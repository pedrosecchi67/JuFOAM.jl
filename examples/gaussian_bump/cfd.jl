using JuFOAM
using JuFOAM.UnstructuredGrids
using JuFOAM.CFD
using JuFOAM.Turbulence
using JuFOAM.Solver

struct Solver
    domain::AbstractDomain
    coarse_domains::Vector{AbstractDomain}
    coarseners::Vector{JuFOAM.Accumulator}
    prolongators::Vector{JuFOAM.Accumulator}
    chimera_interpolator::JuFOAM.Accumulator
    coarse_chimera_interpolators::Vector{JuFOAM.Accumulator}
    wall_distance::AbstractVector
end

function Solver(msh::PolyhedralMesh, n_levels::Int = 4)
    dom, coarse_doms, coarseners, prolongators = MultigridDomain(n_levels, msh)
    chimera_interpolator = ChimeraInterpolator(dom, "ORPHAN")

    coarse_chimera_interpolators = [
        ChimeraInterpolator(cdom, "ORPHAN") for cdom in coarse_doms
    ]

    d = wall_distances(dom, "wall") # add more families if you need to ;)

    Solver(dom, coarse_doms, coarseners, prolongators, 
        chimera_interpolator, coarse_chimera_interpolators, d)
end

struct Solution
    P∞::AbstractVector # freestream primitive variables, p, T, u, v
    P::AbstractMatrix # primitive variables, p, T, u, v
    ν̂∞::Real # freestream SA turbulent variable
    ν̂::AbstractVector # SA turbulent variable
    νₜ::AbstractVector # eddy viscosity
    fluid::Fluid
    wall_bc::FlowBC
    freestream_bc::FlowBC
    symmetry_bc::FlowBC
end

function Solution(
    solv::Solver, p∞::Real, T∞::Real, uvw∞::AbstractVector;
    μref::Real = 1.7894e-5, # for air at 288.15 K
    freestream_turbulence_ratio::Real = 3.0f0,
)
    P∞ = [p∞, T∞, uvw∞...]
    fluid = Fluid(; μref = μref)

    wall_bc = FlowBC(fluid, [p∞, T∞, 0.0, 0.0])
    freestream_bc = FlowBC(fluid, P∞)
    symmetry_bc = FlowBC(fluid, [p∞, T∞, 0.0]; normal_flow = true)

    P = repeat(P∞'; outer = (length(solv.domain), 1))

    ν∞ = dynamic_viscosity(fluid, T∞) / (p∞ / T∞ / fluid.R)
    ν̂∞ = freestream_turbulence_ratio * ν∞
    ν̂ = fill(ν̂∞, length(solv.domain))

    νₜ = similar(ν̂ )
    νₜ .= 0

    Solution(P∞, P, ν̂∞, ν̂, νₜ,
        fluid, wall_bc, freestream_bc, symmetry_bc)
end

# apply BCs to face properties at Pf
function apply_bcs!(
    soln::Solution, dom::AbstractDomain, 
    P::AbstractMatrix, Pf::AbstractMatrix, 
    Pchi::AbstractMatrix
)
    # orphan faces (chimera)
    at_boundary(dom.boundaries["ORPHAN"], Pf) .= Pchi

    for wall in ("wall",) # add more families if you need to ;)
        bdry = dom.boundaries[wall]

        # obtain image point data from cell data
        Pimages = at_images(bdry, P)

        Pbdry = soln.wall_bc(Pimages, bdry.normals)
        at_boundary(bdry, Pf) .= Pbdry
    end

    # symmetry
    for sym in ("symmetry",)
        bdry = dom.boundaries[sym]

        Pimages = at_images(bdry, P)
        Pbdry = soln.symmetry_bc(Pimages, bdry.normals)
        at_boundary(bdry, Pf) .= Pbdry
    end

    # freestream
    for fs in ("farfield",)
        bdry = dom.boundaries[fs]

        Pimages = at_images(bdry, P)
        Pbdry = soln.freestream_bc(Pimages, bdry.normals)
        at_boundary(bdry, Pf) .= Pbdry
    end
end

function residual_and_timescale(
    solv::Solver, soln::Solution, P::AbstractMatrix, νₜ::AbstractVector;
    CFL::Real = 1.0, CFL_global::Real = Inf64,
    multigrid_level::Int = 0,
    high_order::Bool = true,
)
    dom = (
        multigrid_level == 0 ? solv.domain : solv.coarse_domains[multigrid_level]
    )

    R = similar(P)
    R .= 0
    dt = similar(P, (length(dom),))
    dt .= 0

    fluid = soln.fluid

    chimera_interpolator = (
        multigrid_level == 0 ? 
        solv.chimera_interpolator : 
        solv.coarse_chimera_interpolators[multigrid_level]
    )
    Pchi = chimera_interpolator(P)

    dom(R, P, νₜ, dt, "ORPHAN" => Pchi) do dom, R, P, νₜ, dt, (_, Pchi)
        # set up chimera grid values
        p = view(P, :, 1)
        T = view(P, :, 2)
        uvw = @view P[:, 3:end]

        a = speed_of_sound(fluid, T)
        μ = dynamic_viscosity(fluid, T)
        ρ = p ./ T ./ fluid.R
        ν = μ ./ ρ
        μₜ = νₜ .* ρ

        # timescale
        let cosθ = sum(dom.face_normals .* dom.owner_neighbor_directions; 
            dims = 2) |> vec
            λf = sum(at_faces(dom, uvw) .* dom.face_normals; 
                dims = 2) |> vec |> x -> abs.(x) .+ at_faces(dom, a)
            νf = at_faces(dom, ν .+ νₜ)

            dt .= 1.0f0 ./ green_gauss(dom, 
                λf .+ νf .* cosθ ./ dom.owner_neighbor_distances; 
                signed = false)
        end

        # compute gradients
        Pf = at_faces(dom, P)
        apply_bcs!(soln, dom, P, Pf, Pchi)

        ∇P = gradient(dom, Pf)

        # face information
        Po = at_owners(dom, P)
        Pn = at_neighbors(dom, P)

        # MUSCL
        PL = PR = nothing
        if high_order
            PL, PR = MUSCL(dom, P, ∇P)
        else
            PL = copy(Po)
            PR = copy(Pn)
        end

        apply_bcs!(soln, dom, P, PL, Pchi)
        apply_bcs!(soln, dom, P, PR, Pchi)
        apply_bcs!(soln, dom, P, Po, Pchi)
        # we don't correct neighbors, they're in the domain

        # correct gradients for faces
        ∇Pf = face_gradient(dom, ∇P, Po, Pn)

        # inviscid fluxes
        f = inviscid_fluxes(fluid, PL, PR, dom.face_normals)
        # viscous fluxes
        f .-= viscous_fluxes(fluid, Pf, ∇Pf, dom.face_normals; μₜ = at_faces(dom, μₜ))

        # divergent
        Qdot = - green_gauss(dom, f)

        # evolve state variables, update residuals
        # to reflect primitive variables in time
        let Q = primitive2state(fluid, P)
            Pnew = state2primitive(
                fluid, Q .+ dt .* Qdot
            )
            @. R = (Pnew - P) / dt
        end

        ;
    end

    # apply CFL condition
    let dtmin = minimum(dt)
        @. dt = min(dt * CFL, dtmin * CFL_global)
    end

    (R, dt)
end

function solve!(
    solv::Solver, soln::Solution;
    CFL::Real = 10.0,
    CFL_global::Real = 1000.0,
    n_iter::Int = 10,
    n_cycles::Int = 10,
)
    # calc. source term as difference between high-order and low-order residuals
    S, Δt = residual_and_timescale(
        solv, soln, soln.P, soln.νₜ; high_order = true,
        CFL = CFL, CFL_global = CFL_global,
    )
    residuals = sum(S .^ 2; dims = 1) |> vec |> x -> sqrt.(x)

    S .-= (
        residual_and_timescale(
            solv, soln, soln.P, soln.νₜ; high_order = false
        )[1]
    )

    # coarsen time step size vectors
    # and eddy viscosity fields for multigrid
    Δts = [Δt]
    νₜs = [soln.νₜ]
    for coars in solv.coarseners
        push!(Δts, coars(Δts[end]))
        push!(νₜs, coars(νₜs[end]))
    end

    Pold = copy(soln.P)
    Polds = [Pold]
    for coars in solv.coarseners
        push!(Polds, coars(Polds[end]))
    end

    # define residual and relaxation for dual time-stepping
    f = (l, P) -> begin
        dP!dt, dt = residual_and_timescale(solv, soln, P, νₜs[l+1]; 
            high_order = false, multigrid_level = l)
        if l == 0
            @. dP!dt += S
        end

        _dt = Δts[l+1]
        _Pold = Polds[l+1]
        R = @. dP!dt * _dt - (P - _Pold)
        ω = dt ./ _dt ./ 2

        (R, ω)
    end

    FAS!(
        f, soln.P;
        coarseners = solv.coarseners, prolongators = solv.prolongators,
        n_iter = n_iter, n_cycles = n_cycles,
        rtol = 1e-2, atol = 0.0,
    )

    residuals
end

# apply BCs to face properties at ν̂f
function apply_bcs_turb!(
    soln::Solution, dom::AbstractDomain, 
    ν̂::AbstractVector, ν̂f::AbstractVector, 
    ν̂chi::AbstractVector
)
    # orphan faces (chimera)
    at_boundary(dom.boundaries["ORPHAN"], ν̂f) .= ν̂chi

    for wall in ("wall",) # add more families if you need to ;)
        bdry = dom.boundaries[wall]
        at_boundary(bdry, ν̂f) .= 0
    end

    # symmetry
    for sym in ("symmetry",)
        bdry = dom.boundaries[sym]

        ν̂i = at_images(bdry, ν̂ )
        at_boundary(bdry, ν̂f) .= ν̂i
    end

    # freestream
    for fs in ("farfield",)
        bdry = dom.boundaries[fs]

        at_boundary(bdry, ν̂f) .= soln.ν̂∞
    end
end

function residual_and_timescale_turb(
    solv::Solver, soln::Solution, P::AbstractMatrix, 
    ν̂::AbstractVector, d::AbstractVector;
    CFL::Real = 1.0, CFL_global::Real = Inf64,
    multigrid_level::Int = 0,
    high_order::Bool = true,
)
    dom = (
        multigrid_level == 0 ? solv.domain : solv.coarse_domains[multigrid_level]
    )

    r = similar(ν̂ )
    r .= 0
    dt = similar(ν̂ )
    dt .= 0

    fluid = soln.fluid

    chimera_interpolator = (
        multigrid_level == 0 ? 
        solv.chimera_interpolator : 
        solv.coarse_chimera_interpolators[multigrid_level]
    )
    ν̂chi = chimera_interpolator(ν̂ )

    νₜ = similar(ν̂ )
    νₜ .= 0
    dom(r, P, ν̂ , νₜ, d, dt, "ORPHAN" => ν̂chi) do dom, r, P, ν̂ , νₜ, d, dt, (_, ν̂chi)
        # set up chimera grid values
        p = view(P, :, 1)
        T = view(P, :, 2)
        uvw = @view P[:, 3:end]

        μ = dynamic_viscosity(fluid, T)
        ρ = p ./ T ./ fluid.R
        ν = μ ./ ρ

        # obtain gradients
        ν̂f = at_faces(dom, ν̂ )
        apply_bcs_turb!(soln, dom, ν̂ , ν̂f, ν̂chi)

        ∇ν̂ = gradient(dom, ν̂f)

        # face information
        ν̂o = at_owners(dom, ν̂ )
        ν̂n = at_neighbors(dom, ν̂ )

        # MUSCL
        ν̂L = ν̂R = nothing
        if high_order
            ν̂L, ν̂R = MUSCL(dom, ν̂ , ∇ν̂ )
        else
            ν̂L = copy(ν̂o)
            ν̂R = copy(ν̂n)
        end

        apply_bcs_turb!(soln, dom, ν̂ , ν̂L, ν̂chi)
        apply_bcs_turb!(soln, dom, ν̂ , ν̂R, ν̂chi)
        apply_bcs_turb!(soln, dom, ν̂ , ν̂o, ν̂chi)

        uvwf = at_faces(dom, uvw)
        ϕ = sum(uvwf .* dom.face_normals; 
                dims = 2) |> vec

        velocity_gradients = let ∇uvw = gradient(dom, uvwf)
            [
                view(∇uvw[j], :, i) for i = 1:size(uvwf, 2), j = 1:length(∇uvw)
            ]
        end
        Ω = vorticity_magnitude(velocity_gradients)

        nt = Spallart_Allmaras(
            ν, d, Ω, ν̂, hcat(∇ν̂...)
        )

        νₜ .= nt.νₜ
        νSA = nt.νSA
        S = nt.S

        # inviscid fluxes
        f = @. ϕ * (ν̂L + ν̂R) / 2 + abs(ϕ) * (ν̂L - ν̂R) / 2

        # viscous fluxes
        let ∇ν̂f = face_gradient(dom, ∇ν̂,  ν̂o, ν̂n)
            νSAf = at_faces(dom, νSA)

            for (n, g) in zip(
                eachcol(dom.face_normals), ∇ν̂f
            )
                @. f -= n * g * νSAf
            end
        end

        # compute residual
        r .= - green_gauss(dom, f)
        r .+= S

        # timescale
        let cosθ = sum(dom.face_normals .* dom.owner_neighbor_directions; 
            dims = 2) |> vec
            λf = abs.(ϕ)
            νf = at_faces(dom, νSA)

            dt .= 1.0f0 ./ green_gauss(dom, 
                λf .+ νf .* cosθ ./ dom.owner_neighbor_distances; 
                signed = false)
        end
    end

    # apply CFL condition
    let dtmin = minimum(dt)
        @. dt = min(dt * CFL, dtmin * CFL_global)
    end

    # update νₜ at solution
    if multigrid_level == 0
        soln.νₜ .= νₜ
    end

    (r, dt)
end

function solve_turb!(
    solv::Solver, soln::Solution;
    CFL::Real = 10.0,
    CFL_global::Real = 1000.0,
    n_iter::Int = 10,
    n_cycles::Int = 10,
)
    # calc. source term as difference between high-order and low-order residuals
    S, Δt = residual_and_timescale_turb(
        solv, soln, soln.P, soln.ν̂, solv.wall_distance; high_order = true,
        CFL = CFL, CFL_global = CFL_global,
    )
    residual = sum(S .^ 2) |> sqrt

    S .-= (
        residual_and_timescale_turb(
            solv, soln, soln.P, soln.ν̂, solv.wall_distance; high_order = false
        )[1]
    )

    # coarsen time step size vectors
    # and primivitve fields for multigrid
    Δts = [Δt]
    Ps = [soln.P]
    ν̂old = copy(soln.ν̂ )
    ν̂olds = [ν̂old]
    wall_dists = [solv.wall_distance]
    for coars in solv.coarseners
        push!(Δts, coars(Δts[end]))
        push!(Ps, coars(Ps[end]))
        push!(ν̂olds, coars(ν̂olds[end]))
        push!(wall_dists, coars(wall_dists[end]))
    end

    # define residual and relaxation for dual time-stepping
    f = (l, ν̂ ) -> begin
        dν̂!dt, dt = residual_and_timescale_turb(solv, soln, Ps[l+1], ν̂, wall_dists[l+1]; 
            high_order = false, multigrid_level = l)
        if l == 0
            @. dν̂!dt += S
        end

        _dt = Δts[l+1]
        _ν̂old = ν̂olds[l+1]
        r = @. dν̂!dt * _dt - (ν̂ - _ν̂old)
        ω = dt ./ _dt ./ 2

        (r, ω)
    end

    FAS!(
        f, soln.ν̂ ;
        coarseners = solv.coarseners, prolongators = solv.prolongators,
        n_iter = n_iter, n_cycles = n_cycles,
        rtol = 1e-2, atol = 0.0,
    )

    residual
end