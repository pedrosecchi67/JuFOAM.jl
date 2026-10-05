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
end

function Solver(msh::PolyhedralMesh, n_levels::Int = 4)
    dom, coarse_doms, coarseners, prolongators = MultigridDomain(n_levels, msh)
    chimera_interpolator = ChimeraInterpolator(dom, "ORPHAN")

    coarse_chimera_interpolators = [
        ChimeraInterpolator(cdom, "ORPHAN") for cdom in coarse_doms
    ]

    Solver(dom, coarse_doms, coarseners, prolongators, 
        chimera_interpolator, coarse_chimera_interpolators)
end

struct Solution
    P∞::AbstractVector # freestream primitive variables, p, T, u, v
    P::AbstractMatrix # primitive variables, p, T, u, v
    fluid::Fluid
    wall_bc::FlowBC
    freestream_bc::FlowBC
    symmetry_bc::FlowBC
end

function Solution(
    solv::Solver, p∞::Real, T∞::Real, uvw∞::AbstractVector;
    μref::Real = 1.7894e-5 # for air at 288.15 K
)
    P∞ = [p∞, T∞, uvw∞...]
    fluid = Fluid(; μref = μref)

    wall_bc = FlowBC(fluid, [p∞, T∞, 0.0, 0.0])
    freestream_bc = FlowBC(fluid, P∞)
    symmetry_bc = FlowBC(fluid, [p∞, T∞, 0.0]; normal_flow = true)

    P = repeat(P∞'; outer = (length(solv.domain), 1))

    Solution(P∞, P, 
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
    solv::Solver, soln::Solution, P::AbstractMatrix;
    CFL::Real = 1.0, CFL_global::Real = Inf64,
    multigrid_level::Int = 0,
    high_order::Bool = true,
    source_terms::Union{Nothing, AbstractMatrix} = nothing
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

    dom(R, P, dt, "ORPHAN" => Pchi) do dom, R, P, dt, (_, Pchi)
        # set up chimera grid values
        p = view(P, :, 1)
        T = view(P, :, 2)
        uvw = @view P[:, 3:end]

        a = speed_of_sound(fluid, T)
        μ = dynamic_viscosity(fluid, T)
        ρ = p ./ T ./ fluid.R
        ν = μ ./ ρ

        # timescale
        let cosθ = sum(dom.face_normals .* dom.owner_neighbor_directions; 
            dims = 2) |> vec
            λf = sum(at_faces(dom, uvw) .* dom.face_normals; 
                dims = 2) |> vec |> x -> abs.(x) .+ at_faces(dom, a)
            νf = at_faces(dom, ν)

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
        f .-= viscous_fluxes(fluid, Pf, ∇Pf, dom.face_normals)

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
        solv, soln, soln.P; high_order = true,
        CFL = CFL, CFL_global = CFL_global,
    )
    residuals = sum(S .^ 2; dims = 1) |> vec |> x -> sqrt.(x)

    S .-= (
        residual_and_timescale(
            solv, soln, soln.P; high_order = false
        )[1]
    )

    # coarsen time step vectors
    Δts = [Δt]
    for coars in solv.coarseners
        push!(Δts, coars(Δts[end]))
    end

    Pold = copy(soln.P)
    Polds = [Pold]
    for coars in solv.coarseners
        push!(Polds, coars(Polds[end]))
    end

    f = (l, P) -> begin
        dP!dt, dt = residual_and_timescale(solv, soln, P; 
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
