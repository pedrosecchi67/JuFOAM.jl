using JuFOAM
using JuFOAM.UnstructuredGrids
using JuFOAM.CFD
using JuFOAM.Turbulence

struct Solver
    domain::AbstractDomain
    coarse_domains::Vector{AbstractDomain}
    coarseners::Vector{JuFOAM.Accumulator}
    prolongators::Vector{JuFOAM.Accumulator}
    chimera_interpolator::JuFOAM.Accumulator
end

function Solver(msh::PolyhedralMesh, n_levels::Int = 4)
    dom, coarse_doms, coarseners, prolongators = MultigridDomain(n_levels, msh)
    chimera_interpolator = ChimeraInterpolator(dom, "ORPHAN")

    Solver(dom, coarse_doms, coarseners, prolongators, chimera_interpolator)
end

struct Solution
    P::AbstractMatrix # primitive variables, p, T, u, v
    fluid::Fluid
    wall_bc::FlowBC
    freestream_bc::FlowBC
    symmetry_bc::FlowBC
end

function Solution(
    solv::Solver, p∞::Real, T∞::Real, uvw∞::AbstractVector
)
    P∞ = [p∞, T∞, uvw∞...]
    fluid = Fluid()

    wall_bc = FlowBC(fluid, [p∞, T∞, 0.0]; normal_flow = true)
    freestream_bc = FlowBC(fluid, P∞)
    symmetry_bc = FlowBC(fluid, [p∞, T∞, 0.0]; normal_flow = true)

    P = repeat(P∞'; outer = (length(solv.domain), 1))

    Solution(P, fluid, wall_bc, freestream_bc, symmetry_bc)
end
