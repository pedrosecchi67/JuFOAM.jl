using JuFOAM
using JuFOAM.UnstructuredGrids
using JuFOAM.CFD
using JuFOAM.Solver

using Distributed

include("memprofile.jl")

#=
include("unstructured_organization.jl")
include("multigrid.jl")
include("backend_conversion.jl")
=#
include("advection.jl")
include("dissipation.jl")
include("memory.jl")
