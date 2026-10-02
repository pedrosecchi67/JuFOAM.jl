include("cfd.jl")

using Serialization

msh = deserialize("mesh.ufoam")

solv = Solver(msh)
soln = Solution(solv, 1e5, 288.15, [1.0, 0.0])
