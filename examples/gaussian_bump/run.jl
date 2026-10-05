include("cfd.jl")

using Serialization

msh = deserialize("mesh.ufoam")

Mach = 0.2
Re = 3e6
L = 1.0

V = Mach * sqrt(1.4 * 287.05 * 288.15)

μref = let fluid = Fluid()
    fluid = adjust_Reynolds(fluid, [1e5, 288.15, V, 0.0], L, Re)

    fluid.μref
end

solv = Solver(msh)
soln = Solution(solv, 1e5, 288.15, [V, 0.0];
    μref = μref)


P = soln.P
for nit = 1:1000
    residuals = solve!(solv, soln)
    turb_residuals = solve_turb!(solv, soln)

    @show nit residuals turb_residuals
end


p = view(soln.P, :, 1)
T = view(soln.P, :, 2)
uv = view(soln.P, :, 3:4)

ρ = p ./ soln.fluid.R ./ T
ν = dynamic_viscosity(soln.fluid, T) ./ ρ

eddy_viscosity_ratio = soln.νₜ ./ ν

p∞ = soln.P∞[1]
Cp = pressure_coefficient(soln.fluid, p, p∞, Mach)

vtk = vtk_grid("solution", msh)

vtk["p"] = p
vtk["T"] = T
vtk["uv"] = uv'
vtk["rho"] = ρ
vtk["Cp"] = Cp
vtk["eddy_viscosity_ratio"] = eddy_viscosity_ratio
vtk["wall_distance"] = solv.wall_distance

vtk_save(vtk)
