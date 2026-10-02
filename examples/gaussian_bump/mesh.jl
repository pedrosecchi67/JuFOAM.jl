using JuFOAM
using JuFOAM.UnstructuredGrids

using Serialization

function get_exponential_spacing(
    x0::Real, x1::Real,
    h0::Real, growth_ratio::Real = 1.1;
    hmax::Real = Inf64,
    flip::Bool = false,
)
    L = abs(x0 - x1)

    x = [0.0]
    h = h0
    while x[end] < L
        push!(x, x[end] + h)
        h *= growth_ratio
        h = min(hmax, h)
    end

    x ./= x[end]

    if flip
        x = 1.0 .- x[end:-1:1]
    end

    @. x = x * (x1 - x0) + x0

    x
end

function get_block(
    x::AbstractVector, h0::Real, L::Real;
    growth_ratio::Real = 1.1, yfunc = x -> 0.0,
    hmax::Real = Inf64,
)
    y = get_exponential_spacing(0.0, L, h0, growth_ratio; hmax = hmax)

    η = (y .- y[1]) / (y[end] - y[1])
    dy = yfunc.(x)

    x = repeat(x; outer = (1, length(y)))
    y = repeat(y'; outer = (size(x, 1), 1))

    y .+= dy .* (1.0 .- η)'

    (x, y)
end

function bump_height(x::Real)
    y = 0.0
    if 0.3 ≤ x ≤ 1.2
        y = 0.05 * sin(π * x / 0.9 - π / 3.0)^4
    end
    return y
end

meshes = []

let (x, y) = get_block(
        get_exponential_spacing(-25.0, 0.0, 1e-3; flip = true),
        1.4e-5, 5.0; hmax = 0.05,
    )
    PolyhedralMesh(
        x, y;
        families = [
            "farfield" => [(1, false), (2, true)],
            "symmetry" => [(2, false)]
        ]
    )
end |> x -> push!(meshes, x)

let (x, y) = get_block(
        get_exponential_spacing(0.0, 0.75, 1e-3; flip = false,
            hmax = 0.01),
        1.4e-5, 5.0; yfunc = bump_height, hmax = 0.05,
    )
    PolyhedralMesh(
        x, y;
        families = [
            "farfield" => [(2, true)],
            "wall" => [(2, false)]
        ]
    )
end |> x -> push!(meshes, x)

let (x, y) = get_block(
        get_exponential_spacing(0.75, 1.6, 1e-3; flip = true,
            hmax = 0.01),
        1.4e-5, 5.0; yfunc = bump_height, hmax = 0.05,
    )
    PolyhedralMesh(
        x, y;
        families = [
            "farfield" => [(2, true)],
            "wall" => [(2, false)]
        ]
    )
end |> x -> push!(meshes, x)

let (x, y) = get_block(
        get_exponential_spacing(1.6, 26.6, 1e-3; flip = false),
        1.4e-5, 5.0; hmax = 0.05,
    )
    PolyhedralMesh(
        x, y;
        families = [
            "farfield" => [(1, true), (2, true)],
            "symmetry" => [(2, false)]
        ]
    )
end |> x -> push!(meshes, x)

msh = PolyhedralMesh(meshes...)

dom = Domain(msh)
dists = wall_distances(dom, "wall")

vtk = vtk_grid("mesh", msh)
vtk["d"] = dists
vtk_save(vtk)

vtk_grid("surface", msh, "wall") |> vtk_save

@show msh.cells |> length

serialize("mesh.ufoam", msh)
