module UnstructuredGrids

    export PolyhedralMesh,
        set_boundary_projections!,
        set_image_distances!,
        face_information, boundary_information,
        vtk_grid, vtk_save, vtk_multiblock

using DocStringExtensions

using LinearAlgebra

"""
$TYPEDSIGNATURES

Struct describing a polyhedral mesh
"""
struct PolyhedralMesh{Tf <: AbstractFloat, Ti <: Integer}
    points::AbstractMatrix{Tf}
    faces::AbstractVector{Vector{Ti}}
    cells::AbstractVector{Vector{Ti}}
    families::Dict{String, AbstractVector}
    boundary_projections::Dict{String, AbstractMatrix}
    image_point_offsets::Dict{String, AbstractVector}

    function PolyhedralMesh{Tf, Ti}(
        points::AbstractMatrix{Tf},
        faces::AbstractVector{Vector{Ti}},
        cells::AbstractVector{Vector{Ti}},
        families::AbstractDict,
    ) where {Tf <: AbstractFloat, Ti <: Integer}
        new(points, faces, cells, families,
            Dict{String, AbstractMatrix}(),
            Dict{String, AbstractVector}())
    end
end

"""
$TYPEDSIGNATURES

Turn block-structured grid to polyhedral mesh.

Family specifications may be given by a `(dimension, front/back)` notation:

```
families = [
    "inlet" => [
        (1, false), # x-axis, back
        (2, true), # y-axis, right
        (3, false), # z-axis, bottom
        (3, true) # z-axis, top
    ],
    "symmetry" => [
        (2, false) # y-axis, left
    ],
    "outlet" => [
        (1, true) # x-axis, front
    ]
]
```

Unspecified families are labelled as `"ORPHAN"`.
"""
function PolyhedralMesh(
    xyz::AbstractArray...;
    families = [],
)
    Tf = eltype(first(xyz))

    ndim = ndims(first(xyz))
    @assert (ndim in (2, 3)) "Block-structured grid dimensionalities must be 2 or 3"

    points = vec.(xyz) |> x -> hcat(x...)
    npoints = size(points, 1)

    Ti = (npoints > 2e9 / ndim ? Int64 : Int32)

    lattice_size = size(first(xyz))
    ncells = prod((lattice_size .- 1))
    indices = reshape(Ti(1):Ti(npoints), lattice_size)

    get_face = (lattice, inds...) -> (
        ndim == 2 ? let i = inds[1]
            [lattice[i], lattice[i + 1]]
        end : let (i, j) = inds
            [lattice[i, j], lattice[i + 1, j], lattice[i + 1, j + 1], lattice[i, j + 1]]
        end
    )
    faces = Vector{Ti}[]

    push_face! = f -> begin
        push!(faces, f)
        Ti(length(faces))
    end

    cells = [
        Ti[] for _ = 1:ncells
    ] |> x -> reshape(x, (lattice_size .- 1))

    slice2faces = slice -> (
        ndims(slice) == 2 ? [
            push_face!(
                get_face(slice, i, j)
            ) for i = 1:(size(slice, 1) - 1), j = 1:(size(slice, 2) - 1)
        ] : [
            push_face!(
                get_face(slice, i)
            ) for i = 1:(size(slice, 1) - 1)
        ]
    )

    boundary_names = fill("ORPHAN", (ndim, 2))
    for (fname, bfaces) in families
        for (dim, fb) in bfaces
            boundary_names[dim, (fb + 1)] = fname
        end
    end

    families = Dict{String, Vector{Ti}}()
    push_slice_family! = (fname, slice_faces) -> begin
        if haskey(families, fname)
            families[fname] = [families[fname]; vec(slice_faces)]
        else
            families[fname] = vec(slice_faces)
        end
    end

    for dim = 1:ndim
        for k = 1:size(indices, dim)
            lattice_slice = selectdim(indices, dim, k)
            slice_faces = slice2faces(lattice_slice)

            if k < size(indices, dim)
                for (face, cell) in zip(slice_faces, selectdim(cells, dim, k))
                    push!(cell, face)
                end
            end

            if k > 1
                for (face, cell) in zip(slice_faces, selectdim(cells, dim, k - 1))
                    push!(cell, face)
                end
            end

            if k == 1
                push_slice_family!(
                    boundary_names[dim, 1], slice_faces
                )
            elseif k == size(indices, dim)
                push_slice_family!(
                    boundary_names[dim, 2], slice_faces
                )
            end
        end
    end

    cells = vec(cells)

    PolyhedralMesh{Tf, Ti}(points, faces, cells, families)
end

"""
$TYPEDSIGNATURES

Obtain polyhedral mesh from Cartesian grid.
Argument `families` is the same as in `PolyhedralMesh(xyz...)`
(block-structured variant).

Example:

```
mesh = PolyhedralMesh(
    [0.0, 0.0], # hypercube origin
    [1.0, 2.0], # hypercube widths
    (10, 20); # spacing along each axis
    families = [
        "inlet" => [(1, false), (2, false), (2, true)],
        "outlet" => [(1, true)]
    ]
)
```
"""
function PolyhedralMesh(
    origin::AbstractVector{Tf}, widths::AbstractVector{Tf},
    sizes::Tuple;
    families = [], 
) where {Tf <: AbstractFloat}
    xyz = []

    rshape_dims = ones(Int64, length(origin))
    for dim = 1:length(origin)
        coords = Array{Tf}(undef, sizes...)

        rng = LinRange(origin[dim], origin[dim] + widths[dim], sizes[dim])

        rshape_dims[dim] = length(rng)
        coords .= reshape(rng, tuple(rshape_dims...))
        rshape_dims[dim] = 1

        push!(xyz, coords)
    end

    PolyhedralMesh(
        xyz...; families = families,
    )
end

"""
$TYPEDSIGNATURES

Merge multiple polyhedral meshes into one according to given
point merging tolerance.
"""
function PolyhedralMesh(
    grids::PolyhedralMesh...;
    tolerance::Real = 1f-7,
)
    ndim = size(grids[1].points, 2)
    
    nfaces = 0
    npoints = 0
    for grid in grids
        nfaces += length(grid.faces)
        npoints += size(grid.points, 1)
    end
    Ti = (
        max(nfaces, npoints) > 1e9 ? Int64 : Int32
    )
    Tf = typeof(tolerance)

    points, point_keys = let pointhash = Dict{
        NTuple{ndim, Int64}, Ti
    }()
        n0 = Ti(0)
        point_keys = Vector{Ti}[]

        push_point! = pt -> let tag = tuple(
            Int64.(round.(pt ./ tolerance))...
        )
            if haskey(pointhash, tag)
                return pointhash[tag]
            end

            n0 += one(Ti)
            pointhash[tag] = n0

            n0
        end

        for grid in grids
            indices = map(push_point!, eachrow(grid.points))
            push!(point_keys, indices)
        end

        points = Matrix{Tf}(undef, n0, ndim)
        for (grid, key) in zip(grids, point_keys)
            for (pt, i) in zip(eachrow(grid.points), key)
                points[i, :] .= pt
            end
        end

        (points, point_keys)
    end

    faces, face_keys = let facehash = Dict{Tuple, Ti}()
        n0 = Ti(0)
        face2tag = face -> tuple(sort(face)...)
        faces = Vector{Ti}[]
        push_face! = face -> begin
            face = unique(face)
            if length(face) < ndim
                return Ti(0)
            end

            tag = face2tag(face)
            if haskey(facehash, tag)
                return facehash[tag]
            end

            n0 += one(Ti)
            facehash[tag] = n0
            push!(faces, face)

            n0
        end

        face_keys = Vector{Ti}[]
        for (grid, ptkey) in zip(grids, point_keys)
            indices = map(
                fc -> push_face!(ptkey[fc]), grid.faces
            )
            push!(face_keys, indices)
        end

        (faces, face_keys)
    end

    cells = Vector{Ti}[]
    for (fckey, grid) in zip(face_keys, grids)
        for cell in grid.cells
            cell = fckey[cell]
            filter!(f -> f != 0, cell)
            unique!(cell)

            if length(cell) > ndim
                push!(cells, cell)
            end
        end
    end

    families = Dict{String, AbstractVector}()
    push_family! = (fname, indices) -> begin
        if !haskey(families, fname)
            families[fname] = indices
        else
            families[fname] = [families[fname]; indices]
        end
    end
    for (grid, fckey) in zip(grids, face_keys)
        for (fname, faceinds) in grid.families
            ifaces = fckey[faceinds]
            filter!(f -> f != 0, ifaces)

            push_family!(fname, ifaces)
        end
    end

    PolyhedralMesh(
        points, faces, cells, families
    )
end

"""
$TYPEDSIGNATURES

Set face center projection points for immersed boundary.
Gets name of pre-existing family and matrix (each row a proj. point)
with the projections of all centers of faces in `mesh.families[fname]`.
"""
function set_boundary_projections!(
    mesh::PolyhedralMesh{Tf, Ti}, fname::String, projections::AbstractMatrix{Tf}
) where {Tf, Ti}
    mesh.boundary_projections[fname] = copy(projections)
end

"""
$TYPEDSIGNATURES

Set image point distances to boundary for immersed boundary family.
Gets name of pre-existing family and vector of distances between each image point
and the boundary.
"""
function set_image_distances!(
    mesh::PolyhedralMesh{Tf, Ti}, fname::String, distances::AbstractVector{Tf}
) where {Tf, Ti}
    mesh.image_point_offsets[fname] = copy(distances)
end

"""
Obtain simplex normal, area and center
"""
function simplex_nAc(
    simplex::AbstractMatrix
)
    μ = sum(simplex; dims = 1) |> vec
    μ ./= size(simplex, 1)

    p0 = simplex[1, :]
    if length(p0) == 2
        v = simplex[2, :] .- p0
        n = [-v[2], v[1]]

        A = norm(n)
        n ./= (A + 1f-14)

        return (n, A, μ)
    end

    u = simplex[2, :] .- p0
    v = simplex[3, :] .- p0

    n = cross(u, v)
    @. n /= 2
    A = norm(n)
    n ./= (A + 1f-14)

    (n, A, μ)
end

"""
Obtain face normal, area and center
"""
function face_nAc(
    points::AbstractMatrix{Tf}, face::AbstractVector{Ti}
) where {Tf <: AbstractFloat, Ti <: Integer}
    ndim = size(points, 2)
    simplex = zeros(Ti, ndim)
    simplex[1] = face[1]

    if ndim == 2 && length(face) > 2
        @assert length(face) == 2 "2D faces must all be lines: found face with more than two points"
    end

    if ndim == 2
        simplex[2] = face[2]

        M = @view points[simplex, :]
        return simplex_nAc(M)
    end

    c = zeros(Tf, ndim)
    A = zero(Tf)
    n = zeros(Tf, ndim)

    for i = 2:(length(face) - 1)
        inext = i + 1
        simplex[2] = face[i]
        simplex[3] = face[inext]

        M = @view points[simplex, :]
        _n, _A, _c = simplex_nAc(M)

        @. c += _c * _A
        @. n += _n * _A
        A += _A
    end

    @. c /= A
    A = norm(n)
    n ./= (A + 1f-14)

    (n, A, c)
end

"""
$TYPEDSIGNATURES

Obtain face information for unstructured grid.
Returns:

* `face_owners`: vector of indices
* `face_neighbors`: vector of indices
* `face_centers`: matrix, each row a face center
* `face_normals`: matrix, each row a face normal (norm = area)
"""
function face_information(
    msh::PolyhedralMesh{Tf, Ti}
) where {Tf, Ti}
    nfaces = length(msh.faces)
    ndim = size(msh.points, 2)

    face_owners = zeros(Ti, nfaces)
    face_neighbors = zeros(Ti, nfaces)

    for (icell, cell) in enumerate(msh.cells)
        for ifc in cell
            if face_owners[ifc] == 0
                face_owners[ifc] = icell
            else
                face_neighbors[ifc] = icell
            end
        end
    end
    for i = 1:nfaces
        if face_neighbors[i] == 0
            face_neighbors[i] = face_owners[i]
        end
    end

    face_centers = Matrix{Tf}(undef, nfaces, ndim)
    face_normals = Matrix{Tf}(undef, nfaces, ndim)

    for (i, face) in enumerate(msh.faces)
        n, A, c = face_nAc(msh.points, face)

        face_centers[i, :] .= c
        face_normals[i, :] .= n .* A
    end

    (face_owners, face_neighbors, face_centers, face_normals)
end

"""
$TYPEDSIGNATURES

Obtain boundary information as vector of pairs,
each pair pointing to the information needed to define a boundary.

For body-fitted boundaries, includes vectors of
For immersed boundaries
"""
function boundary_information(
    msh::PolyhedralMesh
)
    binfo = []

    for (fname, family) in msh.families
        if haskey(msh.boundary_projections, fname)
            if haskey(msh.image_point_offsets, fname)
                family = (
                    family,
                    msh.boundary_projections[fname],
                    msh.image_point_offsets[fname]
                )
            else
                family = (
                    family,
                    msh.boundary_projections[fname]
                )
            end
        end

        push!(binfo, fname => family)
    end

    binfo
end

using WriteVTK

"""
$TYPEDSIGNATURES

Obtain VTK surface from ring (list of lists of indices) notation
"""
function WriteVTK.vtk_grid(
    fname::String,
    points::AbstractMatrix{Tf}, faces::AbstractVector{Tv};
    vtm::Union{Nothing, WriteVTK.MultiblockFile} = nothing
) where {Tf <: AbstractFloat, Tv <: AbstractVector}
    points, faces = let isval = falses(size(points, 2))
        for face in faces
            isval[face] .= true
        end

        new_inds = cumsum(isval)
        
        (
            points[:, isval],
            [Tv(new_inds[face]) for face in faces]
        )
    end
    Base.GC.gc()

    ctype = (
        size(points, 1) == 2 ?
        VTKCellTypes.VTK_POLY_LINE : VTKCellTypes.VTK_POLYGON
    )
    cells = [
        MeshCell(ctype, conn) for conn in faces
    ]
    Base.GC.gc()

    (
        isnothing(vtm) ?
        vtk_grid(fname, points, cells) :
        vtk_grid(vtm, fname, points, cells)
    )
end

"""
$TYPEDSIGNATURES

Obtain VTK volume from ring (list of lists of indices) and cell 
(list of lists of face indices) notation
"""
function WriteVTK.vtk_grid(
    fname::String,
    points::AbstractMatrix{Tf}, faces::AbstractVector{Tv}, cells::AbstractVector{Tv2};
    vtm::Union{Nothing, WriteVTK.MultiblockFile} = nothing
) where {Tf <: AbstractFloat, Tv <: AbstractVector, Tv2 <: AbstractVector}
    faces, cells = let isval = falses(length(faces))
        for cell in cells
            isval[cell] .= true
        end
        new_inds = cumsum(isval)

        (faces[isval], [Tv(new_inds[cell]) for cell in cells])
    end
    Base.GC.gc()

    points, faces = let isval = falses(size(points, 2))
        for face in faces
            isval[face] .= true
        end
        new_inds = cumsum(isval)
        
        (
            points[:, isval],
            [Tv(new_inds[face]) for face in faces]
        )
    end
    Base.GC.gc()

    if size(points, 1) == 2 # 2D
        cells = WriteVTK.MeshCell[
                let indices = vcat(faces[cell]...)
                    pts = points[:, indices]
                    μ = sum(pts; dims = 2) |> vec |> x -> x ./ length(indices)

                    θ = atan.(pts[2, :] .- μ[2], pts[1, :] .- μ[1])
                    asrt = sortperm(θ) # make a polygon and sort by azymuth

                    WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_POLYGON, indices[asrt])
                end for cell in cells
        ]
    else # 3D
        cells = [
            let faces = faces[cell]
                WriteVTK.VTKPolyhedron(
                    vcat(faces...),
                    map(
                        f -> tuple(f...), faces
                    )...    
                )
            end for cell in cells
        ]
    end
    Base.GC.gc()

    (
        isnothing(vtm) ?
        vtk_grid(fname, points, cells) :
        vtk_grid(vtm, fname, points, cells)
    )
end

"""
$TYPEDSIGNATURES

Obtain vtk file object from WriteVTK for a family in a polyhedral mesh.
"""
WriteVTK.vtk_grid(
    fname::String, msh::PolyhedralMesh, family::String;
    vtm::Union{Nothing, WriteVTK.MultiblockFile} = nothing
) = vtk_grid(
    fname, permutedims(msh.points), view(msh.faces, msh.families[family]); 
    vtm = vtm,
)

"""
$TYPEDSIGNATURES

Obtain vtk file object from WriteVTK for a polyhedral mesh.
"""
WriteVTK.vtk_grid(
    fname::String, msh::PolyhedralMesh;
    vtm::Union{Nothing, WriteVTK.MultiblockFile} = nothing
) = vtk_grid(
    fname, permutedims(msh.points), msh.faces, msh.cells; 
    vtm = vtm,
)

end
