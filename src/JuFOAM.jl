module JuFOAM

    using LinearAlgebra

    using DocStringExtensions

    include("nninterp.jl")
    using .NNInterpolator
    using .NNInterpolator: KDTree
    using .NNInterpolator.ArrayAccumulator
    using .NNInterpolator.ArrayAccumulator.ArrayBackends

    include("stash.jl")
    using .Stash
    using .Stash.UUIDs
    using .Stash.Distributed

    include("metis.jl")
    using .METIS

    include("unstructured_utils.jl")
    using .UnstructuredGrids
    using .UnstructuredGrids.WriteVTK

    include("cfd.jl")
    using .CFD

    @declare_converter FlowBC

    include("turbulence.jl")
    using .Turbulence

    include("solver.jl")
    using .Solver

    export Domain, PartitionedDomain, AbstractDomain,
        vtk_grid, vtk_save, vtk_multiblock,
        at_boundary, at_images, interp2boundary,
        at_owners, at_neighbors, at_faces, green_gauss,
        gradient, divergent, face_gradient, MUSCL, JST_sensor,
        to_backend, 
        MultigridDomain, Interpolator

    """
    $TYPEDFIELDS

    Abstract type defining a domain
    """
    abstract type AbstractDomain{Tf <: AbstractFloat, Ti <: Integer}
    end

    """
    $TYPEDFIELDS

    Struct to define a boundary.
    `η` is a factor which is 1 if the BC is applied at
    the wall and between 0 and 1 if an immersed boundary
        condition.
    """
    struct Boundary{Tf <: AbstractFloat, Ti <: Integer}
        faces::AbstractVector{Ti}
        projections::AbstractMatrix{Tf}
        images::AbstractMatrix{Tf}
        distances::AbstractVector{Tf}
        η::AbstractVector{Tf}
        normals::AbstractMatrix{Tf}
        image_interpolator::Accumulator
    end

    """
    $TYPEDSIGNATURES

    Constructor for a boundary
    """
    function Boundary(
        cell_centers::AbstractMatrix{Tf}, tree::KDTree, face_centers::AbstractMatrix{Tf},
        projections::AbstractMatrix{Tf},
        indices::AbstractVector{Ti},
        image_points::AbstractMatrix{Tf},
        normals::Union{AbstractMatrix, Nothing} = nothing,
        n_neighbors::Int = 0,
    ) where {Tf <: AbstractFloat, Ti <: Integer}
        if isnothing(normals) # assume normals from projections if not given
            normals = image_points .- projections
            normals ./= sqrt.(
                sum(normals .^ 2; dims = 2) .+ 1f-14
            )
        end

        # distance to image points
        distances = sum((image_points .- projections) .* normals; dims = 2) |> vec
        η = sum((view(face_centers, indices, :) .- projections) .* normals; dims = 2) |> vec
        @. η = 1.0f0 - η / distances

        interpolator = Interpolator(
            cell_centers, image_points, tree;
            first_index = true, linear = true, k = n_neighbors
        ) # interpolator to image points

        Boundary{Tf, Ti}(
            indices,
            projections, image_points,
            distances, η, normals, interpolator,
        )
    end

    """
    $TYPEDFIELDS

    Struct defining a finite-volume domain.
    """
    struct Domain{Tf <: AbstractFloat, Ti <: Integer} <: AbstractDomain{
        Tf, Ti
    }
        centers::AbstractMatrix{Tf}
        volumes::AbstractVector{Tf}
        face_centers::AbstractMatrix{Tf}
        face_normals::AbstractMatrix{Tf}
        face_distances::AbstractVector{Tf}
        face_factors::AbstractVector{Tf}
        face_areas::AbstractVector{Tf}
        owner_neighbor_directions::AbstractMatrix{Tf}
        owner_neighbor_distances::AbstractVector{Tf}
        face_owners::AbstractVector{Ti}
        face_neighbors::AbstractVector{Ti}
        face_sum::Accumulator
        boundaries::Dict{String, Boundary{Tf, Ti}}
        part_index::Int64
    end

    @declare_converter Boundary
    @declare_converter Domain

    """
    Obtain graph from face owners and neighbors.
    """
    function owners_and_neighbors2graph(
        owners::AbstractVector{Ti}, neighbors::AbstractVector{Ti},
        face_areas::AbstractVector{Tf},
    ) where {Ti <: Integer, Tf <: AbstractFloat}
        ncells = max(maximum(owners), maximum(neighbors))

        graph = [
            Ti[] for _ = 1:ncells
        ]
        weights = [
            Tf[] for _ = 1:ncells
        ]
        for (i, (o, n, A)) in zip(owners, neighbors, face_areas) |> enumerate
            if o != 0 && n != 0 && o != n
                if !(n in graph[o])
                    push!(graph[o], n)
                    push!(weights[o], A)
                end
                if !(o in graph[n])
                    push!(graph[n], o)
                    push!(weights[n], A)
                end
            end
        end

        (graph, weights)
    end

    """
    $TYPEDSIGNATURES

    Obtain face accumulators from owners and neighbors.
    """
    function face_accumulator(
        owners::AbstractVector{Ti}, neighbors::AbstractVector{Ti},
    ) where {Ti <: Integer}
        ncells = max(maximum(owners), maximum(neighbors))

        graph = [
            Ti[] for _ = 1:ncells
        ]
        weights = [
            Int8[] for _ = 1:ncells
        ]
        for (i, (o, n)) in enumerate(zip(owners, neighbors))
            if o > 0 && o != n
                push!(graph[o], i)
                push!(weights[o], 1)
            end
            if n > 0
                push!(graph[n], i)
                push!(weights[n], -1)
            end
        end

        Accumulator(graph, weights; first_index = true,)
    end

    # body-fitted
    function _boundary_info(
        family::AbstractVector,
        cell_centers::AbstractMatrix,
        face_owners::AbstractVector,
        face_neighbors::AbstractVector,
        face_centers::AbstractMatrix,
        face_normals::AbstractMatrix,
    )
        indices = copy(family)

        owner_centers = cell_centers[face_owners[indices], :]
        normals = @view face_normals[indices, :]
        centers = @view face_centers[indices, :]

        distances = sum(
            normals .* (owner_centers .- centers); dims = 2
        ) |> vec
        @. distances = abs(distances) + 1f-14

        projections = copy(centers)
        images = owner_centers # image points are owner centers
        normals = copy(normals)

        # only one stencil point: image at owner center
        n_neighbors = 1

        (projections, indices, images, normals, n_neighbors)
    end

    # Immersed Boundary Method
    function _boundary_info(
        family::Tuple,
        cell_centers::AbstractMatrix,
        face_owners::AbstractVector,
        face_neighbors::AbstractVector,
        face_centers::AbstractMatrix,
        face_normals::AbstractMatrix,
    )
        indices = copy(family[1])
        projections = copy(family[2])
        distances = nothing # optional third tuple entry: distance to image point
        if length(family) == 3
            distances = copy(family[3])
        elseif length(family) != 2
            error("Unrecognized length for family-defining tuple") |> throw
        end

        centers = @view face_centers[indices, :]
        face_normals = @view face_normals[indices, :]
        normals = @. centers - projections + face_normals * 1f-14
        
        face_distances = sqrt.(
            sum(normals .^ 2; dims = 2) .+ 1f-14
        )
        if isnothing(distances)
            distances = face_distances .* 2
        end
        normals ./= face_distances

        images = @. projections + distances * normals

        # interpolation from nearby stencil
        n_neighbors = min(
            2 ^ size(images, 2), size(cell_centers, 1)
        )

        (projections, indices, images, normals, n_neighbors)
    end

    """
    $TYPEDSIGNATURES

    Obtain domain as a function of the face owner
    and neighbor cell indices (zero or repeated if
    boundary face), the face centers and the face normals
    (matrices, shape `(nfaces, ndims)`). Face normals should
    have norms equal to face areas.

    Families should be provided as tuples or pairs of family names
    and either vectors of face indices for body-conforming grids, 
    or tuples with face indices and face center projections 
    on boundaries, for immersed-boundary grids.
    For IB grids, the distances between face projections
    and image points may also be specified in a third tuple entry.

    Example:

    ```julia
    dom = Domain(
        face_owners,
        face_neighbors,
        face_centers,
        face_normals,
        "wall" => wall_faces,
        "immersed_boundary" => (
            ib_faces, # vector of indices
            ib_face_projections # matrix, each row a proj. point
        ),
        "ib_with_distance" => (
            ib_faces2,
            ib_face_projections2,
            ib_image_point_distances # vector of floats
        )
    )
    ```
    """
    function Domain(
        face_owners::AbstractVector{Ti},
        face_neighbors::AbstractVector{Ti},
        face_centers::AbstractMatrix{Tf},
        face_normals::AbstractMatrix{Tf},
        families...;
        part_index::Int64 = 1,
    ) where {Ti <: Integer, Tf <: AbstractFloat}
        face_centers = copy(face_centers)
        face_normals = copy(face_normals)
        face_owners = copy(face_owners)
        face_neighbors = copy(face_neighbors)

        # fixing o-n references
        for i = 1:length(face_owners)
            if face_owners[i] == 0
                face_owners[i] = face_neighbors[i]
            elseif face_neighbors[i] == 0
                face_neighbors[i] = face_owners[i]
            end

            if face_owners[i] == 0 && face_neighbors[i] == 0
                throw(error("Face $i with no neigh. cell"))
            end
        end

        acc = face_accumulator(face_owners, face_neighbors)

        ncells = max(maximum(face_owners), maximum(face_neighbors))
        nd = size(face_centers, 2)
        
        # calculate cell volumes and centers
        centers = similar(face_centers, (ncells, nd))
        centers .= 0
        volumes = similar(face_centers, (ncells,))
        volumes .= 0
        face_factors = similar(face_centers, (length(face_owners),))
        face_factors .= 0
        face_distances = similar(face_factors)
        face_distances .= 0
        face_areas = similar(face_factors)
        face_areas .= 0
        begin
            references = similar(centers)

            # use one neighboring face center as a nearby reference for vol. calculation
            for (o, n, c) in zip(face_owners, face_neighbors, eachrow(face_centers))
                references[o, :] .= c
                references[n, :] .= c
            end

            # calculate volume of pyramid pointing towards owner and neighbor
            volume_owners = sum(
                face_normals .* (face_centers .- references[face_owners, :]);
                dims = 2
            ) ./ nd |> vec
            volume_neighbors = sum(
                face_normals .* (references[face_neighbors, :] .- face_centers);
                dims = 2
            ) ./ nd |> vec

            @. volume_owners = abs(volume_owners)
            @. volume_neighbors = abs(volume_neighbors)

            for (o, n, vo, vn) in zip(face_owners, face_neighbors,
                volume_owners, volume_neighbors)
                volumes[o] += vo
                volumes[n] += vn * (o != n) # dont add at boundary
            end

            # calculate cell centers
            let η = 1.0f0 / (nd + 1)
                owner_wcenters = (
                    η .* references[face_owners, :] .+ 
                    (1.0f0 - η) .* face_centers
                ) .* volume_owners
                neighbor_wcenters = (
                    η .* references[face_neighbors, :] .+ 
                    (1.0f0 - η) .* face_centers
                ) .* volume_neighbors

                for (owc, nwc, o, n) in zip(
                    eachrow(owner_wcenters), eachrow(neighbor_wcenters),
                    face_owners, face_neighbors
                )
                    centers[o, :] .+= owc
                    centers[n, :] .+= nwc * (o != n) # dont add at boundary
                end

                centers ./= volumes
            end

            face_areas .= sum(face_normals .^ 2; dims = 2)
            @. face_areas = sqrt(face_areas) + 1f-14
            face_normals ./= face_areas

            # re-visit orientations
            let orientation = sum( # make normals point to neighbor centers
                face_normals .* (centers[face_neighbors, :] .- face_centers);
                dims = 2
            ) |> vec
                for i = 1:length(face_areas)
                    if orientation[i] < 0
                        face_normals[i, :] .*= (-1)
                    end
                end
            end

            # calculate interpolation factors for faces
            begin
                owner_distances = sum(
                    (face_centers .- centers[face_owners, :]) .* face_normals;
                    dims = 2
                )
                @. owner_distances = max(owner_distances, 1f-14)
                neighbor_distances = sum(
                    (centers[face_neighbors, :] .- face_centers) .* face_normals;
                    dims = 2
                )
                @. neighbor_distances = max(neighbor_distances, 1f-14)

                @. face_factors = owner_distances / (owner_distances + neighbor_distances)
                @. face_distances = owner_distances + neighbor_distances
            end
        end

        # auxiliary quantities for Laplacian calculation
        owner_neighbor_directions = centers[face_neighbors, :] .- centers[face_owners, :]
        @. owner_neighbor_directions += face_normals * (
            face_neighbors == face_owners
        ) * face_distances
        owner_neighbor_distances = sum(
            owner_neighbor_directions .^ 2; dims = 2
        ) .+ 1f-14 |> vec
        @. owner_neighbor_distances = sqrt(owner_neighbor_distances)
        owner_neighbor_directions ./= owner_neighbor_distances
        @. owner_neighbor_distances = max(owner_neighbor_distances, face_distances)
        # adding eps * face_normals makes the orientation aligned with face normals
        # at boundaries (owner = neighbor)

        boundaries = Dict{String, Boundary{Tf, Ti}}()
        let tree = KDTree(centers')
            for (fname, family) in families
                boundaries[fname] = Boundary(
                    centers, tree, face_centers,
                    _boundary_info(family, centers, # use op. overloading to get bdry info
                        face_owners, face_neighbors,
                        face_centers, face_normals)...
                )
            end
        end

        Domain{Tf, Ti}(
            centers, volumes,
            face_centers, face_normals,
            face_distances, face_factors, face_areas,
            owner_neighbor_directions, owner_neighbor_distances,
            face_owners, face_neighbors,
            acc, boundaries,
            part_index,
        )
    end

    """
    $TYPEDSIGNATURES

    Get number of cells in domain
    """
    Base.length(dom::Domain) = length(dom.volumes)

    """
    $TYPEDFIELDS

    Struct mapping to partitions of a domain
    """
    mutable struct PartitionedDomain{
        Tf <: AbstractFloat, Ti <: Integer
    } <: AbstractDomain{
        Tf, Ti
    }
        n_cells::Int64
        part_map::Dict{Int64, Tuple{Int64, UUID, Vector{Ti}, Int64}}
        family_map::Dict{String, Vector{Tuple{Vector{Ti}, Int64}}}
        conv_to_backend::Any
        conv_from_backend::Any
        lazy_conversion::Bool
    end

    """
    $TYPEDSIGNATURES

    Get number of cells in domain
    """
    Base.length(dom::PartitionedDomain) = dom.n_cells

    function _finalize_pdom(pdom::PartitionedDomain)
        for (pid, key, _, _) in values(pdom.part_map)
            if pid in procs()
                clean_stash!(pid, key)
            end
        end
    end

    """
    $TYPEDSIGNATURES

    Obtain partitions of a domain given face connectivity.

    Example/structure:

    ```
    for part in domain_partitions(
        max_partition_size,
        face_owners, face_neighbors, face_normals, skirt_order
    )
        (
            faces, # face indices in partition
            domain, # cell indices in partition
            n_skirt # number of cells (the last cells) in the skirt
        ) = part
    end
    ```
    """
    function domain_partitions(
        max_size::Int64,
        face_owners::AbstractVector{Ti},
        face_neighbors::AbstractVector{Ti},
        face_normals::AbstractMatrix{Tf},
        skirt_order::Int64
    ) where {Ti <: Integer, Tf <: AbstractFloat}
        # list of lists mapping cells to face indices
        cells2faces = let acc = face_accumulator(
            face_owners, face_neighbors,
        )
            ArrayAccumulator.list_of_lists(acc)[1]
        end

        # partitioning using METIS
        partitions, graph = let n_rounds = log2(max_size) |> floor |> Int64
            face_areas = sum(face_normals .^ 2; dims = 2) |> vec
            @. face_areas = sqrt(face_areas)

            graph, weights = owners_and_neighbors2graph(
                face_owners, face_neighbors, face_areas)

            (
                METIS.partition(
                    graph, n_rounds, weights
                ), graph
            )
        end

        map(
            image -> begin
                sort!(image)

                skirt = METIS.skirt(
                    graph, image, skirt_order
                ) # add skirt
                n_skirt = length(skirt)

                domain = Ti.([image; skirt])

                # list faces from current cells
                faces = Set{Ti}([])
                for i in domain
                    for iface in cells2faces[i]
                        push!(faces, iface)
                    end
                end
                faces = collect(faces)

                (faces, domain, n_skirt)
            end,
            partitions
        )
    end

    _select_boundary(v::AbstractVector, fmap::AbstractDict, i) = map(
        ii -> fmap[v[ii]], i
    ) # select boundary faces according to a set of face indices
    # and their mapping to a new index space
    function _select_boundary(v::Tuple, fmap::AbstractDict, i)
        e1 = _select_boundary(v[1], fmap, i)
        e2 = v[2][i, :]

        if length(v) == 2
            return (e1, e2)
        end

        e3 = v[3][i]

        (e1, e2, e3)
    end

    """
    $TYPEDSIGNATURES

    Constructor for a partitioned domain.
    The structure is the same as in `Domain()`.

    A maximum partition size of `max_partition_size` 
    is used (def. 100_000).
    `order` is used to define domain skirt depths
    and defaults to 2.

    `workers` is an optional vector of MPI workers on which to
    store the domain partitions.
    """
    function PartitionedDomain(
        face_owners::AbstractVector{Ti},
        face_neighbors::AbstractVector{Ti},
        face_centers::AbstractMatrix{Tf},
        face_normals::AbstractMatrix{Tf},
        families...;
        max_partition_size::Int = 100_000,
        order::Int = 1,
        workers::Vector{Int64} = Int64[],
        conv_to_backend = identity,
        conv_from_backend = identity,
        lazy_conversion::Bool = false,
    ) where {Ti <: Integer, Tf <: AbstractFloat}
        # run everything on the current process if not provided
        if length(workers) == 0
            workers = [myid()]
        end
        mypid = ipart -> workers[(ipart - 1) % length(workers) + 1]

        parts = domain_partitions(
            max_partition_size,
            face_owners, face_neighbors, face_normals,
            order
        )

        # dict. mapping boundaries to the
        # indices of their faces at each partition
        # also has distinction between faces of image and skirt cells
        ncells = max(maximum(face_owners), maximum(face_neighbors))
        boundary_face_selector = let cells2faces = face_accumulator(
            face_owners, face_neighbors,
        ) |> acc -> ArrayAccumulator.list_of_lists(acc)[1]
            selectors = Dict{String, Vector{Tuple{Vector{Ti}, Int64}}}()

            in_fam_indices = zeros(Ti, length(face_owners))
            for (fname, fam) in families
                if fam isa Tuple
                    fam = fam[1]
                end

                in_fam_indices[fam] .= 1:length(fam)

                selectors[fname] = [
                    begin
                        image = @view domain[1:(end - n_skirt)]
                        skirt = @view domain[(end - n_skirt + 1):end]

                        n_skirt_faces = 0
                        face_selector = Ti[]

                        for i in image
                            for f in cells2faces[i]
                                idx = in_fam_indices[f]

                                if idx != 0
                                    push!(face_selector, idx)
                                end
                            end
                        end

                        for i in skirt
                            for f in cells2faces[i]
                                idx = in_fam_indices[f]

                                if idx != 0
                                    push!(face_selector, idx)
                                    n_skirt_faces += 1
                                end
                            end
                        end

                        (face_selector, n_skirt_faces)
                    end for (_, domain, n_skirt) in parts
                ]

                in_fam_indices[fam] .= 0
            end

            selectors
        end

        futures = []
        for (ipart, (faces, domain, n_skirt)) in enumerate(parts)
            hmap = Dict(
                [d => Ti(k) for (k, d) in enumerate(domain)]...
            )

            owners = map(
                i -> (haskey(hmap, i) ? hmap[i] : Ti(0)), 
                view(face_owners, faces)
            )
            neighbors = map(
                i -> (haskey(hmap, i) ? hmap[i] : Ti(0)), 
                view(face_neighbors, faces)
            )

            fmap = Dict(
                [f => Ti(k) for (k, f) in enumerate(faces)]...
            )
            boundaries = [
                fname => _select_boundary(
                    fam, fmap, boundary_face_selector[fname][ipart][1]
                ) for (fname, fam) in families
            ]

            args = (
                owners, neighbors,
                face_centers[faces, :], face_normals[faces, :],
                boundaries...
            )

            pid = mypid(ipart)
            future = @spawnat pid begin
                dom = Domain(args...; part_index = ipart)

                if !lazy_conversion
                    dom = to_backend(dom, conv_to_backend)
                end

                key = stash!(pid, dom)

                (pid, key)
            end

            push!(futures, future)

            Base.GC.gc()
        end

        storage_info = fetch.(futures)
        for (i, s) in enumerate(storage_info)
            if s isa RemoteException
                @error "Error in partition $i !"
                throw(s)
            end
        end
        Base.GC.gc()

        part_dict = Dict{Int64, Tuple{Int64, UUID, Vector{Ti}, Int64}}()

        for (ipart, (faces, domain, n_skirt)) in enumerate(parts)
            pid, key = storage_info[ipart]

            part_dict[ipart] = (pid, key, domain, n_skirt)
        end

        pdom = PartitionedDomain{Tf, Ti}(
            ncells, part_dict, boundary_face_selector,
            conv_to_backend, conv_from_backend, lazy_conversion
        )

        finalizer(_finalize_pdom, pdom)

        pdom
    end

    """
    Select part of field array belonging to a given partition
    """
    get_partition(
        u::AbstractArray, dom::PartitionedDomain, ipart::Int
    ) = selectdim(u, 1, dom.part_map[ipart][3]) |> copy
    get_partition(
        u::Union{Pair, Tuple}, dom::PartitionedDomain, ipart::Int
    ) = u[1] => (selectdim(u[2], 1, dom.family_map[u[1]][ipart][1]) |> copy)

    """
    Set part of field array belonging to a given partition
    """
    function set_partition!(
        u::AbstractArray, dom::PartitionedDomain, ipart::Int,
        v::AbstractArray
    )
        _, _, domain, n_skirt = dom.part_map[ipart]
        image = @view domain[1:(end - n_skirt)]
        selectdim(u, 1, image) .= selectdim(v, 1, 1:(size(v, 1) - n_skirt))
        ;
    end
    function set_partition!(
        u::Union{Tuple, Pair}, dom::PartitionedDomain, ipart::Int,
        v::Union{Tuple, Pair}
    )
        bu, uf = u
        bv, vf = v

        @assert bv == bu "Inconsistent correspondence in boundary value update among partitions"

        indices, n_skirt = dom.family_map[bu][ipart]
        image_indices = @view indices[1:(length(indices) - n_skirt)]

        selectdim(uf, 1, image_indices) .= selectdim(
            vf, 1, 1:(size(vf, 1) - n_skirt)
        )
    end

    """
    $TYPEDSIGNATURES

    Run distributed residual calculation function on partitioned domain.
    Example:

    ```
    pdom = PartitionedDomain(
        args... # not gonna write this down
    )

    pdom(u, v) do subdomain, u, v
        # run operations on subdomain and partitions of u, v
        # (first dimension of arrays specifies cell index)
        # and change them in-place

        r # any return value is collected and returned in a vector
    end
    ```

    Kwargs are also forwarded to called function.

    You may also pass arrays corresponding to values at boundary faces:

    ```
    u = rand(length(dom))
    u_at_wall = zeros(length(dom, "wall"))

    dom(u, "wall" => u_at_wall) do subdom, u, pair
        bname, u_at_wall = pair # passed as a pair so you can also get 
        # the boundary name

        bdry = subdom.boundaries[bname]
        u_at_wall .= at_images(bdry, u)
    end
    ```
    """
    function (pdom::PartitionedDomain)(
        f,
        args::Union{AbstractArray, Pair, Tuple}...;
        kwargs...
    )
        lazy_conversion = pdom.lazy_conversion
        conv_to_backend = pdom.conv_to_backend
        conv_from_backend = pdom.conv_from_backend

        futures = Vector{Any}(undef, length(pdom.part_map))
        for (ipart, (pid, key, _, _)) in pdom.part_map
            pargs = map(
                a -> get_partition(a, pdom, ipart),
                args
            )

            future = @spawnat pid begin
                subdom = unstash(pid, key)

                if lazy_conversion
                    subdom = to_backend(subdom, conv_to_backend)
                end

                pargs = map(
                    a -> to_backend(a, conv_to_backend), pargs
                )

                r = subdom(f, pargs...; kwargs...)[1]

                pargs = map(
                    a -> to_backend(a, conv_from_backend), pargs
                )

                (r, pargs)
            end

            futures[ipart] = future
        end

        returns = fetch.(futures)

        for (ipart, ret) in enumerate(returns)
            if ret isa RemoteException
                @error "Error in part $ipart !"
                throw(ret)
            end
        end

        for ipart in keys(pdom.part_map)
            pargs = returns[ipart][2]

            for (a, pa) in zip(args, pargs)
                set_partition!(a, pdom, ipart, pa)
            end
        end

        map(r -> r[1], returns)
    end

    """
    $TYPEDSIGNATURES

    Dummy function to simulate a partitioned domain residual calculation
    call with a single subdomain. Allows for operator overloading over
    all child types of `AbstractDomain.`

    Has the same argument structure.
    """
    (dom::Domain)(
        f,
        args::Union{AbstractArray, Pair, Tuple}...;
        kwargs...
    ) = [
        f(dom, args...; kwargs...)
    ]

    """
    $TYPEDSIGNATURES

    Obtain number of faces at boundary given by family name
    """
    Base.length(dom::Domain, fname::String) = length(dom.boundaries[fname].faces)

    """
    $TYPEDSIGNATURES

    Obtain number of faces at boundary given by family name
    """
    Base.length(dom::PartitionedDomain, fname::String) = dom() do subdom
        length(subdom, fname)
    end |> sum

    """
    $TYPEDSIGNATURES

    Obtain values at boundary image points, given vector of cell values.
    """
    at_images(
        bdry::Boundary, u::AbstractArray
    ) = bdry.image_interpolator(u)

    """
    $TYPEDSIGNATURES

    Obtain view to values at boundary faces, given vector of face values
    """
    at_boundary(bdry::Boundary, uf::AbstractArray) = selectdim(
        uf, 1, bdry.faces
    )

    """
    $TYPEDSIGNATURES

    Interpolate values between image points and the boundary, obtaining 
    face values at a domain boundary's faces.
    """
    function interp2boundary(
        bdry::Boundary, uboundary::AbstractArray, uimage::AbstractArray
    )
        η = bdry.η

        @. η * uboundary + (1.0f0 - η) * uimage
    end

    """
    $TYPEDSIGNATURES

    Build domain from polyhedral mesh
    """
    Domain(
        mesh::PolyhedralMesh{Tf, Ti}
    ) where {Tf, Ti} = Domain(
        face_information(mesh)...,
        boundary_information(mesh)...
    )

    """
    $TYPEDSIGNATURES

    Get number of spatial dimensions in domain
    """
    Base.ndims(dom::Domain) = size(dom.centers, 2)
    
    """
    $TYPEDSIGNATURES

    Build partitioned domain from polyhedral mesh.
    Kwargs are the same as in `PartitionedDomain(...)`.
    """
    PartitionedDomain(
        mesh::PolyhedralMesh{Tf, Ti}; kwargs...
    ) where {Tf, Ti} = PartitionedDomain(
        face_information(mesh)...,
        boundary_information(mesh)...; kwargs...
    )

    """
    $TYPEDSIGNATURES

    Obtain values at face owners
    """
    at_owners(dom::Domain, u::AbstractArray) = selectdim(
        u, 1, dom.face_owners
    ) |> copy

    """
    $TYPEDSIGNATURES

    Obtain values at face neighbors
    """
    at_neighbors(dom::Domain, u::AbstractArray) = selectdim(
        u, 1, dom.face_neighbors
    ) |> copy

    """
    $TYPEDSIGNATURES

    Obtain values at faces
    """
    at_faces(dom::Domain, u::AbstractArray) = let η = dom.face_factors 
        (
            at_owners(dom, u) .* (1.0f0 .- η) .+ at_neighbors(dom, u) .* η
        )
    end

    """
    $TYPEDSIGNATURES

    Obtains Green-Gauss integral at each cell given values at
    faces.

    Applies owner-neighbor sign control if `signed = true` (default).
    """
    green_gauss(
        dom::Domain, uf::AbstractArray; signed::Bool = true
    ) = dom.face_sum(
        uf .* dom.face_areas; f = (signed ? identity : abs)
    ) ./ dom.volumes

    """
    $TYPEDSIGNATURES

    Obtain Green-Gauss gradient of `u` at cell centers
    given `u` at faces.
    Returns tuple with each dimension if `dim = 0`, or 
    a single array if the dimension is specified.
    """
    function gradient(
        dom::Domain, uf::AbstractArray, dim::Int = 0
    )
        if dim == 0
            rs = []
            for dim = 1:ndims(dom)
                push!(rs, gradient(dom, uf, dim))
            end
            return tuple(rs...)
        end

        n = @view dom.face_normals[:, dim]
        green_gauss(dom, uf .* n)
    end

    """
    $TYPEDSIGNATURES

    Obtain face gradient (tuple with a vector for each dimension)
    given cell gradient, value at face owners and value at face neighbors.
    """
    function face_gradient(
        dom::Domain, ∇u::Tuple, u_owners::AbstractArray, u_neighbors::AbstractArray
    )
        ns = dom.owner_neighbor_directions
        du = (u_neighbors .- u_owners) ./ dom.owner_neighbor_distances

        ∇uf = map(
            gu -> at_faces(dom, gu), ∇u
        )
        for (gu, n) in zip(∇uf, eachcol(ns))
            @. du -= n * gu
        end

        for (gu, n) in zip(∇uf, eachcol(ns))
            @. gu += n * du
        end

        ∇uf
    end

    """
    $TYPEDSIGNATURES

    Get divergent of vector field `u`, given as a tuple
    of face values for each component.

    Shorthand for:
    
    ```
    divergent(dom, uf) = let (ufx, ufy) = uf
        nx, ny = eachcol(dom.face_normals)

        green_gauss(dom, nx .* ufx) .+ green_gauss(ny .* ufy)
    end
    ```
    """
    divergent(dom::Domain, uf::Tuple) = [
        green_gauss(dom, uuf .* n) for (uuf, n) in zip(
            uf, eachcol(dom.face_normals)
        )
    ] |> sum

    """
    $TYPEDSIGNATURES

    Minmod operator
    """
    @inline minmod(u1::Real, u2::Real) = (sign(u1) + sign(u2)) / 2 * min(abs(u1), abs(u2))

    """
    $TYPEDSIGNATURES

    Obtain MUSCL reconstruction at faces.

    From cell data and cell gradients (tuple as returned
    by `gradient(dom, u)`), obtain `uL, uR` at left and right
    sides of faces.
    Uses minmod limiter.

    Artificial dissipation sensor `D` at cells may be given
    so that `D = 1` corresponds to MUSCL reconstruction, and `D = 0`,
    to a perfectly centered scheme.
    """
    function MUSCL(
        dom::Domain, u::AbstractArray, ∇u::Tuple;
        D::Union{AbstractArray, Nothing} = nothing,
    )
        ∇uo = map(du -> at_owners(dom, du), ∇u)
        ∇un = map(du -> at_neighbors(dom, du), ∇u)

        uo = at_owners(dom, u)
        un = at_neighbors(dom, u)

        xf = dom.face_centers
        xo = at_owners(dom, dom.centers)
        xn = at_neighbors(dom, dom.centers)

        bu = similar(uo)
        bu .= 0
        for (du, xxo, xxf) in zip(
            ∇uo, eachcol(xo), eachcol(xf)
        )
            @. bu += du * (xxf - xxo)
        end

        fu = similar(uo)
        fu .= 0
        for (du, xxf, xxn) in zip(
            ∇un, eachcol(xf), eachcol(xn)
        )
            @. fu += du * (xxn - xxf)
        end

        @. bu = 2 * bu - (un - uo) / 2
        @. fu = 2 * fu - (un - uo) / 2

        grad = @. minmod(bu, fu)

        uL = @. uo + grad
        uR = @. un - grad

        if !isnothing(D)
            Df = max.(
                at_owners(dom, D), at_neighbors(dom, D)
            )

            uf = at_faces(dom, u)
            @. uL = uL * Df + (1.0f0 - Df) * uf
            @. uR = uR * Df + (1.0f0 - Df) * uf
        end

        (uL, uR)
    end

    """
    $TYPEDSIGNATURES

    Obtain JST shock sensor for generic flow properties
    """
    function CFD.JST_sensor(
        dom::Domain, u::AbstractArray
    )
        δu = at_neighbors(dom, u) .- at_owners(dom, u)

        (
            abs.(
                green_gauss(dom, δu; signed = true)
            ) .+ 1f-14
        ) ./ (
            green_gauss(dom, δu; signed = false) .+ 1f-14
        )
    end

    """
    Obtain coarsener and prolongator given groups as lists of lists
    of indices, as well as volumes.
    """
    function coarsener_and_prolongator(
        volumes::AbstractVector,
        groups_fine::AbstractVector{Vector{Ti}}, 
        groups_coarse::AbstractVector{Vector{Ti}}
    ) where {Ti <: Integer}
        partid_fine = zeros(Ti, length(volumes))
        partid_coarse = zeros(Ti, length(volumes))

        for (k, group) in enumerate(groups_fine)
            partid_fine[group] .= Ti(k)
        end
        for (k, group) in enumerate(groups_coarse)
            partid_coarse[group] .= Ti(k)
        end

        volumes_fine = map(
            group -> sum(i -> volumes[i], group),
            groups_fine
        )
        volumes_coarse = map(
            group -> sum(i -> volumes[i], group),
            groups_coarse
        )

        coarsener = begin
            coarsener_stencils = map(
                group -> unique(
                    view(partid_fine, group)
                ), groups_coarse
            )
            coarsener_weights = map(
                (st, vc) -> volumes_fine[st] ./ vc,
                coarsener_stencils, volumes_coarse
            )

            Accumulator(
                coarsener_stencils, coarsener_weights; first_index = true
            )
        end

        prolongator = let stencils = map(
            group -> [partid_coarse[group[1]]], groups_fine
        )
            Accumulator(stencils; first_index = true)
        end

        (coarsener, prolongator)
    end

    """
    Coarsen face and boundary information given groups vector.
    """
    function coarsen_face_info(
        groups::AbstractVector{Vector{Ti}},
        face_owners::AbstractVector{Ti},
        face_neighbors::AbstractVector{Ti},
        face_centers::AbstractMatrix{Tf},
        face_normals::AbstractMatrix{Tf},
        families...
    ) where {Tf <: AbstractFloat, Ti <: Integer}
        ncells = max(maximum(face_owners), maximum(face_neighbors))
        partid = zeros(Ti, ncells)

        for (k, group) in enumerate(groups)
            partid[group] .= Ti(k)
        end

        # keep faces connecting two groups
        face_mask = view(partid, face_owners) .!= view(partid, face_neighbors)
        # always keep external boundary faces
        @. face_mask = (face_owners == face_neighbors) || face_mask
        face_newinds = Ti.(cumsum(face_mask))

        face_owners = partid[face_owners[face_mask]]
        face_neighbors = partid[face_neighbors[face_mask]]
        face_centers = face_centers[face_mask, :]
        face_normals = face_normals[face_mask, :]

        _families = []
        for (fname, fam) in families
            if fam isa Tuple
                indices = fam[1]
                this_mask = face_mask[indices]

                indices = face_newinds[indices[this_mask]]

                if length(fam) == 2
                    fam = (
                        indices, fam[2][this_mask, :]
                    )
                else # 3
                    fam = (
                        indices, fam[2][this_mask, :], fam[3][this_mask]
                    )
                end
            else
                indices = fam
                this_mask = face_mask[indices]

                indices = face_newinds[indices[this_mask]]

                fam = indices
            end

            push!(_families, fname => fam)
        end

        (
            face_owners, face_neighbors,
            face_centers, face_normals,
            _families...
        )
    end

    """
    $TYPEDSIGNATURES

    Obtain domain, vector of coarse domains,
    vector of coarseners and vector of prolongators.

    Coarseners and prolongators are callables such that `coarseners[i]`
    coarsens properties from level `i` to level `i - 1`, and the opposite
    is true of `prolongators[i]`.

    The interpolations are used as:

    ```
    dom, coarse_doms, coarseners, prolongators = MultigridDomain(
        n_levels,
        args...;
        partitioned = true, # def. false
        max_partition_size = 100_000, # def. true
        kwargs... # any others passed to domain construction.
    )

    P = rand(length(dom), 5) # array, first index identifies cell

    Pc = coarseners[1](P)
    P .= prolongators[1](Pc)
    ```

    Check out `JuFOAM.Solver.FAS!` to use them for PDE solutions!

    `n_coarsening_iter` is the number of p-METIS coarsening iterations
    per level. If absent, defaults to the dimensionality of the domain
    times two (factor of 4 for grid spacing).
    """
    function MultigridDomain(
        n_levels::Int, 
        face_owners::AbstractVector{Ti},
        face_neighbors::AbstractVector{Ti},
        face_centers::AbstractMatrix{Tf},
        face_normals::AbstractMatrix{Tf},
        families...;
        partitioned::Bool = false,
        max_partition_size::Int = 250_000,
        order::Int = 1,
        workers::Vector{Int64} = Int64[],
        conv_to_backend = identity,
        conv_from_backend = identity,
        lazy_conversion::Bool = false,
        n_coarsening_iter::Int = 0,
    ) where {Tf <: AbstractFloat, Ti <: Integer}
        if n_coarsening_iter == 0
            n_coarsening_iter = size(face_normals, 2) * 2
        end

        to_domain = (a...) -> (
            partitioned ? 
            PartitionedDomain(
                a...;
                max_partition_size = max_partition_size, order = order,
                workers = workers, conv_to_backend = conv_to_backend,
                conv_from_backend = conv_from_backend, lazy_conversion = lazy_conversion,
            ) :
            Domain(
                a...
            )
        )

        dom = to_domain(
            face_owners, face_neighbors, face_centers, face_normals,
            families...
        )
        Base.GC.gc()

        # catch volumes for coarsening/prolongation
        volumes = zeros(Tf, length(dom))
        dom(volumes) do dom, volumes
            volumes .= dom.volumes
        end

        groups = map(
            i -> [Ti(i)], 1:length(dom)
        )
        graph, weights = owners_and_neighbors2graph(
            face_owners, face_neighbors, 
            sum(face_normals .^ 2; dims = 2) |> x -> sqrt.(x) |> vec, 
        )
        Base.GC.gc()

        coarseners = Accumulator[]
        prolongators = Accumulator[]
        coarse_domains = AbstractDomain[]
        for _ = 1:n_levels
            old_groups = groups
            for _ = 1:n_coarsening_iter
                groups = METIS.coarsen(graph, groups, weights)
            end

            finfo = coarsen_face_info(groups, 
                    face_owners, face_neighbors,
                    face_centers, face_normals,
                    families...)

            coarse_dom = to_domain(
                finfo...
            )
            Base.GC.gc()

            coarsener, prolongator = coarsener_and_prolongator(
                volumes, old_groups, groups,
            )

            push!(coarseners, coarsener)
            push!(prolongators, prolongator)
            push!(coarse_domains, coarse_dom)
        end

        (dom, coarse_domains, coarseners, prolongators)
    end

    """
    $TYPEDSIGNATURES

    Other method for `MultigridDomain` which takes in a polyhedral mesh.
    Receives the same keyword arguments as the first method.
    """
    MultigridDomain(
        n_levels::Int,
        msh::PolyhedralMesh; kwargs...
    ) = MultigridDomain(
        n_levels,
        face_information(msh)..., boundary_information(msh)...;
        kwargs...
    )

    """
    $TYPEDSIGNATURES

    Obtain interpolator callable object for points within a domain.

    Example:

    ```julia
    u = zeros(length(dom))
    dom(u) do subdom, u
        u .= subdom.centers[:, 1] .+ 2 .* subdom.centers[:, 2]
        ;
    end

    X = [
        0.5 0.7;
        0.2 0.6;
        0.1 0.0
    ] # what's going on with variable u at these three points?

    intp = Interpolator(dom, X; 
        linear = false, # default true, else uses inverse distance weighing
        k = 3) # number of interpolation points. Defaults to dimensionality + 1

    uX = intp(u)
    # [1.9, 1.4, 0.1]
    ```
    """
    function NNInterpolator.Interpolator(
        dom::AbstractDomain{Tf, Ti}, Xc::AbstractMatrix;
        linear::Bool = true, k::Int = 0,
    ) where {Tf, Ti}
        ndim = size(Xc, 2)
        if k == 0
            k = ndim + 1
        end

        X = Matrix{Tf}(undef, length(dom), ndim)
        dom(X) do dom, X
            X .= dom.centers
        end

        tree = KDTree(X')

        Interpolator(X, Xc, tree; first_index = true,
            k = k, linear = linear)
    end

end
