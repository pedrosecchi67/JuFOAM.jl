module CGNSReader

    include("vtk2rings.jl")
    using .VTKConversion

    using DocStringExtensions

    using HDF5

    export read_cgns

    face2tag(face::AbstractVector{Ti}) where {Ti <: Integer} = tuple(sort(face)...)

    """
    $TYPEDSIGNATURES

    Read CGNS file and return volumes and families in ring notation.
    Returns named tuple with:

    ```
    (
        points # matrix, (npts, ndims)
        faces # list of lists of node indices
        cells # list of lists of face indices
        families # dict from string to list of face indices
        zones # dict from string to list of cell indices
    )
    ```
    """
    function read_cgns(
        fname::String
    )
        readdata = x -> read(x)[" data"]

        dset = h5open(fname)

        base = dset["Base"]

        zone_names = [
            zname for zname in keys(base) if zname != " data"
        ]

        face_ledger = Dict{Tuple, Int64}()
        faces = AbstractVector[]
        cells = AbstractVector[]
        push_face! = f -> let tag = face2tag(f)
            (
                haskey(face_ledger, tag) ?
                face_ledger[tag] :
                begin
                    n = length(faces) + 1
                    face_ledger[tag] = n
                    push!(faces, f)
                    n
                end
            )
        end

        pts = []
        families = Dict{String, AbstractVector}()
        zones = Dict{String, AbstractVector}()
        npts = 0
        ncells = 0
        for zname in zone_names
            zone = base[zname]

            if haskey(zone["GridCoordinates"], "CoordinateZ")
                push!(
                    pts, 
                    hcat(
                        zone["GridCoordinates"]["CoordinateX"] |> readdata,
                        zone["GridCoordinates"]["CoordinateY"] |> readdata,
                        zone["GridCoordinates"]["CoordinateZ"] |> readdata,
                    )
                )
            else
                push!(
                    pts, 
                    hcat(
                        zone["GridCoordinates"]["CoordinateX"] |> readdata,
                        zone["GridCoordinates"]["CoordinateY"] |> readdata,
                    )
                )
            end

            face_indices = nothing
            if haskey(zone, "NGonElements") # polyhedral
                # reading faces
                begin
                    offsets = zone["NGonElements"]["ElementStartOffset"] |> readdata
                    conns = zone["NGonElements"]["ElementConnectivity"] |> readdata

                    face_indices = map(
                        i -> npts .+ 
                            (
                                conns[(offsets[i] + 1):min(offsets[i + 1], length(conns))]
                            ) |> push_face!,
                        1:(length(offsets) - 1)
                    )
                end

                # reading volumes
                begin
                    offsets = zone["NFaceElements"]["ElementStartOffset"] |> readdata
                    conns = zone["NFaceElements"]["ElementConnectivity"] |> readdata

                    this_cells = map(
                        i -> let conn = abs.(
                            conns[(offsets[i] + 1):min(offsets[i + 1], length(conns))]
                        )
                            face_indices[conn]
                        end,
                        1:(length(offsets) - 1)
                    )

                    push!(cells, this_cells)

                    zones[zname] = ncells .+ (1:length(this_cells))
                end
            else # unstructured
                # reading faces
                begin
                    offsets = zone["SurfaceElements"]["ElementStartOffset"] |> readdata
                    conns = zone["SurfaceElements"]["ElementConnectivity"] |> readdata

                    face_indices = map(
                        i -> npts .+ 
                            (
                                conns[(offsets[i] + 2):min(offsets[i + 1], length(conns))]
                            ) |> push_face!,
                        1:(length(offsets) - 1)
                    )
                end

                # reading volumes
                begin
                    offsets = zone["VolumeElements"]["ElementStartOffset"] |> readdata
                    conns = zone["VolumeElements"]["ElementConnectivity"] |> readdata

                    this_cells = map(
                        i -> let conn = npts .+ conns[(offsets[i] + 2):min(offsets[i + 1], length(conns))]
                            t = conns[offsets[i] + 1]

                            this_faces = vtk2poly(t, conn)
                            push_face!.(this_faces)
                        end,
                        1:(length(offsets) - 1)
                    )

                    push!(cells, this_cells)

                    zones[zname] = ncells .+ (1:length(this_cells))
                end
            end

            for fname in keys(zone["ZoneBC"])
                fcs = nothing

                if haskey(zone["ZoneBC"][fname], "PointList")
                    fcs = face_indices[readdata(zone["ZoneBC"][fname]["PointList"])] |> vec
                else
                    rng = readdata(zone["ZoneBC"][fname]["PointRange"])
                    fcs = face_indices[rng[1]:rng[2]]
                end

                if haskey(families, fname)
                    families[fname] = vcat(families[fname], fcs)
                else
                    families[fname] = fcs
                end
            end

            npts += size(pts[end], 1)
            ncells += length(zones[zname])
        end
        pts = reduce(vcat, pts)
        cells = reduce(vcat, cells)
        faces = identity.(faces)

        close(dset)

        (
            points = pts,
            faces = faces,
            cells = cells,
            families = families,
            zones = zones,
        )
    end

end