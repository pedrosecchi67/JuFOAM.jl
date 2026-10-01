module VTKConversion

    using DocStringExtensions

    export vtk2poly, VTKCellType

    @enum VTKCellType::Int8 begin
        VTK_LINE = 3
        VTK_POLY_LINE = 4
        VTK_TRIANGLE = 5
        VTK_POLYGON = 7
        VTK_PIXEL = 8
        VTK_QUAD = 9
        VTK_TETRA = 10
        VTK_VOXEL = 11
        VTK_HEXAHEDRON = 12
        VTK_WEDGE = 13
        VTK_PYRAMID = 14
    end

    """
    ```
        vtk2poly(t::VTKCellType, conn::AbstractVector{Ti}) where {Ti <: Integer}
    ```

    Returns list of lists with vtk cell faces as rings of point indices
    """
    function vtk2poly(
        t::VTKCellType, conn::AbstractVector{Ti}
    ) where {Ti <: Integer}
        if t == VTK_LINE || t == VTK_POLY_LINE
            return [[conn[1]], [conn[end]]]
        elseif t == VTK_TRIANGLE || t == VTK_QUAD || t == VTK_POLYGON
            rings = Vector{Ti}[]
            for i = 1:length(conn)
                inext = i % length(conn) + 1
                push!(rings, [conn[i], conn[inext]])
            end

            return rings
        elseif t == VTK_PIXEL
            return [
                [conn[2], conn[1]],
                [conn[3], conn[4]],
                [conn[1], conn[3]],
                [conn[4], conn[2]]
            ]
        elseif t == VTK_TETRA
            return [
                [conn[1], conn[3], conn[2]],
                [conn[2], conn[3], conn[4]],
                [conn[1], conn[2], conn[4]],
                [conn[1], conn[4], conn[3]]
            ]
        elseif t == VTK_VOXEL
            return [
                [conn[2], conn[1], conn[3], conn[4]],
                [conn[5], conn[6], conn[8], conn[7]],
                [conn[1], conn[2], conn[6], conn[5]],
                [conn[2], conn[4], conn[8], conn[6]],
                [conn[4], conn[3], conn[7], conn[8]],
                [conn[3], conn[1], conn[5], conn[7]]
            ]
        elseif t == VTK_HEXAHEDRON
            return [
                [conn[1], conn[4], conn[3], conn[2]],
                [conn[1], conn[2], conn[6], conn[5]],
                [conn[2], conn[3], conn[7], conn[6]],
                [conn[3], conn[4], conn[8], conn[7]],
                [conn[4], conn[1], conn[5], conn[8]],
                [conn[5], conn[6], conn[7], conn[8]],
            ]
        elseif t == VTK_WEDGE
            return [
                [conn[1], conn[2], conn[3]],
                [conn[4], conn[6], conn[5]],
                [conn[1], conn[3], conn[6], conn[4]],
                [conn[1], conn[4], conn[5], conn[2]],
                [conn[2], conn[5], conn[6], conn[3]],
            ]
        elseif t == VTK_PYRAMID
            return [
                [conn[1], conn[4], conn[3], conn[2]],
                [conn[1], conn[2], conn[5]],
                [conn[2], conn[3], conn[5]],
                [conn[3], conn[4], conn[5]],
                [conn[4], conn[1], conn[5]],
            ]
        else
            error("VTK cell type $t not supported") |> throw
        end
    end

    """
    ```
        vtk2poly(t::Integer, conn::AbstractVector{Ti}) where {Ti <: Integer}
    ```

    Returns list of lists with vtk cell faces as rings of point indices
    """
    vtk2poly(t::Integer, conn::AbstractVector{Ti}) where {Ti <: Integer} = vtk2poly(
        VTKCellType(t), conn
    )

end