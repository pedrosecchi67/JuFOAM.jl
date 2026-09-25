begin
    @info "Running unstructured grid data organization test..."

    msh = PolyhedralMesh(
        [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], (10, 10, 10);
        families = [
            "surface" => [(1, false), (2, false), (3, false)]
        ]
    )

    dom = PartitionedDomain(msh; max_partition_size = 100)

    u = zeros(length(dom))
    p = zeros(length(dom))
    uf = zeros(length(dom, "surface"))
    dom(u, p, "surface" => uf) do dom, u, p, pair
        p .= dom.part_index
        u .= sum(dom.centers .^ 2; dims = 2)

        bname, uf = pair
        bdry = dom.boundaries[bname]
        uf .= at_images(bdry, u)
    end

    intp = Interpolator(
        dom,
        [0.5 0.5 0.5;]
    )
    @assert intp(u) ≈ [0.75]

    vtk = vtk_grid("unstructured_test/volume", msh)
    vtk["u"] = u
    vtk["p"] = p
    vtk_save(vtk)

    vtk = vtk_grid("unstructured_test/surface", msh, "surface")
    vtk["u"] = uf
    vtk_save(vtk)
end