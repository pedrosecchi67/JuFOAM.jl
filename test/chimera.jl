begin
    @info "Running chimera grid test..."

    msh = PolyhedralMesh(
        PolyhedralMesh([0.0, 0.0], [0.5, 1.0], (50, 100)),
        PolyhedralMesh([0.5, 0.0], [0.5, 1.0], (50, 100))
    )

    vtk = vtk_grid("chimera/volume", msh)
    vtk_save(vtk)
end
