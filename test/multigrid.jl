begin
    @info "Running multigrid tests..."

    msh = PolyhedralMesh([0.0, 0.0], [1.0, 1.0], (100, 100);
        families = ["wall" => [(1, false)]])

    dom, coarse_doms, coarseners, prolongators = MultigridDomain(
        3, msh; partitioned = true, max_partition_size = 2000,
    )

    u = zeros(length(dom))
    dom(u) do dom, u
        u .= sum(dom.centers .^ 2; dims = 2)
    end

    uc = copy(u)
    for i = 1:length(coarse_doms)
        global uc

        uc = coarseners[i](uc)
        @assert length(uc) == length(coarse_doms[i])
    end
    for i = length(coarse_doms):-1:1
        global uc

        uc = prolongators[i](uc)
    end

    vtk = vtk_grid("multigrid/volume", msh)
    vtk["u"] = u
    vtk["uc"] = uc
    vtk_save(vtk)
end