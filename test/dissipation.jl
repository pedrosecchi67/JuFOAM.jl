begin
    @info "Running dissipation test case..."

    msh = PolyhedralMesh(
        [0.0, 0.0], [1.0, 1.0], (100, 100);
        families = [
            "one" => [(1, false)],
            "zero" => [(2, false), (2, true), (1, true)],
        ]
    )

    dom = PartitionedDomain(msh; max_partition_size = 2000, order = 3)

    u = zeros(length(dom))

    residual = u -> begin
        udot = similar(u)        
        udot .= 0
        dt = similar(u)
        dt .= 0

        dom(udot, u, dt) do dom, udot, u, dt
            cosθ = sum(
                dom.face_normals .* dom.owner_neighbor_directions; dims = 2
            ) |> vec
            a = cosθ ./ dom.owner_neighbor_distances

            dt .= 0.5f0 ./ green_gauss(dom, a; signed = false)

            uf = at_faces(dom, u)

            let bdry = dom.boundaries["one"]
                at_boundary(bdry, uf) .= 1
            end
            let bdry = dom.boundaries["zero"]
                at_boundary(bdry, uf) .= 0
            end

            ∇u = gradient(dom, uf)

            uo = at_owners(dom, u)
            un = at_neighbors(dom, u)

            let bdry = dom.boundaries["one"]
                at_boundary(bdry, uo) .= 1
            end
            let bdry = dom.boundaries["zero"]
                at_boundary(bdry, uo) .= 0
            end

            ∇uf = face_gradient(dom, ∇u, uo, un)

            udot .= divergent(dom, ∇uf)
        end

        (udot, dt)
    end

    for _ = 1:600
        ud, dt = residual(u)
        @. u += dt * ud
    end

    partid = zeros(length(dom))
    dom(partid) do dom, partid
        partid .= dom.part_index
    end

    vtk = vtk_grid("dissipation/volume", msh)
    vtk["u"] = u
    vtk["partid"] = partid
    vtk_save(vtk)

    n = zeros(length(dom, "one"), 2)
    dom("one" => n) do dom, (bname, n)
        let bdry = dom.boundaries[bname]
            n .= at_boundary(bdry, dom.owner_neighbor_directions)
        end
    end

    vtk = vtk_grid("dissipation/surface", msh, "one")
    vtk["n"] = n
    vtk_save(vtk)
end