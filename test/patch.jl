begin
    @info "Running patched advection test case..."

    msh = PolyhedralMesh(
        PolyhedralMesh([0.0, 0.0], [0.5, 1.0], (50, 100);
            families = [
                "upper" => [(1, false)],
                "lower" => [(2, false)],
                "outlet" => [(2, true)]
            ]
        ),
        PolyhedralMesh([0.5, 0.0], [0.5, 1.0], (50, 100);
            families = [
                "lower" => [(2, false)],
                "outlet" => [(1, true), (2, true)]
            ]
        )
    )
    patch_faces!(msh, "ORPHAN")

    CFL = 0.75f0

    residual = (dom, u; low_order = false) -> begin
        C = ones(length(dom), 2)

        udot = similar(u)
        udot .= 0
        dt = similar(u)
        dt .= 0

        dom(udot, u, C, dt) do dom, udot, u, C, dt
            uL = uR = nothing

            if low_order
                uL = at_owners(dom, u)
                uR = at_neighbors(dom, u)
            else
                uf = at_faces(dom, u)
                at_boundary(dom.boundaries["upper"], uf) .= 1
                at_boundary(dom.boundaries["lower"], uf) .= 0
                at_boundary(dom.boundaries["outlet"], uf) .= at_images(dom.boundaries["outlet"], u)
                
                uL, uR = let ∇u = gradient(dom, uf)
                    ν = JST_sensor(dom, u)
                    MUSCL(dom, u, ∇u; D = ν)
                end
            end

            Cf = at_faces(dom, C)
            ϕ = sum(dom.face_normals .* Cf; dims = 2) |> vec

            at_boundary(dom.boundaries["upper"], uL) .= 1
            at_boundary(dom.boundaries["lower"], uL) .= 0
            at_boundary(dom.boundaries["outlet"], uL) .= at_images(dom.boundaries["outlet"], u)

            dt .= CFL ./ green_gauss(dom, ϕ; signed = false)

            udot .= - green_gauss(
                dom, (
                    @. (uL + uR) / 2 * ϕ + (uL - uR) / 2 * abs(ϕ)
                )
            )
        end

        (udot, dt)
    end

    dom, coarse_doms, coarseners, prolongators = MultigridDomain(
        3, msh; 
        max_partition_size = 2000)

    f = (l, u) -> begin
        cdom = (
            l == 0 ? dom : coarse_doms[l]
        )

        ud, dt = residual(cdom, u; low_order = l > 0)

        (ud .* dt, 0.5f0)
    end

    u = zeros(length(dom))

    for nit = 1:20
        println("Iteration $nit..")

        FAS!(
            f, u;
            coarseners = coarseners, prolongators = prolongators,
            n_iter = 20, n_cycles = 4, rtol = 0.01f0
        )
    end

    ν = similar(u)
    ν .= 0
    dom(u, ν) do dom, u, ν
        ν .= JST_sensor(dom, u)
    end

    vtk = vtk_grid("patch/volume", msh)
    vtk["u"] = u
    vtk["JST"] = ν
    vtk_save(vtk)
end
