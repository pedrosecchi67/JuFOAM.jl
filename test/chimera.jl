begin
    @info "Running chimera grid test..."

    msh = PolyhedralMesh(
        PolyhedralMesh([0.0, 0.0], [0.51, 1.0], (50, 100);
            families = [
                "upper" => [(1, false)],
                "lower" => [(2, false)],
                "outlet" => [(2, true)]
            ]
        ),
        PolyhedralMesh([0.49, 0.0], [0.51, 1.0], (50, 100);
            families = [
                "lower" => [(2, false)],
                "outlet" => [(1, true), (2, true)]
            ]
        )
    )

    CFL = 0.75f0

    residual = (dom, u, chimera; low_order = false) -> begin
        C = ones(length(dom), 2)

        udot = similar(u)
        udot .= 0
        dt = similar(u)
        dt .= 0

        dom(udot, u, C, dt, "ORPHAN" => chimera(u)) do dom, udot, u, C, dt, (bname, uch)
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

            bdry = dom.boundaries[bname]
            at_boundary(bdry, uL) .= uch

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

    dom = Domain(msh)
    chimera = ChimeraInterpolator(dom, "ORPHAN")

    u = zeros(length(dom))
    
    for _ = 1:700
        ud, dt = residual(dom, u, chimera)
        @. u += dt * ud
    end

    vtk = vtk_grid("chimera/volume", msh)
    vtk["u"] = u
    vtk_save(vtk)
end
