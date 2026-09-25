begin
    @info "Running backend conversion test..."

    msh = PolyhedralMesh(
        [0.0, 0.0], [1.0, 1.0], (100, 100);
        families = [
            "surf" => [(1, false)],
        ]
    )

    ncalls = 0
    ncalls_back = 0
    conv_to_backend = x -> begin
        global ncalls 
        ncalls += 1
        x
    end
    conv_from_backend = x -> begin
        global ncalls_back
        ncalls_back += 1
        x
    end

    dom = PartitionedDomain(
        msh;
        max_partition_size = 2000,
        conv_to_backend = conv_to_backend,
        conv_from_backend = conv_from_backend,
        lazy_conversion = true
    )

    nparts = dom.part_map |> length

    dom() do dom
    end

    ncalls = ncalls ÷ nparts
    ncalls_back = ncalls_back ÷ nparts

    ncalls_instruct = ncalls

    @assert ncalls > 0
    @assert ncalls_back == 0

    ncalls = 0
    ncalls_back = 0

    u = zeros(length(dom))
    uf = zeros(length(dom, "surf"))

    dom(u, "surf" => uf) do dom, u, (_, uf)
    end

    ncalls = ncalls ÷ nparts
    ncalls_back = ncalls_back ÷ nparts

    @assert ncalls_back == 2
end