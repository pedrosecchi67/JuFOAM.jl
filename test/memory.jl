begin
    @info "Begining memory usage test..."

    tests = [
        "partitioned" => (a...) -> PartitionedDomain(
            a...; max_partition_size = 100_000,
        ),
        "serial" => (a...) -> Domain(a...),
        "multigrid" => (a...) -> MultigridDomain(
            4, a...; partitioned = true,
            max_partition_size = 100_000,
        ),
    ]

    for (test_name, foo) in tests
        @info "Running test in mode: $test_name"
        for nit = 1:2
            @info "Iteration $nit..."

            local N

            N = 1_000_000
            Nside = N ^ (1.0f0 / 3) |> ceil |> Int64

            Base.GC.gc()

            _, prof = @memory_profile 1.0 begin
                msh = PolyhedralMesh(
                    Float64[0.0, 0.0, 0.0], Float64[1.0, 1.0, 1.0], (Nside, Nside, Nside);
                    families = [
                        "inlet" => [(1, false), (2, false), (2, true), (3, false), (3, true)],
                        "outlet" => [(1, true)]
                    ]
                )

                @info "Done generating $N-element mesh"
                @info "Processing domain..."

                let r = foo(msh)
                    if test_name == "partitioned"
                        finalize(r)
                    elseif test_name == "multigrid"
                        finalize(r[1])
                    end
                end

                Base.GC.gc()
            end

            Base.GC.gc()

            toMb = rss -> rss ÷ 2 ^ 20

            max_usage = maximum(prof.gc_live_bytes) |> toMb
            curr_usage = prof.gc_live_bytes[end] |> toMb
            rtime = prof.wall_time[end]

            @info """
            Max. GC live memory (MB): $max_usage
            Last GC live memory (MB): $curr_usage
            Run time (s): $rtime
            """
        end
    end
end
