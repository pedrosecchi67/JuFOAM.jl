"""
```
    macro memory_profile(
        interval::Real e::Expr
    )
```

Macro for memory profiling.
Makes asynchronous memory measurements once every `interval` seconds
and returns the task return value, and a named tuple with vector 
entries `wall_time` and `gc_live_bytes`.
"""
macro memory_profile(
    interval::Real, e::Expr
)
    quote
        rtime = Float64[]
        rss = Int64[]

        task = @async $(esc(e))

        while !istaskdone(task)
            push!(rtime, time())
            push!(rss, Base.gc_live_bytes())

            sleep($(esc(interval)))
        end

        rtime .-= rtime[1]

        (
            fetch(task), (
                wall_time = rtime,
                gc_live_bytes = rss,
            )
        )
    end
end