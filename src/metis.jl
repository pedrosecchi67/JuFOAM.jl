module METIS

    using DocStringExtensions

    """
    $TYPEDSIGNATURES

    Coarsen a graph partitioning.

    The graph should be indicated as a list of lists of neighbor IDs.
    """
    function coarsen(
        graph::AbstractVector,
        partitioning::AbstractVector,
        weights::Union{AbstractVector, Nothing} = nothing
    )

        # sort by length to prioritize merging small partitions
        partsizes = length.(partitioning)
        partitioning = partitioning[sortperm(partsizes)]

        nelements = length(graph)
        nparts = length(partitioning)

        # find node part ID vector
        partid = zeros(Int64, nelements)
        for (id, part) in enumerate(partitioning)
            partid[part] .= id
        end

        # which partitions are still available for merging?
        isavailable = trues(nparts)
        # to which partition will each one merge?
        merge2 = zeros(Int64, nparts)
        # which will remain after merging?
        remains = trues(nparts)

        # for each partition...
        for (npart, partition) in enumerate(partitioning)
            # if it's not available, continue
            if !isavailable[npart]
                continue
            end

            # count connections to each available partition
            count = Dict{Int64, Float64}()

            countme = (n, w) -> begin
                nid = partid[n]

                if nid != npart
                    if isavailable[nid]
                        if !haskey(count, nid)
                            count[nid] = 0.0
                        end

                        count[nid] += w
                    end
                end
            end

            if isnothing(weights)
                for node in partition
                    for n in graph[node]
                        countme(
                            n, 1.0
                        )
                    end
                end
            else
                for node in partition
                    for (n, w) in zip(graph[node], weights[node])
                        countme(
                            n, w
                        )
                    end
                end
            end

            # if we find no available connections:
            if length(count) == 0
                continue
            end

            # find the partition with the most connections
            nmax = 0.0
            partmax = 0
            for (k, v) in count
                if v > nmax
                    nmax = v
                    partmax = k
                end
            end

            # register
            merge2[npart] = partmax
            isavailable[npart] = false
            isavailable[partmax] = false
            remains[partmax] = false
        end

        newpartitions = [
            (
                m2 == 0 ?
                part : [part; partitioning[m2]]
            ) for (m2, part) in zip(merge2[remains], partitioning[remains])
        ]

        newpartitions

    end

    """
    $TYPEDSIGNATURES

    Run `n_rounds` of coarsening on a graph and
    return list of lists of node IDs with the resulting
    partitions.

    You may estimate the number of partitions as `length(graph) / 2 ^ n_rounds`
    """
    function partition(
        graph::AbstractVector,
        n_rounds::Int,
        weights::Union{Nothing, AbstractVector} = nothing
    )

        partitions = map(i -> [i], 1:length(graph))

        for _ = 1:n_rounds
            partitions = coarsen(graph, partitions, weights)
        end

        partitions

    end

    """
    $TYPEDSIGNATURES

    Similar to `partition`, but returns a vector with the
    partitioning state at each coarsening round.
    """
    function grid_levels(
        graph::AbstractVector,
        n_rounds::Int, 
        weights::Union{Nothing, AbstractVector} = nothing,
    )

        partitions = map(i -> [i], 1:length(graph))

        state = [partitions]

        for _ = 1:n_rounds
            partitions = coarsen(graph, partitions, weights)

            push!(state, partitions)
        end

        state

    end

    """
    $TYPEDSIGNATURES

    Obtain connectivity graph between blocks as the ones returned by
    `partition`.

    Returns a new graph identifying connectivity in the coarsened graph.
    """
    function block_connectivity(graph::AbstractVector, blocks::AbstractVector)

        blockids = Vector{Int64}(undef, length(graph))

        for (i, blck) in enumerate(blocks)
            blockids[blck] .= i
        end

        map(
            blck -> let s = Set{Int64}()
                register! = (i, j) -> begin
                    ib = blockids[i]
                    jd = blockids[j]

                    if i != j && ib != jd
                        push!(s, jd)
                    end
                end

                for i in blck
                    for j in graph[i]
                        register!(i, j)
                    end
                end

                collect(s)
            end,
            blocks
        )

    end

    """
    $TYPEDSIGNATURES

    Obtain graph nodes within a given distance of `d` edges from a group
    """
    function skirt(graph::AbstractVector, group::AbstractVector, d::Int = 1)
        skirt_nodes = Set(eltype(group)[])
        group_set = Set(group)

        for i in group_set
            for neigh in graph[i]
                if !(neigh in group_set)
                    push!(skirt_nodes, neigh)
                end
            end
        end

        for _ = 2:d
            for i in collect(skirt_nodes)
                for neigh in graph[i]
                    if !(neigh in group_set)
                        push!(skirt_nodes, neigh)
                    end
                end
            end
        end

        collect(skirt_nodes)
    end

end