"""
Module to obtain the closest point to each query point
in each quadrant. Can be used for meshless methods.
"""
module QuadrantNN

        using LinearAlgebra
        using Statistics

        using Base.Iterators: product
        
        using DocStringExtensions

        export quadrant_rule_graph

        """
        $TYPEDFIELDS

        Struct to hold a KD tree node
        """
        struct KDTree{T <: Real}
                dimension::Int64
                plane::T
                left::Union{KDTree{T}, Nothing}
                right::Union{KDTree{T}, Nothing}
                points::Union{Matrix{T}, Nothing}
                indices::Union{Vector{Int64}, Nothing}
        end

        """
        Find if a KD tree node is a leaf
        """
        isleaf(node::KDTree{T}) where {T <: Real} = isnothing(node.left)

        """
        $TYPEDSIGNATURES

        Build a KD tree from a set of points
        """
        function KDTree(
                points::Matrix{T}, indices::Union{
                        Vector{Int64}, Nothing
                } = nothing;
                dimension::Int = 1,
                leaf_size::Int = 10,
        ) where {T <: Real}
                if isnothing(indices)
                        indices = collect(1:size(points, 2))
                end

                if dimension > size(points, 1)
                        dimension = 1
                end

                if size(points, 2) <= leaf_size
                        return KDTree{T}(
                                dimension, 0.0,
                                nothing, nothing,
                                points, indices
                        )
                end

                x = @view points[dimension, :]
                med = median(x)

                left_inds = findall(
                        (@. x <= med)
                )
                right_inds = findall(
                        (@. x > med)
                )

                if length(left_inds) == 0 || length(right_inds) == 0
                        return KDTree{T}(
                                dimension, 0.0,
                                nothing, nothing,
                                points, indices
                        )
                end

                return KDTree{T}(
                        dimension, med,
                        KDTree(
                                points[:, left_inds], indices[left_inds];
                                dimension = dimension + 1, leaf_size = leaf_size
                        ),
                        KDTree(
                                points[:, right_inds], indices[right_inds];
                                dimension = dimension + 1, leaf_size = leaf_size
                        ), nothing, nothing
                )
        end

        """
        $TYPEDSIGNATURES

        Obtain point closest to a query point within quadrant given by BitVector/vector of booleans.
        """
        function (tree::KDTree{T})(
                x::AbstractVector{T}, quadrant::Union{
                        BitVector, Vector{Bool}
                },
                index::Int64 = 0, distance::Real = Inf32;
                min_distance::Real = 0.0
        ) where {T <: Real}
                if isleaf(tree)
                        for (i, xp) in zip(
                                tree.indices, eachcol(tree.points)
                        )
                                if all(
                                        (xp .> x) .== quadrant
                                )
                                        d = norm(xp .- x)

                                        if d >= min_distance && d < distance
                                                distance = d
                                                index = i
                                        end
                                end
                        end

                        return (index, distance)
                end

                dimension = tree.dimension
                plane = tree.plane

                dleft = max(x[dimension] - plane, 0.0)
                dright = max(plane - x[dimension], 0.0)

                if quadrant[dimension]
                        if dleft == 0.0
                                if dleft < distance
                                        index, distance = tree.left(x, quadrant, index, distance;
                                                min_distance = min_distance)
                                end
                        end

                        if dright < distance
                                index, distance = tree.right(x, quadrant, index, distance;
                                        min_distance = min_distance)
                        end
                else
                        if dright == 0.0
                                if dright < distance
                                        index, distance = tree.right(x, quadrant, index, distance;
                                                min_distance = min_distance)
                                end
                        end

                        if dleft < distance
                                index, distance = tree.left(x, quadrant, index, distance;
                                        min_distance = min_distance)
                        end
                end

                return (index, distance)
        end

        """
        $TYPEDSIGNATURES

        Obtain graph (list of neighbor lists) of points according to octant rule
        """
        function quadrant_rule_graph(
                points::AbstractMatrix{T},
                query_points::Union{AbstractMatrix{T}, Nothing} = nothing; 
                leaf_size::Int = 5,
                n_per_quadrant::Int = 1,
                ϵ::Real = 0.0f0,
        ) where {T <: Real}
                quadrants = product(
                        fill((false, true), size(points, 1))...
                ) |> collect |> x -> map(
                        collect, x
                )
                if ϵ == 0
                        ϵ = eps(T)
                end

                Ti = (
                    size(points, 2) > 1e9 ?
                    Int64 : Int32
                )

                if isnothing(query_points)
                        query_points = points
                end

                graph = [
                        Ti[] for _ = 1:size(query_points, 2)
                ]

                tree = KDTree(points; leaf_size = leaf_size)

                for quad in quadrants
                        for (neighs, x) in zip(
                                graph, eachcol(query_points)
                        )
                                mindist = 0.0

                                for _ = 1:n_per_quadrant
                                        idx, mindist = tree(x, quad; min_distance = mindist + ϵ)

                                        if idx != 0
                                                push!(neighs, idx)
                                        end
                                end
                        end
                end

                graph
        end

end