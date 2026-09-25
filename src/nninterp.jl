module NNInterpolator

    include("accumulator.jl")
    using .ArrayAccumulator
    using .ArrayAccumulator.DocStringExtensions

    using LinearAlgebra

    using NearestNeighbors

    export Interpolator

    function IDW_weights(X::AbstractMatrix, x::AbstractVector)
        w = sum((X .- x) .^ 2; dims = 1) |> vec
        @. w = 1.0f0 / (w + 1f-14)

        sw = sum(w)
        @. w / sw
    end

    function linear_weights(X::AbstractMatrix, x::AbstractVector)
        dX = (X .- x) 
        w = sum(dX .^ 2; dims = 1) |> vec
        @. w = 1.0f0 / (w + 1f-14)

        A = [dX' ones(eltype(dX), size(dX, 2))] .* w
        (pinv(A) .* w')[end, :]
    end

    """
    $TYPEDSIGNATURES

    Obtain interpolator from point cloud
    `X` to points in `Xc`.

    If `first_index = false` (default), `X` is expected to be
    a long matrix. If `true`, `Xc` is expected to be a tall matrix.

    `tree` is an optional, pre-build KDTree.

    `k` nearest neighbors are used.
    Defaults to `ndims + 1`.
    """
    function Interpolator(
        X::AbstractMatrix,
        Xc::AbstractMatrix,
        tree::Union{Nothing, KDTree} = nothing;
        linear::Bool = true,
        first_index::Bool = false,
        bias::Union{AbstractMatrix, Nothing} = nothing,
        k::Int = 0,
    )
        Xq = X
        Xcq = Xc
        biasq = bias
        if first_index
            Xq = permutedims(Xq)
            Xcq = permutedims(Xc)
            if !isnothing(bias)
                biasq = permutedims(bias)
            end
        end

        if isnothing(tree)
            tree = KDTree(Xq)
        end

        get_weights = (i, stencil) -> (
            linear ?
            linear_weights(Xq[:, stencil], Xcq[:, i]) :
            IDW_weights(Xq[:, stencil], Xcq[:, i])
        )

        if k == 0
            k = size(Xq, 1) + 1
        end

        if size(Xcq, 2) == 0
            Accumulator(
                Vector{Int64}[], Vector{eltype(Xcq)}[]; 
                first_index = first_index,
            )
        end

        stencils, _ = knn(
            tree,
            (isnothing(bias) ? Xcq : Xcq .+ biasq), 
            k
        )
        weights = map(
            i -> get_weights(i, stencils[i]), 1:length(stencils)
        )

        Accumulator(stencils, weights;
            first_index = first_index)
    end

end
