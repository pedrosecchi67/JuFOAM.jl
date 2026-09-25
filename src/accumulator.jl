module ArrayAccumulator

    using DocStringExtensions

    include("gpuloops.jl")
    using .GPULoops

    include("arraybends.jl")
    using .ArrayBackends

    export Accumulator, to_backend

    """
    $TYPEDFIELDS

    Struct to define an array accumulator object
    """
    struct Accumulator
        indices::AbstractVector
        pointers::AbstractVector
        weights::Union{AbstractVector, Nothing}
        first_index::Bool
    end

    @declare_converter Accumulator

    """
    $TYPEDSIGNATURES

    Constructor for an accumulator object
    from a list of lists of stencil indices and,
    optionally, a list of lists of weights.
    """
    function Accumulator(
        stencils::AbstractVector,
        weights::Union{AbstractVector, Nothing} = nothing;
        first_index::Bool = false,
    )
        if length(stencils) == 0
            return Accumulator(
                eltype(eltype(stencils))[],
                [0],
                eltype(eltype(stencils))[],
                first_index
            )
        end
        
        indices = reduce(vcat, stencils)

        pointers = similar(indices, (length(stencils) + 1,))
        n0 = 0
        for i = 1:(length(stencils) + 1)
            pointers[i] = n0 + 1

            if i <= length(stencils)
                n0 += length(stencils[i])
            end
        end

        weights = (
            isnothing(weights) ? nothing : reduce(vcat, weights)
        )

        Accumulator(indices, pointers, weights, first_index)
    end

    @gpuloop (i,) function add_csr!(dst, src, idx, ptrs, op, f,)
        s = 0.0f0
        @inbounds for k = ptrs[i]:(ptrs[i + 1] - 1)
            s = op(f(src[idx[k]]), s)
        end
        dst[i] = s
    end

    @gpuloop (i,) function add_csr_weighed!(dst, src, idx, ws, ptrs, op, f,)
        s = 0.0f0
        @inbounds for k = ptrs[i]:(ptrs[i + 1] - 1)
            s = op(f(src[idx[k]] * ws[k]), s)
        end
        dst[i] = s
    end

    """
    $TYPEDSIGNATURES

    Run accumulator on vector.
    Runs reduction operator `op` and deploys function `f` on 
    either `u[i]` or `u[i] * w` before reducing.
    """
    (acc::Accumulator)(
        u::AbstractVector;
        f = identity, op = +,
    ) = let dst = similar(u, (length(acc.pointers) - 1,))
        if isnothing(acc.weights)
            add_csr!(dst, u, acc.indices, acc.pointers, op, f,)
        else
            add_csr_weighed!(dst, u, acc.indices, acc.weights, acc.pointers, op, f)
        end

        dst
    end

    """
    $TYPEDSIGNATURES

    Run accumulator on multi-dimensional array.
    If `first_index` is true upon array construction, works
    along first dimension. Otherwise, works along last dimension (default).
    """
    (acc::Accumulator)(u::AbstractArray) = (
        acc.first_index ?
        mapslices(acc, u; dims = 1) :
        mapslices(acc, u; dims = ndims(u))
    )

    """
    $TYPEDSIGNATURES

    Obtain domain of accumulator, as array of indices,
    and hashmap mapping old indices to the current domain.
    """
    function domain(acc::Accumulator)
        dom = unique(acc.indices)
        Ti = eltype(dom)
        hmap = Dict([i => Ti(k) for (k, i) in enumerate(dom)]...)

        (dom, hmap)
    end

    """
    $TYPEDSIGNATURES

    Re-index accumulator to a local domain given hashmap from 
    `ArrayAccumulator.domain()`.
    """
    function re_index!(acc::Accumulator, hmap::AbstractDict)
        for (k, i) in enumerate(acc.indices)
            acc.indices[k] = hmap[i]
        end
        acc
    end

    """
    $TYPEDSIGNATURES

    Obtain list of lists format for accumulator.
    Returns `nothing` for weights LoL if the accumulator
    doesn't have weights.
    """
    function list_of_lists(
        acc::Accumulator
    )
        tolol = v -> [
            v[acc.pointers[i]:(acc.pointers[i + 1] - 1)] for i = 1:(length(acc.pointers) - 1)
        ]

        (
            tolol(acc.indices),
            (isnothing(acc.weights) ? nothing : tolol(acc.weights))
        )
    end

end