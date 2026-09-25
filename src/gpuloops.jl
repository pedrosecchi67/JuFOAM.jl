module GPULoops

    export @gpuloop

    using KernelAbstractions

    @inline _first_array(a...) = let n = findfirst(
        aa -> isa(aa, AbstractArray), a
    )
        a[n]
    end

    """
    ```
        macro gpuloop(indices, funcdef)
    ```

    Generate a GPU kernel for a function that loops over arrays.

    Example:

    ```
    using CUDA
    using GPULoops

    @gpuloop (i, j) function foo!(e, A, B)
        A[i, j] = B[i, j] ^ e
    end

    A = rand(100, 100) |> cu
    B = rand(100, 100) |> cu

    foo!(2, A, B)

    @assert A ≈ B .^ 2
    ```
    """
    macro gpuloop(indices, funcdef)
        # Extract the function name, positional arguments, and body
        func_head = funcdef.args[1]
        func_body = funcdef.args[2]

        func_name = func_head.args[1]
        func_args = func_head.args[2:end]

        # let's esc all arguments to make them acessible from an also-esc-ed 
        # function
        func_args = esc.(func_args)

        # Define the kernel function
        kernel_func = quote
            @kernel function $(Symbol(func_name, "_kernel!"))($(func_args...))
                $(esc(indices)) = @index(Global, NTuple)
                $(esc(func_body))
            end
        end

        # Define the entry function
        main_func = quote
            function $(esc(func_name))($(func_args...))
                reference = _first_array($(func_args...))
                backend = KernelAbstractions.get_backend(reference)
                groupsize = KernelAbstractions.isgpu(backend) ? 256 : 1024
                kernel! = $(Symbol(func_name, "_kernel!"))(backend, groupsize)
                kernel!($(func_args...); ndrange = size(reference))
                synchronize(backend)
            end
        end

        # Return both the kernel and the main function
        :($kernel_func; $main_func)
    end

end


