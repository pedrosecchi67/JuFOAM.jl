module Stash

    export stash!, unstash, clean_stash!

    using LinearAlgebra
    using Distributed

    using UUIDs

    using DocStringExtensions

    const _STASH = Dict{UUID, Any}()

    """
    $TYPEDSIGNATURES

    Stash value at a given PID and return the storage UUID
    """
    function stash!(
        pid::Int, v::Any
    )
        if pid == myid()
            key = uuid4()
            while haskey(_STASH, key)
                key = uuid4()
            end

            _STASH[key] = v
            return key
        end

        s = @spawnat pid begin
            key = uuid4()
            while haskey(_STASH, key)
                key = uuid4()
            end

            _STASH[key] = v

            key
        end

        fetch(s)
    end
    
    """
    $TYPEDSIGNATURES

    Update stash value at a given PID and return the storage UUID
    """
    function stash!(
        pid::Int, key::UUID, v::Any
    )
        if pid == myid()
            _STASH[key] = v;
            return
        end

        s = @spawnat pid begin
            _STASH[key] = v
						;
        end

        fetch(s)
    end

    """
    $TYPEDSIGNATURES

    Fetch value from stash on a given process
    """
    function unstash(pid::Int, key::UUID)
        if pid == myid()
            return _STASH[key]
        end

        s = @spawnat pid begin
            _STASH[key]
        end

        fetch(s)
    end

    """
    $TYPEDSIGNATURES

    Remove value from stash on a given process
    """
    function clean_stash!(pid::Int, key::UUID)
        if pid == myid()
            if haskey(_STASH, key)
                delete!(_STASH, key)
            end
            return
        end

        s = @spawnat pid begin
            if haskey(_STASH, key)
                delete!(_STASH, key)
            end
        end

        fetch(s)
    end

end