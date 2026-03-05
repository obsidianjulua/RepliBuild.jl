"""
STL Container Wrapper Types for Julia FFI.

Provides Julia-friendly types that wrap opaque handles to C++ STL containers.
All operations dispatch through JIT-compiled MLIR thunks that call real C++
methods via their mangled names.
"""
module STLWrappers

import ..JITManager

export CppVector, CppString

# =============================================================================
# CppVector{T} - Wrapper for std::vector<T>
# =============================================================================

"""
    CppVector{T} <: AbstractVector{T}

Julia wrapper for `std::vector<T>`. Holds an opaque pointer to the C++ object.
Lifetime is managed by a GC finalizer that calls the vector's destructor.

All operations dispatch through JIT-compiled MLIR thunks.
"""
mutable struct CppVector{T} <: AbstractVector{T}
    handle::Ptr{Cvoid}
    owns::Bool
    thunks::Dict{String,String}  # method_name -> mangled_name
    byte_size::Int               # sizeof(std::vector<T>) from DWARF

    function CppVector{T}(handle::Ptr{Cvoid}, owns::Bool,
                          thunks::Dict{String,String};
                          byte_size::Int=24) where T
        obj = new{T}(handle, owns, thunks, byte_size)
        if owns && haskey(thunks, "destructor")
            finalizer(_cpp_vector_destroy, obj)
        end
        return obj
    end
end

function _cpp_vector_destroy(v::CppVector)
    if v.handle != C_NULL && v.owns
        dtor = get(v.thunks, "destructor", "")
        if !isempty(dtor)
            JITManager.invoke("_mlir_ciface_$(dtor)_thunk", v.handle)
        end
        Libc.free(v.handle)
        v.handle = C_NULL
    end
end

# AbstractVector interface
function Base.size(v::CppVector)
    v.handle == C_NULL && error("CppVector: null handle")
    thunk = get(v.thunks, "size", "")
    isempty(thunk) && error("CppVector: no size thunk available")
    n = JITManager.invoke("_mlir_ciface_$(thunk)_thunk", UInt64, v.handle)
    return (Int(n),)
end

Base.length(v::CppVector) = size(v)[1]

function Base.getindex(v::CppVector{T}, i::Integer) where T
    @boundscheck 1 <= i <= length(v) || throw(BoundsError(v, i))
    thunk = get(v.thunks, "subscript", "")
    if !isempty(thunk)
        # operator[] returns a reference (pointer to element)
        elem_ptr = JITManager.invoke(
            "_mlir_ciface_$(thunk)_thunk",
            Ptr{Cvoid}, v.handle, UInt64(i - 1))
        return unsafe_load(Ptr{T}(elem_ptr))
    end
    # Fallback: use data() + offset
    thunk_data = get(v.thunks, "data", "")
    isempty(thunk_data) && error("CppVector: no subscript or data thunk available")
    data_ptr = JITManager.invoke("_mlir_ciface_$(thunk_data)_thunk", Ptr{Cvoid}, v.handle)
    return unsafe_load(Ptr{T}(data_ptr), i)
end

function Base.setindex!(v::CppVector{T}, val, i::Integer) where T
    @boundscheck 1 <= i <= length(v) || throw(BoundsError(v, i))
    thunk = get(v.thunks, "subscript", "")
    if !isempty(thunk)
        elem_ptr = JITManager.invoke(
            "_mlir_ciface_$(thunk)_thunk",
            Ptr{Cvoid}, v.handle, UInt64(i - 1))
        unsafe_store!(Ptr{T}(elem_ptr), convert(T, val))
        return val
    end
    thunk_data = get(v.thunks, "data", "")
    isempty(thunk_data) && error("CppVector: no subscript or data thunk available")
    data_ptr = JITManager.invoke("_mlir_ciface_$(thunk_data)_thunk", Ptr{Cvoid}, v.handle)
    unsafe_store!(Ptr{T}(data_ptr + (i - 1) * sizeof(T)), convert(T, val))
    return val
end

function Base.push!(v::CppVector{T}, val) where T
    v.handle == C_NULL && error("CppVector: null handle")
    thunk = get(v.thunks, "push_back", "")
    isempty(thunk) && error("CppVector: no push_back thunk available")
    converted = convert(T, val)
    val_ref = Ref(converted)
    GC.@preserve val_ref begin
        JITManager.invoke(
            "_mlir_ciface_$(thunk)_thunk",
            v.handle, Base.unsafe_convert(Ptr{Cvoid}, val_ref))
    end
    return v
end

function Base.empty!(v::CppVector)
    v.handle == C_NULL && error("CppVector: null handle")
    thunk = get(v.thunks, "clear", "")
    isempty(thunk) && error("CppVector: no clear thunk available")
    JITManager.invoke("_mlir_ciface_$(thunk)_thunk", v.handle)
    return v
end

function Base.pointer(v::CppVector{T}) where T
    v.handle == C_NULL && error("CppVector: null handle")
    thunk = get(v.thunks, "data", "")
    isempty(thunk) && error("CppVector: no data thunk available")
    raw = JITManager.invoke("_mlir_ciface_$(thunk)_thunk", Ptr{Cvoid}, v.handle)
    return Ptr{T}(raw)
end

"""
    unsafe_wrap(Array, v::CppVector{T}) -> Vector{T}

Create a zero-copy Julia array view of the CppVector's data.
The returned array is only valid while the CppVector is alive.
"""
function Base.unsafe_wrap(::Type{Array}, v::CppVector{T}) where T
    p = pointer(v)
    n = length(v)
    return unsafe_wrap(Array, p, n)
end

function Base.isempty(v::CppVector)
    thunk = get(v.thunks, "empty", "")
    if !isempty(thunk)
        result = JITManager.invoke("_mlir_ciface_$(thunk)_thunk", UInt8, v.handle)
        return result != 0
    end
    return length(v) == 0
end

Base.IndexStyle(::Type{<:CppVector}) = IndexLinear()

function Base.show(io::IO, v::CppVector{T}) where T
    if v.handle == C_NULL
        print(io, "CppVector{$T}(null)")
    else
        n = length(v)
        print(io, "CppVector{$T}(")
        if n <= 10
            for i in 1:n
                i > 1 && print(io, ", ")
                print(io, v[i])
            end
        else
            for i in 1:5
                i > 1 && print(io, ", ")
                print(io, v[i])
            end
            print(io, ", ..., ")
            for i in n-1:n
                i > n-1 && print(io, ", ")
                print(io, v[i])
            end
        end
        print(io, ")")
    end
end

# =============================================================================
# CppString - Wrapper for std::string
# =============================================================================

"""
    CppString

Julia wrapper for `std::string`. Holds an opaque pointer to the C++ object.
Supports conversion to/from Julia `String`.
"""
mutable struct CppString
    handle::Ptr{Cvoid}
    owns::Bool
    thunks::Dict{String,String}
    byte_size::Int               # sizeof(std::string) from DWARF

    function CppString(handle::Ptr{Cvoid}, owns::Bool,
                       thunks::Dict{String,String};
                       byte_size::Int=32)
        obj = new(handle, owns, thunks, byte_size)
        if owns && haskey(thunks, "destructor")
            finalizer(_cpp_string_destroy, obj)
        end
        return obj
    end
end

function _cpp_string_destroy(s::CppString)
    if s.handle != C_NULL && s.owns
        dtor = get(s.thunks, "destructor", "")
        if !isempty(dtor)
            JITManager.invoke("_mlir_ciface_$(dtor)_thunk", s.handle)
        end
        Libc.free(s.handle)
        s.handle = C_NULL
    end
end

function Base.length(s::CppString)
    s.handle == C_NULL && error("CppString: null handle")
    thunk = get(s.thunks, "size", get(s.thunks, "length", ""))
    isempty(thunk) && error("CppString: no size/length thunk available")
    n = JITManager.invoke("_mlir_ciface_$(thunk)_thunk", UInt64, s.handle)
    return Int(n)
end

function Base.String(s::CppString)
    s.handle == C_NULL && error("CppString: null handle")
    thunk = get(s.thunks, "c_str", get(s.thunks, "data", ""))
    isempty(thunk) && error("CppString: no c_str/data thunk available")
    cstr = JITManager.invoke("_mlir_ciface_$(thunk)_thunk", Ptr{UInt8}, s.handle)
    return unsafe_string(cstr)
end

Base.ncodeunits(s::CppString) = length(s)

function Base.show(io::IO, s::CppString)
    if s.handle == C_NULL
        print(io, "CppString(null)")
    else
        print(io, "CppString(\"", String(s), "\")")
    end
end

end # module STLWrappers
