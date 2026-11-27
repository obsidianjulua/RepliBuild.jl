# CxxObjects.jl — a C++ object Julia holds where C++ built it.
#
# A C++ function that returns a `std::vector`, a `std::map`, a
# `std::unique_ptr` or a class with a destructor builds the object in storage
# its caller supplies. C++ never moves such an object by copying its bytes: it
# may point into itself (a `std::list`'s first node points at the list), and
# something has to run its destructor exactly once. A generated wrapper used to
# copy the bytes into a Julia value and drop the storage: the copy was invalid
# wherever the object pointed into itself, and nothing ever ended it.
#
# `Owned{T}` is that storage, kept: a Julia object whose first field is the
# C++ object, so the object's address is the Julia object's and never changes
# (Julia's collector does not move objects). When Julia collects it, a
# finalizer runs the destructor the compiler compiled for the type
# (`replibuild_own_end_<n>`, `Builder/OwnershipUnit.jl`). The same rule as the
# Rust front end's owned values: one holder, ended once.
#
# Passing one on:
#   * where C++ takes a pointer or a reference, the object's address passes;
#   * where C++ takes it BY VALUE, C++ gets a copy made by the type's own copy
#     constructor and the caller keeps its object (John, 2026-11-27: "by
#     value"). A type that cannot be copied (`std::unique_ptr`) is moved: the
#     call takes it, and the Julia object is spent. `move(x)` asks for that
#     on purpose, for a caller who is finished with a large object.
#
# Generated wrappers name this module. It loads after `JITManager`, whose
# pointer conversion (`pointer_arg`) it extends for an owned object.

module CxxObjects

import Libdl
import ..JITManager

export Owned, Kind, Moved, move, release!, isalive

# An object's states. UNBORN is storage C++ has not built into yet.
const UNBORN = 0x00
const LIVE   = 0x01
const ENDED  = 0x02   # its destructor ran: released, or collected
const SPENT  = 0x03   # moved into a C++ call, which took it; ended there

# A value no two processes share: a `Kind`'s function addresses are this
# process's, and a wrapper precompiled into a package carries its `Kind`s
# from the process that compiled it.
const _PROCESS = Ref{UInt}(0)
const _LOCK = ReentrantLock()

function __init__()
    # Not `rand`: loading a package must not move the caller's random stream.
    _PROCESS[] = (UInt(Libc.getpid()) << 32) ⊻ UInt(time_ns()) | UInt(1)
    return nothing
end

"""
    Kind(name, size, align, ends, copies, moves, end_symbol, copy_symbol, move_symbol)

What the compiler said about one C++ type a library returns or takes by value
(the metadata's `ownership`), and the three functions the build compiled for
it. A wrapper holds one per type. `name` is the C++ type, for messages;
`ends` whether its destructor has anything to do; `copies` and `moves` whether
C++ can copy- and move-construct one.

The functions' addresses are found in the wrapped library at first use
(`resolve!`), never when the wrapper is written or precompiled.
"""
mutable struct Kind
    const name::String
    const size::Int
    const align::Int
    const ends::Bool
    const copies::Bool
    const moves::Bool
    const end_symbol::String
    const copy_symbol::String
    const move_symbol::String
    ender::Ptr{Cvoid}
    copier::Ptr{Cvoid}
    mover::Ptr{Cvoid}
    process::UInt
    Kind(name, size, align, ends, copies, moves, end_symbol, copy_symbol, move_symbol) =
        new(String(name), Int(size), Int(align), Bool(ends), Bool(copies), Bool(moves),
            String(end_symbol), String(copy_symbol), String(move_symbol), C_NULL, C_NULL, C_NULL, UInt(0))
end

"""
    resolve!(kind, library) -> kind

Find `kind`'s three functions in `library`, once per process. A library built
before the wrapper asked for them (or a wrapper pointed at another library)
is an error that says to build again: a wrapper never ends an object with a
function it could not find.
"""
function resolve!(kind::Kind, library::AbstractString)
    kind.process == _PROCESS[] && return kind
    lock(_LOCK) do
        kind.process == _PROCESS[] && return
        handle = Libdl.dlopen(library)
        find(sym) = begin
            p = Libdl.dlsym(handle, Symbol(sym); throw_error = false)
            p === nothing && error("RepliBuild: `$library` has no `$sym`, the function its wrapper ends, copies or " *
                                   "moves a C++ `$(kind.name)` with. The wrapper and the library are not from the " *
                                   "same build: run RepliBuild.build and RepliBuild.wrap again.")
            Ptr{Cvoid}(p)
        end
        kind.ender = find(kind.end_symbol)
        kind.copier = find(kind.copy_symbol)
        kind.mover = find(kind.move_symbol)
        kind.process = _PROCESS[]
    end
    return kind
end

"""
    Owned{T}

A C++ object Julia holds where C++ built it. `T` is the Julia type its bytes
have (`NTuple{24, UInt8}` for a `std::vector`, the wrapper's struct for a
class with fields). The C++ object is the Julia object's first field, so its
address is fixed for as long as Julia holds it.

  * `x[]` is a copy of the bytes as a `T`, and `x.field` reads a field of a
    `T` that has fields. Both read the object where it is.
  * It passes wherever C++ takes a pointer or a reference to the type, and
    where C++ takes the type by value (a copy, or a move for a type that
    cannot be copied; see [`move`](@ref)).
  * Its destructor runs when Julia collects it, or at once with
    [`release!`](@ref). Julia's collector does not see the memory a C++ object
    owns: release a large one when you are finished with it.
"""
mutable struct Owned{T}
    bytes::T
    kind::Kind
    state::UInt8
    function Owned{T}(kind::Kind) where {T}
        o = new{T}()
        setfield!(o, :kind, kind)
        setfield!(o, :state, UNBORN)
        return o
    end
end

_kind(o::Owned) = getfield(o, :kind)
_state(o::Owned) = getfield(o, :state)

"""
    slot(T, kind, library) -> Owned{T}

Storage for an object of `kind` that C++ is about to build: uninitialized, and
not yet an object. The wrapper hands its address to the call
(`JITManager.slot_address`) and calls [`born!`](@ref) once the call has
returned; a call that throws leaves storage nothing has to end.
"""
function slot(::Type{T}, kind::Kind, library::AbstractString) where {T}
    isbitstype(T) || error("RepliBuild: cannot hold a C++ `$(kind.name)` as `$T`: it has no fixed C layout.")
    sizeof(T) == kind.size ||
        error("RepliBuild: the wrapper holds a C++ `$(kind.name)` in $(sizeof(T)) bytes and the library says it is " *
              "$(kind.size). The wrapper and the library are not from the same build: run RepliBuild.build and " *
              "RepliBuild.wrap again.")
    resolve!(kind, library)
    return Owned{T}(kind)
end

"""
    born!(x) -> x

C++ has built the object in `x`'s storage: from here on `x` is that object,
and its destructor runs when Julia collects it.
"""
function born!(o::Owned)
    setfield!(o, :state, LIVE)
    _kind(o).ends && finalizer(_end, o)
    return o
end

# The finalizer, and `release!`. It must not throw and must not switch tasks:
# a state check and one C call.
function _end(o::Owned)
    _state(o) == LIVE || return nothing
    setfield!(o, :state, ENDED)
    p = _kind(o).ender
    (_kind(o).ends && p != C_NULL) && ccall(p, Cvoid, (Ptr{Cvoid},), pointer_from_objref(o))
    return nothing
end

"""
    release!(x) -> nothing

Run `x`'s destructor now instead of when Julia collects it. Using `x`
afterwards is an error; releasing it twice is not.
"""
release!(o::Owned) = _end(o)

"Whether `x` is still an object: not released, and not moved into a call."
isalive(o::Owned) = _state(o) == LIVE

@noinline function _refuse_dead(o::Owned)
    how = _state(o) == SPENT ? "was moved into a C++ call, which took it" :
          _state(o) == ENDED ? "was released" : "was never built"
    throw(ArgumentError("this C++ `$(_kind(o).name)` $how, and is no longer an object. " *
                        "Pass `copy(x)`, or keep the original by not moving it."))
end
@inline _alive(o::Owned) = (_state(o) == LIVE || _refuse_dead(o); o)

"The address of the C++ object `x` holds. Valid while `x` is reachable; keep it so with `GC.@preserve`."
address(o::Owned) = (_alive(o); pointer_from_objref(o))

Base.getindex(o::Owned) = getfield(_alive(o), :bytes)
function Base.getproperty(o::Owned{T}, name::Symbol) where {T}
    hasfield(T, name) || error("a C++ `$(_kind(o).name)` held as `$T` has no field `$name`")
    return getfield(getfield(_alive(o), :bytes), name)
end
Base.propertynames(::Owned{T}) where {T} = fieldnames(T)

# Where C++ takes a pointer or reference, the object's own address passes: at
# a `ccall` position typed `Ptr{Cvoid}`, `Ptr{T}` or `Ref{T}`, and at a thunk's
# pointer position (`JITManager.pointer_arg`, whose last method asks
# `unsafe_convert(Ptr{Cvoid}, x)`).
Base.unsafe_convert(::Type{Ptr{Cvoid}}, o::Owned) = address(o)
Base.unsafe_convert(::Type{Ptr{T}}, o::Owned{T}) where {T} = Ptr{T}(address(o))
Base.unsafe_convert(::Type{Ptr{Cvoid}}, o::Owned{Cvoid}) = address(o)   # the two above meet here
Base.cconvert(::Type{Ref{T}}, o::Owned{T}) where {T} = o
Base.cconvert(::Type{Ref{T}}, r::Base.RefValue{Owned{T}}) where {T} = r[]
Base.unsafe_convert(::Type{Ref{T}}, o::Owned{T}) where {T} = Ptr{T}(address(o))
# `Ref(x)` was how a returned container was passed by reference while it was
# its bytes. A `Ref` of an owned object still passes the object, never the
# address of the `Ref`.
JITManager._pointer_arg(::Type{P}, r::Base.RefValue{<:Owned}) where {P} =
    JITManager.PointerArg(address(r[]), r)

function Base.show(io::IO, o::Owned{T}) where {T}
    s = _state(o)
    print(io, "Owned C++ ", _kind(o).name, " (", sizeof(T), " bytes, ",
          s == LIVE ? "live" : s == SPENT ? "moved into a call" : s == ENDED ? "released" : "not built", ")")
end

# ── Passing by value ─────────────────────────────────────────────────────────

"""
    Moved{T}

`move(x)`: an owned object given to a by-value parameter to be taken, not
copied.
"""
struct Moved{T}
    object::Owned{T}
end

"""
    move(x) -> Moved

Give `x` to a C++ by-value parameter without copying it: the call takes the
object, as `f(std::move(x))` does, and `x` is spent afterwards. For a caller
who is finished with a large object. Without it a by-value parameter gets a
copy and `x` stays yours.
"""
move(o::Owned) = Moved(_alive(o))

@noinline _refuse_copy(k::Kind) =
    throw(ArgumentError("a C++ `$(k.name)` cannot be copied (C++ deletes its copy constructor). " *
                        "Pass `RepliBuild.move(x)` to give it to the call."))
@noinline _copy_failed(k::Kind, rc) =
    error(rc == 2 ? "a C++ `$(k.name)` cannot be copied or moved that way" :
                    "copying a C++ `$(k.name)` threw a C++ exception")

"""
    Base.copy(x::Owned) -> Owned

A second object, made by the type's own copy constructor. An error for a type
C++ cannot copy.
"""
function Base.copy(o::Owned{T}) where {T}
    k = _kind(_alive(o))
    k.copies || _refuse_copy(k)
    c = Owned{T}(k)
    rc = GC.@preserve o c ccall(k.copier, Cint, (Ptr{Cvoid}, Ptr{Cvoid}), pointer_from_objref(c), pointer_from_objref(o))
    rc == 0 || _copy_failed(k, rc)
    return born!(c)
end

"""
    Passed{T}

One argument for a parameter C++ takes by value through an address
(`byvalue`): `ptr`, what the call is given; `keep`, what holds the storage
behind it; `temp`, the copy made for the call, which `settle!` ends after it;
and `spend`, the caller's own object when the call takes it, which `settle!`
ends and marks spent.
"""
struct Passed{T}
    ptr::Ptr{Cvoid}
    keep::Any
    temp::Union{Nothing,Owned{T}}
    spend::Union{Nothing,Owned{T}}
end

@noinline _refuse_kind(k::Kind, o::Owned) =
    throw(ArgumentError("a C++ `$(_kind(o).name)` was passed where the function takes a `$(k.name)` by value."))

"""
    byvalue(T, kind, library, x) -> Passed{T}

What a C++ by-value parameter of `kind` gets for the Julia argument `x`:

  * an [`Owned`](@ref) of a type C++ can copy: a temporary made by the copy
    constructor, which `settle!` ends after the call. The caller's object is
    untouched;
  * an `Owned` of a type C++ cannot copy, or `move(x)`: the object itself.
    The call takes it, and `settle!` ends what is left and marks it spent;
  * anything else that names an object's address (a pointer, a `Ref` of the
    bytes a wrapper used to return, the bytes themselves, an STL wrapper's
    object; `JITManager.pointer_arg` decides): a temporary copied from that
    address when the type can be copied, else that address, as before.
"""
function byvalue(::Type{T}, kind::Kind, library::AbstractString, x) where {T}
    resolve!(kind, library)
    return _byvalue(T, kind, x)
end
function _byvalue(::Type{T}, k::Kind, o::Owned{T}) where {T}
    _alive(o)
    (_kind(o) === k || _kind(o).name == k.name) || _refuse_kind(k, o)
    k.copies || return Passed{T}(pointer_from_objref(o), o, nothing, o)
    c = _copy_from(T, k, pointer_from_objref(o), o)
    return Passed{T}(pointer_from_objref(c), c, c, nothing)
end
function _byvalue(::Type{T}, k::Kind, m::Moved{T}) where {T}
    o = _alive(m.object)
    (_kind(o) === k || _kind(o).name == k.name) || _refuse_kind(k, o)
    return Passed{T}(pointer_from_objref(o), o, nothing, o)
end
_byvalue(::Type{T}, k::Kind, x::Union{Owned,Moved}) where {T} =
    _refuse_kind(k, x isa Moved ? x.object : x)
function _byvalue(::Type{T}, k::Kind, x) where {T}
    a = JITManager.pointer_arg(JITManager.ByValue{T}, x)
    (k.copies && a.ptr != C_NULL) || return Passed{T}(a.ptr, a.keep, nothing, nothing)
    c = _copy_from(T, k, a.ptr, a.keep)
    return Passed{T}(pointer_from_objref(c), c, c, nothing)
end

# A temporary copy-constructed from the object at `src`, which `keep` holds.
function _copy_from(::Type{T}, k::Kind, src::Ptr{Cvoid}, keep) where {T}
    c = Owned{T}(k)
    rc = GC.@preserve keep c ccall(k.copier, Cint, (Ptr{Cvoid}, Ptr{Cvoid}), pointer_from_objref(c), src)
    rc == 0 || _copy_failed(k, rc)
    return born!(c)
end

"""
    Lent{A}

One argument for a by-value parameter whose temporary the THUNK makes, with
the class's own copy or move constructor (`FunctionGen.value_temporary`: a
library class with a layout here): `arg` is what the call is given, the
caller's value or its owned object, read where it is; `spend` is that object
when the call takes it.
"""
struct Lent{A,T}
    arg::A
    spend::Union{Nothing,Owned{T}}
end

"""
    lend(T, kind, library, x) -> Lent

What a by-value parameter of a library class gets for `x`: a plain `T` as it
is; an [`Owned`](@ref), which the thunk copies from where it is, and which is
spent after the call when C++ can only move it or the caller said `move(x)`.
"""
function lend(::Type{T}, kind::Kind, library::AbstractString, x) where {T}
    resolve!(kind, library)
    return _lend(T, kind, x)
end
_lend(::Type{T}, k::Kind, x::T) where {T} = Lent{T,T}(x, nothing)
function _lend(::Type{T}, k::Kind, o::Owned{T}) where {T}
    _alive(o)
    (_kind(o) === k || _kind(o).name == k.name) || _refuse_kind(k, o)
    return Lent{Owned{T},T}(o, k.copies ? nothing : o)
end
function _lend(::Type{T}, k::Kind, m::Moved{T}) where {T}
    o = _alive(m.object)
    (_kind(o) === k || _kind(o).name == k.name) || _refuse_kind(k, o)
    return Lent{Owned{T},T}(o, o)
end

"""
    settle!(a) -> nothing

After the call: end the temporary it was given, or what is left of the object
it took, which is spent from here on. C++'s caller destroys a by-value
argument when the call returns; this is that.
"""
function settle!(a::Passed)
    a.temp === nothing || _end(a.temp)
    _spend(a.spend)
    return nothing
end
settle!(a::Lent) = (_spend(a.spend); nothing)

_spend(::Nothing) = nothing
function _spend(o::Owned)
    _end(o)
    setfield!(o, :state, SPENT)
    return nothing
end

end # module CxxObjects
