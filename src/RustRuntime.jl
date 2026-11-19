# RustRuntime.jl — what a generated Rust wrapper calls at run time.
#
# A wrapper for a Rust library (`Wrapper/Rust/GeneratorRust.jl`) makes each
# call one of two ways. Where the function's signature is scalars and
# pointers and the compiler proved it cannot panic, the wrapper holds a plain
# `ccall`. Everywhere else it comes here: the call goes through the function's
# thunk (`RustThunks`), and when the function may panic, through the
# library's own catching frame, so a Rust panic arrives in Julia as a
# `RustPanic` and not as the end of the process.
#
# One `Library` per wrapped library. Its thunks are JIT'd the first time one
# is needed, never when the wrapper is loaded: a wrapper whose every call is
# a plain `ccall` needs no JIT at all, and a precompiling process must not
# start one.

module RustRuntime

using Libdl

import ..RustThunks
import .._read_json

export RustPanic, Library, invoke, thunk, variant

"""
    variant(x) -> Symbol

Which variant a Rust enum value is (`:Circle`, `:Some`), read from its
discriminant as the build's metadata places it. A generated wrapper adds one
method per enum that carries data. `:unknown` where the discriminant is not
one this reads.
"""
function variant end

"""
    RustPanic(message)

A Rust function panicked. `message` is the panic's own text: where it
happened, then what it said (`panicked at src/lib.rs:95:57:` and
`division by zero`). The panic was ended inside the library, by Rust; the
library is as usable afterwards as Rust leaves it.
"""
struct RustPanic <: Exception
    message::String
end
Base.showerror(io::IO, e::RustPanic) = print(io, "RustPanic: ", e.message)

"""
A wrapped Rust library at run time: its file, its metadata, the two functions
of the catching frame (`C_NULL` in a library built before there was one), and
its thunk session once a thunk has been asked for.
"""
mutable struct Library
    path::String
    metadata::String
    handle::Ptr{Cvoid}
    catch_fp::Ptr{Cvoid}
    message_fp::Ptr{Cvoid}
    session::Union{Nothing,RustThunks.Session}
    lock::ReentrantLock
end

"""
    Library(path; metadata = <beside the library>)

Open the library (its own handle, `RTLD_LOCAL`). Nothing is JIT'd here.
"""
function Library(path::AbstractString; metadata::AbstractString = joinpath(dirname(path), "compilation_metadata.json"))
    handle = Libdl.dlopen(String(path), Libdl.RTLD_NOW | Libdl.RTLD_LOCAL)
    return Library(String(path), String(metadata), handle,
                   something(Libdl.dlsym_e(handle, :__rb_catch), C_NULL),
                   something(Libdl.dlsym_e(handle, :__rb_panic_message), C_NULL),
                   nothing, ReentrantLock())
end

# The thunks, JIT'd on first use, once.
function _session(lib::Library)::RustThunks.Session
    s = lib.session
    s === nothing || return s
    return lock(lib.lock) do
        lib.session === nothing || return lib.session
        ccall(:jl_generating_output, Cint, ()) == 0 ||
            error("RustRuntime: a Rust thunk was asked for while Julia is precompiling; " *
                  "a call into the library belongs in `__init__` or later")
        isfile(lib.metadata) ||
            error("RustRuntime: $(lib.metadata) is missing; a Rust library's thunks are generated " *
                  "from the metadata its build wrote beside it")
        lib.session = RustThunks.load(lib.path, _read_json(lib.metadata))
        lib.session
    end
end

"""
    thunk(lib, symbol) -> Ptr{Cvoid}

The thunk for the function `symbol`. An error that says why when there is
none: the function was refused, or its signature is one a thunk does not carry.
"""
function thunk(lib::Library, symbol::AbstractString)::Ptr{Cvoid}
    s = _session(lib)
    fp = get(s.thunks, symbol, C_NULL)
    fp == C_NULL && error("RustRuntime: no thunk for `$symbol` in $(basename(lib.path))" *
                          (haskey(s.skipped, symbol) ? ": " * s.skipped[symbol] : ""))
    return fp
end

"`thunk`, kept in `cell` after the first lookup."
@inline function thunk(cell::Base.RefValue{Ptr{Cvoid}}, lib::Library, symbol::AbstractString)::Ptr{Cvoid}
    fp = cell[]
    fp == C_NULL || return fp
    return cell[] = thunk(lib, symbol)
end

"The text of this thread's last panic in `lib`."
function panic_message(lib::Library)::String
    lib.message_fp == C_NULL && return "a Rust function panicked"
    n = ccall(lib.message_fp, Csize_t, (Ptr{UInt8}, Csize_t), C_NULL, 0)
    n == 0 && return "a Rust function panicked (its payload was not a message)"
    buf = Vector{UInt8}(undef, n)
    ccall(lib.message_fp, Csize_t, (Ptr{UInt8}, Csize_t), buf, n)
    return String(buf)
end

"""
    invoke(lib, fp, guarded, slots, ret)

Call the thunk `fp` with its argument slots and its result pointer. `guarded`
says the function may panic: the call is then made inside the library's
catching frame, and a panic is thrown as a `RustPanic`.
"""
@inline function invoke(lib::Library, fp::Ptr{Cvoid}, guarded::Bool, slots::Vector{Ptr{Cvoid}}, ret::Ptr{Cvoid})
    if guarded
        lib.catch_fp == C_NULL &&
            error("RustRuntime: $(basename(lib.path)) has no catching frame, and this function may panic; " *
                  "rebuild the library with this RepliBuild")
        ok = GC.@preserve slots ccall(lib.catch_fp, UInt8, (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Ptr{Cvoid}), fp, slots, ret)
        ok == 0x01 || throw(RustPanic(panic_message(lib)))
    else
        GC.@preserve slots ccall(fp, Cvoid, (Ptr{Ptr{Cvoid}}, Ptr{Cvoid}), slots, ret)
    end
    return nothing
end

end # module RustRuntime
