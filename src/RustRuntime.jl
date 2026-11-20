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
# What owns memory on the Rust side is ended by Rust: a wrapper's owning
# types call the library's own drop functions (`drop!`, also their
# finalizer), and a `Result`'s `Err` is thrown as a `RustError`.
#
# One `Library` per wrapped library. Its thunks are in a small library of
# their own beside it when `wrap` could compile them (`RustThunks.aot_library`),
# and otherwise are JIT'd the first time one is needed, never when the wrapper
# is loaded: a wrapper whose every call is a plain `ccall` needs no JIT at
# all, a wrapper with its thunks library needs none either, and a
# precompiling process must not start one.

module RustRuntime

using Libdl
import SHA

import ..RustThunks
import .._read_json

export RustPanic, RustError, Library, invoke, thunk, variant, drop!, isowned

"""
    variant(x) -> Symbol

Which variant a Rust enum value is (`:Circle`, `:Some`), read from its
discriminant as the build's metadata places it. A generated wrapper adds one
method per enum that carries data. `:unknown` where the discriminant is not
one this reads.
"""
function variant end

"""
    drop!(x)

End the Rust value `x` holds, now, by Rust's own drop code for its type,
instead of when Julia collects `x`. Nothing happens for a value that was
already moved into Rust or dropped. A generated wrapper adds one method per
type that owns something; each is also that type's finalizer.
"""
function drop! end

"""
    isowned(x) -> Bool

Whether `x` still holds its Rust value: `false` once it was passed to a Rust
function by value (which moves it) or dropped.
"""
function isowned end

"""
    RustError(value)

A Rust function returned `Err(value)`. A wrapped function that returns
`Result<T, E>` gives the `T` of an `Ok` and throws this for an `Err`.
"""
struct RustError <: Exception
    value::Any
end
Base.showerror(io::IO, e::RustError) =
    print(io, "RustError: ", e.value isa AbstractString ? e.value : repr(e.value))

"The `n` elements of type `E` at `data`, copied into a Julia vector: a `Vec` of plain data."
vec_copy(::Type{E}, data::Ptr{Cvoid}, n::Int) where {E} =
    n == 0 ? E[] : copy(unsafe_wrap(Array, Ptr{E}(data), n))

"`f(pointer)` for each of the `n` elements of `stride` bytes at `data`, as a vector of `E`."
vec_read(::Type{E}, f, data::Ptr{Cvoid}, n::Int, stride::Int) where {E} =
    E[f(data + (i - 1) * stride) for i in 1:n]

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
of the catching frame (`C_NULL` in a library built before there was one), its
ahead-of-time thunks library when there is one (`thunks_handle`), and its
thunk session once a thunk had to be JIT'd.
"""
mutable struct Library
    path::String
    metadata::String
    handle::Ptr{Cvoid}
    catch_fp::Ptr{Cvoid}
    message_fp::Ptr{Cvoid}
    session::Union{Nothing,RustThunks.Session}
    lock::ReentrantLock
    thunks_handle::Ptr{Cvoid}
end

"""
    Library(path; metadata = <beside the library>, thunks = "", thunks_sha = "", convention = "rust-thunk-1")

Open the library (its own handle, `RTLD_LOCAL`). Nothing is JIT'd here.

`convention` is the thunk convention the calling wrapper was written for
(what its slots hold). A wrapper written for another one than this
RepliBuild's would hand its thunks the wrong slots, so it is refused here,
before any call, and asked to be generated again.

`thunks` is the library of thunks `wrap` compiled ahead of time, used when it
is there and is the file the wrapper was generated with (`thunks_sha`). When
it is not, the thunks are JIT'd from the metadata at the first call that
needs one, as they are for a wrapper with no thunks library.
"""
function Library(path::AbstractString; metadata::AbstractString = joinpath(dirname(path), "compilation_metadata.json"),
                 thunks::AbstractString = "", thunks_sha::AbstractString = "",
                 convention::AbstractString = "rust-thunk-1")
    convention == RustThunks.CONVENTION ||
        error("RustRuntime: the wrapper of $(basename(path)) was generated for the thunk convention " *
              "`$convention`, and this RepliBuild's is `$(RustThunks.CONVENTION)`: its calls would hand a " *
              "thunk the wrong arguments. Run RepliBuild.wrap on the project again.")
    handle = Libdl.dlopen(String(path), Libdl.RTLD_NOW | Libdl.RTLD_LOCAL)
    return Library(String(path), String(metadata), handle,
                   something(Libdl.dlsym_e(handle, :__rb_catch), C_NULL),
                   something(Libdl.dlsym_e(handle, :__rb_panic_message), C_NULL),
                   nothing, ReentrantLock(), _open_thunks(String(path), String(thunks), String(thunks_sha)))
end

# The ahead-of-time thunks library's handle, or `C_NULL` to JIT: no such
# file, another file than the wrapper was generated with (its thunks may be
# another build's), or one that does not load.
function _open_thunks(library::String, thunks::String, sha::String)::Ptr{Cvoid}
    (isempty(thunks) || !isfile(thunks)) && return C_NULL
    if !isempty(sha) && bytes2hex(open(SHA.sha256, thunks)) != sha
        @warn "RustRuntime: $(basename(thunks)) is not the thunks library the wrapper of $(basename(library)) " *
              "was generated with; it is not used, and the thunks are compiled at first use instead. " *
              "Run RepliBuild.wrap on the project again to have them match."
        return C_NULL
    end
    handle = Libdl.dlopen(thunks, Libdl.RTLD_NOW | Libdl.RTLD_LOCAL; throw_error = false)
    return handle === nothing ? C_NULL : handle
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

The thunk for the function `symbol`: the one in the library's ahead-of-time
thunks library when it has one, else JIT'd. An error that says why when there
is none: the function was refused, or its signature is one a thunk does not
carry.
"""
function thunk(lib::Library, symbol::AbstractString)::Ptr{Cvoid}
    if lib.thunks_handle != C_NULL
        fp = something(Libdl.dlsym_e(lib.thunks_handle, RustThunks.thunk_name(symbol)), C_NULL)
        fp == C_NULL || return fp
    end
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

Call the thunk `fp` with its slots and its result pointer. The slots are one
per parameter the function is handed, then the function's own address, which
the thunk calls (`RustThunks`). `guarded` says the function may panic: the
call is then made inside the library's catching frame, and a panic is thrown
as a `RustPanic`.
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
