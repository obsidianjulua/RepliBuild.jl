"""
    RepliBuild.StdStrings

A `std::basic_string<char>` returned by value, read into a Julia `String`
(2026-11-13).

A C++ function returning `std::string` crosses Tier 2 through a hidden return
pointer, and C++ builds the object there. The object is not trivially
relocatable: libstdc++'s `_M_p` points into the object itself when the string
is short. So it is read, and ended, where the callee built it. That is the
caller's buffer (`FunctionGen.returns_in_caller_buffer`): the wrapper allocates
it (`JITManager.return_slot`), the thunk forwards it as the hidden pointer, and
`take_string!` reads the characters there, calls the destructor on it, and
returns the `String`. The bytes copied into the `String` are all that outlives
the call.

What the slot's bytes mean is the record's own DWARF, recorded by the build
(`record_abi`'s `string_layout`, `Compiler._string_layout`): never a
hardcoded 32 or 24. `string_reader` recognises the two layouts that exist and
refuses anything else, and the generator then keeps the opaque stand-in with
the reason in the wrapper.

The destructor is found per host at first use (`destructor`): through the
wrapped library's own handle, which on ELF also searches the libraries it
depends on (libstdc++ defines `basic_string<char>`'s members when the library
does not), and on PE, where a handle sees only its own DLL, the C++ runtime
by name. A destructor that cannot be found keeps the stand-in and says so once:
a string is never left unended silently.
"""
module StdStrings

import Libdl
import ..host_record_abi

"""
    Reader

How to read a string out of its slot.

- `:pointer_length` (libstdc++): the characters are at the pointer stored at
  `ptr`, and there are as many as the `size_t` at `len` says. The pointer may
  point into the slot (a short string) or to the heap; the slot never moved, so
  either is right.
- `:short_long` (libc++): the bit at `flag_bit` says which arm of the union is
  live. Long: pointer at `ptr`, length at `len`. Short: the characters are in
  the slot at `short_data`, and the length is the `short_len_bits`-bit field at
  `short_len_bit`.

Bit positions are absolute, from the start of the slot, little-endian.
"""
struct Reader
    kind::Symbol
    ptr::Int
    len::Int
    flag_bit::Int
    short_data::Int
    short_len_bit::Int
    short_len_bits::Int
end
Reader(kind::Symbol, ptr::Int, len::Int) = Reader(kind, ptr, len, -1, -1, -1, 0)

_last(path) = (i = findlast('.', path); i === nothing ? path : path[i+1:end])
_ends(path, suffix) = path == suffix || endswith(path, "." * suffix)
_is_pointer(ct) = endswith(strip(String(ct)), "*")

"""
    string_reader(layout) -> Union{Reader,String}

The `Reader` for a recorded `string_layout`, or the reason none can be made (a
`String`, which the generator writes into the wrapper beside the stand-in).

Only `char` strings: a `wchar_t`, `char16_t` or `char32_t` string needs a
transcoding the caller must choose. Recognised by the leaves' own names, at any
nesting (the paths keep it: libc++'s `__rep_` sits inside a compressed pair in
older releases, as a member in newer ones):

- libstdc++: a pointer leaf named `_M_p` and a leaf named `_M_string_length`.
- libc++: the `__l` arm's `__data_` (a pointer) and `__size_`, the `__s` arm's
  `__data_` (an array) and its `__size_` and `__is_long_` bitfields. A libc++
  without the `__is_long_` bit (before LLVM 19 the flag was folded into the
  capacity or the size, differently per layout) is refused, not guessed.
"""
function string_reader(layout::AbstractDict)::Union{Reader,String}
    u = get(layout, "unread", nothing)
    u === nothing || return "its layout could not be read from the DWARF: $(u)"
    ct = get(layout, "char_type", nothing)
    ct == "char" || return "its character type is `$(something(ct, "unknown"))`; only `char` strings are decoded"
    leaves = get(layout, "leaves", Any[])
    isempty(leaves) && return "its layout has no members"
    find(pred) = (i = findfirst(pred, leaves); i === nothing ? nothing : leaves[i])
    path(l) = String(get(l, "path", ""))
    byte(l) = get(l, "offset", nothing)

    p = find(l -> _last(path(l)) == "_M_p" && _is_pointer(get(l, "c_type", "")) && byte(l) !== nothing)
    n = find(l -> _last(path(l)) == "_M_string_length" && byte(l) !== nothing)
    if p !== nothing && n !== nothing
        return Reader(:pointer_length, Int(byte(p)), Int(byte(n)))
    end

    ld = find(l -> _ends(path(l), "__l.__data_") && _is_pointer(get(l, "c_type", "")) && byte(l) !== nothing)
    ln = find(l -> _ends(path(l), "__l.__size_") && byte(l) !== nothing)
    sd = find(l -> _ends(path(l), "__s.__data_") && byte(l) !== nothing)
    sn = find(l -> _ends(path(l), "__s.__size_") && haskey(l, "data_bit_offset"))
    sf = find(l -> _ends(path(l), "__s.__is_long_") && haskey(l, "data_bit_offset") &&
                   Int(get(l, "bit_size", 0)) == 1)
    if ld !== nothing && ln !== nothing && sd !== nothing && sn !== nothing && sf !== nothing
        return Reader(:short_long, Int(byte(ld)), Int(byte(ln)), Int(sf["data_bit_offset"]),
                      Int(byte(sd)), Int(sn["data_bit_offset"]), Int(sn["bit_size"]))
    end
    if ld !== nothing || sd !== nothing
        return "it is a libc++ string without the `__is_long_` bit; that layout is not decoded"
    end
    return "its layout is neither libstdc++'s (`_M_p`, `_M_string_length`) nor libc++'s (`__l`, `__s`)"
end

# `nbits` bits at absolute bit `bit` of the bytes at `p`, little-endian.
function _bits(p::Ptr{UInt8}, bit::Int, nbits::Int)
    v = UInt64(0)
    first, last = bit ÷ 8, (bit + nbits - 1) ÷ 8
    for i in last:-1:first
        v = (v << 8) | unsafe_load(p + i)
    end
    return (v >> (bit - 8first)) & ((UInt64(1) << nbits) - 1)
end

"""
    read_string(p, reader) -> String

Copy the characters of the string object at `p` into a `String`. The caller
keeps the object where it is and ends it after this (`take_string!`).
"""
function read_string(p::Ptr, r::Reader)::String
    q = Ptr{UInt8}(p)
    if r.kind === :pointer_length
        return unsafe_string(unsafe_load(Ptr{Ptr{UInt8}}(q + r.ptr)), unsafe_load(Ptr{Csize_t}(q + r.len)))
    elseif r.kind === :short_long
        if _bits(q, r.flag_bit, 1) == 1
            return unsafe_string(unsafe_load(Ptr{Ptr{UInt8}}(q + r.ptr)), unsafe_load(Ptr{Csize_t}(q + r.len)))
        end
        return unsafe_string(q + r.short_data, Int(_bits(q, r.short_len_bit, r.short_len_bits)))
    end
    throw(ArgumentError("StdStrings.read_string: unknown reader kind $(r.kind)"))
end

"""
    cxx_runtime_libraries(abi) -> Vector{String}

The C++ runtimes a destructor may live in when the wrapped library's own
handle does not reach it, for the host convention `abi`. None on ELF
(`:sysv`, `:aapcs64`): `dlsym` through a library's handle searches the
libraries it depends on. On PE (`:win64`) a handle sees only its own DLL, so
the runtimes MSYS2 CLANG64 and MinGW ship, by name.
"""
cxx_runtime_libraries(abi::Symbol) =
    abi === :win64 ? ["libc++.dll", "libstdc++-6.dll"] : String[]

const _DTORS = Dict{Tuple{String,String},Ptr{Cvoid}}()
const _DTORS_LOCK = ReentrantLock()
const _WARNED = Set{String}()

"""
    destructor(lib, sym; abi = host_record_abi()) -> Ptr{Cvoid}

The address of the destructor `sym` (a complete-object `D1`), found through
`lib`'s own handle, else in `cxx_runtime_libraries(abi)`; `C_NULL` when none
has it. Cached per `(lib, sym)`, so the lookup runs once per host and process.
"""
function destructor(lib::AbstractString, sym::AbstractString; abi::Symbol = host_record_abi())::Ptr{Cvoid}
    lock(_DTORS_LOCK) do
        get!(_DTORS, (String(lib), String(sym))) do
            for l in vcat(String(lib), cxx_runtime_libraries(abi))
                h = Libdl.dlopen(l; throw_error = false)
                h === nothing && continue
                p = Libdl.dlsym(h, sym; throw_error = false)
                p === nothing || return p
            end
            C_NULL
        end
    end
end

"""
    take_string!(slot, reader, lib, sym, fname) -> Union{String, bytes}

The end of a wrapper call returning `std::string`: the C++ object is in
`slot` (`JITManager.return_slot`), where the callee built it. Read it into a
`String`, then call its destructor `sym` on it, even when the read throws. When
this host has no such destructor (`destructor` is `C_NULL`), the slot's bytes
are returned as they always were, and a warning names the function once: what
the string owns on the heap is not freed.
"""
function take_string!(slot, r::Reader, lib::AbstractString, sym::AbstractString, fname::AbstractString)
    d = destructor(lib, sym)
    GC.@preserve slot begin
        p = Ptr{Cvoid}(_slot_address(slot))
        if d == C_NULL
            lock(_DTORS_LOCK) do
                fname in _WARNED || (push!(_WARNED, String(fname));
                    @warn "$fname: std::string returned as its raw bytes, because this host has no " *
                          "`$sym` in $lib or the C++ runtime; what the string owns on the heap is not freed.")
            end
            return slot isa Base.RefValue ? slot[] : slot
        end
        try
            return read_string(p, r)
        finally
            ccall(d, Cvoid, (Ptr{Cvoid},), p)
        end
    end
end
_slot_address(slot::Base.RefValue) = Base.unsafe_convert(Ptr{Cvoid}, slot)
_slot_address(slot) = pointer_from_objref(slot)

end # module StdStrings
