
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Complete set of Julia reserved keywords and soft keywords for enum member escaping
const _JULIA_KEYWORDS = Set([
    "baremodule", "begin", "break", "catch", "const", "continue", "do",
    "else", "elseif", "end", "export", "false", "finally", "for",
    "function", "global", "if", "import", "in", "let", "local",
    "macro", "module", "mutable", "nothing", "quote", "return",
    "struct", "true", "try", "type", "using", "while", "abstract",
    "primitive", "where", "isa",
    # Not technically reserved but conflict as identifiers in enum context
    "and", "or", "not",
])

# Internal/compiler types that leak through DWARF but shouldn't be exported
const _INTERNAL_TYPE_BLOCKLIST = Set([
    "__va_list_tag", "__mbstate_t", "__loadu_pd", "__storeu_pd",
    "__loadu_ps", "__storeu_ps", "__loadu_si128", "__storeu_si128",
    "_va_list_tag", "_mbstate_t", "_loadu_pd", "_storeu_pd",
    "_loadu_ps", "_storeu_ps",
    "ldiv_t", "lldiv_t", "div_t", "max_align_t", "imaxdiv_t",
    "_IO_FILE", "_IO_marker", "_IO_codecvt", "_IO_wide_data",
])

"""Escape a name if it's a Julia keyword, using var\"...\" syntax."""
function _escape_keyword(name::String)::String
    if name in _JULIA_KEYWORDS
        return "var\"$name\""
    end
    return name
end

"""
    _base_shadowing(names) -> Vector{String}

Which of `names` would shadow a Base/Core export if the module were `using`-ed.

A generated module's namespace belongs to the LIBRARY, and the export list is
harvested from every symbol that reached the debug info — libstdc++ included.
Exporting one of these does not break the wrapper itself (that is
`_assert_base_calls_qualified`'s job); it breaks the CONSUMER, silently and at
a distance. Measured on the Hub: llamacpp exports `all`, `error`, `stat` and
`symlink`; sqlite exports `Expr`, `Module` and `stat`; cjson exports `error`.

`using .Llamacpp` therefore made a bare `error("...")` in the caller an
ambiguous binding, so every failure path in that file raised
`UndefVarError: error not defined` instead of its message — invisible until
something actually went wrong, because the happy path never calls it. That cost
a real debugging session in `examples/LlamaChat`, whose fix was to give up on
`using` entirely and go through `L.` for everything.
"""
function _base_shadowing(names_to_export)::Vector{String}
    shadow = Set{String}()
    for m in (Base, Core), n in names(m)
        push!(shadow, String(n))
    end
    return sort!(unique!(filter(n -> n in shadow, collect(String.(names_to_export)))))
end

"""
    _export_statement(all_exports) -> String

The module's `export` line, with Base/Core-shadowing names withheld.

Withheld names are still DEFINED and reachable as `Mod.name` — this drops them
from `export` only, so the library keeps its full API while `using` stops being
a hazard. That is the right trade because the collision is silent for the
consumer and the workaround (`Mod.name`) is both obvious and already what
careful callers do.

Emitted as one derivation because all three generators (C, C++, basic) built
this line themselves, which is exactly how the struct-filler bug shipped in
triplicate.
"""
function _export_statement(all_exports)::String
    names_vec = unique(String.(collect(all_exports)))
    isempty(names_vec) && return ""

    withheld = _base_shadowing(names_vec)
    isempty(withheld) && return "export " * join(names_vec, ", ") * "\n\n"

    keep = filter(n -> !(n in Set(withheld)), names_vec)
    @info "wrap: $(length(withheld)) name(s) withheld from `export` — they would " *
          "shadow a Base/Core binding in any module that `using`s this wrapper. " *
          "Still reachable as <Module>.<name>: " * join(withheld, ", ")

    banner = """
    # ── Withheld from `export` ───────────────────────────────────────────────
    # These $(length(withheld)) name(s) are DEFINED by this module but not exported: each
    # collides with a Base/Core export, and `using` this module would shadow it
    # in the caller's namespace — silently, since the happy path never touches
    # the shadowed binding. Reach them explicitly:  <Module>.$(first(withheld))
    #   $(join(withheld, ", "))
    """
    return banner * "export " * join(keep, ", ") * "\n\n"
end

# Built-in Julia types that never need forward declaration
const _JULIA_BUILTIN_TYPES = Set([
    "Cvoid", "Cint", "Cuint", "Clong", "Culong", "Cshort", "Cushort",
    "Cchar", "Cuchar", "Cfloat", "Cdouble", "Bool", "UInt8", "Int8",
    "UInt16", "Int16", "UInt32", "Int32", "UInt64", "Int64", "Csize_t",
    "Clonglong", "Culonglong", "Cptrdiff_t", "Cssize_t", "Cwchar_t",
    "Cstring", "Float32", "Float64", "Any", "Nothing", "Cintptr_t", "Cuintptr_t",
])

# Type constructors that legitimately appear in a ccall type position but are
# never *defined* by a generated wrapper.
const _JULIA_TYPE_CTORS = Set([
    "Ptr", "Ref", "NTuple", "Tuple", "Union", "Vararg", "String", "Symbol",
])

"""
    _elf_build_id(path) -> Union{Nothing,String}

The GNU build ID of an ELF file as lowercase hex, or `nothing` if absent.

Identifies a BUILD, not a version: two compilations of identical source differ
here. That is exactly the question a generated wrapper needs answered, because
its struct layouts, enum values and blob sizes are a snapshot of one
compilation — point it at a different build and nothing complains, it just
reads the wrong offsets.

Read from the PT_NOTE segment rather than a section header: `strip` drops
sections but keeps segments, so this still works on a stripped library.
"""
function _elf_build_id(path::AbstractString)::Union{Nothing,String}
    isfile(path) || return nothing
    try
        open(path, "r") do io
            read(io, 4) == UInt8[0x7f, 0x45, 0x4c, 0x46] || return nothing  # \x7fELF
            seek(io, 4); read(io, UInt8) == 2 || return nothing             # ELFCLASS64
            seek(io, 0x20); e_phoff = read(io, UInt64)
            seek(io, 0x36); e_phentsize = read(io, UInt16); e_phnum = read(io, UInt16)
            for i in 0:(e_phnum - 1)
                seek(io, e_phoff + i * e_phentsize)
                p_type = read(io, UInt32)
                p_type == 4 || continue                                     # PT_NOTE
                skip(io, 4)                                                 # p_flags
                p_offset = read(io, UInt64); skip(io, 16)                   # vaddr, paddr
                p_filesz = read(io, UInt64)
                pos = p_offset
                stop = p_offset + p_filesz
                while pos + 12 <= stop
                    seek(io, pos)
                    namesz = read(io, UInt32); descsz = read(io, UInt32); ntype = read(io, UInt32)
                    name = String(read(io, namesz))
                    pad(n) = (n + 3) & ~UInt32(3)
                    desc_at = pos + 12 + pad(namesz)
                    if ntype == 3 && startswith(name, "GNU")               # NT_GNU_BUILD_ID
                        seek(io, desc_at)
                        return bytes2hex(read(io, descsz))
                    end
                    pos = desc_at + pad(descsz)
                end
            end
            return nothing
        end
    catch
        return nothing    # unreadable/odd ELF is not worth failing a build over
    end
end

"""
    _slice_declared_symbols(ir) -> Vector{String}

Symbols a slice module binds by `declare` — the exact set the JIT must resolve.

Re-derived from the slice IR that will SHIP rather than carried over from the
Slicer's bookkeeping, so the runtime check can never disagree with the file it
guards. Intrinsics are excluded: LLVM supplies those, they are never dlsym'd.
"""
function _slice_declared_symbols(ir::AbstractString)::Vector{String}
    syms = String[]
    for m in eachmatch(r"^declare[^@\n]*@\"?([A-Za-z0-9_.$]+)\"?\("m, ir)
        push!(syms, m.captures[1])
    end
    for m in eachmatch(r"^@\"?([A-Za-z0-9_.$]+)\"?\s*=\s*external\b"m, ir)
        push!(syms, m.captures[1])
    end
    return sort!(unique!(filter(s -> !startswith(s, "llvm."), syms)))
end

"""
    _defined_type_names(source) -> Set{String}

Every type name a generated module BINDS: struct/mutable struct, `@enum`,
`abstract`/`primitive type`, and `const` aliases.

Derived from the emitted text rather than from generator bookkeeping, so the
forward-reference resolver and the pre-write guard cannot drift apart — they
call this on the same source and agree by construction.
"""
function _defined_type_names(source::AbstractString)::Set{String}
    names = Set{String}()
    for re in (r"^\s*(?:mutable\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)"m,
               r"^\s*@enum\s+([A-Za-z_][A-Za-z0-9_]*)"m,
               r"^\s*abstract\s+type\s+([A-Za-z_][A-Za-z0-9_]*)"m,
               r"^\s*primitive\s+type\s+([A-Za-z_][A-Za-z0-9_]*)"m,
               r"^\s*const\s+([A-Za-z_][A-Za-z0-9_]*)\s*="m)
        for m in eachmatch(re, source)
            push!(names, m.captures[1])
        end
    end
    return names
end

# Docstrings quote the same types they document, so an undefined name shows up
# there too — harmlessly, since it is inside a string. Strip them before
# scanning for real uses.
_strip_docstrings(source::AbstractString) = replace(source, r"\"\"\".*?\"\"\""s => "")

"""
    _undefined_ccall_types(source) -> Vector{Tuple{String,String}}

Type names used in a foreign-call type position that the module never binds,
as `(type, enclosing function)` pairs.

This is the load-blocking class: `ccall(..., (Ptr{Ptr{_IO_FILE}}, ...), ...)`
where `_IO_FILE` is never declared raises `UndefVarError` at module include —
so nothing in the wrapper works, not merely the one function. Checking the
artifact (rather than trusting the generator's own name bookkeeping) is the
point: it catches any path that emits a type, including ones added later.
"""
function _undefined_ccall_types(source::AbstractString)::Vector{Tuple{String,String}}
    code = _strip_docstrings(source)
    known = union(_defined_type_names(code), _JULIA_BUILTIN_TYPES, _JULIA_TYPE_CTORS)

    # Offsets of `function <name>(` so a hit can name where it lives.
    fn_at = [(m.offset, m.captures[1])
             for m in eachmatch(r"^\s*function\s+([A-Za-z_][A-Za-z0-9_!]*)"m, code)]
    enclosing(off) = begin
        i = searchsortedlast([f[1] for f in fn_at], off)
        i == 0 ? "<module scope>" : fn_at[i][2]
    end

    bad = Tuple{String,String}[]
    seen = Set{Tuple{String,String}}()

    # ONLY ccall type positions. `ccall` resolves its return type and argument
    # type tuple eagerly, when the method is defined — that is why an undefined
    # name there kills the module at include. Everywhere else a type name is
    # lazy: a `::T` return annotation, a `cglobal(…, T)` argument, and a
    # function body all resolve on first CALL, so an undeclared type there
    # costs one function and loads fine. stl_test proves the distinction — its
    # wrapper references an undeclared `piecewise_construct_t` in exactly those
    # lazy positions and has always loaded (verified by including it).
    # Scanning more broadly turns working wrappers into refused ones.
    for m in eachmatch(r"ccall\(\([^)]*\)\s*,\s*([^,]+),\s*\(([^)]*)\)", code)
        pos = m.offset
        for entry in vcat(String(m.captures[1]), split(m.captures[2], ','))
            t = strip(String(entry))
            isempty(t) && continue
            # Unwrap Ptr{…}/Ref{…} down to the leaf name.
            while (inner = match(r"^(?:Ptr|Ref)\{(.+)\}$", t)) !== nothing
                t = strip(String(inner.captures[1]))
            end
            occursin(r"^[A-Za-z_][A-Za-z0-9_]*$", t) || continue
            t in known && continue
            key = (t, enclosing(pos))
            key in seen || (push!(seen, key); push!(bad, key))
        end
    end
    return bad
end

"""
    _resolve_forward_ptr(julia_type, defined_names) -> String

Replace `Ptr{X}` (including nested `Ptr{Ptr{X}}`) with `Ptr{Cvoid}` when `X`
is a custom struct type that has not been defined yet. This avoids
`UndefVarError` from forward references while preserving ABI layout
(all pointers are the same size).
"""
function _resolve_forward_ptr(julia_type::AbstractString, defined_names::Set{String})::String
    m = match(r"^Ptr\{(.+)\}$", julia_type)
    isnothing(m) && return julia_type
    inner = m.captures[1]
    # Recurse to resolve the inner type (handles nested Ptr{Ptr{...}})
    resolved_inner = _resolve_forward_ptr(inner, defined_names)
    # If recursion already produced a Ptr (nested case), it's been validated — keep it.
    # Otherwise check if the bare type name is known (builtin or already defined).
    if startswith(resolved_inner, "Ptr{") || resolved_inner in _JULIA_BUILTIN_TYPES || resolved_inner in defined_names
        return "Ptr{$resolved_inner}"
    end
    # Inner type is an unknown bare name — replace with Cvoid to avoid UndefVarError
    return "Ptr{Cvoid}"
end

# =============================================================================
# DISPATCH LOGIC HELPERS
# =============================================================================

"""
    get_julia_aligned_size(members::Vector)

Calculate the size of a struct in Julia including standard padding alignment.
Used to detect if a C++ struct is 'packed' (Julia size > DWARF size).
"""
function get_julia_aligned_size(members::Vector)
    current_offset = 0
    max_align = 1

    for m in members
        # specific size of this member
        m_size = get(m, "size", 0)

        # simple alignment heuristic (size usually equals alignment for primitives)
        # generic pointer/int alignment cap at 8 bytes on 64-bit
        align = m_size > 8 ? 8 : m_size
        align = align == 0 ? 1 : align # handle empty/void

        # update generic alignment requirement
        max_align = max(max_align, align)

        # Add padding to current offset
        padding = (align - (current_offset % align)) % align
        current_offset += padding + m_size
    end

    # Final structure alignment padding
    padding = (max_align - (current_offset % max_align)) % max_align
    return current_offset + padding
end

"""
    _parse_int_or_hex(raw) -> Int

Parse a value that may be decimal, hex (0x…), or a non-string number.
Returns 0 on failure.
"""
function _parse_int_or_hex(raw)::Int
    s = raw isa String ? raw : string(raw)
    try
        (startswith(s, "0x") || startswith(s, "0X")) ? parse(Int, s[3:end], base=16) : parse(Int, s)
    catch
        0
    end
end

"""
    _parse_dwarf_size(s_info) -> Int

Parse byte_size from a DWARF struct info dict, handling both decimal and hex.
"""
function _parse_dwarf_size(s_info)::Int
    _parse_int_or_hex(get(s_info, "byte_size", "0"))
end

# Memoization cache for _is_struct_unsafe. Within a single wrap_introspective
# call the same `s_info` Dict is queried many times (once per function param
# and once per return). Keyed by object identity so we don't pay the cost of
# hashing nested Dicts. Bounded by the number of structs in DWARF, dropped
# when those Dicts go out of scope.
const _STRUCT_UNSAFE_CACHE = IdDict{Any,Bool}()

"""
    _is_struct_unsafe(s_info, dwarf_structs) -> Bool

Check if a struct type is unsafe for ccall (should route to MLIR).
Uses all available DWARF metadata: size, alignment, packing, polymorphism,
inheritance, and member types.
"""
function _is_struct_unsafe(s_info, dwarf_structs)::Bool
    # Only cache top-level queries (those that pass a non-empty dwarf_structs).
    # The recursive nested check at the bottom of this function passes an empty
    # dict to avoid cycles; caching those would produce wrong results when the
    # same struct is queried at the top level later.
    is_top_level = !isempty(dwarf_structs)
    if is_top_level
        cached = get(_STRUCT_UNSAFE_CACHE, s_info, nothing)
        cached !== nothing && return cached
    end
    result = _is_struct_unsafe_impl(s_info, dwarf_structs)
    is_top_level && (_STRUCT_UNSAFE_CACHE[s_info] = result)
    return result
end

function _is_struct_unsafe_impl(s_info, dwarf_structs)::Bool
    # Packed struct: DWARF size != Julia calculated size (alignment mismatch)
    dwarf_size = _parse_dwarf_size(s_info)
    members = get(s_info, "members", [])

    # Resolve member sizes: when a member's c_type is itself a struct in
    # dwarf_structs and the DWARF parser left its size=0, substitute the
    # nested struct's byte_size so get_julia_aligned_size computes correctly.
    resolved_members = map(members) do m
        if get(m, "size", 0) == 0
            m_type = strip(replace(get(m, "c_type", ""), r"\bconst\b" => ""))
            if haskey(dwarf_structs, m_type)
                nested_size = _parse_dwarf_size(dwarf_structs[m_type])
                if nested_size > 0
                    m2 = copy(m)
                    m2["size"] = nested_size
                    return m2
                end
            end
        end
        return m
    end

    julia_size = get_julia_aligned_size(resolved_members)
    if dwarf_size > 0 && julia_size > 0 && dwarf_size != julia_size
        return true
    end

    # Overaligned struct: alignment > 8 means SIMD/cache-line alignment
    # that ccall won't respect (Julia uses natural alignment capped at 8)
    alignment = get(s_info, "alignment", 0)
    alignment = alignment isa String ? (try parse(Int, alignment) catch; 0 end) : alignment
    if alignment > 8
        return true
    end

    # Union: layout is overlapping, ccall can't handle by-value unions
    if get(s_info, "kind", "struct") == "union"
        return true
    end

    # Polymorphic: has vtable pointer, non-trivial copy/move semantics
    if get(s_info, "is_polymorphic", false) == true
        return true
    end

    # Inherits from another class: likely non-trivial layout (vptr, padding)
    base_classes = get(s_info, "base_classes", [])
    if !isempty(base_classes)
        return true
    end

    # Class keyword: C++ class defaults to private members, often non-POD.
    # Only flag if it also has members (empty tag classes are safe).
    if get(s_info, "kind", "struct") == "class" && !isempty(members)
        return true
    end

    # Member contains a nested struct that is itself unsafe
    for m in members
        m_type = strip(replace(get(m, "c_type", ""), r"\bconst\b" => ""))
        # Skip pointers and references — those are just addresses
        if endswith(m_type, "*") || endswith(m_type, "&") || contains(m_type, "*")
            continue
        end
        if haskey(dwarf_structs, m_type)
            nested = dwarf_structs[m_type]
            # Recursive check (one level deep to avoid cycles)
            if _is_struct_unsafe(nested, Dict{String,Any}())
                return true
            end
        end
    end

    return false
end

"""
Canonical set of C/C++ primitive type names that are safe for ccall.
Used by `is_ccall_safe()` to distinguish primitives from struct types
without fragile substring matching (e.g. `contains(ret_type, "int")`
would match "Point").
"""
const _CCALL_SAFE_PRIMITIVES = Set([
    "void", "int", "unsigned int", "signed int",
    "char", "unsigned char", "signed char",
    "short", "unsigned short", "short int",
    "long", "unsigned long", "long int", "long long", "unsigned long long",
    "float", "double",
    "bool", "_Bool",
    "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "size_t", "ssize_t", "ptrdiff_t", "intptr_t", "uintptr_t", "wchar_t",
])

"""
    _is_primitive_type(c_type::AbstractString) -> Bool

Check if a C type string (after stripping const/volatile/whitespace) is a
known primitive. Uses `_CCALL_SAFE_PRIMITIVES` set for O(1) lookup.
"""
function _is_primitive_type(c_type::AbstractString)::Bool
    cleaned = strip(replace(c_type, r"\b(const|volatile|restrict)\b" => ""))
    cleaned = strip(replace(cleaned, r"\s+" => " "))
    return cleaned in _CCALL_SAFE_PRIMITIVES
end

# is_ccall_safe() is defined in Wrapper/DispatchLogic.jl (included after TypesCpp.jl
# to have access to is_stl_container_type)

# =============================================================================
# CSTRING RETURN POLICY
# =============================================================================

"""
    _cstring_policy_lines(free_sym; indent="    ") -> String

Shared body for every `Cstring`-returning wrapper, applied after the call
result is bound to `ptr`:

- `NULL` → `nothing` (a NULL `char*` is a value in C APIs, not an error)
- otherwise copy to a Julia `String`
- when `free_sym` is nonempty the C buffer is released through that library
  symbol after the copy — declared per function in `[wrap.cstring_owned]`
  (`funcname = "free_symbol"`), because ownership of a returned `char*` is
  not recoverable from DWARF.

Every emission site (base wrapper, array convenience, vararg base and typed
overloads, C and C++ generators) splices these same lines so the policy
cannot drift between sites again.
"""
function _cstring_policy_lines(free_sym::String; indent::String="    ")::String
    free_line = isempty(free_sym) ? "" :
        "$(indent)ccall((:$(free_sym), LIBRARY_PATH), Cvoid, (Cstring,), ptr)\n"
    return "$(indent)ptr == C_NULL && return nothing\n" *
           "$(indent)s = unsafe_string(ptr)\n" *
           free_line *
           "$(indent)return s"
end


# =============================================================================
# DUPLICATE-METHOD DEDUPLICATION (package-precompilation safety)
# -----------------------------------------------------------------------------
# Distinct C++ symbols can collapse to the SAME Julia function name and
# argument signature: destructor D1/D2 pairs both become `X_destroy_X(this)`,
# and overloads whose parameter types all map to `::Any` (e.g. a const char*
# and a std::string_view overload) become textually identical methods. At
# script include() that is a benign last-definition-wins warning — but under
# PACKAGE PRECOMPILATION method overwriting is a hard ERROR, so a generated
# wrapper vendored into a package would refuse to precompile (found live
# vendoring the box2d wrapper, 2026-07-19).
#
# The fix preserves include()'s observable semantics exactly: keep the LAST
# definition of each colliding signature and drop the earlier ones (with
# their docstrings, since a chunk is docstring + definition).
# =============================================================================

const _FUNC_SIG_RE = r"^function\s+([A-Za-z_][A-Za-z0-9_!.]*)\((.*?)\)(?:::\S.*)?\s*$"m

"""
    _method_sig_keys(chunk) -> Vector{String}

Dispatch-significant signature keys (`name(argtype,argtype,…)`) for every
top-level `function` definition in an emitted code chunk. Keyword arguments
and default values are ignored — they don't participate in dispatch.
"""
function _method_sig_keys(chunk::AbstractString)
    keys = String[]
    for m in eachmatch(_FUNC_SIG_RE, chunk)
        name = m.captures[1]
        args = first(split(m.captures[2], ';'; limit=2))   # drop kwargs
        argtypes = String[]
        if !isempty(strip(args))
            for a in split(args, ',')
                a = strip(first(split(a, '='; limit=2)))   # drop default value
                isempty(a) && continue
                push!(argtypes, occursin("::", a) ?
                      strip(last(split(a, "::"; limit=2))) : "Any")
            end
        end
        push!(keys, string(name, "(", join(argtypes, ","), ")"))
    end
    return keys
end

"""
    _dedup_method_chunks(chunks) -> Vector{String}

Drop emitted chunks whose every `function` signature is redefined by a LATER
chunk, so the generated module contains exactly one definition per dispatch
signature. Required for the wrapper to precompile inside a package.
"""
function _dedup_method_chunks(chunks::Vector{String})
    seen = Set{String}()
    keep = trues(length(chunks))
    ndropped = 0
    for i in length(chunks):-1:1
        ks = _method_sig_keys(chunks[i])
        isempty(ks) && continue
        if all(k -> k in seen, ks)
            keep[i] = false
            ndropped += 1
        else
            union!(seen, ks)
        end
    end
    if ndropped > 0
        @info "wrap: dropped $ndropped duplicate method definition(s) — identical Julia name+signature from distinct C++ symbols; last definition kept (precompilation-safe)"
    end
    return chunks[keep]
end

# =============================================================================
# BYTE-BLOB STRUCT SETTERS
# =============================================================================
#
# A struct whose layout could not be modelled field-by-field is emitted as an
# immutable `_data::NTuple{N,UInt8}` blob with `Base.getproperty` accessors
# reading each member at its DWARF offset. Nothing was emitted in the other
# direction, so a param struct built BY the library
# (`llama_context_default_params()`) was read-only from Julia — and for
# llama.cpp that is the only path in: an embedding model returns NULL unless
# `ctx_params.embeddings` is set. Callers were left patching bytes through
# hand-rolled offset tables read out of `compilation_metadata.json`.
#
# The offsets and types are exactly what `getproperty` already emits, so the
# setter is that same information in the other direction. Immutability is not
# an obstacle — it only means the setter RETURNS a new value instead of
# mutating. `Base.setproperty!` is defined too, purely to replace Julia's
# "immutable struct cannot be modified" with a message naming the alternative,
# because `x.field = v` is what someone will type first.

"""
    _blob_store_expr(m_name, offset, kind; ...) -> String

The write matching one `getproperty` branch: an `s === :field` condition plus a
body writing through `p`. Emitted as a bare condition so the caller can chain
the branches with `elseif` — one exit, so nothing returns out of the enclosing
`GC.@preserve` region.

`kind` mirrors the getter's own type dispatch exactly. A member the getter
skips gets no setter, and vice versa: a field settable but unreadable (or the
reverse) would be worse than neither.
"""
function _blob_store_expr(m_name::AbstractString, offset::Int, kind::Symbol;
                          julia_type::String="", struct_name::String="",
                          nested_size::Int=0, actual_size::Int=0)
    store = if kind === :pointer
        "unsafe_store!(Ptr{Ptr{Cvoid}}(p + $offset), convert(Ptr{Cvoid}, v))"
    elseif kind === :primitive
        "unsafe_store!(Ptr{$julia_type}(p + $offset), convert($julia_type, v))"
    elseif kind === :typed_struct
        "unsafe_store!(Ptr{$struct_name}(p + $offset), convert($struct_name, v))"
    elseif kind === :blob_struct_full
        "unsafe_store!(Ptr{NTuple{$nested_size,UInt8}}(p + $offset), " *
        "getfield(convert($struct_name, v), :_data))"
    elseif kind === :blob_struct_partial
        # The getter zero-pads a member the struct has no room for; the setter
        # must write only the bytes that FIT or it scribbles over the next one.
        join(["let _b = getfield(convert($struct_name, v), :_data)",
              "                for _i in 1:$actual_size",
              "                    unsafe_store!(Ptr{UInt8}(p + $offset + _i - 1), _b[_i])",
              "                end",
              "            end"], "\n")
    else
        return ""
    end
    return "s === :$m_name\n            $store"
end

"""
    _blob_setter_chunk(name, store_branches, delegate_branches) -> String

Emit `setproperty(::name, ::Symbol, v)` plus the `setproperty!` signpost.

`delegate_branches` are complete `if ... end` blocks placed BEFORE the write
chain, for C11 anonymous-member names that must round-trip through the member
declaring them rather than write bytes directly.
"""
function _blob_setter_chunk(name::String, store_branches::Vector{String},
                            delegate_branches::Vector{String}=String[])
    (isempty(store_branches) && isempty(delegate_branches)) && return ""

    L = String[]
    push!(L, "\"\"\"")
    push!(L, "    setproperty(x::$name, s::Symbol, v) -> $name")
    push!(L, "")
    push!(L, "Return a copy of `x` with field `s` set to `v`. `$name` is an immutable byte")
    push!(L, "blob because it crosses the ABI by value, so this returns a new value rather")
    push!(L, "than mutating it. See also `setproperties`.")
    push!(L, "\"\"\"")
    push!(L, "function setproperty(x::$name, s::Symbol, v)")
    append!(L, delegate_branches)
    if isempty(store_branches)
        push!(L, "    Base.error(\"type $name has no settable field \$s\")")
    else
        push!(L, "    buf = Ref(getfield(x, :_data))")
        push!(L, "    GC.@preserve buf begin")
        push!(L, "        p = Ptr{UInt8}(pointer_from_objref(buf))")
        for (i, b) in enumerate(store_branches)
            push!(L, "        " * (i == 1 ? "if " : "elseif ") * b)
        end
        push!(L, "        else")
        push!(L, "            Base.error(\"type $name has no settable field \$s\")")
        push!(L, "        end")
        push!(L, "    end")
        push!(L, "    return $name(buf[])")
    end
    push!(L, "end")
    push!(L, "")
    push!(L, "function Base.setproperty!(x::$name, s::Symbol, v)")
    push!(L, "    Base.error(\"`$name` is an immutable byte blob (it crosses the ABI by value), \" *")
    push!(L, "          \"so `x.\$s = ...` cannot work. Use `x = setproperty(x, :\$s, v)` \" *")
    push!(L, "          \"or `x = setproperties(x; \$s = v)`.\")")
    push!(L, "end")
    push!(L, "")
    return join(L, "\n") * "\n"
end

"""
    _blob_setproperties_chunk() -> String

The module-level bulk form, emitted once when any struct got a setter. Folding
`setproperty` over keywords is what collapses the repeated rebinding into one
readable call.
"""
_blob_setproperties_chunk() = """
\"\"\"
    setproperties(x; field = value, ...) -> typeof(x)

Return a copy of `x` with each named field set. Byte-blob structs are immutable
(they cross the ABI by value), so this returns a new value rather than mutating:

    cp = setproperties(llama_context_default_params(); n_ctx = 512, embeddings = true)
\"\"\"
setproperties(x; kwargs...) =
    foldl((acc, kv) -> setproperty(acc, first(kv), last(kv)), pairs(kwargs); init = x)

"""
