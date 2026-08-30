
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

# Internal/compiler types that leak through DWARF but shouldn't be exported.
# The set now lives at package level (RepliBuild.INTERNAL_TYPE_BLOCKLIST) so the
# IR generator — which loads before this module — screens the same types. Kept
# under the old name because it is referenced from six files in Wrapper.
import ..INTERNAL_TYPE_BLOCKLIST
const _INTERNAL_TYPE_BLOCKLIST = INTERNAL_TYPE_BLOCKLIST

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
    _defined_names(module_body) -> Set{String}

Every name the emitted module body actually BINDS, read off the parsed source.

This is the counterpart to `emitted_type_names` in the C generator: the export
list must be filtered against what was emitted, never against the generator's
intent bookkeeping, because those two disagreeing IS the bug (see
`_export_statement`).

Collected: `function f(…)`, `f(…) = …`, `const x = …`, `x = …`, `struct`/
`mutable struct`, `abstract type`, `primitive type`, `macro`, and `@enum` —
both the enum type and every member, since `@enum T::U begin A = 1 end` binds
`A` as surely as a `const` does.

Deliberately does NOT descend into function bodies or struct bodies: a local
variable and a field name are not module bindings. Recursion is limited to
`:toplevel`, `:module` and `:block` so a `let` at module scope is the only
over-collection route, and generated wrappers only use `let` on the RHS of a
`const` (never descended into).

An unparseable body returns an empty set rather than throwing —
`_assert_wrapper_parses` owns that diagnosis and reports it far better.
"""
function _defined_names(module_body::AbstractString)::Set{String}
    out = Set{String}()
    ex = try
        Meta.parseall(module_body)
    catch
        return out
    end
    _collect_defined!(out, ex)
    return out
end

# Peel a definition target down to the bound name: `f`, `f(x)`, `f(x) where T`,
# `T{P}`, `T <: S`, `name::Type` all bind `f`/`T`/`name`.
function _bound_name(x)
    x isa Symbol && return String(x)
    x isa Expr || return nothing
    x.head in (:call, :where, :curly, :(<:), :(::)) && !isempty(x.args) &&
        return _bound_name(x.args[1])
    return nothing
end

function _collect_defined!(out::Set{String}, ex)
    ex isa Expr || return out
    h = ex.head

    if h === :function || h === :macro
        n = _bound_name(get(ex.args, 1, nothing)); n === nothing || push!(out, n)
    elseif h === :(=)
        # Covers both `f(x) = …` (short-form method) and `x = …` (global).
        n = _bound_name(ex.args[1]); n === nothing || push!(out, n)
    elseif h === :const || h === :global || h === :local
        for a in ex.args; _collect_defined!(out, a); end
    elseif h === :struct
        n = _bound_name(get(ex.args, 2, nothing)); n === nothing || push!(out, n)
    elseif h === :abstract || h === :primitive
        n = _bound_name(get(ex.args, 1, nothing)); n === nothing || push!(out, n)
    elseif h === :macrocall && !isempty(ex.args) && ex.args[1] === Symbol("@enum")
        # `@enum Name::T begin A = 1 … end` and `@enum Name A B C` both bind the
        # type AND every member. Members carry the emitter's FINAL spelling,
        # which is the whole reason this is read back rather than re-derived.
        rest = filter(a -> !(a isa LineNumberNode), ex.args[2:end])
        if !isempty(rest)
            n = _bound_name(rest[1]); n === nothing || push!(out, n)
            for m in rest[2:end], s in (m isa Expr && m.head === :block ? m.args : (m,))
                s isa LineNumberNode && continue
                nm = s isa Expr && s.head === :(=) ? _bound_name(s.args[1]) : _bound_name(s)
                nm === nothing || push!(out, nm)
            end
        end
    elseif h === :macrocall
        # A macro WRAPPING a definition still defines it, so descend. The
        # generators lean on this constantly and it is entirely invisible in the
        # source text: a docstring written directly above a definition, with no
        # blank line between, parses as `@doc "…" <definition>` — so
        # `function get_hdr(…)` becomes a macrocall argument, not a toplevel
        # `:function`. Skipping these dropped every DOCUMENTED name while
        # keeping the undocumented ones, which is how the first draft of this
        # filter deleted `setproperty`, `g_count` and the bitfield accessors
        # from a wrapper's export line. Also covers `@generated function
        # _TIER1_x(…)` and `@inline f() = …`, both real emission shapes.
        for a in ex.args; _collect_defined!(out, a); end
    end

    if h === :toplevel || h === :module || h === :block
        for a in ex.args; _collect_defined!(out, a); end
    end
    return out
end

"""
    _exported_names(module_body) -> Vector{String}

Every name the emitted module body EXPORTS, read off the parsed source.

The other half of `_defined_names`: together they let a guard compare the two
lists the generator is supposed to keep in agreement, without consulting either
of the bookkeeping structures that produced them.
"""
function _exported_names(module_body::AbstractString)::Vector{String}
    out = String[]
    ex = try
        Meta.parseall(module_body)
    catch
        return out
    end
    _collect_exports!(out, ex)
    return unique!(out)
end

function _collect_exports!(out::Vector{String}, ex)
    ex isa Expr || return out
    if ex.head === :export
        for a in ex.args
            n = _bound_name(a); n === nothing || push!(out, n)
        end
    elseif ex.head === :toplevel || ex.head === :module || ex.head === :block
        for a in ex.args; _collect_exports!(out, a); end
    end
    return out
end

"""
    _export_statement(all_exports, module_body="") -> String

The module's `export` line: names the body does not define are dropped, and
Base/Core-shadowing names are withheld.

Withheld names are still DEFINED and reachable as `Mod.name` — that drops them
from `export` only, so the library keeps its full API while `using` stops being
a hazard. That is the right trade because the collision is silent for the
consumer and the workaround (`Mod.name`) is both obvious and already what
careful callers do.

Dropped names are different: exporting a name the module never binds is a
promise it cannot keep. `using` still succeeds (Julia does not check export
targets at load), so it stays invisible until someone reaches the name — or
until a doc generator, an introspection pass, or plain REPL tab-completion
walks `names(Mod)` and hits `UndefVarError` on a name Julia itself suggested.

Measured across the Hub before this filter existed: **102 undefined exports in
5 of 18 packages** (sqlite 64, llamacpp 22, lua 12, miniaudio 2, zlib 2), from
two derivations that had drifted apart:
  - union accessors screened out of the DEFINITIONS but left in the export list
  - enum members exported under their raw C spelling while `@enum` binds the
    sanitized one (`__RLIMIT_NICE` vs `_RLIMIT_NICE`, `COPY_` vs `COPY`,
    `ma_dr_wav__…` vs `ma_dr_wav_…` — leading, trailing and interior
    underscore transforms, all the same class)

Emitted as one derivation because all three generators (C, C++, basic) built
this line themselves, which is exactly how the struct-filler bug shipped in
triplicate.
"""
function _export_statement(all_exports, module_body::AbstractString = "")::String
    names_vec = unique(String.(collect(all_exports)))
    isempty(names_vec) && return ""

    if !isempty(module_body)
        defined = _defined_names(module_body)
        if !isempty(defined)      # empty ⇒ unparseable; leave the list alone
            dropped = sort!(filter(n -> !(n in defined), names_vec))
            if !isempty(dropped)
                @info "wrap: $(length(dropped)) name(s) dropped from `export` — the " *
                      "module never binds them, so `using` would offer a name that " *
                      "raises UndefVarError: " * join(first(dropped, 12), ", ") *
                      (length(dropped) > 12 ? " … (+$(length(dropped) - 12) more)" : "")
                names_vec = filter(n -> n in defined, names_vec)
                isempty(names_vec) && return ""
            end
        end
    end

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
    _assert_no_bound_name_rejected(rejected, what)

Refuse to drop generated code because of a name that is already BOUND.

Every "is this type declared?" screen in the generators works by regex-scraping
names out of emitted source and testing them against the set of names the
module binds. That test answers the right question only for names that
*identify a type*; a scrape that also picks up a type CONSTRUCTOR asks it of
`Ptr`, which no wrapper defines and every module has — so the screen rejects on
a name that was never in doubt.

That is not hypothetical: the union-accessor filter's `::([A-Za-z_]\\w*)` regex
captures `Ptr` from `::Ptr{Cvoid}`, and screening it dropped **every**
pointer-typed union accessor in the Hub regardless of its pointee — 76 of them,
all fully resolvable (sqlite's `p4union` kept 1 member of 16, lua's `Value` 3 of
6). The drop is silent by construction: the accessor is simply absent, so the
only symptom is a union you cannot read a pointer out of, and nothing
distinguishes that from a member the generator was right to skip.

So the rejects themselves get screened. A name bound in `Base`/`Core` resolves
inside the generated module (which does `using Base`) and therefore CANNOT be
the undeclared-type case these filters exist to catch. Two ways to get here,
both needing a human: a constructor missing from `_JULIA_TYPE_CTORS`, or a
wrapper type genuinely colliding with a `Base` name — where silently keeping the
accessor would bind `Base`'s type and misread the member's bytes. Hard error,
because both are wrong answers, not risky ones.
"""
function _assert_no_bound_name_rejected(rejected, what::AbstractString)
    bound = sort!(collect(Set(String(n) for n in rejected
                              if isdefined(Base, Symbol(n)) || isdefined(Core, Symbol(n)))))
    isempty(bound) && return nothing
    Base.error("""
        $what rejected on $(length(bound)) name(s) that are already bound: $(join(bound, ", "))

        These screens drop code that names a type the module never binds. A name
        bound in Base/Core always resolves inside the generated module, so it can
        never be that case — the screen is answering the wrong question about it.

        Either it is a type constructor (add it to `_JULIA_TYPE_CTORS` in
        Wrapper/Utils.jl), or a wrapper type collides with a Base name and needs a
        deliberate decision — keeping it would silently bind Base's type.
        """)
end

"""
    _library_sha256(path) -> String

SHA-256 of a library file as lowercase hex, or `""` if it cannot be read.

Identifies a BUILD, not a version: two compilations of identical source differ
here. That is exactly the question a generated wrapper needs answered, because
its struct layouts, enum values and blob sizes are a snapshot of one
compilation — point it at a different build and nothing complains, it just
reads the wrong offsets.

This used to read the GNU build ID out of PT_NOTE. A build ID is only present
if the linker was asked to emit one, and RepliBuild never asks: the C bucket
links through `Clang_unified_jll` (see `_clang_for_c_bucket`), which unlike the
system driver emits no note by default, so every C library built after that
routing landed carried no build ID and the check silently did nothing. Hashing
the file depends on no linker flag, behaves identically for the C and C++
buckets, and is stronger — it also catches post-link modification.

The one thing it gives up: a build ID survives `strip`, a file hash does not.
The `binary` stage strips before `wrap` runs, so the hash is taken of the
already-stripped file and the normal flow is unaffected; stripping a library
*after* generating its wrapper will warn.
"""
function _library_sha256(path::AbstractString)::String
    isfile(path) || return ""
    try
        return bytes2hex(open(SHA.sha256, path))
    catch
        return ""    # unreadable file is not worth failing a build over
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
    function record!(entries, pos)
        for entry in entries
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

    for m in eachmatch(r"ccall\(\([^)]*\)\s*,\s*([^,]+),\s*\(([^)]*)\)", code)
        record!(vcat(String(m.captures[1]), split(m.captures[2], ',')), m.offset)
    end

    # The `@ccall` form is EQUALLY EAGER and was not scanned. It is the shape
    # `generate_vararg_wrappers` emits — `@ccall LIB.var"sym"(a::T, b::U; va_1::V)::R`
    # — so every variadic function was invisible to this guard. libcurl proved
    # it: `curl_mfprintf`'s `fd::Ptr{_IO_FILE}` was written out, refused by
    # Julia at include, and took all 1004 functions with it while this function
    # reported nothing wrong. Argument annotations here are types, not values,
    # and lower to a foreigncall with the same eager resolution as above.
    for m in eachmatch(r"@ccall\s+[A-Za-z_][\w.]*\.var\"[^\"]*\"\(([^)]*)\)::([A-Za-z_][\w{}, ]*)", code)
        anns = String[]
        for part in _split_toplevel_commas(replace(String(m.captures[1]), ';' => ','))
            occursin("::", part) && push!(anns, last(split(part, "::"; limit = 2)))
        end
        record!(vcat(anns, String(m.captures[2])), m.offset)
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

"""
    _cstring_wrapper_pair(julia_name, param_sig, raw_bind, raw_return, free_sym;
                          doc_comment="") -> String

The COMPLETE emission for a `char*`-returning function: the policy wrapper
(`Union{String,Nothing}`) and its raw `<name>_ptr` sibling, as one chunk.

The dispatch tier supplies only the two call bodies — `raw_bind` leaves the raw
pointer in a local named `ptr`, `raw_return` returns it directly — and decides
nothing else. The NULL policy, the copy, the `[wrap.cstring_owned]` free, the
`_ptr` sibling and its docstring are derived here, once, for every tier and both
generators.

**Why this is one function and not a branch in each caller.** It used to be
inline in the ccall path of both generators, so the C++ MLIR-dispatch path —
which `continue`s past that code — emitted a bare `Cstring` instead: 77
functions across five Hub packages returned an unwrapped pointer with no `_ptr`
sibling, and any `[wrap.cstring_owned]` declaration on one of them was silently
ignored (found 2026-08-12). A tier decides HOW to call, never how the result is
presented. [`_assert_cstring_policy`](@ref) enforces that on the way out.
"""
function _cstring_wrapper_pair(julia_name::AbstractString, param_sig::AbstractString,
                               raw_bind::AbstractString, raw_return::AbstractString,
                               free_sym::AbstractString; doc_comment::AbstractString="")::String
    owned_note = isempty(free_sym) ? "" : ", NOT freed — caller owns the buffer"
    return """
    $doc_comment
    function $julia_name($param_sig)::Union{String,Nothing}
    $raw_bind
    $(_cstring_policy_lines(String(free_sym)))
    end

    \"\"\"
        $(julia_name)_ptr($param_sig) -> Cstring

    Raw-pointer variant of `$julia_name`: returns the C `char*` unchanged
    (no copy, no NULL check$(owned_note)).
    \"\"\"
    function $(julia_name)_ptr($param_sig)::Cstring
    $raw_return
    end

    """
end


# =============================================================================
# DISPATCH INTROSPECTION + LAYOUT FACTS
# -----------------------------------------------------------------------------
# Both of these exist because twelve Hub consumers wrote them by hand first.
#
# `kernel_emits_llvmcall` — reach into the private `_TIER1_*` kernel, call
# `code_typed`, string-match "llvmcall" — appears in 12 consumer files, 11 of
# them byte-identical. They do that because `TIER1_FUNCTIONS` records what the
# generator INTENDED and the `@generated` kernel can demote at generation time,
# so nobody trusts the exported set. And `struct_size`/`meta_offset` — re-parsing
# `compilation_metadata.json` for facts the generator held while emitting —
# appears in four more. A helper written once is ergonomics; a helper written
# twelve times independently is a missing feature.
# =============================================================================

"""
    _extra_link_libs_snippet(config) -> String

`__init__` prologue that pre-loads `[ingest] extra_link_libs`, or `""` when the
config declares none.

**Must run BEFORE the main library is opened.** An ingested `.so` is someone
else's build: if it was linked without recording a dependency in `DT_NEEDED`, or
that dependency lives somewhere the loader will not search, opening the main
library fails with an undefined-symbol error naming a symbol this wrapper never
mentions — the least actionable error the pipeline can produce. Opening these
`RTLD_GLOBAL` first puts their symbols in the global namespace so resolution
succeeds.

Names follow the `-l` convention the TOML key is named for, so `"m"` means
`libm`. A value that is already a path or a full soname is passed through: the
candidates are tried in order and the first that opens wins.

`-l` is a LINK-time name and does not always survive into a runtime one. On
glibc ≥ 2.34 `m`, `pthread`, `dl` and `rt` are all merged into libc and
`/usr/lib/libm.so` is a linker script rather than an object, so `-lm` links and
`dlopen("libm.so")` fails — correctly, since there is nothing left to preload.
Prefer a full soname or a path (`"libfoo.so.1"`, `"/opt/foo/lib/libfoo.so"`) for
anything that genuinely needs loading.

This was declared, documented and serialized for its whole life without ever
being read — the only code that touched it was the TOML writer and a test
asserting it round-tripped through the parser. See `test_config_surface.jl`,
which now refuses that class.
"""
function _extra_link_libs_snippet(config)
    # Plain field access, not `getfield`: the config-surface guard finds consumers
    # by reading the source, and `getfield(config, :ingest)` is invisible to it —
    # it reported this very function's field as unconsulted.
    ing = config.ingest
    ing === nothing && return ""
    libs = ing.extra_link_libs
    isempty(libs) && return ""
    lit = "[" * join(("\"" * escape_string(l) * "\"" for l in libs), ", ") * "]"
    return """
        # [ingest] extra_link_libs — opened RTLD_GLOBAL before the main library so
        # its undefined symbols can resolve against them. `-l` naming: "m" => libm.
        for _lib in $lit
            _h = nothing
            for _cand in (_lib, "lib" * _lib, "lib" * _lib * ".so")
                _h = Libdl.dlopen(_cand, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL; throw_error = false)
                _h === nothing || break
            end
            _h === nothing && @warn "extra_link_libs: no candidate soname opened. Harmless if this names a library merged into libc (m, pthread, dl, rt on glibc >= 2.34) — there is nothing left to preload. Otherwise the main library may fail to resolve; use a full soname or path." lib=_lib
        end
"""
end

"""
    _aot_fptr_const_name!(taken, mangled) -> String

Name the wrapper constant holding `mangled`'s resolved AOT thunk address.

Keyed on the MANGLED symbol, never on the Julia name, for the reason
`_slice_const_name!` records: `julia_name` is not injective over `mangled` (the
`replibuild_shim_` strip, the `_+` collapse and the trailing-`_` rstrip each
merge distinct symbols), so a Julia-keyed slot would let two functions share one
address and each call whichever thunk `__init__` happened to resolve last — a
wrong-function call with no error anywhere.

`taken` maps each issued constant to its owning symbol, so sanitizing a
character that is not legal in an identifier cannot reintroduce the collision.
"""
function _aot_fptr_const_name!(taken::Dict{String,String}, mangled::AbstractString)
    base = "_FPTR_" * replace(mangled, r"[^A-Za-z0-9_]" => "_")
    name = base
    n = 1
    while get(taken, name, mangled) != mangled
        n += 1
        name = "$(base)_$n"
    end
    taken[name] = mangled
    return name
end

"""
    _aot_thunk_slot_chunk(func_text, taken) -> String

Emit the thunk-address slots plus the `_THUNK_SLOTS` table `__init__` fills.

Derived from the FINAL wrapper text, never from `taken` alone — same discipline
as `_tier1_emit_slices!` and `_dispatch_facts`. `taken` records what the
generator INTENDED to emit, and dedup runs after it: a chunk dropped for
colliding on its dispatch signature would otherwise leave a slot that nothing
reads, and (worse) the reverse drift would be invisible. Scanning the text that
actually ships means the set of slots defined and the set referenced are one
derivation.

A `_FPTR_*` name in the text with no entry in `taken` is generator drift and
raises here, at wrap time, rather than becoming an `UndefVarError` on whichever
call the user makes first.

Always emits the table — empty if there are no thunks — because `__init__`
iterates it unconditionally, and a table that exists only sometimes is a second
thing to keep in sync.
"""
# _aot_thunk_slot_names(func_text, taken) -> Vector{String}
#
# The `_FPTR_*` slots the FINAL wrapper text actually reads, sorted and validated
# against `taken`. Split out so the emitted table and the wrap-time presence check
# (`_assert_aot_thunks_present`) share one derivation — two scans of the same text
# is exactly the drift this whole area exists to prevent.
function _aot_thunk_slot_names(func_text::AbstractString, taken::Dict{String,String})
    referenced = String[]
    seen = Set{String}()
    for m in eachmatch(r"\b(_FPTR_\w+)\b", func_text)
        nm = m.captures[1]
        nm in seen && continue
        push!(seen, nm)
        haskey(taken, nm) || error(
            "Wrapper text references AOT thunk slot `$nm`, which no emission " *
            "site registered. The slot namer and the emission site have drifted " *
            "— see `_aot_fptr_const_name!`.")
        push!(referenced, nm)
    end
    return sort!(referenced)
end

# _aot_thunk_symbol(taken, slot) -> String
#
# The dynamic symbol a slot resolves. One spelling, used by the emitted
# `_THUNK_SLOTS` table AND by the presence check — a second copy of this format
# string is a way for the wrapper to look for a name the check never verified.
_aot_thunk_symbol(taken::Dict{String,String}, slot::AbstractString) =
    "_mlir_ciface_$(taken[slot])_thunk"

function _aot_thunk_slot_chunk(func_text::AbstractString, taken::Dict{String,String})
    referenced = _aot_thunk_slot_names(func_text, taken)

    io = IOBuffer()
    print(io, """

    # =============================================================================
    # AOT thunk addresses
    #
    # One slot per thunk, resolved once in `__init__` and read directly at each
    # call site. These are `Ref`s rather than plain `const Ptr`s because a raw
    # pointer cannot be serialized into a precompile image — the address has to be
    # found in the process that is actually going to call it.
    # =============================================================================
    """)
    for nm in referenced
        println(io, "const $nm = Ref{Ptr{Cvoid}}(C_NULL)")
    end
    println(io, "\nconst _THUNK_SLOTS = Tuple{Base.RefValue{Ptr{Cvoid}},String}[")
    for nm in referenced
        println(io, "    ($nm, \"$(_aot_thunk_symbol(taken, nm))\"),")
    end
    println(io, "]")
    return String(take!(io))
end

"""
    _assert_aot_thunks_present(func_text, taken, thunks_lib_path)

Refuse to ship a wrapper that binds AOT thunks the thunks library does not
define.

**The thunk set is derived twice and the two derivations can disagree.** AOT runs
at BUILD, from `compilation_metadata.json` via `JLCSIRGenerator`; wrap runs later
and takes a thunk for every method `is_ccall_safe` rejects or `DAGDiff` marks.
A header-inline method that only became a symbol because some later TU emitted it
can appear in wrap's list and be absent from AOT's. Nothing reconciled them, so
the wrapper emitted `invoke_aot_ptr` sites whose slots resolve to `C_NULL`, and
the failure surfaced as a load-time warning plus a hard error on whichever call
the user made first — the silent-until-you-touch-it class.

This does not fix the double derivation; it makes the disagreement stop the wrap
instead of reaching a caller. Both facts are available here: the slots come from
the final text, and the thunks library exists by now (AOT ran at build, wrap runs
after), so it can simply be asked what it defines.

Reported by Clipper2 2.0.1 — 7 of 12 thunk sites missing, the set growing 1 → 7
when an export TU instantiated more header-inline methods
(`packages/clipper2/GENERATOR-aot-thunk-gap.md`).
"""
function _assert_aot_thunks_present(func_text::AbstractString,
                                    taken::Dict{String,String},
                                    thunks_lib_path::AbstractString)
    slots = _aot_thunk_slot_names(func_text, taken)
    isempty(slots) && return nothing

    if isempty(thunks_lib_path) || !isfile(thunks_lib_path)
        error("""
        Wrapper binds $(length(slots)) AOT thunk(s) but the thunks library is missing:
          $(isempty(thunks_lib_path) ? "<no path>" : thunks_lib_path)
        Every `invoke_aot_ptr` call site would raise. Rebuild with
        `[compile] aot_thunks = true`, or turn it off so the generator emits the
        JIT dispatch path instead.""")
    end

    (out, code) = BuildBridge.execute("nm", ["-D", "--defined-only", thunks_lib_path])
    code == 0 || error("""
    Cannot verify AOT thunks: `nm -D --defined-only $thunks_lib_path` exited $code.
    Refusing to emit a wrapper whose thunk bindings are unverified — an unresolved
    slot is a hard error at the call site, not a fallback.
    $out""")

    defined = Set{String}()
    for line in eachsplit(out, '\n')
        parts = split(strip(line))
        isempty(parts) || push!(defined, String(parts[end]))
    end

    missing_syms = [(nm, _aot_thunk_symbol(taken, nm))
                    for nm in slots if !(_aot_thunk_symbol(taken, nm) in defined)]
    isempty(missing_syms) && return nothing

    listed = join(("  $sym" for (_, sym) in missing_syms), "\n")
    error("""
    AOT thunks library is missing $(length(missing_syms)) of $(length(slots)) symbol(s)
    the wrapper binds:
    $listed

    Library: $thunks_lib_path

    The AOT pass (build) and the wrapper generator (wrap) derived different thunk
    sets. Wrap's set is the one the emitted call sites use, so these would resolve
    to C_NULL and raise on first call. Typical cause: header-inline or template
    methods that only became symbols in a TU compiled after AOT ran.

    Rebuild so AOT sees the same symbols, or set `[compile] aot_thunks = false`
    for this package to take the JIT dispatch path.""")
end

"""
    _dispatch_facts(func_chunks) -> (Dict{String,Symbol}, Dict{String,String})

Classify every emitted function by the tier it actually dispatches through, and
record the Tier-1 kernel each Tier-1 wrapper delegates to.

Read off the FINAL chunks (post-dedup, post-slice-emission), never off the
generator's own bookkeeping — the same discipline as `_tier1_emit_slices!` and
`_assert_cstring_policy`, and for the same reason: a table built from intent
agrees with the bug when intent and emission disagree.
"""
function _dispatch_facts(func_chunks)
    tier = Dict{String,Symbol}()
    kernels = Dict{String,String}()
    unclassified = String[]
    current = ""
    body = IOBuffer()

    finish! = function ()
        isempty(current) && return
        b = String(take!(body))
        # A Tier-1 wrapper delegates to its kernel; the kernel itself carries
        # the llvmcall. Either shape means Tier 1 for the function in hand.
        k = match(r"\b(_TIER1_\w+)\(", b)
        t = if k !== nothing || occursin("Base.llvmcall(", b)
            :tier1
        # THREE Tier-2 spellings, because AOT has TWO emission shapes and they
        # are not variants of each other:
        #   1. `JITManager.invoke(...)`                    — JIT, any function
        #   2. `JITManager.invoke_aot_ptr(_FPTR_x[], …)`   — AOT, ordinary fn
        #   3. `ccall((:thunk_<mangled>, THUNKS_LIBRARY_PATH), …)`
        #                                                  — AOT, VIRTUAL method
        #
        # Shape 2 was `invoke_aot(THUNKS_HANDLE[], "…")` until eager thunk
        # resolution moved the lookup into `__init__`. Both spellings are matched:
        # `invoke_aot` still exists and is still correct for a caller that has a
        # handle and a name, and a wrapper generated before the change is a file
        # on someone's disk, not a thing that gets migrated.
        # Shape 3 is `GeneratorCpp.jl`'s AOT virtual-dispatch branch and has
        # always been here; `THUNKS_LIBRARY_PATH` is its tell. Shape 2 arrived
        # with `invoke_aot` and this classifier was never taught it, so every
        # AOT-dispatched ORDINARY function matched nothing, fell past the ccall
        # test as well, and was omitted from the table ENTIRELY rather than
        # landing in it wrongly. On a library with no virtual methods what
        # survived was whatever genuinely ccalls — which then read as a UNIFORM
        # Tier-3 wrapper, and `_dispatch_tier_chunk` replaced the whole
        # introspection API with a sentence claiming Tier 3 for a module that
        # was 4/5 thunks. llamacpp shipped that sentence over 2280 call sites.
        #
        # ORDER IS LOAD-BEARING: shape 3 contains `ccall((:`, so this branch
        # must precede the Tier-3 test or every AOT virtual method is filed as
        # a plain ccall. Deleting the `THUNKS_LIBRARY_PATH` term while "fixing"
        # shape 2 does exactly that, silently — test_introspection's AOT
        # fixture is the thing that refuses it.
        elseif occursin("JITManager.invoke(", b) ||
               occursin("JITManager.invoke_aot(", b) ||
               occursin("JITManager.invoke_aot_ptr(", b) ||
               occursin("THUNKS_LIBRARY_PATH", b)
            :tier2
        elseif occursin("ccall((:", b) || occursin("@ccall ", b)
            :tier3
        else
            # A chunk that NAMES the dispatch machinery and matched no branch
            # above means this classifier has fallen behind emission. Recorded
            # and raised after the walk. Keyed on `JITManager.`/`llvmcall`
            # rather than on `ccall`, because an ordinary helper may ccall libc
            # (`memcpy`, `free`) without dispatching to the wrapped library —
            # those, and pure-Julia helpers, correctly classify as nothing and
            # must not be flagged.
            (occursin("JITManager.", b) || occursin("llvmcall", b)) &&
                push!(unclassified, current)
            nothing
        end
        if t !== nothing
            # One Julia name can carry methods on DIFFERENT tiers — a Tier-1
            # primary plus a Tier-3 convenience overload is the common shape (48
            # names across 5 Hub packages: cglm 35, miniaudio 5, llamacpp 3, lz4
            # 3, imgui 2). A last-write-wins Dict answers confidently and
            # WRONGLY for those, which is the failure this whole surface exists
            # to remove. Record the disagreement instead.
            prior = get(tier, current, nothing)
            if prior === nothing
                tier[current] = t
                t === :tier1 && k !== nothing && (kernels[current] = String(k.captures[1]))
            elseif prior !== t
                tier[current] = :mixed
                delete!(kernels, current)   # no single kernel can answer for it
            elseif t === :tier1 && k !== nothing && !haskey(kernels, current)
                kernels[current] = String(k.captures[1])
            end
        end
        current = ""
    end

    for chunk in func_chunks, line in eachsplit(String(chunk), '\n')
        m = match(r"^(?:@generated\s+)?function\s+([A-Za-z_][A-Za-z0-9_!]*)\s*\(", line)
        if m !== nothing
            finish!()
            nm = String(m.captures[1])
            # The kernels are an implementation detail the consumer should never
            # have to name — that reach-in is exactly what this replaces.
            current = startswith(nm, "_TIER1_") ? "" : nm
            continue
        end
        isempty(current) || print(body, line, '\n')
        line == "end" && finish!()
    end
    finish!()

    # REFUSE rather than ship a table that is silently short. An omitted
    # function does not merely lose its own row — it can make the remainder
    # look uniform, and a uniform table is emitted as a SENTENCE asserting the
    # tier. That is how a wrong answer here becomes a wrapper that describes
    # itself falsely, so the incomplete case has to be louder than the wrong
    # one.
    if !isempty(unclassified)
        names = join(sort(unique(unclassified)), ", ")
        error("""
              _dispatch_facts could not classify $(length(unique(unclassified))) emitted function(s) \
              that call the dispatch machinery: $names

              A dispatch shape exists that this classifier does not recognise. Add it to
              the tier test — do NOT let these fall out of `DISPATCH_TIER`, because an
              incomplete table reads as uniform and makes the wrapper state a tier it
              does not use.
              """)
    end

    return (tier, kernels)
end

"""
Prepended to the dispatch section of any wrapper that has Tier-2 call sites.

Answers "what is `invoke_aot`?" at the one place the reader is guaranteed to be
standing when they ask it — their own generated wrapper — rather than in a docs
page they reach only after deciding to trust the tool.
"""
const _THUNK_NOTE = """
# ── What a thunk is ───────────────────────────────────────────────────────
# A Tier-2 call goes through a *thunk*: the `extern "C"` shim you would
# hand-write to call this C++ function from C, except RepliBuild generated it
# and compiled it for you. `ccall` cannot express a virtual dispatch, a
# by-value struct return, or a call that may throw — the thunk puts the
# arguments where the C++ ABI wants them, and makes the call.
#
# Thunks are written in RepliBuild's MLIR dialect and compiled from there:
# ahead of time into a companion `_thunks.so` (`[compile] aot_thunks = true`),
# or JIT-compiled when this module loads. Either way the marshalling was
# compiled in rather than decided per call — the cost is one extra call frame
# and an argument array. gdb steps into the generated thunk by file and line.

"""

"""
    _dispatch_tier_chunk(tier, kernels) -> String

Emit `DISPATCH_TIER` (what was emitted) and `dispatch_tier(f)` (what actually
runs). Generated inline, because a wrapper cannot depend on RepliBuild at
runtime.
"""
function _dispatch_tier_chunk(tier::Dict{String,Symbol}, kernels::Dict{String,String})::String
    isempty(tier) && return ""
    entries = join(("    :$(k) => :$(tier[k])," for k in sort(collect(keys(tier)))), "\n")

    # A UNIFORM TABLE IS A CONSTANT — emit a sentence, not 100 rows.
    #
    # `dispatch_tier` exists to answer "which tier does this go through?". When
    # every wrapped function has the same answer the question is already
    # answered, and the table is one fact repeated once per function: cjson with
    # slicing off emitted 100 rows of `=> :tier3` plus a lookup that could
    # return only `:tier3` or `:unknown`.
    #
    # Keyed on UNIFORMITY rather than on language, though for C the two coincide:
    # the C generator has no Tier-2 dispatch path at all (0 `JITManager.invoke`
    # across all 14 C packages), so C with `[wrap.tier1] enable = false` is
    # necessarily all-Tier-3. Gating on `language == :c` would also strip C++
    # wrappers, where a tier2/tier3 mix is real information. Uniformity asks the
    # question directly and stays correct if either generator gains a tier.
    #
    # Consequence, deliberate: a wrapper in this state has no `DISPATCH_TIER` and
    # no `dispatch_tier`. Consumers asserting `dispatch_tier(:f) === :tier1` get
    # an UndefVarError rather than a confident `:tier3` — louder, and those
    # assertions are stale by construction once a package is unbolted.
    # `isempty(kernels)` is load-bearing and NOT redundant with uniformity: a
    # wrapper whose table is uniformly `:tier1` still needs the probe, because
    # Tier 1 is the one tier whose emitted intent can disagree with what runs —
    # a missing slice file, an unresolvable declare or output-mode generation all
    # demote it to ccall. Gating on uniformity alone deleted `dispatch_tier` from
    # exactly the wrapper that most needs it; caught by test_introspection's
    # single-function all-tier1 fixture.
    tiers = unique(values(tier))

    # The first thing a C++ user meets in their own wrapper is `invoke_aot` at
    # every call site — `is_ccall_safe` routes anything not marked `noexcept` to
    # Tier 2, so a C++ module is mostly thunk calls. This is where that question
    # gets asked, so this is where it is answered. Emitted only when the wrapper
    # HAS Tier-2 call sites; a pure-ccall wrapper never mentions thunks.
    note = (:tier2 in tiers || :mixed in tiers) ? _THUNK_NOTE : ""

    if isempty(kernels) && length(tiers) == 1
        only_tier = first(tiers)
        return """
        $note# ── Dispatch ──────────────────────────────────────────────────────────────
        # Every function in this module dispatches through $(only_tier === :tier3 ?
            "Tier 3 (`ccall` straight into the library)" :
            (only_tier === :tier2 ? "Tier 2 (MLIR thunk)" : "Tier 1 (sliced `llvmcall`)")).
        # No `DISPATCH_TIER` table and no `dispatch_tier` probe are emitted: with a
        # single possible answer there is nothing to look up. A mixed-tier wrapper
        # emits both.

        """
    end

    # NO TIER-1 KERNELS ⇒ NO TIER-1 MACHINERY. Everything below the table exists
    # to answer "did this kernel's @generated body actually splice llvmcall, or
    # did it demote?" — a question with no referent when no kernel was emitted.
    # Without this branch a `[wrap.tier1] enable = false` wrapper still shipped an
    # empty `_TIER1_KERNEL`, a `code_typed` probe that could never fire, and a
    # docstring about bitcode slices.
    #
    # The emitted table is the whole truth here, so `dispatch_tier` is a lookup.
    # It also drops the output-mode `:deferred` guard, deliberately: that guard
    # exists because probing a Tier-1 kernel FORCES generation and would freeze a
    # worker's answer into the pkgimage. With nothing to probe the answer is
    # static, so `const T = dispatch_tier(:f)` at module scope is now safe and
    # correct — a behavioural difference between tier1-on and tier1-off wrappers,
    # and the tier1-off one is the stronger guarantee.
    if isempty(kernels)
        return """
        $note# ── Dispatch introspection ────────────────────────────────────────────────
        # What the generator emitted for each function. Tier 2 is an MLIR thunk,
        # Tier 3 a plain ccall. This wrapper has no Tier-1 (sliced) call sites, so
        # the table is exhaustive and needs no runtime probe.
        const DISPATCH_TIER = Dict{Symbol,Symbol}(
        $entries
        )

        \"\"\"
            dispatch_tier(f) -> Symbol

        Which tier `f` dispatches through: `:tier2` (MLIR thunk), `:tier3`
        (`ccall`), or `:unknown` for a name this module did not wrap. Accepts the
        function or its `Symbol`.

        This wrapper was generated with per-function bitcode slicing disabled, so
        no call site can demote at runtime and this is a plain lookup — unlike a
        Tier-1-enabled wrapper, where the answer must be probed and is refused
        during precompilation.
        \"\"\"
        function dispatch_tier(f)
            # Every Base name qualified: this module's namespace belongs to the
            # LIBRARY, which is free to export `get`, `nameof`, `string`…
            name = f isa Symbol ? f : Base.nameof(f)
            return Base.get(DISPATCH_TIER, name, :unknown)
        end

        """
    end

    kentries = join(("    :$(k) => :$(kernels[k])," for k in sort(collect(keys(kernels)))), "\n")

    return """
    # ── Dispatch introspection ────────────────────────────────────────────────
    # What the generator EMITTED for each function. Tier 1 is `Base.llvmcall` on
    # a per-function bitcode slice, Tier 2 an MLIR thunk, Tier 3 a plain ccall.
    const DISPATCH_TIER = Dict{Symbol,Symbol}(
    $entries
    )

    # Tier-1 wrappers delegate to these `@generated` kernels. Private: ask
    # `dispatch_tier` instead of naming one.
    const _TIER1_KERNEL = Dict{Symbol,Symbol}(
    $kentries
    )

    \"\"\"
        dispatch_tier(f) -> Symbol

    Which tier `f` **actually** dispatches through right now: `:tier1`
    (`Base.llvmcall` on a bitcode slice), `:tier2` (MLIR thunk), `:tier3`
    (`ccall`), `:unknown` for a name this module did not wrap, or `:mixed` when
    this name carries methods on more than one tier — typically a Tier-1 primary
    plus a Tier-3 convenience overload, where no single answer is true. Ask
    about a name that resolves to one method, or read the emitted source.

    Differs from `DISPATCH_TIER[nameof(f)]`, which is what the generator emitted.
    The two disagree exactly when a Tier-1 kernel demotes — a missing `slices/`
    directory, an unresolvable declare, or generation inside a precompile worker
    — and the demoted answer is the honest one, because that is the code that
    will run. Accepts the function or its `Symbol`.

    Answering for a Tier-1 function forces its kernel to generate — that is what
    makes the answer real rather than declared, and it is also why this is **not
    a read-only call**.

    Returns `:deferred` during precompilation and refuses to probe. A Tier-1
    kernel generating inside a precompile worker deliberately splices its `ccall`
    body, so an answer taken there would describe the worker and not the session
    that runs: `const T = dispatch_tier(:f)` at module scope would freeze
    `:tier3` into the pkgimage for a function that re-generates to `llvmcall` on
    load. **Ask at runtime**, in the session whose behaviour you care about.
    \"\"\"
    function dispatch_tier(f)
        # OUTPUT MODE: refuse, do not probe. Two reasons, both measured.
        # (1) The answer would be frozen and WRONG — a Tier-1 kernel generating
        #     inside a precompile worker deliberately splices its ccall body, so
        #     `const T = dispatch_tier(:f)` at module scope bakes :tier3 into the
        #     pkgimage while the same function re-generates to llvmcall in the
        #     next session. Verified with an observer/control package pair.
        # (2) Probing FORCES that generation. This function looks read-only and
        #     is not; making it inert here keeps an introspection call from
        #     having any effect on the code that ships.
        # `ccall` is syntax, not a binding — it cannot be shadowed and must not
        # be qualified. Same spelling the Tier-1 kernels already use.
        ccall(:jl_generating_output, Cint, ()) == 1 && return :deferred
        # Every Base name here is qualified: this module's namespace belongs to
        # the LIBRARY, which is free to export `get`, `string`, `methods`… (see
        # _assert_base_calls_qualified — llama.cpp already takes `error`, `all`,
        # `stat` and `symlink`).
        name = f isa Symbol ? f : Base.nameof(f)
        t = Base.get(DISPATCH_TIER, name, :unknown)
        t === :tier1 || return t
        kern = Base.get(_TIER1_KERNEL, name, nothing)
        kern === nothing && return :tier1
        fn = Base.getfield(@__MODULE__, kern)
        ms = Base.collect(Base.methods(fn))
        Base.isempty(ms) && return :tier1
        ct = try
            Base.code_typed(fn, Base.tuple_type_tail(ms[1].sig))
        catch
            return :tier1
        end
        return (!Base.isempty(ct) && Base.occursin("llvmcall", Base.string(ct))) ? :tier1 : :tier3
    end

    """
end

"""
    _layout_chunk(dwarf_structs, emitted_names, sanitize) -> String

Emit the byte sizes and member offsets the generator read from DWARF, for the
types this module actually declares.

Scoped to `emitted_names` on purpose: a struct the wrapper never declared is not
something a caller can name, and llama.cpp's metadata carries 2864 of them
(mostly libstdc++ internals).

**`sanitize` must be the caller's OWN type-name sanitizer** — the one that
produced the emitted struct names — not a re-implementation. The first version
of this re-derived the spelling with a generic `[^A-Za-z0-9_] => "_"`, which
differs from both generators on any templated or scoped name: they collapse
`_+` and trim, so `ImChunkStream<ImGuiTableSettings>` emits as
`ImChunkStream_ImGuiTableSettings` while the generic rule yields a trailing
underscore. The mismatch failed the `emitted_names` gate and dropped the type
SILENTLY — 100 of imgui's 262 declared structs, 29 of tinyxml2's 42, i.e.
precisely the templated types whose size a caller cannot compute any other way.
"""
function _layout_chunk(dwarf_structs, emitted_names, sanitize)::String
    _num(v) = v isa Integer ? Int(v) :
        (let s = string(v); startswith(s, "0x") ? parse(Int, s[3:end], base=16) : parse(Int, s) end)

    sizes = String[]
    offsets = String[]
    for key in sort(collect(keys(dwarf_structs)))
        info = dwarf_structs[key]
        info isa AbstractDict || continue
        name = String(sanitize(String(key)))
        name in emitted_names || continue
        bs = try _num(get(info, "byte_size", 0)) catch; 0 end
        bs > 0 || continue
        push!(sizes, "    :$(name) => $(bs),")

        members = String[]
        for m in get(info, "members", [])
            m isa AbstractDict || continue
            raw = String(get(m, "name", ""))
            isempty(raw) && continue
            # Member names come from DWARF and are NOT all Julia identifiers:
            # a polymorphic class carries a synthesized `_vptr$Class`, and `$`
            # in a `const` Dict literal is interpolation — `UndefVarError: $`
            # at module load, taking the whole wrapper with it. (The existing
            # `getproperty` branches emit the same raw name, but inside a
            # function body, so it only bites the caller who asks for that
            # field. This table is evaluated eagerly, so it must be clean.)
            mn = _sanitize_type_name_for_layout(raw)
            (isempty(mn) || !(isletter(mn[1]) || mn[1] == '_')) && continue
            off = try _num(get(m, "offset", -1)) catch; -1 end
            off < 0 && continue
            push!(members, ":$(mn) => $(off)")
        end
        isempty(members) || push!(offsets, "    :$(name) => Dict{Symbol,Int}($(join(members, ", "))),")
    end
    isempty(sizes) && return ""

    return """
    # ── Layout facts (from DWARF, as emitted) ────────────────────────────────
    # The sizes and offsets this module was generated from. Exposed because
    # callers need them for the cases the type system cannot cover — stack
    # space for an in-place constructor, and reading a member through an opaque
    # pointer the API hands back as `Ptr{Cvoid}`. Previously every consumer
    # re-parsed `compilation_metadata.json` to recover them.
    const STRUCT_SIZES = Dict{Symbol,Int}(
    $(join(sizes, "\n"))
    )

    const STRUCT_OFFSETS = Dict{Symbol,Dict{Symbol,Int}}(
    $(join(offsets, "\n"))
    )

    \"\"\"
        struct_size(name) -> Int

    Byte size of a wrapped struct, as DWARF reported it. Throws for an unknown
    name rather than returning 0, which would silently under-allocate.
    \"\"\"
    function struct_size(name)
        s = Base.Symbol(name)
        Base.haskey(STRUCT_SIZES, s) || Base.error(Base.string(
            "struct_size: no layout for ", s, ". ",
            Base.length(STRUCT_SIZES), " struct(s) known; see STRUCT_SIZES."))
        return STRUCT_SIZES[s]
    end

    \"\"\"
        member_offset(name, member) -> Int

    Byte offset of `member` within a wrapped struct. For structs with named
    Julia fields prefer `getproperty`; this is for reading through a raw pointer.
    \"\"\"
    function member_offset(name, member)
        s = Base.Symbol(name); m = Base.Symbol(member)
        Base.haskey(STRUCT_OFFSETS, s) || Base.error(Base.string(
            "member_offset: no layout for ", s, "."))
        ms = STRUCT_OFFSETS[s]
        Base.haskey(ms, m) || Base.error(Base.string(
            "member_offset: ", s, " has no member ", m, ". Members: ",
            Base.join(Base.sort(Base.string.(Base.keys(ms))), ", ")))
        return ms[m]
    end

    """
end

# Layout keys come from DWARF and carry C++ spellings; the emitted Julia type
# name is what a caller can actually name, so the table is keyed on that.
_sanitize_type_name_for_layout(s::AbstractString) =
    replace(String(s), r"[^A-Za-z0-9_]" => "_")

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
    _split_toplevel_commas(s) -> Vector{SubString}

Split on commas at bracket depth 0. A parameter's TYPE may contain commas —
`Union{AbstractString,Cstring}`, `NTuple{8,UInt8}`, `Dict{Symbol,Int}` — and a
flat `split(s, ',')` turns one argument into two, silently changing the
dispatch key this module keys dedup on. Depth is tracked over `{}`, `()` and
`[]` and clamped at 0 so an unbalanced closer can't drive it negative.
"""
function _split_toplevel_commas(s::AbstractString)
    parts = SubString{String}[]
    str = String(s)
    depth = 0
    start = 1
    for (i, c) in pairs(str)
        if c in ('{', '(', '[')
            depth += 1
        elseif c in ('}', ')', ']')
            depth = max(0, depth - 1)
        elseif c == ',' && depth == 0
            push!(parts, SubString(str, start, prevind(str, i)))
            start = nextind(str, i)
        end
    end
    push!(parts, SubString(str, start))
    return parts
end

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
            for a in _split_toplevel_commas(args)
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
    # Which chunk claimed each signature, so a drop can name what shadowed it.
    claimed_by = Dict{String,String}()
    dropped = Tuple{String,String,String}[]   # (signature, dropped symbol, kept symbol)
    for i in length(chunks):-1:1
        ks = _method_sig_keys(chunks[i])
        isempty(ks) && continue
        if all(k -> k in seen, ks)
            keep[i] = false
            sym = _chunk_mangled_symbol(chunks[i])
            for k in ks
                push!(dropped, (k, sym, get(claimed_by, k, "?")))
            end
        else
            union!(seen, ks)
            sym = _chunk_mangled_symbol(chunks[i])
            for k in ks
                get!(claimed_by, k, sym)
            end
        end
    end
    if !isempty(dropped)
        # Naming the losers is the point. The count alone cannot distinguish a
        # D1/D2 destructor pair — where dropping one is exactly right — from an
        # ::Any-collapsed overload pair, where a DISTINCT C++ entry point became
        # unreachable and only the symbol names show it (imgui's
        # `TreeNode(const char*, const char*, ...)` losing to the `void const*`
        # form: same `(Any, Any)` signature, different function).
        shown = first(dropped, 12)
        # Same symbol on both sides means one C++ entry point emitted two
        # chunks (the D1/D2 destructor pair aliasing to one definition) — a
        # correct drop, and naming the shadower twice would only read as noise.
        detail = join([lost == kept ? "$sig ⟵ $lost" : "$sig ⟵ $lost (shadowed by $kept)"
                       for (sig, lost, kept) in shown], ", ")
        more = length(dropped) > 12 ? " … (+$(length(dropped) - 12) more)" : ""
        @info "wrap: dropped $(length(dropped)) duplicate method definition(s) — identical Julia name+signature from distinct C++ symbols; last definition kept (precompilation-safe). Unreachable now: $detail$more"
    end
    return chunks[keep]
end

# The mangled symbol a chunk was generated from, as recorded in its own
# docstring ("- Mangled symbol: `_ZN…`"). Read back off the emitted text rather
# than threaded through, for the same reason the export lists and layout tables
# are: the text is what ships, so it cannot disagree with itself.
function _chunk_mangled_symbol(chunk::AbstractString)
    m = match(r"Mangled symbol: `([^`]+)`", chunk)
    m !== nothing && return String(m.captures[1])
    # Varargs chunks document the demangled prototype instead, so fall back to
    # the symbol their own @ccall names — which is the thing that would go
    # unreachable, and the whole reason to print a symbol at all.
    m = match(r"LIBRARY_PATH\.var\"([^\"]+)\"", chunk)
    m !== nothing && return String(m.captures[1])
    m = match(r"ccall\(\(:([A-Za-z_][A-Za-z0-9_]*),\s*LIBRARY_PATH\)", chunk)
    m !== nothing && return String(m.captures[1])
    return "<unknown symbol>"
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
