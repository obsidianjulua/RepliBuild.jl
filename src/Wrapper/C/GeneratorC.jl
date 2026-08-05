# Julia-side (size, alignment) of primitive field types, used to reproduce C
# struct layouts exactly. Julia lays out isbits struct fields with natural
# alignment — identical to the C rules for these types on x86_64 SysV.
const _C_PRIM_FIELD_LAYOUT = Dict{String,Tuple{Int,Int}}(
    "Cchar" => (1, 1), "Cuchar" => (1, 1), "Int8" => (1, 1), "UInt8" => (1, 1), "Bool" => (1, 1),
    "Cshort" => (2, 2), "Cushort" => (2, 2), "Int16" => (2, 2), "UInt16" => (2, 2),
    "Cint" => (4, 4), "Cuint" => (4, 4), "Int32" => (4, 4), "UInt32" => (4, 4),
    "Cfloat" => (4, 4), "Float32" => (4, 4), "Cwchar_t" => (4, 4),
    "Clong" => (8, 8), "Culong" => (8, 8), "Clonglong" => (8, 8), "Culonglong" => (8, 8),
    "Int64" => (8, 8), "UInt64" => (8, 8), "Cdouble" => (8, 8), "Float64" => (8, 8),
    "Csize_t" => (8, 8), "Cssize_t" => (8, 8), "Cptrdiff_t" => (8, 8),
    "Cintptr_t" => (8, 8), "Cuintptr_t" => (8, 8), "Cstring" => (8, 8),
)

_align_up(x::Int, a::Int) = a <= 1 ? x : (x + a - 1) & ~(a - 1)

# Julia field type + (size, align) for one struct member, or `nothing` when the
# member cannot be typed with exactly known layout. Struct-typed members are
# only accepted once the member's own struct has been emitted with verified
# named fields (`resolved_layouts`, topological emission order guarantees the
# member type is processed first).
function _field_layout(jt::String, ct::String,
                       resolved_layouts::Dict{String,Tuple{Int,Int}},
                       enum_layouts::Dict{String,Tuple{Int,Int}})
    # Pointers (incl. function pointers): 8/8 regardless of pointee
    if endswith(ct, "*") || endswith(ct, "&")
        return (startswith(jt, "Ptr{") || jt == "Cstring") ? (jt, 8, 8) : ("Ptr{Cvoid}", 8, 8)
    end
    startswith(jt, "Ptr{") && return (jt, 8, 8)
    if haskey(_C_PRIM_FIELD_LAYOUT, jt)
        sz, al = _C_PRIM_FIELD_LAYOUT[jt]
        return (jt, sz, al)
    end
    nt = match(r"^NTuple\{(\d+),\s*(.+)\}$", jt)
    if nt !== nothing
        n = parse(Int, nt.captures[1])
        elem = String(strip(nt.captures[2]))
        if haskey(_C_PRIM_FIELD_LAYOUT, elem)
            esz, eal = _C_PRIM_FIELD_LAYOUT[elem]
            return (jt, n * esz, eal)
        end
        es = _sanitize_c_type_name(elem)
        if haskey(resolved_layouts, es)
            esz, eal = resolved_layouts[es]
            return ("NTuple{$n, $es}", n * esz, eal)
        end
        return nothing
    end
    base = _sanitize_c_type_name(jt == "Any" ? ct : jt)
    if haskey(enum_layouts, base)
        esz, eal = enum_layouts[base]
        return (base, esz, eal)
    end
    if haskey(resolved_layouts, base)
        ssz, sal = resolved_layouts[base]
        return (base, ssz, sal)
    end
    return nothing
end

"""
    _resolve_exact_layout(members, byte_size, resolved_layouts, enum_layouts)

Type every struct member with a Julia field type of exactly known size and
alignment, then prove that the emitter's layout (explicit align-1 `_pad_N`
fields filling DWARF offset gaps + Julia natural field alignment) reproduces
every DWARF member offset and the DWARF total size. On success returns
`(members′, max_align)` with `julia_type`/`size` rewritten; on any doubt
returns `nothing` and the caller keeps the opaque byte blob — exact or
opaque, never approximate.
"""
function _resolve_exact_layout(members::Vector, byte_size::Int,
                               resolved_layouts::Dict{String,Tuple{Int,Int}},
                               enum_layouts::Dict{String,Tuple{Int,Int}})
    byte_size <= 0 && return nothing
    isempty(members) && return nothing
    plans = Tuple{Any,String,Int,Int,Int}[]   # (member, ftype, fsize, falign, offset)
    for m in members
        haskey(m, "bit_size") && return nothing   # bitfields use the accessor path
        off_raw = get(m, "offset", nothing)
        off_raw === nothing && return nothing
        off = _parse_int_or_hex(off_raw)
        jt = String(get(m, "julia_type", "Any"))
        ct = String(strip(replace(String(get(m, "c_type", "")), r"\bconst\b" => "")))
        lay = _field_layout(jt, ct, resolved_layouts, enum_layouts)
        lay === nothing && return nothing
        push!(plans, (m, lay[1], lay[2], lay[3], off))
    end
    sort!(plans, by = p -> p[5])
    cur = 0
    maxal = 1
    for (_, _, fsize, falign, off) in plans
        off < cur && return nothing               # overlapping members (union-like)
        cur = max(cur, off)                       # explicit pad bytes (align 1)
        _align_up(cur, falign) == off || return nothing
        cur = off + fsize
        maxal = max(maxal, falign)
    end
    cur > byte_size && return nothing
    # Trailing pad bytes are align-1; Julia then rounds the struct size to maxal,
    # which must land exactly on the DWARF size.
    _align_up(byte_size, maxal) == byte_size || return nothing
    members′ = Vector{Any}(undef, length(plans))
    for (i, (m, ftype, fsize, _, _)) in enumerate(plans)
        m2 = copy(m)
        m2["julia_type"] = ftype
        m2["size"] = fsize
        members′[i] = m2
    end
    return (members′, maxal)
end

"""
    _dwarf_member_align(jt, ct, dwarf_structs, best_dwarf_key, enum_layouts, seen) -> Int | nothing
    _dwarf_aggregate_align(jname, dwarf_structs, best_dwarf_key, enum_layouts, seen) -> Int | nothing

Alignment of a DWARF-described type, derived structurally from its members.

`_field_layout` can only answer for types whose Julia field type already exists
— primitives, pointers, enums, and aggregates emitted with verified layout. An
anonymous union needs its alignment BEFORE any of that: the alignment is what
picks the storage type of the opaque region that stands in for it, and its
members may themselves be unemitted anonymous aggregates or bitfields.

Aggregates take the max over their members; a bitfield contributes the
alignment of its declared base type. `nothing` on any unknown leaf or on a
by-value cycle — an alignment that cannot be proven is never guessed.
"""
function _dwarf_member_align(jt::String, ct::String, dwarf_structs,
                             best_dwarf_key::Dict{String,String},
                             enum_layouts::Dict{String,Tuple{Int,Int}},
                             seen::Set{String}=Set{String}())
    ctc = String(strip(replace(ct, r"\bconst\b" => "")))
    (endswith(ctc, "*") || endswith(ctc, "&") || startswith(jt, "Ptr{") || jt == "Cstring") && return 8
    haskey(_C_PRIM_FIELD_LAYOUT, jt) && return _C_PRIM_FIELD_LAYOUT[jt][2]
    nt = match(r"^NTuple\{(\d+),\s*(.+)\}$", jt)
    if nt !== nothing
        elem = String(strip(nt.captures[2]))
        haskey(_C_PRIM_FIELD_LAYOUT, elem) && return _C_PRIM_FIELD_LAYOUT[elem][2]
        return _dwarf_aggregate_align(_sanitize_c_type_name(elem), dwarf_structs,
                                      best_dwarf_key, enum_layouts, seen)
    end
    base = _sanitize_c_type_name(jt == "Any" ? ctc : jt)
    haskey(enum_layouts, base) && return enum_layouts[base][2]
    return _dwarf_aggregate_align(base, dwarf_structs, best_dwarf_key, enum_layouts, seen)
end

# DWARF entry for an aggregate addressed by its sanitized Julia name.
function _lookup_dwarf_agg(jname::String, dwarf_structs, best_dwarf_key::Dict{String,String})
    isempty(jname) && return nothing
    key = get(best_dwarf_key, jname, nothing)
    if key !== nothing && haskey(dwarf_structs, key)
        return dwarf_structs[key]
    elseif haskey(dwarf_structs, jname)
        return dwarf_structs[jname]
    end
    for (sk, sv) in dwarf_structs
        if _sanitize_c_type_name(String(sk)) == jname
            return sv
        end
    end
    return nothing
end

"""
    _exposed_member_names(jname, dwarf_structs, best_dwarf_key, seen) -> Vector{String}

Names an aggregate exposes through `getproperty`: its own members, plus — for
each C11 anonymous member — the names that member's own type exposes, because C
injects an anonymous member's names into the enclosing scope (`n.x`, not
`n._anon1.x`). Recursion terminates on the by-value cycle set; each level only
needs its immediate injections, since the level below emits its own.
"""
function _exposed_member_names(jname::String, dwarf_structs,
                               best_dwarf_key::Dict{String,String},
                               seen::Set{String}=Set{String}())
    out = String[]
    (isempty(jname) || jname in seen) && return out
    info = _lookup_dwarf_agg(jname, dwarf_structs, best_dwarf_key)
    info === nothing && return out
    push!(seen, jname)
    for m in get(info, "members", [])
        nm = get(m, "name", nothing)
        nm === nothing && continue
        # Bitfields are reached through the generated `get_<name>(s)` accessors,
        # not through `getproperty` — injecting them would emit a branch that
        # delegates to a property the target type does not have.
        haskey(m, "bit_size") && continue
        if get(m, "anonymous_member", false) == true
            # `_anonN` is a name WE invented to give the region a field; C never
            # exposes it, and injecting it upward would shadow the enclosing
            # struct's own `_anonN` field. Only what it contains travels up.
            inner = _sanitize_c_type_name(String(get(m, "julia_type", "")))
            isempty(inner) && (inner = _sanitize_c_type_name(String(get(m, "c_type", ""))))
            append!(out, _exposed_member_names(inner, dwarf_structs, best_dwarf_key, seen))
        else
            push!(out, String(nm))
        end
    end
    delete!(seen, jname)
    return out
end

function _dwarf_aggregate_align(jname::String, dwarf_structs,
                                best_dwarf_key::Dict{String,String},
                                enum_layouts::Dict{String,Tuple{Int,Int}},
                                seen::Set{String}=Set{String}())
    isempty(jname) && return nothing
    jname in seen && return nothing          # a type cannot contain itself by value
    info = _lookup_dwarf_agg(jname, dwarf_structs, best_dwarf_key)
    info === nothing && return nothing
    ms = get(info, "members", [])
    isempty(ms) && return nothing
    push!(seen, jname)
    al = 1
    for m in ms
        a = _dwarf_member_align(String(get(m, "julia_type", "Any")),
                                String(get(m, "c_type", "")),
                                dwarf_structs, best_dwarf_key, enum_layouts, seen)
        if a === nothing
            delete!(seen, jname)
            return nothing
        end
        al = max(al, a)
    end
    delete!(seen, jname)
    return al
end

"""
    _aligned_region_storage(byte_size, align; all_float=false) -> (elem_type, count) | nothing

Julia storage for a `byte_size`-byte opaque region that must carry a C
alignment of `align`.

`NTuple{N, UInt8}` is align 1 — embedding one where C wants align 8 silently
UNDER-ALIGNS the enclosing struct, and Julia then lays the following fields out
at the wrong offsets. So the element type is chosen to CARRY the alignment and
the count follows from the size. Refuses (rather than under-aligning) when the
size is not a whole number of elements, or when the alignment exceeds what a
Julia primitive can express.

`all_float` picks a floating-point element instead. SysV classifies an eightbyte
as SSE only when EVERY field overlapping it is float/double, so an integer
element is the correct class for any aggregate with a non-float member — but for
an all-float one (`union { float f; double d; }`) it would claim INTEGER where
the value travels in XMM.
"""
function _aligned_region_storage(byte_size::Int, align::Int; all_float::Bool=false)
    (byte_size <= 0 || align <= 0 || align > 8) && return nothing
    elem, esz = if all_float && align >= 8
        ("Float64", 8)
    elseif all_float && align >= 4
        ("Float32", 4)
    elseif align >= 8
        ("UInt64", 8)
    elseif align >= 4
        ("UInt32", 4)
    elseif align >= 2
        ("UInt16", 2)
    else
        ("UInt8", 1)
    end
    byte_size % esz == 0 || return nothing
    return (elem, byte_size ÷ esz)
end

# Is EVERY (transitive) member of this struct a float/double? The dual of
# `_struct_has_float_member`: that one asks "could this be SSE-class", this one
# asks "must it be". Unknown types answer false — an unprovable claim of
# uniform float class is worse than falling back to integer storage.
function _struct_all_float_members(struct_name::String, dwarf_structs, seen::Set{String}=Set{String}())
    struct_name in seen && return false
    push!(seen, struct_name)
    info = get(dwarf_structs, struct_name, nothing)
    info === nothing && return false
    ms = get(info, "members", [])
    isempty(ms) && return false
    for m in ms
        haskey(m, "bit_size") && return false
        ct = String(strip(get(m, "c_type", "")))
        (endswith(ct, "*") || endswith(ct, "&")) && return false
        base = String(strip(replace(replace(ct, r"\[\d*\]" => ""), r"\bconst\b" => "")))
        if base == "float" || base == "double"
            continue
        elseif haskey(dwarf_structs, base)
            _struct_all_float_members(base, dwarf_structs, seen) || return false
        else
            return false
        end
    end
    return true
end

# Does this struct (transitively) contain float/double members? Used to judge
# whether an opaque ≤16-byte blob is in the SysV SSE-class danger window.
# Unknown types count as risky.
function _struct_has_float_member(struct_name::String, dwarf_structs, seen::Set{String}=Set{String}())
    struct_name in seen && return false
    push!(seen, struct_name)
    info = get(dwarf_structs, struct_name, nothing)
    info === nothing && return true
    for m in get(info, "members", [])
        ct = String(strip(get(m, "c_type", "")))
        endswith(ct, "*") && continue
        endswith(ct, "&") && continue
        occursin("float", ct) && return true
        occursin("double", ct) && return true
        base = String(strip(replace(replace(ct, r"\[\d*\]" => ""), r"\bconst\b" => "")))
        if haskey(dwarf_structs, base) && _struct_has_float_member(base, dwarf_structs, seen)
            return true
        end
    end
    return false
end

"""
    _symbol_resolves_via(handle, sym) -> Bool

Whether `dlsym` finds `sym` through `handle`. A `C_NULL` handle is
`RTLD_DEFAULT` — the whole process — which is what ORC searches when it links
a slice at `llvmcall` time. A real handle scopes the lookup to that library
and its `DT_NEEDED` chain.
"""
_symbol_resolves_via(handle::Ptr{Cvoid}, sym::AbstractString) =
    ccall(:dlsym, Ptr{Cvoid}, (Ptr{Cvoid}, Cstring), handle, sym) != C_NULL

"""
    _slice_path_expr(mangled) -> String

The Julia source expression a generated wrapper uses to locate `mangled`'s slice.
Emitted twice per Tier-1 function — once for the `read` that feeds `llvmcall`,
once for the `include_dependency` that makes the file a precompile dependency —
so both must be built from here rather than written out twice.
"""
_slice_path_expr(mangled::AbstractString) =
    "joinpath(@__DIR__, \"slices\", \"$mangled.ll\")"

"""
    _slice_const_name!(taken, mangled) -> String

Name the wrapper constant that holds the path to `mangled`'s slice.

Keyed on the MANGLED symbol, never on the Julia name. `julia_name` is not
injective over `mangled` — the `replibuild_shim_` strip, the `_+` collapse and
the trailing-`_` rstrip each merge distinct symbols — so a Julia-keyed const
silently rebinds one function to another function's slice module. `Base.llvmcall`
resolves the const at codegen, i.e. on the FIRST CALL, so the loser fails long
after wrap time: `luaL_checkversion_` and `replibuild_shim_luaL_checkversion`
both produced `_SLICE_luaL_checkversion`, and the three-argument method died
with "Module IR does not contain specified entry function".

`taken` maps each issued constant to the symbol that owns it, so the sanitizer
(a mangled name may carry characters that are not legal in an identifier)
cannot reintroduce the very collision this exists to remove.
"""
function _slice_const_name!(taken::Dict{String,String}, mangled::AbstractString)
    base = "_SLICE_" * replace(mangled, r"[^A-Za-z0-9_]" => "_")
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
    _tier1_kernel_name(slice_const) -> String

Kernel function name for a Tier-1 call site, derived from the slice-path
constant so it inherits `_slice_const_name!`'s mangled-symbol keying and
collision suffixes — two symbols can no more share a kernel than a const.
"""
_tier1_kernel_name(slice_const::AbstractString) =
    replace(slice_const, r"^_SLICE_" => "_TIER1_")

"""
    _tier1_kernel_chunk(slice_const, kernel, mangled, ret_type, param_types, arg_names) -> String

Emit the per-function Tier-1 kernel: a `@generated` function that decides
ccall vs llvmcall ONCE, at generation time. llvmcall is opportunistic — a
passenger tier, never the driver — so every doubt resolves to the ccall body:

  - **output mode** (`jl_generating_output`): emitting a sliced llvmcall inside
    a precompile worker deadlocks the JIT engine lock whenever a `declare`
    binds a dlopened library's symbol (2026-07-31; an untaken top-level branch
    reaches it through inference alone). The ccall body precompiles clean, and
    a runtime first call regenerates to the slice.
  - **missing slice file**: a wrapper vendored or relocated without `slices/`
    stays a working ccall wrapper instead of failing at load or call time.

The slice is read at GENERATION time (first call), so module load does no
slice I/O, uncalled functions keep nothing resident, and the `.ji` no longer
stores a second copy of IR that already ships as `.ll` files. The
`include_dependency` (content-tracked on 1.11+) still invalidates the cache
when a present slice changes; it is skipped when the file is absent so a
slice-less wrapper can still precompile.
"""
function _tier1_kernel_chunk(slice_const::String, kernel::String, mangled::String,
                             ret_type::String, param_types::Vector{String},
                             arg_names::Vector{String})
    sig = join(("$n::$t" for (n, t) in zip(arg_names, param_types)), ", ")
    tuple_types = join(param_types, ", ")
    ccall_types = isempty(param_types) ? "()" : "(" * tuple_types * ",)"
    arg_tail = isempty(arg_names) ? "" : ", " * join(arg_names, ", ")
    return """
    const $slice_const = $(_slice_path_expr(mangled))
    isfile($slice_const) && include_dependency($slice_const)

    @generated function $kernel($sig)
        if ccall(:jl_generating_output, Cint, ()) == 1 || !isfile($slice_const) ||
           !_slice_symbols_resolve(get(TIER1_DECLARES, "$mangled", String[]))
            return :(ccall((:$mangled, LIBRARY_PATH), $ret_type, $ccall_types$arg_tail))
        end
        ir = read($slice_const, String)
        return :(Base.llvmcall((\$ir, "$mangled"), $ret_type, Tuple{$tuple_types}$arg_tail))
    end
    """
end

"""
    _tier1_preflight!(accepted, results, lib_path) -> Dict{String,Vector{String}}

Pre-flight every accepted slice's binding contract: each name the slice binds
by `declare` must be supplied by the `.so` itself or its `DT_NEEDED` chain.
Functions with a miss are deleted from `accepted` and returned in the
`function => missing symbols` map.

This exists because an unresolved slice declaration does NOT surface as a
catchable error: ORC blocks on the pending symbol and the JIT deadlocks on the
first call ("Symbols not found"), with no stack to read. Checking the same
lookup up-front converts that whole class into a clean Tier-3 fallback plus a
warning naming the symbol — the discipline the macro-shim collision guard uses.

A `.so` that cannot be loaded at all leaves every slice unverified, so Tier 1
is disabled wholesale rather than shipped on faith.
"""
function _tier1_preflight!(accepted::Set{String}, results, lib_path::String)
    unresolved = Dict{String,Vector{String}}()
    strays = Dict{String,Vector{String}}()

    # RTLD_LOCAL, and closed again below. The check resolves through the
    # HANDLE, so the library never needs to enter the global namespace — and
    # must not. `dlsym(RTLD_DEFAULT, …)` searches every globally-loaded
    # library in THIS process, so an RTLD_GLOBAL load that is never closed
    # leaks its symbols into every later wrap in the same session: wrapping B
    # after A verified B's slices against A's exports too, and re-wrapping
    # after an edit verified against the previous `.so`. Those symbols do not
    # exist in the consumer's process, so the slice shipped and deadlocked the
    # JIT there — precisely the class this pre-flight exists to prevent.
    handle = try
        Libdl.dlopen(lib_path, Libdl.RTLD_NOW | Libdl.RTLD_LOCAL)
    catch e
        @warn "Tier 1: cannot dlopen '$(basename(lib_path))' to pre-flight slice " *
              "symbols, so no slice can be verified — Tier 1 disabled for this " *
              "wrap (all functions dispatch via ccall)." exception=(e, catch_backtrace())
        empty!(accepted)
        return unresolved
    end

    try
        for name in sort!(collect(accepted))
            # Scoped to the library and its DT_NEEDED chain — what a consumer
            # gets when the wrapper's __init__ dlopens the `.so` RTLD_GLOBAL.
            # Strictly narrower than ORC's process-wide lookup, so a miss can
            # only over-demote (safe), never wrongly accept.
            missing_syms = filter(s -> !_symbol_resolves_via(handle, s),
                                  results[name].declares)
            isempty(missing_syms) && continue
            delete!(accepted, name)
            unresolved[name] = missing_syms
            # A symbol the library cannot supply but this process can is the
            # contamination signature — worth naming, because "missing" and
            # "only here because something else is loaded" have very different
            # fixes.
            stray = filter(s -> _symbol_resolves_via(C_NULL, s), missing_syms)
            isempty(stray) || (strays[name] = stray)
        end
    finally
        try
            Libdl.dlclose(handle)
        catch
        end
    end

    if !isempty(unresolved)
        detail = join(("$fn → " * join(syms, ", ") for (fn, syms) in sort!(collect(unresolved))),
                      "\n    ")
        @warn """
        Tier 1: $(length(unresolved)) function(s) demoted to ccall — their slices
        `declare` symbols that '$(basename(lib_path))' and its dependencies do not
        supply. Each would have deadlocked the JIT on first call instead of
        erroring. A miss here means static promotion did not export something the
        slice reached: check `promoted_symbols` in compilation_metadata.json and
        `[link] promote_statics`.
            $detail
        """
    end
    if !isempty(strays)
        detail = join(("$fn → " * join(syms, ", ") for (fn, syms) in sort!(collect(strays))),
                      "\n    ")
        @warn """
        Tier 1: some of those symbols DO resolve in this build process but are not
        supplied by the library — they come from something else loaded here (an
        earlier wrap in this session, or the host). A consumer loading only
        '$(basename(lib_path))' would not find them, so they are treated as missing.
            $detail
        """
    end
    return unresolved
end

"""
    _tier1_slice_prepass(config, functions, dwarf_structs, lib_path) -> Union{Nothing,Dict{String,String}}

Run the Slicer over every Tier-1 candidate function (non-varargs, not
excluded, `is_c_lto_safe`), apply the hazard/size policy, and pre-flight the
surviving slices' declarations against the real `.so`. Returns `mangled => IR`
for every accepted slice, or `nothing` when the promoted module is missing
(promotion off / fallback build) — Tier 1 then disables loudly for this wrap.

Nothing is written here. Acceptance only makes a function *eligible*; whether a
call site actually materialises is decided later by `lto_shape_ok` and by the
signature dedup, so `_tier1_emit_slices!` writes the files once the final
wrapper text is known.

Hazard policy: `:varargs_callee` (calling printf via declare is fine) and
`:noinline` (correct, just not spliced) are allowed; `:setjmp_family` is gated
unless `[wrap.tier1] allow_setjmp = true`; everything else (`:weak`,
`:inline_asm`, `:module_asm`) demotes to Tier 3.
"""
function _tier1_slice_prepass(config::RepliBuildConfig, functions, dwarf_structs,
                              lib_path::String)
    build_dir = get_build_path(config)
    abi_ll = joinpath(build_dir, "$(config.project.name)_abi.ll")
    if !isfile(abi_ll)
        @warn "Tier 1 enabled but promoted module not found ($abi_ll) — " *
              "promotion is off or this is a fallback/ingest build. Tier 1 " *
              "disabled for this wrap; all functions dispatch via ccall."
        return nothing
    end

    tier1 = config.wrap.tier1
    candidates = String[]
    for func in functions
        get(func, "is_vararg", false) && continue
        mangled = String(get(func, "mangled", get(func, "name", "")))
        isempty(mangled) && continue
        (mangled in tier1.exclude || String(get(func, "name", "")) in tier1.exclude) && continue
        is_c_lto_safe(func, dwarf_structs) || continue
        push!(candidates, mangled)
    end

    slices_dir = joinpath(get_output_path(config), "slices")
    # A policy change must not leave stale slices behind
    isdir(slices_dir) && rm(slices_dir, recursive=true, force=true)
    mkpath(slices_dir)
    isempty(candidates) && return Dict{String,String}()

    results = Slicer.slice_library(abi_ll; targets=unique(candidates),
                                   cache_dir=get_cache_path(config))

    allowed_hazards = Set{Symbol}([:varargs_callee, :noinline])
    tier1.allow_setjmp && push!(allowed_hazards, :setjmp_family)
    max_bytes = tier1.max_slice_kb * 1024

    accepted = Set{String}()
    n_refused = 0; n_hazard = 0; n_oversize = 0
    for (name, r) in results
        if !Slicer.sliced(r)
            n_refused += 1
        elseif !all(h -> h in allowed_hazards, r.hazards)
            n_hazard += 1
        elseif length(r.ir) > max_bytes
            n_oversize += 1
        else
            push!(accepted, name)
        end
    end

    # Symbol pre-flight runs BEFORE anything is handed to the emission loop —
    # a demoted function must not leave a slice for a call site to pick up.
    n_unresolved = length(_tier1_preflight!(accepted, results, lib_path))

    demoted = n_refused + n_hazard + n_oversize + n_unresolved
    println("  tier1: $(length(accepted)) slices accepted" *
            (demoted == 0 ? "" :
             " ($n_refused refused, $n_hazard hazard-gated, $n_oversize oversize, " *
             "$n_unresolved unresolved-symbol → ccall)"))
    return Dict{String,String}(name => results[name].ir for name in accepted)
end

"""
    _render_declares_literal(declares) -> String

Render the per-slice symbol table as a Julia `Dict` literal. Emitted rather
than loaded from a side file so a vendored wrapper carries its own contract:
one file to copy, and no way to ship the table without the kernels that read it.
"""
function _render_declares_literal(declares::Dict{String,Vector{String}})::String
    isempty(declares) && return "Dict{String,Vector{String}}()"
    io = IOBuffer()
    println(io, "Dict{String,Vector{String}}(")
    for k in sort!(collect(keys(declares)))
        println(io, "        ", repr(k), " => ", repr(declares[k]), ",")
    end
    print(io, "    )")
    return String(take!(io))
end

"""
    _tier1_emit_slices!(config, func_chunks, slice_ir, const_owner) -> Vector{String}

Write exactly the slices the FINAL wrapper text reads, and return the Julia
names of the call sites that read them (the `TIER1_FUNCTIONS` surface).

The write set is derived from the post-dedup chunks rather than from the
pre-pass's accepted set, because acceptance is a strictly weaker condition than
emission and nothing else reconciles the two: the pre-pass gates on
`is_c_lto_safe`, while a call site additionally needs `lto_shape_ok` (no
Cstring/struct crossing) and must survive `_dedup_method_chunks`. Writing on
acceptance left slices on disk that no call site could reach — 19 of them in
the Hub lua wrapper, every one a `Cstring` return — sliced, pre-flighted,
shipped inside the package, and dead.

Deriving both the files and the registry from the emitted text is what keeps
them in step: a future emission branch cannot add a `_SLICE_` const without
also getting its slice written and its name registered.

A Tier-1 chunk is recognized by its `@generated function _TIER1_*` kernel; the
write set comes from the `const _SLICE_* = joinpath(...)` path constants the
kernel reads (the `read` itself now lives inside the kernel's generator, so
the const line — not a `read(` line — is the scan anchor).
"""
function _tier1_emit_slices!(config::RepliBuildConfig, func_chunks::Vector{String},
                             slice_ir::Dict{String,String},
                             const_owner::Dict{String,String})
    slices_dir = joinpath(get_output_path(config), "slices")
    mkpath(slices_dir)
    emitted = String[]
    written = Set{String}()
    declares = Dict{String,Vector{String}}()
    for chunk in func_chunks
        occursin("@generated function _TIER1_", chunk) || continue

        for m in eachmatch(r"^const (_SLICE_\w+) = joinpath\(@__DIR__, \"slices\", \"([^\"]+)\.ll\"\)"m, chunk)
            const_name, path_mangled = m.captures
            mangled = get(const_owner, const_name, nothing)
            mangled === nothing &&
                error("Tier 1: emitted constant $const_name has no owning symbol — " *
                      "every slice const must come from _slice_const_name!")
            mangled == path_mangled ||
                error("Tier 1: emitted constant $const_name points at " *
                      "'$path_mangled.ll' but is owned by '$mangled' — the const " *
                      "and its path have drifted apart")
            haskey(slice_ir, mangled) ||
                error("Tier 1: emitted constant $const_name reads a slice for " *
                      "'$mangled', which the pre-pass never accepted")
            mangled in written && continue
            write(joinpath(slices_dir, mangled * ".ll"), slice_ir[mangled])
            push!(written, mangled)
            # Recorded off the IR being written, so the runtime check and the
            # file it guards cannot disagree (see _slice_declared_symbols).
            declares[mangled] = _slice_declared_symbols(slice_ir[mangled])
        end

        fm = match(r"^function ([A-Za-z_][A-Za-z0-9_!]*)\("m, chunk)
        fm === nothing || push!(emitted, fm.captures[1])
    end
    isempty(written) ||
        println("  tier1: $(length(written)) slices emitted " *
                "($(length(unique(emitted))) functions)")
    return emitted, declares
end

function generate_introspective_module_c(config::RepliBuildConfig, lib_path::String,
                                      metadata, module_name::String,
                                      registry::TypeRegistry, generate_docs::Bool,
                                      thunks_lib_path::String="")

    # Track exported symbols
    exports = String[]

    # Header with metadata
    header = """
    # Auto-generated Julia wrapper for $(config.project.name)
    # Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    # Generator: RepliBuild Wrapper (Introspective: DWARF metadata)
    # Library: $(basename(lib_path))
    # Metadata: compilation_metadata.json

    module $module_name
    
    const Cintptr_t = Int
    const Cuintptr_t = UInt

    using Libdl
    import RepliBuild
    import Base: unsafe_convert

    # Resolve the library next to this file first: build artifacts travel as a
    # unit (wrapper + .so side by side, e.g. in ~/.replibuild/builds/<hash>/),
    # so the sibling copy is the one that belongs to this wrapper. The
    # generation-time absolute path is only a fallback — shared dirs like
    # ~/.replibuild/registry/julia/ get overwritten by later builds, stranding
    # any wrapper that baked them in.
    const LIBRARY_PATH = let baked = \"$(abspath(lib_path))\"
        sibling = joinpath(@__DIR__, basename(baked))
        isfile(sibling) ? sibling : baked
    end
    const THUNKS_LIBRARY_PATH = let baked = \"$(thunks_lib_path)\"
        sibling = isempty(baked) ? \"\" : joinpath(@__DIR__, basename(baked))
        !isempty(sibling) && isfile(sibling) ? sibling : baked
    end

    # Verify library exists
    if !isfile(LIBRARY_PATH)
        Base.error("Library not found: \$LIBRARY_PATH (no sibling copy in \$(@__DIR__) either)")
    end

    # ── Build identity ────────────────────────────────────────────────────────
    # This wrapper is portable SOURCE, but its contents are a snapshot of ONE
    # compilation: struct field offsets, blob byte sizes, enum values and the
    # sliced IR all come from the library named below. Point it at a different
    # build — a JLL, a distro package, a rebuild with different flags — and
    # nothing fails loudly; it just reads the wrong offsets. So the identity of
    # the library it was generated against travels with it.
    const BUILD_ID = \"$(_elf_build_id(lib_path) === nothing ? "" : _elf_build_id(lib_path))\"
    const BUILD_TARGET = \"$(Sys.MACHINE)\"
    const BUILD_GENERATOR = \"RepliBuild v$(pkgversion(@__MODULE__) === nothing ? "?" : pkgversion(@__MODULE__))\"

    \"\"\"GNU build ID of the ELF at `path`, or \"\" — read from PT_NOTE, so it survives stripping.\"\"\"
    function _read_build_id(path::AbstractString)
        try
            open(path, \"r\") do io
                read(io, 4) == UInt8[0x7f, 0x45, 0x4c, 0x46] || return \"\"
                seek(io, 4); read(io, UInt8) == 2 || return \"\"
                seek(io, 0x20); phoff = read(io, UInt64)
                seek(io, 0x36); phentsize = read(io, UInt16); phnum = read(io, UInt16)
                for i in 0:(phnum - 1)
                    seek(io, phoff + i * phentsize)
                    read(io, UInt32) == 4 || continue
                    skip(io, 4); off = read(io, UInt64); skip(io, 16); sz = read(io, UInt64)
                    pos, stop = off, off + sz
                    while pos + 12 <= stop
                        seek(io, pos)
                        namesz = read(io, UInt32); descsz = read(io, UInt32); ntype = read(io, UInt32)
                        name = String(read(io, namesz))
                        pad(n) = (n + 3) & ~UInt32(3)
                        desc_at = pos + 12 + pad(namesz)
                        if ntype == 3 && startswith(name, \"GNU\")
                            seek(io, desc_at); return bytes2hex(read(io, descsz))
                        end
                        pos = desc_at + pad(descsz)
                    end
                end
                return \"\"
            end
        catch
            return \"\"
        end
    end

    \"\"\"
    Warn if the library beside this wrapper is not the one it was generated from.

    Only a warning: a rebuild of the same source is usually fine, and refusing
    to load would break legitimate workflows. But layout drift is silent and
    unrecoverable at the call site, so the user gets told once. Tier 1 protects
    itself separately and automatically — see `_slice_symbols_resolve`.
    \"\"\"
    function _check_build_identity()
        isempty(BUILD_ID) && return nothing          # no build ID to compare
        actual = _read_build_id(LIBRARY_PATH)
        (isempty(actual) || actual == BUILD_ID) && return nothing
        @warn \"\"\"
        $(module_name): the library loaded is NOT the build this wrapper was generated from.
        Struct layouts, enum values, blob sizes and any sliced IR in this wrapper are a
        snapshot of the original build; a mismatch can read wrong offsets with no error.
        Regenerate the wrapper against this library if the two are not known-compatible.\"\"\" *
        \"\\n  expected build id: \$BUILD_ID\\n  actual build id:   \$actual\\n  library:           \$LIBRARY_PATH\" *
        \"\\n  generated for:     \$BUILD_TARGET by \$BUILD_GENERATOR\"
        return nothing
    end

    # Flush C stdout so printf output appears immediately in the Julia REPL
    @inline _flush_cstdout() = ccall(:fflush, Cint, (Ptr{Cvoid},), C_NULL)

    \"\"\"
        with(s; field=value, ...) -> typeof(s)

    Copy a generated immutable struct with selected fields replaced — the
    idiomatic way to customize C `*Def`-style structs before passing them in:

        def = with(DefaultDef(); gravity = Vec2(0, -10))

    Padding fields are carried over untouched. Not exported; call as
    `$module_name.with(...)`.

    Byte-blob structs (a single `_data::NTuple{N,UInt8}`, used for layouts that
    could not be modelled field-by-field) have no real fields to rebuild from,
    so they are forwarded to `setproperties`, which writes at DWARF offsets.
    One verb either way.
    \"\"\"
    function with(s::T; kw...) where {T}
        isempty(kw) && return s
        if fieldnames(T) === (:_data,) && isdefined(@__MODULE__, :setproperties)
            return setproperties(s; kw...)
        end
        vals = Any[getfield(s, f) for f in fieldnames(T)]
        for (k, v) in kw
            i = findfirst(==(k), fieldnames(T))
            i === nothing && throw(ArgumentError(string(T) * " has no field " * string(k)))
            ft = fieldtype(T, i)
            vals[i] = v isa ft ? v : convert(ft, v)
        end
        return T(vals...)
    end

    """

    # Track if JIT is required (for virtual methods or complex ABI)
    requires_jit = false

    # Metadata section
    compiler_info = get(metadata, "compiler_info", Dict())
    lto_name = config.project.name
    lto_ir_block = if config.link.enable_lto && config.compile.aot_thunks
        # AOT thunks mode: skip monolithic LTO bitcode (avoids OOM/hang on large projects),
        # only load per-function thunks bitcode
        """
    const LTO_IR = UInt8[]  # Monolithic LTO disabled (AOT thunks mode)
    const THUNKS_LTO_IR_PATH = joinpath(@__DIR__, "$(lto_name)_thunks_lto.bc")
    const THUNKS_LTO_IR = isfile(THUNKS_LTO_IR_PATH) ? read(THUNKS_LTO_IR_PATH) : UInt8[]

    """
    elseif config.link.enable_lto
        """
    # LTO: load LLVM bitcode at module parse time for Base.llvmcall zero-cost dispatch
    # Using bitcode (.bc) is significantly faster for Julia to parse than text IR (.ll)
    const LTO_IR_PATH = joinpath(@__DIR__, "$(lto_name)_lto.bc")
    const LTO_IR = isfile(LTO_IR_PATH) ? read(LTO_IR_PATH) : UInt8[]
    const THUNKS_LTO_IR_PATH = joinpath(@__DIR__, "$(lto_name)_thunks_lto.bc")
    const THUNKS_LTO_IR = isfile(THUNKS_LTO_IR_PATH) ? read(THUNKS_LTO_IR_PATH) : UInt8[]

    """
    else
        """
    const LTO_IR = ""  # LTO disabled for this build
    const THUNKS_LTO_IR = ""

    """
    end
    metadata_section = """
    # =============================================================================
    # Compilation Metadata
    # =============================================================================

    const METADATA = Dict(
        "llvm_version" => "$(get(compiler_info, "llvm_version", "unknown"))",
        "clang_version" => "$(get(compiler_info, "clang_version", "unknown"))",
        "optimization" => "$(get(compiler_info, "optimization_level", "unknown"))",
        "target_triple" => "$(get(compiler_info, "target_triple", "unknown"))",
        "function_count" => $(get(metadata, "function_count", 0)),
        "generated_at" => "$(get(metadata, "timestamp", "unknown"))"
    )

    """ * lto_ir_block

    # Extract metadata
    functions = metadata["functions"]
    dwarf_structs = get(metadata, "struct_definitions", Dict())

    # =========================================================================
    # TIER 1 SLICE PRE-PASS (per-function llvmcall over bitcode slices)
    # =========================================================================
    # Slices every eligible function and returns `mangled => IR`; the emission
    # loop routes accepted functions through Base.llvmcall on the slice instead
    # of ccall. nothing ⇒ tier off. The `.ll` files are written afterwards by
    # _tier1_emit_slices!, from the post-dedup chunks, so only slices a call
    # site actually reads reach disk.
    tier1_slices = nothing
    tier1_const_owner = Dict{String,String}()   # _SLICE_* const => mangled symbol
    if config.wrap.tier1.enable && config.wrap.language == :c
        tier1_slices = _tier1_slice_prepass(config, functions, dwarf_structs, lib_path)
    else
        # Tier 1 off: clear any slices a previous Tier-1 wrap left behind.
        # The prepass owns the wipe when enabled; without this branch a config
        # flip from on to off ships orphan .ll files no call site reads
        # (found live on cjson, 84 orphans, 2026-07-31).
        rm(joinpath(get_output_path(config), "slices"), recursive=true, force=true)
    end

    # Layout registries shared between struct emission and function emission:
    # - resolved_layouts: julia struct name => (byte_size, alignment) for every
    #   struct emitted with VERIFIED named fields (eligible as inline members
    #   and ABI-exact for by-value crossings)
    # - blob_struct_names/sizes/float_risk: structs that stayed opaque byte
    #   blobs; ≤16B float-bearing blobs may not cross the ABI by value
    resolved_layouts = Dict{String,Tuple{Int,Int}}()
    blob_struct_names = Set{String}()
    blob_struct_sizes = Dict{String,Int}()
    # Named-field types whose Julia fields do NOT reproduce the SysV register
    # class — an opaque region standing in for a mixed float/integer aggregate,
    # and (transitively) anything that embeds one. Emission is topological, so
    # a member's risk is always known before its container is emitted.
    abi_class_risk = Set{String}()
    blob_float_risk = Dict{String,Bool}()

    # Struct definitions
    # Collect all struct names from DWARF (excluding enums which have __enum__ prefix)
    struct_types = Set{String}()
    for (name, info) in dwarf_structs
        if !startswith(name, "__enum__") && haskey(info, "members")
            push!(struct_types, name)
        end
    end

    # =============================================================================
    # ENUM GENERATION (from DWARF)
    # =============================================================================

    enum_chunks = String[]

    # Extract enums (stored with __enum__ prefix)
    enum_types = filter(k -> startswith(k, "__enum__"), keys(dwarf_structs))

    # Build set of enum names (without prefix) to exclude from struct generation
    enum_names = Set{String}()
    for enum_key in enum_types
        enum_name = replace(enum_key, "__enum__" => "")
        push!(enum_names, enum_name)
    end

    # Underlying-integer layout of each DWARF enum, for use as struct members
    enum_layouts = Dict{String,Tuple{Int,Int}}()
    for enum_key in enum_types
        enum_name = replace(enum_key, "__enum__" => "")
        ju = String(get(dwarf_structs[enum_key], "julia_type", "Int32"))
        (ju == "Any" || ju == "unknown") && (ju = "Int32")
        if haskey(_C_PRIM_FIELD_LAYOUT, ju)
            enum_layouts[_sanitize_c_type_name(enum_name)] = _C_PRIM_FIELD_LAYOUT[ju]
        end
    end

    if !isempty(enum_types)
        push!(enum_chunks, """
        # =============================================================================
        # Enum Definitions (from DWARF debug info)
        # =============================================================================

        """)

        for enum_key in sort(collect(enum_types))
            enum_name = replace(enum_key, "__enum__" => "")
            enum_info = dwarf_structs[enum_key]

            # Get enum metadata
            underlying_type = get(enum_info, "underlying_type", "int")
            julia_underlying = get(enum_info, "julia_type", "Int32")

            # Fallback to Int32 if type is unknown or Any
            if julia_underlying == "Any" || julia_underlying == "unknown"
                julia_underlying = "Int32"
            end

            enumerators = get(enum_info, "enumerators", [])

            if !isempty(enumerators)
                push!(enum_chunks, """
                # C enum: $enum_name (underlying type: $underlying_type)
                @enum $enum_name::$julia_underlying begin
                """)

                seen_values = Set{Any}()
                seen_names = Set{String}()
                duplicate_defs = String[]
                for (i, enumerator) in enumerate(enumerators)
                    name = get(enumerator, "name", "Unknown")
                    name = _sanitize_c_type_name(name)
                    value = get(enumerator, "value", 0)
                    name = _escape_keyword(name)

                    if name in seen_names
                        continue
                    end
                    push!(seen_names, name)

                    if value in seen_values
                        push!(duplicate_defs, "const $name = $enum_name($value)")
                    else
                        push!(seen_values, value)
                        push!(enum_chunks, "    $name = $value\n")
                    end
                end

                push!(enum_chunks, """
                end

                """)
                if !isempty(duplicate_defs)
                    push!(enum_chunks, join(duplicate_defs, "\n") * "\n\n")
                end
            end
        end

        push!(enum_chunks, "\n")
    end

    # =============================================================================
    # ENUM GENERATION (from Headers - supplementary)
    # =============================================================================
    # Add enums found in headers that weren't in DWARF (unused/optimized away)
    header_enums = get(metadata, "header_enums", Dict())
    if !isempty(header_enums)
        added_header_enums = 0
        for (enum_name, members) in header_enums
            # Skip if already defined from DWARF
            if enum_name in enum_names
                continue
            end

            if !isempty(members)
                if added_header_enums == 0
                    push!(enum_chunks, """
                    # =============================================================================
                    # Enum Definitions (from Headers - supplementary)
                    # =============================================================================
                    # These enums were not in DWARF (unused code eliminated by compiler)

                    """)
                end

                # Choose underlying type based on value range
                min_val = minimum(v for (_, v) in members)
                max_val = maximum(v for (_, v) in members)
                underlying = if min_val >= 0 && max_val > typemax(Int32)
                    "UInt32"
                elseif min_val < typemin(Int32) || max_val > typemax(Int32)
                    "Int64"
                else
                    "Cint"
                end

                push!(enum_chunks, """
                # C enum: $enum_name (from header - not in DWARF)
                @enum $enum_name::$underlying begin
                """)

                seen_values = Set{Any}()
                seen_names = Set{String}()
                duplicate_defs = String[]
                for (member_name, value) in members
                    member_name = _sanitize_c_type_name(string(member_name))
                    member_name = _escape_keyword(member_name)

                    if member_name in seen_names
                        continue
                    end
                    push!(seen_names, member_name)

                    if value in seen_values
                        push!(duplicate_defs, "const $member_name = $enum_name($value)")
                    else
                        push!(seen_values, value)
                        push!(enum_chunks, "    $member_name = $value\n")
                    end
                end

                push!(enum_chunks, """
                end

                """)
                if !isempty(duplicate_defs)
                    push!(enum_chunks, join(duplicate_defs, "\n") * "\n\n")
                end
                added_header_enums += 1
            end
        end

        if added_header_enums > 0
            push!(enum_chunks, "\n")
        end
    end

    # =============================================================================
    # STRUCT GENERATION (from DWARF)
    # =============================================================================

    # Create a mapping from sanitized Julia name back to raw C name for resolving dependencies
    julia_to_cpp_struct = Dict{String, String}()
    for struct_name in struct_types
        julia_to_cpp_struct[_sanitize_c_type_name(struct_name)] = struct_name
    end

    # Scan for ALL referenced struct types that need forward declarations.
    # This includes:
    # 1. Types referenced via Ptr{T} that have no DWARF definition (truly opaque)
    # 2. Types referenced via Ptr{T} that DO have definitions but may appear later
    #    in topological order (forward reference problem)
    # We use `mutable struct Foo end` for truly opaque types, and for defined types
    # we don't forward-declare (they'll be defined in topo order).
    opaque_structs = Set{String}()
    ptr_referenced_structs = Set{String}()  # Struct types referenced via Ptr{X}

    for (name, info) in dwarf_structs
        if !startswith(name, "__enum__") && haskey(info, "members")
            members = get(info, "members", [])
            for member in members
                julia_type = get(member, "julia_type", "Any")
                # Extract inner type from Ptr{T} (recursively for Ptr{Ptr{T}})
                inner = julia_type
                found_ptr = false
                while startswith(inner, "Ptr{") && endswith(inner, "}")
                    inner = inner[5:end-1]
                    found_ptr = true
                end

                # Scan for ALL struct type references in this julia_type
                # Handles: Ptr{Foo}, NTuple{N, Ptr{Foo}}, Ptr{Ptr{Foo}}, Ref{Foo}, etc.
                builtin_types = Set(["Cvoid", "Cint", "Cuint", "Cintptr_t", "Cuintptr_t", "Clong", "Culong", "Cshort", "Cushort",
                                     "Cchar", "Cuchar", "Cfloat", "Cdouble", "Bool", "UInt8", "Int8",
                                     "UInt16", "Int16", "UInt32", "Int32", "UInt64", "Int64", "Csize_t",
                                     "Clonglong", "Culonglong", "Cptrdiff_t", "Cssize_t", "Cwchar_t",
                                     "Cstring", "Float32", "Float64", "Any", "Nothing"])

                # Extract the base type by stripping known container prefixes
                base_ref = julia_type
                while true
                    if startswith(base_ref, "Ptr{") && endswith(base_ref, "}")
                        base_ref = base_ref[5:end-1]
                    elseif startswith(base_ref, "Ref{") && endswith(base_ref, "}")
                        base_ref = base_ref[5:end-1]
                    elseif startswith(base_ref, "NTuple{")
                        ntuple_match = match(r"NTuple\{\d+,\s*([^}]+)\}", base_ref)
                        if !isnothing(ntuple_match)
                            base_ref = strip(ntuple_match.captures[1])
                        else
                            break
                        end
                    else
                        break
                    end
                end
                
                base_ref = strip(base_ref)
                
                # If the inner type is not a builtin, register it
                if !(base_ref in builtin_types) && !isempty(base_ref)
                    # We compare the raw string with struct_types
                    if base_ref in struct_types
                        push!(ptr_referenced_structs, base_ref)
                    else
                        push!(opaque_structs, base_ref)
                    end
                end
            end
        end
    end

    # Also scan function parameter and return types for struct references.
    # Types like sqlite3_blob only appear in function signatures, not struct members.
    _builtin_types = Set(["Cvoid", "Cint", "Cuint", "Cintptr_t", "Cuintptr_t", "Clong", "Culong", "Cshort", "Cushort",
                          "Cchar", "Cuchar", "Cfloat", "Cdouble", "Bool", "UInt8", "Int8",
                          "UInt16", "Int16", "UInt32", "Int32", "UInt64", "Int64", "Csize_t",
                          "Clonglong", "Culonglong", "Cptrdiff_t", "Cssize_t", "Cwchar_t",
                          "Cstring", "Float32", "Float64", "Any", "Nothing"])
    for func in functions
        all_types = String[]
        for param in get(func, "parameters", [])
            push!(all_types, get(param, "julia_type", "Any"))
        end
        ret = get(func, "return_type", nothing)
        if !isnothing(ret)
            push!(all_types, get(ret, "julia_type", "Cvoid"))
        end
        for julia_type in all_types
            base_ref = julia_type
            while true
                if startswith(base_ref, "Ptr{") && endswith(base_ref, "}")
                    base_ref = base_ref[5:end-1]
                elseif startswith(base_ref, "Ref{") && endswith(base_ref, "}")
                    base_ref = base_ref[5:end-1]
                elseif startswith(base_ref, "NTuple{")
                    ntuple_match = match(r"NTuple\{\d+,\s*([^}]+)\}", base_ref)
                    if !isnothing(ntuple_match)
                        base_ref = strip(ntuple_match.captures[1])
                    else
                        break
                    end
                else
                    break
                end
            end
            base_ref = strip(base_ref)
            if !(base_ref in _builtin_types) && !isempty(base_ref)
                if base_ref in struct_types
                    push!(ptr_referenced_structs, base_ref)
                else
                    push!(opaque_structs, base_ref)
                end
            end
        end
    end

    struct_chunks = String[]
    union_accessor_chunks = String[]  # Deferred union accessors (emitted after all struct defs)
    blob_setters_emitted = false  # gates the one module-level `setproperties`

    # Emit forward declarations for:
    # 1. Truly opaque types (no DWARF definition) — as mutable struct
    # 2. Struct types referenced via Ptr{X} — as empty struct (will be redefined with fields later)
    # This ensures Ptr{X} fields can reference types defined later in the file.
    # BUT: skip forward declarations for types that will get a real DWARF definition,
    # since Julia doesn't allow redefining a struct with the same name.
    all_forward_decls = union(opaque_structs, ptr_referenced_structs)

    # Build set of sanitized names that will get real DWARF definitions
    dwarf_defined_names = Set{String}()
    for (name, info) in dwarf_structs
        s = _sanitize_c_type_name(name)
        s = replace(s, "*" => "Ptr")
        s = replace(s, "&" => "Ref")
        s = replace(s, r"[^a-zA-Z0-9_]" => "_")
        push!(dwarf_defined_names, s)
    end
    # Also exclude enum names — they're already defined via @enum
    union!(dwarf_defined_names, enum_names)

    if !isempty(all_forward_decls)
        push!(struct_chunks, """
        # =============================================================================
        # Forward Declarations (Opaque + Ptr-referenced types)
        # =============================================================================

        """)
        seen_forward_decls = Set{String}()
        for name in sort(collect(all_forward_decls))
            # Skip blocklisted internal/system types
            name in _INTERNAL_TYPE_BLOCKLIST && continue

            # Sanitize
            s_name = _sanitize_c_type_name(name)
            s_name = replace(s_name, "*" => "Ptr")
            s_name = replace(s_name, "&" => "Ref")
            s_name = replace(s_name, r"[^a-zA-Z0-9_]" => "_")

            # Skip names that are Julia keywords or builtins
            if s_name in ("char", "int", "void", "bool", "float", "double", "long", "short")
                continue
            end

            # Skip duplicates (different C names can sanitize to the same Julia identifier)
            s_name in seen_forward_decls && continue
            push!(seen_forward_decls, s_name)

            # Skip forward declaration if this type will get a real DWARF definition later
            s_name in dwarf_defined_names && continue

            # Both opaque types and pointer-referenced types should be immutable structs
            # to preserve inline ABI layout (0-byte empty structs) if they are used by value.
            push!(struct_chunks, "struct $s_name end\n")
        end
        push!(struct_chunks, "\n")
    end

    if !isempty(struct_types)
        push!(struct_chunks, """
        # =============================================================================
        # Struct Definitions (from DWARF debug info)
        # =============================================================================

        """)

        # Topologically sort structs by dependencies
        # Build dependency graph: struct_name => [dependencies]
        struct_deps = Dict{String, Set{String}}()
        struct_soft_deps = Dict{String, Set{String}}()

        for struct_name in struct_types
            deps = Set{String}()
            soft_deps = Set{String}()

            if haskey(dwarf_structs, struct_name)
                struct_info = dwarf_structs[struct_name]
                members = get(struct_info, "members", [])

                for member in members
                    julia_type = get(member, "julia_type", "Any")

                    # Extract type name from various wrapper types:
                    #   Ptr{Foo} -> Foo (soft dep)
                    #   NTuple{N, Foo} -> Foo (hard dep, needs full definition first)
                    #   Ref{Foo} -> Foo (hard dep)
                    #   Foo -> Foo (hard dep, direct by-value embedding)
                    dep_type = nothing
                    is_soft = false

                    if startswith(julia_type, "Ptr{")
                        ptr_match = match(r"Ptr\{([^}]+)\}", julia_type)
                        if !isnothing(ptr_match)
                            dep_type = strip(ptr_match.captures[1])
                            is_soft = true
                        end
                    elseif startswith(julia_type, "NTuple{")
                        ntuple_match = match(r"NTuple\{\d+,\s*([^}]+)\}", julia_type)
                        if !isnothing(ntuple_match)
                            dep_type = strip(ntuple_match.captures[1])
                        end
                    elseif startswith(julia_type, "Ref{")
                        ref_match = match(r"Ref\{([^}]+)\}", julia_type)
                        if !isnothing(ref_match)
                            dep_type = strip(ref_match.captures[1])
                        end
                    else
                        dep_type = julia_type
                    end

                    if !isnothing(dep_type) && haskey(julia_to_cpp_struct, dep_type)
                        cpp_dep = julia_to_cpp_struct[dep_type]
                        if cpp_dep != struct_name
                            if is_soft
                                push!(soft_deps, cpp_dep)
                            else
                                push!(deps, cpp_dep)
                            end
                        end
                    end
                end
            end

            struct_deps[struct_name] = deps
            struct_soft_deps[struct_name] = soft_deps
        end

        # Topological sort using Kahn's algorithm
        sorted_structs = String[]
        remaining_hard = Dict(k => copy(v) for (k, v) in struct_deps)
        remaining_soft = Dict(k => copy(v) for (k, v) in struct_soft_deps)

        while !isempty(remaining_hard)
            # Find structs with no hard AND no soft dependencies
            ready = [name for (name, deps) in remaining_hard if isempty(deps) && isempty(remaining_soft[name])]

            if isempty(ready)
                # Break soft dependency cycle: find structs with no hard dependencies
                ready = [name for (name, deps) in remaining_hard if isempty(deps)]
            end

            if isempty(ready)
                # Circular hard dependency - just take alphabetically first
                ready = [sort(collect(keys(remaining_hard)))[1]]
            end

            for name in sort(ready)
                push!(sorted_structs, name)
                delete!(remaining_hard, name)
                delete!(remaining_soft, name)

                # Remove this struct from all dependency lists
                for deps in values(remaining_hard)
                    delete!(deps, name)
                end
                for deps in values(remaining_soft)
                    delete!(deps, name)
                end
            end
        end

        # Helper to recursively flatten struct members from base classes
        function flatten_struct_members(s_name, visited=Set{String}())
            if s_name in visited
                return []
            end
            push!(visited, s_name)
            
            all_members = []
            if haskey(dwarf_structs, s_name)
                info = dwarf_structs[s_name]
                
                                own_members = get(info, "members", [])
                append!(all_members, own_members)
            end
            return all_members
        end

        # Pre-merge DWARF keys that sanitize to the same Julia name.
        # Multiple C++ template nesting depths (e.g., "Matrix<double,-1,-1> >" vs
        # "Matrix<double,-1,-1> > >") sanitize to the same identifier.
        # Keep the one with the largest byte_size and most members.
        best_dwarf_key = Dict{String, String}()  # sanitized_name -> best struct_name
        for struct_name in sorted_structs
            struct_name in enum_names && continue
            
            jname = _sanitize_c_type_name(struct_name)
            jname = replace(jname, "*" => "Ptr")
            jname = replace(jname, "&" => "Ref")
            jname = replace(jname, r"[^a-zA-Z0-9_]" => "_")
            
            if !haskey(best_dwarf_key, jname)
                best_dwarf_key[jname] = struct_name
            else
                old_key = best_dwarf_key[jname]
                if haskey(dwarf_structs, struct_name) && haskey(dwarf_structs, old_key)
                    old_info = dwarf_structs[old_key]
                    new_info = dwarf_structs[struct_name]
                    old_bs = try; s = get(old_info, "byte_size", "0"); startswith(string(s), "0x") ? parse(Int, string(s)[3:end], base=16) : parse(Int, string(s)); catch; 0; end
                    new_bs = try; s = get(new_info, "byte_size", "0"); startswith(string(s), "0x") ? parse(Int, string(s)[3:end], base=16) : parse(Int, string(s)); catch; 0; end
                    old_mc = length(get(old_info, "members", []))
                    new_mc = length(get(new_info, "members", []))
                    if new_bs > old_bs || (new_bs == old_bs && new_mc > old_mc)
                        best_dwarf_key[jname] = struct_name
                    end
                elseif haskey(dwarf_structs, struct_name) && !haskey(dwarf_structs, old_key)
                    best_dwarf_key[jname] = struct_name
                end
            end
        end

        seen_struct_defs = Set{String}()
        defined_struct_names = Set{String}()
        if @isdefined(seen_forward_decls)
            # Only seed with forward decls that were actually emitted (not DWARF-defined skips)
            for s in seen_forward_decls
                if !(s in dwarf_defined_names)
                    push!(defined_struct_names, s)
                end
            end
        end
        for struct_name in sorted_structs
            # Skip if this is actually an enum (enums are generated separately)
            if struct_name in enum_names
                continue
            end

            # Skip STL internal implementation types (they leak from DWARF when
            # template instantiation is forced, but are not useful to Julia users)

            # Skip compiler/libc internal types
            if struct_name in _INTERNAL_TYPE_BLOCKLIST
                continue
            end

            # Sanitize struct name for Julia (replace < > , with _)
            julia_struct_name = _sanitize_c_type_name(struct_name)
            julia_struct_name = replace(julia_struct_name, "*" => "Ptr")
            julia_struct_name = replace(julia_struct_name, "&" => "Ref")
            julia_struct_name = replace(julia_struct_name, r"[^a-zA-Z0-9_]" => "_")

            # Skip duplicate sanitized names — only process the "best" DWARF key
            # (the one with largest byte_size / most members)
            if julia_struct_name in seen_struct_defs
                continue
            end
            if haskey(best_dwarf_key, julia_struct_name) && best_dwarf_key[julia_struct_name] != struct_name
                continue
            end
            push!(seen_struct_defs, julia_struct_name)
            push!(defined_struct_names, julia_struct_name)

            # Check if we have DWARF member information for this struct
            if haskey(dwarf_structs, struct_name)
                struct_info = dwarf_structs[struct_name]
                kind = get(struct_info, "kind", "struct")
                
                # SPECIAL HANDLING FOR UNIONS
                #
                # ANONYMOUS unions are deliberately excluded: this path emits a
                # `mutable struct` (its accessors need `pointer_from_objref`),
                # and a mutable type is a REFERENCE when used as a field, so it
                # can never be embedded in the parent that declared it — which is
                # the only thing an anonymous union is for. Those fall through to
                # the struct path, where overlapping members fail exact layout and
                # the opaque-region branch gives them an immutable, correctly
                # aligned, embeddable representation with `getproperty` accessors.
                if kind == "union" && get(struct_info, "anonymous", false) != true
                    byte_size = _parse_dwarf_size(struct_info)
                    members = get(struct_info, "members", [])

                    if byte_size == 0
                        # Fallback if size missing
                        for m in members
                            m_size = get(m, "size", 0)
                            byte_size = max(byte_size, m_size)
                        end
                        if byte_size == 0; byte_size = 8; end # Panic fallback
                    end

                    push!(struct_chunks, """
                    # C union: $struct_name (size $byte_size bytes)
                    mutable struct $julia_struct_name
                        data::NTuple{$byte_size, UInt8}
                    end
                    $julia_struct_name() = $julia_struct_name(ntuple(i -> 0x00, $byte_size))

                    """)

                    # Collect all struct names being generated for reference
                    known_struct_names = Set{String}()
                    for sn in keys(dwarf_structs)
                        push!(known_struct_names, _sanitize_c_type_name(sn))
                    end

                    # Julia primitive types that are always available
                    julia_builtins = Set(["Cvoid", "Cint", "Cuint", "Cchar", "Cuchar",
                        "Cshort", "Cushort", "Clong", "Culong", "Clonglong", "Culonglong",
                        "Cfloat", "Cdouble", "Cstring", "Csize_t", "Cssize_t", "Cptrdiff_t",
                        "Bool", "Int8", "UInt8", "Int16", "UInt16", "Int32", "UInt32",
                        "Int64", "UInt64", "Float32", "Float64", "Nothing"])

                    # Generate typed accessor functions for each union member
                    for m in members
                        m_name = get(m, "name", "")
                        m_julia_type = get(m, "julia_type", "Any")
                        if isempty(m_name) || m_julia_type == "Any"
                            continue
                        end

                        # Sanitize member type name to match struct definitions
                        m_julia_type = _sanitize_c_type_name(m_julia_type)

                        # Sanitize member name for Julia identifier
                        safe_m_name = replace(m_name, r"[^A-Za-z0-9_]" => "_")

                        # Skip complex types (nested structs, arrays) — only primitive/pointer accessors
                        if startswith(m_julia_type, "NTuple{") || startswith(m_julia_type, "Vector{")
                            continue
                        end

                        # Check that Ptr{X} inner types are defined; fall back to Ptr{Cvoid} if not
                        ptr_match = match(r"^Ptr\{(.+)\}$", m_julia_type)
                        if !isnothing(ptr_match)
                            inner = ptr_match.captures[1]
                            # Recursively unwrap nested Ptr{}
                            while (pm = match(r"^Ptr\{(.+)\}$", inner)) !== nothing
                                inner = pm.captures[1]
                            end
                            if inner ∉ julia_builtins && inner ∉ known_struct_names
                                m_julia_type = "Ptr{Cvoid}"
                            end
                        elseif m_julia_type ∉ julia_builtins && m_julia_type ∉ known_struct_names
                            # Non-pointer, non-primitive, non-generated struct — skip
                            continue
                        end

                        # Scope accessor names to the union type to avoid collisions
                        # when multiple unions have members with the same name.
                        accessor_prefix = "$(julia_struct_name)_$(safe_m_name)"
                        push!(union_accessor_chunks, """
                        \"\"\"Get union member `$m_name` as `$m_julia_type` from `$julia_struct_name`.\"\"\"
                        function get_$(accessor_prefix)(u::$julia_struct_name)::$m_julia_type
                            return unsafe_load(Ptr{$m_julia_type}(pointer_from_objref(u)))
                        end

                        \"\"\"Set union member `$m_name` as `$m_julia_type` in `$julia_struct_name`.\"\"\"
                        function set_$(accessor_prefix)!(u::$julia_struct_name, v::$m_julia_type)
                            unsafe_store!(Ptr{$m_julia_type}(pointer_from_objref(u)), v)
                        end

                        """)
                        push!(exports, "get_$(accessor_prefix)")
                        push!(exports, "set_$(accessor_prefix)!")
                    end

                    continue
                end

                # BUG FIX: Recursively get all members including base classes
                members = flatten_struct_members(struct_name)

                if !isempty(members)
                    # Sort members by offset to ensure correct order
                    sort!(members, by = m -> begin
                        off = get(m, "offset", "0x0")
                        isnothing(off) ? 0 : parse(Int, off)
                    end)

                    # Check if this struct has bitfield members
                    has_bitfields = any(m -> haskey(m, "bit_size"), members)

                    if has_bitfields
                        byte_size = _parse_dwarf_size(struct_info)
                        if byte_size == 0; byte_size = 8; end

                        push!(blob_struct_names, julia_struct_name)
                        blob_struct_sizes[julia_struct_name] = byte_size
                        blob_float_risk[julia_struct_name] = _struct_has_float_member(struct_name, dwarf_structs)
                        push!(struct_chunks, """
                        # C struct with bitfields: $struct_name (size $byte_size bytes)
                        struct $julia_struct_name
                            _data::NTuple{$byte_size, UInt8}
                        end
                        $julia_struct_name() = $julia_struct_name(ntuple(i -> 0x00, $byte_size))

                        """)

                        # Generate accessor functions for each member
                        for member in members
                            member_name = get(member, "name", "unknown")
                            julia_type = get(member, "julia_type", "Any")
                            safe_member = replace(member_name, r"[^A-Za-z0-9_]" => "_")

                            # Sanitize julia_type to avoid < > in function signatures
                            sanitized_type = julia_type
                            if occursin(r"[<>]", julia_type)
                                if occursin(r"Ptr\{[^}]+\}", julia_type)
                                    type_match = match(r"Ptr\{([^}]+)\}", julia_type)
                                    if !isnothing(type_match)
                                        inner_type = String(type_match.captures[1])
                                        sanitized_type = "Ptr{$(_sanitize_c_type_name(inner_type))}"
                                    end
                                else
                                    sanitized_type = _sanitize_c_type_name(julia_type)
                                end
                            end
                            julia_type = sanitized_type

                            if haskey(member, "bit_size")
                                bit_size = member["bit_size"]

                                # Determine absolute bit offset
                                bit_offset = if haskey(member, "data_bit_offset")
                                    member["data_bit_offset"]
                                elseif haskey(member, "bit_offset_legacy")
                                    byte_off = _parse_int_or_hex(get(member, "offset", "0"))
                                    byte_off * 8 + member["bit_offset_legacy"]
                                else
                                    0
                                end

                                byte_pos = bit_offset >> 3
                                bit_within_byte = bit_offset & 7
                                mask = (1 << bit_size) - 1

                                # Getter with bit extraction
                                if bit_size <= 8 && bit_within_byte + bit_size <= 8
                                    # Single byte access
                                    push!(struct_chunks, """
                                    \"\"\"Get bitfield `$member_name` ($bit_size bits) from `$julia_struct_name`.\"\"\"
                                    function get_$(safe_member)(s::$julia_struct_name)::UInt32
                                        return UInt32((s._data[$(byte_pos + 1)] >> $bit_within_byte) & $(mask))
                                    end

                                    \"\"\"Set bitfield `$member_name` ($bit_size bits) in `$julia_struct_name` (returns new instance).\"\"\"
                                    function set_$(safe_member)(s::$julia_struct_name, v::Integer)::$julia_struct_name
                                        data = collect(s._data)
                                        cleared = data[$(byte_pos + 1)] & ~UInt8($(mask) << $bit_within_byte)
                                        data[$(byte_pos + 1)] = cleared | UInt8((UInt32(v) & $(mask)) << $bit_within_byte)
                                        return $julia_struct_name(NTuple{$byte_size, UInt8}(data))
                                    end

                                    """)
                                else
                                    # Multi-byte bitfield — assemble exactly the spanned
                                    # bytes instead of loading a power-of-2 container.
                                    # A container that overhangs the spanned region can
                                    # read (getter) or WRITE (setter) past the struct
                                    # tail — an OOB heap write for fields near the end.
                                    n_bytes_span = (bit_within_byte + bit_size + 7) >> 3
                                    if byte_pos + n_bytes_span > byte_size
                                        @warn "Bitfield $member_name in $struct_name spans past DWARF byte_size ($byte_size B); clamping accessor to the recorded size"
                                        n_bytes_span = max(1, byte_size - byte_pos)
                                    end
                                    acc_type = n_bytes_span <= 8 ? "UInt64" : "UInt128"
                                    # Return the smallest unsigned type that fits the field
                                    ret_type = bit_size <= 16 ? "UInt16" :
                                               bit_size <= 32 ? "UInt32" : "UInt64"
                                    # Wrapping mask expression: valid up to bit_size == 64
                                    # ($acc_type(1) << 64 == 0, minus 1 wraps to all-ones)
                                    amask = "(($acc_type(1) << $bit_size) - $acc_type(1))"
                                    push!(struct_chunks, """
                                    \"\"\"Get bitfield `$member_name` ($bit_size bits) from `$julia_struct_name`.\"\"\"
                                    function get_$(safe_member)(s::$julia_struct_name)::$ret_type
                                        data = s._data
                                        acc = zero($acc_type)
                                        @inbounds for i in 0:$(n_bytes_span - 1)
                                            acc |= $acc_type(data[$(byte_pos + 1) + i]) << (8 * i)
                                        end
                                        return $ret_type((acc >> $bit_within_byte) & $amask)
                                    end

                                    \"\"\"Set bitfield `$member_name` ($bit_size bits) in `$julia_struct_name` (returns new instance).\"\"\"
                                    function set_$(safe_member)(s::$julia_struct_name, v::Integer)::$julia_struct_name
                                        data = collect(s._data)
                                        acc = zero($acc_type)
                                        @inbounds for i in 0:$(n_bytes_span - 1)
                                            acc |= $acc_type(data[$(byte_pos + 1) + i]) << (8 * i)
                                        end
                                        acc = (acc & ~($amask << $bit_within_byte)) | (((v % $acc_type) & $amask) << $bit_within_byte)
                                        @inbounds for i in 0:$(n_bytes_span - 1)
                                            data[$(byte_pos + 1) + i] = UInt8((acc >> (8 * i)) & 0xff)
                                        end
                                        return $julia_struct_name(NTuple{$byte_size, UInt8}(data))
                                    end

                                    """)
                                end
                                push!(exports, "get_$(safe_member)")
                                push!(exports, "set_$(safe_member)")
                            else
                                # Non-bitfield member in a bitfield struct — byte-offset accessor
                                byte_off = _parse_int_or_hex(get(member, "offset", "0"))
                                # The member type must be one the module will
                                # actually bind. `Any`/`NTuple` were the only
                                # exclusions, so a member whose struct type never
                                # gets defined (miniaudio: `memoryStream` typed
                                # ma_dr_flac__memory_stream) emitted an accessor
                                # naming an undeclared type — UndefVarError at
                                # include, whole module dead. dwarf_defined_names
                                # is computed BEFORE emission, so gating on it has
                                # no ordering hazard.
                                _acc_type_ok = julia_type in _JULIA_BUILTIN_TYPES ||
                                               julia_type in dwarf_defined_names ||
                                               !isnothing(match(r"^Ptr\{", julia_type))
                                if julia_type != "Any" && !startswith(julia_type, "NTuple{") && _acc_type_ok
                                    push!(struct_chunks, """
                                    \"\"\"Get non-bitfield member `$member_name` from `$julia_struct_name`.\"\"\"
                                    function get_$(safe_member)(s::$julia_struct_name)::$julia_type
                                        buf = collect(s._data)
                                        GC.@preserve buf begin
                                            p = pointer(buf) + $byte_off
                                            return unsafe_load(Ptr{$julia_type}(p))
                                        end
                                    end

                                    """)
                                    push!(exports, "get_$(safe_member)")
                                end
                            end
                        end

                        continue
                    end

                    # Exact member-layout resolution: type every member with a
                    # Julia field type of known size+alignment — primitives,
                    # pointers, enums, NTuple{N,·}, and structs already emitted
                    # with verified named fields — then prove Julia's natural
                    # layout reproduces every DWARF offset and the total size.
                    # Anything unprovable keeps the opaque byte blob: exact or
                    # opaque, never approximate.
                    byte_size = _parse_dwarf_size(struct_info)
                    layout_verified = false
                    struct_max_align = 1
                    exact_layout = _resolve_exact_layout(members, byte_size, resolved_layouts, enum_layouts)
                    if exact_layout !== nothing
                        members = exact_layout[1]
                        struct_max_align = exact_layout[2]
                        layout_verified = true
                    end

                    if !layout_verified && byte_size > 0
                        member_count = length(members)
                        _is_anon_agg = get(struct_info, "anonymous", false) == true
                        _float_risk = _struct_has_float_member(struct_name, dwarf_structs)

                        # An ANONYMOUS aggregate (a C11 `union {...};` member, or
                        # the type of a `union {...} u;` member) exists only to be
                        # EMBEDDED in its parent, so an align-1 `NTuple{N,UInt8}`
                        # is not good enough: it would under-align the parent and
                        # shift every field after it. When the alignment can be
                        # proven from DWARF, the region carries it in its element
                        # type and the aggregate becomes an ABI-exact embeddable
                        # member — which is what lets the parent keep its own named
                        # fields instead of collapsing to a blob (toml_datum_t).
                        #
                        # Withheld at ≤16 bytes with float risk: there the SysV
                        # register class depends on the real member types, and an
                        # integer-typed region would claim INTEGER where the struct
                        # travels in SSE. Those stay blobs so the existing
                        # by-value crossing guard still sees them.
                        _region = nothing
                        _region_align = 0
                        _all_float = _struct_all_float_members(struct_name, dwarf_structs)
                        if _is_anon_agg
                            _al = _dwarf_aggregate_align(julia_struct_name, dwarf_structs,
                                                         best_dwarf_key, enum_layouts)
                            if _al !== nothing
                                _region = _aligned_region_storage(byte_size, _al;
                                                                  all_float=_all_float)
                                _region !== nothing && (_region_align = _al)
                            end
                        end

                        if _region !== nothing
                            _elem, _count = _region
                            resolved_layouts[julia_struct_name] = (byte_size, _region_align)
                            # MIXED float/non-float members: the region's single
                            # element type can only claim one SysV class for every
                            # eightbyte, and the real aggregate may split them
                            # (`union { double d; struct { int a; double b; } s; }`
                            # is INTEGER,SSE). Above 16 bytes it is MEMORY class on
                            # both views and none of this matters; at or below, the
                            # type is flagged so the by-value crossing guard still
                            # refuses loudly instead of miscompiling silently.
                            if byte_size <= 16 && _float_risk && !_all_float
                                push!(abi_class_risk, julia_struct_name)
                            end
                            push!(struct_chunks, """
                            # Anonymous C $kind: $struct_name ($member_count members,
                            # $byte_size-byte opaque region, alignment $_region_align)
                            struct $julia_struct_name
                                _data::NTuple{$_count, $_elem}
                            end

                            # Zero-initializer for $julia_struct_name
                            function $julia_struct_name()
                                return $julia_struct_name(ntuple(i -> $(_elem)(0), $_count))
                            end

                            """)
                        else
                            push!(blob_struct_names, julia_struct_name)
                            blob_struct_sizes[julia_struct_name] = byte_size
                            blob_float_risk[julia_struct_name] = _float_risk
                            push!(struct_chunks, """
                            # C struct: $struct_name ($member_count members, byte blob for ABI safety)
                            struct $julia_struct_name
                                _data::NTuple{$(byte_size), UInt8}
                            end

                            # Zero-initializer for $julia_struct_name
                            function $julia_struct_name()
                                return $julia_struct_name(ntuple(i -> 0x00, $byte_size))
                            end

                            """)
                        end

                        # Generate Base.getproperty accessor for named member access on byte-blob structs
                        _accessor_branches = String[]
                        # …and the matching writes. Built in the SAME loop off the
                        # same offset/type decision, so a member can never be
                        # readable but unsettable (see _blob_store_expr).
                        _setter_branches = String[]
                        _setter_delegates = String[]
                        _loadable_primitives = Dict(
                            "Cdouble" => ("Cdouble", 8), "Cfloat" => ("Cfloat", 4),
                            "Cint" => ("Cint", 4), "Cuint" => ("Cuint", 4),
                            "Clong" => ("Clong", 8), "Culong" => ("Culong", 8),
                            "Clonglong" => ("Clonglong", 8), "Culonglong" => ("Culonglong", 8),
                            "Cshort" => ("Cshort", 2), "Cushort" => ("Cushort", 2),
                            "Cchar" => ("Cchar", 1), "Cuchar" => ("Cuchar", 1),
                            "Csize_t" => ("Csize_t", 8), "Cptrdiff_t" => ("Cptrdiff_t", 8),
                            "Cssize_t" => ("Cssize_t", 8), "Bool" => ("Bool", 1),
                            "UInt8" => ("UInt8", 1), "Int8" => ("Int8", 1),
                            "UInt16" => ("UInt16", 2), "Int16" => ("Int16", 2),
                            "UInt32" => ("UInt32", 4), "Int32" => ("Int32", 4),
                            "UInt64" => ("UInt64", 8), "Int64" => ("Int64", 8),
                            "Float32" => ("Float32", 4), "Float64" => ("Float64", 8),
                        )
                        # Pre-compute in-context sizes for each member from offset gaps
                        _member_offsets = Int[]
                        for m in members
                            m_offset_str = get(m, "offset", nothing)
                            push!(_member_offsets, isnothing(m_offset_str) ? -1 : _parse_int_or_hex(m_offset_str))
                        end
                        for (mi, m) in enumerate(members)
                            m_name = get(m, "name", nothing)
                            isnothing(m_name) && continue
                            m_offset = _member_offsets[mi]
                            m_offset < 0 && continue
                            m_julia_type = get(m, "julia_type", "")
                            m_c_type = strip(get(m, "c_type", ""))

                            # Compute available space for this member from offset gaps.
                            # Union members all sit at offset 0, so the gap to the
                            # "next" member is 0 — every one of them owns the whole
                            # aggregate instead.
                            available_size = byte_size - m_offset
                            if kind != "union"
                                next_offset = byte_size
                                for j in (mi+1):length(_member_offsets)
                                    if _member_offsets[j] >= 0
                                        next_offset = _member_offsets[j]
                                        break
                                    end
                                end
                                available_size = next_offset - m_offset
                            end

                            # Pointer types
                            # (the Ref temporary is the GC root that must be
                            # preserved — preserving `x`, an immutable value,
                            # would not keep the box alive)
                            if endswith(m_c_type, "*")
                                push!(_accessor_branches, """
                                    if s === :$m_name
                                        r = Ref(getfield(x, :_data))
                                        return GC.@preserve r unsafe_load(Ptr{Ptr{Cvoid}}(pointer_from_objref(r) + $m_offset))
                                    end""")
                                push!(_setter_branches, _blob_store_expr(m_name, m_offset, :pointer))
                            # Primitive types we can unsafe_load directly
                            elseif haskey(_loadable_primitives, m_julia_type)
                                jt, _ = _loadable_primitives[m_julia_type]
                                push!(_accessor_branches, """
                                    if s === :$m_name
                                        r = Ref(getfield(x, :_data))
                                        return GC.@preserve r unsafe_load(Ptr{$jt}(pointer_from_objref(r) + $m_offset))
                                    end""")
                                push!(_setter_branches, _blob_store_expr(m_name, m_offset, :primitive; julia_type=jt))
                            # Enum members. `@enum T::U` is a primitive type of the
                            # underlying integer's size, so it loads like one — but it
                            # is not in _loadable_primitives, so every enum member of a
                            # blob struct used to be silently dropped (toml_datum_t
                            # lost `type`, which is the tag that says which union arm
                            # is live — the single most important field on the struct).
                            elseif haskey(enum_layouts, _sanitize_c_type_name(m_julia_type))
                                _ejt = _sanitize_c_type_name(m_julia_type)
                                push!(_accessor_branches, """
                                    if s === :$m_name
                                        r = Ref(getfield(x, :_data))
                                        return GC.@preserve r unsafe_load(Ptr{$_ejt}(pointer_from_objref(r) + $m_offset))
                                    end""")
                                push!(_setter_branches, _blob_store_expr(m_name, m_offset, :primitive; julia_type=_ejt))
                            # Fixed-size arrays (`char tag[16]`, `double raw[3]`) —
                            # NTuple loads directly and is exactly the member's size.
                            elseif startswith(m_julia_type, "NTuple{") &&
                                   match(r"^NTuple\{\d+,\s*(.+)\}$", m_julia_type) !== nothing &&
                                   haskey(_C_PRIM_FIELD_LAYOUT,
                                          String(strip(match(r"^NTuple\{\d+,\s*(.+)\}$", m_julia_type).captures[1])))
                                push!(_accessor_branches, """
                                    if s === :$m_name
                                        r = Ref(getfield(x, :_data))
                                        return GC.@preserve r unsafe_load(Ptr{$m_julia_type}(pointer_from_objref(r) + $m_offset))
                                    end""")
                                push!(_setter_branches, _blob_store_expr(m_name, m_offset, :primitive; julia_type=m_julia_type))
                            # Nested struct/union types — extract sub-blob
                            else
                                # Prefer the Julia type name: an anonymous aggregate
                                # is carried under a synthesized name that needs no
                                # sanitizing, and so is every plain C struct name.
                                m_sanitized = _sanitize_c_type_name(
                                    isempty(String(m_julia_type)) || m_julia_type == "Any" ?
                                    String(m_c_type) : String(m_julia_type))
                                # NOTE: this used to also require `m_sanitized !=
                                # m_c_type`, i.e. it only fired for names that HAD to
                                # be sanitized (C++ template spellings). Every plainly
                                # named nested struct — `toml_datum_t toptab;` — was
                                # therefore dropped from its parent's accessors. The
                                # dwarf_structs lookup below is the real guard.
                                if !isempty(m_sanitized)
                                    # Find the nested struct's byte_size using best_dwarf_key map
                                    nested_info = nothing
                                    best_key = get(best_dwarf_key, m_sanitized, nothing)
                                    if !isnothing(best_key) && haskey(dwarf_structs, best_key)
                                        nested_info = dwarf_structs[best_key]
                                    else
                                        # Fallback: search all structs for sanitized name match
                                        for (sk, sv) in dwarf_structs
                                            if _sanitize_c_type_name(sk) == m_sanitized
                                                nested_info = sv
                                                break
                                            end
                                        end
                                    end
                                    if !isnothing(nested_info)
                                        nested_bs_str = get(nested_info, "byte_size", "0x0")
                                        nested_bs = isnothing(nested_bs_str) ? 0 :
                                            (startswith(string(nested_bs_str), "0x") ?
                                             parse(Int, string(nested_bs_str)[3:end], base=16) :
                                             parse(Int, string(nested_bs_str)))
                                        if nested_bs > 0
                                            if m_sanitized in blob_struct_names
                                                # Target is a byte-blob struct — extract bytes
                                                actual_size = min(nested_bs, available_size)
                                                if actual_size == nested_bs
                                                    push!(_accessor_branches, """
                                    if s === :$m_name
                                        r = Ref(getfield(x, :_data))
                                        bytes = GC.@preserve r ntuple(i -> unsafe_load(Ptr{UInt8}(pointer_from_objref(r) + $m_offset + i - 1)), $nested_bs)
                                        return $(m_sanitized)(bytes)
                                    end""")
                                                    push!(_setter_branches, _blob_store_expr(m_name, m_offset, :blob_struct_full;
                                                        struct_name=m_sanitized, nested_size=nested_bs))
                                                else
                                                    push!(_accessor_branches, """
                                    if s === :$m_name
                                        r = Ref(getfield(x, :_data))
                                        raw = GC.@preserve r ntuple(i -> unsafe_load(Ptr{UInt8}(pointer_from_objref(r) + $m_offset + i - 1)), $actual_size)
                                        padded = ntuple(i -> i <= $actual_size ? raw[i] : 0x00, $nested_bs)
                                        return $(m_sanitized)(padded)
                                    end""")
                                                    push!(_setter_branches, _blob_store_expr(m_name, m_offset, :blob_struct_partial;
                                                        struct_name=m_sanitized, nested_size=nested_bs, actual_size=actual_size))
                                                end
                                            else
                                                # Target is a normal typed struct — unsafe_load directly
                                                push!(_accessor_branches, """
                                    if s === :$m_name
                                        r = Ref(getfield(x, :_data))
                                        return GC.@preserve r unsafe_load(Ptr{$m_sanitized}(pointer_from_objref(r) + $m_offset))
                                    end""")
                                                push!(_setter_branches, _blob_store_expr(m_name, m_offset, :typed_struct;
                                                    struct_name=m_sanitized))
                                            end
                                        end
                                    end
                                end
                            end
                        end
                        # C11 anonymous members inject their names into the
                        # enclosing scope — `n.x`, not `n._anon1.x`. A real member
                        # of THIS aggregate always wins (an injected name may never
                        # shadow a declared one), and each injected name is emitted
                        # once.
                        _real_names = Set{String}(String(get(m, "name", ""))
                                                  for m in members if !isnothing(get(m, "name", nothing)))
                        _injected = Set{String}()
                        for m in members
                            get(m, "anonymous_member", false) == true || continue
                            _fname = get(m, "name", nothing)
                            isnothing(_fname) && continue
                            _inner = _sanitize_c_type_name(String(get(m, "julia_type", "")))
                            for _nm in _exposed_member_names(_inner, dwarf_structs, best_dwarf_key)
                                (_nm in _real_names || _nm in _injected) && continue
                                push!(_injected, _nm)
                                push!(_accessor_branches, """
                                    if s === :$_nm
                                        return getproperty(getproperty(x, :$_fname), :$_nm)
                                    end""")
                                # An injected name writes through the member that
                                # declares it: set it on the inner value, then set
                                # that value back. Handled before the byte-write
                                # chain because it does not address `p` at all.
                                push!(_setter_delegates, """
                                    if s === :$_nm
                                        return setproperty(x, :$_fname, setproperty(getproperty(x, :$_fname), :$_nm, v))
                                    end""")
                            end
                        end

                        if !isempty(_accessor_branches)
                            accessor_code = join(_accessor_branches, "\n")
                            push!(struct_chunks, """
                            function Base.getproperty(x::$julia_struct_name, s::Symbol)
                                s === :_data && return getfield(x, :_data)
                            $accessor_code
                                Base.error("type $julia_struct_name has no field \$s")
                            end

                            """)
                            filter!(!isempty, _setter_branches)
                            setter_code = _blob_setter_chunk(julia_struct_name, _setter_branches, _setter_delegates)
                            if !isempty(setter_code)
                                push!(struct_chunks, setter_code)
                                blob_setters_emitted = true
                            end
                        end

                        continue
                    end

                    # NOTE: packed structs (DWARF byte_size < natural aligned size)
                    # need no dedicated branch here — every `!layout_verified`
                    # struct with a positive byte_size already became an opaque
                    # blob (with getproperty accessors) in the branch above, which
                    # subsumes the old size-heuristic packed detection. Reaching
                    # this point means layout_verified == true or byte_size <= 0.

                    member_count = length(members)
                    push!(struct_chunks, """
                    # C struct: $struct_name ($member_count members)
                    struct $julia_struct_name
                    """)

                    current_offset = 0
                    pad_idx = 0

                    for member in members
                        member_name = get(member, "name", "unknown")
                        julia_type = get(member, "julia_type", "Any")
                        
                        offset = _parse_int_or_hex(get(member, "offset", "0"))
                        
                        # Insert padding if needed
                        if offset > current_offset
                            pad_size = offset - current_offset
                            push!(struct_chunks, "    _pad_$(pad_idx)::NTuple{$(pad_size), UInt8}\n")
                            pad_idx += 1
                            current_offset = offset
                        end

                        # Sanitize member name for Julia (replace invalid characters and keywords)
                        sanitized_name = replace(member_name, '$' => '_')
                        sanitized_name = make_c_identifier(sanitized_name)

                        # Sanitize member types that reference other structs with template syntax
                        # Only sanitize custom struct names, not built-in Julia types like NTuple
                        sanitized_type = julia_type

                        # Don't sanitize built-in Julia types (NTuple, Ptr{Cint}, etc.)
                        builtin_types = ["NTuple", "Ptr", "Cint", "Cuint", "Cintptr_t", "Cuintptr_t", "Cdouble", "Cfloat", "Clong", "Culong", "Cshort", "Cushort", "Cchar", "Cuchar", "Culonglong", "Clonglong", "Cvoid", "Csize_t", "Cptrdiff_t", "Cssize_t", "Cwchar_t", "Cstring", "Bool", "UInt8", "Int8", "UInt16", "Int16", "UInt32", "Int32", "UInt64", "Int64", "Float32", "Float64"]
                        is_builtin = any(startswith(julia_type, bt) for bt in builtin_types)

                        if !is_builtin || occursin(r"[<>]", julia_type)
                            if occursin(r"Ptr\{[^}]+\}", julia_type)
                                # Extract type from Ptr{Type} for custom struct references
                                type_match = match(r"Ptr\{([^}]+)\}", julia_type)
                                if !isnothing(type_match)
                                    inner_type = String(type_match.captures[1])
                                    # Only sanitize if inner type is a custom struct (contains template chars)
                                    if occursin(r"[<>]", inner_type)
                                        sanitized_inner = _sanitize_c_type_name(inner_type)
                                            sanitized_type = "Ptr{$sanitized_inner}"
                                    end
                                end
                            elseif occursin(r"[<>]", julia_type)
                                # Direct custom struct reference with template syntax
                                sanitized_type = _sanitize_c_type_name(julia_type)
                            end
                        end

                        # If the field is Ptr{X} and X hasn't been defined yet,
                        # substitute Ptr{Cvoid} to avoid UndefVarError (same ABI size).
                        sanitized_type = _resolve_forward_ptr(sanitized_type, defined_struct_names)

                        push!(struct_chunks, "    $sanitized_name::$sanitized_type\n")
                        
                        # Update current offset
                        member_size = get(member, "size", 0)
                        # If size is 0 (e.g. unknown type), we can't reliably track offset
                        # But typically we rely on the next member's offset to insert padding
                        current_offset += member_size
                    end

                    # Add trailing padding if the struct size from DWARF is larger than the final offset
                    if byte_size > current_offset
                        pad_size = byte_size - current_offset
                        push!(struct_chunks, "    _pad_tail::NTuple{$(pad_size), UInt8}\n")
                    end

                    push!(struct_chunks, """
                    end

                    # Zero-initializer for $julia_struct_name
                    function $julia_struct_name()
                        ref = Ref{$julia_struct_name}()
                        GC.@preserve ref begin
                            ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), Base.unsafe_convert(Ptr{Cvoid}, ref), 0, sizeof($julia_struct_name))
                        end
                        return ref[]
                    end

                    """)

                    # A struct with real named fields still has to honour C11
                    # name injection: `union { ... };` with no member name puts
                    # its members in THIS scope. The synthetic `_anonN` field is
                    # the storage; these branches are the C-visible spelling.
                    # `getfield` fallthrough keeps every real field working, so
                    # this getproperty is purely additive.
                    _inject_branches = String[]
                    _real_names = Set{String}(
                        make_c_identifier(replace(String(get(m, "name", "")), '$' => '_'))
                        for m in members if !isnothing(get(m, "name", nothing)))
                    _injected = Set{String}()
                    for member in members
                        get(member, "anonymous_member", false) == true || continue
                        _fname = get(member, "name", nothing)
                        isnothing(_fname) && continue
                        _fname = make_c_identifier(replace(String(_fname), '$' => '_'))
                        _inner = _sanitize_c_type_name(String(get(member, "julia_type", "")))
                        for _nm in _exposed_member_names(_inner, dwarf_structs, best_dwarf_key)
                            # A declared field of this struct always wins.
                            (_nm in _real_names || _nm in _injected) && continue
                            push!(_injected, _nm)
                            push!(_inject_branches, """
                                s === :$_nm && return getproperty(getfield(x, :$_fname), :$_nm)""")
                        end
                    end
                    if !isempty(_inject_branches)
                        push!(struct_chunks, """
                        # C11 anonymous-member name injection for $julia_struct_name
                        function Base.getproperty(x::$julia_struct_name, s::Symbol)
                        $(join(_inject_branches, "\n"))
                            return getfield(x, s)
                        end

                        """)
                    end

                    # Embedding a type whose region misstates its SysV class makes
                    # THIS struct misstate it too. Members are emitted first, so
                    # one hop per struct closes over the whole chain.
                    for member in members
                        _mt = _sanitize_c_type_name(String(get(member, "julia_type", "")))
                        if _mt in abi_class_risk
                            push!(abi_class_risk, julia_struct_name)
                            break
                        end
                    end

                    # Verified named-field structs become eligible as inline
                    # members of later structs (topological emission order) and
                    # are ABI-exact for by-value crossings.
                    if layout_verified
                        resolved_layouts[julia_struct_name] = (byte_size, struct_max_align)
                    end
                else
                    # Struct found but no members (empty struct or incomplete info)
                    # Use DWARF byte_size if available for correct ABI layout
                    opaque_size = _parse_dwarf_size(struct_info)
                    if opaque_size <= 0; opaque_size = 8; end
                    push!(struct_chunks, """
                    # Opaque struct: $struct_name (no member info, $opaque_size bytes from DWARF)
                    mutable struct $julia_struct_name
                        data::NTuple{$opaque_size, UInt8}
                    end

                    """)
                end
            else
                # No DWARF info available — only safe behind Ptr{}, use pointer-sized placeholder
                push!(struct_chunks, """
                # Opaque struct: $struct_name (no DWARF info — use only via Ptr)
                mutable struct $julia_struct_name
                    data::NTuple{8, UInt8}
                end

                """)
            end
        end

        push!(struct_chunks, "\n")
    end

    # =============================================================================
    # GLOBAL VARIABLES GENERATION
    # =============================================================================
    
    global_vars = get(metadata, "globals", Dict())
    func_chunks = String[]

    if !isempty(global_vars)
        push!(func_chunks, """
        # =============================================================================
        # Global Variables
        # =============================================================================

        """)

        for (var_name, var_info) in global_vars
            julia_type = get(var_info, "julia_type", "Any")
            # Sanitize name
            safe_name = make_c_identifier(var_name)

            # A global whose type didn't resolve cannot get a value getter:
            # `cglobal(..., Any)` + unsafe_load reads the memory as a boxed
            # Julia object pointer — garbage or a crash. Same for type names
            # that aren't clean Julia type expressions. Emit only the raw
            # pointer accessor for those.
            _type_usable = julia_type != "Any" && julia_type != "_UnsafeUnknown" &&
                           !occursin(r"[^A-Za-z0-9_{}, ]", julia_type)

            if _type_usable
                push!(func_chunks, """
                \"""
                    $safe_name()

                Get value of global variable `$var_name`.
                \"""
                function $safe_name()::$julia_type
                    ptr = cglobal((:$var_name, LIBRARY_PATH), $julia_type)
                    return unsafe_load(ptr)
                end

                \"""
                    $(safe_name)_ptr()

                Get pointer to global variable `$var_name`.
                \"""
                function $(safe_name)_ptr()::Ptr{$julia_type}
                    return cglobal((:$var_name, LIBRARY_PATH), $julia_type)
                end

                """)
                push!(exports, safe_name)
            else
                push!(func_chunks, """
                \"""
                    $(safe_name)_ptr()

                Get pointer to global variable `$var_name` (type unresolved — raw pointer only).
                \"""
                function $(safe_name)_ptr()::Ptr{Cvoid}
                    return cglobal((:$var_name, LIBRARY_PATH))
                end

                """)
            end

            push!(exports, "$(safe_name)_ptr")
        end

        push!(func_chunks, "\n")
    end



    # Track exported function names
    # exports already initialized at top

    # Every type name the module will actually BIND, read back off the chunks
    # already emitted (enums + structs + forward declarations). Signature types
    # are resolved against this below: a `Ptr{X}` whose X is never declared is
    # an UndefVarError at module include, which kills the ENTIRE wrapper, not
    # just the one function.
    #
    # This is not hypothetical bookkeeping — `_INTERNAL_TYPE_BLOCKLIST`
    # deliberately suppresses the DECLARATION of system types that leak through
    # DWARF (`_IO_FILE` et al.), while nothing suppressed their USES, so any
    # library with a `FILE*` in its API generated an unloadable module
    # (miniaudio's ma_fopen/ma_wfopen, 2026-08-02). Degrading the pointer to
    # `Ptr{Cvoid}` is ABI-identical and keeps the function callable.
    emitted_type_names = _defined_type_names(join(enum_chunks) * join(struct_chunks))

    # Union accessors were built during struct emission and screened against the
    # INTENDED struct set (`known_struct_names`). A type that was planned but
    # then skipped — dedup, blocklist, a DWARF-key collision — leaves an
    # accessor naming something the module never binds. Same class as the
    # signature fix, different path: miniaudio's `get_..._memoryStream` returned
    # `ma_dr_flac__memory_stream`, which no chunk defines. Drop those; the
    # member stays reachable through the parent struct's bytes.
    filter!(union_accessor_chunks) do chunk
        used = String[]
        for re in (r"::([A-Za-z_][A-Za-z0-9_]*)", r"Ptr\{([A-Za-z_][A-Za-z0-9_]*)\}")
            for m in eachmatch(re, chunk)
                push!(used, m.captures[1])
            end
        end
        all(t -> t in emitted_type_names || t in _JULIA_BUILTIN_TYPES, used)
    end

    for func in functions
        func_name = func["name"]
        mangled = func["mangled"]
        demangled = func["demangled"]

        # =========================================================
        # TIERED DISPATCH DECISION
        # =========================================================
        
        # BUG FIX: Make copies to allow modification (injecting 'this', refining types) without affecting metadata
        params = copy(func["parameters"])
        return_type = copy(func["return_type"])
        is_vararg = get(func, "is_vararg", false)

        # BUG FIX: Refine return types (Ptr{Cvoid} -> Ptr{Struct})
        # If c_type indicates a pointer to a known struct, but julia_type is generic Ptr{Cvoid}, fix it.
        c_ret = get(return_type, "c_type", "")
        j_ret = get(return_type, "julia_type", "")
        
        if (j_ret == "Ptr{Cvoid}" || j_ret == "Any") && endswith(c_ret, "*")
             base_c = strip(c_ret[1:end-1])
             # Remove const/struct/class prefixes
             base_c = replace(base_c, r"^(const\s+|struct\s+|class\s+)+" => "")
             base_c = strip(base_c)
             
             if base_c in struct_types && !(base_c in _INTERNAL_TYPE_BLOCKLIST)
                 # Sanitize
                 safe_base = _sanitize_c_type_name(base_c)
                 return_type["julia_type"] = "Ptr{$safe_base}"
             end
        end

        # Replace blocklisted internal types in return type (e.g. Ptr{_IO_FILE} -> Ptr{Cvoid})
        let rm = match(r"^(Ref|Ptr)\{(.*)\}$", get(return_type, "julia_type", "Cvoid"))
            if rm !== nothing && String(rm.captures[2]) in _INTERNAL_TYPE_BLOCKLIST
                return_type["julia_type"] = "Ptr{Cvoid}"
            end
        end

        # BUG FIX: Sanitize return types that are template instantiations (e.g. Box<double> -> Box_double)
        # This handles by-value returns of template types
        ret_type_str = get(return_type, "julia_type", "Cvoid")
        if occursin(r"[<>]", ret_type_str) && !startswith(ret_type_str, "Ptr{") && !startswith(ret_type_str, "Ref{")
            safe_ret = _sanitize_c_type_name(ret_type_str)
            return_type["julia_type"] = safe_ret
        end

        # =========================================================
        # VARARGS INTERCEPTION
        # =========================================================
        if is_vararg
            # Sanitize function name for Julia (same logic as below)
            va_julia_name = func_name
            va_julia_name = replace(va_julia_name, "::" => "_")
            va_julia_name = replace(va_julia_name, "<" => "_")
            va_julia_name = replace(va_julia_name, ">" => "_")
            va_julia_name = replace(va_julia_name, "," => "_")
            va_julia_name = replace(va_julia_name, " " => "_")
            va_julia_name = replace(va_julia_name, "(" => "_")
            va_julia_name = replace(va_julia_name, ")" => "")
            va_julia_name = replace(va_julia_name, "&" => "ref")
            va_julia_name = replace(va_julia_name, "[" => "_")
            va_julia_name = replace(va_julia_name, "]" => "")
            va_julia_name = replace(va_julia_name, ":" => "_")
            va_julia_name = replace(va_julia_name, r"_+" => "_")
            va_julia_name = String(rstrip(va_julia_name, '_'))

            overloads = get(config.wrap.varargs_overloads, func_name, Vector{Vector{String}}())
            if isempty(overloads)
                @warn "Varargs function '$func_name' has no overloads configured in [wrap.varargs]. Generating base wrapper only."
            end

            va_code, va_exports = generate_vararg_wrappers(
                func_name, mangled, va_julia_name,
                params, return_type, overloads,
                generate_docs, demangled, :c,
                cstring_free=get(config.wrap.cstring_owned, func_name, "")
            )
            push!(func_chunks, va_code)
            append!(exports, va_exports)
            continue  # Skip normal wrapper generation
        end

        # Build parameter list with ergonomic types
        param_names = String[]
        param_types = String[]  # C types for ccall
        julia_param_types = String[]  # Julia types for function signature (may differ)
        needs_conversion = Bool[]

        for (i, param) in enumerate(params)
            # Sanitize parameter name (e.g., avoid 'end', 'function' keywords)
            safe_name = make_c_identifier(param["name"])
            # Ensure unique parameter names (duplicate names cause Julia syntax errors)
            if safe_name in param_names
                safe_name = "$(safe_name)_$(i)"
            end
            push!(param_names, safe_name)
            julia_type = param["julia_type"]

            # Replace blocklisted internal types (e.g. _IO_FILE) with Cvoid
            let bm = match(r"^(Ref|Ptr)\{(.*)\}$", julia_type)
                if bm !== nothing && String(bm.captures[2]) in _INTERNAL_TYPE_BLOCKLIST
                    julia_type = "Ptr{Cvoid}"
                end
            end

            # Sanitize C++ template types in parameter julia_type (e.g. "Ref{allocator<int> >}")
            if occursin(r"[<>]", julia_type)
                m = match(r"^(Ref|Ptr)\{(.*)\}$", julia_type)
                if m !== nothing
                    wrapper_kw, inner = m.captures
                    inner_safe = _sanitize_c_type_name(inner)
                    julia_type = "$wrapper_kw{$inner_safe}"
                else
                    julia_type = _sanitize_c_type_name(julia_type)
                end
            end
            # Also sanitize namespace separators (::) and other non-identifier chars
            # that can appear in Ptr{namespace::type} style references from DWARF
            # Allow spaces (e.g. "NTuple{4, UInt8}") to pass through unsanitized
            if occursin(r"::|[^A-Za-z0-9_{},\[\] ]", julia_type) && julia_type != "Any"
                m = match(r"^(Ref|Ptr)\{(.*)\}$", julia_type)
                if m !== nothing
                    wrapper_kw, inner = m.captures
                    inner_safe = _sanitize_c_type_name(inner)
                    julia_type = isempty(inner_safe) ? "Ptr{Cvoid}" : "$wrapper_kw{$inner_safe}"
                else
                    julia_type = _sanitize_c_type_name(julia_type)
                end
            end
            c_type_name = get(param, "c_type", "")

            # Determine the actual C type for ccall
            # If julia_type is "Any" but c_type has a name, it's likely a struct
            actual_c_type = if julia_type == "Any" && !isempty(c_type_name) && c_type_name in struct_types
                c_type_name  # Use the struct name directly
            else
                julia_type
            end

            # Degrade Ptr{X} to Ptr{Cvoid} when X is never declared in this
            # module (blocklisted system type, or any name that reached a
            # signature without reaching the struct emitters). Same ABI.
            actual_c_type = _resolve_forward_ptr(actual_c_type, emitted_type_names)

            push!(param_types, actual_c_type)

            # Map C integer types to natural Julia types with range checking
            if actual_c_type in ("Cint", "Cuint", "Clong", "Culong", "Cshort", "Cushort", "Clonglong", "Culonglong", "Csize_t", "Cssize_t")
                push!(julia_param_types, "Integer")
                push!(needs_conversion, true)
            elseif actual_c_type in ("Cfloat", "Cdouble")
                # Same ergonomics as the integer path: accept any Real
                # (1/60 is Float64, literals are often Int) and convert.
                push!(julia_param_types, "Real")
                push!(needs_conversion, true)
            elseif startswith(actual_c_type, "Ptr{")
                # Relax pointer types to Any to allow Managed wrappers via Base.unsafe_convert
                push!(julia_param_types, "Any")
                push!(needs_conversion, false)
            else
                push!(julia_param_types, actual_c_type)
                push!(needs_conversion, false)
            end
        end

        # Julia function name (avoid conflicts and sanitize)
        julia_name = func_name

        # Sanitize function name - remove invalid characters
        julia_name = replace(julia_name, "::" => "_")
        julia_name = replace(julia_name, "<" => "_")
        julia_name = replace(julia_name, ">" => "_")
        julia_name = replace(julia_name, "," => "_")
        julia_name = replace(julia_name, " " => "_")
        julia_name = replace(julia_name, "+" => "plus")
        julia_name = replace(julia_name, "=" => "assign")
        julia_name = replace(julia_name, "-" => "minus")
        julia_name = replace(julia_name, "*" => "mul")
        julia_name = replace(julia_name, "/" => "div")
        julia_name = replace(julia_name, "(" => "_")
        julia_name = replace(julia_name, ")" => "")
        julia_name = replace(julia_name, "&" => "ref")
        julia_name = replace(julia_name, "[" => "_")
        julia_name = replace(julia_name, "]" => "")
        julia_name = replace(julia_name, ":" => "_")
        julia_name = replace(julia_name, r"_+" => "_")  # collapse consecutive underscores
        julia_name = replace(julia_name, r"^replibuild_shim_" => "") # Remove macro shim prefix
        julia_name = String(rstrip(julia_name, '_'))

        # Build function signature using ergonomic Julia types
        param_sig_parts = String[]
        for (name, typ) in zip(param_names, julia_param_types)
            if name == "varargs..."
                push!(param_sig_parts, name)
            else
                push!(param_sig_parts, "$name::$typ")
            end
        end
        param_sig = join(param_sig_parts, ", ")

        # Documentation
        doc_comment = ""
        if generate_docs
            # Build detailed argument documentation, detecting function pointers
            arg_docs = String[]
            callback_docs = String[]

            for (i, param) in enumerate(params)
                name = param["name"]
                c_type_name = get(param, "c_type", "")
                julia_type = param["julia_type"]

                # Check if this is a callback parameter (Ptr{Cvoid} from function pointer)
                if julia_type == "Ptr{Cvoid}" && (startswith(c_type_name, "function_ptr") || contains(c_type_name, "(*"))
                    # Try to resolve the actual signature from typedef or DWARF
                    fp_sig = nothing

                    # First: Check if we have a direct function pointer signature from DWARF
                    if haskey(param, "function_pointer_signature")
                        fp_sig = param["function_pointer_signature"]
                    end

                    # Second: Try to match against header typedef signatures (better than DWARF incomplete sigs)
                    # Even if we have a DWARF signature, prefer typedef if available
                    if haskey(metadata, "function_pointer_typedefs")
                        # Try to find the best matching typedef
                        # Strategy: Match by parameter name similarity or function name
                        best_typedef = nothing
                        best_score = 0

                        for (typedef_name, typedef_info) in metadata["function_pointer_typedefs"]
                            score = 0

                            # Check if typedef name is contained in parameter name or vice versa
                            # e.g., "TransformCallback" in "callback" or "transform" in "TransformCallback"
                            typedef_lower = lowercase(typedef_name)
                            param_lower = lowercase(name)
                            func_lower = lowercase(func_name)

                            if contains(typedef_lower, param_lower) || contains(param_lower, typedef_lower)
                                score += 10
                            end

                            # Check if typedef name is related to function name
                            # e.g., "Transform" in "register_transform_callback"
                            typedef_base = replace(typedef_lower, "callback" => "")
                            if contains(func_lower, typedef_base)
                                score += 20
                            end

                            if score > best_score
                                best_score = score
                                best_typedef = (typedef_name, typedef_info)
                            end
                        end

                        # Only a positive name-based match may override the DWARF
                        # signature. The old "first typedef in the table" fallback
                        # documented an arbitrary signature — a wrong @cfunction
                        # example is worse than none (users build crashing callbacks
                        # from it). No match → keep DWARF's fp_sig or say unknown.
                        if !isnothing(best_typedef)
                            typedef_name, typedef_info = best_typedef
                            ret_type = typedef_info["return_type"]
                            params_list = typedef_info["parameters"]

                            # Construct DWARF-style signature for parsing
                            if isempty(params_list)
                                fp_sig = "function_ptr($ret_type)"
                            else
                                fp_sig = "function_ptr($ret_type; $(join(params_list, ", ")))"
                            end
                        end
                    end

                    # Generate documentation if we have a signature
                    if !isnothing(fp_sig)
                        julia_sig = parse_function_pointer_signature(fp_sig, registry)

                        if !isnothing(julia_sig)
                            push!(arg_docs, "- `$(name)::Ptr{Cvoid}` - Callback function")
                            push!(callback_docs, """
                                **Callback `$name`**: Create using `@cfunction`
                                ```julia
                                callback = @cfunction($julia_sig) Ptr{Cvoid}
                                ```""")
                        else
                            # Fallback if parsing fails
                            push!(arg_docs, "- `$(name)::$(julia_type)` - Callback function (signature: $fp_sig)")
                        end
                    else
                        # No signature info available
                        push!(arg_docs, "- `$(name)::$(julia_type)` - Callback function (signature unknown)")
                    end
                else
                    # Regular parameter
                    push!(arg_docs, "- `$(name)::$(julia_type)`")
                end
            end

            # Build the docstring. Struct returns carry julia_type "Any" in the
            # metadata even though codegen resolves the concrete struct — show
            # the resolved name, not the sentinel.
            doc_ret = String(return_type["julia_type"])
            # Cstring returns are copied to String (NULL → nothing) — document
            # what the wrapper actually returns, not the C-level type.
            doc_ret == "Cstring" && (doc_ret = "Union{String,Nothing}")
            if doc_ret == "Any"
                _doc_rct = String(strip(replace(String(get(return_type, "c_type", "")), r"\bconst\b" => "")))
                if !isempty(_doc_rct) && (_doc_rct in struct_types || haskey(dwarf_structs, _doc_rct))
                    doc_ret = _sanitize_c_type_name(_doc_rct)
                end
            end
            doc_parts = """
            \"\"\"
                $julia_name($param_sig) -> $doc_ret

            Wrapper for `$demangled`

            # Arguments
            $(join(arg_docs, "\n"))

            # Returns
            - `$doc_ret`"""

            # Add callback documentation if any
            if !isempty(callback_docs)
                doc_parts *= "\n\n            # Callback Signatures\n"
                doc_parts *= join(callback_docs, "\n\n")
            end

            doc_parts *= """


            # Metadata
            - Mangled symbol: `$mangled`
            \"\"\"
            """

            doc_comment = doc_parts
        end

        # =========================================================
        # PER-FUNCTION C SAFETY DECISION
        # =========================================================
        # C-specific gate: only packed struct returns and union-by-value
        # returns are unsafe.  Unlike C++ (which routes to MLIR), C thunks
        # are compiled by the same Clang that built the library — no LLVM
        # version mismatch, no sanitizing needed.  DAG is not wired to the
        # C path — Julia's internal LLVM handles the C ABI natively.
        c_safe = is_c_lto_safe(func, dwarf_structs)

        # SysV register-class guard: an opaque byte-blob struct ≤16 bytes with
        # float members has unknowable register classes from the byte image
        # (the real struct travels in SSE eightbytes, NTuple{N,UInt8} claims
        # INTEGER). A by-value crossing would silently read/write the wrong
        # registers — refuse loudly instead. Pointer crossings and >16-byte
        # blobs (MEMORY class on both views) remain fine.
        blob_abi_offenders = String[]
        let _rjt = String(get(return_type, "julia_type", "Any")),
            _rct = String(strip(replace(String(get(return_type, "c_type", "void")), r"\bconst\b" => "")))
            _rname = _rjt == "Any" ? _sanitize_c_type_name(_rct) : _rjt
            if _rname in blob_struct_names && 1 <= get(blob_struct_sizes, _rname, 0) <= 16 &&
               get(blob_float_risk, _rname, true)
                # A ≤16B struct with an unaligned member is MEMORY class — BRANCH 0
                # returns it through an explicit sret buffer, which is exact even
                # for a blob. Only register-class (aligned) returns are unfixable.
                _runaligned = false
                _rinfo = get(dwarf_structs, _rct, nothing)
                if _rinfo !== nothing
                    for _m in get(_rinfo, "members", [])
                        haskey(_m, "bit_size") && continue
                        _msz = get(_m, "size", 0)
                        _msz <= 0 && continue
                        _mal = _msz > 8 ? 8 : _msz
                        if _parse_int_or_hex(get(_m, "offset", "0")) % _mal != 0
                            _runaligned = true
                            break
                        end
                    end
                end
                if !_runaligned
                    push!(blob_abi_offenders, "returns `$_rname` by value")
                end
            end
            # Same refusal for a named-field struct that carries an opaque region
            # of mixed SysV class (see `abi_class_risk`): it has real fields, so
            # the blob test above never sees it, but its register class is just
            # as unknowable from the Julia side.
            if _rname in abi_class_risk && 1 <= get(resolved_layouts, _rname, (0, 0))[1] <= 16
                push!(blob_abi_offenders, "returns `$_rname` by value")
            end
        end
        # A ≤16B blob param with a misaligned (packed) member is MEMORY class in
        # the real SysV ABI — passed on the stack — while its NTuple byte image
        # classifies INTEGER and travels in registers. That mismatch corrupts
        # arguments silently even with no float member, so trap it alongside
        # the SSE-class (float-risk) case. Aligned all-integer blobs classify
        # INTEGER on both views and stay callable; >16B blobs are MEMORY on
        # both views and stay callable.
        _blob_param_misaligned = function(jname::String)
            raw = get(best_dwarf_key, jname, jname)
            info = get(dwarf_structs, raw, nothing)
            info === nothing && return false
            for _m in get(info, "members", [])
                haskey(_m, "bit_size") && continue
                _msz = get(_m, "size", 0)
                _msz <= 0 && continue
                _mal = _msz > 8 ? 8 : _msz
                if _parse_int_or_hex(get(_m, "offset", "0")) % _mal != 0
                    return true
                end
            end
            return false
        end
        for _pt in param_types
            if _pt in blob_struct_names && 1 <= get(blob_struct_sizes, _pt, 0) <= 16 &&
               (get(blob_float_risk, _pt, true) || _blob_param_misaligned(_pt))
                push!(blob_abi_offenders, "takes `$_pt` by value")
            elseif _pt in abi_class_risk && 1 <= get(resolved_layouts, _pt, (0, 0))[1] <= 16
                push!(blob_abi_offenders, "takes `$_pt` by value")
            end
        end

        # =========================================================
        # BRANCH 0: MEMORY-CLASS RETURN — explicit-sret ccall
        # =========================================================
        # SysV classifies an aggregate return as MEMORY only when it has an
        # unaligned non-bitfield field (true `__attribute__((packed))`) or
        # is larger than 16 bytes. MEMORY-class returns require the caller
        # to allocate storage and pass its address as a hidden first arg —
        # we emit the ccall with that exact shape (`Cvoid` return,
        # `Ptr{Ret}` first arg) so it matches the SysV sret lowering of
        # `Ret f(args...)`. No C thunk needed.
        #
        # Bitfield-only structs and aligned aggregates ≤ 16 bytes are
        # INTEGER class (returned in registers) and fall through to
        # BRANCH 2's normal ccall.
        if !c_safe && isempty(blob_abi_offenders)
            sret_c_ret = get(return_type, "c_type", "void")
            cleaned_c_ret = String(strip(replace(sret_c_ret, r"\bconst\b" => "")))
            ret_struct_info = get(dwarf_structs, cleaned_c_ret, nothing)
            needs_sret = false
            if ret_struct_info !== nothing
                rs_size = _parse_dwarf_size(ret_struct_info)
                if rs_size > 16
                    needs_sret = true
                else
                    for m in get(ret_struct_info, "members", [])
                        haskey(m, "bit_size") && continue
                        m_size = get(m, "size", 0)
                        m_size <= 0 && continue
                        m_align = m_size > 8 ? 8 : m_size
                        m_off = _parse_int_or_hex(get(m, "offset", "0"))
                        if m_off % m_align != 0
                            needs_sret = true
                            break
                        end
                    end
                end
            end

            if needs_sret
                sret_ret_type = return_type["julia_type"]
                if sret_ret_type == "Any" && sret_c_ret != "void"
                    if haskey(dwarf_structs, sret_c_ret)
                        sret_ret_type = _sanitize_c_type_name(sret_c_ret)
                    else
                        matched_key = _fuzzy_dwarf_lookup(sret_c_ret, dwarf_structs)
                        if matched_key !== nothing
                            sret_ret_type = _sanitize_c_type_name(matched_key)
                        end
                    end
                end

                sret_ccall_types = "(Ptr{$sret_ret_type}, $(join(param_types, ", "))$(isempty(param_types) ? "" : ","))"
                sret_ccall_args  = "ret_buf, $(join(param_names, ", "))"

                func_def = """
                $doc_comment
                function $julia_name($param_sig)
                    ret_buf = Ref($sret_ret_type())
                    GC.@preserve ret_buf begin
                        ccall((:$mangled, LIBRARY_PATH), Cvoid, $sret_ccall_types, $sret_ccall_args)
                    end
                    return ret_buf[]
                end

                """

                push!(func_chunks, func_def)
                push!(exports, julia_name)
                continue
            end
            # else: fall through to BRANCH 2 — INTEGER-class register return
        end

        # =========================================================
        # BRANCH 2: CCALL (Fast Path) - Existing Logic
        # =========================================================

        # Build conversion logic for parameters
        conversion_code = ""
        ccall_param_names = String[]

        for (i, (name, c_type, needs_conv)) in enumerate(zip(param_names, param_types, needs_conversion))
            if needs_conv
                converted_name = "$(name)_c"
                push!(ccall_param_names, converted_name)

                # Generate range-checked conversion
                conversion_code *= "    $converted_name = $c_type($name)\n"
            else
                push!(ccall_param_names, name)
            end
        end

        ccall_args = join(ccall_param_names, ", ")

        # ccall needs tuple expression (Type1, Type2) not Tuple{Type1, Type2}
        ccall_types = if isempty(param_types)
            "()"
        else
            "($(join(param_types, ", ")),)"  # Note: trailing comma for single-element tuples
        end

        # llvmcall needs Tuple{Type1, Type2} — built from raw param_types (no trailing comma)
        # CRITICAL: Ref{T} in Julia maps to ptr addrspace(10) in LLVM, but C IR uses
        # plain ptr (addrspace 0). For llvmcall we must use Ptr{T} and convert explicitly.
        llvmcall_param_types = String[]
        llvmcall_arg_names = String[]
        llvmcall_ref_args = String[]  # args that need GC.@preserve
        llvmcall_conversion_lines = String[]
        for (i, pt) in enumerate(param_types)
            arg_name = ccall_param_names[i]
            m_ref = match(r"^Ref\{(.+)\}$", pt)
            m_ptr = match(r"^Ptr\{(.+)\}$", pt)
            if m_ref !== nothing
                # Ref{T} → Ptr{T}: llvmcall sees Ref as ptr addrspace(10),
                # but C IR uses plain ptr (addrspace 0)
                inner_type = m_ref.captures[1]
                push!(llvmcall_param_types, "Ptr{$inner_type}")
                ptr_name = "__ptr_$(arg_name)"
                push!(llvmcall_arg_names, ptr_name)
                push!(llvmcall_ref_args, arg_name)
                push!(llvmcall_conversion_lines, "        $ptr_name = Base.unsafe_convert(Ptr{$inner_type}, $arg_name)")
            elseif m_ptr !== nothing
                # Ptr{T} params: the Julia signature is relaxed to ::Any,
                # so callers may pass Ref{T} or Vector{T}. ccall auto-converts
                # via cconvert/unsafe_convert, but llvmcall doesn't.
                # Mirror ccall's conversion chain here.
                push!(llvmcall_param_types, pt)
                ptr_name = "__ptr_$(arg_name)"
                cc_name = "__cc_$(arg_name)"
                push!(llvmcall_arg_names, ptr_name)
                push!(llvmcall_ref_args, cc_name)  # preserve cconvert result (holds the GC root)
                push!(llvmcall_conversion_lines, "        $cc_name = Base.cconvert($pt, $arg_name)")
                push!(llvmcall_conversion_lines, "        $ptr_name = Base.unsafe_convert($pt, $cc_name)")
            else
                push!(llvmcall_param_types, pt)
                push!(llvmcall_arg_names, arg_name)
            end
        end
        llvmcall_types = isempty(llvmcall_param_types) ? "" : join(llvmcall_param_types, ", ")
        llvmcall_args = join(llvmcall_arg_names, ", ")
        has_ref_params = !isempty(llvmcall_ref_args)

        # Generate function body based on return type and conversions
        julia_return_type = return_type["julia_type"]
        c_return_type = return_type["c_type"]

        # A returned pointer to an undeclared type is the same load-blocking
        # class as a parameter one (see emitted_type_names above).
        julia_return_type = _resolve_forward_ptr(julia_return_type, emitted_type_names)

        # Check if return type is a struct (not primitive, not pointer)
        is_struct_return = julia_return_type == "Any" && !contains(c_return_type, "*") && !contains(c_return_type, "void") && c_return_type != "unknown"
        
        # Also detect resolved struct returns: julia_return_type was mapped to a
        # known struct name (e.g. Matrix_double_minus_1...) but is_struct_return
        # missed it because julia_return_type != "Any".
        returns_known_struct = is_struct_return ||
            haskey(julia_to_cpp_struct, julia_return_type) ||
            c_return_type in struct_types

        # Build the llvmcall expression with proper Ref{T} → Ptr{T} handling.
        # ccall handles Ref{T} → C pointer automatically, but llvmcall sees Ref{T}
        # as ptr addrspace(10) while C IR uses plain ptr (addrspace 0).
        function _wrap_ffi_callsite(call_expr, indent)
            if !has_ref_params
                return "$(indent)return $call_expr"
            end
            lines = String[]
            for l in llvmcall_conversion_lines
                push!(lines, "$indent$l")
            end
            preserve_list = join(llvmcall_ref_args, " ")
            push!(lines, "$(indent)GC.@preserve $preserve_list begin")
            push!(lines, "$(indent)    return $call_expr")
            push!(lines, "$(indent)end")
            return join(lines, "\n")
        end
        function _build_llvmcall_expr(ret_type_str, indent="        "; ir_src::String="LTO_IR")
            call_expr = "Base.llvmcall(($ir_src, \"$mangled\"), $ret_type_str, Tuple{$llvmcall_types}, $llvmcall_args)"
            return _wrap_ffi_callsite(call_expr, indent)
        end
        # Tier-1 call sites route through the per-function @generated kernel
        # (ccall-vs-llvmcall decided at generation time); same conversion and
        # GC.@preserve discipline as a direct llvmcall.
        _build_kernel_call(kernel, indent="        ") =
            _wrap_ffi_callsite("$kernel($llvmcall_args)", indent)

        # llvmcall shape gate: safe for primitive/pointer args only.
        # Struct-by-value params are excluded because Base.llvmcall doesn't
        # reliably map Julia struct layouts to LLVM aggregate types.
        #
        # `Bool` is excluded for a narrower but just as fatal reason: Julia
        # passes a Bool across the llvmcall boundary as **i8**, while clang
        # emits a C `_Bool` parameter as **i1 zeroext**. The slice therefore
        # declares `i1` and llvmcall refuses the call with
        #     Malformed llvmcall: argument N type i1 does not match
        #     expected argument type i8
        # `Base.llvmcall` resolves at CODEGEN — i.e. on the FIRST CALL — so this
        # never surfaces at wrap time or module load, only when a user calls the
        # function. ccall converts Bool→i1 correctly, so Tier 3 is right here;
        # per the standing doctrine (llvmcall is a passenger tier, any doubt
        # resolves to ccall) these demote rather than getting a conversion shim.
        # Found 2026-08-01 via mpack_write_bool; latent in blake3
        # (blake3_hash_many) and box2d3 too.
        lto_shape_ok = !returns_known_struct &&
            julia_return_type != "Cstring" &&
            julia_return_type != "Bool" &&
            !any(t -> t == "Cstring", param_types) &&
            !any(t -> t == "Bool", param_types) &&
            !any(t -> t in struct_types, param_types)

        # Whole-module LTO path (legacy, scale-limited — see CLAUDE.md caveats)
        lto_eligible = config.link.enable_lto && lto_shape_ok

        # Tier 1 sliced llvmcall: the pre-pass produced a verified slice AND
        # the ABI shape gates pass AND the C safety gate passes. Takes
        # precedence over the whole-module path.
        tier1_this = tier1_slices !== nothing && haskey(tier1_slices, mangled) &&
            lto_shape_ok && c_safe

        # Check for _UnsafeUnknown trap to prevent segfaults
        has_unknown_param = any(t -> t == "_UnsafeUnknown", param_types)
        is_unknown_return = julia_return_type == "_UnsafeUnknown"

        if !isempty(blob_abi_offenders)
            offender_list = join(blob_abi_offenders, "; ")
            func_def = """
            $doc_comment
            function $julia_name($param_sig)
                Base.error(\"\"\"
                ABI Safety Trap: cannot call '$julia_name' through ccall.
                This function crosses an opaque byte-blob struct by value inside
                the SysV register window (≤16 bytes, float-bearing or packed): $offender_list.
                A byte blob cannot reproduce the struct's real argument classes
                (SSE eightbytes, or MEMORY-class stack passing for packed layouts),
                so the call would silently corrupt data. The struct is opaque because
                its layout could not be reproduced exactly (packed layout, bitfields,
                or unresolvable member types). Use a pointer-taking variant of this
                C function, or make the struct's members resolvable.
                \"\"\")
            end

            """
        elseif has_unknown_param || is_unknown_return
            func_def = """
            $doc_comment
            function $julia_name($param_sig)
                Base.error(\"\"\"
                FFI Safety Trap: Cannot call function '$julia_name'.
                One or more types could not be mapped to Julia safely:
                Return type: $julia_return_type (C: $c_return_type)
                Parameter types: $(join(param_types, ", "))
                
                To fix this, add the missing type mapping to your replibuild.toml.
                Calling this function would have caused a segmentation fault.
                \"\"\")
            end

            """
        elseif is_struct_return
            # Struct-valued return - Julia uses the struct type directly
            # ccall will handle struct returns automatically if the Julia type matches
            
            # Sanitize C return type if it contains template chars (Box<int> -> Box_int)
            safe_c_ret = c_return_type
            if occursin(r"[<>]", safe_c_ret)
                 safe_c_ret = _sanitize_c_type_name(safe_c_ret)
            end

            if lto_eligible
                llvmcall_body = _build_llvmcall_expr(safe_c_ret)
                func_def = """
                $doc_comment
                function $julia_name($param_sig)::$safe_c_ret
                $conversion_code    if !isempty(LTO_IR)
                $llvmcall_body
                    else
                        return ccall((:$mangled, LIBRARY_PATH), $safe_c_ret, $ccall_types, $ccall_args)
                    end
                end

                """
            else
                func_def = """
                $doc_comment
                function $julia_name($param_sig)::$safe_c_ret
                $conversion_code    return ccall((:$mangled, LIBRARY_PATH), $safe_c_ret, $ccall_types, $ccall_args)
                end

                """
            end
        elseif julia_return_type == "Cstring"
            # Cstring return: NULL → nothing, else copy to String; owned buffers
            # ([wrap.cstring_owned]) are freed through the library's own
            # deallocator after the copy. A raw `_ptr` variant is emitted
            # alongside for callers that need the pointer itself (lifetime
            # management, passing along, avoiding the copy).
            cstring_free_sym = get(config.wrap.cstring_owned, func_name, "")
            func_def = """
            $doc_comment
            function $julia_name($param_sig)::Union{String,Nothing}
            $conversion_code    ptr = ccall((:$mangled, LIBRARY_PATH), Cstring, $ccall_types, $ccall_args)
            $(_cstring_policy_lines(cstring_free_sym))
            end

            \"\"\"
                $(julia_name)_ptr($param_sig) -> Cstring

            Raw-pointer variant of `$julia_name`: returns the C `char*` unchanged
            (no copy, no NULL check$(isempty(cstring_free_sym) ? "" : ", NOT freed — caller owns the buffer")).
            \"\"\"
            function $(julia_name)_ptr($param_sig)::Cstring
            $conversion_code    return ccall((:$mangled, LIBRARY_PATH), Cstring, $ccall_types, $ccall_args)
            end

            """
            push!(exports, "$(julia_name)_ptr")
        elseif !isempty(conversion_code)
            # Has parameter conversions
            if tier1_this
                slice_const = _slice_const_name!(tier1_const_owner, mangled)
                kernel = _tier1_kernel_name(slice_const)
                kernel_chunk = _tier1_kernel_chunk(slice_const, kernel, mangled,
                                                   julia_return_type,
                                                   llvmcall_param_types, llvmcall_arg_names)
                call_body = _build_kernel_call(kernel, "    ")
                func_def = """
                $kernel_chunk
                $doc_comment
                function $julia_name($param_sig)::$julia_return_type
                $conversion_code$call_body
                end

                """
            elseif lto_eligible
                llvmcall_body = _build_llvmcall_expr(julia_return_type)
                func_def = """
                $doc_comment
                function $julia_name($param_sig)::$julia_return_type
                $conversion_code    if !isempty(LTO_IR)
                $llvmcall_body
                    else
                        return ccall((:$mangled, LIBRARY_PATH), $julia_return_type, $ccall_types, $ccall_args)
                    end
                end

                """
            else
                func_def = """
                $doc_comment
                function $julia_name($param_sig)::$julia_return_type
                $conversion_code    return ccall((:$mangled, LIBRARY_PATH), $julia_return_type, $ccall_types, $ccall_args)
                end

                """
            end
        else
            # Standard wrapper - no conversions needed
            if tier1_this
                slice_const = _slice_const_name!(tier1_const_owner, mangled)
                kernel = _tier1_kernel_name(slice_const)
                kernel_chunk = _tier1_kernel_chunk(slice_const, kernel, mangled,
                                                   julia_return_type,
                                                   llvmcall_param_types, llvmcall_arg_names)
                call_body = _build_kernel_call(kernel, "    ")
                func_def = """
                $kernel_chunk
                $doc_comment
                function $julia_name($param_sig)::$julia_return_type
                $call_body
                end

                """
            elseif lto_eligible
                llvmcall_body = _build_llvmcall_expr(julia_return_type)
                func_def = """
                $doc_comment
                function $julia_name($param_sig)::$julia_return_type
                    if !isempty(LTO_IR)
                $llvmcall_body
                    else
                        return ccall((:$mangled, LIBRARY_PATH), $julia_return_type, $ccall_types, $ccall_args)
                    end
                end

                """
            else
                func_def = """
                $doc_comment
                function $julia_name($param_sig)::$julia_return_type
                    ccall((:$mangled, LIBRARY_PATH), $julia_return_type, $ccall_types, $ccall_args)
                end

                """
            end
        end

        push!(func_chunks, func_def)
        push!(exports, julia_name)

        # Generate convenience wrappers for ergonomic APIs
        # One type: const primitive array pointers -> accept Vector directly with GC.@preserve
        #
        # Deliberately NOT emitted: struct-by-value overloads for Ptr{Struct} params
        # (f(x::MyStruct) passing Ref(local copy) to the ccall). A C function taking
        # T* may free, mutate-and-retain, or store that pointer; handing it a pointer
        # into a temporary Julia-owned copy is undefined behavior for every such
        # function (crash-proven: cJSON_Delete(::cJSON) → glibc double-free abort).
        # Ownership is not recoverable from DWARF, so no name heuristic can gate this
        # safely. The base wrapper's ::Any params already accept Ref(x)/pointers.

        has_array_ptr_params = false
        array_ptr_indices = Int[]
        convenience_param_types = String[]
        convenience_param_names = String[]

        for (i, (ptype, pname)) in enumerate(zip(param_types, param_names))
            # Non-C-prefixed pointer types (struct pointers, Ptr{Int32}-style aliases)
            # pass through unchanged (no by-value overload — see note above); this arm
            # also keeps them out of the array-pointer heuristic below.
            if startswith(ptype, "Ptr{") && !startswith(ptype, "Ptr{C") && ptype != "Ptr{Cvoid}"
                push!(convenience_param_types, ptype)
                push!(convenience_param_names, pname)
            # Check if this is Ptr{Cdouble}, Ptr{Cfloat}, Ptr{Cint}, etc. (array pointers)
            # These benefit from Vector{T} wrapper with automatic GC.@preserve
            elseif ptype in ["Ptr{Cdouble}", "Ptr{Cfloat}", "Ptr{Cint}", "Ptr{Cuint}",
                             "Ptr{Int32}", "Ptr{Int64}", "Ptr{UInt32}", "Ptr{UInt64}",
                             "Ptr{Float32}", "Ptr{Float64}"]
                # Check if parameter name suggests it's an input array (not output)
                # Common patterns: x, y, data, array, vector, signal, values, input, src
                pname_lower = lowercase(pname)
                is_likely_input = any(pattern -> contains(pname_lower, pattern),
                    ["x", "y", "data", "array", "vec", "signal", "value", "input", "src", "coef"])

                # Also check if it's NOT likely an output
                is_likely_output = any(pattern -> contains(pname_lower, pattern),
                    ["out", "result", "dest", "dst", "buffer"])

                if is_likely_input && !is_likely_output
                    has_array_ptr_params = true
                    push!(array_ptr_indices, i)
                    # Convert Ptr{Cdouble} -> Vector{Float64}, etc.
                    elem_type = replace(ptype, "Ptr{" => "", "}" => "")
                    julia_vec_type = if elem_type == "Cdouble"
                        "Vector{Float64}"
                    elseif elem_type == "Cfloat"
                        "Vector{Float32}"
                    elseif elem_type in ["Cint", "Int32"]
                        "Vector{Int32}"
                    elseif elem_type in ["Cuint", "UInt32"]
                        "Vector{UInt32}"
                    elseif elem_type == "Int64"
                        "Vector{Int64}"
                    elseif elem_type == "UInt64"
                        "Vector{UInt64}"
                    elseif elem_type == "Float64"
                        "Vector{Float64}"
                    elseif elem_type == "Float32"
                        "Vector{Float32}"
                    else
                        ptype  # Fallback to pointer type
                    end
                    push!(convenience_param_types, julia_vec_type)
                    push!(convenience_param_names, pname)
                else
                    push!(convenience_param_types, ptype)
                    push!(convenience_param_names, pname)
                end
            else
                push!(convenience_param_types, ptype)
                push!(convenience_param_names, pname)
            end
        end

        # Generate convenience wrapper if we have array pointer parameters
        if has_array_ptr_params
            convenience_sig = join(["$pname::$ptype" for (pname, ptype) in zip(convenience_param_names, convenience_param_types)], ", ")

            # Resolve the return type the same way the base wrapper does. For a
            # struct-valued return, julia_return_type is the "Any" sentinel — using
            # it as the ccall return type makes Julia treat the returned struct as a
            # boxed object pointer and segfault. Mirror the base wrapper's safe_c_ret.
            convenience_ret_type = julia_return_type
            if is_struct_return
                convenience_ret_type = occursin(r"[<>]", c_return_type) ?
                    _sanitize_c_type_name(c_return_type) : c_return_type
            end

            # Build the ccall arguments
            convenience_ccall_args = String[]
            for (i, pname) in enumerate(param_names)
                if i in array_ptr_indices
                    # Array parameter: use pointer()
                    push!(convenience_ccall_args, "pointer($pname)")
                else
                    # Regular parameter: pass as-is
                    push!(convenience_ccall_args, pname)
                end
            end
            convenience_ccall = join(convenience_ccall_args, ", ")

            # The vectors backing pointer() must stay rooted across the call
            preserve_vars = join([param_names[i] for i in array_ptr_indices], " ")

            if julia_return_type == "Cstring"
                # Match the base wrapper's Cstring policy (NULL → nothing,
                # String copy, [wrap.cstring_owned] free)
                convenience_func = """
                # Convenience wrapper - accepts arrays directly with automatic GC preservation
                function $julia_name($convenience_sig)::Union{String,Nothing}
                    ptr = GC.@preserve $preserve_vars begin
                        ccall((:$mangled, LIBRARY_PATH), Cstring, $ccall_types, $convenience_ccall)
                    end
                $(_cstring_policy_lines(get(config.wrap.cstring_owned, func_name, "")))
                end

                """
            else
                convenience_func = """
                # Convenience wrapper - accepts arrays directly with automatic GC preservation
                function $julia_name($convenience_sig)::$convenience_ret_type
                    return GC.@preserve $preserve_vars begin
                        ccall((:$mangled, LIBRARY_PATH), $convenience_ret_type, $ccall_types, $convenience_ccall)
                    end
                end

                """
            end

            push!(func_chunks, convenience_func)
        end
    end
    # Export statement (unique to handle overloaded functions)
    # Include enum types, enum values, struct types, and functions
    all_exports = copy(exports)

    # Add enum types
    for enum_key in enum_types
        enum_name = replace(enum_key, "__enum__" => "")
        push!(all_exports, enum_name)

        # Add enum constants
        if haskey(dwarf_structs, enum_key)
            enum_info = dwarf_structs[enum_key]
            enumerators = get(enum_info, "enumerators", [])
            for enumerator in enumerators
                enum_const_name = get(enumerator, "name", "")
                if !isempty(enum_const_name)
                    push!(all_exports, enum_const_name)
                end
            end
        end
    end

    # Add struct types (filter internal/compiler types)
    for struct_name in struct_types
        if !(struct_name in enum_names)
            # Sanitize struct name for export
            julia_struct_name = _sanitize_c_type_name(struct_name)
            julia_struct_name = replace(julia_struct_name, "*" => "Ptr")
            julia_struct_name = replace(julia_struct_name, "&" => "Ref")
            julia_struct_name = replace(julia_struct_name, r"[^a-zA-Z0-9_]" => "_")
            # Skip internal/compiler types
            if julia_struct_name in _INTERNAL_TYPE_BLOCKLIST || struct_name in _INTERNAL_TYPE_BLOCKLIST
                continue
            end
            push!(all_exports, julia_struct_name)
        end
    end

    # The bulk setter, once, only if some struct actually got a `setproperty`.
    # Skipped outright if the library already owns the name: our definition
    # takes no type-annotated argument, so it would not merely add a method —
    # it would shadow the library's own `setproperties` for every call. The
    # per-struct `setproperty` methods are safe either way (they dispatch on a
    # concrete blob type, and `_dedup_method_chunks` settles exact collisions).
    if blob_setters_emitted
        if !("setproperties" in all_exports)
            push!(struct_chunks, _blob_setproperties_chunk())
            push!(all_exports, "setproperties")
        else
            @warn "wrap: the library exports `setproperties`; skipping the generated " *
                  "bulk field setter. Use `setproperty(x, :field, v)` per field."
        end
        "setproperty" in all_exports || push!(all_exports, "setproperty")
    end

    export_statement = if !isempty(all_exports)
        "export " * join(unique(all_exports), ", ") * "\n\n"
    else
        ""
    end

    # Footer
    footer = """

    end # module $module_name
    """

    # Unbuffer C stdout so printf output is visible in the Julia REPL.
    # Must be in __init__ (not module body) for precompilation safety.
    # Build-identity check, shared by every __init__ variant below.
    _identity_snippet = """
            _check_build_identity()
    """

    _setvbuf_snippet = """
            # Unbuffer C stdout so printf output appears immediately in the REPL
            let c_stdout = unsafe_load(cglobal(:stdout, Ptr{Cvoid}))
                ccall(:setvbuf, Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Cint, Csize_t), c_stdout, C_NULL, 2, 0)
            end"""

    # Generate initialization block
    init_block = if config.compile.aot_thunks
        """
        # Library handles for manual management if needed
        const LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
        const THUNKS_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

        function __init__()
            # Load main library explicitly to ensure symbols are available
            LIB_HANDLE[] = Libdl.dlopen(LIBRARY_PATH, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)

            # Load AOT thunks library if it was successfully generated
            if !isempty(THUNKS_LIBRARY_PATH) && isfile(THUNKS_LIBRARY_PATH)
                THUNKS_HANDLE[] = Libdl.dlopen(THUNKS_LIBRARY_PATH, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
            elseif $requires_jit
                @warn "AOT Thunks library not found, but advanced FFI features are required. These features will fail at runtime."
            end
        $_identity_snippet$_setvbuf_snippet
        end
        """
    else
        if requires_jit
            """
            function __init__()
                # Initialize the global JIT context with this library's vtables
                RepliBuild.JITManager.initialize_global_jit(LIBRARY_PATH)
            $_identity_snippet$_setvbuf_snippet
            end
            """
        else
            """
            # Library handle for manual management if needed
            const LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

            function __init__()
                # Load library explicitly to ensure symbols are available
                LIB_HANDLE[] = Libdl.dlopen(LIBRARY_PATH, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
            $_identity_snippet$_setvbuf_snippet
            end
            """
        end
    end

    # Same precompilation-safety dedup as the C++ generator: exactly one
    # definition per dispatch signature (last one wins, as include() would).
    func_chunks = _dedup_method_chunks(func_chunks)

    # Slices are written from the FINAL chunks, after dedup — see the docstring.
    tier1_emitted, tier1_declares = tier1_slices === nothing ?
        (String[], Dict{String,Vector{String}}()) :
        _tier1_emit_slices!(config, func_chunks, tier1_slices, tier1_const_owner)

    tier1_registry = """
    # ── Tier 1: sliced llvmcall ───────────────────────────────────────────────
    # Each function named below carries a `_SLICE_*` path const and a
    # `@generated` kernel that picks ccall vs llvmcall ONCE, at generation time.
    # llvmcall is opportunistic — a passenger tier, never the driver — so both
    # doubts resolve to the ccall body, which is why the kernels below look
    # like they distrust their own slice:
    #   * output mode (`jl_generating_output`) — a sliced llvmcall inside a
    #     precompile worker deadlocks the JIT engine lock whenever a `declare`
    #     binds a dlopened library's symbol, so precompilation takes ccall and
    #     the first runtime call regenerates to the slice;
    #   * missing slice file — a wrapper vendored or relocated without its
    #     `slices/` directory stays a working ccall wrapper rather than failing
    #     at load or call time;
    #   * unresolvable declare — see _slice_symbols_resolve below.
    # Empty set ⇒ every function dispatches through ccall/thunk.
    const TIER1_FUNCTIONS = Set{String}($(isempty(tier1_emitted) ? "String[]" : repr(sort(unique(tier1_emitted)))))

    # Symbols each slice binds by `declare`, recorded off the shipped .ll files.
    const TIER1_DECLARES = $(_render_declares_literal(tier1_declares))

    \"\"\"
    Whether every symbol a slice declares can be resolved in this process.

    The wrap-time pre-flight already checked this against the library it was
    generated from — but a wrapper is portable source, and the library beside it
    at RUNTIME may not be that library. A JLL or distro build has none of the
    `__rb_*` promoted statics RepliBuild's own build exports, and an unresolved
    slice declare does not raise: ORC prints `Symbols not found` and then blocks
    FOREVER on the first call. So the check is repeated here, against the
    library actually loaded, before any slice is trusted. A miss silently uses
    the ccall body, which is always correct.
    \"\"\"
    function _slice_symbols_resolve(syms)
        for s in syms
            ccall(:dlsym, Ptr{Cvoid}, (Ptr{Cvoid}, Cstring), C_NULL, s) == C_NULL && return false
        end
        return true
    end

    """

    return join([header, init_block, metadata_section, join(enum_chunks), join(struct_chunks), join(union_accessor_chunks), tier1_registry, export_statement, join(func_chunks), footer])
end

