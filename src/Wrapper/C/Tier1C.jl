# =============================================================================
# Tier 1 — `Base.llvmcall` over per-function bitcode slices (C generator only)
# =============================================================================
#
# EVERYTHING llvmcall-shaped in the C wrapper generator lives in this file, and
# nothing else in `src/Wrapper/C/` knows how a slice is cut, named, pre-flighted
# or emitted. That isolation is the point: Tier 1 is an EXPERIMENTAL side
# project inside RepliBuild — it ships, it works, it is **not** a supported
# tier, and it is not load-bearing. `[wrap.tier1] enable` defaults to false and
# every Hub config leaves it off, so on a default build not one line of this
# file runs.
#
# Split out of GeneratorC.jl (was lines 317-677) on 2026-08-22 for two reasons:
#
#   * DEVELOPMENT — the experimental tier is worked on and broken independently
#     of the generator it hangs off. One file to read, one file to revert.
#   * GATING — with the emission surface in one place, "the toggle is off"
#     means a single guarded call rather than a rule scattered across the
#     generator. Before this, a tier1-off wrapper still carried the whole
#     llvmcall registry: `TIER1_FUNCTIONS` (empty), `TIER1_DECLARES` (empty),
#     an unreachable `_slice_symbols_resolve`, and a 14-line comment block
#     explaining a mechanism the file did not use. See `_tier1_registry_chunk`.
#
# THE DOCTRINE, which every function here obeys: llvmcall is a passenger tier,
# never the driver. Every doubt — output mode, a missing slice file, an
# unresolvable declare, a refused or hazardous slice — resolves to the `ccall`
# body, which is always correct. That is why the emitted kernels read as though
# they distrust their own slices; they do.
#
# Related: `src/IRGen/Slicer.jl` cuts the slices; `Wrapper/Utils.jl`
# `_slice_declared_symbols` reads a slice's declares back off the shipped `.ll`.

"""
    _symbol_resolves_via(handle, sym) -> Bool

Whether `dlsym` finds `sym` through `handle`. A `C_NULL` handle is
`RTLD_DEFAULT` — the whole process — which is what ORC searches when it links
a slice at `llvmcall` time. A real handle scopes the lookup to that library
and its `DT_NEEDED` chain.
"""
function _symbol_resolves_via(handle::Ptr{Cvoid}, sym::AbstractString)
    # `dlsym` is a POSIX libdl export. Windows has no such symbol in the
    # process, so the raw ccall raised `could not load symbol "dlsym"` rather
    # than answering — which took the whole pre-flight down on the one platform
    # whose answer differs from the caller's assumption. Libdl is the portable
    # spelling of the same lookup.
    if handle != C_NULL
        return Libdl.dlsym(handle, sym; throw_error = false) !== nothing
    end

    # A null handle is RTLD_DEFAULT: every globally-loaded library in this
    # process. Windows has no RTLD_DEFAULT — GetProcAddress always takes a
    # module — so the loaded-module list stands in for it. Only the `stray`
    # diagnostic below uses this arm, and only for symbols already known to be
    # missing, so walking the list costs nothing in the common path.
    @static if Sys.iswindows()
        for m in Libdl.dllist()
            h = Libdl.dlopen(m, Libdl.RTLD_LAZY; throw_error = false)
            h === nothing && continue
            Libdl.dlsym(h, sym; throw_error = false) === nothing || return true
        end
        return false
    else
        return ccall(:dlsym, Ptr{Cvoid}, (Ptr{Cvoid}, Cstring), C_NULL, sym) != C_NULL
    end
end

"""
    _lto_unresolved_symbols(lto_ll, lib_path) -> Vector{String}

Symbols the monolithic LTO module `declare`s that nothing reachable from
`lib_path` can supply. Empty ⇒ the module is safe to hand to `Base.llvmcall`.

Monolithic LTO embeds the WHOLE post-LTO module in the wrapper and llvmcall's
it, so every symbol that module declares has to resolve in the consumer's
process the first time a function referencing it is materialised. On ELF they
do: the C runtime is a shared libc, its symbols are in the process, and this
question never had to be asked.

PE does not work that way. mingw links its C runtime statically into every DLL,
so a helper like `snprintf` is a defined symbol in the binary's COFF table and
absent from its export directory — nothing in the process can hand ORC an
address for it. ORC then prints

    JIT session error: Symbols not found: [ snprintf ]

and DEADLOCKS rather than raising, which is why this must be answered before
the wrapper is written and not discovered at the call site.

`_tier1_preflight!` asks exactly this question for per-function slices. The
monolithic path never had an equivalent.
"""
function _lto_unresolved_symbols(lto_ll::String, lib_path::String)::Vector{String}
    isfile(lto_ll) || return String[]

    declared = Set{String}()
    for line in eachline(lto_ll)
        startswith(line, "declare") || continue
        m = match(r"@\"?([A-Za-z0-9_.\$]+)\"?\s*\(", line)
        m === nothing && continue
        name = String(m.captures[1])
        # Intrinsics are lowered by the backend and never looked up.
        startswith(name, "llvm.") && continue
        push!(declared, name)
    end
    isempty(declared) && return String[]

    # RTLD_LOCAL and closed again: the same reasoning as _tier1_preflight!.
    # Loading it globally would leak this library's exports into every later
    # wrap in the session and verify the next package against symbols its
    # consumer will not have.
    handle = try
        Libdl.dlopen(lib_path, Libdl.RTLD_NOW | Libdl.RTLD_LOCAL)
    catch
        return String[]   # cannot verify ⇒ do not demote
    end
    try
        return sort!(filter(s -> !_symbol_resolves_via(handle, s), collect(declared)))
    finally
        try
            Libdl.dlclose(handle)
        catch
        end
    end
end

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


"""
    _tier1_registry_chunk(emitted, declares) -> String

The Tier-1 surface a wrapper needs to run sliced `llvmcall` call sites:
`TIER1_FUNCTIONS`, `TIER1_DECLARES`, and the runtime `_slice_symbols_resolve`
pre-flight.

**Returns `""` when `emitted` is empty — that is the gate.** Before it, a
wrapper generated with `[wrap.tier1] enable = false` still carried the entire
registry: an empty `TIER1_FUNCTIONS`, an empty `TIER1_DECLARES`, a
`_slice_symbols_resolve` nothing could call, and fourteen lines of comment
explaining a mechanism the file did not use. Roughly fifty lines of llvmcall
machinery in a wrapper containing no llvmcall.

Keyed on what was actually EMITTED, not on the config flag, which is the same
discipline as `_tier1_emit_slices!`: acceptance is strictly weaker than
emission (a slice can be accepted and still have no call site that reads it),
so a config-keyed gate could emit a registry for zero kernels. If nothing was
emitted there is nothing to register, whatever the toml says.
"""
function _tier1_registry_chunk(emitted::Vector{String},
                               declares::Dict{String,Vector{String}})::String
    isempty(emitted) && return ""
    return """
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
    const TIER1_FUNCTIONS = Set{String}($(repr(sort(unique(emitted)))))

    # Symbols each slice binds by `declare`, recorded off the shipped .ll files.
    const TIER1_DECLARES = $(_render_declares_literal(declares))

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
end
