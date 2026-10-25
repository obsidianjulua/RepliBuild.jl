#!/usr/bin/env julia
# Slicer.jl — per-function bitcode slicing for the llvmcall LTO path (M2)
#
# A slice is a tiny LLVM module holding exactly ONE definition — the target
# function, verbatim from the post-promotion `<name>_abi.ll` — plus `declare`
# lines for everything the target references. At JIT time the declarations
# resolve against the RTLD_GLOBAL-loaded `.so` (ORC process-symbol lookup), so
# every mutable datum keeps its single copy in the `.so` (static-promotion
# contract, `_promote_statics_libllvm`), and `Base.llvmcall` embeds kilobytes
# instead of the whole linked module (the box2d3-scale segfault class).
#
# That lookup is PROCESS-wide, not library-wide: ORC takes the first definition
# of a name in the process, and the pre-flight only asks whether the library
# has it. So a slice never declares a symbol its own library defines by the
# public name. Promotion gives every exported definition an exported alias
# `__rb_<lib>_<name>` (`_alias_exports!`), and the slice declares that
# (`_claim_rb_aliases!`): a name no other library carries. Julia loads its
# own `libz.so.1` at startup, and the Hub's zlib ran `gzclose` as a slice whose
# `gzclose_w` was Julia's (2026-10-24). A module without the aliases (built
# before the pass, or with Tier 1 off) refuses such slices rather than guess.
#
# Boundary policy per reached GlobalValue:
#   function / global the library    → declare its `__rb_` alias; refuse when
#     defines and exports               the module has none for it
#   function only declared           → declare (libc and friends: the library
#                                      binds them through the process as well)
#   function promoted to `__rb_*`    → declare (already a unique name)
#   mutable global                  → declare (never embed — cJSON class)
#   internal constant, unnamed_addr → embed   (no symbol exists, and only the
#                                     contents are observable, so a second copy
#                                     cannot be told apart from the first)
#   internal constant, addr-signif. → refuse  (embedding would put the slice's
#                                     copy at a DIFFERENT address than the
#                                     .so's; that is the cJSON divergence class
#                                     rotated from value identity onto address
#                                     identity, and just as silent)
#   external constant               → declare (a symbol exists; bind it)
#   alias / ifunc                   → refuse the slice (unbuilt, ledger class)
#   blockaddress ≠ self             → refuse (jump table into a deleted body)
#
# Everything runs on Julia's resident libLLVM via LLVM.jl — same in-process,
# version-locked philosophy as the C build bucket. Every produced slice is
# verified before it is returned; a slice that cannot be produced comes back
# as a refusal with a reason, never as silently-wrong IR. That includes the
# Slicer's own failures: an exception while slicing one target comes back as
# that target's refusal (`errored`), so a Slicer bug costs one function its
# slice and never fails the wrap.
#
# LLVM.jl 9 and 10 both load here. Where a name means different things in the
# two (attribute kinds, `isconstant`, `unnamed_addr`, module asm), the read goes
# through `LLVM.API`, the C API binding, which is the same in both. Two of those
# changes (attribute kinds, `isconstant`) are silent on 10: the old spelling
# still runs and answers a different question.

module Slicer

using LLVM
using SHA
using JSON
import .._read_json

export SliceResult, slice_library, slice_function, sliced

"Bump to invalidate every cached slice (policy/algorithm change)."
# 4: entries are named by `slice_file_stem`; a v3 cache on a case-insensitive
# filesystem can hold one symbol's IR under another's name.
# 5: reads that changed meaning in LLVM.jl 10 go through `LLVM.API`. A v4 entry
# written on LLVM.jl 10 can lack its `:setjmp_family` hazard, which would let a
# setjmp closure through to llvmcall, and on LLVM.jl 9 module asm was never
# actually cleared.
# 6: a slice declares the library's own exported symbols by their `__rb_` alias.
# A v5 entry declares the public name, which binds through the whole process.
const SLICER_VERSION = "6"

struct SliceResult
    name::String
    ir::Union{Nothing,String}          # nothing ⇒ refused
    hazards::Vector{Symbol}            # M3 dispatch gates on these
    refusal::Union{Nothing,String}
    n_declared_fns::Int
    n_declared_globals::Int
    n_embedded_constants::Int
    declares::Vector{String}           # symbols the JIT must resolve (M3 pre-flight)
end

"Whether a usable slice was produced."
sliced(r::SliceResult) = r.ir !== nothing

"""
Whether slicing threw instead of reaching a verdict. The refusal then says
nothing about the function: it is a Slicer bug or a transient failure. So it
is never cached, and the wrap reports it apart from the policy refusals.
"""
errored(r::SliceResult) = :slicer_error in r.hazards

_refuse(name, why::String, hazards=Symbol[]) =
    SliceResult(name, nothing, hazards, why, 0, 0, 0, String[])

# One line naming the exception and the innermost Slicer frame it came through,
# so a report from a wrap log points at the read that broke. Capped: a
# systematic break refuses every target it reaches with the same text.
function _error_reason(e, bt)
    T = string(nameof(typeof(e)))
    msg = replace(strip(sprint(showerror, e)), r"\s+" => " ")
    msg = chopprefix(msg, T * ": ")     # most `showerror`s open with the type
    length(msg) > 300 && (msg = first(msg, 300) * "…")
    site = ""
    for fr in stacktrace(bt)
        basename(String(fr.file)) == "Slicer.jl" || continue
        fn = string(fr.func)
        # An anonymous closure's name (`#9`) says nothing; the line does.
        site = (startswith(fn, "#") ? "" : " in $fn") * " at Slicer.jl:$(fr.line)"
        break
    end
    return "slicing threw $T$site: $msg"
end

# ── attribute / classification helpers ───────────────────────────────────────

const SETJMP_FAMILY = Set([
    "setjmp", "_setjmp", "sigsetjmp", "__sigsetjmp",
    "longjmp", "_longjmp", "siglongjmp",
])

# Looked up by kind ID through `LLVM.API`. `LLVM.kind(attr)` is that integer ID
# on LLVM.jl 9 but a Symbol on 10, where comparing it to the ID is always false:
# `returns_twice` callees would lose their hazard, and a `noinline` target would
# get `alwaysinline` too, which the verifier rejects.
function _has_enum_attr(f, attr_name::String)::Bool
    kind_id = LLVM.API.LLVMGetEnumAttributeKindForName(attr_name, Csize_t(length(attr_name)))
    fn_idx = reinterpret(LLVM.API.LLVMAttributeIndex, LLVM.API.LLVMAttributeFunctionIndex)
    return LLVM.API.LLVMGetEnumAttributeAtIndex(f, fn_idx, kind_id) != C_NULL
end

_is_vararg(f) = LLVM.API.LLVMIsFunctionVarArg(LLVM.API.LLVMGlobalGetValueType(f)) != 0

_is_local_linkage(gv) =
    LLVM.linkage(gv) in (LLVM.API.LLVMInternalLinkage, LLVM.API.LLVMPrivateLinkage)

"""
Whether duplicating `gv` is unobservable — LLVM's own marker for "only the
contents matter, not where they live". This is the exact precondition for
embedding a constant into a slice, because the `.so` keeps its copy and the
JIT gets a second one at a different address.

`local_unnamed_addr` suffices for the globals this gates: they have INTERNAL
linkage, so nothing outside the library can name them, and "not
address-significant within the module" therefore already covers every observer
that exists. Without either marker the address is live — the pointer may be
compared, stored and compared later, or used as a sentinel — and two copies
answer that question differently depending on which tier ran.

Read through `LLVM.API`: LLVM.jl 9 has `unnamed_addr` and `local_unnamed_addr`
as two Bools, while LLVM.jl 10 drops the second and makes `unnamed_addr` return
the three-way `LLVM.UnnamedAddr` enum.
"""
_addr_insignificant(gv) =
    LLVM.API.LLVMGetUnnamedAddress(gv) != LLVM.API.LLVMNoUnnamedAddr

"""
Whether `gv` is declared `constant`, so its contents never change at run time.

Not `LLVM.isconstant(gv)`: on LLVM.jl 10 that asks whether a value is an LLVM
`Constant`, and every global variable is one. A mutable static would then take
the constant branch of the closure walk: embedded when it carries an
`unnamed_addr` marker (the cJSON divergence class), refused for the wrong reason
otherwise. LLVM.jl 10 spells the old meaning `gv.constant`.
"""
_is_global_constant(gv) = LLVM.API.LLVMIsGlobalConstant(gv) != 0

_is_weak_linkage(gv) =
    LLVM.linkage(gv) in (LLVM.API.LLVMWeakAnyLinkage, LLVM.API.LLVMWeakODRLinkage,
                         LLVM.API.LLVMLinkOnceAnyLinkage, LLVM.API.LLVMLinkOnceODRLinkage,
                         LLVM.API.LLVMExternalWeakLinkage, LLVM.API.LLVMCommonLinkage)

"""
Whether `gv` reaches the dynamic symbol table under its own name: a definition
with external linkage and default visibility. Weak, `available_externally` and
local definitions do not qualify, nor does any declaration.
"""
_is_exported_definition(gv) =
    !LLVM.isdeclaration(gv) &&
    LLVM.linkage(gv) == LLVM.API.LLVMExternalLinkage &&
    LLVM.visibility(gv) == LLVM.API.LLVMDefaultVisibility

"The name of a raw `LLVMValueRef`; aliases have no LLVM.jl wrapper type."
function _ref_name(ref::LLVM.API.LLVMValueRef)::String
    len = Ref{Csize_t}(0)
    p = LLVM.API.LLVMGetValueName2(ref, len)
    return p == C_NULL ? "" : unsafe_string(Ptr{UInt8}(p), len[])
end

# ── exported aliases ─────────────────────────────────────────────────────────

"""
    _claim_rb_aliases!(mod) -> Dict{String,String}

The module's exported `__rb_*` aliases (`Compiler._alias_exports!`) as
`aliasee name => alias name`, with every alias in the module then taken out of
the slice's way: internal linkage, so `globaldce` removes it (there is no C API
to erase an alias), and for the `__rb_` ones a scratch name, so the aliasee can
take the alias name when `_extract!` turns it into a declaration. No alias ever
survives into a slice: an alias to a function whose body the slice deleted is
not valid IR, and a reached alias is refused by the closure walk before this
matters.

Only an alias whose aliasee is a function or global variable directly (not a
constant expression over one) is recorded; nothing else produces one.
"""
function _claim_rb_aliases!(mod)
    alias_of = Dict{String,String}()
    a = LLVM.API.LLVMGetFirstGlobalAlias(mod)
    while a != C_NULL
        nxt = LLVM.API.LLVMGetNextGlobalAlias(a)
        name = _ref_name(a)
        if startswith(name, "__rb_")
            aliasee = LLVM.API.LLVMAliasGetAliasee(a)
            kind = LLVM.API.LLVMGetValueKind(aliasee)
            if kind == LLVM.API.LLVMFunctionValueKind ||
               kind == LLVM.API.LLVMGlobalVariableValueKind
                alias_of[_ref_name(aliasee)] = name
            end
            scratch = name * ".slicer_claimed"
            LLVM.API.LLVMSetValueName2(a, scratch, Csize_t(sizeof(scratch)))
        end
        LLVM.API.LLVMSetLinkage(a, LLVM.API.LLVMInternalLinkage)
        a = nxt
    end
    return alias_of
end

# ── closure walk ─────────────────────────────────────────────────────────────

"""
Declarations-only dependence closure: the target's instruction operands, plus
the transitive constant DAG of every *embedded* (internal constant) global.
Reached functions get their bodies deleted and reached mutable globals get
their initializers dropped, so neither contributes further edges.

`alias_of` is the module's exported-alias map (`_claim_rb_aliases!`). A reached
function or global the library defines and exports under its own name must have
an entry, which `renames` then maps it to (`name => alias`): the slice declares
the alias, never the public name. One without an entry refuses the slice.

Returns `(reached_fns, reached_gvs, embedded_gvs, renames, hazards, refusal)`
where the sets hold names. `refusal` is `nothing` when the closure is sliceable.
"""
function _closure(mod, target, alias_of::Dict{String,String})
    reached_fns = Set{String}()
    reached_gvs = Set{String}()
    embedded_gvs = Set{String}()
    renames = Dict{String,String}()
    hazards = Set{Symbol}()
    refusal = Ref{Union{Nothing,String}}(nothing)

    # An exported definition is a name the process may already hold from
    # another library, so the slice may only bind it through its alias.
    bind_via_alias!(v, name) = begin
        startswith(name, "__rb_") && return         # promoted: already unique
        alias = get(alias_of, name, nothing)
        if alias === nothing
            refusal[] = "reached exported definition '$name', which has no `__rb_` " *
                        "alias in the module. A slice declaring the public name " *
                        "binds through the whole process, where a library loaded " *
                        "earlier may define it (Julia ships zlib, zstd, pcre2, …). " *
                        "Rebuild so promotion adds the aliases (delete " *
                        ".replibuild_cache/project_hash), or exclude this function."
        else
            renames[name] = alias
        end
    end

    target_name = LLVM.name(target)
    seen = Set{UInt}()  # visited constant/value refs

    gv_queue = Any[]   # embedded globals whose initializers still need walking

    visit(v) = begin
        refusal[] === nothing || return
        ref = UInt(Base.unsafe_convert(LLVM.API.LLVMValueRef, v))
        ref in seen && return
        push!(seen, ref)

        kind = LLVM.API.LLVMGetValueKind(v)
        if kind == LLVM.API.LLVMFunctionValueKind
            name = LLVM.name(v)
            name == target_name && return
            push!(reached_fns, name)
            if name in SETJMP_FAMILY || _has_enum_attr(v, "returns_twice")
                push!(hazards, :setjmp_family)
            end
            if !startswith(name, "llvm.") && _is_vararg(v)
                push!(hazards, :varargs_callee)
            end
            if !startswith(name, "llvm.") && !LLVM.isdeclaration(v) && _is_local_linkage(v)
                refusal[] = "reached internal-linkage function '$name' — module is not " *
                            "static-promoted (slice from <name>_abi.ll, not _opt.ll)"
            elseif !startswith(name, "llvm.") && _is_exported_definition(v)
                bind_via_alias!(v, name)
            end
            _is_weak_linkage(v) && push!(hazards, :weak)
        elseif kind == LLVM.API.LLVMGlobalVariableValueKind
            name = LLVM.name(v)
            name in reached_gvs && return
            push!(reached_gvs, name)
            _is_weak_linkage(v) && push!(hazards, :weak)
            is_const = _is_global_constant(v)
            if is_const && _is_local_linkage(v) && _addr_insignificant(v)
                # no symbol exists and the address is dead → embed; its
                # initializer contributes edges
                push!(embedded_gvs, name)
                push!(gv_queue, v)
            elseif is_const && _is_local_linkage(v)
                refusal[] = "reached address-significant internal constant " *
                            "'$name' — it carries no `unnamed_addr`, so its " *
                            "address is observable, and embedding would give " *
                            "the slice a second copy at a different address " *
                            "than the .so's. Promote it (a declared symbol " *
                            "keeps one copy) or exclude this function."
            elseif !is_const && _is_local_linkage(v)
                refusal[] = "reached internal-linkage mutable global '$name' — module " *
                            "is not static-promoted (slice from <name>_abi.ll)"
            elseif _is_exported_definition(v)
                bind_via_alias!(v, name)
            end
            # non-local (declared) globals: initializer will be dropped, no edges
        elseif kind == LLVM.API.LLVMGlobalAliasValueKind
            refusal[] = "reached global alias '$(LLVM.name(v))' — alias slicing unbuilt"
        elseif kind == LLVM.API.LLVMGlobalIFuncValueKind
            refusal[] = "reached ifunc '$(LLVM.name(v))' — ifunc slicing unbuilt"
        elseif kind == LLVM.API.LLVMBlockAddressValueKind
            # blockaddress(@fn, %bb): fine when fn is the target (its body stays);
            # fatal otherwise (the table would point into a deleted body).
            ba_fn = first(LLVM.operands(v))
            if LLVM.name(ba_fn) != target_name
                refusal[] = "blockaddress into '$(LLVM.name(ba_fn))' — jump table " *
                            "into a function the slice would declare"
            end
        elseif kind == LLVM.API.LLVMInlineAsmValueKind
            push!(hazards, :inline_asm)
        elseif v isa LLVM.User
            # constant expressions / aggregates — recurse the constant DAG
            for op in LLVM.operands(v)
                visit(op)
            end
        end
    end

    for bb in LLVM.blocks(target), inst in LLVM.instructions(bb)
        refusal[] === nothing || break
        for op in LLVM.operands(inst)
            visit(op)
        end
    end
    while refusal[] === nothing && !isempty(gv_queue)
        gv = popfirst!(gv_queue)
        init = LLVM.initializer(gv)
        init === nothing && continue
        visit(init)
    end

    return reached_fns, reached_gvs, embedded_gvs, renames, collect(hazards), refusal[]
end

# ── extraction ───────────────────────────────────────────────────────────────

"""
Extract one target from a clone of the promoted module. The clone is consumed.
Returns a `SliceResult`.
"""
function _extract!(mod, target_name::String)
    # Locate the target
    target = nothing
    for f in LLVM.functions(mod)
        if LLVM.name(f) == target_name
            target = f
            break
        end
    end
    target === nothing && return _refuse(target_name, "function not found in module")
    LLVM.isdeclaration(target) &&
        return _refuse(target_name, "function has no body in module (declaration)")
    _is_vararg(target) &&
        return _refuse(target_name, "variadic target — llvmcall cannot pass varargs",
                       [:target_varargs])

    hazards = Symbol[]

    # Module-level asm: the target's code never needs it (per-function asm is a
    # value, not module asm) — clear it so the JIT link stays clean. Cleared
    # through `LLVM.API`: LLVM.jl 9's `inline_asm!(mod, "")` APPENDS unless given
    # `overwrite=true`, so it never cleared anything, and LLVM.jl 10 has no
    # `inline_asm!` at all. (`isempty` reads both the String that `inline_asm`
    # returns on 9 and the view it returns on 10.)
    if !isempty(LLVM.inline_asm(mod))
        push!(hazards, :module_asm)
        LLVM.API.LLVMSetModuleInlineAsm2(mod, "", 0)
    end

    # The exported aliases, claimed before the walk so the walk can demand one
    # for every exported definition it reaches, and moved aside so the renames
    # below find their names free.
    alias_of = _claim_rb_aliases!(mod)

    reached_fns, reached_gvs, embedded_gvs, renames, closure_hazards, refusal =
        _closure(mod, target, alias_of)
    refusal === nothing || return _refuse(target_name, refusal, closure_hazards)
    append!(hazards, closure_hazards)

    # Functions: reached defs → declarations; unreached defs → internalize+DCE.
    for f in collect(LLVM.functions(mod))
        name = LLVM.name(f)
        name == target_name && continue
        LLVM.isdeclaration(f) && continue
        if name in reached_fns
            LLVM.API.LLVMFunctionDeleteBody(f)  # C++ deleteBody: safe, sets external
            if LLVM.API.LLVMHasPersonalityFn(f) != 0
                # LLVM 20 folded the Extra `*2` NULL-tolerant setter into
                # `LLVMSetPersonalityFn`. LLVM.jl picks the right one.
                LLVM.personality!(f, nothing)
            end
        else
            LLVM.linkage!(f, LLVM.API.LLVMInternalLinkage)
        end
    end

    # Globals: embedded constants keep their definitions; everything else
    # reached → declaration; unreached → internalize+DCE.
    for gv in collect(LLVM.globals(mod))
        name = LLVM.name(gv)
        if name in embedded_gvs
            # internal constant, embedded as-is
        elseif name in reached_gvs
            if !LLVM.isdeclaration(gv)
                # Same LLVM 20 fold as personality: Extra `LLVMSetInitializer2`
                # is not in LLVM.API when Julia's libLLVM is 20+.
                LLVM.initializer!(gv, nothing)
            end
            LLVM.linkage!(gv, LLVM.API.LLVMExternalLinkage)
        elseif !LLVM.isdeclaration(gv)
            LLVM.linkage!(gv, LLVM.API.LLVMInternalLinkage)
        end
    end

    # Exported definitions the slice reached are declarations now; give each
    # the alias name it binds by. The alias itself was renamed away by
    # `_claim_rb_aliases!`, so the name is free: a suffix here means something
    # else in the module holds it, which is a Slicer bug, not a verdict.
    for (old, new) in renames
        gv = haskey(LLVM.functions(mod), old) ? LLVM.functions(mod)[old] :
                                                LLVM.globals(mod)[old]
        LLVM.name!(gv, new)
        LLVM.name(gv) == new ||
            error("Slicer: '$old' could not take its alias name '$new' " *
                  "(got '$(LLVM.name(gv))')")
    end

    # Aliases are all internal now (`_claim_rb_aliases!`) and nothing references
    # one — the closure refused any *reached* alias — so DCE sweeps them.

    tm = LLVM.JITTargetMachine()
    LLVM.run!("globaldce,strip-dead-prototypes", mod, tm)

    # alwaysinline on the entry so Julia splices the body into the caller —
    # unless the compiler explicitly said noinline (respect its judgment).
    if _has_enum_attr(target, "noinline")
        push!(hazards, :noinline)
    else
        push!(LLVM.function_attributes(target), LLVM.EnumAttribute("alwaysinline", 0))
    end

    # Adopt Julia's datalayout/triple at link time; avoids mismatch warnings.
    LLVM.API.LLVMSetDataLayout(mod, "")
    LLVM.API.LLVMSetTarget(mod, "")

    LLVM.verify(mod)

    # Post-DCE inventory of what actually survived. The declaration names are
    # the slice's binding contract: at llvmcall time ORC resolves each one
    # against the process (the RTLD_GLOBAL-loaded `.so` plus its dependencies,
    # plus everything loaded before them — which is why the library's own
    # symbols are declared by their `__rb_` alias), and a single miss deadlocks
    # the JIT rather than erroring. M3 pre-flights this list with dlsym so a
    # miss demotes the function to ccall instead. `llvm.*` intrinsics are
    # lowered by the backend and never looked up.
    decl_fns = [LLVM.name(f) for f in LLVM.functions(mod)
                if LLVM.isdeclaration(f) && LLVM.name(f) != target_name]
    decl_gvs = [LLVM.name(g) for g in LLVM.globals(mod) if LLVM.isdeclaration(g)]
    n_decl_fns = length(decl_fns)
    n_decl_gvs = length(decl_gvs)
    n_embed = count(g -> !LLVM.isdeclaration(g), collect(LLVM.globals(mod)))
    declares = sort!(filter(n -> !startswith(n, "llvm."), vcat(decl_fns, decl_gvs)))

    ir = string(mod)
    # Module flags carry nothing for a slice (no debug info) but conflict with
    # Julia's (Dwarf 5 vs 4) on every llvmcall compile. Orphaned numbered
    # metadata nodes that remain are legal and ignored.
    ir = replace(ir, r"^!llvm\.(module\.flags|ident)[^\n]*\n"m => "")

    return SliceResult(target_name, ir, hazards, nothing,
                       n_decl_fns, n_decl_gvs, n_embed, declares)
end

# ── public API ───────────────────────────────────────────────────────────────

function _clone(master, src_text::String)
    ref = LLVM.API.LLVMCloneModule(master)
    return LLVM.Module(ref)
end

"""
    _slice_cache_key(src_bytes; llvm_major=Base.libllvm_version.major) -> String

The slice-cache directory for one promoted module: its content hash,
`SLICER_VERSION`, and the libLLVM major that prints the slices.

The major is there because a cached slice is IR text, printed by whichever
Julia sliced it. Switch Julia and `wrap` alone (no rebuild) and the module text,
so the content hash, is unchanged: without the major, Julia 1.12 was served
slices Julia 1.13 had printed, and their `icmp samesign` failed to parse in
LLVM 18 at the first call.
"""
_slice_cache_key(src_bytes; llvm_major::Integer=Base.libllvm_version.major) =
    bytes2hex(sha256(src_bytes))[1:16] * "-v" * SLICER_VERSION * "-llvm" * string(llvm_major)

"""
    slice_library(abi_ll; targets, cache_dir=nothing) -> Dict{String,SliceResult}

Slice `targets` out of the post-promotion module at `abi_ll`. Parses the
module once and clones per target. With `cache_dir`, results are cached by
(module content hash, target, SLICER_VERSION, libLLVM major; see
`_slice_cache_key`) and served on hit.

An exception while slicing one target becomes that target's refusal, tagged
`:slicer_error` (see `errored`), and is never cached; the other targets are
unaffected. A failure on the module as a whole (reading, parsing, the shared
prep) still throws, because no target can be sliced from it.
"""
function slice_library(abi_ll::String; targets::Vector{String},
                       cache_dir::Union{Nothing,String}=nothing)
    isfile(abi_ll) || error("slice source not found: $abi_ll")
    src_bytes = read(abi_ll)
    results = Dict{String,SliceResult}()

    key_dir = nothing
    if cache_dir !== nothing
        key_dir = joinpath(cache_dir, "slices", _slice_cache_key(src_bytes))
        mkpath(key_dir)
    end

    to_compute = String[]
    for t in targets
        cached = key_dir === nothing ? nothing : _cache_load(key_dir, t)
        if cached === nothing
            push!(to_compute, t)
        else
            results[t] = cached
        end
    end

    if !isempty(to_compute)
        src_text = String(copy(src_bytes))
        LLVM.Context() do _
            master = parse(LLVM.Module, src_text)

            # Target-independent prep on the master — clones inherit it, so the
            # per-target cost drops sharply (debug metadata dominates the IR).
            LLVM.strip_debuginfo!(master)
            # llvm.* intrinsic globals (compiler.used pins every shim through
            # DCE; the .so owns ctors/dtors — a slice must never re-run them).
            for gv in collect(LLVM.globals(master))
                if startswith(LLVM.name(gv), "llvm.")
                    isempty(collect(LLVM.uses(gv))) ||
                        error("Slicer: intrinsic global $(LLVM.name(gv)) has uses")
                    LLVM.API.LLVMDeleteGlobal(gv)
                end
            end

            for t in to_compute
                clone = _clone(master, src_text)
                r = try
                    _extract!(clone, t)
                catch e
                    e isa InterruptException && rethrow()
                    _refuse(t, _error_reason(e, catch_backtrace()), [:slicer_error])
                finally
                    LLVM.dispose(clone)
                end
                results[t] = r
                # An error is not a verdict on the function. Cached, it would be
                # served until the module or SLICER_VERSION changed.
                (key_dir === nothing || errored(r)) || _cache_store(key_dir, r)
            end
            LLVM.dispose(master)
        end
    end

    return results
end

"Slice a single function (convenience wrapper over `slice_library`)."
function slice_function(abi_ll::String, target::String;
                        cache_dir::Union{Nothing,String}=nothing)
    return slice_library(abi_ll; targets=[target], cache_dir=cache_dir)[target]
end

"""
    slice_file_stem(name) -> String

File name, without extension, for `name`'s slice: in this cache and in a wrapper's
`slices/` directory. The symbol itself when it has no uppercase letter, otherwise
the symbol plus 8 hex digits of its SHA-1, so two names that differ only in case
get two files. Windows (and macOS by default) folds case: lua's value macro
`LUA_YIELD` and function-like macro `lua_yield` both wrote
`replibuild_shim_LUA_YIELD.ll`, the later write won, and `LUA_YIELD()` threw
"Module IR does not contain specified entry function".
"""
slice_file_stem(name::AbstractString) =
    any(isuppercase, name) ? string(name, "-", bytes2hex(sha1(String(name)))[1:8]) : String(name)

# ── cache ────────────────────────────────────────────────────────────────────

function _cache_load(key_dir::String, target::String)
    stem = slice_file_stem(target)
    meta_path = joinpath(key_dir, stem * ".json")
    isfile(meta_path) || return nothing
    try
        meta = _read_json(meta_path)
        ir = nothing
        if get(meta, "sliced", false)
            ll_path = joinpath(key_dir, stem * ".ll")
            isfile(ll_path) || return nothing
            ir = read(ll_path, String)
        end
        return SliceResult(target, ir,
                           Symbol.(get(meta, "hazards", String[])),
                           get(meta, "refusal", nothing),
                           get(meta, "n_declared_fns", 0),
                           get(meta, "n_declared_globals", 0),
                           get(meta, "n_embedded_constants", 0),
                           String[String(s) for s in get(meta, "declares", String[])])
    catch
        return nothing  # corrupt entry → recompute
    end
end

# Writes go through a temp file + rename so a concurrent build of the same
# project can never read a torn entry: _cache_load validates nothing about
# the .ll body, so a partial write would be served as slice IR and fail at
# llvmcall time. rename(2) within a directory is atomic on POSIX.
function _atomic_write(path::String, contents::AbstractString)
    tmp = path * ".tmp-" * string(getpid())
    write(tmp, contents)
    mv(tmp, path; force=true)
end

function _cache_store(key_dir::String, r::SliceResult)
    stem = slice_file_stem(r.name)
    if sliced(r)
        _atomic_write(joinpath(key_dir, stem * ".ll"), r.ir)
    end
    meta = sprint() do io
        JSON.print(io, Dict(
            "sliced" => sliced(r),
            "hazards" => String.(r.hazards),
            "refusal" => r.refusal,
            "n_declared_fns" => r.n_declared_fns,
            "n_declared_globals" => r.n_declared_globals,
            "n_embedded_constants" => r.n_embedded_constants,
            "declares" => r.declares,
        ), 2)
    end
    _atomic_write(joinpath(key_dir, stem * ".json"), meta)
end

end # module Slicer
