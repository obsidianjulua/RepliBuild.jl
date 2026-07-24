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
# Boundary policy per reached GlobalValue:
#   function            → declare (promotion guarantees a symbol exists)
#   mutable global      → declare (never embed — cJSON divergence class)
#   internal constant   → embed   (no symbol exists; duplication is harmless)
#   external constant   → declare (a symbol exists; bind, don't duplicate)
#   alias / ifunc       → refuse the slice (unbuilt, ledger class)
#   blockaddress ≠ self → refuse (jump table into a body we'd be deleting)
#
# Everything runs on Julia's resident libLLVM via LLVM.jl — same in-process,
# version-locked philosophy as the C build bucket. Every produced slice is
# verified before it is returned; a slice that cannot be produced comes back
# as a refusal with a reason, never as silently-wrong IR.

module Slicer

using LLVM
using SHA
using JSON

export SliceResult, slice_library, slice_function, sliced

"Bump to invalidate every cached slice (policy/algorithm change)."
const SLICER_VERSION = "1"

struct SliceResult
    name::String
    ir::Union{Nothing,String}          # nothing ⇒ refused
    hazards::Vector{Symbol}            # M3 dispatch gates on these
    refusal::Union{Nothing,String}
    n_declared_fns::Int
    n_declared_globals::Int
    n_embedded_constants::Int
end

"Whether a usable slice was produced."
sliced(r::SliceResult) = r.ir !== nothing

_refuse(name, why::String, hazards=Symbol[]) =
    SliceResult(name, nothing, hazards, why, 0, 0, 0)

# ── attribute / classification helpers ───────────────────────────────────────

const SETJMP_FAMILY = Set([
    "setjmp", "_setjmp", "sigsetjmp", "__sigsetjmp",
    "longjmp", "_longjmp", "siglongjmp",
])

function _has_enum_attr(f, attr_name::String)::Bool
    kind_id = LLVM.API.LLVMGetEnumAttributeKindForName(attr_name, Csize_t(length(attr_name)))
    for attr in collect(LLVM.function_attributes(f))
        attr isa LLVM.EnumAttribute || continue
        LLVM.kind(attr) == kind_id && return true
    end
    return false
end

_is_vararg(f) = LLVM.API.LLVMIsFunctionVarArg(LLVM.API.LLVMGlobalGetValueType(f)) != 0

_is_local_linkage(gv) =
    LLVM.linkage(gv) in (LLVM.API.LLVMInternalLinkage, LLVM.API.LLVMPrivateLinkage)

_is_weak_linkage(gv) =
    LLVM.linkage(gv) in (LLVM.API.LLVMWeakAnyLinkage, LLVM.API.LLVMWeakODRLinkage,
                         LLVM.API.LLVMLinkOnceAnyLinkage, LLVM.API.LLVMLinkOnceODRLinkage,
                         LLVM.API.LLVMExternalWeakLinkage, LLVM.API.LLVMCommonLinkage)

# ── closure walk ─────────────────────────────────────────────────────────────

"""
Declarations-only dependence closure: the target's instruction operands, plus
the transitive constant DAG of every *embedded* (internal constant) global.
Reached functions get their bodies deleted and reached mutable globals get
their initializers dropped, so neither contributes further edges.

Returns `(reached_fns, reached_gvs, embedded_gvs, hazards, refusal)` where the
sets hold names. `refusal` is `nothing` when the closure is sliceable.
"""
function _closure(mod, target)
    reached_fns = Set{String}()
    reached_gvs = Set{String}()
    embedded_gvs = Set{String}()
    hazards = Set{Symbol}()
    refusal = Ref{Union{Nothing,String}}(nothing)

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
            end
            _is_weak_linkage(v) && push!(hazards, :weak)
        elseif kind == LLVM.API.LLVMGlobalVariableValueKind
            name = LLVM.name(v)
            name in reached_gvs && return
            push!(reached_gvs, name)
            _is_weak_linkage(v) && push!(hazards, :weak)
            is_const = LLVM.isconstant(v)
            if is_const && _is_local_linkage(v)
                # no symbol exists → embed; its initializer contributes edges
                push!(embedded_gvs, name)
                push!(gv_queue, v)
            elseif !is_const && _is_local_linkage(v)
                refusal[] = "reached internal-linkage mutable global '$name' — module " *
                            "is not static-promoted (slice from <name>_abi.ll)"
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

    return reached_fns, reached_gvs, embedded_gvs, collect(hazards), refusal[]
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
    # value, not module asm) — clear it so the JIT link stays clean.
    if !isempty(LLVM.inline_asm(mod))
        push!(hazards, :module_asm)
        LLVM.inline_asm!(mod, "")
    end

    reached_fns, reached_gvs, embedded_gvs, closure_hazards, refusal = _closure(mod, target)
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
                LLVM.API.LLVMSetPersonalityFn2(f, C_NULL)
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
                LLVM.API.LLVMSetInitializer2(gv, C_NULL)
            end
            LLVM.linkage!(gv, LLVM.API.LLVMExternalLinkage)
        elseif !LLVM.isdeclaration(gv)
            LLVM.linkage!(gv, LLVM.API.LLVMInternalLinkage)
        end
    end

    # Aliases can't be internalized away reliably — the closure already refused
    # any *reached* alias; unreached ones are swept by DCE if unreferenced.

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

    # Post-DCE stats from what actually survived
    n_decl_fns = count(f -> LLVM.isdeclaration(f) && LLVM.name(f) != target_name,
                       collect(LLVM.functions(mod)))
    n_decl_gvs = count(g -> LLVM.isdeclaration(g), collect(LLVM.globals(mod)))
    n_embed = count(g -> !LLVM.isdeclaration(g), collect(LLVM.globals(mod)))

    ir = string(mod)
    # Module flags carry nothing for a slice (no debug info) but conflict with
    # Julia's (Dwarf 5 vs 4) on every llvmcall compile. Orphaned numbered
    # metadata nodes that remain are legal and ignored.
    ir = replace(ir, r"^!llvm\.(module\.flags|ident)[^\n]*\n"m => "")

    return SliceResult(target_name, ir, hazards, nothing, n_decl_fns, n_decl_gvs, n_embed)
end

# ── public API ───────────────────────────────────────────────────────────────

function _clone(master, src_text::String)
    ref = LLVM.API.LLVMCloneModule(master)
    return LLVM.Module(ref)
end

"""
    slice_library(abi_ll; targets, cache_dir=nothing) -> Dict{String,SliceResult}

Slice `targets` out of the post-promotion module at `abi_ll`. Parses the
module once and clones per target. With `cache_dir`, results are cached by
(module content hash, target, SLICER_VERSION) and served on hit.
"""
function slice_library(abi_ll::String; targets::Vector{String},
                       cache_dir::Union{Nothing,String}=nothing)
    isfile(abi_ll) || error("slice source not found: $abi_ll")
    src_bytes = read(abi_ll)
    results = Dict{String,SliceResult}()

    key_dir = nothing
    if cache_dir !== nothing
        modkey = bytes2hex(sha256(src_bytes))[1:16] * "-v" * SLICER_VERSION
        key_dir = joinpath(cache_dir, "slices", modkey)
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
                r = _extract!(clone, t)
                LLVM.dispose(clone)
                results[t] = r
                key_dir === nothing || _cache_store(key_dir, r)
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

# ── cache ────────────────────────────────────────────────────────────────────

function _cache_load(key_dir::String, target::String)
    meta_path = joinpath(key_dir, target * ".json")
    isfile(meta_path) || return nothing
    try
        meta = JSON.parsefile(meta_path)
        ir = nothing
        if get(meta, "sliced", false)
            ll_path = joinpath(key_dir, target * ".ll")
            isfile(ll_path) || return nothing
            ir = read(ll_path, String)
        end
        return SliceResult(target, ir,
                           Symbol.(get(meta, "hazards", String[])),
                           get(meta, "refusal", nothing),
                           get(meta, "n_declared_fns", 0),
                           get(meta, "n_declared_globals", 0),
                           get(meta, "n_embedded_constants", 0))
    catch
        return nothing  # corrupt entry → recompute
    end
end

function _cache_store(key_dir::String, r::SliceResult)
    if sliced(r)
        write(joinpath(key_dir, r.name * ".ll"), r.ir)
    end
    open(joinpath(key_dir, r.name * ".json"), "w") do io
        JSON.print(io, Dict(
            "sliced" => sliced(r),
            "hazards" => String.(r.hazards),
            "refusal" => r.refusal,
            "n_declared_fns" => r.n_declared_fns,
            "n_declared_globals" => r.n_declared_globals,
            "n_embedded_constants" => r.n_embedded_constants,
        ), 2)
    end
end

end # module Slicer
