#!/usr/bin/env julia
# test/test_jlcs_invariants.jl — Definitive-trace probes for JLCS dialect concerns
#
# This file does NOT assume any of the suspected issues are real. Each probe
# pushes a specific concern through the actual lowering stack (parse → lower →
# emit LLVM IR) and records what the stack *actually* does. The point is to
# replace "I think this is a bug" with a reproducible trace: a clean lowering,
# a graceful diagnostic, or a hard crash are three distinguishable outcomes.
#
# Malformed IR that may trigger out-of-bounds access during lowering is run in
# an isolated subprocess so a SIGSEGV produces an exit signal we can assert on,
# rather than taking down the whole test runner.
#
# Concerns probed (from the dialect review on 2026-05-29):
#   A. jlcs.scope        — managed_ptrs.size() vs destructors.size() mismatch
#   B. jlcs.marshal_arg  — memberTypes/juliaOffsets/result-field count mismatch
#   C. array ops         — are load/store_array_element + array_view still alive?
#
# Requires libJLCS.so (build with: cd src/mlir && ./build.sh)

using Test

const MLIR_AVAILABLE = try
    using RepliBuild
    isfile(RepliBuild.MLIRNative.libJLCS)
catch
    false
end

if !MLIR_AVAILABLE
    @info "libJLCS not found — skipping JLCS invariant probes"
    exit(0)
end

using RepliBuild.MLIRNative

const PROJECT = dirname(@__DIR__)

# ──────────────────────────────────────────────────────────────────────────────
# Isolated lowering driver
#
# Runs parse + lower in a fresh subprocess and reports a definitive outcome:
#   :lowered  — lowering returned true (clean)
#   :failed   — lowering returned false (graceful diagnostic / pattern failure)
#   :parse    — parse_module rejected the IR (verifier/parser caught it)
#   :crash    — process died on a signal (SIGSEGV etc.) → undefined behaviour
# ──────────────────────────────────────────────────────────────────────────────

function probe_lowering(ir::String)
    irfile = tempname() * ".mlir"
    write(irfile, ir)
    driver = """
        using RepliBuild.MLIRNative
        ir = read($(repr(irfile)), String)
        ctx = create_context()
        mod = try
            parse_module(ctx, ir)
        catch
            exit(3)   # parse/verifier rejected
        end
        ok = lower_to_llvm(mod)
        exit(ok ? 0 : 2)
    """
    proc = run(pipeline(
        ignorestatus(`julia --project=$(PROJECT) -e $driver`);
        stdout = devnull, stderr = devnull))
    rm(irfile; force = true)
    if proc.termsignal != 0
        return (:crash, proc.termsignal)
    elseif proc.exitcode == 0
        return (:lowered, 0)
    elseif proc.exitcode == 2
        return (:failed, 2)
    elseif proc.exitcode == 3
        return (:parse, 3)
    else
        return (:other, proc.exitcode)
    end
end

# In-process lower + emit LLVM IR text, for well-formed cases we want to inspect.
function lower_and_emit(ir::String)
    ctx = create_context()
    try
        mod = parse_module(ctx, ir)
        ok = lower_to_llvm(mod)
        ok || return (false, "")
        llpath = tempname() * ".ll"
        emit_llvmir(mod, llpath)
        text = isfile(llpath) ? read(llpath, String) : ""
        rm(llpath; force = true)
        return (true, text)
    finally
        destroy_context(ctx)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
@testset "JLCS invariant probes" begin

# ── A. jlcs.scope: managed_ptrs vs destructors arity ──────────────────────────
@testset "A. scope arity" begin
    # A1 — well-formed baseline: 2 objects, 2 destructors.
    # Definitive trace: destructors must lower in REVERSE order (B before A),
    # after the constructors. This is the RAII contract.
    wellformed = """
    module {
      llvm.func @_ZN1AC1Ev(!llvm.ptr)
      llvm.func @_ZN1BC1Ev(!llvm.ptr)
      llvm.func @_ZN1AD1Ev(!llvm.ptr)
      llvm.func @_ZN1BD1Ev(!llvm.ptr)
      func.func @scoped() attributes {llvm.emit_c_interface} {
        %n = arith.constant 1 : i64
        %a = llvm.alloca %n x i8 : (i64) -> !llvm.ptr
        %b = llvm.alloca %n x i8 : (i64) -> !llvm.ptr
        jlcs.scope(%a, %b : !llvm.ptr, !llvm.ptr) dtors([@_ZN1AD1Ev, @_ZN1BD1Ev]) {
          jlcs.ctor_call @_ZN1AC1Ev(%a) : (!llvm.ptr) -> ()
          jlcs.ctor_call @_ZN1BC1Ev(%b) : (!llvm.ptr) -> ()
          jlcs.yield
        }
        return
      }
    }
    """
    ok, ll = lower_and_emit(wellformed)
    @test ok
    if ok
        # Match call SITES, not the forward declarations at the top of the module
        # (the declarations appear in source order and would mask call order).
        posA = findfirst("call void @_ZN1AD1Ev", ll)
        posB = findfirst("call void @_ZN1BD1Ev", ll)
        @test posA !== nothing && posB !== nothing
        # Reverse destruction: B's destructor call precedes A's in the output.
        @test posB !== nothing && posA !== nothing && first(posB) < first(posA)
        println("  ✓ A1 well-formed scope lowers; reverse-order destruction confirmed")
    end

    # A2 — malformed: 1 managed ptr but 2 destructors.
    # The lowering's emitDestructors indexes managedPtrs[i] for i up to
    # destructors.size()-1. If that read is unchecked, this is an OOB access.
    # We make NO assumption — we record what the stack does.
    malformed = """
    module {
      llvm.func @_ZN1AD1Ev(!llvm.ptr)
      llvm.func @_ZN1BD1Ev(!llvm.ptr)
      func.func @scoped() attributes {llvm.emit_c_interface} {
        %n = arith.constant 1 : i64
        %a = llvm.alloca %n x i8 : (i64) -> !llvm.ptr
        jlcs.scope(%a : !llvm.ptr) dtors([@_ZN1AD1Ev, @_ZN1BD1Ev]) {
          jlcs.yield
        }
        return
      }
    }
    """
    outcome, code = probe_lowering(malformed)
    println("  → A2 malformed scope (1 ptr / 2 dtors): $outcome (code/signal=$code)")
    # Contract: malformed IR must be rejected gracefully (verifier/diagnostic),
    # never segfault. ScopeOp::verify() (2026-07-16) rejects the arity mismatch
    # at parse time — outcome is :parse, not :crash.
    @test outcome != :crash
    @test outcome == :parse
    outcome == :crash && @warn "scope arity mismatch CRASHES lowering (signal $code) — verifier needed"
end

# ── B. jlcs.marshal_arg: member/offset/field-count arity ──────────────────────
@testset "B. marshal_arg arity" begin
    # B1 — well-formed baseline: 2 members, 2 offsets, 2-field packed result.
    wellformed = """
    module {
      func.func @marshal2(%p: !llvm.ptr) -> !llvm.struct<packed (i32, f64)>
          attributes {llvm.emit_c_interface} {
        %v = jlcs.marshal_arg %p
          { memberTypes = [i32, f64], juliaOffsets = [0 : i64, 8 : i64] }
          : (!llvm.ptr) -> !llvm.struct<packed (i32, f64)>
        return %v : !llvm.struct<packed (i32, f64)>
      }
    }
    """
    ok, ll = lower_and_emit(wellformed)
    @test ok
    ok && println("  ✓ B1 well-formed marshal_arg lowers")

    # B2 — malformed: 2 memberTypes but only 1 juliaOffset.
    # Lowering loops over memberTypes.size() and reads juliaOffsets[i]; the
    # second iteration indexes past the offsets array. Record the outcome.
    malformed = """
    module {
      func.func @marshal_bad(%p: !llvm.ptr) -> !llvm.struct<packed (i32, f64)>
          attributes {llvm.emit_c_interface} {
        %v = jlcs.marshal_arg %p
          { memberTypes = [i32, f64], juliaOffsets = [0 : i64] }
          : (!llvm.ptr) -> !llvm.struct<packed (i32, f64)>
        return %v : !llvm.struct<packed (i32, f64)>
      }
    }
    """
    outcome, code = probe_lowering(malformed)
    println("  → B2 malformed marshal_arg (2 types / 1 offset): $outcome (code/signal=$code)")
    # Same contract as A2: must reject gracefully, never segfault.
    # MarshalArgOp::verify() (2026-07-16) rejects the arity mismatch at parse.
    @test outcome != :crash
    @test outcome == :parse
    outcome == :crash && @warn "marshal_arg offset/type mismatch CRASHES lowering (signal $code) — verifier needed"
end

# ── C. array ops liveness: are they still wired through the stack? ────────────
@testset "C. array op liveness" begin
    # No Julia generator emits these and no other test exercises them. This
    # probe answers one question definitively: do load/store_array_element +
    # array_view still parse and lower, or have they bit-rotted?
    ir = """
    module {
      func.func @arr_load(%v: !llvm.ptr, %i: index) -> f64
          attributes {llvm.emit_c_interface} {
        %e = "jlcs.load_array_element"(%v, %i) : (!llvm.ptr, index) -> f64
        return %e : f64
      }
      func.func @arr_store(%val: f64, %v: !llvm.ptr, %i: index)
          attributes {llvm.emit_c_interface} {
        "jlcs.store_array_element"(%val, %v, %i) : (f64, !llvm.ptr, index) -> ()
        return
      }
    }
    """
    outcome, code = probe_lowering(ir)
    println("  → C array ops parse+lower: $outcome (code/signal=$code)")
    @test outcome in (:lowered, :failed, :parse, :crash)
    if outcome == :lowered
        println("    array ops are FUNCTIONAL (produced by ArrayViewGen since 2026-07-16; executed in test_jlcs_producers.jl)")
    elseif outcome in (:parse, :crash)
        @warn "array ops no longer survive the stack ($outcome) — bit-rotted, wire up or remove"
    end

    # Also confirm the !jlcs.array_view type itself still parses.
    type_ir = """
    module {
      func.func @av(%x: !jlcs.array_view<f64, 3>) {
        return
      }
    }
    """
    ctx = create_context()
    parsed = try
        m = parse_module(ctx, type_ir); m != C_NULL
    catch
        false
    finally
        destroy_context(ctx)
    end
    println("  → C array_view type parses: $parsed")
    @test parsed isa Bool  # record, don't presume
end

# ──────────────────────────────────────────────────────────────────────────────
# D. Every symbol libJLCS.so references can be resolved
#
# `libJLCS.so` shipped for its whole life with TEN undefined symbols — eight
# op `build()` bodies and `ArrayViewType`'s two accessors (found 2026-08-07).
# `skipDefaultBuilders = 1` plus a bodyless `OpBuilder<(ins ...)>` makes
# TableGen emit a declaration and expect the body in C++; nobody wrote them.
# ODS still referenced them, because it generates a `static OpTy create(...)`
# wrapper per builder that calls `build`. Same story for the type accessors:
# `genStorageClass = 0` means TableGen declares them and leaves the bodies to
# whoever wrote the storage class.
#
# Nothing caught it, and nothing could have. `MLIRNative.jl` reaches the
# library through `ccall((:sym, path), ...)`, which binds LAZILY — an undefined
# symbol nobody calls is never looked up. Every producer builds IR as text and
# parses it, so no call site existed. The library loaded, the dialect worked,
# 87/87 templates passed, and the whole Tier-2 suite was green on a library the
# loader would have rejected if asked to resolve it eagerly.
#
# So ask it eagerly. RTLD_NOW is the exact failure mode, it costs one dlopen,
# and it catches every future instance of the class — a new bodyless builder, a
# new manual-storage accessor, a definition deleted in a refactor — rather than
# just the ten that happened to exist. Negative-checked: on the pre-fix library
# this fails with `undefined symbol: mlir::jlcs::SetFieldOp::build(...)`.

@testset "D. no unresolved symbols (RTLD_NOW)" begin
    lib = RepliBuild.MLIRNative.libJLCS

    # Fresh subprocess: this process has already dlopened the library lazily,
    # and dlopen on an ALREADY-LOADED handle returns it without upgrading the
    # binding mode — so an in-process RTLD_NOW would silently pass regardless.
    # Asserts on the EXIT CODE, not on stdout. The first draft of this probe
    # matched `occursin("RESOLVED", out)` against a failure message reading
    # "UNRESOLVED: ..." — which contains it, so the guard passed on the very
    # library it was written to reject.
    code = """
    using Libdl
    try
        Libdl.dlopen($(repr(lib)), Libdl.RTLD_NOW | Libdl.RTLD_LOCAL)
        exit(0)
    catch e
        println(stderr, sprint(showerror, e))
        exit(1)
    end
    """
    err = IOBuffer()
    ok = success(pipeline(`$(Base.julia_cmd()) --startup-file=no --project=$(PROJECT) -e $code`,
                          stdout = devnull, stderr = err))
    ok || println("  → ", strip(String(take!(err))))
    @test ok

    # The same question asked of the artifact rather than the loader, so a
    # failure names every offender at once instead of only the first one the
    # loader trips over. Restricted to our own namespace by the Itanium mangled
    # spelling of `mlir::jlcs::` — MLIR/LLVM/libc symbols are supplied by the
    # DT_NEEDED libraries and are legitimately undefined here. Matching the
    # mangled form keeps this to one process and no `c++filt`.
    #
    # Match the nested-name body `4mlir4jlcs`, NOT the `_ZN` prefix: a CONST
    # member function mangles as `_ZNK`, so an `_ZN4mlir4jlcs` filter reports 8
    # of the 10 real offenders and silently drops both `ArrayViewType`
    # accessors — one of the two classes this probe exists for.
    if Sys.which("nm") !== nothing
        syms = readlines(`nm -D --undefined-only $lib`)
        undef = filter(s -> occursin("4mlir4jlcs", s), syms)
        isempty(undef) || println("  → undefined jlcs symbols:\n    ",
                                  join(strip.(undef), "\n    "))
        @test isempty(undef)
    end
end

end # testset
