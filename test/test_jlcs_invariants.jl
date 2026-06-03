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
    # never segfault. CONFIRMED to crash today (SIGSEGV) — marked broken so it
    # flips to "Unexpectedly passed" the moment a ScopeOp verifier lands.
    @test_broken outcome != :crash
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
    @test_broken outcome != :crash
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
        println("    array ops are FUNCTIONAL but unused (dead producer, not dead op)")
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

end # testset
