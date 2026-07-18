#!/usr/bin/env julia
# test/test_struct_abi.jl — trace tests for the pugixml JIT-init segfault and
# the by-value small-struct ABI (docs/pugixml-jit-init-segfault.md, 2026-07-18).
#
# Three regressions pinned, each of which was a live crash or miscompile:
#
#  A. NESTED PACKED STRUCT (the segfault). StructGen emits padding-free
#     structs as !jlcs.c_struct; referencing one via its alias from inside an
#     !llvm.struct body survives lowering (the type converter treats
#     !llvm.struct as legal and never rewrites its body) and SIGSEGVs
#     translateModuleToLLVMIR (PtrLikeTypeInterface::getMemorySpace). Fixed by
#     inlining the byte-identical LLVM packed literal instead of the alias.
#
#  B. JIT PRE-FLIGHT GUARD. Even when a foreign type sneaks through, create_jit
#     must fail catchably (null → Julia error → Tier 2 disabled), not kill the
#     process inside translateModuleToLLVMIR.
#
#  C. SysV SMALL-STRUCT ABI. try_call/ffe_call lowering forced sret for EVERY
#     "packed" (padding-free) struct return — but ≤16-byte aligned structs are
#     register-class: native code returns {void*} in RAX, so the sret call
#     shifted `this` into the sret slot and returned stack garbage (pugixml
#     first_child, found live 2026-07-18). Register-class structs now coerce
#     one scalar per eightbyte, clang-style, for returns AND by-value args.
#     Verified against a REAL clang++-compiled callee — self-JIT'd callees
#     share the JIT's own convention and cannot catch a mismatch.
#
# Requires libJLCS.so + clang++ (devtests tier); skips cleanly without them.

using Test
using Libdl

const MLIR_AVAILABLE = try
    using RepliBuild
    isfile(RepliBuild.MLIRNative.libJLCS)
catch
    false
end

if !MLIR_AVAILABLE
    @info "libJLCS not found — skipping struct-ABI trace tests"
    exit(0)
end

const CLANGXX = Sys.which("clang++")
if CLANGXX === nothing
    @info "clang++ not found — skipping struct-ABI trace tests"
    exit(0)
end

using RepliBuild.MLIRNative
using RepliBuild.JLCSIRGenerator
using RepliBuild.DWARFParser

# ── Fixture build ─────────────────────────────────────────────────────────────

const ABI_DIR = joinpath(@__DIR__, "struct_abi")
const ABI_SRC = joinpath(ABI_DIR, "src", "abi_fixture.cpp")
const ABI_LIB = joinpath(ABI_DIR, "libabi_fixture.so")

if !isfile(ABI_LIB) || mtime(ABI_SRC) > mtime(ABI_LIB)
    run(`$CLANGXX -shared -fPIC -O1 -o $ABI_LIB $ABI_SRC`)
end
const JLCS = RepliBuild.MLIRNative.libJLCS

@testset "struct ABI traces" begin

# ── A. nested packed struct lowers + JITs (was: whole-process SIGSEGV) ───────
@testset "A: c_struct nested in llvm.struct body" begin
    vtinfo = DWARFParser.VtableInfo(Dict{String,DWARFParser.ClassInfo}(),
                                    Dict{String,UInt64}(), Dict{String,UInt64}())
    inner = Dict{String,Any}("kind" => "struct", "byte_size" => "0x8",
        "members" => [Dict{String,Any}("name" => "x", "c_type" => "long", "size" => 8, "offset" => 0)])
    outer = Dict{String,Any}("kind" => "struct", "byte_size" => "0x18",
        "members" => [Dict{String,Any}("name" => "a", "c_type" => "Inner", "size" => 8, "offset" => 0),
                      Dict{String,Any}("name" => "b", "c_type" => "int", "size" => 4, "offset" => 8)])
    mkfn(name; ret="void", params=String[]) = Dict{String,Any}(
        "mangled" => name, "name" => name, "demangled" => "$(name)()",
        "return_type" => Dict{String,Any}("c_type" => ret, "size" => 0, "julia_type" => "Any"),
        "parameters" => [Dict{String,Any}("name" => "p$i", "c_type" => p, "size" => 0)
                         for (i, p) in enumerate(params)],
        "is_method" => false, "is_vararg" => false, "exported" => true, "is_noexcept" => false)
    metadata = Dict{String,Any}(
        "language" => "c++",
        "struct_definitions" => Dict{String,Any}("Inner" => inner, "Outer" => outer),
        "functions" => Any[mkfn("make_outer"; ret="Outer"),
                           mkfn("take_outer"; params=["Outer"])])

    ir = JLCSIRGenerator.generate_jlcs_ir(vtinfo, metadata)
    # The packed member must be inlined as an LLVM literal, not alias-referenced
    @test occursin("!llvm.struct<packed (i64)>", ir)
    @test !occursin(r"!llvm\.struct<\"Outer\", \(!Struct_Inner", ir)

    ctx = create_context()
    mod = parse_module(ctx, ir)
    @test lower_to_llvm(mod)
    jit = create_jit(mod, opt_level=1, shared_libs=[JLCS])
    @test jit != C_NULL
    destroy_jit(jit)
    destroy_context(ctx)
end

# ── B. pre-flight guard: bad type → catchable error, not SIGSEGV ─────────────
# This is the EXACT shape that segfaulted the whole process pre-fix: the
# hybrid type lives in a BODY op (llvm.load result). Signature positions get
# rewritten by the type converter ("_Converted.<name>"), but llvm.* ops are
# already legal during conversion, so their types are never revisited and the
# foreign type reaches translateModuleToLLVMIR. create_jit must now refuse it
# with a catchable Julia error instead of dying in getMemorySpace.
@testset "B: create_jit refuses untranslatable types" begin
    bad_ir = """
    !GInner = !jlcs.c_struct<"GuardInner", [i64], [[0 : i64]], packed = true>
    !GOuter = !llvm.struct<"GuardOuter", (!GInner, i32, !llvm.array<12 x i8>)>
    module {
      func.func private @guard_take(!GOuter)
      func.func @guard_take_thunk(%args_ptr: !llvm.ptr) attributes { llvm.emit_c_interface } {
        %idx = arith.constant 0 : i64
        %ap = llvm.getelementptr %args_ptr[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
        %vp = llvm.load %ap : !llvm.ptr -> !llvm.ptr
        %v = llvm.load %vp : !llvm.ptr -> !GOuter
        jlcs.try_call %v { callee = @guard_take } : (!GOuter) -> ()
        return
      }
    }
    """
    ctx = create_context()
    mod = parse_module(ctx, bad_ir)          # parses fine — that's the trap
    lower_to_llvm(mod)                       # body op keeps the foreign type
    @test_throws ErrorException create_jit(mod, opt_level=1, shared_libs=[JLCS])
    destroy_context(ctx)
end

# ── C. small-struct SysV ABI against a real clang++ callee ───────────────────
@testset "C: register-class struct returns/args through try_call" begin
    abi_ir = """
    module {
      func.func private @h1_make(!llvm.ptr) -> !llvm.struct<packed (!llvm.ptr)>
      func.func @h1_make_thunk(%args_ptr: !llvm.ptr) -> !llvm.struct<packed (!llvm.ptr)> attributes { llvm.emit_c_interface } {
        %i0 = arith.constant 0 : i64
        %s0 = llvm.getelementptr %args_ptr[%i0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
        %p0 = llvm.load %s0 : !llvm.ptr -> !llvm.ptr
        %v0 = llvm.load %p0 : !llvm.ptr -> !llvm.ptr
        %r = jlcs.try_call %v0 { callee = @h1_make } : (!llvm.ptr) -> !llvm.struct<packed (!llvm.ptr)>
        return %r : !llvm.struct<packed (!llvm.ptr)>
      }

      func.func private @p2_make(i32, i32) -> !llvm.struct<packed (i32, i32)>
      func.func @p2_make_thunk(%args_ptr: !llvm.ptr) -> !llvm.struct<packed (i32, i32)> attributes { llvm.emit_c_interface } {
        %i0 = arith.constant 0 : i64
        %i1 = arith.constant 1 : i64
        %s0 = llvm.getelementptr %args_ptr[%i0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
        %p0 = llvm.load %s0 : !llvm.ptr -> !llvm.ptr
        %a = llvm.load %p0 : !llvm.ptr -> i32
        %s1 = llvm.getelementptr %args_ptr[%i1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
        %p1 = llvm.load %s1 : !llvm.ptr -> !llvm.ptr
        %b = llvm.load %p1 : !llvm.ptr -> i32
        %r = jlcs.try_call %a, %b { callee = @p2_make } : (i32, i32) -> !llvm.struct<packed (i32, i32)>
        return %r : !llvm.struct<packed (i32, i32)>
      }

      func.func private @p2_sum(!llvm.struct<packed (i32, i32)>) -> i32
      func.func @p2_sum_thunk(%args_ptr: !llvm.ptr) -> i32 attributes { llvm.emit_c_interface } {
        %i0 = arith.constant 0 : i64
        %s0 = llvm.getelementptr %args_ptr[%i0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
        %p0 = llvm.load %s0 : !llvm.ptr -> !llvm.ptr
        %v = llvm.load %p0 : !llvm.ptr -> !llvm.struct<packed (i32, i32)>
        %r = jlcs.try_call %v { callee = @p2_sum } : (!llvm.struct<packed (i32, i32)>) -> i32
        return %r : i32
      }

      func.func private @f2_make(f32, f32) -> !llvm.struct<packed (f32, f32)>
      func.func @f2_make_thunk(%args_ptr: !llvm.ptr) -> !llvm.struct<packed (f32, f32)> attributes { llvm.emit_c_interface } {
        %i0 = arith.constant 0 : i64
        %i1 = arith.constant 1 : i64
        %s0 = llvm.getelementptr %args_ptr[%i0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
        %p0 = llvm.load %s0 : !llvm.ptr -> !llvm.ptr
        %x = llvm.load %p0 : !llvm.ptr -> f32
        %s1 = llvm.getelementptr %args_ptr[%i1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
        %p1 = llvm.load %s1 : !llvm.ptr -> !llvm.ptr
        %y = llvm.load %p1 : !llvm.ptr -> f32
        %r = jlcs.try_call %x, %y { callee = @f2_make } : (f32, f32) -> !llvm.struct<packed (f32, f32)>
        return %r : !llvm.struct<packed (f32, f32)>
      }

      func.func private @f2_sum(!llvm.struct<packed (f32, f32)>) -> f32
      func.func @f2_sum_thunk(%args_ptr: !llvm.ptr) -> f32 attributes { llvm.emit_c_interface } {
        %i0 = arith.constant 0 : i64
        %s0 = llvm.getelementptr %args_ptr[%i0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
        %p0 = llvm.load %s0 : !llvm.ptr -> !llvm.ptr
        %v = llvm.load %p0 : !llvm.ptr -> !llvm.struct<packed (f32, f32)>
        %r = jlcs.try_call %v { callee = @f2_sum } : (!llvm.struct<packed (f32, f32)>) -> f32
        return %r : f32
      }

      func.func private @b3_make(i64, i64, i64) -> !llvm.struct<(i64, i64, i64)>
      func.func @b3_make_thunk(%args_ptr: !llvm.ptr) -> !llvm.struct<(i64, i64, i64)> attributes { llvm.emit_c_interface } {
        %i0 = arith.constant 0 : i64
        %i1 = arith.constant 1 : i64
        %i2 = arith.constant 2 : i64
        %s0 = llvm.getelementptr %args_ptr[%i0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
        %p0 = llvm.load %s0 : !llvm.ptr -> !llvm.ptr
        %a = llvm.load %p0 : !llvm.ptr -> i64
        %s1 = llvm.getelementptr %args_ptr[%i1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
        %p1 = llvm.load %s1 : !llvm.ptr -> !llvm.ptr
        %b = llvm.load %p1 : !llvm.ptr -> i64
        %s2 = llvm.getelementptr %args_ptr[%i2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
        %p2 = llvm.load %s2 : !llvm.ptr -> !llvm.ptr
        %c = llvm.load %p2 : !llvm.ptr -> i64
        %r = jlcs.try_call %a, %b, %c { callee = @b3_make } : (i64, i64, i64) -> !llvm.struct<(i64, i64, i64)>
        return %r : !llvm.struct<(i64, i64, i64)>
      }
    }
    """
    ctx = create_context()
    mod = parse_module(ctx, abi_ir)
    @test mod != C_NULL
    @test lower_to_llvm(mod)
    jit = create_jit(mod, opt_level=1, shared_libs=[abspath(ABI_LIB), JLCS])
    @test jit != C_NULL

    ciface(name) = MLIRNative.lookup(jit, "_mlir_ciface_$(name)_thunk")

    # H1 {void*}: INTEGER eightbyte → RAX. Pre-fix this returned stack garbage.
    let fp = ciface("h1_make")
        @test fp != C_NULL
        marker = Ptr{Cvoid}(UInt(0xdeadbeef00abcdef))
        arg = Ref(marker)
        GC.@preserve arg begin
            slots = Ptr{Cvoid}[Base.unsafe_convert(Ptr{Cvoid}, arg)]
            out = Ref{NTuple{1,UInt64}}((0,))
            GC.@preserve slots out ccall(fp, Cvoid,
                (Ptr{NTuple{1,UInt64}}, Ptr{Ptr{Cvoid}}), out, slots)
            @test out[][1] == UInt(marker)
        end
    end

    # P2 {int,int}: both ints share RAX (LLVM's per-element lowering would
    # split them into EAX/EDX — the latent pre-fix mismatch).
    let fp = ciface("p2_make")
        a = Ref(Int32(41)); b = Ref(Int32(1))
        GC.@preserve a b begin
            slots = Ptr{Cvoid}[Base.unsafe_convert(Ptr{Cvoid}, a),
                               Base.unsafe_convert(Ptr{Cvoid}, b)]
            out = Ref{NTuple{2,Int32}}((0, 0))
            GC.@preserve slots out ccall(fp, Cvoid,
                (Ptr{NTuple{2,Int32}}, Ptr{Ptr{Cvoid}}), out, slots)
            @test out[] == (Int32(41), Int32(1))
        end
    end
    let fp = ciface("p2_sum")
        # By-value struct arg: the slot holds &struct DIRECTLY (thunk arg-slot
        # convention — see CLAUDE.md), not a pointer to a Ref of the pointer.
        val = Ref{NTuple{2,Int32}}((Int32(40), Int32(2)))
        GC.@preserve val begin
            vp = Base.unsafe_convert(Ptr{NTuple{2,Int32}}, val)
            slots = Ptr{Cvoid}[Ptr{Cvoid}(vp)]
            r = GC.@preserve slots ccall(fp, Int32, (Ptr{Ptr{Cvoid}},), slots)
            @test r == Int32(42)
        end
    end

    # F2 {float,float}: one SSE eightbyte — both floats travel in XMM0.
    let fp = ciface("f2_make")
        x = Ref(Float32(1.5)); y = Ref(Float32(2.25))
        GC.@preserve x y begin
            slots = Ptr{Cvoid}[Base.unsafe_convert(Ptr{Cvoid}, x),
                               Base.unsafe_convert(Ptr{Cvoid}, y)]
            out = Ref{NTuple{2,Float32}}((0f0, 0f0))
            GC.@preserve slots out ccall(fp, Cvoid,
                (Ptr{NTuple{2,Float32}}, Ptr{Ptr{Cvoid}}), out, slots)
            @test out[] == (Float32(1.5), Float32(2.25))
        end
    end
    let fp = ciface("f2_sum")
        val = Ref{NTuple{2,Float32}}((Float32(1.5), Float32(2.25)))
        GC.@preserve val begin
            vp = Base.unsafe_convert(Ptr{NTuple{2,Float32}}, val)
            slots = Ptr{Cvoid}[Ptr{Cvoid}(vp)]
            r = GC.@preserve slots ccall(fp, Float32, (Ptr{Ptr{Cvoid}},), slots)
            @test r == Float32(3.75)
        end
    end

    # B3 {long,long,long}: 24B MEMORY class — the sret path must stay intact
    # (this is the xml_parse_result shape that already worked).
    let fp = ciface("b3_make")
        a = Ref(Int64(7)); b = Ref(Int64(8)); c = Ref(Int64(9))
        GC.@preserve a b c begin
            slots = Ptr{Cvoid}[Base.unsafe_convert(Ptr{Cvoid}, a),
                               Base.unsafe_convert(Ptr{Cvoid}, b),
                               Base.unsafe_convert(Ptr{Cvoid}, c)]
            out = Ref{NTuple{3,Int64}}((0, 0, 0))
            GC.@preserve slots out ccall(fp, Cvoid,
                (Ptr{NTuple{3,Int64}}, Ptr{Ptr{Cvoid}}), out, slots)
            @test out[] == (Int64(7), Int64(8), Int64(9))
        end
    end

    destroy_jit(jit)
    destroy_context(ctx)
end

end # testset

println("✅ struct-ABI trace tests passed")
