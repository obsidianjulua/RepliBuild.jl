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

# ── D. MEMORY-class by-value args + emitted struct size (2026-08-05) ─────────
#
# Two defects that only a real clang++ callee can arbitrate, driven through the
# WHOLE generator (metadata → StructGen → FunctionGen → lowering) rather than
# hand-written IR, because the first of them lives in StructGen:
#
#  1. EMITTED SIZE. A non-packed struct used to be closed with one trailing
#     filler of `byte_size - sum(member sizes)`, double-counting the interior
#     alignment padding LLVM inserts anyway. `Gap` (24 bytes, members summing
#     to 19) came out 32. Since `emit_c_interface` stores a MEMORY-class result
#     straight into the caller's buffer, the thunk wrote 8 bytes past it — and
#     past the `Ref{T}` a real wrapper hands it. Measured on llama.cpp:
#     `llama_context_default_params` overran a 160-byte Ref by 34 bytes.
#
#  2. BY-VALUE ARGS. A MEMORY-class struct argument was passed as an LLVM
#     first-class aggregate (backend splits it per element across registers) or,
#     when packed, as a bare pointer. SysV wants a caller stack copy: `byval`.
#     llama.cpp's `llama_model_load_from_file(path, llama_model_params)`
#     segfaulted on the first shape.
#
# The sentinel probe is the load-bearing assertion for (1): a size check on the
# type string would pass on a body that is merely mis-PADDED, whereas "which
# bytes did the callee actually write" cannot be satisfied by the wrong layout.
@testset "D: MEMORY-class by-value args and exact struct size" begin
    vtinfo = DWARFParser.VtableInfo(Dict{String,DWARFParser.ClassInfo}(),
                                    Dict{String,UInt64}(), Dict{String,UInt64}())
    mem(n, c, o, s) = Dict{String,Any}("name" => n, "c_type" => c, "offset" => o, "size" => s)
    gap = Dict{String,Any}("kind" => "struct", "byte_size" => "0x18",
        "members" => [mem("a", "int", 0, 4), mem("p", "void*", 8, 8),
                      mem("b", "int", 16, 4), mem("f1", "bool", 20, 1),
                      mem("f2", "bool", 21, 1), mem("f3", "bool", 22, 1)])
    b3 = Dict{String,Any}("kind" => "struct", "byte_size" => "0x18",
        "members" => [mem("a", "long", 0, 8), mem("b", "long", 8, 8),
                      mem("c", "long", 16, 8)])
    fn(name, ret, params) = Dict{String,Any}(
        "mangled" => name, "name" => name, "demangled" => "$(name)()",
        "return_type" => Dict{String,Any}("c_type" => ret, "size" => 0, "julia_type" => "Any"),
        "parameters" => [Dict{String,Any}("name" => "p$i", "c_type" => p, "size" => 0)
                         for (i, p) in enumerate(params)],
        "is_method" => false, "is_vararg" => false, "exported" => true, "is_noexcept" => true)
    metadata = Dict{String,Any}("language" => "c",
        "struct_definitions" => Dict{String,Any}("Gap" => gap, "B3" => b3),
        "functions" => Any[fn("gap_make", "Gap", ["int", "void*", "int"]),
                           fn("gap_probe", "long", ["Gap"]),
                           fn("b3_sum", "long", ["B3"])])

    ir = JLCSIRGenerator.generate_jlcs_ir(vtinfo, metadata)
    # B3 has no interior padding (sum == byte_size) so it is "packed" by the
    # generator's test; Gap must carry EXPLICIT interior padding and no
    # oversized tail. Pre-fix it ended in `!llvm.array<5 x i8>`.
    @test occursin("!llvm.array<4 x i8>", ir)
    @test !occursin("!llvm.array<5 x i8>", ir)

    ctx = create_context()
    mod = parse_module(ctx, ir)
    @test mod != C_NULL
    @test lower_to_llvm(mod)
    jit = create_jit(mod, opt_level=1, shared_libs=[abspath(ABI_LIB), JLCS])
    @test jit != C_NULL
    ciface(name) = MLIRNative.lookup(jit, "_mlir_ciface_$(name)_thunk")

    # (1) sret return writes EXACTLY sizeof(Gap) == 24 bytes. The buffer is
    # sentinel-filled and oversized, so an overrun shows up as a touched byte
    # rather than as corruption somewhere else.
    let fp = ciface("gap_make")
        @test fp != C_NULL
        a = Ref(Int32(11)); p = Ref(Ptr{Cvoid}(UInt(0x1234))); b = Ref(Int32(22))
        buf = fill(0xAA, 128)
        GC.@preserve a p b buf begin
            slots = Ptr{Cvoid}[Base.unsafe_convert(Ptr{Cvoid}, a),
                               Base.unsafe_convert(Ptr{Cvoid}, p),
                               Base.unsafe_convert(Ptr{Cvoid}, b)]
            GC.@preserve slots ccall(fp, Cvoid, (Ptr{UInt8}, Ptr{Ptr{Cvoid}}), buf, slots)
        end
        last_written = something(findlast(i -> buf[i] != 0xAA, eachindex(buf)), 0)
        @test last_written == 24                       # pre-fix: 32
        @test reinterpret(Int32, buf[1:4])[1] == 11
        @test reinterpret(UInt64, buf[9:16])[1] == 0x1234
        @test reinterpret(Int32, buf[17:20])[1] == 22
        @test (buf[21], buf[22], buf[23]) == (0x01, 0x00, 0x01)
    end

    # (2) MEMORY-class struct passed BY VALUE, both with interior padding (Gap,
    # non-packed → used to cross as an LLVM first-class aggregate) and without
    # (B3, `is_struct_packed` → used to cross as a bare pointer). Every field
    # feeds the result, so a register-split or by-reference crossing is a wrong
    # NUMBER rather than a crash we might misread.
    #
    # Negative-checked against the pre-fix dialect: `gap_probe` returns
    # 22000000 instead of 11022105 — the aggregate was split per element, so
    # the callee read `b` where `a` should be and zeros for the rest. `b3_sum`
    # happens to survive the old bare-pointer path on this fixture (the thunk's
    # alloca lands where the callee looks for its stack copy), so it pins the
    # packed path's correctness without discriminating the fix. Gap is the
    # discriminating case; keep it that way if this fixture ever changes.
    let fp = ciface("gap_probe")
        @test fp != C_NULL
        g = zeros(UInt8, 24)
        g[1:4]   = reinterpret(UInt8, Int32[11])
        g[9:16]  = reinterpret(UInt8, UInt64[0x1234])
        g[17:20] = reinterpret(UInt8, Int32[22])
        g[21], g[22], g[23] = 0x01, 0x00, 0x01
        GC.@preserve g begin
            # Thunk arg-slot convention: the slot holds &struct DIRECTLY.
            slots = Ptr{Cvoid}[Ptr{Cvoid}(pointer(g))]
            r = GC.@preserve slots ccall(fp, Int64, (Ptr{Ptr{Cvoid}},), slots)
            @test r == 11 * 1000000 + 22 * 1000 + 1 + 4 + 100
        end
    end
    let fp = ciface("b3_sum")
        @test fp != C_NULL
        v = Ref{NTuple{3,Int64}}((Int64(100), Int64(20), Int64(3)))
        GC.@preserve v begin
            slots = Ptr{Cvoid}[Ptr{Cvoid}(Base.unsafe_convert(Ptr{NTuple{3,Int64}}, v))]
            r = GC.@preserve slots ccall(fp, Int64, (Ptr{Ptr{Cvoid}},), slots)
            @test r == 123
        end
    end

    destroy_jit(jit)
    destroy_context(ctx)
end

end # testset

println("✅ struct-ABI trace tests passed")
