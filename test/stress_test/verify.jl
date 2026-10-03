# test/stress_test/verify.jl — Stress test verification
#
# Covers: numerics (dense matrix, vectors, stats), vtable dispatch,
#         and (conditionally) MLIR / AOT / RAII dialect operations.

using Test

# ── Load wrapper ──────────────────────────────────────────────────────────────

wrapper_path = joinpath(@__DIR__, "julia", "StressTest.jl")
if !isfile(wrapper_path)
    error("Wrapper not found at $wrapper_path. Did you run build + wrap?")
end

include(wrapper_path)
using .StressTest

# ── Helpers ───────────────────────────────────────────────────────────────────

function heap_alloc(mat::StressTest.DenseMatrix)
    buf = Libc.malloc(sizeof(StressTest.DenseMatrix))
    ptr = Ptr{StressTest.DenseMatrix}(buf)
    unsafe_store!(ptr, mat)
    return ptr
end

# ══════════════════════════════════════════════════════════════════════════════
# 1. NUMERICS
# ══════════════════════════════════════════════════════════════════════════════

@testset "StressTest: Numerics" begin

    @testset "DenseMatrix (JIT struct return)" begin
        mat = StressTest.dense_matrix_create(Csize_t(3), Csize_t(3))
        mat_ptr = heap_alloc(mat)
        StressTest.dense_matrix_set_identity(mat_ptr)

        @test StressTest.matrix_trace(mat_ptr) == 3.0

        mat_t = StressTest.matrix_transpose(mat_ptr)
        mat_t_ptr = heap_alloc(mat_t)
        @test StressTest.matrix_trace(mat_t_ptr) == 3.0

        mat_sq = StressTest.matrix_multiply(mat_ptr, mat_ptr)
        mat_sq_ptr = heap_alloc(mat_sq)
        @test StressTest.matrix_trace(mat_sq_ptr) == 3.0

        mat_sum = StressTest.matrix_add(mat_ptr, mat_ptr)
        mat_sum_ptr = heap_alloc(mat_sum)
        @test StressTest.matrix_trace(mat_sum_ptr) == 6.0

        mat_copy = StressTest.dense_matrix_copy(mat_ptr)
        mat_copy_ptr = heap_alloc(mat_copy)
        @test StressTest.matrix_trace(mat_copy_ptr) == 3.0

        for p in [mat_ptr, mat_t_ptr, mat_sq_ptr, mat_sum_ptr, mat_copy_ptr]
            StressTest.dense_matrix_destroy(p)
            Libc.free(p)
        end
    end

    @testset "Statistics (ccall)" begin
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        @test StressTest.compute_mean(data, Csize_t(5)) == 3.0
        @test StressTest.compute_median(data, Csize_t(5)) == 3.0
        @test StressTest.compute_stddev(data, Csize_t(5)) > 0.0
    end

    @testset "Vector operations (ccall)" begin
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        @test StressTest.vector_dot(pointer(a), pointer(b), Csize_t(3)) == 32.0
        @test StressTest.vector_norm(pointer(a), Csize_t(3)) ≈ sqrt(14.0)
    end

end

# ══════════════════════════════════════════════════════════════════════════════
# 2. VTABLE DISPATCH (from vtable_test)
# ══════════════════════════════════════════════════════════════════════════════

@testset "StressTest: VTable Dispatch" begin
    rect_ptr = StressTest.create_rectangle(10.0, 20.0)
    @test rect_ptr != C_NULL

    circle_ptr = StressTest.create_circle(5.0)
    @test circle_ptr != C_NULL

    @test StressTest.get_area(rect_ptr) ≈ 200.0
    @test StressTest.get_perimeter(rect_ptr) ≈ 60.0

    @test StressTest.get_area(circle_ptr) ≈ (π * 25.0)
    @test StressTest.get_perimeter(circle_ptr) ≈ (2π * 5.0)

    StressTest.delete_shape(rect_ptr)
    StressTest.delete_shape(circle_ptr)

end

# ══════════════════════════════════════════════════════════════════════════════
# 3. MLIR / AOT / RAII (conditional — requires libJLCS)
# ══════════════════════════════════════════════════════════════════════════════

const MLIR_AVAILABLE = try
    using RepliBuild
    isfile(RepliBuild.MLIRNative.libJLCS)
catch
    false
end

if MLIR_AVAILABLE
    using RepliBuild.MLIRNative
    using RepliBuild.DWARFParser
    using RepliBuild.JLCSIRGenerator

    # ── MLIR IR Generation ────────────────────────────────────────────────

    @testset "StressTest: MLIR IR Generation" begin
        vm = VirtualMethod("foo", "_ZN4Base3fooEv", 0, "int", [])
        ci = ClassInfo("Base", 0, String[], [vm], MemberInfo[], 8)

        ir_type = JLCSIRGenerator.generate_type_info_ir("Base", ci, UInt64(0x1000))
        @test contains(ir_type, "jlcs.type_info \"Base\"")

        ir_method = JLCSIRGenerator.generate_virtual_method_ir(vm, UInt64(0x2000))
        @test contains(ir_method, "thunk__ZN4Base3fooEv")

        ctx = create_context()
        @test ctx != C_NULL

        mod = create_module(ctx)
        @test mod != C_NULL

        valid_ir = """
        module {
            func.func @test(%arg0: i32) -> i32 {
                return %arg0 : i32
            }
        }
        """
        parsed_mod = parse_module(ctx, valid_ir)
        @test parsed_mod != C_NULL

        # Generate & parse round-trip
        vm2 = VirtualMethod("bar", "_ZN4Base3barEv", 0, "void", ["int"])
        ci2 = ClassInfo("Base", 0, String[], [vm2], MemberInfo[], 8)
        classes = Dict("Base" => ci2)
        vtable_addrs = Dict("Base" => UInt64(0x1000))
        method_addrs = Dict("_ZN4Base3barEv" => UInt64(0x2000))
        vtinfo = VtableInfo(classes, vtable_addrs, method_addrs)
        generated_ir = generate_jlcs_ir(vtinfo)
        parsed = parse_module(ctx, generated_ir)
        @test parsed != C_NULL

        destroy_context(ctx)
    end

    # ── AOT Compilation ───────────────────────────────────────────────────

    @testset "StressTest: AOT Compilation" begin
        using Libdl

        @testset "Basic emit_object" begin
            ctx = MLIRNative.create_context()
            try
                mod_str = "module { func.func @my_thunk(%arg0: i32) -> i32 { return %arg0 : i32 } }"
                mod = MLIRNative.parse_module(ctx, mod_str)
                @test mod != C_NULL
                @test MLIRNative.lower_to_llvm(mod)

                obj_path = tempname() * ".o"
                @test MLIRNative.emit_object(mod, obj_path)
                @test isfile(obj_path) && filesize(obj_path) > 0
                rm(obj_path, force=true)
            finally
                MLIRNative.destroy_context(ctx)
            end
        end

        @testset "VTable thunks AOT pipeline" begin
            lib_path = joinpath(@__DIR__, "julia", "libstress_test." * Libdl.dlext)
            metadata_path = joinpath(@__DIR__, "julia", "compilation_metadata.json")

            @test isfile(lib_path)
            @test isfile(metadata_path)

            ctx = MLIRNative.create_context()
            try
                vtable_info = DWARFParser.parse_vtables(lib_path)
                metadata = RepliBuild._read_json(metadata_path)
                ir_source = JLCSIRGenerator.generate_jlcs_ir(vtable_info, metadata)
                @test !isempty(ir_source)

                mod = MLIRNative.parse_module(ctx, ir_source)
                @test mod != C_NULL
                @test MLIRNative.lower_to_llvm(mod)

                thunks_obj = joinpath(@__DIR__, "julia", "thunks.o")
                @test MLIRNative.emit_object(mod, thunks_obj)
                @test isfile(thunks_obj) && filesize(thunks_obj) > 0

                # This link was `gcc -shared -o … thunks.o` and nothing else,
                # which is a weaker link than the one ThunkBuilder actually
                # performs. Two separate reasons it could not hold on Windows:
                #
                #   * `gcc` is not a given. MSYS2 CLANG64 — the
                #     x86_64-w64-windows-gnu environment this port targets —
                #     ships clang and no gcc at all, so the command spawned
                #     nothing (ENOENT).
                #   * An ELF shared object may keep undefined symbols and let
                #     `dlopen` bind them later against an RTLD_GLOBAL library.
                #     PE has no such thing: every symbol in a DLL must resolve at
                #     LINK time. Unlinked, these thunks left ~17 undefined —
                #     the C++ runtime (`operator new`, `__cxa_begin_catch`), the
                #     wrapped library's own symbols (`Shape::Shape()`,
                #     `compute_eigen`), and libJLCS's exception shim.
                #
                # So link what the thunks actually reference, the way
                # Builder/ThunkBuilder.jl does: clang++ for the C++ runtime, the
                # library under test, and libJLCS. The `-l:` NEEDED has no
                # SONAME, so without `$ORIGIN` RUNPATH `dlopen` of the thunks
                # library cannot find the sibling even after
                # `dlopen(abspath(lib_path))`.
                cc      = RepliBuild.LLVMEnvironment.get_tool("clang++")
                lib_dir = joinpath(@__DIR__, "julia")
                jlcs    = RepliBuild.MLIRNative.libJLCS
                thunks_so = joinpath(lib_dir, "libstress_test_thunks." * Libdl.dlext)
                # Match ThunkBuilder: PE has no RUNPATH, sibling DLLs resolve
                # from the loading module's directory.
                if Sys.iswindows()
                    run(`$cc -shared -o $thunks_so $thunks_obj
                         -L$lib_dir -l:$(basename(lib_path)) $jlcs`)
                else
                    origin_rpath = "-Wl,-rpath,\$ORIGIN"
                    jlcs_rpath = "-Wl,-rpath," * dirname(jlcs)
                    run(`$cc -shared -o $thunks_so $thunks_obj
                         -L$lib_dir -l:$(basename(lib_path)) $jlcs
                         $origin_rpath -Wl,-rpath,$lib_dir $jlcs_rpath`)
                end
                @test isfile(thunks_so)

                main_lib = Libdl.dlopen(abspath(lib_path), Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
                @test main_lib != C_NULL
                thunks_lib = Libdl.dlopen(abspath(thunks_so), Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
                @test thunks_lib != C_NULL

                Libdl.dlclose(thunks_lib)
                Libdl.dlclose(main_lib)
            finally
                MLIRNative.destroy_context(ctx)
            end
        end

    end

    # ── RAII Dialect ──────────────────────────────────────────────────────

    @testset "StressTest: constructor by-value struct arguments" begin
        # DWARF names a constructor with the unified C4 linkage name and the symbol
        # table has C1/C2, so the exact-name join missed EVERY constructor. The
        # signature was then guessed from the demangled string, where `Point2D` is
        # only a name: both arguments became `Any` and the thunk stored garbage
        # (msdfgen's `LinearSegment(Vector2, Vector2, EdgeColor)`, 2026-09-26).
        meta = RepliBuild._read_json(joinpath(@__DIR__, "julia", "compilation_metadata.json"))
        ctor = first(f for f in meta["functions"] if occursin("geom9Segment2DC", f["mangled"]))
        @test ctor["parameters_source"] == "dwarf"
        @test [(p["name"], p["c_type"]) for p in ctor["parameters"]] ==
              [("this", "Segment2D*"), ("from", "Point2D"), ("to", "Point2D")]

        seg = zeros(Float64, 4)                      # geom::Segment2D { Point2D a, b; }
        GC.@preserve seg begin
            StressTest.geom_Segment2D_Segment2D(Ptr{StressTest.Segment2D}(pointer(seg)),
                                                StressTest.Point2D(1.0, 2.0),
                                                StressTest.Point2D(4.0, 6.0))
            @test seg == [1.0, 2.0, 4.0, 6.0]
            @test StressTest.geom_Segment2D_length2(Ptr{StressTest.Segment2D}(pointer(seg))) == 25.0
        end
    end

    @testset "StressTest: by-value records without a scalar spelling" begin
        # Each of these reached the thunk as `!llvm.ptr` (2026-09-27). The empty
        # class shifted every later argument one register (`unit_scaled` gave
        # 257-style garbage, never an error); the template record went in as a
        # pointer and came back from RAX while the callee wrote XMM0; the
        # string_view was one pointer where the callee reads {len, ptr}.
        meta = RepliBuild._read_json(joinpath(@__DIR__, "julia", "compilation_metadata.json"))
        @test meta["record_abi"]["Unit"] == Dict("byte_size" => "0x1", "pass" => "value", "empty" => true)

        @test StressTest.geom_unit_scaled(StressTest.Unit(), 4, StressTest.Unit(), 2) == 42
        @test StressTest.geom_unit_make(7) === StressTest.Unit()     # void callee, singleton back
        @test StressTest.geom_cell_doubled(StressTest.Cell_double(2.5)).v == 5.0

        # No layout in the metadata: refused where it is called, with the
        # reason, and the rest of the module still loads.
        err = try StressTest.geom_name_length("hello"); nothing catch e; e end
        @test err isa ErrorException
        @test occursin("ABI Safety Trap", sprint(showerror, err))
        @test occursin("string_view", sprint(showerror, err))

        # DW_CC_pass_by_reference: returned through an sret slot the thunk owns.
        @test meta["record_abi"]["Ticket"]["pass"] == "reference"
        @test StressTest.geom_ticket_make(42).n == 42
    end

    @testset "StressTest: packed structs hold the C layout" begin
        # 5 and 14 bytes, like C — not the natural 8 and 16.
        @test sizeof(StressTest.Packed) == 5
        @test sizeof(StressTest.Pack2) == 14
        p = StressTest.geom_packed_make(5)
        @test (Int(p.c), p.i) == (5, 6)
        @test StressTest.geom_packed_sum(p) == 11.0
        q = StressTest.geom_pack2_make(5)
        @test (Int(q.c), q.i, q.d) == (5, 6, 5.5)
        @test StressTest.geom_pack2_sum(q) == 16.5
        # Reads through a pointer into C memory: where natural alignment was wrong.
        t = Ptr{StressTest.Packed}(StressTest.geom_packed_table())
        @test [unsafe_load(t, k).i for k in 1:3] == [10, 20, 30]
    end

    @testset "StressTest: global-scope constructor" begin
        # At global scope `class` is the bare name, and the old "name == class,
        # skip it" rule dropped exactly these constructors (audit F25,
        # 2026-09-26). Same naming as geom's: Span_Span.
        @test isdefined(StressTest, :Span_Span)
        s = zeros(Int32, 2)                          # Span { int lo, hi; }
        GC.@preserve s begin
            p = Ptr{StressTest.Span}(pointer(s))
            StressTest.Span_Span(p, 3, 10)
            @test s == Int32[3, 10]
            @test StressTest.Span_width(p) == 7
        end
    end

    @testset "StressTest: anonymous union member" begin
        # The anonymous union is an immutable 4-byte region inlined in its
        # parent, so Variant keeps named fields, is 16 bytes like C, and `d`
        # sits at offset 8. As a `mutable struct` the union was a reference
        # field and the record could not hold the C layout (audit F22).
        @test hasfield(StressTest.Variant, :u) && hasfield(StressTest.Variant, :d)
        @test sizeof(StressTest.Variant) == 16
        if hasfield(StressTest.Variant, :u)
            U = fieldtype(StressTest.Variant, :u)
            @test !ismutabletype(U)
            @test sizeof(U) == 4
            v = StressTest.geom_variant_make(5)
            @test v.d == 5.25
            @test reinterpret(Int32, collect(v.u.data))[1] == 5
            @test StressTest.geom_variant_sum(v) == 10.25
        end
    end

    @testset "StressTest: float-array member" begin
        # `float v[3]` records 12 bytes, so Arr3 keeps its float field and comes
        # back from XMM as floats, not as an integer byte blob (audit F21).
        @test hasfield(StressTest.Arr3, :v)
        if hasfield(StressTest.Arr3, :v)
            @test fieldtype(StressTest.Arr3, :v) == NTuple{3, Cfloat}
            a = StressTest.geom_arr3_make(1f0)
            @test collect(a.v) == Float32[1, 2, 3]
            @test StressTest.geom_arr3_sum(StressTest.Arr3((1f0, 2f0, 3f0))) == 6.0
        end
    end

    @testset "StressTest: RAII Dialect" begin
        @testset "Parse ctor_call / dtor_call IR" begin
            ctx = create_context()
            ir = """
            module {
              func.func private @tracker_ctor(!llvm.ptr)
              func.func private @tracker_dtor(!llvm.ptr)

              func.func @test_raii(%ptr: !llvm.ptr) attributes {llvm.emit_c_interface} {
                jlcs.ctor_call @tracker_ctor(%ptr) : (!llvm.ptr) -> ()
                jlcs.dtor_call @tracker_dtor(%ptr) : (!llvm.ptr) -> ()
                return
              }
            }
            """
            mod = parse_module(ctx, ir)
            @test mod != C_NULL
            destroy_context(ctx)
        end

        @testset "Lower ctor_call / dtor_call to LLVM" begin
            ctx = create_context()
            ir = """
            module {
              func.func private @tracker_ctor(!llvm.ptr)
              func.func private @tracker_dtor(!llvm.ptr)

              func.func @test_raii(%ptr: !llvm.ptr) attributes {llvm.emit_c_interface} {
                jlcs.ctor_call @tracker_ctor(%ptr) : (!llvm.ptr) -> ()
                jlcs.dtor_call @tracker_dtor(%ptr) : (!llvm.ptr) -> ()
                return
              }
            }
            """
            mod = parse_module(ctx, ir)
            mod_jit = clone_module(mod)
            @test lower_to_llvm(mod_jit) == true
            destroy_context(ctx)
        end

        @testset "TypeInfoOp with destructor name" begin
            ctx = create_context()
            ir = """
            module {
              jlcs.type_info "Tracker",
                !jlcs.c_struct<"Tracker", [i32, i32], [[0 : i64, 4 : i64]], packed = false>,
                "", "_ZN7TrackerD1Ev"
            }
            """
            mod = parse_module(ctx, ir)
            @test mod != C_NULL

            mod_jit = clone_module(mod)
            @test lower_to_llvm(mod_jit)

            destroy_context(ctx)
        end

    end

else
    @info "libJLCS not found — skipping MLIR/AOT/RAII tests"
end

