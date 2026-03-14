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

    println("  ✓ numerics")
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

    println("  ✓ vtable dispatch")
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
        println("  ✓ MLIR IR generation")
    end

    # ── MLIR Type Safety ──────────────────────────────────────────────────

    @testset "StressTest: MLIR Type Safety" begin
        ctx = create_context()

        ir = """
        module {
            func.func @add(%arg0: i32, %arg1: i32) -> i32 attributes {llvm.emit_c_interface} {
                %0 = arith.addi %arg0, %arg1 : i32
                return %0 : i32
            }
        }
        """

        mod = parse_module(ctx, ir)
        mod_jit = clone_module(mod)
        @test lower_to_llvm(mod_jit) == true

        jit = create_jit(mod_jit)

        # Safe invocation
        res_ref = Ref(Int32(0))
        invoke_safe(jit, mod, "add", Int32(10), Int32(20), res_ref)
        @test res_ref[] == 30

        # Type mismatch detection
        bad_ref = Ref(Int32(0))
        @test_throws ErrorException invoke_safe(jit, mod, "add", Int64(10), Int32(20), bad_ref)
        @test_throws ErrorException invoke_safe(jit, mod, "add", Float32(10.0), Int32(20), bad_ref)

        destroy_jit(jit)
        destroy_context(ctx)
        println("  ✓ MLIR type safety")
    end

    # ── AOT Compilation ───────────────────────────────────────────────────

    @testset "StressTest: AOT Compilation" begin
        using JSON
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
            lib_path = joinpath(@__DIR__, "julia", "libstress_test.so")
            metadata_path = joinpath(@__DIR__, "julia", "compilation_metadata.json")

            @test isfile(lib_path)
            @test isfile(metadata_path)

            ctx = MLIRNative.create_context()
            try
                vtable_info = DWARFParser.parse_vtables(lib_path)
                metadata = JSON.parsefile(metadata_path)
                ir_source = JLCSIRGenerator.generate_jlcs_ir(vtable_info, metadata)
                @test !isempty(ir_source)

                mod = MLIRNative.parse_module(ctx, ir_source)
                @test mod != C_NULL
                @test MLIRNative.lower_to_llvm(mod)

                thunks_obj = joinpath(@__DIR__, "julia", "thunks.o")
                @test MLIRNative.emit_object(mod, thunks_obj)
                @test isfile(thunks_obj) && filesize(thunks_obj) > 0

                thunks_so = joinpath(@__DIR__, "julia", "libstress_test_thunks.so")
                run(`gcc -shared -o $thunks_so $thunks_obj`)
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

        println("  ✓ AOT compilation")
    end

    # ── RAII Dialect ──────────────────────────────────────────────────────

    @testset "StressTest: RAII Dialect" begin
        # Build libtracker.so from tracker.cpp in stress_test/src
        tracker_lib = joinpath(@__DIR__, "julia", "libstress_test.so")

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

        @testset "JIT execute ctor_call / dtor_call" begin
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
            @test lower_to_llvm(mod_jit)

            jit = create_jit(mod_jit, shared_libs=[tracker_lib])
            @test jit != C_NULL

            tracker = Int32[0, 0]
            tracker_ptr = Ref(Ptr{Cvoid}(pointer(tracker)))
            args = [Base.unsafe_convert(Ptr{Cvoid}, tracker_ptr)]
            GC.@preserve tracker tracker_ptr begin
                @test jit_invoke(jit, "test_raii", args) == true
            end

            @test tracker[1] == 42   # tracker_ctor
            @test tracker[2] == 99   # tracker_dtor

            destroy_jit(jit)
            destroy_context(ctx)
        end

        @testset "ctor_call with arguments" begin
            ctx = create_context()
            ir = """
            module {
              func.func private @tracker_ctor_val(!llvm.ptr, i32)
              func.func private @tracker_dtor(!llvm.ptr)

              func.func @test_raii_val(%ptr: !llvm.ptr, %val: i32) attributes {llvm.emit_c_interface} {
                jlcs.ctor_call @tracker_ctor_val(%ptr, %val) : (!llvm.ptr, i32) -> ()
                jlcs.dtor_call @tracker_dtor(%ptr) : (!llvm.ptr) -> ()
                return
              }
            }
            """
            mod = parse_module(ctx, ir)
            mod_jit = clone_module(mod)
            @test lower_to_llvm(mod_jit)

            jit = create_jit(mod_jit, shared_libs=[tracker_lib])
            @test jit != C_NULL

            tracker = Int32[0, 0]
            val = Ref(Int32(123))
            tracker_ptr = Ref(Ptr{Cvoid}(pointer(tracker)))
            args = [Base.unsafe_convert(Ptr{Cvoid}, tracker_ptr),
                    Base.unsafe_convert(Ptr{Cvoid}, val)]
            GC.@preserve tracker tracker_ptr val begin
                @test jit_invoke(jit, "test_raii_val", args) == true
            end

            @test tracker[1] == 123
            @test tracker[2] == 99

            destroy_jit(jit)
            destroy_context(ctx)
        end

        @testset "jlcs.scope (parse, lower, JIT)" begin
            ctx = create_context()
            ir = """
            module {
              func.func private @tracker_ctor(!llvm.ptr)
              func.func private @tracker_dtor(!llvm.ptr)

              func.func @test_scope(%ptr: !llvm.ptr) attributes {llvm.emit_c_interface} {
                jlcs.scope(%ptr : !llvm.ptr) dtors([@tracker_dtor]) {
                  jlcs.ctor_call @tracker_ctor(%ptr) : (!llvm.ptr) -> ()
                }
                return
              }
            }
            """
            mod = parse_module(ctx, ir)
            @test mod != C_NULL

            mod_jit = clone_module(mod)
            @test lower_to_llvm(mod_jit)

            jit = create_jit(mod_jit, shared_libs=[tracker_lib])
            @test jit != C_NULL

            tracker = Int32[0, 0]
            tracker_ptr = Ref(Ptr{Cvoid}(pointer(tracker)))
            args = [Base.unsafe_convert(Ptr{Cvoid}, tracker_ptr)]
            GC.@preserve tracker tracker_ptr begin
                @test jit_invoke(jit, "test_scope", args) == true
            end

            @test tracker[1] == 42   # ctor inside scope
            @test tracker[2] == 99   # dtor at scope exit

            destroy_jit(jit)
            destroy_context(ctx)
        end

        @testset "Scope with multiple managed objects" begin
            ctx = create_context()
            ir = """
            module {
              func.func private @tracker_ctor(!llvm.ptr)
              func.func private @tracker_dtor(!llvm.ptr)

              func.func @test_scope_multi(%ptr1: !llvm.ptr, %ptr2: !llvm.ptr) attributes {llvm.emit_c_interface} {
                jlcs.scope(%ptr1, %ptr2 : !llvm.ptr, !llvm.ptr) dtors([@tracker_dtor, @tracker_dtor]) {
                  jlcs.ctor_call @tracker_ctor(%ptr1) : (!llvm.ptr) -> ()
                  jlcs.ctor_call @tracker_ctor(%ptr2) : (!llvm.ptr) -> ()
                }
                return
              }
            }
            """
            mod = parse_module(ctx, ir)
            mod_jit = clone_module(mod)
            @test lower_to_llvm(mod_jit)

            jit = create_jit(mod_jit, shared_libs=[tracker_lib])
            @test jit != C_NULL

            tracker1 = Int32[0, 0]
            tracker2 = Int32[0, 0]
            ptr1 = Ref(Ptr{Cvoid}(pointer(tracker1)))
            ptr2 = Ref(Ptr{Cvoid}(pointer(tracker2)))
            args = [Base.unsafe_convert(Ptr{Cvoid}, ptr1),
                    Base.unsafe_convert(Ptr{Cvoid}, ptr2)]
            GC.@preserve tracker1 tracker2 ptr1 ptr2 begin
                @test jit_invoke(jit, "test_scope_multi", args) == true
            end

            @test tracker1[2] == 99
            @test tracker2[2] == 99

            destroy_jit(jit)
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

        println("  ✓ RAII dialect")
    end

else
    @info "libJLCS not found — skipping MLIR/AOT/RAII tests"
end

println("\n✅ stress_test: verification complete")
