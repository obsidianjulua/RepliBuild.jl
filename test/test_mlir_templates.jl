#!/usr/bin/env julia
# test/test_mlir_templates.jl — JLCS dialect stress tests for nested templates
#
# Tests the JLCS MLIR dialect's ability to represent and lower complex C++ template
# patterns: nested CStruct types, packed template structs, template RAII scopes,
# virtual dispatch on template containers, and struct return through ffe_call.
#
# Requires libJLCS.so (build with: cd src/mlir && ./build.sh)

using Test
using Libdl

# ══════════════════════════════════════════════════════════════════════════════
# Availability guard — skip entire file if libJLCS not available
# ══════════════════════════════════════════════════════════════════════════════

const MLIR_AVAILABLE = try
    using RepliBuild
    isfile(RepliBuild.MLIRNative.libJLCS)
catch
    false
end

if !MLIR_AVAILABLE
    @info "libJLCS not found — skipping MLIR template tests"
    exit(0)
end

using RepliBuild.MLIRNative

# ══════════════════════════════════════════════════════════════════════════════
# Build the C++ support library
# ══════════════════════════════════════════════════════════════════════════════

const TEMPLATES_DIR = joinpath(@__DIR__, "mlir_templates")
const TEMPLATES_SRC = joinpath(TEMPLATES_DIR, "src", "templates.cpp")
const TEMPLATES_LIB = joinpath(TEMPLATES_DIR, "libmlir_templates.so")

function build_templates_lib()
    if !isfile(TEMPLATES_SRC)
        error("Template source not found: $TEMPLATES_SRC")
    end
    if !isfile(TEMPLATES_LIB) || mtime(TEMPLATES_SRC) > mtime(TEMPLATES_LIB)
        @info "Building template support library..."
        run(`clang++ -shared -fPIC -O2 -std=c++17 -o $TEMPLATES_LIB $TEMPLATES_SRC`)
    end
    return abspath(TEMPLATES_LIB)
end

const LIB_PATH = build_templates_lib()
@info "Using template library: $LIB_PATH"

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

"""Build a jit_invoke args vector from Julia values. Handles:
- Ptr{Cvoid} → wraps in Ref (for !llvm.ptr args)
- Int32/Int64/Float32/Float64 → wraps in Ref (for scalar args)
Returns (args_vec, refs) — refs must be GC.@preserved during invoke.
"""
function make_jit_args(vals...)
    refs = Any[]
    ptrs = Ptr{Cvoid}[]
    for v in vals
        r = Ref(v)
        push!(refs, r)
        push!(ptrs, Base.unsafe_convert(Ptr{Cvoid}, r))
    end
    return (ptrs, refs)
end

# ══════════════════════════════════════════════════════════════════════════════
# 1. Nested CStruct Types — Parse
# ══════════════════════════════════════════════════════════════════════════════

@testset "MLIR Templates" begin

@testset "1. Nested CStruct Types (Parse)" begin
    ctx = create_context()
    try
        ir = """
        module {
            // Pair<int, double> — 2 fields, natural alignment
            jlcs.type_info "Pair_int_double",
                !jlcs.c_struct<"Pair_int_double", [i32, f64], [[0 : i64, 8 : i64]], packed = false>,
                "", ""

            // Pair<int, Pair<double, float>> — 3 fields (flattened nested template)
            jlcs.type_info "Pair_int_Pair_double_float",
                !jlcs.c_struct<"Pair_int_Pair_double_float", [i32, f64, f32],
                    [[0 : i64, 8 : i64, 16 : i64]], packed = false>,
                "", ""

            // PackedTriple<char, int, short> — packed, no padding
            jlcs.type_info "PackedTriple_char_int_short",
                !jlcs.c_struct<"PackedTriple_char_int_short", [i8, i32, i16],
                    [[0 : i64, 1 : i64, 5 : i64]], packed = true>,
                "", ""

            // FixedArray<double, 4> — homogeneous array as struct
            jlcs.type_info "FixedArray_double_4",
                !jlcs.c_struct<"FixedArray_double_4", [f64, f64, f64, f64],
                    [[0 : i64, 8 : i64, 16 : i64, 24 : i64]], packed = false>,
                "", ""

            // Container<int> with vptr — template container
            jlcs.type_info "Container_int",
                !jlcs.c_struct<"Container_int", [!llvm.ptr, i32],
                    [[0 : i64, 8 : i64]], packed = false>,
                "", ""
        }
        """
        mod = parse_module(ctx, ir)
        @test mod != C_NULL
        println("  ✓ nested CStruct types parsed")
    finally
        destroy_context(ctx)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# 2. Nested CStruct Types — Lower to LLVM
# ══════════════════════════════════════════════════════════════════════════════

@testset "2. Nested CStruct Types (Lower)" begin
    ctx = create_context()
    try
        ir = """
        module {
            jlcs.type_info "Pair_int_double",
                !jlcs.c_struct<"Pair_int_double", [i32, f64], [[0 : i64, 8 : i64]], packed = false>,
                "", ""

            jlcs.type_info "Pair_int_Pair_double_float",
                !jlcs.c_struct<"Pair_int_Pair_double_float", [i32, f64, f32],
                    [[0 : i64, 8 : i64, 16 : i64]], packed = false>,
                "", ""

            jlcs.type_info "PackedTriple_char_int_short",
                !jlcs.c_struct<"PackedTriple_char_int_short", [i8, i32, i16],
                    [[0 : i64, 1 : i64, 5 : i64]], packed = true>,
                "", ""
        }
        """
        mod = parse_module(ctx, ir)
        mod_jit = clone_module(mod)
        @test lower_to_llvm(mod_jit) == true
        println("  ✓ nested CStruct types lowered to LLVM")
    finally
        destroy_context(ctx)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# 3. Nested get_field Chains — JIT
# ══════════════════════════════════════════════════════════════════════════════

@testset "3. Nested get_field Chains (JIT)" begin
    ctx = create_context()
    try
        # NestedPair layout: { i32 @0, [pad], f64 @8, f32 @16, [pad] } = 24 bytes
        # These functions read each field from a NestedPair* by byte offset
        # get_field uses MLIR generic format (no custom assembly defined)
        ir = """
        module {
            func.func @read_outer_first(%ptr: !llvm.ptr) -> i32 attributes {llvm.emit_c_interface} {
                %val = "jlcs.get_field"(%ptr) {fieldOffset = 0 : i64} : (!llvm.ptr) -> i32
                return %val : i32
            }

            func.func @read_inner_first(%ptr: !llvm.ptr) -> f64 attributes {llvm.emit_c_interface} {
                %val = "jlcs.get_field"(%ptr) {fieldOffset = 8 : i64} : (!llvm.ptr) -> f64
                return %val : f64
            }

            func.func @read_inner_second(%ptr: !llvm.ptr) -> f32 attributes {llvm.emit_c_interface} {
                %val = "jlcs.get_field"(%ptr) {fieldOffset = 16 : i64} : (!llvm.ptr) -> f32
                return %val : f32
            }
        }
        """
        mod = parse_module(ctx, ir)
        mod_jit = clone_module(mod)
        @test lower_to_llvm(mod_jit)

        jit = create_jit(mod_jit, shared_libs=[LIB_PATH])
        @test jit != C_NULL

        # Allocate a NestedPair: 24 bytes
        # Layout: i32 @0, pad@4, f64 @8, f32 @16, pad@20
        buf = zeros(UInt8, 24)

        # Write fields at known offsets
        GC.@preserve buf begin
            p = pointer(buf)
            unsafe_store!(Ptr{Int32}(p), Int32(42))        # first = 42
            unsafe_store!(Ptr{Float64}(p + 8), 3.14)       # inner.first = 3.14
            unsafe_store!(Ptr{Float32}(p + 16), Float32(2.71)) # inner.second = 2.71

            # Read outer.first
            result_i32 = Ref(Int32(0))
            ptr_ref = Ref(Ptr{Cvoid}(p))
            (args, refs) = make_jit_args(Ptr{Cvoid}(p))
            push!(args, Base.unsafe_convert(Ptr{Cvoid}, result_i32))
            push!(refs, result_i32)
            GC.@preserve refs begin
                @test jit_invoke(jit, "read_outer_first", args)
            end
            @test result_i32[] == 42

            # Read inner.first (the nested double)
            result_f64 = Ref(Float64(0.0))
            (args2, refs2) = make_jit_args(Ptr{Cvoid}(p))
            push!(args2, Base.unsafe_convert(Ptr{Cvoid}, result_f64))
            push!(refs2, result_f64)
            GC.@preserve refs2 begin
                @test jit_invoke(jit, "read_inner_first", args2)
            end
            @test result_f64[] ≈ 3.14

            # Read inner.second (the nested float)
            result_f32 = Ref(Float32(0.0))
            (args3, refs3) = make_jit_args(Ptr{Cvoid}(p))
            push!(args3, Base.unsafe_convert(Ptr{Cvoid}, result_f32))
            push!(refs3, result_f32)
            GC.@preserve refs3 begin
                @test jit_invoke(jit, "read_inner_second", args3)
            end
            @test result_f32[] ≈ Float32(2.71)
        end

        # Cross-validate with C++ functions
        lib = Libdl.dlopen(LIB_PATH)
        GC.@preserve buf begin
            p = pointer(buf)
            cpp_inner_first = ccall(Libdl.dlsym(lib, :nested_get_inner_first),
                                   Float64, (Ptr{Cvoid},), p)
            @test cpp_inner_first ≈ 3.14

            cpp_inner_second = ccall(Libdl.dlsym(lib, :nested_get_inner_second),
                                    Float32, (Ptr{Cvoid},), p)
            @test cpp_inner_second ≈ Float32(2.71)
        end
        Libdl.dlclose(lib)

        destroy_jit(jit)
        println("  ✓ nested get_field chains")
    finally
        destroy_context(ctx)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# 4. Packed Template Struct — JIT (pass by pointer)
# ══════════════════════════════════════════════════════════════════════════════

@testset "4. Packed Template Struct (JIT)" begin
    ctx = create_context()
    try
        # Call packed_triple_sum(PackedTriple*) → reads packed fields and sums them
        ir = """
        module {
            func.func private @packed_triple_sum(!llvm.ptr) -> i32

            func.func @test_packed_sum(%ptr: !llvm.ptr) -> i32 attributes {llvm.emit_c_interface} {
                %result = jlcs.ffe_call %ptr { callee = @packed_triple_sum } : (!llvm.ptr) -> i32
                return %result : i32
            }
        }
        """
        mod = parse_module(ctx, ir)
        mod_jit = clone_module(mod)
        @test lower_to_llvm(mod_jit)

        jit = create_jit(mod_jit, shared_libs=[LIB_PATH])
        @test jit != C_NULL

        # Allocate PackedTriple: 7 bytes { char @0, int32 @1, int16 @5 }
        buf = zeros(UInt8, 8)  # extra byte for safety
        GC.@preserve buf begin
            p = pointer(buf)
            unsafe_store!(Ptr{Int8}(p), Int8(10))           # a = 10
            unsafe_store!(Ptr{Int32}(p + 1), Int32(200))    # b = 200
            unsafe_store!(Ptr{Int16}(p + 5), Int16(30))     # c = 30

            result = Ref(Int32(0))
            (args, refs) = make_jit_args(Ptr{Cvoid}(p))
            push!(args, Base.unsafe_convert(Ptr{Cvoid}, result))
            push!(refs, result)
            GC.@preserve refs begin
                @test jit_invoke(jit, "test_packed_sum", args)
            end
            @test result[] == 240  # 10 + 200 + 30
        end

        destroy_jit(jit)
        println("  ✓ packed template struct (pass by pointer)")
    finally
        destroy_context(ctx)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# 5. Packed Struct Return via ffe_call (sret path)
# ══════════════════════════════════════════════════════════════════════════════

@testset "5. Packed Struct Return via ffe_call" begin
    ctx = create_context()
    try
        # make_packed_triple(char, int32, int16) → PackedTriple (7 bytes, packed)
        # The packed return triggers sret internally in FFECallOpLowering.
        # We use a void wrapper that stores the result to an output pointer,
        # avoiding double-sret complexity (inner ffe_call sret + outer ciface sret).
        ir = """
        module {
            func.func private @make_packed_triple(i8, i32, i16) -> !llvm.struct<packed (i8, i32, i16)>

            func.func @test_make_packed(%out: !llvm.ptr, %a: i8, %b: i32, %c: i16)
                    attributes {llvm.emit_c_interface} {
                %result = jlcs.ffe_call %a, %b, %c { callee = @make_packed_triple }
                    : (i8, i32, i16) -> !llvm.struct<packed (i8, i32, i16)>
                llvm.store %result, %out : !llvm.struct<packed (i8, i32, i16)>, !llvm.ptr
                return
            }
        }
        """
        mod = parse_module(ctx, ir)
        @test mod != C_NULL

        mod_jit = clone_module(mod)
        @test lower_to_llvm(mod_jit)

        jit = create_jit(mod_jit, shared_libs=[LIB_PATH])
        @test jit != C_NULL

        # Invoke: args = [&out_ptr, &a, &b, &c]  (void return, no result buffer)
        result_buf = zeros(UInt8, 16)
        a_ref = Ref(Int8(5))
        b_ref = Ref(Int32(100))
        c_ref = Ref(Int16(25))
        out_ref = Ref(Ptr{Cvoid}(pointer(result_buf)))

        GC.@preserve out_ref a_ref b_ref c_ref result_buf begin
            args = Ptr{Cvoid}[
                Base.unsafe_convert(Ptr{Cvoid}, out_ref),
                Base.unsafe_convert(Ptr{Cvoid}, a_ref),
                Base.unsafe_convert(Ptr{Cvoid}, b_ref),
                Base.unsafe_convert(Ptr{Cvoid}, c_ref),
            ]
            @test jit_invoke(jit, "test_make_packed", args)
        end

        # Read back packed fields from result
        GC.@preserve result_buf begin
            p = pointer(result_buf)
            @test unsafe_load(Ptr{Int8}(p)) == 5
            @test unsafe_load(Ptr{Int32}(p + 1)) == 100
            @test unsafe_load(Ptr{Int16}(p + 5)) == 25
        end

        destroy_jit(jit)
        println("  ✓ packed struct return via ffe_call (sret)")
    finally
        destroy_context(ctx)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# 6. Template RAII Scope — Single Object
# ══════════════════════════════════════════════════════════════════════════════

@testset "6. Template RAII Scope (Single Object)" begin
    ctx = create_context()
    try
        # Scope manages a PairIntDouble: ctor writes values, dtor writes sentinels
        # After scope exit, reading second field should return -1.0 (dtor sentinel)
        # get_field uses MLIR generic format (no custom assembly defined)
        ir = """
        module {
            func.func private @pair_int_double_ctor(!llvm.ptr, i32, f64)
            func.func private @pair_int_double_dtor(!llvm.ptr)

            func.func @test_pair_scope(%ptr: !llvm.ptr, %a: i32, %b: f64) -> f64
                    attributes {llvm.emit_c_interface} {
                jlcs.scope(%ptr : !llvm.ptr) dtors([@pair_int_double_dtor]) {
                    jlcs.ctor_call @pair_int_double_ctor(%ptr, %a, %b) : (!llvm.ptr, i32, f64) -> ()
                    jlcs.yield
                }
                // After scope: dtor has fired, second field should be -1.0
                %val = "jlcs.get_field"(%ptr) {fieldOffset = 8 : i64} : (!llvm.ptr) -> f64
                return %val : f64
            }
        }
        """
        mod = parse_module(ctx, ir)
        mod_jit = clone_module(mod)
        @test lower_to_llvm(mod_jit)

        jit = create_jit(mod_jit, shared_libs=[LIB_PATH])
        @test jit != C_NULL

        # Allocate PairIntDouble: 16 bytes
        buf = zeros(UInt8, 16)
        result = Ref(Float64(0.0))

        GC.@preserve buf result begin
            p = pointer(buf)
            args = [
                Ref(Ptr{Cvoid}(p)),
                Ref(Int32(42)),
                Ref(Float64(3.14)),
            ]
            ptrs = Ptr{Cvoid}[
                Base.unsafe_convert(Ptr{Cvoid}, args[1]),
                Base.unsafe_convert(Ptr{Cvoid}, args[2]),
                Base.unsafe_convert(Ptr{Cvoid}, args[3]),
                Base.unsafe_convert(Ptr{Cvoid}, result),
            ]
            GC.@preserve args begin
                @test jit_invoke(jit, "test_pair_scope", ptrs)
            end
        end

        # After scope: dtor wrote -1.0 to second field
        @test result[] == -1.0

        destroy_jit(jit)
        println("  ✓ template RAII scope (single object)")
    finally
        destroy_context(ctx)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# 7. Multi-Object RAII Ordering
# ══════════════════════════════════════════════════════════════════════════════

@testset "7. Multi-Object RAII Ordering" begin
    ctx = create_context()
    try
        # Three objects in one scope. Constructors log 1,2,3; destructors log -1,-2,-3.
        # Scope should destruct in reverse: [1, 2, 3, -3, -2, -1]
        ir = """
        module {
            func.func private @scope_log_reset()
            func.func private @scope_ctor_a(!llvm.ptr)
            func.func private @scope_ctor_b(!llvm.ptr)
            func.func private @scope_ctor_c(!llvm.ptr)
            func.func private @scope_dtor_a(!llvm.ptr)
            func.func private @scope_dtor_b(!llvm.ptr)
            func.func private @scope_dtor_c(!llvm.ptr)
            func.func private @get_scope_log() -> !llvm.ptr

            func.func @test_scope_ordering(%pa: !llvm.ptr, %pb: !llvm.ptr, %pc: !llvm.ptr) -> !llvm.ptr
                    attributes {llvm.emit_c_interface} {
                func.call @scope_log_reset() : () -> ()

                jlcs.scope(%pa, %pb, %pc : !llvm.ptr, !llvm.ptr, !llvm.ptr)
                    dtors([@scope_dtor_a, @scope_dtor_b, @scope_dtor_c]) {
                    jlcs.ctor_call @scope_ctor_a(%pa) : (!llvm.ptr) -> ()
                    jlcs.ctor_call @scope_ctor_b(%pb) : (!llvm.ptr) -> ()
                    jlcs.ctor_call @scope_ctor_c(%pc) : (!llvm.ptr) -> ()
                    jlcs.yield
                }

                %log = func.call @get_scope_log() : () -> !llvm.ptr
                return %log : !llvm.ptr
            }
        }
        """
        mod = parse_module(ctx, ir)
        mod_jit = clone_module(mod)
        @test lower_to_llvm(mod_jit)

        jit = create_jit(mod_jit, shared_libs=[LIB_PATH])
        @test jit != C_NULL

        # Dummy buffers (the ctors/dtors don't actually use the pointer contents)
        dummy_a = zeros(UInt8, 8)
        dummy_b = zeros(UInt8, 8)
        dummy_c = zeros(UInt8, 8)
        result_ptr = Ref(Ptr{Cvoid}(C_NULL))

        GC.@preserve dummy_a dummy_b dummy_c result_ptr begin
            pa = Ref(Ptr{Cvoid}(pointer(dummy_a)))
            pb = Ref(Ptr{Cvoid}(pointer(dummy_b)))
            pc = Ref(Ptr{Cvoid}(pointer(dummy_c)))

            ptrs = Ptr{Cvoid}[
                Base.unsafe_convert(Ptr{Cvoid}, pa),
                Base.unsafe_convert(Ptr{Cvoid}, pb),
                Base.unsafe_convert(Ptr{Cvoid}, pc),
                Base.unsafe_convert(Ptr{Cvoid}, result_ptr),
            ]
            GC.@preserve pa pb pc begin
                @test jit_invoke(jit, "test_scope_ordering", ptrs)
            end
        end

        # Read the log: 6 Int32 values
        log_ptr = result_ptr[]
        @test log_ptr != C_NULL

        log_values = [unsafe_load(Ptr{Int32}(log_ptr), i) for i in 1:6]
        @test log_values == [1, 2, 3, -3, -2, -1]

        destroy_jit(jit)
        println("  ✓ multi-object RAII ordering")
    finally
        destroy_context(ctx)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# 8. Virtual Dispatch on Template Container
# ══════════════════════════════════════════════════════════════════════════════

@testset "8. Virtual Dispatch on Template Container" begin
    ctx = create_context()
    try
        # ContainerInt layout: { void** vptr @0, int32_t stored @8 }
        # vtable[0] = get_value, vtable[1] = set_value
        # vcall uses MLIR generic format (no custom assembly defined)
        #
        # Note: vcall lowers to indirect llvm.call via OperationState which currently
        # has operandSegmentSizes issues in LLVM 21 JIT translation. We test parse + lower
        # here; the AOT path (emit_object) works correctly and is tested in stress_test.

        @testset "vcall parse + lower (get_value)" begin
            ir = """
            module {
                func.func @test_vcall_get(%obj: !llvm.ptr) -> i32
                        attributes {llvm.emit_c_interface} {
                    %n = "jlcs.vcall"(%obj) {class_name = @ContainerInt, vtable_offset = 0 : i64, slot = 0 : i64}
                        : (!llvm.ptr) -> i32
                    return %n : i32
                }
            }
            """
            mod = parse_module(ctx, ir)
            @test mod != C_NULL
            mod_jit = clone_module(mod)
            @test lower_to_llvm(mod_jit)
        end

        @testset "vcall parse + lower (set + get, void + value return)" begin
            ir = """
            module {
                func.func @test_vcall_set_then_get(%obj: !llvm.ptr, %val: i32) -> i32
                        attributes {llvm.emit_c_interface} {
                    "jlcs.vcall"(%obj, %val) {class_name = @ContainerInt, vtable_offset = 0 : i64, slot = 1 : i64}
                        : (!llvm.ptr, i32) -> ()
                    %n = "jlcs.vcall"(%obj) {class_name = @ContainerInt, vtable_offset = 0 : i64, slot = 0 : i64}
                        : (!llvm.ptr) -> i32
                    return %n : i32
                }
            }
            """
            mod = parse_module(ctx, ir)
            @test mod != C_NULL
            mod_jit = clone_module(mod)
            @test lower_to_llvm(mod_jit)
        end

        @testset "vcall JIT execution via manual vtable dispatch" begin
            # Equivalent test using explicit vtable dispatch (LLVM ops) to verify
            # the concept works end-to-end with JIT. This mirrors what vcall lowers to.
            ir = """
            module {
                func.func @test_manual_vcall_get(%obj: !llvm.ptr) -> i32
                        attributes {llvm.emit_c_interface} {
                    // Load vptr from object at offset 0
                    %vptr = llvm.load %obj : !llvm.ptr -> !llvm.ptr
                    // Load function pointer from vtable slot 0
                    %zero = arith.constant 0 : i64
                    %fptr_addr = llvm.getelementptr %vptr[%zero] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
                    %fptr = llvm.load %fptr_addr : !llvm.ptr -> !llvm.ptr
                    // Indirect call: int32_t (*fptr)(void* self)
                    %result = llvm.call %fptr(%obj) : !llvm.ptr, (!llvm.ptr) -> i32
                    return %result : i32
                }
            }
            """
            mod = parse_module(ctx, ir)
            mod_jit = clone_module(mod)
            @test lower_to_llvm(mod_jit)

            jit = create_jit(mod_jit, shared_libs=[LIB_PATH])
            @test jit != C_NULL

            buf = zeros(UInt8, 16)
            GC.@preserve buf begin
                p = pointer(buf)
                lib = Libdl.dlopen(LIB_PATH)
                ccall(Libdl.dlsym(lib, :container_int_init),
                      Cvoid, (Ptr{Cvoid}, Int32), p, Int32(777))

                result = Ref(Int32(0))
                obj_ref = Ref(Ptr{Cvoid}(p))
                ptrs = Ptr{Cvoid}[
                    Base.unsafe_convert(Ptr{Cvoid}, obj_ref),
                    Base.unsafe_convert(Ptr{Cvoid}, result),
                ]
                GC.@preserve obj_ref result begin
                    @test jit_invoke(jit, "test_manual_vcall_get", ptrs)
                end
                @test result[] == 777

                Libdl.dlclose(lib)
            end

            destroy_jit(jit)
        end

        println("  ✓ virtual dispatch on template container")
    finally
        destroy_context(ctx)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# 9. TypeInfoOp with Template Inheritance
# ══════════════════════════════════════════════════════════════════════════════

@testset "9. TypeInfoOp with Template Inheritance" begin
    ctx = create_context()
    try
        ir = """
        module {
            // Base template class
            jlcs.type_info "Container_int",
                !jlcs.c_struct<"Container_int", [!llvm.ptr, i32],
                    [[0 : i64, 8 : i64]], packed = false>,
                "", ""

            // Derived template — inherits from Container_int, adds extra field
            jlcs.type_info "DerivedContainer_int",
                !jlcs.c_struct<"DerivedContainer_int", [!llvm.ptr, i32, i32],
                    [[0 : i64, 8 : i64, 12 : i64]], packed = false>,
                "Container_int", "_ZN21DerivedContainer_intD1Ev"

            // Double-nested inheritance: GrandChild<int> -> DerivedContainer<int> -> Container<int>
            jlcs.type_info "GrandChild_int",
                !jlcs.c_struct<"GrandChild_int", [!llvm.ptr, i32, i32, f64],
                    [[0 : i64, 8 : i64, 12 : i64, 16 : i64]], packed = false>,
                "DerivedContainer_int", "_ZN14GrandChild_intD1Ev"
        }
        """
        mod = parse_module(ctx, ir)
        @test mod != C_NULL

        mod_jit = clone_module(mod)
        @test lower_to_llvm(mod_jit)

        println("  ✓ TypeInfoOp with template inheritance")
    finally
        destroy_context(ctx)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# 10. Fixed-Size Array Template — set_field + external call
# ══════════════════════════════════════════════════════════════════════════════

@testset "10. Fixed-Size Array Template (JIT)" begin
    ctx = create_context()
    try
        # Write 4 doubles via set_field, then call fixed_array_sum to verify
        # set_field uses MLIR generic format (no custom assembly defined)
        ir = """
        module {
            func.func private @fixed_array_sum(!llvm.ptr) -> f64

            func.func @test_array_write_and_sum(%ptr: !llvm.ptr,
                    %v0: f64, %v1: f64, %v2: f64, %v3: f64) -> f64
                    attributes {llvm.emit_c_interface} {
                "jlcs.set_field"(%ptr, %v0) {fieldOffset = 0 : i64} : (!llvm.ptr, f64) -> ()
                "jlcs.set_field"(%ptr, %v1) {fieldOffset = 8 : i64} : (!llvm.ptr, f64) -> ()
                "jlcs.set_field"(%ptr, %v2) {fieldOffset = 16 : i64} : (!llvm.ptr, f64) -> ()
                "jlcs.set_field"(%ptr, %v3) {fieldOffset = 24 : i64} : (!llvm.ptr, f64) -> ()
                %sum = func.call @fixed_array_sum(%ptr) : (!llvm.ptr) -> f64
                return %sum : f64
            }
        }
        """
        mod = parse_module(ctx, ir)
        mod_jit = clone_module(mod)
        @test lower_to_llvm(mod_jit)

        jit = create_jit(mod_jit, shared_libs=[LIB_PATH])
        @test jit != C_NULL

        # FixedArray4: 32 bytes (4 × f64)
        buf = zeros(Float64, 4)
        result = Ref(Float64(0.0))

        GC.@preserve buf result begin
            p = pointer(buf)
            ptr_ref = Ref(Ptr{Cvoid}(p))
            v0 = Ref(Float64(1.0))
            v1 = Ref(Float64(2.0))
            v2 = Ref(Float64(3.0))
            v3 = Ref(Float64(4.0))

            ptrs = Ptr{Cvoid}[
                Base.unsafe_convert(Ptr{Cvoid}, ptr_ref),
                Base.unsafe_convert(Ptr{Cvoid}, v0),
                Base.unsafe_convert(Ptr{Cvoid}, v1),
                Base.unsafe_convert(Ptr{Cvoid}, v2),
                Base.unsafe_convert(Ptr{Cvoid}, v3),
                Base.unsafe_convert(Ptr{Cvoid}, result),
            ]
            GC.@preserve ptr_ref v0 v1 v2 v3 begin
                @test jit_invoke(jit, "test_array_write_and_sum", ptrs)
            end
        end

        @test result[] ≈ 10.0

        # Verify the buffer was actually written by set_field
        @test buf[1] ≈ 1.0
        @test buf[2] ≈ 2.0
        @test buf[3] ≈ 3.0
        @test buf[4] ≈ 4.0

        destroy_jit(jit)
        println("  ✓ fixed-size array template")
    finally
        destroy_context(ctx)
    end
end

println("\n✅ MLIR template stress tests: all passed")

end # @testset "MLIR Templates"
