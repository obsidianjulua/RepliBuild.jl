
using Test
using RepliBuild.MLIRNative

@testset "MLIR Type Safety Tests" begin

    @testset "JIT Invocation Safety" begin
        # 1. Setup Context and JIT
        ctx = create_context()
        
        # Define a simple function that adds two 32-bit integers
        # func @add(%arg0: i32, %arg1: i32) -> i32
        ir = """
        module {
            func.func @add(%arg0: i32, %arg1: i32) -> i32 {
                %0 = arith.addi %arg0, %arg1 : i32
                return %0 : i32
            }
        }
        """
        
        mod = parse_module(ctx, ir)
        jit = create_jit(mod)
        
        # 2. Correct Usage (Baseline)
        # We need to pass pointers to values.
        # For i32, we pass Ref{Int32}
        
        a = Int32(10)
        b = Int32(20)
        res = Int32(0) # Output buffer
        
        # Arguments must be pointers to the values
        # The return value is also passed as a pointer (last argument for void return, 
        # but for packed invoke, return values are usually handled via pointer too if they are structs,
        # otherwise for primitives it might depend on ABI. 
        # WAIT: mlirExecutionEngineInvokePacked expects arguments AND return value pointers in the list.
        # For: func(i32, i32) -> i32
        # Args: [Ptr{i32}, Ptr{i32}, Ptr{i32}]  <-- Last one is return value
        
        args = [
            pointer_from_objref(Ref(a)),
            pointer_from_objref(Ref(b)),
            pointer_from_objref(Ref(res)) 
        ]
        
        # We need to cast these to Ptr{Cvoid} for the current invoke signature
        void_args = map(x -> reinterpret(Ptr{Cvoid}, x), args)
        
        success = invoke(jit, "add", void_args)
        @test success == true
        
        # Check result
        # Note: res is immutable (Int32), so Ref(res) created a copy. 
        # We need to use a mutable container or Ref properly to get the value back.
        
        a_ref = Ref(Int32(10))
        b_ref = Ref(Int32(20))
        res_ref = Ref(Int32(0))
        
        args_correct = [
            unsafe_convert(Ptr{Cvoid}, a_ref),
            unsafe_convert(Ptr{Cvoid}, b_ref),
            unsafe_convert(Ptr{Cvoid}, res_ref)
        ]
        
        invoke(jit, "add", args_correct)
        @test res_ref[] == 30
        
        # 3. UNSAFE Usage (The issue to fix)
        # Passing wrong types (e.g. Int64 instead of Int32)
        # This effectively interprets the first 32 bits of the Int64 as the Int32
        
        bad_a = Ref(Int64(10)) 
        bad_b = Ref(Int64(20))
        bad_res = Ref(Int64(0))
        
        args_bad_type = [
            unsafe_convert(Ptr{Cvoid}, bad_a),
            unsafe_convert(Ptr{Cvoid}, bad_b),
            unsafe_convert(Ptr{Cvoid}, bad_res)
        ]
        
        # This will "work" but produce garbage or crash if memory alignment was different
        # Or if we passed fewer arguments than expected
        
        invoke(jit, "add", args_bad_type)
        
        # Verify it didn't crash (at least)
        @test true 
        
        # 4. Incorrect Argument Count
        # Missing arguments usually causes reads from stack garbage -> Segfault risk
        args_missing = [
            unsafe_convert(Ptr{Cvoid}, a_ref)
        ]
        
        # We expect this might crash the process, so running it in a test suite is risky.
        # Ideally, our safe invoke should catch this before calling C++.
        
        # Proposed API for safety:
        # invoke_safe(jit, "add", (Int32, Int32), Int32, 10, 20)
        
        destroy_jit(jit)
        destroy_context(ctx)
    end
end
