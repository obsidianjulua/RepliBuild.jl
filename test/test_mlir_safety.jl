
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
        
        # Clone module for JIT compilation (lowering modifies it)
        mod_jit = clone_module(mod)
        
        # Lower to LLVM IR
        success_lower = lower_to_llvm(mod_jit)
        @test success_lower == true
        
        jit = create_jit(mod_jit)
        
        # 2. Correct Usage (Baseline)
        a = Int32(10)
        b = Int32(20)
        res = Int32(0) # Output buffer
        
        # Arguments must be pointers to the values
        args = [
            pointer_from_objref(Ref(a)),
            pointer_from_objref(Ref(b)),
            pointer_from_objref(Ref(res)) 
        ]
        
        void_args = map(x -> reinterpret(Ptr{Cvoid}, x), args)
        success = jit_invoke(jit, "add", void_args)
        @test success == true
        
        # 3. Safe Usage (New API)
        # We use 'mod' (the high-level MLIR) for type introspection
        # We use 'jit' (compiled from mod_jit) for execution
        res_ref = Ref(Int32(0))
        
        # Should succeed
        invoke_safe(jit, mod, "add", Int32(10), Int32(20), res_ref)
        @test res_ref[] == 30
        
        # 4. Safe Usage - Detection of Type Mismatch
        
        bad_res_ref = Ref(Int32(0))
        
        # Pass Int64 instead of Int32 -> Should throw Error
        @test_throws ErrorException invoke_safe(jit, mod, "add", Int64(10), Int32(20), bad_res_ref)
        
        # Pass Float32 instead of Int32 -> Should throw Error
        @test_throws ErrorException invoke_safe(jit, mod, "add", Float32(10.0), Int32(20), bad_res_ref)
        
        destroy_jit(jit)
        destroy_context(ctx)
    end
end
