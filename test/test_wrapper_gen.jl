using Test
using RepliBuild.JLCSIRGenerator
using RepliBuild.Wrapper

@testset "Wrapper Generation Logic" begin
    
    @testset "Safety Checks" begin
        # Mock struct definitions
        structs = Dict(
            "SafeStruct" => Dict("kind" => "struct", "byte_size" => "16", "members" => []),
            "UnsafeStruct" => Dict("kind" => "struct", "byte_size" => "24", "members" => []), # Too big
            "ClassType" => Dict("kind" => "class", "byte_size" => "8", "members" => []),      # Class
            "UnionType" => Dict("kind" => "union", "byte_size" => "8", "members" => [])       # Union
        )
        
        # Test Case 1: Safe function
        f1 = Dict("return_type" => Dict("c_type" => "void"), 
                  "parameters" => [Dict("c_type" => "int")])
        @test Wrapper.is_ccall_safe(f1, structs) == true
        
        # Test Case 2: Unsafe return (large struct by value)
        f2 = Dict("return_type" => Dict("c_type" => "UnsafeStruct"), 
                  "parameters" => [])
        @test Wrapper.is_ccall_safe(f2, structs) == false
        
        # Test Case 3: Unsafe param (Class)
        f3 = Dict("return_type" => Dict("c_type" => "void"), 
                  "parameters" => [Dict("c_type" => "ClassType")])
        @test Wrapper.is_ccall_safe(f3, structs) == false
        
        # Test Case 4: Unsafe param (Union)
        f4 = Dict("return_type" => Dict("c_type" => "void"), 
                  "parameters" => [Dict("c_type" => "UnionType")])
        @test Wrapper.is_ccall_safe(f4, structs) == false
    end

    @testset "Thunk IR Generation" begin
        using RepliBuild.JLCSIRGenerator.FunctionGen
        
        # Mock function metadata including a method
        functions = [
            Dict(
                "name" => "my_method",
                "mangled" => "_ZN5Class9my_methodEi",
                "return_type" => Dict("c_type" => "int"),
                "parameters" => [
                    Dict("c_type" => "Class*", "name" => "this"),
                    Dict("c_type" => "int", "name" => "arg1")
                ],
                "is_method" => true,
                "class" => "Class"
            )
        ]
        
        ir = generate_function_thunks(functions)
        println(ir)
        
        # Verify correctness
        # 1. Should have external declaration with __real_ prefix
        @test contains(ir, "llvm.func @__real__ZN5Class9my_methodEi")
        
        # 2. Should have thunk with mangled name
        @test contains(ir, "func.func @_ZN5Class9my_methodEi")
        
        # 3. Should unpack 2 arguments
        @test contains(ir, "%idx_1 = arith.constant 0") # param 1 (this)
        @test contains(ir, "%idx_2 = arith.constant 1") # param 2 (arg1)
    end
end
