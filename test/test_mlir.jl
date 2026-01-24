using Test
using RepliBuild.JLCSIRGenerator
using RepliBuild.MLIRNative
using RepliBuild.DWARFParser

@testset "MLIR Integration Tests" begin
    
    @testset "JLCSIRGenerator" begin
        # Create mock data
        vm = VirtualMethod("foo", "_ZN4Base3fooEv", 0, "int", [])
        ci = ClassInfo("Base", 0, [], [vm], 8)
        
        # Test type info generation
        ir_type = JLCSIRGenerator.generate_type_info_ir("Base", ci, UInt64(0x1000))
        @test contains(ir_type, "jlcs.type_info @\"Base\"")
        @test contains(ir_type, "size = 8 : i64")
        
        # Test virtual method IR generation
        ir_method = JLCSIRGenerator.generate_virtual_method_ir(vm, UInt64(0x2000))
        @test contains(ir_method, "func.func @_ZN4Base3fooEv(%arg0: !llvm.ptr) -> i32")
        @test contains(ir_method, "llvm.call")
        
        # Test vcall example
        ir_vcall = generate_vcall_example("Base", "foo", 0, 0, "int")
        @test contains(ir_vcall, "jlcs.vcall @Base::foo")
    end

    @testset "MLIRNative" begin
        # Check library presence
        @test isfile(MLIRNative.libJLCS)
        
        # Test Context creation/destruction
        ctx = create_context()
        @test ctx != C_NULL
        
        # Test Module creation
        mod = create_module(ctx)
        @test mod != C_NULL
        
        # Test Parsing
        valid_ir = """
        module {
            func.func @test(%arg0: i32) -> i32 {
                return %arg0 : i32
            }
        }
        """
        parsed_mod = parse_module(ctx, valid_ir)
        @test parsed_mod != C_NULL
        
        # Cleanup
        destroy_context(ctx)
    end
    
    @testset "Integration: Generate & Parse" begin
        # Create context
        ctx = create_context()
        
        # Generate complete module IR
        # We need a VtableInfo object
        vm = VirtualMethod("bar", "_ZN4Base3barEv", 0, "void", ["int"])
        ci = ClassInfo("Base", 0, [], [vm], 8)
        
        classes = Dict("Base" => ci)
        vtable_addrs = Dict("Base" => UInt64(0x1000))
        method_addrs = Dict("_ZN4Base3barEv" => UInt64(0x2000))
        
        vtinfo = VtableInfo(classes, vtable_addrs, method_addrs)
        
        generated_ir = generate_jlcs_ir(vtinfo)
        
        println("\nGenerated IR for verification:")
        println(generated_ir)
        
        # Parse generated IR
        # Note: generated_ir contains "module { ... }", parse_module expects standard MLIR text
        # If generate_jlcs_ir produces valid MLIR, it should parse.
        
        parsed_mod = parse_module(ctx, generated_ir)
        @test parsed_mod != C_NULL
        
        destroy_context(ctx)
    end
end
