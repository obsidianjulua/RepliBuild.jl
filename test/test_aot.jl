using Test
using RepliBuild
using RepliBuild.MLIRNative
using RepliBuild.DWARFParser
using RepliBuild.JLCSIRGenerator
using JSON
using Libdl

@testset "AOT Compilation" begin

    @testset "Basic emit_object" begin
        ctx = MLIRNative.create_context()
        try
            mod_str = "module { func.func @my_thunk(%arg0: i32) -> i32 { return %arg0 : i32 } }"
            mod = MLIRNative.parse_module(ctx, mod_str)
            @test mod != C_NULL

            @test MLIRNative.lower_to_llvm(mod)

            obj_path = tempname() * ".o"
            @test MLIRNative.emit_object(mod, obj_path)
            @test isfile(obj_path)
            @test filesize(obj_path) > 0

            rm(obj_path, force=true)
        finally
            MLIRNative.destroy_context(ctx)
        end
    end

    @testset "VTable thunks AOT pipeline" begin
        vtable_dir = joinpath(@__DIR__, "vtable_test")
        lib_path = joinpath(vtable_dir, "julia", "libvtable_test.so")
        metadata_path = joinpath(vtable_dir, "julia", "compilation_metadata.json")

        # Build vtable_test if not already built
        toml_path = joinpath(vtable_dir, "replibuild.toml")
        if !isfile(lib_path) || !isfile(metadata_path)
            @test isfile(toml_path)
            RepliBuild.build(toml_path)
        end

        @test isfile(lib_path)
        @test isfile(metadata_path)

        ctx = MLIRNative.create_context()
        try
            # Parse vtables and generate JLCS IR
            vtable_info = DWARFParser.parse_vtables(lib_path)
            metadata = JSON.parsefile(metadata_path)
            ir_source = JLCSIRGenerator.generate_jlcs_ir(vtable_info, metadata)

            @test !isempty(ir_source)
            @test contains(ir_source, "module")

            # Parse and lower
            mod = MLIRNative.parse_module(ctx, ir_source)
            @test mod != C_NULL

            @test MLIRNative.lower_to_llvm(mod)

            # Emit object file
            thunks_obj = joinpath(vtable_dir, "julia", "thunks.o")
            @test MLIRNative.emit_object(mod, thunks_obj)
            @test isfile(thunks_obj)
            @test filesize(thunks_obj) > 0

            # Link to shared library
            thunks_so = joinpath(vtable_dir, "julia", "libvtable_test_thunks.so")
            run(`gcc -shared -o $thunks_so $thunks_obj`)
            @test isfile(thunks_so)

            # dlopen both libraries
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
