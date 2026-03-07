#!/usr/bin/env julia
# RepliBuild.jl — Registry Test Suite
#
# Lightweight tests for `Pkg.test()` / AutoMerge CI.
# No C++ toolchain required — validates package loading, types, and API surface.
# For full integration tests: julia --project=. test/devtests.jl

using Test
using RepliBuild

@testset "RepliBuild.jl" begin

    @testset "Package loads" begin
        @test RepliBuild.VERSION == v"2.3.0"
        @test isdefined(RepliBuild, :discover)
        @test isdefined(RepliBuild, :build)
        @test isdefined(RepliBuild, :wrap)
        @test isdefined(RepliBuild, :clean)
        @test isdefined(RepliBuild, :info)
        @test isdefined(RepliBuild, :check_environment)
    end

    @testset "ConfigurationManager" begin
        cfg = RepliBuild.ConfigurationManager

        mktempdir() do dir
            toml_path = joinpath(dir, "replibuild.toml")
            write(toml_path, """
            [project]
            name = "test_pkg"
            uuid = "00000000-0000-0000-0000-000000000000"
            root = "."

            [compile]
            source_files = ["src/foo.cpp"]
            include_dirs = ["include"]
            flags = ["-std=c++17"]

            [link]
            enable_lto = false
            optimization_level = "2"

            [binary]
            type = "shared"

            [wrap]
            style = "module"
            """)

            config = cfg.load_config(toml_path)
            @test config.project.name == "test_pkg"
            @test "-std=c++17" in config.compile.flags
            @test config.link.optimization_level == "2"
            @test config.binary.type == :shared
        end
    end

    @testset "DWARFParser types" begin
        dp = RepliBuild.DWARFParser

        vm = dp.VirtualMethod("foo", "_ZN1A3fooEv", 0, "int", String[])
        @test vm.name == "foo"
        @test vm.slot == 0

        ci = dp.ClassInfo("A", 0, String[], [vm], dp.MemberInfo[], 8)
        @test ci.name == "A"
        @test length(ci.virtual_methods) == 1
    end

    @testset "JLCSIRGenerator" begin
        dp = RepliBuild.DWARFParser
        gen = RepliBuild.JLCSIRGenerator

        vm = dp.VirtualMethod("bar", "_ZN1B3barEv", 0, "void", ["int"])
        ci = dp.ClassInfo("B", 0, String[], [vm], dp.MemberInfo[], 8)

        ir = gen.generate_type_info_ir("B", ci, UInt64(0x1000))
        @test contains(ir, "jlcs.type_info")
        @test contains(ir, "\"B\"")

        ir_m = gen.generate_virtual_method_ir(vm, UInt64(0x2000))
        @test contains(ir_m, "thunk__ZN1B3barEv")
    end

    @testset "MLIR native library" begin
        mlir = RepliBuild.MLIRNative
        if isfile(mlir.libJLCS)
            ctx = mlir.create_context()
            @test ctx != C_NULL

            mod = mlir.parse_module(ctx, """
                module { func.func @id(%x: i32) -> i32 { return %x : i32 } }
            """)
            @test mod != C_NULL

            mlir.destroy_context(ctx)
        else
            @info "libJLCS not found — skipping MLIR native tests (Tier 1 still works)"
        end
    end

    @testset "Environment doctor" begin
        status = RepliBuild.check_environment(verbose=false)
        @test hasproperty(status, :ready) || hasfield(typeof(status), :ready)
    end
end
