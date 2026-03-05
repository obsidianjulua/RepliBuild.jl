using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using RepliBuild
using Test
using JSON

@testset "STL Container Support" begin
    toml_path = joinpath(@__DIR__, "replibuild.toml")
    @test isfile(toml_path)

    @testset "Build" begin
        library_path = RepliBuild.build(toml_path, clean=true)
        @test isfile(library_path)

        # Verify metadata was generated
        metadata_path = joinpath(dirname(library_path), "compilation_metadata.json")
        @test isfile(metadata_path)

        # Check that STL methods were captured in metadata
        metadata = JSON.parsefile(metadata_path)
        stl_methods = get(metadata, "stl_methods", Dict())
        println("STL containers found: ", collect(keys(stl_methods)))
        for (container, methods) in stl_methods
            println("  $container: $(length(methods)) methods")
            for m in methods
                println("    $(m["method"]): $(m["mangled"])")
            end
        end
        @test !isempty(stl_methods)
    end

    @testset "Wrap" begin
        wrapper_path = RepliBuild.wrap(toml_path)
        @test isfile(wrapper_path)

        # Print the generated wrapper for inspection
        println("\n=== Generated Wrapper (first 200 lines) ===")
        lines = readlines(wrapper_path)
        for (i, line) in enumerate(lines)
            i > 200 && break
            println(line)
        end
        println("=== End Wrapper ($(length(lines)) total lines) ===\n")
    end

    @testset "C-linkage functions (Tier 1)" begin
        wrapper_path = joinpath(@__DIR__, "julia", "StlTest.jl")
        @test isfile(wrapper_path)
        include(wrapper_path)

        @test isdefined(Main, :StlTest)
        mod = Main.StlTest

        # These C-linkage functions should work via ccall
        if isdefined(mod, :add_numbers)
            @test mod.add_numbers(Cint(3), Cint(4)) == 7
        end
        if isdefined(mod, :multiply)
            @test mod.multiply(Cdouble(2.5), Cdouble(4.0)) == 10.0
        end
    end
end

