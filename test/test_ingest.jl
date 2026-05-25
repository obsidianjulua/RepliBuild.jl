#!/usr/bin/env julia
# RepliBuild.jl — Ingest mode tests (no toolchain required)
#
# Covers config parsing, scaffolding, validation, and the cheap DWARF-presence
# error path. End-to-end ingest (build a real .so, then ingest it) lives in
# devtests.jl since it needs Clang.

using Test
using RepliBuild

@testset "Ingest mode" begin

    @testset "API surface" begin
        @test isdefined(RepliBuild, :ingest)
        @test isdefined(RepliBuild, :IngestConfig)
        @test isdefined(RepliBuild.Compiler, :ingest_library)
    end

    @testset "Config parsing — [ingest] section" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "replibuild.toml")
            write(toml_path, """
            [project]
            name = "sample"

            [ingest]
            library = "build/libsample.so"
            headers = ["include", "vendor/include"]
            extra_link_libs = ["m", "pthread"]

            [wrap]
            language = "c"
            module_name = "Sample"
            """)

            cfg = RepliBuild.ConfigurationManager.load_config(toml_path)
            @test cfg.ingest !== nothing
            # Relative library path resolved against the toml's directory
            @test cfg.ingest.library == abspath(joinpath(dir, "build/libsample.so"))
            @test cfg.ingest.headers == ["include", "vendor/include"]
            @test cfg.ingest.extra_link_libs == ["m", "pthread"]
            # binary.output_name auto-derived from ingest.library so wrap() finds it
            @test cfg.binary.output_name == "libsample.so"
        end
    end

    @testset "Config parsing — [ingest] absent means source mode" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "replibuild.toml")
            write(toml_path, """
            [project]
            name = "src_mode"

            [compile]
            source_files = ["src/foo.c"]
            """)

            cfg = RepliBuild.ConfigurationManager.load_config(toml_path)
            @test cfg.ingest === nothing
        end
    end

    @testset "ingest() helper scaffolds a valid TOML" begin
        mktempdir() do dir
            fake_lib = joinpath(dir, "libfake.so.6")
            touch(fake_lib)

            project_dir = joinpath(dir, "fake_project")
            toml_path = RepliBuild.ingest(
                fake_lib,
                headers=["/usr/include"],
                extra_link_libs=["dl"],
                project_dir=project_dir,
                language=:c,
                register=false,
            )

            @test isfile(toml_path)
            @test dirname(toml_path) == project_dir

            cfg = RepliBuild.ConfigurationManager.load_config(toml_path)
            @test cfg.ingest !== nothing
            @test cfg.ingest.library == abspath(fake_lib)
            @test cfg.ingest.headers == ["/usr/include"]
            @test cfg.ingest.extra_link_libs == ["dl"]
            # Project name derived from "libfake.so.6" → "fake"
            @test cfg.project.name == "fake"
            @test cfg.wrap.module_name == "Fake"
            @test cfg.wrap.language == :c
        end
    end

    @testset "ingest() rejects missing libraries" begin
        @test_throws Exception RepliBuild.ingest("/nonexistent/libfoo.so")
    end

    @testset "ingest_library errors on stripped/empty .so" begin
        mktempdir() do dir
            # Empty placeholder file — no DWARF, no symbols
            fake_lib = joinpath(dir, "libempty.so")
            touch(fake_lib)

            project_dir = joinpath(dir, "project")
            toml_path = RepliBuild.ingest(fake_lib, project_dir=project_dir, name="empty_lib", register=false)
            cfg = RepliBuild.ConfigurationManager.load_config(toml_path)

            # Cheap DWARF check should refuse to proceed
            @test_throws ErrorException RepliBuild.Compiler.ingest_library(cfg)
        end
    end

    @testset "Round-trip: save_config preserves [ingest]" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "replibuild.toml")
            write(toml_path, """
            [project]
            name = "rt"

            [ingest]
            library = "$(joinpath(dir, "libfoo.so"))"
            headers = ["inc"]
            """)
            touch(joinpath(dir, "libfoo.so"))

            cfg = RepliBuild.ConfigurationManager.load_config(toml_path)
            RepliBuild.ConfigurationManager.save_config(cfg)

            cfg2 = RepliBuild.ConfigurationManager.load_config(toml_path)
            @test cfg2.ingest !== nothing
            @test cfg2.ingest.headers == ["inc"]
        end
    end
end
