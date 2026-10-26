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
            # PARSING ONLY. This assertion held for the entire period in which
            # `extra_link_libs` was parsed, documented as loading libraries, and
            # never read by anything — a round-trip through the parser is true
            # whether or not the feature exists. The behavioural assertion (the
            # emitted snippet is EXECUTED) lives in test_config_surface.jl; do not
            # treat this line as coverage of the feature.
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

    # discover's rule: only `build=true, wrap=true` registers, and `register`
    # builds, vendors, and adds a package to a Julia environment. Until
    # 2026-10-26 the default (`build=false`) registered too, which since 4.3.0
    # built, wrapped and changed `@v#.#`, with any failure logged at `@debug`.
    @testset "ingest() registers only with build and wrap" begin
        mktempdir() do dir
            fake_lib = joinpath(dir, "libfake.so.6")
            touch(fake_lib)
            home = joinpath(dir, "rb_home")
            withenv("REPLIBUILD_HOME" => home, "REPLIBUILD_JULIA_ENV" => "none") do
                toml_path = RepliBuild.ingest(fake_lib, project_dir=joinpath(dir, "bare"))
                @test isfile(toml_path)
                @test isempty(RepliBuild.PackageRegistry._load_index().entries)
                @test !isdir(joinpath(dir, "bare", "julia"))

                # A registration failure surfaces. A file where the registry
                # home should be makes `register` fail before it builds; a
                # swallowed failure would reach the build instead, which
                # refuses the empty library with an ErrorException.
                write(home * "_file", "")
                withenv("REPLIBUILD_HOME" => home * "_file") do
                    @test_throws Base.IOError RepliBuild.ingest(
                        fake_lib, project_dir=joinpath(dir, "full"), build=true, wrap=true)
                end
            end
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
            # `escape_string` because this is a TOML *basic* string and `dir` is a
            # real filesystem path: on Windows it is `C:\\Users\\...`, and the raw
            # backslash reaches the parser as an escape (`\U` → "invalid unicode
            # scalar"). Interpolating a path into TOML unescaped is the same bug
            # the ingest() writer had; a test that reproduces it is not a fixture.
            write(toml_path, """
            [project]
            name = "rt"

            [ingest]
            library = "$(escape_string(joinpath(dir, "libfoo.so")))"
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
