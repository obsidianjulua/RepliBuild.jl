# test_portable_paths.jl — a replibuild.toml is not tied to one checkout.
#
# Two halves. load_config resolves relative `[compile] source_files` and
# `include_dirs` against the project root (they used to reach clang as written,
# so they meant "relative to the process cwd"). save_config, which discover
# goes through, writes paths under the project relative with `/` separators; a
# discovered Hub config had carried `/home/…` into every clone, where it could
# not build on any other machine, let alone on Windows.
#
# No toolchain.

using Test
using RepliBuild
import TOML

const _PP_CM = RepliBuild.ConfigurationManager

@testset "portable config paths" begin
    @testset "_portable_path" begin
        base = joinpath(mktempdir(), "proj")
        @test _PP_CM._portable_path(base, base) == "."
        @test _PP_CM._portable_path(base * Base.Filesystem.path_separator, base) == "."
        @test _PP_CM._portable_path(joinpath(base, "src", "a.c"), base) == "src/a.c"
        # A sibling sharing the prefix is outside, not under.
        sibling = base * "2"
        @test _PP_CM._portable_path(joinpath(sibling, "x.h"), base) == joinpath(sibling, "x.h")
        # Already relative: only separators are normalised.
        @test _PP_CM._portable_path("include\\gen", base) == "include/gen"
    end

    mktempdir() do dir
        mkpath(joinpath(dir, "src")); mkpath(joinpath(dir, "include"))
        write(joinpath(dir, "src", "a.c"), "int a(void){return 1;}\n")
        toml = joinpath(dir, "replibuild.toml")
        open(toml, "w") do io
            TOML.print(io, Dict(
                "project" => Dict("name" => "pathprobe", "root" => dir),
                "compile" => Dict("source_files" => [joinpath(dir, "src", "a.c")],
                                  "include_dirs" => [joinpath(dir, "include"), dir]),
                "wrap"    => Dict("language" => "c")))
        end

        @testset "save_config writes paths relative to the project" begin
            _PP_CM.save_config(_PP_CM.load_config(toml))
            raw = TOML.parsefile(toml)
            @test raw["project"]["root"] == "."
            @test raw["compile"]["source_files"] == ["src/a.c"]
            @test sort(raw["compile"]["include_dirs"]) == [".", "include"]
            @test !occursin(dir, read(toml, String))       # nothing machine-local left
        end

        @testset "load_config resolves them against the root, not the cwd" begin
            elsewhere = mktempdir()
            cfg = cd(() -> _PP_CM.load_config(toml), elsewhere)
            @test all(isabspath, cfg.compile.source_files)
            @test all(isabspath, cfg.compile.include_dirs)
            @test isfile(only(cfg.compile.source_files))
            @test normpath(only(cfg.compile.source_files)) == normpath(joinpath(dir, "src", "a.c"))
            @test any(d -> rstrip(normpath(d), ['/', '\\']) == rstrip(normpath(dir), ['/', '\\']),
                      cfg.compile.include_dirs)
            @test isempty(_PP_CM.validate_config(cfg))     # the source-exists check passes from anywhere
        end
    end
end
