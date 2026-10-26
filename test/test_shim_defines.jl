#!/usr/bin/env julia
# test/test_shim_defines.jl — [wrap] shim_defines (runtests; the guard case
# needs the JLL clang, as test_shim_header_guard.jl does)
#
# `shim_defines` puts `#define`s at the top of the generated shim file, before
# its #includes, so a single-header library's implementation section compiles
# into the library from the TOML alone: `shim_headers = ["stb_image.h"]` plus
# `shim_defines = ["STB_IMAGE_IMPLEMENTATION"]`, no source file, no macro entry.
# Before the key, the only TOML-only way was a -D in [compile] flags plus a
# throwaway [wrap.macros] entry (no entry, no shim file), and the -D reached
# every file the build compiles. These pin the file written, the refusals at
# load, the round trip and the header guard; test_shim_defines_live.jl
# (devtests) builds and calls one.

using Test
using TOML
using RepliBuild

const _SD_CM = RepliBuild.ConfigurationManager
const _SD_C = RepliBuild.Compiler

function _sd_config(dir::String, wrap_extra::String; compile_extra::String = "")
    toml = joinpath(dir, "replibuild.toml")
    write(toml, """
    [project]
    name = "shimdef"
    # escape_string: a Windows path in a TOML basic string would parse `\\U`.
    root = "$(escape_string(dir))"

    [compile]
    $compile_extra

    [wrap]
    language = "c"
    $wrap_extra
    """)
    return _SD_CM.load_config(toml)
end

# The shim file's text, and the offset of each needle in it (nothing if absent).
function _sd_shim(cfg)
    files = _SD_C.generate_macro_shims(cfg, String[])
    @test length(files) == 1
    return read(only(files), String)
end
_sd_at(txt, needle) = (r = findfirst(needle, txt); r === nothing ? typemax(Int) : first(r))

@testset "[wrap] shim_defines" begin

    @testset "defines alone write the file, before the includes" begin
        mktempdir() do dir
            cfg = _sd_config(dir, """
            shim_headers = ["hol.h"]
            shim_defines = ["HOL_IMPLEMENTATION", "HOL_ALLOC=my_alloc"]
            """)
            @test cfg.wrap.shim_defines == ["HOL_IMPLEMENTATION", "HOL_ALLOC=my_alloc"]
            @test isempty(cfg.wrap.macros)
            txt = _sd_shim(cfg)
            inc = _sd_at(txt, "#include \"hol.h\"")
            # NAME alone is 1, as -DNAME would be, so `#if NAME` works too.
            @test _sd_at(txt, "#define HOL_IMPLEMENTATION 1\n") < inc
            @test _sd_at(txt, "#define HOL_ALLOC my_alloc\n") < inc
            @test inc < typemax(Int)
            # No macros: no shim functions, so nothing of ours is exported.
            @test !occursin("RB_SHIM_EXPORT", txt)
            @test !occursin("replibuild_shim_", txt)
        end
    end

    @testset "defines and macros share one file" begin
        mktempdir() do dir
            cfg = _sd_config(dir, """
            shim_headers = ["hol.h"]
            shim_defines = ["HOL_IMPLEMENTATION"]

            [wrap.macros.HOL_SQUARE]
            ret = "int"
            args = ["int"]
            """)
            txt = _sd_shim(cfg)
            @test _sd_at(txt, "#define HOL_IMPLEMENTATION 1\n") < _sd_at(txt, "#include \"hol.h\"")
            @test occursin("RB_SHIM_EXPORT int replibuild_shim_HOL_SQUARE(int arg0)", txt)
            @test occursin("return HOL_SQUARE(arg0);", txt)
            # The two RB_SHIM_EXPORT definitions plus one per macro, as before.
            @test count("RB_SHIM_EXPORT", txt) == 2 + 1
        end
    end

    @testset "neither key: no file, sources unchanged" begin
        mktempdir() do dir
            cfg = _sd_config(dir, """shim_headers = ["hol.h"]""")
            @test _SD_C.generate_macro_shims(cfg, ["a.c"]) == ["a.c"]
            @test !isfile(joinpath(_SD_C.get_cache_path(cfg), "replibuild_shims.c"))
        end
    end

    @testset "shim_define_line" begin
        @test _SD_CM.shim_define_line("X") == "#define X 1"
        @test _SD_CM.shim_define_line("X=2") == "#define X 2"
        @test _SD_CM.shim_define_line("X=a=b") == "#define X a=b"
        @test _SD_CM.shim_define_line("X=") == "#define X "
    end

    @testset "refused at load" begin
        checked = 0
        for bad in ("\"HOL_IMPL\\nint injected;\"",   # a line break would add code
                    "\"HOL_IMPL\\n\"",                 # even a trailing one (`\$` matches before it)
                    "\"1BAD\"", "\"A B\"", "\"\"", "\"-DX\"", "3")
            mktempdir() do dir
                @test_throws ErrorException _sd_config(dir, """
                shim_headers = ["hol.h"]
                shim_defines = [$bad]
                """)
            end
            checked += 1
        end
        @test checked == 7
        # Defines with nothing to apply to.
        mktempdir() do dir
            err = try
                _sd_config(dir, """shim_defines = ["HOL_IMPLEMENTATION"]"""); nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("shim_headers is empty", sprint(showerror, err))
        end
    end

    @testset "save_config writes it back" begin
        mktempdir() do dir
            cfg = _sd_config(dir, """
            shim_headers = ["hol.h"]
            shim_defines = ["HOL_IMPLEMENTATION"]
            """)
            _SD_CM.save_config(cfg)
            doc = TOML.parsefile(joinpath(dir, "replibuild.toml"))
            @test doc["wrap"]["shim_defines"] == ["HOL_IMPLEMENTATION"]
            @test _SD_CM.load_config(joinpath(dir, "replibuild.toml")).wrap.shim_defines ==
                  ["HOL_IMPLEMENTATION"]
        end
    end

    # The guard used to return when there were no macros, so a defines-only
    # shim could compile a system copy of the header's implementation unchecked.
    @testset "header guard runs for a defines-only shim" begin
        proj = mktempdir()
        system = mktempdir()      # stands in for /usr/include, outside the tree
        write(joinpath(system, "collide.h"), "int collide_add(int a, int b);\n")
        cfg = _sd_config(proj, """
            shim_headers = ["collide.h"]
            shim_defines = ["COLLIDE_IMPLEMENTATION"]
            """; compile_extra = """include_dirs = ["$(escape_string(system))"]""")
        files = _SD_C.generate_macro_shims(cfg, String[])
        err = try
            _SD_C.verify_shim_headers(cfg, only(files)); nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test err !== nothing && occursin("header collision", sprint(showerror, err))
    end
end
