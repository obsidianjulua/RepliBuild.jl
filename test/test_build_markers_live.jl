# test/test_build_markers_live.jl — the project-hash marker names what was built.
#
# `compile_project` returns the existing library without looking at anything
# when the stored project hash matches. Three ways it matched a library that
# was not built from those inputs (audit 2026-11-05):
#
#   B1  A build that failed after the link kept the previous build's marker
#       beside its new library. Reverting the change restored those inputs,
#       and the fast path returned the failed build's library.
#   B2  The marker was hashed AFTER the build, so a file saved while the build
#       ran was recorded as built over a library compiled from the version
#       before it.
#   B3  A compiler that never started counted as a compile: the previous IR
#       was linked and the build reported success.
#
# Real C builds (JLL clang), with `f()` called in a child process: Windows
# locks a loaded DLL, and glibc would hand back the first mapping of a path.
# Negative-checked against the pre-fix `compile_project` and
# `compile_single_to_ir`: B1's revert takes the fast path and `f` is not
# exported, B2 returns 2, and B3's build succeeds.

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) || include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const _BM_C  = RepliBuild.Compiler
const _BM_CM = RepliBuild.ConfigurationManager

_bm_tp(p) = replace(p, '\\' => '/')

function _bm_toml(dir, src; compile = "", defines = Dict{String,String}())
    defs = isempty(defines) ? "" :
        "[compile.defines]\n" * join(["$k = \"$v\"" for (k, v) in sort(collect(defines))], "\n")
    return """
    [project]
    name = "markers"
    root = "$(_bm_tp(dir))"

    [compile]
    flags = ["-O1"]
    source_files = ["$(_bm_tp(src))"]
    $compile
    $defs

    [link]
    enable_lto = false

    [binary]
    type = "shared"

    [wrap]
    language = "c"

    [cache]
    enabled = true
    """
end

function _bm_project(dir, source)
    src = joinpath(dir, "src", "a.c")
    mkpath(dirname(src))
    write(src, source)
    return (src, joinpath(dir, "replibuild.toml"))
end

# `f()` as the library answers it, or `nothing` when it does not export `f`.
function _bm_call_f(lib)
    code = "using Libdl; h = dlopen(ARGS[1]); print(ccall(dlsym(h, :f), Cint, ())); dlclose(h)"
    out = IOBuffer()
    p = run(pipeline(ignorestatus(`$(Base.julia_cmd()) --startup-file=no -e $code $lib`);
                     stdout = out, stderr = devnull))
    return p.exitcode == 0 ? parse(Int, String(take!(out))) : nothing
end

_bm_fast_path(toml) = _BM_C.is_project_cache_valid(_BM_CM.load_config(toml))
_bm_marker(toml) = _BM_C._project_hash_path(_BM_CM.load_config(toml))

@testset "Build markers name what was built" begin

    @testset "B1: a failed build is not revived by a revert" begin
        mktempdir() do d
            (src, toml) = _bm_project(d, "int f(void) { return 1; }\n")
            good = _bm_toml(d, src)
            write(toml, good)
            lib = RepliBuild.build(toml)
            @test _bm_call_f(lib) == 1

            # Links, then the wrap-surface guard refuses it (R1): every function
            # went hidden and static promotion renamed it away.
            write(toml, _bm_toml(d, src; compile = "visibility = \"hidden\""))
            @test_throws ErrorException RepliBuild.build(toml)
            @test !isfile(_bm_marker(toml))

            write(toml, good)
            @test !_bm_fast_path(toml)
            lib = RepliBuild.build(toml)
            @test _bm_call_f(lib) == 1
        end
    end

    @testset "B2: a file saved during the build is not recorded as built" begin
        mktempdir() do d
            (src, toml) = _bm_project(d, "int f(void) { return 2; }\n")
            write(toml, _bm_toml(d, src))
            # The child saves the next version while its build runs: after the
            # compile and link, before the marker is written.
            hook = """
            using RepliBuild
            @eval RepliBuild.Compiler function verify_wrap_surface(config::RepliBuildConfig,
                                                                 binary_path::String,
                                                                 metadata_path::Union{Nothing,String})
                write(ARGS[2], "int f(void) { return 3; }\\n")
                return nothing
            end
            RepliBuild.build(ARGS[1])
            """
            r = run_isolated(`-e $hook $toml $src`; timeout = 600)
            @test r.ok
            @test occursin("return 3", read(src, String))   # the hook ran
            @test !_bm_fast_path(toml)
            lib = RepliBuild.build(toml)
            @test _bm_call_f(lib) == 3
        end
    end

    @testset "B3: a compiler that cannot start fails the build" begin
        mktempdir() do d
            (src, toml) = _bm_project(d, "int f(void) { return VAL; }\n")
            write(toml, _bm_toml(d, src; defines = Dict("VAL" => "1")))
            lib = RepliBuild.build(toml)
            @test _bm_call_f(lib) == 1
            ir = _BM_C._ir_output_path(_BM_C.get_build_path(_BM_CM.load_config(toml)), src)
            @test isfile(ir)

            # 200,000 characters in one argument: the spawn fails on Windows
            # (32,767 for the command line) and Linux (128 KiB per argument).
            write(toml, _bm_toml(d, src; defines = Dict("VAL" => "2", "PAD" => "x"^200_000)))
            @test_throws ErrorException RepliBuild.build(toml)
            @test !isfile(ir)
            @test !isfile(_bm_marker(toml))

            write(toml, _bm_toml(d, src; defines = Dict("VAL" => "2")))
            lib = RepliBuild.build(toml)
            @test _bm_call_f(lib) == 2
        end
    end
end
