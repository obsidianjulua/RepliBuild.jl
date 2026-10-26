# =============================================================================
# [wrap] shim_defines, live
#
# stbish.h is a single-header library in the stb style. Built from the TOML
# alone (`shim_headers` plus `shim_defines`, no source file, no [wrap.macros]
# entry), its implementation section compiles into the library and the wrapper
# calls it. Built beside a library source that includes the same header
# (user.c), the define reaches only the shim file, so the implementation is
# defined once and user.c links against it. The same project with the define
# in [compile] flags instead defines every function twice and the link stops:
# that is the case the key exists for. Each wrapper is called in a child
# process. Needs the C bucket only.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const SD_FIXTURE = joinpath(@__DIR__, "shim_defines_test")
const SD_PROBE = joinpath(SD_FIXTURE, "probe_shim_defines.jl")

# A project in a temp dir with copies of the fixture's files: the shim header
# guard requires the header inside the project tree.
function _sd_project(name::String, wrap::String; with_source::Bool,
                     flags::String = "\"-O2\", \"-fPIC\"")
    root = mktempdir()
    cp(joinpath(SD_FIXTURE, "include"), joinpath(root, "include"))
    sources = String[]
    if with_source
        cp(joinpath(SD_FIXTURE, "src"), joinpath(root, "src"))
        push!(sources, "\"src/user.c\"")
    end
    toml = joinpath(root, "replibuild.toml")
    write(toml, """
    [project]
    name = "$name"
    # escape_string: a Windows path in a TOML basic string would parse `\\U`.
    root = "$(escape_string(root))"

    [compile]
    flags = [$flags]
    source_files = [$(join(sources, ", "))]
    include_dirs = ["include"]

    [link]
    enable_lto = false
    optimization_level = "2"

    [binary]
    type = "shared"

    [wrap]
    language = "c"
    $wrap

    [types]
    strictness = "warn"
    allow_unknown_structs = true
    allow_function_pointers = true

    [cache]
    enabled = false
    """)
    return toml
end

function _sd_probe(wrapper::String, mode::String, expected::Vector{String})
    r = run_isolated(`$SD_PROBE $wrapper $mode`)
    v = probe_verdicts(r.output)
    @test r.ok
    @test haskey(v, "PROBE_DONE")
    checked = 0
    for p in expected
        @test (p, get(v, p, "no PROBE line")) == (p, "PASS")
        checked += 1
    end
    @test checked == length(expected)
end

@testset "[wrap] shim_defines, live" begin
    @testset "the TOML alone builds the implementation" begin
        toml = _sd_project("sd_alone", """
            shim_headers = ["stbish.h"]
            shim_defines = ["STBISH_IMPLEMENTATION"]
            """; with_source = false)
        RepliBuild.build(toml)
        wrapper = RepliBuild.wrap(toml)
        @test isfile(wrapper)
        _sd_probe(wrapper, "alone", ["add", "range", "span"])
    end

    @testset "beside a source that includes the header, with macros" begin
        toml = _sd_project("sd_user", """
            shim_headers = ["stbish.h"]
            shim_defines = ["STBISH_IMPLEMENTATION"]

            [wrap.macros.STBISH_VERSION]
            ret = "int"

            [wrap.macros.stbish_twice]
            ret = "int"
            args = ["int"]
            """; with_source = true)
        RepliBuild.build(toml)
        wrapper = RepliBuild.wrap(toml)
        @test isfile(wrapper)
        _sd_probe(wrapper, "user", ["add", "range", "span", "user_total", "version", "twice"])
    end

    @testset "the define in [compile] flags reaches user.c too" begin
        toml = _sd_project("sd_flags", """
            shim_headers = ["stbish.h"]

            [wrap.macros.STBISH_VERSION]
            ret = "int"
            """; with_source = true,
            flags = "\"-O2\", \"-fPIC\", \"-DSTBISH_IMPLEMENTATION\"")
        @test_throws ErrorException RepliBuild.build(toml)
    end
end
