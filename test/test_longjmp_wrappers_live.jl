# =============================================================================
# A Julia callback leaves through a wrapped C function that longjmps, live
# (2026-11-21)
#
# test_cstring_arg_roots.jl reads the compiled wrapper. This makes the jump.
#
# test/longjmp_wrapper_test/escape.c is the shape of Lua's error path with no
# Lua in it: `lj_protected` sets a jump point and calls back, and the callback
# calls `lj_fail(const char*)` or the variadic `lj_failf`, which longjmp to
# it. On the 4.6.0 release sweep the Hub's lua died in the collector on this
# path, every run: the wrapper held a GC root across a call that never
# returned, and the abandoned GC frame was read at the next collection.
#
# The probe runs in a child (its failure is a crash): 200 jumps per shape,
# the abandoned stack overwritten and the collector run inside the frame that
# made the protected call. With `cstring_arg` as it was, the child aborts with
# "GC error (probable corruption)" before its first PROBE line.
#
# Also here, in this process, since they return: bytes passed with their
# length (`codeunits`, a byte vector) cross with their NULs, which is what the
# refusal of a NUL-holding `String` tells a caller to pass.
#
# Needs the C bucket only.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())

const LJ_DIR = joinpath(@__DIR__, "longjmp_wrapper_test")

function lj_body(wsrc::AbstractString, name::AbstractString)
    m = match(Regex("^function $(name)\\(.*?^end\$", "ms"), wsrc)
    return m === nothing ? "" : String(m.match)
end

function lj_build()
    for d in ("build", "julia", ".replibuild_cache", ".debug")
        p = joinpath(LJ_DIR, d)
        ispath(p) && RepliBuild._rm_tree(p)
    end
    toml_path = joinpath(LJ_DIR, "replibuild.toml")
    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "longjmp_wrapper"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(LJ_DIR))"

    [compile]
    flags = ["-O1", "-fPIC"]
    source_files = ["$(escape_string(joinpath(LJ_DIR, "escape.c")))"]
    include_dirs = ["$(escape_string(LJ_DIR))"]

    [link]
    enable_lto = false
    optimization_level = "0"

    [binary]
    type = "shared"

    [wrap]
    language = "c"

    [wrap.varargs]
    lj_failf = [["Cint"]]

    [types]
    strictness = "warn"
    allow_unknown_structs = true

    [cache]
    enabled = false
    """)
    withenv("REPLIBUILD_JULIA_ENV" => "none") do
        redirect_stdout(devnull) do
            RepliBuild.build(toml_path)
        end
        RepliBuild.wrap(toml_path)
    end
    return joinpath(LJ_DIR, "julia", "LongjmpWrapper.jl")
end

@testset "A callback leaves through a wrapper that longjmps, live" begin
    wrapper = lj_build()
    @test isfile(wrapper)
    wsrc = isfile(wrapper) ? read(wrapper, String) : ""

    @testset "the wrappers on the path bind the text through cstring_arg" begin
        @test occursin("RepliBuild.JITManager.cstring_arg(message)", lj_body(wsrc, "lj_fail"))
        @test occursin("RepliBuild.JITManager.cstring_arg(fmt)", lj_body(wsrc, "lj_failf_Cint"))
    end

    @testset "what returns: text, and bytes passed with their length" begin
        M = Base.include(Module(), wrapper)
        @test M.lj_len("hello") == 5
        @test M.lj_len(SubString("hello world", 1, 5)) == 5
        # A NUL among bytes that travel with their length is data.
        @test M.lj_count_nul(codeunits("a\0b\0"), 4) == 2
        @test M.lj_count_nul(UInt8[0x00, 0x61, 0x00, 0x00], 4) == 3
        # The same bytes as a String are refused, and the message names both.
        err = try
            M.lj_count_nul("a\0b\0", 4)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("codeunits", err.msg)
    end

    @testset "200 jumps per shape, the collector run on the abandoned frames" begin
        if Sys.iswindows()
            # mingw's longjmp unwinds through the JIT'd frames by SEH; that it
            # lands has not been run. The compiled-wrapper checks in
            # test_cstring_arg_roots.jl run on every host.
            @info "Skipping the longjmp probe: not run on Windows"
            @test_skip false
        else
            r = run_isolated(`$(joinpath(LJ_DIR, "probe.jl")) $wrapper 200`; timeout = 600)
            v = probe_verdicts(r.output)
            @test r.ok
            @test haskey(v, "PROBE_DONE")
            for name in ("plain", "variadic", "returns", "still_calls")
                @test (name, startswith(get(v, name, "no PROBE line"), "PASS")) == (name, true)
            end
        end
    end
end
