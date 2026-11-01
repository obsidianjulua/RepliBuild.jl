# =============================================================================
# SIMD vectors by value, live
#
# Builds simd_live_test twice, as C (GeneratorC: Tier 3 ccalls) and as C++
# (its take_/make_ go through Tier 2 thunks), and has a child process call every
# record of simd_live.h and the bare 8- and 16-byte vectors in both directions,
# with the SSE registers poisoned first, values checked against C. Each host
# asserts its own convention:
#
#   * SysV classifies a vector by its size. Tier 2 states that class
#     (`sysv_record_abi`), so every record holding a vector of 8 bytes or less
#     is called; the 16-byte one is one SSE register, which no call statement
#     states, and is refused. Tier 3's `ccall` classifies the `NTuple` standing
#     in for a vector lane by lane, so only the records whose lanes classify as
#     the vector does (`{<2 x float>}`, `{<2 x float>; int}`) are called and the
#     rest are refused (`sysv_vector_misread`). This is the only place a SysV
#     vector call runs: CI's tier2-linux.yml.
#   * Win64 passes and returns every record by size, on both tiers. The
#     16-byte `{<4 x float>}` comes back through GeneratorC's 16-aligned
#     buffer (BRANCH 0), since the callee stores it with `movaps`.
#   * A bare vector never crosses a thunk (one register per lane), and no Julia
#     type returns one, so both are refused on both hosts. A bare 16-byte
#     argument through Tier 3 is right on Win64 (both sides pass it by
#     reference). A bare 8-byte one is not: clang passes it by reference and
#     ccall by value, and the callee faults. That call runs in a child of its
#     own and is `@test_broken` until the Win64 refusal is decided.
#
# Needs clang, clang++ and libJLCS. The library-free side is
# test_simd_vectors.jl in runtests.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const SIMD_LIVE_DIR = joinpath(@__DIR__, "simd_live_test")
const SIMD_LIVE_RECORDS = ["r_v1f", "r_v2i", "r_v2f", "r_v1d", "r_v2f_i", "r_v4f"]
const SIMD_LIVE_BARE = ["v2f", "v4f"]

"""
    simd_live_expected(lang, name, dir) -> String

What the probe must print for `<name>/<dir>` from the `lang` (`:c` or `:cpp`)
build on this host: `PASS`, `TRAP`, or `BROKEN` for the Win64 bare 8-byte
argument through Tier 3, which faults and runs on its own.
"""
function simd_live_expected(lang::Symbol, name::String, dir::String)
    win = Sys.iswindows()
    if name in SIMD_LIVE_BARE
        dir == "make" && return "TRAP"          # no Julia type returns a vector
        lang === :cpp && return "TRAP"          # a thunk would pass the lanes
        win || return "TRAP"                    # ccall reads the lanes, C the vector
        return name == "v4f" ? "PASS" : "BROKEN"
    end
    win && return "PASS"                        # Win64: by size, on both tiers
    name == "r_v4f" && return "TRAP"            # one SSE register (SSE, SSEUP)
    lang === :cpp && return "PASS"              # Tier 2 states the vector's class
    return name in ("r_v2f", "r_v2f_i") ? "PASS" : "TRAP"
end

function simd_live_project(lang::Symbol, opt::String)
    dir = joinpath(mktempdir(), "simd_live")
    cp(SIMD_LIVE_DIR, dir)
    src = joinpath(dir, "src", lang === :c ? "simd_live.c" : "simd_live.cpp")
    toml = joinpath(dir, "replibuild.toml")
    write(toml, """
    [project]
    name = "simd_live_$lang"
    root = "$(escape_string(dir))"

    [compile]
    flags = [$(lang === :cpp ? "\"-std=c++17\", " : "")"-fPIC"]
    source_files = ["$(escape_string(src))"]
    include_dirs = ["$(escape_string(joinpath(dir, "include")))"]

    [link]
    enable_lto = false
    optimization_level = "$opt"

    [binary]
    type = "shared"

    [wrap]
    language = "$(lang === :c ? "c" : "cpp")"

    [types]
    strictness = "warn"
    allow_unknown_structs = true

    [cache]
    enabled = false
    """)
    return toml
end

if requires("SIMD vectors by value, live", :libJLCS, :clang, :clangxx)
@testset "SIMD vectors by value, live" begin
    probe = joinpath(SIMD_LIVE_DIR, "probe_simd_live.jl")
    for lang in (:c, :cpp), opt in ("0", "2")
        @testset "$(lang === :c ? "C (Tier 3)" : "C++ (Tier 2)"), callee at -O$opt" begin
            toml = simd_live_project(lang, opt)
            withenv("REPLIBUILD_HOME" => mktempdir(), "REPLIBUILD_JULIA_ENV" => "none") do
                RepliBuild.build(toml)
                RepliBuild.wrap(toml)
            end
            wrapper = joinpath(dirname(toml), "julia", RepliBuild.ConfigurationManager.wrapper_module_name(toml) * ".jl")
            @test isfile(wrapper)
            code = read(wrapper, String)

            # Every C++ take_/make_ that is called goes through its thunk.
            if lang === :cpp
                tiers = Dict(m.captures[1] => m.captures[2] for m in eachmatch(r"\n\s+:(\w+) => :(tier\d)", code))
                for r in SIMD_LIVE_RECORDS, dir in ("take", "make")
                    f = "$(dir)_$r"
                    want = simd_live_expected(lang, r, dir) == "PASS" ? "tier2" : "missing"
                    @test (f, get(tiers, f, "missing")) == (f, want)
                end
            end
            # Win64 returns the 16-aligned record through a buffer GeneratorC
            # aligned itself, which a plain ccall result cannot tell apart.
            if lang === :c && Sys.iswindows()
                i = findfirst("function make_r_v4f(", code)
                @test i !== nothing && occursin("ret_ptr", code[first(i):first(findnext("\nend", code, first(i)))])
            end

            broken = [n for n in SIMD_LIVE_BARE, d in ("take", "make") if simd_live_expected(lang, n, d) == "BROKEN"]
            skips = ["$n/take" for n in broken]
            r = run_isolated(`$probe $wrapper $skips`)
            v = probe_verdicts(r.output)
            @test r.ok
            @test haskey(v, "PROBE_DONE")
            probed = 0
            for name in vcat(SIMD_LIVE_RECORDS, SIMD_LIVE_BARE), dir in ("take", "make")
                want = simd_live_expected(lang, name, dir)
                want == "BROKEN" && continue
                got = get(v, "$name/$dir", "no PROBE line")
                @test ("$name/$dir", first(split(got, ' '))) == ("$name/$dir", want)
                probed += 1
            end
            @test probed == 2 * (length(SIMD_LIVE_RECORDS) + length(SIMD_LIVE_BARE)) - length(broken)
            @test !any(k -> endswith(k, "/size"), keys(v))

            # The Win64 bare 8-byte argument through Tier 3, in a child of its
            # own: clang passes it by reference and ccall by value.
            for n in broken
                others = ["$m/$d" for m in vcat(SIMD_LIVE_RECORDS, SIMD_LIVE_BARE), d in ("take", "make")
                          if "$m/$d" != "$n/take"]
                rb = run_isolated(`$probe $wrapper $others`; echo = false)
                @test_broken startswith(get(probe_verdicts(rb.output), "$n/take", "no PROBE line"), "PASS")
            end
        end
    end
end
end # requires
