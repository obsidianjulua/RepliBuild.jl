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
#     type returns one, so both are refused on both hosts. Through Tier 3 on
#     Win64, clang passes every bare vector by reference to a copy aligned to
#     its size: the 16-byte one crosses inside GeneratorC's `_RB_ByVal16{1}`
#     (a 16-aligned copy) and is called; the 8-byte one, which ccall would put
#     in a register, is refused (it faulted, 2026-11-01 audit of #60).
#   * The `take_c_*`/`make_c_*` functions have C linkage in the C++ build too,
#     so GeneratorCpp's ccall gate (`is_ccall_safe`) judges them: a vector
#     crossing takes the thunk, which gives the same answers as the C++ ones
#     above. They used to be plain ccalls of the shapes the C build refuses.
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
# C linkage in both builds: the records and the bare vector `take_c_`/`make_c_` cross.
const SIMD_LIVE_CLINK_RECORDS = ["c_r_v4f", "c_r_v2f"]
const SIMD_LIVE_CLINK_BARE = ["c_v4f"]                   # take only

"""
    simd_live_expected(lang, name, dir) -> String

What the probe must print for `<name>/<dir>` from the `lang` (`:c` or `:cpp`)
build on this host: `PASS` or `TRAP`. A `c_` name is the C-linkage twin of the
record or vector after the prefix, which crosses the same way in the C build
and, in the C++ build, the way the thunk crosses it.
"""
function simd_live_expected(lang::Symbol, name::String, dir::String)
    win = Sys.iswindows()
    base = replace(name, r"^c_" => "")
    if base in SIMD_LIVE_BARE
        dir == "make" && return "TRAP"          # no Julia type returns a vector
        lang === :cpp && return "TRAP"          # a thunk would pass the lanes
        win || return "TRAP"                    # ccall reads the lanes, C the vector
        return base == "v4f" ? "PASS" : "TRAP"  # Win64: 16 bytes in the carrier; 8 bytes would go in a register
    end
    win && return "PASS"                        # Win64: by size, on both tiers
    base == "r_v4f" && return "TRAP"            # one SSE register (SSE, SSEUP)
    lang === :cpp && return "PASS"              # Tier 2 states the vector's class
    return base in ("r_v2f", "r_v2f_i") ? "PASS" : "TRAP"
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

            # Every C++ take_/make_ that is called goes through its thunk, the
            # C-linkage twins included: the gate sends a vector crossing there.
            if lang === :cpp
                tiers = Dict(m.captures[1] => m.captures[2] for m in eachmatch(r"\n\s+:(\w+) => :(tier\d)", code))
                for r in vcat(SIMD_LIVE_RECORDS, SIMD_LIVE_CLINK_RECORDS), dir in ("take", "make")
                    f = "$(dir)_$r"
                    want = simd_live_expected(lang, r, dir) == "PASS" ? "tier2" : "missing"
                    @test (f, get(tiers, f, "missing")) == (f, want)
                end
                @test get(tiers, "take_c_v4f", "missing") == "missing"
                @test !occursin(r"function take_c_v4f\([^\n]*\n\s*(?:#[^\n]*\n\s*)*ccall\(", code)
            end
            # Win64, C: the 16-aligned record comes back through a buffer
            # GeneratorC aligned itself, and the 16-byte bare vector goes in
            # through the carrier; neither is a plain ccall by its result. The
            # 8-byte one is a refusal, named as such.
            if lang === :c && Sys.iswindows()
                body(f) = let i = findfirst("function $f(", code)
                    i === nothing ? "" : code[first(i):first(findnext("\nend", code, first(i)))]
                end
                @test occursin("ret_ptr", body("make_r_v4f"))
                @test occursin("_rb_byval16(", body("take_v4f"))
                @test occursin("_rb_byval16(", body("take_c_v4f"))
                @test occursin("_rb_byval16(", body("take_r_v4f"))
                @test occursin("ABI Safety Trap", body("take_v2f")) && occursin("in a register", body("take_v2f"))
            end

            r = run_isolated(`$probe $wrapper`)
            v = probe_verdicts(r.output)
            @test r.ok
            @test haskey(v, "PROBE_DONE")
            probed = 0
            cases = vcat(vec([(n, d) for n in vcat(SIMD_LIVE_RECORDS, SIMD_LIVE_BARE, SIMD_LIVE_CLINK_RECORDS), d in ("take", "make")]),
                         [(n, "take") for n in SIMD_LIVE_CLINK_BARE])
            for (name, dir) in cases
                want = simd_live_expected(lang, name, dir)
                got = get(v, "$name/$dir", "no PROBE line")
                @test ("$name/$dir", first(split(got, ' '))) == ("$name/$dir", want)
                probed += 1
            end
            @test probed == length(cases)
            @test !any(k -> endswith(k, "/size"), keys(v))
        end
    end
end
end # requires
