# =============================================================================
# Tier 2 SysV record classes, live
#
# Builds sysv_live_test, a C++ library whose take_R(R) / make_R() cross one
# record per way a by-value record can be classified under x86-64 SysV, wraps
# it, and has a child process call every one of them in both directions, with
# the SSE registers poisoned first. The records are the ones
# test_sysv_abi_oracle.jl diffs against LLVM's classifier, and the ones its
# families named when the dialect still classified from StructGen's body:
#
#   * an eightbyte of a float and padding went in a general register
#     (`{float; double}`, `{void*; float}`, `{int : 32; float}`, `aligned(16)
#     float`, `_Complex float`, a `{double; float}` nested in a tiled body);
#   * a misaligned field inside a member the body spelled as bytes went unseen,
#     so a packed `{char; {short; float}}`, a union holding a packed record and
#     a `#pragma pack(1)` record holding an over-aligned one crossed in
#     registers where C uses memory (abi_fuzz seeds 21 and 22 segfaulted on
#     that shape: the callee read a shifted pointer);
#   * `#pragma pack(1) {char; int : 31}` went in memory where C uses one
#     register;
#   * `long double` and `_Complex double` members were `!llvm.ptr`, so their
#     records were 8 bytes;
#   * `aligned(16)` and `__int128` records got 8-aligned byval and sret slots.
#
# FunctionGen now classifies each record from its DWARF facts and states the
# verdict on the call (`sysv_args`, `sysv_ret`), which the dialect acts on. The
# probe checks values, not classes, so it holds on Win64 too, where the records
# cross by size. One record is refused on SysV by design: `make_c_ld` returns a
# `long double` record in st0, where no thunk looks; the wrapper traps it.
#
# Both optimisation levels of the callee are run: at -O0 a callee reads its
# arguments from the stack slots the prologue spilled, at -O2 straight from
# the registers, and the two miss a wrong register differently.
#
# Needs clang++ and libJLCS. The library-free side (the classifier over
# synthetic metadata, the statements the thunk emits) is test_sysv_record_abi.jl
# in runtests.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const SYSV_LIVE_DIR = joinpath(@__DIR__, "sysv_live_test")

# Every record, with what the probe must print for its take_ and make_
# directions on this host.
const SYSV_LIVE_RECORDS = [
    "c_f_d", "c_d_f", "c_p_f", "c_ubf_f", "c_f_pad", "c_wrap_df", "c_al16_f", "c_cf", "c_cd", "c_ubf_d",
    "c_ld", "c_u_ld_i", "c_s_f", "c_pk_sf", "c_pk_ci", "c_u_pk", "c_pk_alnest", "c_pp1_bf", "c_u_fi",
    "c_bf2_f", "c_i128_l", "c_al16_big", "c_f_i", "c_d2", "c_c_d",
]

function sysv_live_expected(record::String, direction::String)
    # SysV returns `{ long double }` in st0, which the thunk does not read, so
    # the generator traps the return. Win64 returns the 16-byte record through
    # a hidden pointer and the call works.
    record == "c_ld" && direction == "make" && !Sys.iswindows() && return "TRAP"
    return "PASS"
end

function sysv_live_toml(opt::String)
    src = joinpath(SYSV_LIVE_DIR, "src", "sysv_live.cpp")
    inc = joinpath(SYSV_LIVE_DIR, "include")
    return """
    [project]
    name = "sysv_live"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(SYSV_LIVE_DIR))"

    [compile]
    flags = ["-std=c++17", "-fPIC"]
    source_files = ["$(escape_string(src))"]
    include_dirs = ["$(escape_string(inc))"]

    [link]
    enable_lto = false
    optimization_level = "$opt"

    [binary]
    type = "shared"

    [wrap]
    language = "cpp"

    [types]
    strictness = "warn"
    allow_unknown_structs = true
    allow_function_pointers = true

    [cache]
    enabled = false
    """
end

if requires("SysV record classes, live", :libJLCS, :clangxx)
@testset "Tier 2 SysV record classes, live" begin
    toml_path = joinpath(SYSV_LIVE_DIR, "replibuild.toml")
    for opt in ("0", "2")
        @testset "callee at -O$opt" begin
            # Regenerated each run: it holds absolute, machine-local paths.
            write(toml_path, sysv_live_toml(opt))
            RepliBuild.build(toml_path)
            RepliBuild.wrap(toml_path)
            wrapper = joinpath(SYSV_LIVE_DIR, "julia", "SysvLive.jl")
            @test isfile(wrapper)
            code = read(wrapper, String)

            # Every take_/make_ is a C++ function that may throw: Tier 2. The
            # one the generator refuses is a trap stub, which has no tier.
            tiers = Dict(m.captures[1] => m.captures[2]
                         for m in eachmatch(r"\n\s+:(\w+) => :(tier\d)", code))
            checked = 0
            for r in SYSV_LIVE_RECORDS, dir in ("take", "make")
                f = "$(dir)_$r"
                want = sysv_live_expected(r, dir) == "TRAP" ? "missing" : "tier2"
                @test (f, get(tiers, f, "missing")) == (f, want)
                checked += 1
            end
            @test checked == 2 * length(SYSV_LIVE_RECORDS)

            r = run_isolated(`$(joinpath(SYSV_LIVE_DIR, "probe_sysv_live.jl")) $wrapper`)
            v = probe_verdicts(r.output)
            @test r.ok
            @test haskey(v, "PROBE_DONE")
            probed = 0
            for rec in SYSV_LIVE_RECORDS, dir in ("take", "make")
                name = "$rec/$dir"
                want = sysv_live_expected(rec, dir)
                got = get(v, name, "no PROBE line")
                @test (name, first(split(got, ' '))) == (name, want)
                probed += 1
            end
            @test probed == 2 * length(SYSV_LIVE_RECORDS)
            # No record's Julia type disagrees with C about its size.
            @test !any(k -> endswith(k, "/size"), keys(v))
        end
    end
end
end # requires
