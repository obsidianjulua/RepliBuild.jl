# =============================================================================
# `char` by value, signed bitfields and enum parameters, live (2026-11-18)
#
# Three wrong answers the 2026-11-17 audit found (B14, B15):
#   * a signed bitfield read as unsigned: `int c:7 = -3` came back 125, and
#     its setter threw InexactError for -3;
#   * a `char` parameter typed and passed as `UInt8` while a `char` return is
#     `Cchar`: `f(Cchar(-5))` was an InexactError, and `ccall` passed the byte
#     zero-extended where clang's callee reads a `char` sign-extended (x86-64);
#     a Tier 2 thunk extended it by the metadata's `UInt8` the same way;
#   * a C enum parameter refused an integer, which C converts implicitly.
#
# This builds a C library and a C++ one with the toolchains, wraps them, and
# calls each from a child (`run_isolated`). The callees return `(int)c`, which
# clang compiles to a move of the argument register, so the extension the
# caller did is what comes back. Every expectation is stated through the
# host's `char` (`reinterpret(Cchar, 0xfb)`): signed on x86-64, unsigned on
# AArch64. The C++ half needs clang++ and libJLCS.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())
ENV["REPLIBUILD_JULIA_ENV"] = "none"

const CB_C_DIR = joinpath(@__DIR__, "char_bitfield_c_test")
const CB_CPP_DIR = joinpath(@__DIR__, "char_bitfield_cpp_test")

# Regenerated each run: it holds absolute, machine-local paths.
function _cb_toml(dir, name, lang, src, flags)
    toml = joinpath(dir, "replibuild.toml")
    write(toml, """
    [project]
    name = "$name"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(dir))"

    [compile]
    flags = [$(join(repr.(flags), ", "))]
    source_files = ["$(escape_string(joinpath(dir, src)))"]
    include_dirs = ["$(escape_string(dir))"]

    [link]
    enable_lto = false
    optimization_level = "2"

    [binary]
    type = "shared"

    [wrap]
    language = "$lang"

    [types]
    strictness = "warn"
    allow_unknown_structs = true

    [cache]
    enabled = false
    """)
    return toml
end

function _cb_run(dir, probe, wrapper, names)
    r = run_isolated(`$(joinpath(dir, probe)) $wrapper`)
    v = probe_verdicts(r.output)
    @test r.ok
    @test haskey(v, "PROBE_DONE")
    for n in names
        @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
    end
    @test count(n -> haskey(v, n), names) == length(names)
end

@testset "char, signed bitfields and enum parameters: C, live" begin
    toml = _cb_toml(CB_C_DIR, "charbitc", "c", "cb.c", ["-std=c11", "-fPIC", "-O2"])
    redirect_stdout(devnull) do
        RepliBuild.build(toml)
    end
    RepliBuild.wrap(toml)
    wrapper = joinpath(CB_C_DIR, "julia", "Charbitc.jl")
    @test isfile(wrapper)
    _cb_run(CB_C_DIR, "probe_char_bitfield_c.jl", wrapper,
            ["char_takes_cchar_and_bytes", "char_return_is_cchar", "signed_char_is_int8",
             "char_into_a_record", "signed_bitfields_read", "signed_bitfields_write",
             "enum_bitfield_follows_underlying", "enum_param_takes_integer"])
end

if requires("char and signed bitfields through C++, live", :libJLCS, :clangxx)
@testset "char and signed bitfields: C++, live" begin
    toml = _cb_toml(CB_CPP_DIR, "charbitcpp", "cpp", "cbx.cpp",
                    ["-std=c++17", "-fPIC", "-O2", "-fstandalone-debug"])
    redirect_stdout(devnull) do
        RepliBuild.build(toml)
    end
    RepliBuild.wrap(toml)
    wrapper = joinpath(CB_CPP_DIR, "julia", "Charbitcpp.jl")
    @test isfile(wrapper)
    _cb_run(CB_CPP_DIR, "probe_char_bitfield_cpp.jl", wrapper,
            ["char_value_thunk", "char_value_ccall", "char_return", "char_in_an_overload_set",
             "signed_bitfield_in_a_cpp_record"])
end
end # requires
