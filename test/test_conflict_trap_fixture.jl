# =============================================================================
# A name several types share never crosses by value, live (2026-10-23)
#
# RepliBuild keys types by their bare DWARF name and keeps one definition per
# name, so a function taking or returning one of several types that share a
# name, or a record holding one, has no layout it can trust. Two fixtures give
# one name to distinct types, built from source:
#
#   * conflict_trap_test: two C++ units, each with an `impl`, a `Mode` and a
#     `Kind` in its own namespace. `Mode` is 4 bytes in both; `Kind` is 1 byte
#     in one and 4 in the other.
#   * conflict_trap_c_test: two C files, each with its own `struct state`.
#
# The real DWARF must give every definition's size (`conflict_sizes`). Each
# function a conflicted name crosses by value, directly or inside a record, must
# be a trap with no tier row and no thunk, and the rest must keep their tiers. A
# child process then calls everything: the traps raise, and a record that is one
# type, a pointer to a conflicted one and `Mode` return the right values.
#
# The C half needs only the C bucket; the C++ half needs clang++ and libJLCS
# (two of its controls are Tier 2 thunks).
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const CONFLICT_TRAP_DIR = joinpath(@__DIR__, "conflict_trap_test")
const CONFLICT_TRAP_C_DIR = joinpath(@__DIR__, "conflict_trap_c_test")
const CONFLICT_TRAP_PROBE = joinpath(CONFLICT_TRAP_DIR, "probe_conflicts.jl")

# A generated function's body, from its `function` line to the next top-level `end`.
_ct_body(code, f) = (m = match(Regex("\\nfunction $f\\(.*?\\n(.*?)\\nend\\n", "s"), code);
                     m === nothing ? "" : String(m.captures[1]))

function _ct_toml(dir, name, lang, sources; extra = "")
    toml_path = joinpath(dir, "replibuild.toml")
    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "$name"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(dir))"

    [compile]
    flags = $(lang == "cpp" ? "[\"-std=c++17\", \"-fPIC\"]" : "[\"-O0\", \"-fPIC\"]")
    source_files = [$(join(["\"$(escape_string(joinpath(dir, "src", s)))\"" for s in sources], ", "))]
    $extra

    [link]
    enable_lto = false
    optimization_level = "0"

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
    return toml_path
end

function _ct_live(wrapper, lang, names)
    r = run_isolated(`$CONFLICT_TRAP_PROBE $wrapper $lang`)
    v = probe_verdicts(r.output)
    @test r.ok
    @test haskey(v, "PROBE_DONE")
    for n in names
        @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
    end
    @test count(n -> haskey(v, n), names) == length(names)
end

@testset "A name several types share never crosses by value: C, live" begin
    toml_path = _ct_toml(CONFLICT_TRAP_C_DIR, "conflict_trap_c", "c", ["one.c", "two.c"])
    RepliBuild.build(toml_path)
    RepliBuild.wrap(toml_path)
    jdir = joinpath(CONFLICT_TRAP_C_DIR, "julia")
    sd = RepliBuild.Compiler._read_json(joinpath(jdir, "compilation_metadata.json"))["struct_definitions"]
    wrapper = joinpath(jdir, "ConflictTrapC.jl")
    @test isfile(wrapper)
    code = read(wrapper, String)

    @testset "the metadata keeps every definition's size" begin
        @test get(sd["state"], "conflict", false) == true
        @test sd["state"]["conflict_sizes"] == [8]       # one size, two layouts: still a trap
        @test !haskey(sd["pt"], "conflict")
    end

    @testset "a trap for each function that passes `state`" begin
        for f in ("st_one", "st_two")
            b = _ct_body(code, f)
            @test (f, occursin("Base.error(\"ABI Safety Trap: cannot call '$f'", b)) == (f, true)
            @test (f, occursin("ccall", b)) == (f, false)
        end
        @test occursin("distinct records in this library are named `state` (all 8 bytes)", _ct_body(code, "st_one"))
        @test occursin("ccall((:pt_sum, LIBRARY_PATH)", _ct_body(code, "pt_sum"))
    end

    @testset "every function, live" begin
        _ct_live(wrapper, "c", ["pt_sum", "st_one", "st_two"])
    end
end

if requires("conflicted names, C++", :libJLCS, :clangxx)
@testset "A name several types share never crosses by value: C++, live" begin
    toml_path = _ct_toml(CONFLICT_TRAP_DIR, "conflict_trap", "cpp", ["alpha.cpp", "beta.cpp"];
                         extra = "include_dirs = [\"$(escape_string(joinpath(CONFLICT_TRAP_DIR, "include")))\"]")
    RepliBuild.build(toml_path)
    RepliBuild.wrap(toml_path)
    jdir = joinpath(CONFLICT_TRAP_DIR, "julia")
    sd = RepliBuild.Compiler._read_json(joinpath(jdir, "compilation_metadata.json"))["struct_definitions"]
    wrapper = joinpath(jdir, "ConflictTrap.jl")
    @test isfile(wrapper)
    code = read(wrapper, String)

    @testset "the metadata keeps every definition's size" begin
        @test sd["impl"]["conflict"] == true
        @test sd["impl"]["conflict_sizes"] == [8, 16]
        @test sd["__enum__Mode"]["conflict_sizes"] == [4]
        @test sd["__enum__Kind"]["conflict_sizes"] == [1, 4]
        @test !haskey(sd["Duo"], "conflict")
        @test !haskey(sd["Holder"], "conflict")
    end

    trapped = ["ct_impl_sum", "ct_holder_sum", "ct_alpha_kind", "ct_vocab_score", "ct_kind"]

    @testset "traps have no tier and no thunk; the rest keep theirs" begin
        tiers = Dict(String(m.captures[1]) => Symbol(m.captures[2])
                     for m in eachmatch(r"\n\s+:(\w+) => :(tier\d)", code))
        want = Dict("ct_duo_sum" => :tier3, "ct_mode_value" => :tier3, "ct_impl_ptr_sum" => :tier3,
                    "ct_duo_twice" => :tier2, "ct_beta_mode" => :tier2)
        checked = 0
        for (f, t) in sort(collect(want); by = first)
            @test (f, get(tiers, f, :missing)) == (f, t)
            checked += 1
        end
        @test checked == length(want)
        manifest = read(joinpath(jdir, "thunk_manifest.json"), String)
        for f in trapped
            @test (f, get(tiers, f, :none)) == (f, :none)
            @test (f, occursin("Base.error(\"ABI Safety Trap: cannot call '$f'", _ct_body(code, f))) == (f, true)
            @test (f, occursin(f, manifest)) == (f, false)
        end
        @test occursin("ct_duo_twice", manifest)
        @test occursin("which holds `impl` by value", _ct_body(code, "ct_holder_sum"))
        @test occursin("enums of different widths in this library are named `Kind` (1 and 4 bytes)",
                       _ct_body(code, "ct_kind"))
    end

    @testset "every function, live" begin
        _ct_live(wrapper, "cpp", ["ct_duo_sum", "ct_mode_value", "ct_impl_ptr_sum", "ct_duo_twice",
                                  "ct_beta_mode", trapped...])
    end
end
end # requires
