# =============================================================================
# The ownership unit, live (2026-11-27)
#
# A C++ object that C++ will not copy bytewise (a `std::vector`, a
# `std::map`, a `std::unique_ptr`, a class with a destructor) has to be ended
# once and copied only by its copy constructor. Whether a type needs that is
# the compiler's answer, so the build asks: a generated unit names each such
# type the library returns or takes by value, and the compiler states its
# size, whether it must be ended and whether it can be copied or moved, and
# compiles the functions that do each (`OwnershipUnit`, the build's step 4d).
#
# ownership_test is built here and the answers are read from the metadata.
# The functions are then called on objects the library makes, in a child: a
# wrong one is a double free or a leak, which the fixture's own count of live
# objects shows.
#
# A second build has an umbrella header that DEFINES a function: the library's
# own unit already includes it, and the generated unit including it again
# would be "multiple definition of" at the link. It is left out of the unit,
# and the answers are the same.
#
# Needs clang++.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())

const OWN_DIR = joinpath(@__DIR__, "ownership_test")
const _OWN_FG = RepliBuild.JLCSIRGenerator.FunctionGen

_own_toml(dir, sources; wrap = "") = """
[project]
name = "ownership"
root = "$(escape_string(dir))"

[compile]
flags = ["-std=c++17", "-fPIC", "-fstandalone-debug"]
source_files = [$(join(["\"$(escape_string(s))\"" for s in sources], ", "))]

[link]
enable_lto = false
optimization_level = "0"

[binary]
type = "shared"

[wrap]
language = "cpp"
$wrap
[types]
strictness = "warn"
allow_unknown_structs = true

[cache]
enabled = false
"""

# The answer for the one type whose key starts with `prefix`.
function _own_type(types, prefix)
    hits = sort!([k for k in keys(types) if startswith(k, prefix)])
    return length(hits) == 1 ? (hits[1], types[hits[1]]) : (nothing, nothing)
end

# (prefix of the key, must be ended, can be copied, can be moved)
const _OWN_EXPECTED = (("Tracked", true, true, true),
                       ("Copied", false, true, true),
                       ("vector<int", true, true, true),
                       ("map<int, int", true, true, true),
                       ("list<int", true, true, true),
                       ("unique_ptr<own::Tracked", true, false, true),
                       ("Token", true, false, true))

if requires("The ownership unit, live", :clangxx)
@testset "The ownership unit, live" begin
    toml_path = joinpath(OWN_DIR, "replibuild.toml")
    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, _own_toml(OWN_DIR, [joinpath(OWN_DIR, "src", "own.cpp")]))
    lib = RepliBuild.build(toml_path)
    @test isfile(lib)
    md = RepliBuild._read_json(joinpath(dirname(lib), "compilation_metadata.json"))
    own = md["ownership"]
    types = own["types"]

    @testset "the compiler's answer for each type" begin
        @test isempty(own["skipped"])
        @test length(types) == length(_OWN_EXPECTED)
        seen = 0
        for (prefix, ends, copy, move) in _OWN_EXPECTED
            (key, t) = _own_type(types, prefix)
            @test (prefix, key === nothing) == (prefix, false)
            key === nothing && continue
            @test (prefix, t["ends"], t["copy"], t["move"]) == (prefix, ends, copy, move)
            @test (prefix, t["bitwise"]) == (prefix, false)
            # The compiler's size is the one DWARF gives the type.
            dwarf = something(get(md["struct_definitions"], key, nothing), get(md["record_abi"], key, nothing))
            @test (prefix, t["size"]) == (prefix, _OWN_FG._parse_num(dwarf["byte_size"]))
            @test t["align"] in (4, 8)
            seen += 1
        end
        @test seen == length(_OWN_EXPECTED)
        # A pointer is all a unique_ptr with the default deleter is.
        @test _own_type(types, "unique_ptr<own::Tracked")[2]["size"] == sizeof(Ptr{Cvoid})
        # Plain data is not asked about: C++ copies it bytewise.
        @test !any(k -> startswith(k, "Plain"), keys(types))
    end

    @testset "every type the functions cross has its answer" begin
        # Filed under the key a generator asks with, for returns and parameters.
        asked = 0
        for f in md["functions"]
            rkey = _OWN_FG.owned_key(String(get(f["return_type"], "c_type", "void")),
                                     md["struct_definitions"], md["record_abi"]; ret = true)
            rkey === nothing || (asked += 1; @test (f["mangled"], haskey(types, rkey)) == (f["mangled"], true))
            for p in get(f, "parameters", Any[])
                pkey = _OWN_FG.owned_key(String(get(p, "c_type", "")), md["struct_definitions"], md["record_abi"])
                pkey === nothing || (asked += 1; @test (f["mangled"], haskey(types, pkey)) == (f["mangled"], true))
            end
        end
        # 9 by-value returns (points, index, chain, boxed, tracked, copied,
        # token, broken, iota) and 6 by-value parameters (take, keep, adopt,
        # see, redeem, fail).
        @test asked == 15
    end

    @testset "nothing of the unit is the library's API" begin
        unit = joinpath(OWN_DIR, ".replibuild_cache", "replibuild_own.cpp")
        @test isfile(unit)
        @test count(l -> startswith(l, "RB_OWN_ROW("), eachline(unit)) == length(types)
        @test !any(f -> occursin("replibuild", String(f["mangled"])) || occursin("replibuild", String(get(f, "demangled", ""))),
                   md["functions"])
        @test !any(k -> occursin("own<", k) || occursin("own_args", k) || occursin("own_facts", k),
                   keys(md["struct_definitions"]))
        @test !any(k -> occursin("replibuild", String(k)), keys(md["globals"]))
        # ...and every function a wrapper will call is one the loader binds.
        runtime = RepliBuild.Compiler._runtime_defined_names(lib)
        @test runtime !== nothing
        for (_, t) in types, k in ("end_symbol", "copy_symbol", "move_symbol")
            @test (t[k], t[k] in runtime) == (t[k], true)
        end
        # A wrapper is still written for the library.
        @test isfile(RepliBuild.wrap(toml_path))
    end

    @testset "ending, copying and moving, on the library's own objects" begin
        names = String[]
        for prefix in ("Tracked", "vector<int", "unique_ptr<own::Tracked")
            t = _own_type(types, prefix)[2]
            append!(names, [t["end_symbol"], t["copy_symbol"], t["move_symbol"]])
        end
        r = run_isolated(`$(joinpath(OWN_DIR, "probe_helpers.jl")) $lib $names`; timeout = 300)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        cases = ("made_is_one_alive", "copy_makes_a_second", "copy_has_the_value", "copy_leaves_the_original",
                 "move_takes_the_value", "moved_from_is_empty_and_alive", "ending_two_leaves_one",
                 "ending_all_leaves_none", "vector_copy_reads_the_same", "vector_copy_outlives_the_original",
                 "vector_move_takes_the_elements", "unique_ptr_copy_is_refused", "unique_ptr_moves",
                 "nothing_left_alive")
        for name in cases
            @test (name, startswith(get(v, name, "no PROBE line"), "PASS")) == (name, true)
        end
        @test count(k -> k != "PROBE_DONE", keys(v)) == length(cases)
    end

    @testset "an umbrella header that defines a function is left out of the unit" begin
        mktempdir() do dir
            mkpath(joinpath(dir, "src"))
            for f in ("own.hpp", "own.cpp")
                cp(joinpath(OWN_DIR, "src", f), joinpath(dir, "src", f))
            end
            # An implementation header: one translation unit of the library
            # includes it, and it defines a function with external linkage.
            write(joinpath(dir, "src", "impl.hpp"), "#pragma once\nint own_impl_marker() { return 7; }\n")
            write(joinpath(dir, "src", "impl.cpp"), "#include \"impl.hpp\"\n")
            toml = joinpath(dir, "replibuild.toml")
            write(toml, _own_toml(dir, [joinpath(dir, "src", "own.cpp"), joinpath(dir, "src", "impl.cpp")];
                                  wrap = "shim_headers = [\"$(escape_string(joinpath(dir, "src", "impl.hpp")))\"]\n"))
            lib2 = RepliBuild.build(toml)
            @test isfile(lib2)
            own2 = RepliBuild._read_json(joinpath(dirname(lib2), "compilation_metadata.json"))["ownership"]
            # The same answers, and the library has the unit's functions: the
            # link did not fail on a second definition.
            @test isempty(own2["skipped"])
            @test sort!(collect(keys(own2["types"]))) == sort!(collect(keys(types)))
            unit = read(joinpath(dir, ".replibuild_cache", "replibuild_own.cpp"), String)
            @test !occursin("impl.hpp", unit)
            @test occursin("own.hpp", unit)
            runtime = RepliBuild.Compiler._runtime_defined_names(lib2)
            @test all(t -> t["end_symbol"] in runtime, values(own2["types"]))
            @test "own_impl_marker" in runtime || "_Z15own_impl_markerv" in runtime
        end
    end
end
end # requires
