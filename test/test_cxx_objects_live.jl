# =============================================================================
# A C++ object Julia holds where C++ built it, live (2026-11-27)
#
# The rules of test_cxx_objects.jl, on a library: ownership_test is built and
# wrapped, and its wrapper is called from Julia in a child.
#
#   * a `std::vector`, a `std::map`, a `std::list`, a `std::unique_ptr` and a
#     class with a destructor come back held where C++ built them, and the
#     library reads each right (a map's last key, a walk to a list's end: both
#     were wrong while the bytes were moved);
#   * each is ended once: at `release!`, or when Julia collects it. The
#     fixture counts its live objects, so a leak and a double ending both show;
#   * a by-value parameter gets a copy and the caller keeps its object; what
#     C++ cannot copy is moved, `move(x)` asks for that, and a moved object is
#     an error to use;
#   * a call that throws ends what it was given and leaves nothing half-built.
#
# Needs clang++ and libJLCS.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())

const CXO_LIVE_DIR = joinpath(@__DIR__, "ownership_test")

if requires("A C++ object Julia holds where C++ built it, live", :libJLCS, :clangxx)
@testset "A C++ object Julia holds where C++ built it, live" begin
    toml_path = joinpath(CXO_LIVE_DIR, "replibuild.toml")
    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "ownership"
    root = "$(escape_string(CXO_LIVE_DIR))"

    [compile]
    flags = ["-std=c++17", "-fPIC", "-fstandalone-debug"]
    source_files = ["$(escape_string(joinpath(CXO_LIVE_DIR, "src", "own.cpp")))"]

    [link]
    enable_lto = false
    optimization_level = "0"

    [binary]
    type = "shared"

    [wrap]
    language = "cpp"

    [types]
    strictness = "warn"
    allow_unknown_structs = true

    [cache]
    enabled = false
    """)
    RepliBuild.build(toml_path)
    wrapper = RepliBuild.wrap(toml_path)
    @test isfile(wrapper)

    code = read(wrapper, String)
    body(f) = [String(m.match) for m in eachmatch(Regex("\\nfunction $f\\(.*?\\nend", "s"), code)]

    @testset "the wrapper" begin
        # Held in place: the storage, born once C++ has built into it.
        held = ("own_iota" => "NTuple{24, UInt8}", "own_Shape_points" => "NTuple{24, UInt8}",
                "own_Shape_index" => "NTuple{48, UInt8}", "own_Shape_chain_abi_cxx11" => "NTuple{24, UInt8}",
                "own_Shape_boxed" => "NTuple{8, UInt8}", "own_Shape_tracked" => "Tracked",
                "own_Shape_token" => "Token", "own_Shape_broken" => "Tracked")
        for (f, T) in held
            b = body(f)
            @test (f, length(b)) == (f, 1)
            isempty(b) && continue
            @test (f, occursin(Regex("_rb_ret = RepliBuild\\.CxxObjects\\.slot\\($(replace(T, "{" => "\\{", "}" => "\\}")), _RB_OWN_\\d+, LIBRARY_PATH\\)"), b[1])) == (f, true)
            @test (f, occursin("return RepliBuild.CxxObjects.born!(_rb_ret)", b[1])) == (f, true)
            @test (f, occursin("slot_value", b[1])) == (f, false)
        end
        # Nothing to end: the value, as before.
        for f in ("own_Shape_copied", "own_Shape_plain")
            b = only(body(f))
            @test (f, occursin("CxxObjects", b)) == (f, false)
        end
        # By value: a temporary the wrapper makes (a container), or the
        # caller's value or object lent to the thunk (a class with a layout).
        for f in ("own_Shape_take", "own_Shape_keep", "own_Shape_adopt", "own_Shape_fail")
            b = only(body(f))
            @test (f, occursin(r"_rb_v2 = RepliBuild\.CxxObjects\.byvalue\(NTuple\{\d+, UInt8\}, _RB_OWN_\d+, LIBRARY_PATH, \w+\)", b)) == (f, true)
            @test (f, occursin("finally RepliBuild.CxxObjects.settle!(_rb_v2) end", b)) == (f, true)
            @test (f, occursin("ByValue", b)) == (f, false)
        end
        for (f, T) in ("own_Shape_see" => "Tracked", "own_Shape_redeem" => "Token")
            b = only(body(f))
            @test (f, occursin("::Union{$T, RepliBuild.CxxObjects.Owned{$T}, RepliBuild.CxxObjects.Moved{$T}})", b)) == (f, true)
            @test (f, occursin("RepliBuild.CxxObjects.lend($T, _RB_OWN_", b)) == (f, true)
            @test (f, occursin("finally RepliBuild.CxxObjects.settle!(_rb_v2) end", b)) == (f, true)
        end
        # A reference to a class a return hands back as the object takes the object.
        @test length(body("own_peek")) == 2
        @test any(b -> occursin("RepliBuild.CxxObjects.Owned{Tracked}", b), body("own_peek"))
        @test occursin("RepliBuild.CxxObjects.Owned{NTuple{8, UInt8}}", only(body("own_boxed_value")))
        # One constant per kind a call site names, each defined once.
        named = sort!(unique(String[m.match for m in eachmatch(r"\b_RB_OWN_\d+\b", join(vcat((body(String(f)) for f in
                    ("own_iota", "own_Shape_points", "own_Shape_index", "own_Shape_chain_abi_cxx11", "own_Shape_boxed",
                     "own_Shape_tracked", "own_Shape_token", "own_Shape_take", "own_Shape_see", "own_Shape_redeem"))...)))]))
        defined = String[m.captures[1] for m in eachmatch(r"^const (_RB_OWN_\d+) = RepliBuild\.CxxObjects\.Kind\("m, code)]
        @test length(named) == 6                       # vector, map, list, unique_ptr, Tracked, Token
        @test sort(defined) == named && allunique(defined)
    end

    @testset "called from Julia" begin
        r = run_isolated(`$(joinpath(CXO_LIVE_DIR, "probe_owned.jl")) $wrapper`; timeout = 600)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        names = ["returns_are_held_or_plain", "held_objects_read_right", "a_reference_is_the_callers_object",
                 "overloads_take_their_own_object", "released_once", "collected_once",
                 "a_return_that_throws_leaves_nothing", "by_value_is_a_copy", "uncopyable_is_moved_and_spent",
                 "move_gives_the_object", "a_throw_settles_the_argument", "another_type_is_refused",
                 "nothing_left_alive"]
        for n in names
            @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
        end
        @test count(n -> haskey(v, n), names) == length(names)
        @test count(k -> k != "PROBE_DONE", keys(v)) == length(names)
    end
end
end # requires
