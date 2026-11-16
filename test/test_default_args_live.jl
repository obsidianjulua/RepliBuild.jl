# =============================================================================
# C++ default arguments, live (2026-11-15)
#
# test_default_args.jl (unit) checks the candidate rows, the unit's lines and
# the identity rewrite. This builds test/default_arg_test (a constructor and
# member functions with trailing defaults, as Bullet's btDefaultMotionState
# and btHingeConstraint::setLimit have), wraps it, and makes the short calls
# from a child, reading the omitted arguments' values back through the
# library. Needs clang++ and libJLCS.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const DEFAULT_ARG_DIR = joinpath(@__DIR__, "default_arg_test")

if requires("C++ default arguments, live", :libJLCS, :clangxx)
@testset "C++ default arguments, live" begin
    toml_path = joinpath(DEFAULT_ARG_DIR, "replibuild.toml")
    src = joinpath(DEFAULT_ARG_DIR, "src", "default_arg.cpp")
    inc = joinpath(DEFAULT_ARG_DIR, "include")

    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "defaultarg"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(DEFAULT_ARG_DIR))"

    [compile]
    flags = ["-std=c++17", "-fPIC", "-O2", "-fstandalone-debug"]
    source_files = ["$(escape_string(src))"]
    include_dirs = ["$(escape_string(inc))"]

    [link]
    enable_lto = false
    optimization_level = "2"

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
    build_log = mktemp() do path, io
        redirect_stdout(io) do
            RepliBuild.build(toml_path)
        end
        close(io)
        read(path, String)
    end
    RepliBuild.wrap(toml_path)
    wrapper = joinpath(DEFAULT_ARG_DIR, "julia", "Defaultarg.jl")
    @test isfile(wrapper)
    code = read(wrapper, String)
    md = RepliBuild._read_json(joinpath(DEFAULT_ARG_DIR, "julia", "compilation_metadata.json"))

    # (member, parameters kept): the short calls the header has defaults for.
    want = sort([("da::Hinge::Hinge(da::Vec const&)", 1), ("da::State::State(da::Vec const&)", 1),
                 ("da::Hinge::setLimit(float, float)", 3), ("da::Hinge::setLimit(float, float, float)", 2),
                 ("da::Hinge::setLimit(float, float, float, float)", 1),
                 ("da::Hinge::step(float)", 2), ("da::Hinge::step(float, int)", 1),
                 ("da::Hinge::scaled(float) const", 1)])

    @testset "the unit keeps exactly the calls the header has defaults for" begin
        @test occursin(r"defaults: 9 call\(s\) of 6 function\(s\) with default arguments left out", build_log)
        unit = read(joinpath(DEFAULT_ARG_DIR, ".replibuild_cache", "replibuild_ctors.cpp"), String)
        # Nine rows compiled: the eight below and the static member's, which
        # is a valid C++ call and is refused later, in the metadata.
        @test count("\nnamespace replibuild_default { ", unit) == 9
        @test occursin("self->setLimit(static_cast<replibuild_force::id<float>&&>(a0), static_cast<replibuild_force::id<float>&&>(a1))", unit)
        # No default, no row: setLimit(lo), none(a), prot(a), count(a).
        @test !occursin("self->none(", unit) && !occursin("self->prot(", unit) && !occursin("self->count(", unit)
        @test !occursin(r"self->setLimit\(static_cast<replibuild_force::id<float>&&>\(a0\)\)", unit)
    end

    @testset "the metadata: each function carries its member's identity" begin
        bound = md["default_arguments"]["bound"]
        @test isempty(md["default_arguments"]["candidates"])
        emitted = sort([(String(r["demangled"]), Int(r["omitted"])) for r in bound if r["emitted"] == true])
        @test emitted == want
        # The static member's short call compiled and is not a function of the wrapper.
        refused = [String(r["demangled"]) for r in bound if r["emitted"] != true]
        @test refused == ["da::Hinge::make(int)"]
        fns = [f for f in md["functions"] if haskey(f, "default_arguments")]
        @test length(fns) == length(want)
        for f in fns
            @test startswith(String(f["mangled"]), "_ZN18replibuild_default")
            @test f["parameters"][1]["name"] == "this"
            @test f["has_object_pointer"] == true
            @test String(f["class"]) in ("da::Hinge", "da::State")
        end
        sl = only(f for f in fns if f["demangled"] == "da::Hinge::setLimit(float, float)")
        @test [p["name"] for p in sl["parameters"]] == ["this", "lo", "hi"]
        @test sl["default_arguments"]["of"] == "_ZN2da5Hinge8setLimitEfffff"
        # Nothing of the unit's own namespace is a function of the wrapper by that name.
        @test !any(f -> occursin("replibuild_default", String(get(f, "demangled", ""))), md["functions"])
    end

    @testset "the wrapper binds the short calls under the member's name" begin
        @test occursin(r"\nfunction da_Hinge_setLimit\(this::Any, lo::Cfloat, hi::Cfloat\)\n", code)
        @test occursin(r"\nfunction da_Hinge_setLimit\(this::Any, lo::Cfloat, hi::Cfloat, soft::Cfloat, bias::Cfloat, relax::Cfloat\)\n", code)
        @test occursin(r"\nfunction da_Hinge_Hinge\(this::Any, a::[^,)]+\)\n", code)
        @test occursin(r"\nfunction da_State_State\(this::Any, start::[^,)]+\)\n", code)
        @test !occursin("replibuild_default_", code)             # no function is named after the helper
        @test !occursin(r"\nfunction da_Hinge_make\(a::[^,)]+\)\n", code)
    end

    @testset "the short calls, from a child" begin
        r = run_isolated(`$(joinpath(DEFAULT_ARG_DIR, "probe_default_arg.jl")) $wrapper`)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        names = ["constructor_default", "inline_constructor_default", "method_trailing_defaults",
                 "overload_beside_defaults", "by_value_return", "no_default_no_binding"]
        for n in names
            @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
        end
        @test count(n -> haskey(v, n), names) == length(names)
    end
end
end # requires
