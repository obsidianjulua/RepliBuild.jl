# =============================================================================
# Forced inline member functions, live (2026-11-15)
#
# test_inline_methods.jl (unit) checks the parser's rows, the plan, the unit's
# text and the two readings of a compile. This builds test/inline_method_test
# (a class whose everyday API is defined in its body and called by nothing in
# the library, as Bullet's btRigidBody is), wraps it, and calls each forced
# member from a child against the library's own readers.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const INLINE_METHOD_DIR = joinpath(@__DIR__, "inline_method_test")

if requires("forced inline member functions, live", :libJLCS, :clangxx)
@testset "Forced inline member functions, live" begin
    toml_path = joinpath(INLINE_METHOD_DIR, "replibuild.toml")
    src = joinpath(INLINE_METHOD_DIR, "src", "inline_method.cpp")
    inc = joinpath(INLINE_METHOD_DIR, "include")

    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "inlinemethod"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(INLINE_METHOD_DIR))"

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
    wrapper = joinpath(INLINE_METHOD_DIR, "julia", "Inlinemethod.jl")
    @test isfile(wrapper)
    code = read(wrapper, String)
    md = RepliBuild._read_json(joinpath(INLINE_METHOD_DIR, "julia", "compilation_metadata.json"))

    # The members the unit defines. `mass` is called by the library, but at
    # -O2 that call is inlined and no definition of it was left either.
    forced = ["_ZN2im4Body11defaultMassEv",
              "_ZN2im4Body11setVelocityERKNS_4Vec3E", "_ZN2im4Body12applyImpulseERKNS_4Vec3E",
              "_ZN2im4Body3idsEv", "_ZN2im4Body5scaleEff", "_ZN2im4Body5scaleEi", "_ZN2im4Body8positionEv",
              "_ZN2im4Body9setActiveEb", "_ZN2im4BodypLERKNS_4Vec3E", "_ZNK2im4Body11getVelocityEv",
              "_ZNK2im4Body4massEv", "_ZNK2im4Body7invMassEv", "_ZNK2im4Body8getScaleEv",
              "_ZNK2im4Body8isActiveEv", "_ZNK2im4Body8positionEv"]

    @testset "the build ran a second pass for the unit" begin
        @test occursin("methods: $(length(forced)) inline member function(s) of 1 class(es) declared in headers and defined nowhere", build_log)
        @test !occursin("ctors:", build_log)                    # no constructor to force here
        @test count("link: libinlinemethod", build_log) == 2
        unit = joinpath(INLINE_METHOD_DIR, ".replibuild_cache", "replibuild_ctors.cpp")
        @test isfile(unit)
        text = read(unit, String)
        @test occursin("#include \"$(joinpath(inc, "inline_method.h"))\"", text)
        @test occursin("replibuild_force::args<im::Vec3 const&>::m(&im::Body::setVelocity);", text)
        @test occursin("replibuild_force::args<>::mc(&im::Body::getVelocity);", text)
        @test count("RB_FORCE_MEMBER replibuild_member_", text) == length(forced)
        # The member with no body was in the plan and is not in the unit: its
        # address would be an undefined symbol in the library.
        @test !occursin("declaredOnly", text)
        @test !occursin("outOfLine", text) && !occursin("privateInline", text) && !occursin("protectedInline", text)
        @test !occursin("&im::Body::kind", text) && !occursin("&im::Body::as", text)
        # The library now defines them, and nothing of the unit's own is API.
        mangled = Set(String(f["mangled"]) for f in md["functions"])
        missing_syms = [s for s in forced if !(s in mangled)]
        @test missing_syms == String[]
        @test !("_ZN2im4Body12declaredOnlyEi" in mangled)
        @test !any(f -> occursin("replibuild_", String(get(f, "demangled", ""))), md["functions"])
        @test !any(g -> occursin("replibuild_member", String(g)), keys(get(md, "globals", Dict())))
    end

    @testset "the metadata names what was forced and what was skipped, with reasons" begin
        plan = md["inline_methods"]
        @test sort([String(f["symbol"]) for f in plan["forced"] if f["emitted"] == true]) == sort(forced)
        @test all(f -> f["emitted"] == true, plan["forced"])
        skipped = Dict(String(s["symbol"]) => String(s["reason"]) for s in plan["skipped"])
        tmpl = RepliBuild.Compiler._TEMPLATE_MEMBER_SKIP
        @test skipped == Dict("_ZN2im4Body12declaredOnlyEi" => "declared in its header with no body there",
                              # A class template's members are `[types] templates`' to ask for.
                              "_ZN2im3ArrIiE2atEi" => tmpl, "_ZNK2im3ArrIiE4sizeEv" => tmpl)
        # The parser's rows behind the plan.
        rows = Dict(String(r["symbol"]) => r for r in md["record_abi"]["Body"]["methods"])
        @test rows["_ZN2im4Body13privateInlineEv"]["accessibility"] == "private"
        @test rows["_ZN2im4Body15protectedInlineEv"]["accessibility"] == "protected"
        @test rows["_ZNK2im4Body4kindEv"]["virtual"] == true
        @test rows["_ZN2im4Body11setVelocityERKNS_4Vec3E"]["accessibility"] == "public"
        # No constructor was forced: the constructor plan is empty, not absent.
        @test isempty(md["inline_constructors"]["forced"])
    end

    @testset "the wrapper binds a forced member like any other" begin
        @test occursin(r"\nfunction im_Body_setVelocity\(this::Any, v::", code)
        @test occursin(r"\nfunction im_Body_getVelocity\(this::Any\)", code)
        @test occursin(r"\nfunction im_Body_applyImpulse\(this::Any, j::", code)
        @test occursin(r"\nfunction im_Body_defaultMass\(\)", code)
        @test !occursin("function im_Arr_int_size(", code)
        @test !occursin("function im_Body_declaredOnly(", code)
        @test !occursin("function im_Body_privateInline(", code)
    end

    @testset "called through the wrapper, from a child" begin
        r = run_isolated(`$(joinpath(INLINE_METHOD_DIR, "probe_inline_method.jl")) $wrapper`)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        names = ["set_then_get_round_trips", "impulse_over_mass", "bool_setter_and_getter",
                 "overloads_by_parameters", "static_and_operator",
                 "defined_untouched_and_refused_unbound"]
        for n in names
            @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
        end
        @test count(n -> haskey(v, n), names) == length(names)
    end
end
end # requires
