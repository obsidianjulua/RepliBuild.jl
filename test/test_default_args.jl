# =============================================================================
# C++ default arguments (2026-11-15)
#
# A default argument exists only in the header: the mangled name spells every
# parameter and clang writes no DW_AT_default_value. So
# btDefaultMotionState(start) had to be called with its second transform
# spelled out, and btHingeConstraint::setLimit(lo, hi) with all five
# arguments. The forced unit now asks the compiler: one function in namespace
# replibuild_default per call that leaves trailing arguments out; a call the
# header has no defaults for is a compile error on its row and is dropped
# (Compiler._default_argument_plan, generate_inline_definitions).
#
# Toolchain-free: the candidate rows over hand-made metadata, each row's line
# in the unit, the row tracing, and the identity a surviving function takes
# in the metadata. test_default_args_live.jl builds test/default_arg_test.
# =============================================================================

using Test
using RepliBuild

const _DA = RepliBuild.Compiler

@testset "C++ default arguments" begin
    hdr = ["/src/lib/include/lib.h"]
    def(; kw...) = merge(Dict{String,Any}("kind" => "struct", "decl_files" => hdr),
                         Dict{String,Any}(String(k) => v for (k, v) in kw))
    row(sym; acc = "public", artificial = false, deleted = false, virt = false) =
        Dict{String,Any}("symbol" => sym, "accessibility" => acc, "artificial" => artificial,
                         "deleted" => deleted, "virtual" => virt)
    sd = Dict{String,Any}("Hinge" => def(), "Arr<int>" => def(), "Twice" => def(conflict = true),
                          "NoHdr" => def(decl_files = ["/src/lib/src/x.cpp"]), "pair<int, int>" => def(toolchain = true))
    ra = Dict{String,Any}(
        "Hinge" => Dict{String,Any}("methods" => [
            row("_ZN2da5HingeC4ERKNS_3VecES3_"),          # Hinge(Vec const&, Vec const&)
            row("_ZN2da5Hinge8setLimitEfffff"),           # setLimit(float ×5)
            row("_ZNK2da5Hinge5countEii"),                # count(int, int) const
            row("_ZNK2da5Hinge5countEi"),                 # count(int) const: the short call's owner
            row("_ZNK2da5Hinge2loEv"),                    # no parameters
            row("_ZN2da5Hinge4protEii"; acc = "protected"),
            row("_ZN2da5HingeD4Ev"),                      # a destructor
            row("_ZN2da5HingepLERKNS_3VecE"),             # an operator
            row("_ZN2da5HingeC4ERKS0_"; artificial = true),
            row("_ZN2da5Hinge4goneEi"; deleted = true),
            row("_ZN2da5Hinge6hiddenEi")]),               # dropped by pattern
        "Arr<int>" => Dict{String,Any}("methods" => [row("_ZN2da3ArrIiE6resizeEii")]),
        "Twice" => Dict{String,Any}("methods" => [row("_ZN2da5Twice1fEii")]),
        "NoHdr" => Dict{String,Any}("methods" => [row("_ZN2da5NoHdr1fEii")]),
        "pair<int, int>" => Dict{String,Any}("methods" => [row("_ZNSt4pairIiiE4swapERS0_")]))
    symbols = [Dict{String,Any}("mangled" => m) for m in ("_ZN2da5HingeC2ERKNS_3VecES3_", "_ZNK2da5Hinge5countEii")]
    plan = _DA._default_argument_plan(sd, ra, symbols, ["*hidden*"])
    rows = plan["candidates"]

    @testset "one candidate per shorter call, longest first" begin
        got = [(r["symbol"], r["name"], r["parameters"], r["omitted"]) for r in rows]
        @test got == [
            ("_ZN2da5HingeC4ERKNS_3VecES3_", "Hinge", ["da::Vec const&"], 1),
            ("_ZN2da5HingeC4ERKNS_3VecES3_", "Hinge", String[], 2),
            ("_ZN2da5Hinge8setLimitEfffff", "setLimit", fill("float", 4), 1),
            ("_ZN2da5Hinge8setLimitEfffff", "setLimit", fill("float", 3), 2),
            ("_ZN2da5Hinge8setLimitEfffff", "setLimit", fill("float", 2), 3),
            ("_ZN2da5Hinge8setLimitEfffff", "setLimit", fill("float", 1), 4),
            ("_ZN2da5Hinge8setLimitEfffff", "setLimit", String[], 5),
            # count(int) is a member of its own: only the empty call is a candidate.
            ("_ZNK2da5Hinge5countEii", "count", String[], 2),
        ]
        @test [r["id"] for r in rows] == collect(0:length(rows)-1)
        by(sym, n) = only(r for r in rows if r["symbol"] == sym && length(r["parameters"]) == n)
        c = by("_ZN2da5HingeC4ERKNS_3VecES3_", 1)
        @test (c["ctor"], c["const"], c["type"], c["class"], c["headers"]) == (true, false, "da::Hinge", "Hinge", hdr)
        @test c["defined"] == true                      # the symbol table has its C2
        @test c["demangled"] == "da::Hinge::Hinge(da::Vec const&)"
        s = by("_ZN2da5Hinge8setLimitEfffff", 2)
        @test (s["ctor"], s["defined"]) == (false, false)      # inline: the unit must define it
        @test s["demangled"] == "da::Hinge::setLimit(float, float)"
        @test by("_ZNK2da5Hinge5countEii", 0)["demangled"] == "da::Hinge::count() const"
        # Nothing for: a protected, implicit or deleted member, a destructor,
        # an operator, a member with no parameters, an excluded symbol, a
        # class template's specialization, a conflicted or headerless record,
        # a toolchain record.
        named = Set(r["symbol"] for r in rows)
        @test named == Set(["_ZN2da5HingeC4ERKNS_3VecES3_", "_ZN2da5Hinge8setLimitEfffff", "_ZNK2da5Hinge5countEii"])
        # Metadata without `methods` rows (an older build), and no demangler: nothing.
        @test isempty(_DA._default_argument_plan(sd, Dict{String,Any}("Hinge" => Dict{String,Any}()), symbols, String[])["candidates"])
        @test isempty(_DA._default_argument_plan(sd, ra, symbols, String[]; demangle = _ -> nothing)["candidates"])
    end

    @testset "a row's line: the short call, on one line" begin
        by(sym, n) = only(r for r in rows if r["symbol"] == sym && length(r["parameters"]) == n)
        m = _DA._default_argument_line(by("_ZN2da5Hinge8setLimitEfffff", 2))
        id = by("_ZN2da5Hinge8setLimitEfffff", 2)["id"]
        @test m == "namespace replibuild_default { auto d$(id)(da::Hinge* self, replibuild_force::id<float> a0, " *
                   "replibuild_force::id<float> a1) -> decltype(self->setLimit(static_cast<replibuild_force::id<float>&&>(a0), " *
                   "static_cast<replibuild_force::id<float>&&>(a1))) { return self->setLimit(" *
                   "static_cast<replibuild_force::id<float>&&>(a0), static_cast<replibuild_force::id<float>&&>(a1)); } }" *
                   "   // _ZN2da5Hinge8setLimitEfffff less 3"
        c = _DA._default_argument_line(by("_ZN2da5HingeC4ERKNS_3VecES3_", 1))
        @test occursin("void d$(by("_ZN2da5HingeC4ERKNS_3VecES3_", 1)["id"])(da::Hinge* self, replibuild_force::id<da::Vec const&> a0) " *
                       "{ ::new (static_cast<void*>(self)) da::Hinge(static_cast<replibuild_force::id<da::Vec const&>&&>(a0)); }", c)
        # A const member is called on a const object, so the const overload is the one taken.
        k = _DA._default_argument_line(by("_ZNK2da5Hinge5countEii", 0))
        @test occursin("(const da::Hinge* self) -> decltype(self->count())", k)
        @test !any(l -> occursin('\n', l), (m, c, k))
        # The symbol prefix a row's function has, whatever its parameters.
        @test _DA._default_shim_prefix(7) == "_ZN18replibuild_default2d7E"
        @test _DA._default_shim_prefix(123) == "_ZN18replibuild_default4d123E"
        @test !startswith("_ZN18replibuild_default3d12EPN2da5HingeE", _DA._default_shim_prefix(1))
    end

    @testset "the unit: rows after the constructors and members, each traced by its line" begin
        text = _DA._inline_ctor_unit_text(hdr, Any[], Any[], rows)
        @test occursin("namespace replibuild_force { template <class T> using id = T; }\n", text)
        @test occursin("#include <new>", text)                       # placement new in a constructor row
        @test count("\nnamespace replibuild_default { ", text) == length(rows)
        traced = _DA._unit_row_of_line(text)
        @test sort([v[2] for v in values(traced) if v[1] === :default]) == collect(1:length(rows))
        @test all(v -> v[1] === :default, values(traced))
        lines = split(text, '\n')
        for (ln, (_, k)) in traced
            @test endswith(lines[ln], "// $(rows[k]["symbol"]) less $(rows[k]["omitted"])")
        end
        # No default rows: the unit is what it was.
        @test _DA._inline_ctor_unit_text(hdr, Any[], Any[]) == _DA._inline_ctor_unit_text(hdr, Any[], Any[], Any[])
        @test !occursin("replibuild_default", _DA._inline_ctor_unit_text(hdr, Any[], Any[]))
    end

    @testset "a surviving function takes the identity of the member it calls" begin
        p(name, c) = Dict{String,Any}("name" => name, "c_type" => c, "julia_type" => "Cfloat")
        target = Dict{String,Any}("mangled" => "_ZN2da5Hinge8setLimitEfffff", "name" => "setLimit", "class" => "da::Hinge",
                                  "has_object_pointer" => true, "parameters_source" => "dwarf",
                                  "parameters" => Any[p("this", "Hinge*"), p("lo", "float"), p("hi", "float"),
                                                      p("soft", "float"), p("bias", "float"), p("relax", "float")])
        shim(sym, n) = Dict{String,Any}("mangled" => sym, "name" => "setLimit", "class" => "da::Hinge",
                                        "has_object_pointer" => false, "parameters_source" => "dwarf",
                                        "parameters" => Any[p("self", "Hinge*"); [p("a$(i-1)", "float") for i in 1:n]])
        static_target = Dict{String,Any}("mangled" => "_ZN2da5Hinge4makeEii", "name" => "make", "class" => "da::Hinge",
                                         "has_object_pointer" => false, "parameters_source" => "dwarf",
                                         "parameters" => Any[p("a", "int"), p("b", "int")])
        s2 = shim("_ZN18replibuild_default2d4EPN2da5HingeEff", 2)
        s_static = shim("_ZN18replibuild_default2d9EPN2da5HingeEi", 1)
        s_inferred = merge(shim("_ZN18replibuild_default3d11EPN2da5HingeEf", 1), Dict{String,Any}("parameters_source" => "inferred"))
        s_short = shim("_ZN18replibuild_default3d12EPN2da5HingeEf", 0)          # the object is missing
        s_orphan = shim("_ZN18replibuild_default3d13EPN2da5HingeEf", 1)         # its member has no entry
        other = Dict{String,Any}("mangled" => "_ZN2da5Hinge2loEv", "parameters" => Any[])
        fns = Any[target, static_target, s2, s_static, s_inferred, s_short, s_orphan, other]
        brow(sym, n, omitted; ctor = false) = Dict{String,Any}("symbol" => sym, "parameters" => fill("float", n),
                                                               "omitted" => omitted, "ctor" => ctor)
        bound = Dict{String,Any}(
            s2["mangled"] => brow("_ZN2da5Hinge8setLimitEfffff", 2, 3),
            s_static["mangled"] => brow("_ZN2da5Hinge4makeEii", 1, 1),
            s_inferred["mangled"] => brow("_ZN2da5Hinge8setLimitEfffff", 1, 4),
            s_short["mangled"] => brow("_ZN2da5Hinge8setLimitEfffff", 1, 4),
            s_orphan["mangled"] => brow("_ZN2da5Hinge7missingEff", 1, 1))
        _DA._apply_default_argument_identity!(fns, bound)
        kept = [f["mangled"] for f in fns]
        # Removed: a static member's, one whose parameters DWARF did not
        # supply, one whose list is not the object plus the prefix.
        @test kept == [target["mangled"], static_target["mangled"], s2["mangled"], s_orphan["mangled"], other["mangled"]]
        @test [q["name"] for q in s2["parameters"]] == ["this", "lo", "hi"]         # the member's own names
        @test s2["has_object_pointer"] == true
        @test s2["default_arguments"] == Dict{String,Any}("of" => "_ZN2da5Hinge8setLimitEfffff", "omitted" => 3)
        # A member with no entry of its own: the function still stands, with its own names.
        @test [q["name"] for q in s_orphan["parameters"]] == ["this", "a0"]
        # The member itself is untouched, and nothing happens with nothing bound.
        @test [q["name"] for q in target["parameters"]] == ["this", "lo", "hi", "soft", "bias", "relax"]
        @test !haskey(target, "default_arguments")
        before = deepcopy(fns)
        @test _DA._apply_default_argument_identity!(fns, Dict{String,Any}()) == before
    end
end
