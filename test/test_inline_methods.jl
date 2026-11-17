# =============================================================================
# Forced inline member functions (2026-11-15)
#
# A member function defined in a class body and called by no translation unit
# of the library has no definition in it: the class DIE declares it, `nm` has
# no symbol, and the wrapper had no binding (Bullet's
# btRigidBody::applyCentralImpulse, setLinearVelocity, getLinearVelocity,
# btCollisionObject::isActive, btTransform::getOrigin; Clipper2's
# Clipper64::AddSubject). The build now takes each one's address in the forced
# unit that already defines the inline constructors
# (Compiler.generate_inline_definitions), so clang emits the definitions.
#
# Toolchain-free: the parser's `methods` rows over the vendored readelf dump
# of test/inline_ctor_test, the member spelling, the declared-minus-defined
# plan over hand-made metadata, the unit's text, and the two readings the
# generator makes of a compile: which rows a diagnostic names, and which
# members the unit's IR only declares. test_inline_methods_live.jl builds
# test/inline_method_test and calls the members through the wrapper.
# =============================================================================

using Test
using RepliBuild

const _IM = RepliBuild.Compiler
const IM_DUMP = read(joinpath(@__DIR__, "fixtures", "inline_ctor.readelf.txt"), String)

@testset "Forced inline member functions" begin

    @testset "the parser records every member function a record declares" begin
        rt, sd, gv, td, ra = _IM.parse_dwarf_dump(IM_DUMP)
        rows(name) = Dict(String(r["symbol"]) => r for r in ra[name]["methods"])
        @test haskey(ra["Priv"], "methods")
        priv = rows("Priv")
        # `make` and `get` follow `public:`; the constructor before it is private.
        @test priv["_ZN2ic4Priv4makeEi"]["accessibility"] == "public"
        @test priv["_ZNK2ic4Priv3getEv"]["accessibility"] == "public"
        @test priv["_ZN2ic4PrivC4Ei"]["accessibility"] == "private"
        @test all(r -> r["virtual"] == false && r["deleted"] == false, values(priv))
        # A virtual member says so: it is emitted with the vtable, never forced.
        abs = rows("Abs")
        @test abs["_ZNK2ic3Abs3getEv"]["virtual"] == true
        @test abs["_ZN2ic3AbsC4Ei"]["virtual"] == false
        # The constructor rows are untouched by the second key.
        @test [r["symbol"] for r in ra["Priv"]["constructors"]] == ["_ZN2ic4PrivC4Ei"]
        @test !haskey(ra["Priv"]["constructors"][1], "virtual")
        # Rows are sorted by symbol: the metadata is stable.
        syms = [String(r["symbol"]) for r in ra["Priv"]["methods"]]
        @test syms == sort(syms)
    end

    @testset "how the unit names a member: the last paren group, the last depth-0 ::" begin
        sp = _IM._method_spelling
        @test sp("btRigidBody::applyCentralImpulse(btVector3 const&)") ==
              (type = "btRigidBody", name = "applyCentralImpulse", parameters = ["btVector3 const&"],
               isconst = false, structor = false)
        @test sp("im::Body::getVelocity() const") ==
              (type = "im::Body", name = "getVelocity", parameters = String[], isconst = true, structor = false)
        # A nested class and a class template's specialization keep their spelling.
        @test sp("btCollisionWorld::RayResultCallback::hasHit() const").type == "btCollisionWorld::RayResultCallback"
        s = sp("btAlignedObjectArray<btCollisionObject*>::at(int)")
        @test (s.type, s.name, s.parameters) == ("btAlignedObjectArray<btCollisionObject*>", "at", ["int"])
        @test sp("ns::Map<std::pair<int, int>, ns::V>::find(std::pair<int, int> const&) const").type ==
              "ns::Map<std::pair<int, int>, ns::V>"
        # Operators: `operator()`'s own parens are its name, a conversion
        # operator's name holds a type, and `<`/`>` are not template brackets.
        @test (sp("C::operator()(int) const").name, sp("C::operator()(int) const").parameters) == ("operator()", ["int"])
        @test sp("C::operator()()").name == "operator()"
        @test sp("im::Body::operator+=(im::Vec3 const&)").name == "operator+="
        @test sp("C::operator float*() const").name == "operator float*"
        @test (sp("C::operator<(C const&) const").type, sp("C::operator<(C const&) const").name) == ("C", "operator<")
        @test sp("C::operator->()").name == "operator->"
        # A member whose name merely starts with `operator` is an ordinary name.
        @test sp("C::operatorName(int)").name == "operatorName"
        # A function-pointer parameter: the list is the whole last group.
        @test sp("C::each(void (*)(int, void*), void*)").parameters == ["void (*)(int, void*)", "void*"]
        # Constructors and destructors are named as such, for the plan to pass over.
        @test sp("ic::Plain::Plain(int)").structor
        @test sp("ic::Plain::~Plain()").structor
        @test sp("Arr<int>::Arr(int)").structor
        @test !sp("ic::Plain::plain(int)").structor
        # What the unit cannot name.
        for bad in ("int im::Body::as<int>() const",              # a member template: a return type
                    "bool ns::C::operator==<int>(int const&) const",
                    "C::f() volatile", "C::f() &", "C::f() &&", "C::f() const &",
                    "C::log(char const*, ...)",
                    "(anonymous namespace)::C::f(int)",
                    "f(int)::Local::g()",
                    "C::f[abi:cxx11](int)",
                    "free_function(int)", "")
            @test (bad, sp(bad)) == (bad, nothing)
        end
    end

    # Hand-made metadata: each record is one way a member is forced or passed over.
    hdr = ["/src/lib/include/lib.h"]
    def(; kw...) = merge(Dict{String,Any}("kind" => "struct", "decl_files" => hdr),
                         Dict{String,Any}(String(k) => v for (k, v) in kw))
    sd = Dict{String,Any}(
        "Body" => def(), "Arr<int>" => def(), "Twice" => def(conflict = true),
        "NoHdr" => def(decl_files = ["/src/lib/src/nohdr.cpp"]),
        "pair<int, int>" => def(toolchain = true), "basic_string<char>" => def(scope = "std::__cxx11"),
        "__enum__E" => Dict{String,Any}("kind" => "enum"))
    row(sym; acc = "public", artificial = false, deleted = false, virt = false) =
        Dict{String,Any}("symbol" => sym, "accessibility" => acc, "artificial" => artificial,
                         "deleted" => deleted, "virtual" => virt)
    ra = Dict{String,Any}(
        "Body" => Dict{String,Any}("methods" => [
            row("_ZN2im4Body11setVelocityERKNS_4Vec3E"),          # forced
            row("_ZNK2im4Body11getVelocityEv"),                   # forced, const
            row("_ZN2im4Body8positionEv"),                        # forced
            row("_ZNK2im4Body8positionEv"),                       # forced, const
            row("_ZN2im4Body11defaultMassEv"),                    # forced (static: one spelling)
            row("_ZN2im4BodypLERKNS_4Vec3E"),                     # forced, an operator
            row("_ZN2im4Body9outOfLineEi"),                       # defined by the library
            row("_ZN2im4BodyC4Ef"),                               # a constructor: the other plan's
            row("_ZN2im4BodyD4Ev"; virt = true),                  # a destructor
            row("_ZNK2im4Body4kindEv"; virt = true),              # virtual: with the vtable
            row("_ZN2im4Body15protectedInlineEv"; acc = "protected"),
            row("_ZN2im4Body13privateInlineEv"; acc = "private"),
            row("_ZN2im4BodyaSERKS0_"; artificial = true),        # implicit operator=
            row("_ZN2im4Body6hiddenEv"),                          # dropped by pattern
            row("_ZNK2im4Body2asIiEET_v"),                        # a member template
            row("_ZN2im4Body6deadedEv"; deleted = true)]),
        "Arr<int>" => Dict{String,Any}("methods" => [row("_ZNK2im3ArrIiE4sizeEv"), row("_ZN2im3ArrIiE2atEi")]),
        "Twice" => Dict{String,Any}("methods" => [row("_ZN2im5Twice1fEv")]),
        "NoHdr" => Dict{String,Any}("methods" => [row("_ZN2im5NoHdr1fEv")]),
        "pair<int, int>" => Dict{String,Any}("methods" => [row("_ZNSt4pairIiiE4swapERS0_")]),
        "basic_string<char>" => Dict{String,Any}("methods" => [row("_ZNKSs4sizeEv")]))
    symbols = [Dict{String,Any}("mangled" => "_ZN2im4Body9outOfLineEi", "demangled" => "im::Body::outOfLine(int)")]

    @testset "declared minus defined: public, non-virtual, nameable" begin
        plan = _IM._inline_method_plan(sd, ra, symbols, ["*hidden*"])
        forced = plan["forced"]
        @test [(f["class"], f["symbol"]) for f in forced] == [
            ("Body", "_ZN2im4Body11setVelocityERKNS_4Vec3E"), ("Body", "_ZNK2im4Body11getVelocityEv"),
            ("Body", "_ZN2im4Body8positionEv"), ("Body", "_ZNK2im4Body8positionEv"),
            ("Body", "_ZN2im4Body11defaultMassEv"), ("Body", "_ZN2im4BodypLERKNS_4Vec3E")]
        by = Dict(f["symbol"] => f for f in forced)
        f = by["_ZN2im4Body11setVelocityERKNS_4Vec3E"]
        @test (f["type"], f["name"], f["parameters"], f["const"], f["headers"]) ==
              ("im::Body", "setVelocity", ["im::Vec3 const&"], false, hdr)
        @test by["_ZNK2im4Body8positionEv"]["const"] == true && by["_ZN2im4Body8positionEv"]["const"] == false
        @test by["_ZN2im4BodypLERKNS_4Vec3E"]["name"] == "operator+="
        # Named: what a recipe or a rebuild could change.
        skipped = Dict(s["symbol"] => s["reason"] for s in plan["skipped"])
        @test skipped == Dict(
            # A class template's members are `[types] templates`' to instantiate.
            "_ZNK2im3ArrIiE4sizeEv" => _IM._TEMPLATE_MEMBER_SKIP,
            "_ZN2im3ArrIiE2atEi" => _IM._TEMPLATE_MEMBER_SKIP,
            "_ZN2im4Body6hiddenEv" => "dropped by [wrap] exclude_symbols",
            "_ZNK2im4Body2asIiEET_v" => "cannot be named in C++ (a member template, or a qualifier the unit does not spell)",
            "_ZN2im5Twice1fEv" => "the name has distinct definitions in this library",
            "_ZN2im5NoHdr1fEv" => "declared in no header the unit can include")
        # Silent: the library's own definition, a constructor and a destructor,
        # a virtual, a protected, a private, an implicit and a deleted member,
        # a toolchain record's and a standard-library scope's.
        named = Set(vcat([f["symbol"] for f in forced], collect(keys(skipped))))
        for quiet in ("_ZN2im4Body9outOfLineEi", "_ZN2im4BodyC4Ef", "_ZN2im4BodyD4Ev", "_ZNK2im4Body4kindEv",
                      "_ZN2im4Body15protectedInlineEv", "_ZN2im4Body13privateInlineEv", "_ZN2im4BodyaSERKS0_",
                      "_ZN2im4Body6deadedEv", "_ZNSt4pairIiiE4swapERS0_", "_ZNKSs4sizeEv")
            @test (quiet, quiet in named) == (quiet, false)
        end
        # No demangler: named, not guessed.
        p2 = _IM._inline_method_plan(sd, ra, symbols, String[]; demangle = _ -> nothing)
        @test isempty(p2["forced"])
        @test !isempty(p2["skipped"]) &&
              all(s -> s["reason"] == "the symbol does not demangle on this host", p2["skipped"])
        # Metadata without the key (an older build): nothing planned.
        @test _IM._inline_method_plan(sd, Dict{String,Any}("Body" => Dict{String,Any}()), symbols, String[]) ==
              Dict{String,Any}("forced" => Any[], "skipped" => Any[])
    end

    plan = _IM._inline_method_plan(sd, ra, symbols, ["*hidden*"])
    ctor = Dict{String,Any}("class" => "Body", "symbol" => "_ZN2im4BodyC4Ef", "type" => "im::Body",
                            "parameters" => ["float"], "headers" => hdr)
    text = _IM._inline_ctor_unit_text(hdr, [ctor], plan["forced"])

    @testset "the unit: each member's address through the overload-selecting helper, one row per line" begin
        @test occursin("static constexpr auto m(R (C::*f)(A...)) -> R (C::*)(A...) { return f; }", text)
        @test occursin("static constexpr auto m(R (*f)(A...)) -> R (*)(A...) { return f; }", text)
        @test occursin("static constexpr auto mc(R (C::*f)(A...) const) -> R (C::*)(A...) const { return f; }", text)
        @test occursin("#define RB_FORCE_MEMBER __attribute__((used)) static const auto\n", text)
        @test occursin("\nRB_FORCE_MEMBER replibuild_member_0 = replibuild_force::args<im::Vec3 const&>::m(" *
                       "&im::Body::setVelocity);   // _ZN2im4Body11setVelocityERKNS_4Vec3E\n", text)
        @test occursin("replibuild_force::args<>::mc(&im::Body::position);   // _ZNK2im4Body8positionEv\n", text)
        @test occursin("replibuild_force::args<>::m(&im::Body::position);   // _ZN2im4Body8positionEv\n", text)
        @test occursin("replibuild_force::args<im::Vec3 const&>::m(&im::Body::operator+=);", text)
        @test count("RB_FORCE_MEMBER replibuild_member_", text) == length(plan["forced"]) == 6
        # The constructor part is as it was, and comes first.
        @test occursin("\ntemplate void replibuild_force::ctor<im::Body, float>(void*, float);   // _ZN2im4BodyC4Ef\n", text)
        @test findfirst("replibuild_force::ctor<im::Body", text).start < findfirst("RB_FORCE_MEMBER replibuild_member_0", text).start
        # No member rows: no member preamble, the #105 unit byte for byte.
        only_ctors = _IM._inline_ctor_unit_text(hdr, [ctor])
        @test !occursin("RB_FORCE_MEMBER", only_ctors) && !occursin("struct args", only_ctors)
        @test only_ctors == _IM._inline_ctor_unit_text(hdr, [ctor], Any[])
        # Only members: no constructor instantiation.
        @test !occursin("template void replibuild_force::ctor<", _IM._inline_ctor_unit_text(hdr, Any[], plan["forced"]))
    end

    @testset "a diagnostic is traced to its row by its line" begin
        rows = _IM._unit_row_of_line(text)
        lines = split(text, '\n')
        @test count(v -> v[1] === :ctor, values(rows)) == 1
        @test sort([v[2] for v in values(rows) if v[1] === :method]) == collect(1:6)
        for (ln, (kind, k)) in rows
            want = kind === :ctor ? "_ZN2im4BodyC4Ef" : plan["forced"][k]["symbol"]
            @test endswith(lines[ln], "// $want")
        end
        l_at = findfirst(l -> occursin("&im::Body::defaultMass)", l), lines)
        l_set = findfirst(l -> occursin("&im::Body::setVelocity)", l), lines)
        out = """
        /p/.replibuild_cache/replibuild_ctors_probe.cpp:$(l_set):78: error: no matching function for call to 'm'
        /p/.replibuild_cache/replibuild_ctors_probe.cpp:40:27: note: candidate template ignored: couldn't infer template argument 'R'
        /src/lib/include/lib.h:12:20: error: invalid operands to binary expression ('const V' and 'const V')
        /p/.replibuild_cache/replibuild_ctors_probe.cpp:$(l_at):61: note: in instantiation of member function 'im::Body::defaultMass' requested here
        /usr/include/other.h:3:1: error: something unrelated
        2 errors generated.
        """
        bad = _IM._unit_bad_lines(out, "/p/.replibuild_cache/replibuild_ctors_probe.cpp")
        # An error at a row is that row's; a note at a row belongs to the error
        # printed before it, in the header. Line 40 is the preamble, not a row.
        @test bad[l_set] == "no matching function for call to 'm'"
        @test bad[l_at] == "invalid operands to binary expression ('const V' and 'const V')"
        @test sort(collect(keys(bad))) == sort([l_set, l_at, 40])
        @test get(rows, 40, nothing) === nothing
        @test rows[l_set][1] === :method && rows[l_at][1] === :method
        # Diagnostics of another file name no line of the unit.
        @test isempty(_IM._unit_bad_lines("/src/lib/include/lib.h:12:20: error: x\n", "/p/unit.cpp"))
    end

    @testset "the unit's IR says which members have a body" begin
        mktempdir() do dir
            ll = joinpath(dir, "unit.ll")
            write(ll, """
            @_ZL19replibuild_member_0 = internal constant { i64, i64 } { i64 ptrtoint (ptr @_ZN2im4Body11setVelocityERKNS_4Vec3E to i64), i64 0 }, align 8
            define linkonce_odr hidden void @_ZN2im4Body11setVelocityERKNS_4Vec3E(ptr noundef %this, ptr noundef %v) #0 align 2 {
              ret void
            }
            define linkonce_odr dso_local noundef nonnull align 4 dereferenceable(12) ptr @_ZNK2im4Body11getVelocityEv(ptr noundef %this) {
              ret ptr %this
            }
            declare noundef i32 @_ZN2im4Body12declaredOnlyEi(ptr noundef nonnull align 8 dereferenceable(56), i32 noundef) #1
            declare i32 @__gxx_personality_v0(...)
            define weak_odr dso_local void @"_ZN4quot3fooEv"() {
              ret void
            }
            """)
            (defined, declared) = _IM._ir_defined_declared(ll)
            @test defined == Set(["_ZN2im4Body11setVelocityERKNS_4Vec3E", "_ZNK2im4Body11getVelocityEv", "_ZN4quot3fooEv"])
            @test declared == Set(["_ZN2im4Body12declaredOnlyEi", "__gxx_personality_v0"])
        end
    end

    @testset "the generated units' helper records are not the library's" begin
        sd = Dict{String,Any}(
            "ctor_impl<true, ic::Plain, int>" => Dict{String,Any}("scope" => "replibuild_force", "byte_size" => "0x1"),
            "args<int>" => Dict{String,Any}("scope" => "replibuild_force"),
            "Inner" => Dict{String,Any}("scope" => "replibuild_force::detail"),
            "Plain" => Dict{String,Any}("scope" => "ic"),
            "Mine" => Dict{String,Any}("scope" => "replibuild_forceful"),     # a library's own namespace
            "NoScope" => Dict{String,Any}(),
            "__enum__E" => Dict{String,Any}("kind" => "enum"))
        ra = Dict{String,Any}("ctor_impl<true, ic::Plain, int>" => Dict{String,Any}("empty" => true),
                              "Plain" => Dict{String,Any}("empty" => false))
        _IM._drop_force_helper_records!(sd, ra)
        @test sort(collect(keys(sd))) == ["Mine", "NoScope", "Plain", "__enum__E"]
        @test collect(keys(ra)) == ["Plain"]
        # The vendored dump of the constructor fixture was taken before any
        # unit existed; a parse of it loses nothing.
        rt, sd2, gv, td, ra2 = _IM.parse_dwarf_dump(IM_DUMP)
        @test haskey(sd2, "Plain") && haskey(ra2, "Plain")
    end

    @testset "the wrap report names what was forced, by kind" begin
        mplan = Dict{String,Any}(
            "forced" => Any[Dict{String,Any}("class" => "Body", "symbol" => "a", "emitted" => true),
                            Dict{String,Any}("class" => "Body", "symbol" => "b", "emitted" => true)],
            "skipped" => Any[Dict{String,Any}("class" => "Body", "symbol" => "c",
                                              "reason" => "declared in its header with no body there")])
        out = mktemp() do path, io
            redirect_stdout(io) do
                RepliBuild.Wrapper._report_inline_constructors(mplan; what = "member function")
            end
            close(io)
            read(path, String)
        end
        @test occursin("wrap: 2 inline member function(s) of 1 class(es) declared in headers and defined by the build's forced unit: Body", out)
        @test occursin("wrap: 1 class(es) with an inline member function left undefined, declared in its header with no body there: Body", out)
        @test !occursin("constructor", out)
        # A header's failure is reported as the header's, apart from rows
        # that do not compile themselves.
        hplan = Dict{String,Any}("forced" => Any[], "skipped" => Any[
            Dict{String,Any}("class" => "Wrapper", "symbol" => "w",
                             "reason" => "its header does not compile on its own (wrap_begin.h: x.h:4:1: error: expected '}')"),
            Dict{String,Any}("class" => "Odd", "symbol" => "o", "reason" => "does not compile: no member named 'q'")])
        hout = mktemp() do path, io
            redirect_stdout(io) do
                RepliBuild.Wrapper._report_inline_constructors(hplan)
            end
            close(io)
            read(path, String)
        end
        @test occursin("left undefined, its header does not compile on its own (a fragment", hout)
        @test occursin(r"its header does not compile on its own[^\n]*: Wrapper\n", hout)
        @test occursin(r"does not compile in the unit[^\n]*: Odd\n", hout)
    end
end
