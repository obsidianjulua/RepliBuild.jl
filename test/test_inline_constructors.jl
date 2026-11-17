# =============================================================================
# Forced inline constructors (2026-11-15)
#
# A constructor defined inline in a header and used by no translation unit of
# the library has no definition in it: the class DIE declares it, `nm` has no
# `C1`/`C2`, and the class cannot be built from Julia (Bullet's btSphereShape,
# btDefaultMotionState, btRigidBodyConstructionInfo). The build now writes a
# unit that odr-uses each such constructor through placement new, so clang
# emits the definitions into the library (Compiler.generate_inline_constructors).
#
# Toolchain-free: the parser over a vendored readelf dump of the fixture
# (test/inline_ctor_test, -fstandalone-debug), the declared-minus-defined
# plan over hand-made metadata, the symbol spelling, the unit's text and the
# export rewrite. test_inline_ctors_live.jl builds the fixture and constructs
# through the wrapper.
# =============================================================================

using Test
using RepliBuild

const _IC = RepliBuild.Compiler
const IC_DUMP = read(joinpath(@__DIR__, "fixtures", "inline_ctor.readelf.txt"), String)

# The function's source text from Compiler.jl, so a neutered definition can be
# put back exactly as it was.
function _ic_source(name::String)
    src = read(joinpath(@__DIR__, "..", "src", "Builder", "Compiler.jl"), String)
    start = findfirst("function $name(", src)
    start === nothing && error("no $name in Compiler.jl")
    stop = findnext("\nend\n", src, start.stop)
    return src[start.start:stop.stop]
end

_ic_rows(ra, name) = sort([Dict(r) for r in ra[name]["constructors"]]; by = r -> r["symbol"])

@testset "Forced inline constructors" begin

    @testset "the parser records each library record's declared constructors with the declaration's facts" begin
        rt, sd, gv, td, ra = _IC.parse_dwarf_dump(IC_DUMP)
        checked = 0
        # (record, [(symbol, accessibility, deleted)])
        for (name, want) in (("Plain", [("_ZN2ic5PlainC4Ei", "public", false), ("_ZN2ic5PlainC4Eii", "public", false)]),
                             ("Dflt", [("_ZN2ic4DfltC4Eii", "public", false)]),
                             ("Expl", [("_ZN2ic4ExplC4Ef", "public", false)]),
                             ("Abs", [("_ZN2ic3AbsC4Ei", "public", false)]),
                             ("Priv", [("_ZN2ic4PrivC4Ei", "private", false)]),      # a `class`: no attribute means private
                             ("Defined", [("_ZN2ic7DefinedC4Ei", "public", false)]),
                             ("Del", [("_ZN2ic3DelC4Ei", "public", false), ("_ZN2ic3DelC4Ev", "public", true)]))
            @test (name, haskey(sd, name)) == (name, true)
            @test (name, haskey(ra, name) && haskey(ra[name], "constructors")) == (name, true)
            (haskey(ra, name) && haskey(ra[name], "constructors")) || continue
            rows = _ic_rows(ra, name)
            @test (name, [(r["symbol"], r["accessibility"], r["deleted"]) for r in rows]) == (name, want)
            @test (name, all(r -> r["artificial"] == false, rows)) == (name, true)
            checked += 1
        end
        @test checked == 7
        # Enum keys and toolchain records carry no rows.
        @test !any(k -> startswith(k, "__enum__") && haskey(get(ra, k, Dict()), "constructors"), keys(sd))
    end

    @testset "a constructor's spelling is its symbol's demangling: class qualified, parameters as the compiler wrote them" begin
        dm = _IC._demangle("_ZN11btRigidBody27btRigidBodyConstructionInfoC4EfP13btMotionStateP16btCollisionShapeRK9btVector3")
        @test dm == "btRigidBody::btRigidBodyConstructionInfo::btRigidBodyConstructionInfo(float, btMotionState*, btCollisionShape*, btVector3 const&)"
        @test _IC._constructor_spelling(dm) ==
              ("btRigidBody::btRigidBodyConstructionInfo", ["float", "btMotionState*", "btCollisionShape*", "btVector3 const&"])
        @test _IC._constructor_spelling("ic::Plain::Plain()") == ("ic::Plain", String[])
        @test _IC._constructor_spelling("ns::Vec<int, std::pair<int, int> >::Vec(std::pair<int, int> const&, int)") ==
              ("ns::Vec<int, std::pair<int, int> >", ["std::pair<int, int> const&", "int"])
        # What the unit cannot name in C++.
        @test _IC._constructor_spelling("(anonymous namespace)::X::X(int)") === nothing
        @test _IC._constructor_spelling("f()::Local::Local(int)") === nothing
        @test _IC._constructor_spelling("S::S(f()::{lambda(int)#1})") === nothing
        @test _IC._constructor_spelling("S::S(std::string[abi:cxx11] const&)") === nothing
        @test _IC._constructor_spelling("S::S(int, ...)") === nothing
        @test _IC._constructor_spelling("bare(int)") === nothing
        # The symbol-table variants of a unified key.
        @test _IC._constructor_variants("_ZN2ic5PlainC4Eii") == ["_ZN2ic5PlainC1Eii", "_ZN2ic5PlainC2Eii", "_ZN2ic5PlainC3Eii"]
        @test _IC._constructor_variants("_ZN3FooC4IiEEi") == ["_ZN3FooC1IiEEi", "_ZN3FooC2IiEEi", "_ZN3FooC3IiEEi"]
        @test _IC._constructor_variants("_ZN3Foo3barEv") == String[]
    end

    # Hand-made metadata: library records with declared constructors, a symbol
    # table defining some of them.
    row(sym; acc = "public", deleted = false, artificial = false) =
        Dict{String,Any}("symbol" => sym, "accessibility" => acc, "deleted" => deleted, "artificial" => artificial)
    function rec(; kw...)
        d = Dict{String,Any}("kind" => "struct", "byte_size" => 4, "members" => Any[], "scope" => "ic",
                             "decl_files" => ["/src/lib/include/lib.h"])
        for (k, v) in kw
            d[String(k)] = v
        end
        return d
    end
    sd = Dict{String,Any}(
        "Plain" => rec(), "Dflt" => rec(), "Priv" => rec(kind = "class"), "Defined" => rec(), "Del" => rec(),
        "Templ" => rec(), "Imp" => rec(), "Twice" => rec(conflict = true), "NoHdr" => rec(decl_files = ["/src/lib/src/nohdr.cpp"]),
        "pair<int, int>" => rec(scope = "std", toolchain = true), "Hidden" => rec(), "Odd" => rec(),
        "__enum__Color" => Dict{String,Any}("kind" => "enum"))
    ra = Dict{String,Any}(
        "Plain" => Dict{String,Any}("constructors" => [row("_ZN2ic5PlainC4Ei"), row("_ZN2ic5PlainC4Eii")]),
        "Dflt" => Dict{String,Any}("constructors" => [row("_ZN2ic4DfltC4Eii")]),
        "Priv" => Dict{String,Any}("constructors" => [row("_ZN2ic4PrivC4Ei"; acc = "private")]),
        "Defined" => Dict{String,Any}("constructors" => [row("_ZN2ic7DefinedC4Ei")]),
        "Del" => Dict{String,Any}("constructors" => [row("_ZN2ic3DelC4Ev"; deleted = true), row("_ZN2ic3DelC4Ei")]),
        "Templ" => Dict{String,Any}("constructors" => [row("_ZN2ic5TemplC4IiEET_")]),
        "Imp" => Dict{String,Any}("constructors" => [row("_ZN2ic3ImpC4ERKS0_"; artificial = true)]),
        "Twice" => Dict{String,Any}("constructors" => [row("_ZN2ic5TwiceC4Ei")]),
        "NoHdr" => Dict{String,Any}("constructors" => [row("_ZN2ic5NoHdrC4Ei")]),
        "pair<int, int>" => Dict{String,Any}("constructors" => [row("_ZNSt4pairIiiEC4ERKiS2_")]),
        "Hidden" => Dict{String,Any}("constructors" => [row("_ZN2ic6HiddenC4Ei")]),
        "Odd" => Dict{String,Any}("constructors" => [row("_ZN2ic3OddC4Ei")]))
    symbols = [Dict{String,Any}("mangled" => m, "demangled" => _IC._demangle(m)) for m in
               ("_ZN2ic7DefinedC1Ei", "_ZN2ic7DefinedC2Ei", "_ZN2ic3DelC1Ei", "_ZN2ic3DelC2Ei", "_ZN2ic5PlainC1Ei")]
    # `Odd`'s demangling is unspellable on purpose, `Hidden` is excluded by pattern.
    demangle = s -> s == "_ZN2ic3OddC4Ei" ? "(anonymous namespace)::Odd::Odd(int)" : _IC._demangle(s)

    @testset "declared minus defined, and why each row is skipped" begin
        plan = _IC._inline_constructor_plan(sd, ra, symbols, ["*Hidden*"]; demangle = demangle)
        forced = plan["forced"]
        @test [(f["class"], f["symbol"]) for f in forced] == [("Dflt", "_ZN2ic4DfltC4Eii"), ("Plain", "_ZN2ic5PlainC4Eii")]
        @test all(f -> f["headers"] == ["/src/lib/include/lib.h"], forced)
        @test [f["type"] for f in forced] == ["ic::Dflt", "ic::Plain"]
        @test [f["parameters"] for f in forced] == [["int", "int"], ["int", "int"]]
        skipped = Dict(s["class"] => s["reason"] for s in plan["skipped"])
        @test skipped == Dict("Priv" => "private", "Del" => "deleted", "Templ" => "a constructor template",
                              "Imp" => "an implicit constructor, never declared",
                              "Twice" => "the name has distinct definitions in this library",
                              "NoHdr" => "declared in no header the unit can include (/src/lib/src/nohdr.cpp)",
                              "Hidden" => "dropped by [wrap] exclude_symbols",
                              "Odd" => "cannot be spelled in C++: (anonymous namespace)::Odd::Odd(int)")
        # Defined by the library (either variant): not forced, not reported.
        @test !any(s -> s["class"] in ("Defined", "Plain") && s["symbol"] in ("_ZN2ic7DefinedC4Ei", "_ZN2ic5PlainC4Ei"), plan["skipped"])
        # A toolchain record is nobody's to force and is not reported either.
        @test !any(s -> s["class"] == "pair<int, int>", plan["skipped"])
        # No demangler: named, not guessed.
        p2 = _IC._inline_constructor_plan(sd, ra, symbols, String[]; demangle = _ -> nothing)
        @test isempty(p2["forced"])
        @test all(s -> s["class"] in ("Priv", "Del", "Templ", "Imp", "Twice", "NoHdr") ||
                       s["reason"] == "the symbol does not demangle on this host", p2["skipped"])
    end

    @testset "the unit: one include per header, the guarded helper, one explicit instantiation per row" begin
        plan = _IC._inline_constructor_plan(sd, ra, symbols, ["*Hidden*"]; demangle = demangle)
        text = _IC._inline_ctor_unit_text(["btBulletDynamicsCommon.h", "/src/lib/include/lib.h"], plan["forced"])
        @test occursin("#include \"btBulletDynamicsCommon.h\"\n#include \"/src/lib/include/lib.h\"\n", text)
        @test occursin("#include <new>", text) && occursin("#include <type_traits>", text)
        @test occursin("::new (p) T(static_cast<A&&>(a)...)", text)
        @test occursin("std::is_abstract<T>::value && std::is_constructible<T, A...>::value", text)
        @test occursin("\ntemplate void replibuild_force::ctor<ic::Dflt, int, int>(void*, int, int);   // _ZN2ic4DfltC4Eii\n", text)
        @test occursin("\ntemplate void replibuild_force::ctor<ic::Plain, int, int>(void*, int, int);   // _ZN2ic5PlainC4Eii\n", text)
        @test count("template void replibuild_force::ctor<", text) == 2
        @test !occursin("Defined", text) && !occursin("Priv", text)
        @test _IC._inline_ctor_unit_text(["<vector>"], Any[]) |> t -> occursin("#include <vector>\n", t)
        # The unit and the templates TU are the generated force units: `-fno-inline`
        # keeps the constructor a definition instead of a body inlined into `go`.
        @test _IC._template_tu_compile_flags("/x/.replibuild_cache/replibuild_ctors.cpp") == ["-fno-inline"]
        @test _IC._template_tu_compile_flags("/x/src/lib.cpp") == String[]
        @test _IC._file_compile_fingerprint("abc", "/x/replibuild_ctors.cpp") ==
              _IC._file_compile_fingerprint("abc", "/x/replibuild_templates.cpp")
        @test _IC._file_compile_fingerprint("abc", "/x/replibuild_ctors.cpp") != "abc"
    end

    @testset "the forced definitions are made exportable, on ELF and PE, and nothing else moves" begin
        syms = Set(["_ZN2ic5PlainC1Eii", "_ZN2ic5PlainC2Eii"])
        d = "define linkonce_odr hidden void @_ZN2ic5PlainC2Eii(ptr %0, i32 %1) unnamed_addr #3 {"
        @test _IC._export_global_line(d, syms; pe = false) ==
              "define linkonce_odr void @_ZN2ic5PlainC2Eii(ptr %0, i32 %1) unnamed_addr #3 {"
        @test _IC._export_global_line(d, syms; pe = true) ==
              "define linkonce_odr dso_local dllexport void @_ZN2ic5PlainC2Eii(ptr %0, i32 %1) unnamed_addr #3 {"
        a = "@_ZN2ic5PlainC1Eii = linkonce_odr hidden unnamed_addr alias void (ptr, i32, i32), ptr @_ZN2ic5PlainC2Eii"
        @test _IC._export_global_line(a, syms; pe = false) ==
              "@_ZN2ic5PlainC1Eii = linkonce_odr unnamed_addr alias void (ptr, i32, i32), ptr @_ZN2ic5PlainC2Eii"
        @test _IC._export_global_line(a, syms; pe = true) ==
              "@_ZN2ic5PlainC1Eii = linkonce_odr dso_local dllexport unnamed_addr alias void (ptr, i32, i32), ptr @_ZN2ic5PlainC2Eii"
        # Already exportable: unchanged. Another symbol: unchanged. A call: unchanged.
        @test _IC._export_global_line("define dso_local dllexport void @_ZN2ic5PlainC2Eii(ptr %0) {", syms; pe = true) ==
              "define dso_local dllexport void @_ZN2ic5PlainC2Eii(ptr %0) {"
        other = "define hidden void @_ZN2ic5PlainC2Ei(ptr %0) {"
        @test _IC._export_global_line(other, syms; pe = false) == other
        call = "  call void @_ZN2ic5PlainC2Eii(ptr %0, i32 1, i32 2)"
        @test _IC._export_global_line(call, syms; pe = false) == call

        mktempdir() do dir
            ir = joinpath(dir, "linked.ll")
            write(ir, join([d, "  ret void", "}", a, other, call, "define hidden void @keep() {", "  ret void", "}"], "\n") * "\n")
            (out, n) = _IC._forced_exportable_ir([ir], syms, joinpath(dir, "exported"); pe = false)
            @test n == 2
            @test out[1] != ir && isfile(out[1])
            got = read(out[1], String)
            @test occursin("define linkonce_odr void @_ZN2ic5PlainC2Eii(", got)
            @test occursin("@_ZN2ic5PlainC1Eii = linkonce_odr unnamed_addr alias", got)
            @test occursin("define hidden void @_ZN2ic5PlainC2Ei(", got)     # not in the set
            @test occursin("define hidden void @keep() {", got)
            @test count("\n", got) == count("\n", read(ir, String))
            # Nothing to change: the file itself comes back.
            @test _IC._forced_exportable_ir([ir], Set(["@nobody"]), joinpath(dir, "exported2"); pe = false) == ([ir], 0)
            @test _IC._forced_exportable_ir([ir], Set{String}(), joinpath(dir, "exported3"); pe = false) == ([ir], 0)

            # A library-sized symbol set against a library-sized file: the
            # rewrite asks the set about each definition's name, never each
            # symbol about each line (Jolt's second link, 2026-11-17: 12,000
            # symbols, 38 s). The same lines move, and only those.
            many = Set{String}("_ZN2ic5Other$(i)C2Ev" for i in 1:20_000)
            big = joinpath(dir, "big.ll")
            open(big, "w") do io
                for i in 1:20_000
                    println(io, "define linkonce_odr hidden void @_ZN2ic5Other$(i)C2Ev(ptr %0) unnamed_addr {")
                    println(io, "  call void @_ZN2ic5Other$(i)C2Ev(ptr %0)")
                    println(io, "  ret void\n}")
                    println(io, "define hidden void @_ZN2ic4Kept$(i)Ev(ptr %0) {\n  ret void\n}")
                end
            end
            started = time()
            (bout, bn) = _IC._forced_exportable_ir([big], many, joinpath(dir, "exported4"); pe = false)
            @test bn == 20_000
            @test time() - started < 30      # seconds; the scan took minutes at this size
            bgot = read(bout[1], String)
            @test count("define linkonce_odr void @_ZN2ic5Other", bgot) == 20_000
            @test count("define hidden void @_ZN2ic4Kept", bgot) == 20_000
            @test !occursin("define linkonce_odr hidden", bgot)
        end
    end

    @testset "the rows that do not compile are found without one compile per row" begin
        # `_alone_verdicts` stands in front of the per-row compile: rows are
        # tried in groups and a failing group is halved. `probe` here is the
        # compiler: a set of rows compiles when none of them is bad.
        for (n, bad) in ((100, Int[]), (100, [1]), (100, [100]), (100, [3, 4, 17, 40, 41, 99]),
                         (37, collect(1:37)), (1, [1]), (1, Int[]), (0, Int[]), (2139, collect(500:572)))
            badset = Set(bad)
            calls = Threads.Atomic{Int}(0)
            singles = Vector{Int}[]
            lk = ReentrantLock()
            probe = function (g, round, k)
                Threads.atomic_add!(calls, 1)
                length(g) == 1 && lock(() -> push!(singles, g), lk)
                return any(in(badset), g) ? "row $(first(g)) of $(length(g))" : nothing
            end
            got = _IC._alone_verdicts(probe, n)
            # What one compile per row says, text included: a row is only ever
            # given the error of its own compile.
            @test got == Union{Nothing,String}[i in badset ? "row $i of 1" : nothing for i in 1:n]
            @test Set(first.(singles)) ⊇ badset
            # Few failures: far fewer compiles than rows. All failing: no more
            # than the halving costs.
            isempty(bad) && @test calls[] == cld(n, _IC._ALONE_GROUP_SIZE)
            (n == 2139) && @test calls[] < n ÷ 4
            @test calls[] <= 2n
        end
        # A group that fails while each of its rows compiles alone blames no
        # row: the unit compiled whole decides, as it did.
        @test _IC._alone_verdicts((g, _, _) -> length(g) > 1 ? "together only" : nothing, 40) ==
              fill(nothing, 40)
        # The group size is a parameter; 1 is one compile per row.
        seen = Vector{Int}[]
        lk1 = ReentrantLock()
        @test _IC._alone_verdicts((g, _, _) -> (lock(() -> push!(seen, g), lk1); g == [2] ? "no" : nothing), 3; group = 1) ==
              [nothing, "no", nothing]
        @test sort(seen) == [[1], [2], [3]]
    end

    @testset "negative check: without the defined-symbol subtraction the library's own constructors are forced" begin
        # The fix under test is declared MINUS defined. Neuter the variant
        # derivation, so no declared constructor ever matches a symbol, and the
        # plan must come to include `Defined`, whose constructor the library has.
        plan_before = _IC._inline_constructor_plan(sd, ra, symbols, String[]; demangle = demangle)
        @test !any(f -> f["class"] == "Defined", plan_before["forced"])
        orig = _ic_source("_constructor_variants")
        Core.eval(_IC, :(_constructor_variants(::AbstractString) = String[]))
        plan_neutered = Base.invokelatest(_IC._inline_constructor_plan, sd, ra, symbols, String[]; demangle = demangle)
        @test any(f -> f["class"] == "Defined", plan_neutered["forced"])
        @test length(plan_neutered["forced"]) > length(plan_before["forced"])
        Core.eval(_IC, Meta.parse(orig))
        plan_after = Base.invokelatest(_IC._inline_constructor_plan, sd, ra, symbols, String[]; demangle = demangle)
        @test plan_after == plan_before
        @test Base.invokelatest(_IC._constructor_variants, "_ZN2ic5PlainC4Eii") == ["_ZN2ic5PlainC1Eii", "_ZN2ic5PlainC2Eii", "_ZN2ic5PlainC3Eii"]
    end
end
