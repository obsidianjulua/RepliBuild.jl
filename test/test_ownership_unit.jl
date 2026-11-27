# test_ownership_unit.jl — what the build asks the compiler about a C++ type
# that crosses by value and that C++ will not copy bytewise.
#
# A `std::vector`, a `std::unique_ptr`, a class with a destructor: whether one
# must be ENDED and whether it can be COPIED are the compiler's answers, never
# the shape's. The build writes one line per such type into a generated unit
# (`OwnershipUnit`), reads the answers off the unit's IR, and files them in the
# metadata's `ownership` under the key both generator paths ask with
# (`FunctionGen.owned_key`).
#
# Here, without a compiler: which types have a key, on each calling
# convention; how a type is named (a function's return through `decltype`, a
# parameter by its demangled spelling); the unit's text, one row a line; the
# facts read off IR lines; and the rows a check refuses.
#
# test_ownership_unit_live.jl builds a library and calls what the unit defines.

using Test
using RepliBuild

const _OU = RepliBuild.OwnershipUnit
const _OU_FG = RepliBuild.JLCSIRGenerator.FunctionGen

# A library's tables, as the extraction writes them: one class with a
# destructor and a layout, one container, one type with no layout that passes
# by reference, and one plain record.
function _ou_tables()
    structs = Dict{String,Any}(
        "Tracked" => Dict{String,Any}("kind" => "class", "byte_size" => "0x8", "scope" => "own",
            "decl_files" => ["/src/own.hpp"],
            "members" => [Dict{String,Any}("name" => "box_", "c_type" => "int*", "offset" => 0, "size" => 8)]),
        "Shape" => Dict{String,Any}("kind" => "class", "byte_size" => "0x18", "scope" => "own",
            "decl_files" => ["/src/own.hpp"],
            "members" => [Dict{String,Any}("name" => "a", "c_type" => "long", "offset" => 0, "size" => 8),
                          Dict{String,Any}("name" => "b", "c_type" => "long", "offset" => 8, "size" => 8),
                          Dict{String,Any}("name" => "c", "c_type" => "long", "offset" => 16, "size" => 8)]),
        "Plain" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x10", "scope" => "own",
            "decl_files" => ["/src/own.hpp"],
            "members" => [Dict{String,Any}("name" => "a", "c_type" => "int", "offset" => 0, "size" => 4),
                          Dict{String,Any}("name" => "b", "c_type" => "double", "offset" => 8, "size" => 8)]))
    vec = "vector<int, std::allocator<int> >"
    fn = "function<int (int)>"
    record_abi = Dict{String,Any}(
        "Tracked" => Dict{String,Any}("byte_size" => "0x8", "pass" => "reference", "empty" => false),
        "Shape" => Dict{String,Any}("byte_size" => "0x18", "pass" => "value", "empty" => false,
            "methods" => [Dict{String,Any}("symbol" => "_ZNK3own5Shape6pointsEi"),
                          Dict{String,Any}("symbol" => "_ZNK3own5Shape7trackedEi"),
                          Dict{String,Any}("symbol" => "_ZNK3own5Shape4takeESt6vectorIiSaIiEE"),
                          Dict{String,Any}("symbol" => "_ZNK3own5Shape5chainB5cxx11Ei")]),
        "Plain" => Dict{String,Any}("byte_size" => "0x10", "pass" => "value", "empty" => false),
        vec => Dict{String,Any}("byte_size" => "0x18", "pass" => "reference", "empty" => false),
        fn => Dict{String,Any}("byte_size" => "0x20", "pass" => "reference", "empty" => false))
    return (structs, record_abi, vec, fn)
end

_ou_fn(mangled, demangled, ret; params = Any[], extra...) = merge(
    Dict{String,Any}("mangled" => mangled, "demangled" => demangled,
                     "return_type" => Dict{String,Any}("c_type" => ret), "parameters" => params),
    Dict{String,Any}(string(k) => v for (k, v) in extra))
_ou_this() = Dict{String,Any}("name" => "this", "c_type" => "Shape*")

@testset "which types have a key, on every convention" begin
    (structs, abi, vec, fn) = _ou_tables()
    checked = 0
    for conv in (:sysv, :win64, :aapcs64), ret in (false, true)
        key(t) = _OU_FG.owned_key(t, structs, abi; abi = conv, ret)
        # A class with a destructor and a layout: its table entry.
        @test key("Tracked") == "Tracked"
        @test key("const Tracked") == "Tracked"
        @test key("own::Tracked") == "Tracked"
        # A container, which has no layout here: its spelling.
        @test key(vec) == vec
        @test key("const " * vec) == vec
        # No layout and passed by reference: its spelling.
        @test key(fn) == fn
        # Not by value, or nothing C++ refuses to copy bytewise.
        for t in ("Tracked*", "Tracked&", "const " * vec * "&", vec * "*", "Plain", "Shape", "int", "double",
                  "void", "", "char*")
            @test (conv, ret, t, key(t)) == (conv, ret, t, nothing)
        end
        checked += 1
    end
    @test checked == 6
    # A pointer INSIDE the template arguments is not the type's own.
    abi["vector<own::Tracked *, std::allocator<own::Tracked *> >"] =
        Dict{String,Any}("byte_size" => "0x18", "pass" => "reference", "empty" => false)
    @test _OU_FG.owned_key("vector<own::Tracked *, std::allocator<own::Tracked *> >", structs, abi) ==
          "vector<own::Tracked *, std::allocator<own::Tracked *> >"
end

@testset "a function with no class in front of its name" begin
    @test _OU.function_spelling("iota(int)") == (name = "iota", parameters = ["int"])
    @test _OU.function_spelling("operator*(btMatrix3x3 const&, btMatrix3x3 const&)") ==
          (name = "operator*", parameters = ["btMatrix3x3 const&", "btMatrix3x3 const&"])
    @test _OU.function_spelling("make()") == (name = "make", parameters = String[])
    @test _OU.function_spelling("pick(std::vector<int, std::allocator<int> >, int)") ==
          (name = "pick", parameters = ["std::vector<int, std::allocator<int> >", "int"])
    # A scope is `_method_spelling`'s; a return type in front is a template.
    @test _OU.function_spelling("own::iota(int)") === nothing
    @test _OU.function_spelling("std::vector<int> make<int>(int)") === nothing
    @test _OU.function_spelling("int twice<int>(int)") === nothing
    @test _OU.function_spelling("log(char const*, ...)") === nothing
    @test _OU.function_spelling("not a function") === nothing
end

@testset "how each type is named" begin
    (structs, abi, vec, fn) = _ou_tables()
    functions = Any[
        # A member whose class a header declares: the return through decltype.
        _ou_fn("_ZNK3own5Shape6pointsEi", "own::Shape::points(int) const", vec;
               params = Any[_ou_this(), Dict{String,Any}("name" => "n", "c_type" => "int")]),
        _ou_fn("_ZNK3own5Shape7trackedEi", "own::Shape::tracked(int) const", "Tracked";
               params = Any[_ou_this(), Dict{String,Any}("name" => "v", "c_type" => "int")]),
        # A by-value parameter: its own demangled spelling.
        _ou_fn("_ZNK3own5Shape4takeESt6vectorIiSaIiEE",
               "own::Shape::take(std::vector<int, std::allocator<int> >) const", "long";
               params = Any[_ou_this(), Dict{String,Any}("name" => "v", "c_type" => vec)]),
        # A function in a namespace, which no class declares.
        _ou_fn("_ZN3own4iotaEi", "own::iota(int)", vec;
               params = Any[Dict{String,Any}("name" => "n", "c_type" => "int")]),
        # One with no scope at all.
        _ou_fn("_Z4makev", "make()", "Tracked"),
        # A type only a parameter names.
        _ou_fn("_ZN3own5applyESt8functionIFiiEE", "own::apply(std::function<int (int)>)", "int";
               params = Any[Dict{String,Any}("name" => "f", "c_type" => fn)]),
        # Not asked about: plain data, a reference, a pure virtual row.
        _ou_fn("_ZNK3own5Shape5plainEv", "own::Shape::plain() const", "Plain"; params = Any[_ou_this()]),
        _ou_fn("_ZN3own3sumERKSt6vectorIiSaIiEE", "own::sum(std::vector<int, std::allocator<int> > const&)", "long";
               params = Any[Dict{String,Any}("name" => "v", "c_type" => "const " * vec * "&")]),
        _ou_fn("_ZNK3own5Shape4pureEv", "own::Shape::pure() const", vec; params = Any[_ou_this()], vtable_only = true),
    ]
    p = _OU.plan(functions, structs, abi; demangle = _ -> nothing)
    rows = p["candidates"]
    @test isempty(p["skipped"])
    @test sort!(unique(String[r["key"] for r in rows])) == sort!(["Tracked", vec, fn])
    @test [Int(r["id"]) for r in rows] == collect(0:length(rows)-1)
    of(key) = [r for r in rows if r["key"] == key]

    # The container: the member's return first, then the parameter's own
    # spelling, then the namespace function's return.
    v = of(vec)
    @test length(v) == 3
    @test v[1]["expr"] == "decltype(replibuild_force::own_args<int>::mc(&own::Shape::points))"
    @test v[1]["headers"] == ["/src/own.hpp"]
    @test v[1]["ret"] == true
    @test v[2]["expr"] == "std::vector<int, std::allocator<int> >"
    @test v[2]["ret"] == false
    @test occursin("parameter 1 of _ZNK3own5Shape4take", v[2]["from"])
    @test v[3]["expr"] == "decltype(replibuild_force::own_args<int>::m(&own::iota))"
    @test isempty(v[3]["headers"])
    @test all(r -> r["bytes"] == 24, v)

    t = of("Tracked")
    @test length(t) == 3
    @test t[1]["expr"] == "decltype(replibuild_force::own_args<int>::mc(&own::Shape::tracked))"
    @test t[2]["expr"] == "decltype(replibuild_force::own_args<>::m(&::make))"
    # Last, its own name under the scope its definition was read in, with its own header.
    @test t[3]["expr"] == "own::Tracked" && t[3]["from"] == "its own name" && t[3]["rank"] == _OU.OWN_NAME_RANK
    @test t[3]["headers"] == ["/src/own.hpp"]
    @test all(r -> r["bytes"] == 8, t)

    f = of(fn)
    @test length(f) == 1
    @test f[1]["expr"] == "std::function<int (int)>"
    @test f[1]["bytes"] == 32

    # No more candidates than the cap, however many functions return a type.
    many = Any[_ou_fn("_ZN3own1f$(i)Ev", "own::f$(i)()", vec) for i in 1:9]
    @test length(_OU.plan(many, structs, abi)["candidates"]) == _OU.MAX_CANDIDATES
    # The same plan whatever order the functions come in.
    @test _OU.plan(reverse(functions), structs, abi; demangle = _ -> nothing) == p
end

@testset "a type's own name, where no function is declared" begin
    (structs, abi, vec, fn) = _ou_tables()
    # The scope a record's facts carry is the one its definition was read in.
    abi[vec]["scope"] = "std"
    lst = "list<int, std::allocator<int> >"
    wide = "basic_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t> >"
    abi[lst] = Dict{String,Any}("byte_size" => "0x18", "pass" => "reference", "empty" => false, "scope" => "std")
    abi[wide] = Dict{String,Any}("byte_size" => "0x20", "pass" => "reference", "empty" => false, "scope" => "std::__cxx11")
    @test _OU.own_name(vec, vec, structs, abi) == ("std::vector<int, std::allocator<int> >", true)
    @test _OU.own_name(wide, "const " * wide, structs, abi) ==
          ("std::__cxx11::basic_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t> >", true)
    # A library record: its definition's scope, and it is the project's.
    @test _OU.own_name("Tracked", "Tracked", structs, abi) == ("own::Tracked", false)
    structs["Top"] = Dict{String,Any}("kind" => "struct", "byte_size" => "0x8", "scope" => "")
    @test _OU.own_name("Top", "Top", structs, abi) == ("Top", false)
    # No scope known, two scopes, two layouts, or a name no unit can write: nothing.
    @test _OU.own_name(fn, fn, structs, abi) === nothing
    abi["Twice"] = Dict{String,Any}("byte_size" => "0x8", "pass" => "reference", "empty" => false, "scope_ambiguous" => true)
    @test _OU.own_name("Twice", "Twice", structs, abi) === nothing
    abi["Clash"] = Dict{String,Any}("byte_size" => "0x8", "pass" => "reference", "empty" => false, "scope" => "a", "conflict" => true)
    @test _OU.own_name("Clash", "Clash", structs, abi) === nothing
    structs["Both"] = Dict{String,Any}("kind" => "struct", "byte_size" => "0x8", "scope" => "a", "conflict" => true)
    @test _OU.own_name("Both", "Both", structs, abi) === nothing
    for bad in ("(lambda at /src/x.cpp:3:5)", "function<void ((anonymous namespace)::T)>")
        abi[bad] = Dict{String,Any}("byte_size" => "0x8", "pass" => "reference", "empty" => false, "scope" => "std")
        @test _OU.own_name(bad, bad, structs, abi) === nothing
    end

    # The standard headers a spelling's names are declared in.
    @test _OU.std_headers("std::vector<int, std::allocator<int> >") == ["<memory>", "<vector>"]
    @test _OU.std_headers("std::map<int, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, " *
                          "std::less<int>, std::allocator<std::pair<const int, int> > >") ==
          ["<functional>", "<map>", "<memory>", "<string>", "<utility>"]
    @test _OU.std_headers("std::filesystem::path") == ["<filesystem>"]
    @test _OU.std_headers("std::__1::unique_ptr<own::Tracked, std::__1::default_delete<own::Tracked> >") == ["<memory>"]
    @test _OU.std_headers("std::optional<std::chrono::duration<long, std::ratio<1L, 1000L> > >") ==
          ["<chrono>", "<optional>", "<ratio>"]
    @test isempty(_OU.std_headers("own::Tracked")) && isempty(_OU.std_headers("mystd::vector<int>"))
    @test isempty(_OU.std_headers("std::not_a_standard_name<int>"))      # no header: the compile says so

    # A library whose functions no header declares: the return's line cannot
    # compile, and the type's own name is there beside it.
    free = Any[_ou_fn("_Z4makei", "make(int)", lst), _ou_fn("_Z4widev", "wide()", wide)]
    p = _OU.plan(free, structs, abi)
    rows = [r for r in p["candidates"] if r["key"] == lst]
    @test [r["expr"] for r in rows] == ["decltype(replibuild_force::own_args<int>::m(&::make))",
                                        "std::list<int, std::allocator<int> >"]
    @test rows[2]["headers"] == ["<list>", "<memory>"] && rows[2]["ret"] == true && rows[2]["bytes"] == 24
    @test [r["headers"] for r in p["candidates"] if r["key"] == wide][end] == ["<memory>", "<string>"]
    # It is kept beside the cap, however many functions return the type.
    many = Any[_ou_fn("_ZN3own1f$(i)Ev", "own::f$(i)()", vec) for i in 1:9]
    q = _OU.plan(many, structs, abi)["candidates"]
    @test length(q) == _OU.MAX_CANDIDATES + 1
    @test q[end]["expr"] == "std::vector<int, std::allocator<int> >" && q[end]["rank"] == _OU.OWN_NAME_RANK
    # A type only a function template returns had no line at all: now it has its name.
    tmpl = Any[_ou_fn("_ZN3own4makeIiEESt4listIT_SaIS2_EEi", "std::list<int> own::make<int>(int)", lst)]
    q = _OU.plan(tmpl, structs, abi)
    @test isempty(q["skipped"]) && [r["expr"] for r in q["candidates"]] == ["std::list<int, std::allocator<int> >"]
    # A parameter written by name gets its standard headers too.
    take = Any[_ou_fn("_Z4takeSt6vectorIiSaIiEE", "take(std::vector<int, std::allocator<int> >)", "long";
                      params = Any[Dict{String,Any}("name" => "v", "c_type" => vec)])]
    q = _OU.plan(take, structs, Dict{String,Any}(vec => Dict{String,Any}("byte_size" => "0x18", "pass" => "reference", "empty" => false)))
    @test [(r["expr"], r["headers"]) for r in q["candidates"]] == [("std::vector<int, std::allocator<int> >", ["<memory>", "<vector>"])]
    # The project's records a template argument names bring their header.
    vt = "vector<own::Tracked, std::allocator<own::Tracked> >"
    abi[vt] = Dict{String,Any}("byte_size" => "0x18", "pass" => "reference", "empty" => false, "scope" => "std")
    q = _OU.plan(Any[_ou_fn("_Z3allv", "all()", vt)], structs, abi)["candidates"]
    @test q[end]["expr"] == "std::vector<own::Tracked, std::allocator<own::Tracked> >"
    @test q[end]["headers"] == ["/src/own.hpp", "<memory>", "<vector>"]

    # The facts of one name read in two scopes carry no scope.
    C = RepliBuild.Compiler
    facts = Dict{String,Dict{String,Any}}()
    entry(scope) = Dict{String,Any}("byte_size" => "0x8", "pass" => "reference", "empty" => false, "scope" => scope)
    C._merge_record_facts!(facts, "T", entry("a"))
    C._merge_record_facts!(facts, "T", entry("a"))
    @test facts["T"]["scope"] == "a" && !haskey(facts["T"], "scope_ambiguous") && !haskey(facts["T"], "conflict")
    C._merge_record_facts!(facts, "T", entry("b"))
    @test !haskey(facts["T"], "scope") && facts["T"]["scope_ambiguous"] == true && !haskey(facts["T"], "conflict")
    C._merge_record_facts!(facts, "T", entry("a"))
    @test !haskey(facts["T"], "scope")                      # once ambiguous, it stays so
    # A definition whose scope the tree did not settle is no evidence for the other's.
    C._merge_record_facts!(facts, "U", entry("a"))
    C._merge_record_facts!(facts, "U", delete!(entry("a"), "scope"))
    @test !haskey(facts["U"], "scope") && facts["U"]["scope_ambiguous"] == true
end

@testset "what a line cannot name is said, never guessed" begin
    (structs, abi, vec, _) = _ou_tables()
    lst = "list<int, std::allocator<int> >"
    abi[lst] = Dict{String,Any}("byte_size" => "0x18", "pass" => "reference", "empty" => false)

    # An ABI tag is part of the symbol, not of the name a unit writes.
    tagged = Any[_ou_fn("_ZNK3own5Shape5chainB5cxx11Ei", "own::Shape::chain[abi:cxx11](int) const", lst;
                        params = Any[_ou_this(), Dict{String,Any}("name" => "n", "c_type" => "int")])]
    p = _OU.plan(tagged, structs, abi)
    @test length(p["candidates"]) == 1
    @test p["candidates"][1]["expr"] == "decltype(replibuild_force::own_args<int>::mc(&own::Shape::chain))"

    # A function template's return cannot be named this way, and neither can a
    # default-argument entry point, whose name is the member's with fewer
    # parameters than any overload takes.
    for f in (_ou_fn("_ZN3own4makeIiEESt4listIT_SaIS2_EEi", "std::list<int> own::make<int>(int)", lst),
              _ou_fn("_ZN18replibuild_default2d0EPN3own5ShapeE", "own::Shape::chain()", lst;
                     params = Any[_ou_this()], default_arguments = Dict{String,Any}("of" => "x", "omitted" => 1)))
        q = _OU.plan(Any[f], structs, abi)
        @test isempty(q["candidates"])
        @test length(q["skipped"]) == 1
        @test q["skipped"][1]["key"] == lst
        @test occursin("no function the unit can name returns it", q["skipped"][1]["reason"])
    end
    # Named by another function, it is not skipped.
    both = vcat(tagged, Any[_ou_fn("_ZN3own4makeIiEESt4listIT_SaIS2_EEi", "std::list<int> own::make<int>(int)", lst)])
    @test isempty(_OU.plan(both, structs, abi)["skipped"])

    # A parameter the demangled list does not line up with, or cannot spell.
    for (dm, why) in (("own::Shape::take(std::vector<int, std::allocator<int> >, int) const", "cannot be written"),
                      ("own::Shape::take((anonymous namespace)::V) const", "cannot be written"))
        f = _ou_fn("_ZNK3own5Shape4takeX", dm, "long"; params = Any[_ou_this(), Dict{String,Any}("name" => "v", "c_type" => vec)])
        q = _OU.plan(Any[f], structs, abi)
        @test isempty(q["candidates"])
        @test occursin(why, q["skipped"][1]["reason"])
    end
end

@testset "the unit: one row a line" begin
    rows = Any[Dict{String,Any}("id" => 0, "expr" => "decltype(replibuild_force::own_args<int>::mc(&own::Shape::points))",
                               "from" => "_ZNK3own5Shape6pointsEi returns it"),
               Dict{String,Any}("id" => 4, "expr" => "std::map<int, int, std::less<int>, std::allocator<std::pair<int const, int> > >",
                               "from" => "parameter 1 of _Z1f\nwith a break\\")]
    text = _OU.unit_text(["own.hpp", "<vector>"], rows)
    lines = split(text, '\n')
    @test "#include \"own.hpp\"" in lines
    @test "#include <vector>" in lines
    ids = _OU.row_of_line(text)
    @test sort!(collect(values(ids))) == [0, 4]
    for (line, id) in ids
        @test startswith(lines[line], "RB_OWN_ROW($id, ")
        # Nothing of a row runs onto a second line, and none ends in a
        # backslash, which would splice the next line into its comment.
        @test !endswith(lines[line], "\\")
    end
    @test count(l -> startswith(l, "RB_OWN_ROW("), lines) == 2
    # The helper templates live where the extraction already keeps them out
    # of the library's records and functions.
    @test occursin("namespace replibuild_force {", text)
    @test !occursin("namespace replibuild_own", text)
    # The functions a wrapper calls are named by row.
    @test occursin("replibuild_own_end_##n", text)
    @test _OU.helper("end", 4) == "replibuild_own_end_4"
    @test _OU.helper("copy", 4) == "replibuild_own_copy_4"
    @test _OU.helper("move", 4) == "replibuild_own_move_4"
    # No row, no rows.
    @test isempty(_OU.row_of_line(_OU.unit_text(["own.hpp"], Any[])))
end

# IR as clang writes it for a row: the instantiation whose name states the
# facts, and the three functions.
function _ou_ir(dir, rows; drop = String[])
    path = joinpath(dir, "own.ll")
    open(path, "w") do io
        println(io, "target triple = \"x86_64-pc-linux-gnu\"")
        for (id, size, align, flags, letter, quoted) in rows
            name = "_ZN16replibuild_force9own_factsIL$(letter)$(id)EL$(letter)$(size)EL$(letter)$(align)EL$(letter)$(flags)EEEvv"
            println(io, "define weak_odr dso_local void @", quoted ? "\"$name\"" : name, "() #0 comdat {")
            println(io, "  ret void\n}")
            for k in ("end", "copy", "move")
                h = _OU.helper(k, id)
                h in drop && continue
                println(io, "define dso_local ", k == "end" ? "void" : "i32", " @$h(ptr noundef %0) #1 {")
                println(io, "  ret ", k == "end" ? "void" : "i32 0", "\n}")
            end
        end
        println(io, "declare void @_ZdlPv(ptr noundef) #2")
    end
    return path
end

@testset "the compiler's answers, read off the unit's IR" begin
    mktempdir() do dir
        ir = _ou_ir(dir, [(0, 24, 8, 7, 'y', false), (3, 8, 8, 5, 'm', false), (12, 16, 4, 6, 'y', true)])
        f = _OU.facts(ir)
        @test sort!(collect(keys(f))) == [0, 3, 12]
        @test f[0] == (size = 24, align = 8, flags = 7)
        @test f[3] == (size = 8, align = 8, flags = 5)      # `unsigned long` on another target
        @test f[12] == (size = 16, align = 4, flags = 6)    # a quoted name
        @test isempty(_OU.facts(joinpath(dir, "absent.ll")))
        # What the flags say.
        @test (7 & _OU.ENDS) != 0 && (7 & _OU.COPYABLE) != 0 && (7 & _OU.MOVABLE) != 0 && (7 & _OU.BITWISE) == 0
        @test (5 & _OU.COPYABLE) == 0
        @test (6 & _OU.ENDS) == 0
    end
end

@testset "a row whose answer cannot be trusted is refused" begin
    (structs, abi, vec, _) = _ou_tables()
    row(id, key, bytes; c_type = key, ret = true) = Dict{String,Any}(
        "id" => id, "key" => key, "bytes" => bytes, "c_type" => c_type, "ret" => ret,
        "expr" => "x", "headers" => String[], "from" => "f$id returns it")
    mktempdir() do dir
        ir = _ou_ir(dir, [(0, 24, 8, 7, 'y', false), (1, 8, 8, 5, 'y', false), (2, 8, 8, 23, 'y', false),
                          (3, 40, 8, 7, 'y', false), (5, 8, 8, 7, 'y', false)];
                    drop = [_OU.helper("copy", 5)])
        p = Dict{String,Any}("skipped" => Dict{String,Any}[],
            "rows" => Dict{String,Any}[row(0, vec, 24), row(1, "Tracked", 8), row(2, "Ref", nothing),
                                       row(3, "Big", 24), row(4, "Absent", 8), row(5, "Half", 8)])
        _OU.checked!(p, ir)
        @test [r["key"] for r in p["rows"]] == [vec, "Tracked"]
        @test p["rows"][1]["size"] == 24 && p["rows"][1]["flags"] == 7
        why = Dict(s["key"] => s["reason"] for s in p["skipped"])
        @test occursin("named a reference", why["Ref"])
        @test occursin("40 bytes where the library's is 24", why["Big"])
        @test occursin("states nothing for it", why["Absent"])
        @test occursin("does not define its functions", why["Half"])

        # A function the library does not export cannot be called.
        runtime = Set{String}(vcat([_OU.helper(k, 0) for k in ("end", "copy", "move")],
                                   [_OU.helper(k, 1) for k in ("end", "copy")]))
        q = deepcopy(p)
        _OU.exported!(q, runtime)
        @test [r["key"] for r in q["rows"]] == [vec]
        @test any(s -> s["key"] == "Tracked" && occursin("does not export", s["reason"]), q["skipped"])
        # "Could not look" refuses nothing.
        @test length(_OU.exported!(deepcopy(p), nothing)["rows"]) == 2

        # The metadata's table: the answers by key, and each function's name.
        m = _OU.metadata(p, structs, abi)
        @test sort!(collect(keys(m["types"]))) == sort!(["Tracked", vec])
        v = m["types"][vec]
        @test (v["size"], v["align"], v["ends"], v["copy"], v["move"], v["bitwise"]) == (24, 8, true, true, true, false)
        @test (v["end_symbol"], v["copy_symbol"], v["move_symbol"]) ==
              ("replibuild_own_end_0", "replibuild_own_copy_0", "replibuild_own_move_0")
        @test m["types"]["Tracked"]["copy"] == false
        @test [s["key"] for s in m["skipped"]] == sort!(["Absent", "Big", "Half", "Ref"])

        # A library built without the unit has no answer for any type.
        r = _OU.refused!(deepcopy(p), "multiple definition of `f`")
        @test isempty(r["rows"])
        @test count(s -> occursin("did not build with its ownership unit: multiple definition", s["reason"]), r["skipped"]) == 2
        @test isempty(_OU.metadata(r, structs, abi)["types"])
    end
end

@testset "an answer is filed under the key the generator will ask with" begin
    (structs, abi, _, fn) = _ou_tables()
    # Planned while the type had no layout here: filed under its spelling.
    row = Dict{String,Any}("id" => 0, "key" => fn, "c_type" => fn, "ret" => false, "bytes" => 32,
                           "size" => 32, "align" => 8, "flags" => 7, "from" => "parameter 1 of f")
    p = Dict{String,Any}("rows" => Dict{String,Any}[row], "skipped" => Dict{String,Any}[])
    @test collect(keys(_OU.metadata(p, structs, abi)["types"])) == [fn]
    # The unit's own debug info gave it one: the final extraction's key is its
    # table entry, and that is where the answer goes.
    structs[fn] = Dict{String,Any}("kind" => "class", "byte_size" => "0x20",
        "members" => [Dict{String,Any}("name" => "a", "c_type" => "long", "offset" => 0, "size" => 8)])
    @test _OU_FG.owned_key(fn, structs, abi) == fn      # the entry's name is the spelling here
    structs2 = Dict{String,Any}("Later" => structs[fn])
    abi2 = Dict{String,Any}("Later" => Dict{String,Any}("byte_size" => "0x20", "pass" => "reference", "empty" => false))
    row2 = merge(row, Dict{String,Any}("key" => "spelled::Later", "c_type" => "spelled::Later"))
    q = Dict{String,Any}("rows" => Dict{String,Any}[row2], "skipped" => Dict{String,Any}[])
    @test collect(keys(_OU.metadata(q, structs2, abi2)["types"])) == ["Later"]
    # A type the final tables no longer know keeps the key it was planned under.
    @test collect(keys(_OU.metadata(q, Dict{String,Any}(), Dict{String,Any}())["types"])) == ["spelled::Later"]
end

@testset "the build keeps the unit's functions out of the library's API" begin
    src = read(joinpath(RepliBuild.SRC_DIR, "Builder", "Compiler.jl"), String)
    # Its three functions per row are never wrapped as functions...
    @test occursin("startswith(mangled_name, \"replibuild_own_\") && continue", src)
    # ...count as ours where a DLL's export intent is read...
    @test occursin("occursin(\"replibuild_own_\", line)", src)
    # ...and a stale unit never shapes a build's first link.
    @test occursin("rm(_own_unit_path(config); force=true)", src)
    @test _OU.HELPER_PREFIX == "replibuild_own_"
end
