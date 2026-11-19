# =============================================================================
# Pure virtual member functions (library-free, CI-safe)
#
# `virtual const char *name() const = 0;` has no definition, so no symbol, and
# the wrap surface is built from the symbol table: an interface's own methods
# had no wrapper, and nothing could call `name()` on the `Animal*` a factory
# returned (2026-11-17). What reaches such a function is the object's vtable,
# and its declaration DIE says where. So:
#
#   * the parser records, on every member function a record declares, whether
#     it is pure and its vtable slot;
#   * the build gives each pure virtual no symbol defines a function row
#     marked `vtable_only` (private and protected ones, destructors,
#     variadic ones and those of a class with internal linkage are passed
#     over, each with a reason), and marks the records such a function
#     leaves abstract;
#   * the thunk generator writes a thunk that is only a `jlcs.vcall`, with no
#     declaration of a callee that does not exist, or none at all where the
#     signature passes a record by value;
#   * the JIT's pre-flight does not look the name up;
#   * GeneratorCpp binds the thunk (never a `ccall`), refuses where there is no
#     thunk, and refuses the constructor of an abstract class, which run on
#     storage of its own would leave an object whose pure slots end the process.
#
# test/fixtures/pure_virtual.readelf.txt is `readelf --debug-dump=info` of
# (clang++ 23 -std=c++17 -g -fstandalone-debug -O0 -fPIC
#  -fdebug-compilation-dir=/src/pv -c):
#
#   namespace pv {
#   struct Span { double lo; double hi; };
#   class Animal {
#   public:
#       explicit Animal(int legs);
#       virtual ~Animal();
#       virtual const char *name() const = 0;
#       virtual int legs() const;
#       virtual double speed(double terrain) const = 0;
#       virtual Span span() const = 0;
#       virtual int weigh(Span s) const = 0;
#       virtual int log(const char *fmt, ...) = 0;
#   protected:
#       virtual int secret() const = 0;
#       int legs_;
#   private:
#       virtual int hidden() const = 0;
#   };
#   struct Dog : Animal { Dog(); …every one of them overridden… };
#   struct Token { int id; Token(); virtual ~Token() = 0; };
#   struct Shape { int tag; Shape(); virtual ~Shape();
#                  virtual int area() const = 0; virtual int sides() const = 0; };
#   struct Poly : Shape { int n; explicit Poly(int n_); int sides() const override; };
#   struct Square : Poly { int side; explicit Square(int side_); int area() const override; };
#   struct Base { virtual ~Base(); virtual int tag() const = 0; };
#   struct VRoot { VRoot(); virtual ~VRoot(); virtual int who() const = 0; };
#   struct VLeft : virtual VRoot { int l; VLeft(); };
#   struct VImpl : virtual VRoot { VImpl(); int who() const override; };
#   struct VJoin : VLeft, VImpl { VJoin(); };
#   int Base::tag() const { return 7; }
#   void use(Animal *, Dog *, Token *, Square *, Base *, VJoin *) {}
#   }
#
# test_pure_virtual_live.jl (devtests) builds test/pure_virtual_test and calls
# the functions through the wrapper, with JIT and with AOT thunks.
# =============================================================================

using Test
using RepliBuild
using Libdl
using Logging

const _PV_C = RepliBuild.Compiler
const _PV_G = RepliBuild.JLCSIRGenerator
const _PV_W = RepliBuild.Wrapper
const _PV_J = RepliBuild.JITManager
const _PV_DP = RepliBuild.DWARFParser
const _PV_LIB = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
const _PV_DUMP = read(joinpath(@__DIR__, "fixtures", "pure_virtual.readelf.txt"), String)

# What a library built from that unit, with every declared member defined,
# has in its symbol table: no pure virtual among them but `Base::tag`, the one
# that is defined as well. An abstract class has its base-object constructor
# only (clang emits no `C1` for one).
const _PV_SYMBOLS = Dict{String,Any}[Dict{String,Any}("mangled" => m, "demangled" => d) for (m, d) in (
    ("_ZN2pv6AnimalC2Ei", "pv::Animal::Animal(int)"),
    ("_ZN2pv6AnimalD0Ev", "pv::Animal::~Animal()"), ("_ZN2pv6AnimalD1Ev", "pv::Animal::~Animal()"),
    ("_ZNK2pv6Animal4legsEv", "pv::Animal::legs() const"),
    ("_ZN2pv3DogC1Ev", "pv::Dog::Dog()"),
    ("_ZNK2pv3Dog4nameEv", "pv::Dog::name() const"),
    ("_ZNK2pv3Dog5speedEd", "pv::Dog::speed(double) const"),
    ("_ZN2pv5TokenC2Ev", "pv::Token::Token()"),
    ("_ZN2pv5TokenD0Ev", "pv::Token::~Token()"), ("_ZN2pv5TokenD1Ev", "pv::Token::~Token()"),
    ("_ZN2pv5ShapeC2Ev", "pv::Shape::Shape()"),
    ("_ZN2pv4PolyC2Ei", "pv::Poly::Poly(int)"),
    ("_ZNK2pv4Poly5sidesEv", "pv::Poly::sides() const"),
    ("_ZN2pv6SquareC1Ei", "pv::Square::Square(int)"),
    ("_ZNK2pv6Square4areaEv", "pv::Square::area() const"),
    ("_ZNK2pv4Base3tagEv", "pv::Base::tag() const"),
)]

const _PV_PURE = ["_ZNK2pv6Animal4nameEv", "_ZNK2pv6Animal4spanEv", "_ZNK2pv6Animal5speedEd",
                  "_ZNK2pv6Animal5weighENS_4SpanE", "_ZNK2pv5Shape4areaEv", "_ZNK2pv5Shape5sidesEv",
                  "_ZNK2pv5VRoot3whoEv"]

# The metadata the build assembles from the dump and those symbols: the
# parser's tables, the abstract records, the pure virtuals' rows in front of
# the symbols' rows (`Compiler.extract_compilation_metadata` does the same).
function _pv_metadata(; dump = _PV_DUMP, mark = true)
    (fns, structs, _, _, rabi) = with_logger(NullLogger()) do
        _PV_C.parse_dwarf_dump(dump)
    end
    mark && _PV_C._mark_abstract_records!(rabi, structs)
    plan = _PV_C._vtable_only_plan(structs, rabi, _PV_SYMBOLS, fns)
    functions = Any[]
    append!(functions, _PV_C._vtable_only_function_rows(plan, fns))
    for row in _PV_C.parse_function_signatures(_PV_SYMBOLS)
        key = haskey(fns, row["mangled"]) ? row["mangled"] : _PV_C._structor_dwarf_key(row["mangled"], fns)
        key === nothing && error("no DIE for $(row["mangled"]): the fixture and the symbol list have drifted")
        info = fns[key]
        row["return_type"] = Dict{String,Any}("c_type" => info["c_type"], "julia_type" => info["julia_type"],
                                              "size" => info["size"])
        row["parameters"] = Any[Dict{String,Any}(p) for p in info["parameters"]]
        row["return_type_source"] = row["parameters_source"] = "dwarf"
        row["is_vararg"] = false
        row["is_noexcept"] = false
        push!(functions, row)
    end
    return Dict{String,Any}("functions" => functions, "struct_definitions" => structs, "record_abi" => rabi,
                            "globals" => Dict{String,Any}(), "language" => "c++", "vtable_methods" => plan)
end

const _PV_NO_VTABLES = _PV_DP.VtableInfo(Dict{String,_PV_DP.ClassInfo}(), Dict{String,UInt64}(), Dict{String,UInt64}())

# The thunk `sym`'s definition in a generated module, or "".
function _pv_thunk(ir, sym)
    i = findfirst("func.func @$(sym)_thunk(", ir)
    i === nothing && return ""
    j = findnext("\n}\n", ir, last(i))
    return ir[first(i):(j === nothing ? lastindex(ir) : last(j))]
end

function _pv_config(dir)
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "pvirt"
    root = "$(escape_string(dir))"
    [link]
    enable_lto = false
    [wrap]
    language = "cpp"
    [types]
    strictness = "warn"
    allow_unknown_structs = true
    [cache]
    enabled = false
    """)
    return RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml"))
end

# The wrapper text and the thunk set the generator derives from `meta`.
function _pv_generate(meta)
    dir = mktempdir()
    cfg = _pv_config(dir)
    res = redirect_stdout(devnull) do
        with_logger(NullLogger()) do
            _PV_W.generate_introspective_module_cpp(cfg, _PV_LIB, meta, "Pvirt", _PV_W.create_type_registry(cfg), true)
        end
    end
    manifest = RepliBuild._read_json(joinpath(RepliBuild.ConfigurationManager.get_output_path(cfg), "thunk_manifest.json"))
    return (String(first(res)), Set{String}(String.(manifest["function_thunks"])))
end
_pv_body(code, name) = (m = match(Regex("^function $name\\(.*?\\n^end", "ms"), code); m === nothing ? "" : String(m.match))
_pv_defined(code, name) = occursin(Regex("^function $name\\(", "m"), code)

@testset "Pure virtual member functions" begin

    @testset "the parser records whether a member is pure, and its vtable slot" begin
        (_, _, _, _, ra) = _PV_C.parse_dwarf_dump(_PV_DUMP)
        rows(name) = Dict(String(r["symbol"]) => r for r in ra[name]["methods"])
        animal = rows("Animal")
        checked = 0
        for (sym, pure, slot) in (("_ZNK2pv6Animal4nameEv", true, 2), ("_ZNK2pv6Animal4legsEv", false, 3),
                                  ("_ZNK2pv6Animal5speedEd", true, 4), ("_ZNK2pv6Animal4spanEv", true, 5),
                                  ("_ZNK2pv6Animal5weighENS_4SpanE", true, 6), ("_ZN2pv6Animal3logEPKcz", true, 7),
                                  ("_ZNK2pv6Animal6secretEv", true, 8), ("_ZNK2pv6Animal6hiddenEv", true, 9),
                                  ("_ZN2pv6AnimalD4Ev", false, 0))
            @test (sym, haskey(animal, sym)) == (sym, true)
            haskey(animal, sym) || continue
            @test (sym, animal[sym]["pure"], get(animal[sym], "vtable_slot", nothing)) == (sym, pure, slot)
            @test (sym, animal[sym]["virtual"]) == (sym, true)
            checked += 1
        end
        @test checked == 9
        # A member that is not virtual has no slot and is not pure.
        ctor = animal["_ZN2pv6AnimalC4Ei"]
        @test ctor["virtual"] == false && ctor["pure"] == false && !haskey(ctor, "vtable_slot")
        # An override is virtual and not pure, at the slot of what it overrides.
        dog = rows("Dog")
        @test (dog["_ZNK2pv3Dog4nameEv"]["pure"], dog["_ZNK2pv3Dog4nameEv"]["vtable_slot"]) == (false, 2)
        # A pure virtual destructor is pure too.
        @test rows("Token")["_ZN2pv5TokenD4Ev"]["pure"] == true
        # Accessibility as before: written, else the record's default.
        @test animal["_ZNK2pv6Animal6secretEv"]["accessibility"] == "protected"
        @test animal["_ZNK2pv6Animal6hiddenEv"]["accessibility"] == "private"
        @test animal["_ZNK2pv6Animal4nameEv"]["accessibility"] == "public"
        # The constructor rows carry neither key.
        @test all(r -> !haskey(r, "pure") && !haskey(r, "vtable_slot"), ra["Animal"]["constructors"])

        # `DW_OP_lit<n>` says what `DW_OP_constu: n` says; an expression that is
        # not one constant is no slot.
        line = "DW_AT_vtable_elem_location: 2 byte block: 10 4 \t(DW_OP_constu: 4)"
        @test occursin(line, _PV_DUMP)
        lit = replace(_PV_DUMP, line => "DW_AT_vtable_elem_location: 1 byte block: 34 \t(DW_OP_lit4)")
        @test lit != _PV_DUMP
        speed(dump) = only(r for r in _PV_C.parse_dwarf_dump(dump)[5]["Animal"]["methods"]
                           if r["symbol"] == "_ZNK2pv6Animal5speedEd")
        @test speed(lit)["vtable_slot"] == 4
        odd = replace(_PV_DUMP, line => "DW_AT_vtable_elem_location: 3 byte block: 10 4 22 \t(DW_OP_constu: 4; DW_OP_plus)")
        @test !haskey(speed(odd), "vtable_slot") && speed(odd)["pure"] == true
    end

    @testset "which records a pure virtual leaves abstract" begin
        (_, structs, _, _, ra) = _PV_C.parse_dwarf_dump(_PV_DUMP)
        _PV_C._mark_abstract_records!(ra, structs)
        left(name) = get(get(ra, name, Dict{String,Any}()), "pure_virtuals", nothing)
        @test left("Animal") == ["pv::Animal::hidden() const", "pv::Animal::log(char const*, ...)",
                                 "pv::Animal::name() const", "pv::Animal::secret() const", "pv::Animal::span() const",
                                 "pv::Animal::speed(double) const", "pv::Animal::weigh(pv::Span) const"]
        # Its own, and the one it inherits and does not declare again.
        @test left("Shape") == ["pv::Shape::area() const", "pv::Shape::sides() const"]
        @test left("Poly") == ["pv::Shape::area() const"]
        # A pure virtual destructor makes its own class abstract and no other.
        @test left("Token") == ["pv::Token::~Token()"]
        # Defined as well, it is still pure.
        @test left("Base") == ["pv::Base::tag() const"]
        @test left("VRoot") == ["pv::VRoot::who() const"]
        # Every pure virtual overridden: not abstract.
        @test left("Dog") === nothing && left("Square") === nothing && left("VImpl") === nothing
        # Through a virtual base another branch can supply the overrider
        # (`VJoin` is not abstract: `VImpl::who` is `VRoot::who`'s final
        # overrider). Nothing is claimed along such a path, so `VLeft`, which
        # IS abstract, is not called so either: its constructor is refused by
        # the virtual-base rule (`_structor_bindings`).
        @test left("VJoin") === nothing && left("VLeft") === nothing

        # Hand-made hierarchies, with a demangler that knows these names.
        names = Dict("f_A" => "A::f()", "g_A" => "A::g(int) const", "f_B" => "B::f()", "g_B" => "B::g(int)",
                     "dA" => "A::~A()", "dB" => "B::~B()", "h_C" => "C::h()", "f_E" => "E::f()", "x_F" => "F::x()")
        dm(s) = get(names, s, nothing)
        row(sym, pure) = Dict{String,Any}("symbol" => sym, "pure" => pure, "virtual" => true, "accessibility" => "public")
        base(t; virtual = false) = Dict{String,Any}("type" => t, "offset" => 0, "virtual" => virtual)
        rec(bases...; kw...) = merge(Dict{String,Any}("kind" => "struct", "byte_size" => "0x8", "members" => Any[],
                                                      "base_classes" => Any[bases...]),
                                     Dict{String,Any}(String(k) => v for (k, v) in kw))
        function marked(structs, rows)
            abi = Dict{String,Any}(k => Dict{String,Any}("methods" => v) for (k, v) in rows)
            _PV_C._mark_abstract_records!(abi, structs; demangle = dm)
            return Dict(k => get(v, "pure_virtuals", nothing) for (k, v) in abi)
        end
        # B overrides f and declares a `g` that is another function (no `const`).
        got = marked(Dict{String,Any}("A" => rec(), "B" => rec(base("A"))),
                     Dict("A" => Any[row("f_A", true), row("g_A", true), row("dA", true)],
                          "B" => Any[row("f_B", false), row("g_B", false), row("dB", false)]))
        @test got["A"] == ["A::f()", "A::g(int) const", "A::~A()"]
        @test got["B"] == ["A::g(int) const"]             # never the destructor
        # B declares f pure again: it is B's own.
        got = marked(Dict{String,Any}("A" => rec(), "B" => rec(base("A"))),
                     Dict("A" => Any[row("f_A", true)], "B" => Any[row("f_B", true)]))
        @test got["B"] == ["B::f()"]
        # A base the metadata does not describe, a toolchain record and a name
        # with several definitions say nothing; a record with no rows of its
        # own still inherits.
        got = marked(Dict{String,Any}("A" => rec(), "C" => rec(base("A"), base("Gone")),
                                      "T" => rec(; toolchain = true), "D" => rec(base("T")),
                                      "K" => rec(; conflict = true), "L" => rec(base("K"))),
                     Dict("A" => Any[row("f_A", true)], "C" => Any[row("h_C", false)], "T" => Any[row("f_E", true)],
                          "D" => Any[row("x_F", false)], "K" => Any[row("f_B", true)], "L" => Any[row("x_F", false)]))
        @test got["C"] == ["A::f()"]
        @test got["T"] === nothing && got["D"] === nothing && got["K"] === nothing && got["L"] === nothing
        # A member whose name cannot be read might be the overrider: nothing
        # inherited is claimed then.
        got = marked(Dict{String,Any}("A" => rec(), "B" => rec(base("A"))),
                     Dict("A" => Any[row("f_A", true)], "B" => Any[row("unreadable", false)]))
        @test got["A"] == ["A::f()"] && got["B"] === nothing
        # ...but only a virtual member can be: one that is not (a member
        # template, whose name opens with a return type) changes nothing.
        got = marked(Dict{String,Any}("A" => rec(), "B" => rec(base("A"))),
                     Dict("A" => Any[row("f_A", true)],
                          "B" => Any[merge(row("unreadable", false), Dict{String,Any}("virtual" => false))]))
        @test got["B"] == ["A::f()"]
        # A bare name two scopes define (`a::N`, `b::N`) holds the rows of both.
        got = marked(Dict{String,Any}("A" => rec(; scopes = Any["a", "b"]), "B" => rec(base("A"))),
                     Dict("A" => Any[row("f_A", true)], "B" => Any[row("x_F", false)]))
        @test got["A"] === nothing && got["B"] === nothing
        # Behind a virtual base: not claimed.
        got = marked(Dict{String,Any}("A" => rec(), "B" => rec(base("A"; virtual = true))),
                     Dict("A" => Any[row("f_A", true)], "B" => Any[row("x_F", false)]))
        @test got["B"] === nothing
        # A cycle in hand-made metadata ends.
        got = marked(Dict{String,Any}("A" => rec(base("B")), "B" => rec(base("A"))),
                     Dict("A" => Any[row("f_A", true)], "B" => Any[row("x_F", false)]))
        @test got["A"] == ["A::f()"]
    end

    @testset "declared minus defined: the rows, and what is passed over with its reason" begin
        (fns, structs, _, _, ra) = _PV_C.parse_dwarf_dump(_PV_DUMP)
        plan = _PV_C._vtable_only_plan(structs, ra, _PV_SYMBOLS, fns)
        @test [r["symbol"] for r in plan["bound"]] == _PV_PURE
        @test [(r["class"], r["slot"]) for r in plan["bound"]] ==
              [("Animal", 2), ("Animal", 5), ("Animal", 4), ("Animal", 6), ("Shape", 2), ("Shape", 3), ("VRoot", 2)]
        skipped = Dict(String(r["symbol"]) => String(r["reason"]) for r in plan["skipped"])
        @test sort!(collect(keys(skipped))) == ["_ZN2pv5TokenD4Ev", "_ZN2pv6Animal3logEPKcz",
                                                "_ZNK2pv6Animal6hiddenEv", "_ZNK2pv6Animal6secretEv"]
        @test startswith(skipped["_ZN2pv5TokenD4Ev"], "a destructor")
        @test startswith(skipped["_ZN2pv6Animal3logEPKcz"], "variadic")
        @test startswith(skipped["_ZNK2pv6Animal6hiddenEv"], "private")
        @test startswith(skipped["_ZNK2pv6Animal6secretEv"], "protected")
        # A pure virtual the library defines has a symbol, and a row from it.
        @test !any(r -> r["symbol"] == "_ZNK2pv4Base3tagEv", vcat(plan["bound"], plan["skipped"]))
        # ...and one that is not pure is no business of this plan, defined or not.
        nothing_defined = _PV_C._vtable_only_plan(structs, ra, Dict{String,Any}[], fns)
        @test [r["symbol"] for r in nothing_defined["bound"]] ==
              vcat(_PV_PURE[1:4], ["_ZNK2pv4Base3tagEv"], _PV_PURE[5:7])       # by class, then symbol
        @test !any(r -> occursin("legs", r["symbol"]), nothing_defined["bound"])

        # No slot, no signature, a symbol that does not demangle, a toolchain
        # record: each is named or passed over, never bound.
        strip_slot = deepcopy(ra)
        for r in strip_slot["Shape"]["methods"]
            delete!(r, "vtable_slot")
        end
        p = _PV_C._vtable_only_plan(structs, strip_slot, _PV_SYMBOLS, fns)
        @test !any(r -> r["class"] == "Shape", p["bound"])
        @test count(r -> r["class"] == "Shape" && occursin("no vtable slot", r["reason"]), p["skipped"]) == 2
        p = _PV_C._vtable_only_plan(structs, ra, _PV_SYMBOLS, filter(kv -> first(kv) != "_ZNK2pv5Shape4areaEv", fns))
        @test only(r["reason"] for r in p["skipped"] if r["symbol"] == "_ZNK2pv5Shape4areaEv") ==
              "the debug information has no signature for it"
        p = _PV_C._vtable_only_plan(structs, ra, _PV_SYMBOLS, fns; demangle = s -> occursin("Shape", s) ? nothing : _PV_C._demangle(s))
        @test count(r -> r["class"] == "Shape" && occursin("does not demangle", r["reason"]), p["skipped"]) == 2
        # A class of one translation unit (an unnamed namespace, a local
        # class) is no part of the interface.
        internal(s) = (d = _PV_C._demangle(s); d === nothing ? nothing :
                       replace(d, "pv::Shape::" => "pv::(anonymous namespace)::Shape::",
                               "pv::VRoot::" => "pv::make()::VRoot::"))
        p = _PV_C._vtable_only_plan(structs, ra, _PV_SYMBOLS, fns; demangle = internal)
        @test !any(r -> r["class"] in ("Shape", "VRoot"), p["bound"])
        @test count(r -> r["class"] in ("Shape", "VRoot") && startswith(r["reason"], "its class has internal linkage"),
                    p["skipped"]) == 3
        @test count(r -> r["class"] == "Animal", p["bound"]) == 4
        # A `(` inside template arguments is a function type, not a scope.
        slot(s) = (d = _PV_C._demangle(s); d === nothing ? nothing : replace(d, "pv::Shape::" => "pv::Slot<void (int)>::"))
        p = _PV_C._vtable_only_plan(structs, ra, _PV_SYMBOLS, fns; demangle = slot)
        @test [r["demangled"] for r in p["bound"] if r["class"] == "Shape"] ==
              ["pv::Slot<void (int)>::area() const", "pv::Slot<void (int)>::sides() const"]
        toolchain = deepcopy(structs)
        toolchain["Shape"]["toolchain"] = true
        p = _PV_C._vtable_only_plan(toolchain, ra, _PV_SYMBOLS, fns)
        @test !any(r -> r["class"] == "Shape", vcat(p["bound"], p["skipped"]))
        # Metadata from before the rows said `pure` plans nothing.
        old = deepcopy(ra)
        for (_, v) in old, r in get(v, "methods", Any[])
            delete!(r, "pure")
        end
        p = _PV_C._vtable_only_plan(structs, old, _PV_SYMBOLS, fns)
        @test isempty(p["bound"]) && isempty(p["skipped"])

        # The function rows.
        rows = Dict(String(r["mangled"]) => r for r in _PV_C._vtable_only_function_rows(plan, fns))
        @test sort!(collect(keys(rows))) == sort(_PV_PURE)
        name = rows["_ZNK2pv6Animal4nameEv"]
        @test (name["name"], name["class"], name["demangled"]) == ("name", "pv::Animal", "pv::Animal::name() const")
        @test name["vtable_only"] == true && name["vtable_slot"] == 2
        @test name["is_method"] == true && name["has_object_pointer"] == true && name["is_vararg"] == false
        @test name["return_type"]["c_type"] == "const char*" && name["return_type"]["julia_type"] == "Cstring"
        @test [String(p["name"]) for p in name["parameters"]] == ["this"]
        @test name["return_type_source"] == "dwarf" && name["parameters_source"] == "dwarf"
        # It claims no export: the coverage tool leaves a row out that says `false`.
        @test !haskey(name, "exported")
        # A declaration's parameters have no names; the types are the DIE's.
        speed = rows["_ZNK2pv6Animal5speedEd"]
        @test [(String(p["name"]), String(p["c_type"])) for p in speed["parameters"]] ==
              [("this", "const Animal*"), ("arg1", "double")]
        @test all(_PV_C._is_vtable_only, values(rows))
        @test !_PV_C._is_vtable_only(Dict{String,Any}("mangled" => "_ZNK2pv4Base3tagEv"))
        @test !_PV_C._is_vtable_only(nothing)
    end

    @testset "the thunk is a vtable call and nothing else" begin
        meta = _pv_metadata()
        all_symbols = Set{String}(String(f["mangled"]) for f in meta["functions"])
        checked = 0
        for needed in (nothing, all_symbols)
            # No class table at all: the slot is the row's own.
            ir = _PV_G.generate_jlcs_ir(_PV_NO_VTABLES, deepcopy(meta); needed_symbols = needed)
            for (sym, cls, slot) in (("_ZNK2pv6Animal4nameEv", "Animal", 2), ("_ZNK2pv6Animal5speedEd", "Animal", 4),
                                     ("_ZNK2pv5Shape4areaEv", "Shape", 2), ("_ZNK2pv5Shape5sidesEv", "Shape", 3),
                                     ("_ZNK2pv5VRoot3whoEv", "VRoot", 2))
                body = _pv_thunk(ir, sym)
                @test (sym, !isempty(body)) == (sym, true)
                @test (sym, occursin("class_name = @$cls, vtable_offset = 0 : i64, slot = $slot : i64, this_offset = 0 : i64", body)) == (sym, true)
                @test (sym, count("\"jlcs.vcall\"", body)) == (sym, 1)
                # An override may throw: the landing pad.
                @test (sym, occursin("may_throw", body)) == (sym, true)
                # Nothing names the symbol: no declaration, no callee.
                @test (sym, occursin("func.func private @$sym(", ir)) == (sym, false)
                @test (sym, occursin("@$sym,", ir) || occursin("@$sym ", ir) || occursin("@$sym(", ir)) == (sym, false)
                @test (sym, occursin("_call", body)) == (sym, false)
                checked += 1
            end
            # A record by value in either direction: no thunk, and the line says why.
            for (sym, what) in (("_ZNK2pv6Animal4spanEv", "returns a record by value"),
                                ("_ZNK2pv6Animal5weighENS_4SpanE", "takes a record by value"))
                @test (sym, _pv_thunk(ir, sym)) == (sym, "")
                @test (sym, occursin("// no thunk for $sym: it is pure virtual and $what", ir)) == (sym, true)
                @test (sym, occursin("@$sym", replace(ir, "// no thunk for $sym" => ""))) == (sym, false)
            end
            # The module calls nothing it does not declare.
            @test _PV_G.undeclared_callees(ir) == String[]
            # A virtual with a symbol is declared and called as it was.
            @test occursin("func.func private @_ZNK2pv4Base3tagEv(", ir)
            @test occursin("func.func private @_ZNK2pv3Dog4nameEv(", ir)
        end
        @test checked == 10

        # Left out of the manifest, a pure virtual gets nothing at all.
        ir = _PV_G.generate_jlcs_ir(_PV_NO_VTABLES, deepcopy(meta); needed_symbols = Set(["_ZNK2pv3Dog4nameEv"]))
        @test !occursin("_ZNK2pv6Animal4nameEv", ir)

        # With the class table llvm-dwarfdump gives, the answer is the same one.
        vm(name, sym, slot) = _PV_DP.VirtualMethod(name, sym, slot, "int", String[])
        animal = _PV_DP.ClassInfo("Animal", 0, String[], [vm("name", "_ZNK2pv6Animal4nameEv", 2), vm("legs", "_ZNK2pv6Animal4legsEv", 3)],
                                  [_PV_DP.MemberInfo("legs_", "int", 8)], 16)
        vt = _PV_DP.VtableInfo(Dict("Animal" => animal), Dict{String,UInt64}(),
                               Dict{String,UInt64}("_ZNK2pv6Animal4legsEv" => UInt64(0x1000)))
        ir = _PV_G.generate_jlcs_ir(vt, deepcopy(meta); needed_symbols = all_symbols)
        @test occursin("class_name = @Animal, vtable_offset = 0 : i64, slot = 2 : i64", _pv_thunk(ir, "_ZNK2pv6Animal4nameEv"))
        @test !occursin("func.func private @_ZNK2pv6Animal4nameEv(", ir)
        @test !occursin("llvm.func @_ZNK2pv6Animal4nameEv(", ir)
        # ...and the virtual that has a symbol dispatches at its own slot.
        @test occursin("slot = 3 : i64", _pv_thunk(ir, "_ZNK2pv6Animal4legsEv"))
    end

    @testset "one verdict for the thunk generator and the wrapper" begin
        meta = _pv_metadata()
        rows = Dict(String(f["mangled"]) => f for f in meta["functions"])
        structs = meta["struct_definitions"]
        rabi = meta["record_abi"]
        checked = 0
        # The reasons do not depend on the convention: a record is a record.
        for abi in (:sysv, :win64, :aapcs64)
            ask(sym) = _PV_G.vtable_only_refusal(rows[sym], structs, rabi; abi = abi)
            for sym in ("_ZNK2pv6Animal4nameEv", "_ZNK2pv6Animal5speedEd", "_ZNK2pv5Shape4areaEv")
                @test (abi, sym, ask(sym)) == (abi, sym, nothing)
            end
            @test (abi, occursin("returns a record by value", something(ask("_ZNK2pv6Animal4spanEv"), ""))) == (abi, true)
            @test (abi, occursin("takes a record by value", something(ask("_ZNK2pv6Animal5weighENS_4SpanE"), ""))) == (abi, true)
            checked += 1
        end
        @test checked == 3
        # A row with no slot (hand-made metadata) and a variadic one.
        noslot = Dict{String,Any}(rows["_ZNK2pv6Animal4nameEv"])
        delete!(noslot, "vtable_slot")
        @test occursin("no vtable slot", something(_PV_G.vtable_only_refusal(noslot, structs, rabi), ""))
        variadic = merge(Dict{String,Any}(rows["_ZNK2pv6Animal4nameEv"]), Dict{String,Any}("is_vararg" => true))
        @test occursin("variadic", something(_PV_G.vtable_only_refusal(variadic, structs, rabi), ""))
        # A by-value record with no layout is the crossing's own refusal.
        opaque = Dict{String,Any}(rows["_ZNK2pv6Animal5weighENS_4SpanE"])
        opaque["parameters"] = Any[opaque["parameters"][1],
                                   Dict{String,Any}("name" => "arg1", "c_type" => "Unseen", "julia_type" => "Any")]
        @test occursin("takes `Unseen` by value", something(_PV_G.vtable_only_refusal(opaque, structs, rabi), ""))
    end

    @testset "the pre-flight does not look a pure virtual up" begin
        meta = _pv_metadata()
        defined = Set{String}(String(s["mangled"]) for s in _PV_SYMBOLS)
        every = Set{String}(String(f["mangled"]) for f in meta["functions"])
        for needed in (nothing, every)
            callees = _PV_J._module_callees(meta, needed === nothing ? nothing : copy(needed))
            @test isempty(intersect(callees, _PV_PURE))
            @test issubset(defined, callees)
            # A library that has exactly its symbols resolves everything asked.
            (bound, unresolved) = _PV_J._preflight_callees(_ -> nothing, callees, defined)
            @test isempty(bound) && isempty(unresolved)
        end
        # The module the generator writes declares nothing a pre-flight missed.
        ir = _PV_G.generate_jlcs_ir(_PV_NO_VTABLES, deepcopy(meta); needed_symbols = every)
        stray = @test_logs min_level = Logging.Warn _PV_J._check_module_declarations(
            ir, defined, Dict{String,Ptr{Cvoid}}(), "libpvirt.so")
        @test isempty(stray)
    end

    @testset "the wrapper binds the thunk, never a ccall" begin
        (code, thunks) = _pv_generate(_pv_metadata())
        checked = 0
        for (name, sym) in (("pv_Animal_name", "_ZNK2pv6Animal4nameEv"), ("pv_Animal_speed", "_ZNK2pv6Animal5speedEd"),
                            ("pv_Shape_area", "_ZNK2pv5Shape4areaEv"), ("pv_Shape_sides", "_ZNK2pv5Shape5sidesEv"),
                            ("pv_VRoot_who", "_ZNK2pv5VRoot3whoEv"))
            b = _pv_body(code, name)
            @test (name, occursin("RepliBuild.JITManager.invoke(\"_mlir_ciface_$(sym)_thunk\"", b)) == (name, true)
            @test (name, occursin("RepliBuild.JITManager.pointer_arg(", b)) == (name, true)
            @test (name, sym in thunks) == (name, true)
            checked += 1
        end
        @test checked == 5
        # No `ccall` names a pure virtual, anywhere in the text.
        for sym in _PV_PURE
            @test (sym, occursin(":$sym,", code) || occursin("var\"$sym\"", code)) == (sym, false)
        end
        # A `const char*` return gets the string pair, as on every path.
        @test _pv_defined(code, "pv_Animal_name") && _pv_defined(code, "pv_Animal_name_ptr")
        @test occursin("\"_mlir_ciface__ZNK2pv6Animal4nameEv_thunk\"", _pv_body(code, "pv_Animal_name_ptr"))
        # The docstring says what the call is.
        @test occursin("Wrapper for `pv::Animal::name() const`\n\n**Pure virtual.**", code)
        @test !occursin("Wrapper for `pv::Animal::legs() const`\n\n**Pure virtual.**", code)
        # A record by value: a refusal that says so, and no thunk asked for.
        for (name, sym, what) in (("pv_Animal_span", "_ZNK2pv6Animal4spanEv", "returns a record by value"),
                                  ("pv_Animal_weigh", "_ZNK2pv6Animal5weighENS_4SpanE", "takes a record by value"))
            b = _pv_body(code, name)
            @test (name, occursin("Base.error(\"ABI Safety Trap: cannot call '$name'", b)) == (name, true)
            @test (name, occursin("It is pure virtual and $what", b)) == (name, true)
            @test (name, occursin("invoke", b)) == (name, false)
            @test (name, sym in thunks) == (name, false)
        end
        # Private, protected, variadic, and the pure virtual destructor: no function.
        for name in ("pv_Animal_secret", "pv_Animal_hidden", "pv_Animal_log")
            @test (name, _pv_defined(code, name)) == (name, false)
        end
        @test !occursin("_ZN2pv5TokenD4Ev", code)
        # The destructors the library defines are bound as they were.
        @test occursin("\"_mlir_ciface__ZN2pv5TokenD1Ev_thunk\"", _pv_body(code, "pv_Token_destroy_Token"))
        # A virtual with a symbol is unchanged: its own thunk, by its own name.
        @test occursin("\"_mlir_ciface__ZNK2pv6Animal4legsEv_thunk\"", _pv_body(code, "pv_Animal_legs"))
        @test occursin("\"_mlir_ciface__ZNK2pv4Base3tagEv_thunk\"", _pv_body(code, "pv_Base_tag"))
        @test !occursin("Wrapper for `pv::Base::tag() const`\n\n**Pure virtual.**", code)
        # The text parses.
        @test !any(a -> a isa Expr && a.head in (:incomplete, :error), Meta.parseall(code).args)

        # Nothing else about a row decides it: hand-made metadata that calls a
        # pure virtual `noexcept` with a `nounwind` definition, and has no
        # `methods` rows to say it is virtual, would be ccall-safe.
        meta = _pv_metadata()
        for f in meta["functions"]
            _PV_C._is_vtable_only(f) || continue
            f["is_noexcept"] = true
            f["nounwind"] = true
        end
        for (_, v) in meta["record_abi"]
            delete!(v, "methods")
        end
        (bare, bare_thunks) = _pv_generate(meta)
        @test occursin("\"_mlir_ciface__ZNK2pv6Animal5speedEd_thunk\"", _pv_body(bare, "pv_Animal_speed"))
        @test "_ZNK2pv6Animal5speedEd" in bare_thunks
        @test !occursin(":_ZNK2pv6Animal5speedEd,", bare)
        # ...nor a row that says it is variadic, which the build never makes:
        # the varargs path calls by name, so it is a refusal.
        meta = _pv_metadata()
        for f in meta["functions"]
            f["mangled"] == "_ZNK2pv6Animal5speedEd" && (f["is_vararg"] = true)
        end
        (odd, odd_thunks) = _pv_generate(meta)
        @test occursin("ABI Safety Trap: cannot call 'pv_Animal_speed'", _pv_body(odd, "pv_Animal_speed"))
        @test occursin("variadic", _pv_body(odd, "pv_Animal_speed"))
        @test !occursin("_ZNK2pv6Animal5speedEd", replace(odd, r"Mangled symbol: `[^`]*`" => ""))
        @test !("_ZNK2pv6Animal5speedEd" in odd_thunks)
    end

    @testset "no constructor of an abstract class is bound" begin
        (code, thunks) = _pv_generate(_pv_metadata())
        checked = 0
        for (name, sym, cls, why) in (("pv_Animal_Animal", "_ZN2pv6AnimalC2Ei", "pv::Animal", "pv::Animal::hidden() const"),
                                      ("pv_Shape_Shape", "_ZN2pv5ShapeC2Ev", "pv::Shape", "pv::Shape::area() const"),
                                      ("pv_Poly_Poly", "_ZN2pv4PolyC2Ei", "pv::Poly", "pv::Shape::area() const"),
                                      ("pv_Token_Token", "_ZN2pv5TokenC2Ev", "pv::Token", "pv::Token::~Token()"))
            b = _pv_body(code, name)
            @test (name, occursin("Base.error(\"ABI Safety Trap: cannot call '$name'", b)) == (name, true)
            @test (name, occursin("`$cls` is an abstract class: `$why`", b)) == (name, true)
            @test (name, occursin("invoke", b)) == (name, false)
            @test (name, sym in thunks) == (name, false)
            checked += 1
        end
        @test checked == 4
        # The refusal keeps the constructor's parameters.
        @test occursin(r"^function pv_Poly_Poly\(this::\w+, arg1::\w+\)$"m, code)
        # A class that defines every pure virtual is constructed as before.
        @test occursin("\"_mlir_ciface__ZN2pv6SquareC1Ei_thunk\"", _pv_body(code, "pv_Square_Square"))
        @test occursin("\"_mlir_ciface__ZN2pv3DogC1Ev_thunk\"", _pv_body(code, "pv_Dog_Dog"))
        @test "_ZN2pv6SquareC1Ei" in thunks && "_ZN2pv3DogC1Ev" in thunks
        # Destructors are not constructors.
        @test occursin("invoke", _pv_body(code, "pv_Animal_destroy_Animal"))

        # Negative check: metadata that does not say which records are
        # abstract binds the base-object constructor, as the wrapper did.
        (old, old_thunks) = _pv_generate(_pv_metadata(; mark = false))
        @test occursin("\"_mlir_ciface__ZN2pv6AnimalC2Ei_thunk\"", _pv_body(old, "pv_Animal_Animal"))
        @test "_ZN2pv6AnimalC2Ei" in old_thunks
    end

    @testset "the AOT stage is handed the same set" begin
        dir = mktempdir()
        cfg_path = joinpath(dir, "replibuild.toml")
        _pv_config(dir)
        write(cfg_path, read(cfg_path, String) * "\n[compile]\naot_thunks = true\n")
        cfg = RepliBuild.ConfigurationManager.load_config(cfg_path)
        @test cfg.compile.aot_thunks
        handed = Set{String}[]
        stage = needed -> (push!(handed, Set{String}(needed)); false)     # writes no library
        @test_throws _PV_W.AOTStageFailed redirect_stdout(devnull) do
            with_logger(NullLogger()) do
                _PV_W.generate_introspective_module_cpp(cfg, _PV_LIB, _pv_metadata(), "Pvirt", _PV_W.create_type_registry(cfg),
                                                        true, joinpath(dir, "libpvirt_thunks." * Libdl.dlext); aot_stage = stage)
            end
        end
        @test length(handed) == 1
        if length(handed) == 1
            set = only(handed)
            @test issubset(["_ZNK2pv6Animal4nameEv", "_ZNK2pv6Animal5speedEd", "_ZNK2pv5Shape4areaEv"], set)
            @test !("_ZNK2pv6Animal4spanEv" in set) && !("_ZN2pv6AnimalC2Ei" in set)
        end
    end

    @testset "negative check: a row that is not told apart names a symbol no library has" begin
        # The predicate, neutered for the rows this file makes (a method more
        # specific than the real one's untyped argument); deleted afterwards.
        meta = _pv_metadata()
        every = Set{String}(String(f["mangled"]) for f in meta["functions"])
        @eval _PV_C _is_vtable_only(func::Dict{String,Any}) = false
        neutered = Base.invokelatest(which, _PV_C._is_vtable_only, Tuple{Dict{String,Any}})
        try
            ir = Base.invokelatest(_PV_G.generate_jlcs_ir, _PV_NO_VTABLES, deepcopy(meta); needed_symbols = every)
            # The thunk declares the pure virtual as a callee and calls it by name.
            @test occursin("func.func private @_ZNK2pv6Animal4nameEv(", ir)
            @test occursin("callee = @_ZNK2pv6Animal4nameEv", _pv_thunk(ir, "_ZNK2pv6Animal4nameEv"))
            @test !occursin("jlcs.vcall", _pv_thunk(ir, "_ZNK2pv6Animal4nameEv"))
            # ...and the pre-flight asks the library for it.
            @test issubset(_PV_PURE, Base.invokelatest(_PV_J._module_callees, meta, nothing))
        finally
            Base.delete_method(neutered)
        end
        @test Base.invokelatest(_PV_C._is_vtable_only, meta["functions"][1])
        ir = Base.invokelatest(_PV_G.generate_jlcs_ir, _PV_NO_VTABLES, deepcopy(meta); needed_symbols = every)
        @test !occursin("func.func private @_ZNK2pv6Animal4nameEv(", ir)
    end

    @testset "a library with no pure virtual: the same text with and without the new keys" begin
        # `Dog` and `Square` alone: nothing pure in the metadata's functions.
        function concrete(with_keys)
            meta = _pv_metadata()
            filter!(f -> String(get(f, "class", "")) in ("pv::Dog", "pv::Square") && !_PV_C._is_vtable_only(f),
                    meta["functions"])
            for k in ("Animal", "Shape", "Poly", "Token", "Base", "VRoot", "VLeft", "VImpl", "VJoin")
                delete!(meta["struct_definitions"], k)
                delete!(meta["record_abi"], k)
            end
            for k in ("Dog", "Square")
                meta["struct_definitions"][k]["base_classes"] = Any[]
            end
            if !with_keys
                delete!(meta, "vtable_methods")
                for (_, v) in meta["record_abi"], r in get(v, "methods", Any[])
                    delete!(r, "pure")
                    delete!(r, "vtable_slot")
                end
            end
            return meta
        end
        (with, with_thunks) = _pv_generate(concrete(true))
        (without, without_thunks) = _pv_generate(concrete(false))
        stamp(text) = replace(text, r"^# Generated:.*$"m => "", r"\"generated_at\" => \"[^\"]*\"" => "")
        @test _pv_defined(with, "pv_Dog_name")
        @test stamp(with) == stamp(without)
        @test with_thunks == without_thunks
        @test !occursin("Pure virtual", with) && !occursin("abstract class", with)
    end
end
