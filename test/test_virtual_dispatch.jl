# =============================================================================
# Which methods a thunk dispatches through the vtable (library-free, CI-safe)
#
# A wrapper `Dog_name(p)` must do what `p->name()` does: read the object's
# vtable, so a `Puppy*` runs `Puppy::name`. The thunk does that with
# `jlcs.vcall`; a call by symbol is `p->Dog::name()`. Three ways a virtual
# method was still called by symbol until 2026-11-17:
#
#   * A class that adds no data member to its base (`struct Dog : Animal` with
#     only methods; an interface's implementation) was skipped by the vcall
#     producer and got no `jlcs.type_info`, because the test for "is this a
#     class worth describing" was "does it have data members".
#     `Dog_name(puppy)` returned "dog".
#   * A class a later compile unit only DECLARES (a unit that defines some of
#     its members, under clang's default debug info) lost its definition in
#     `DWARFParser`'s class table, where the last DIE of a name replaced the
#     others: no size, so no type_info and no vcall, whatever its members.
#   * A `noexcept` virtual with scalar arguments is ccall-safe, so the wrapper
#     called its symbol with a plain `ccall` and no thunk was involved at all.
#
# The thunk module is generated here from the class table `DWARFParser` hands
# it, the class table from llvm-dwarfdump text, and the wrapper from synthetic
# metadata. test_virtual_classes_live.jl (devtests) builds the same hierarchy
# from C++, in two files, and calls it.
# =============================================================================

using Test
using RepliBuild
using Libdl
using Logging

const _VD_DP = RepliBuild.DWARFParser
const _VD_G = RepliBuild.JLCSIRGenerator
const _VD_W = RepliBuild.Wrapper
const _VD_LIB = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))

_vd_vm(name, sym, slot; ret = "int", params = String[]) = _VD_DP.VirtualMethod(name, sym, slot, ret, params)
_vd_class(name, size, bases, members, methods) =
    _VD_DP.ClassInfo(name, 0, String[bases...], _VD_DP.VirtualMethod[methods...], _VD_DP.MemberInfo[members...], size,
                     zeros(Int, length(bases)), fill(false, length(bases)))

# struct Animal { int legs_; virtual ~Animal(); virtual int legs() const; virtual int rank() const noexcept; };
# struct Dog : Animal { int legs() const override; int rank() const noexcept override; int fetch(int); };
# struct Puppy : Dog { int legs() const override; };      Dog and Puppy add no data member.
# struct Tag {};                                          an empty class
# struct Seen;  with one virtual method declared          a declaration (no size)
const _VD_CLASSES = Dict{String,_VD_DP.ClassInfo}(
    "Animal" => _vd_class("Animal", 16, [], [_VD_DP.MemberInfo("_vptr\$Animal", "int (**)()", 0),
                                             _VD_DP.MemberInfo("legs_", "int", 8)],
                          [_vd_vm("~Animal", "_ZN6AnimalD4Ev", 0; ret = "void"), _vd_vm("legs", "_ZNK6Animal4legsEv", 2),
                           _vd_vm("rank", "_ZNK6Animal4rankEv", 3)]),
    "Dog" => _vd_class("Dog", 16, ["Animal"], [],
                       [_vd_vm("legs", "_ZNK3Dog4legsEv", 2), _vd_vm("rank", "_ZNK3Dog4rankEv", 3),
                        _vd_vm("~Dog", "_ZN3DogD4Ev", 0; ret = "void")]),
    "Puppy" => _vd_class("Puppy", 16, ["Dog"], [], [_vd_vm("legs", "_ZNK5Puppy4legsEv", 2)]),
    "Tag" => _vd_class("Tag", 1, [], [], []),
    "Seen" => _vd_class("Seen", 0, [], [], [_vd_vm("peek", "_ZNK4Seen4peekEv", 2)]),
)
const _VD_METHODS = ["_ZNK6Animal4legsEv", "_ZNK6Animal4rankEv", "_ZNK3Dog4legsEv", "_ZNK3Dog4rankEv",
                     "_ZNK5Puppy4legsEv", "_ZNK4Seen4peekEv"]
const _VD_VT = _VD_DP.VtableInfo(_VD_CLASSES, Dict{String,UInt64}(),
                                 Dict{String,UInt64}(m => UInt64(0x1000 + 16i) for (i, m) in enumerate(_VD_METHODS)))

_vd_p(n, jt, ct) = Dict{String,Any}("name" => n, "julia_type" => jt, "c_type" => ct)
const _VD_INT = Dict{String,Any}("c_type" => "int", "julia_type" => "Cint", "size" => 4)
const _VD_VOID = Dict{String,Any}("c_type" => "void", "julia_type" => "Cvoid", "size" => 0)
function _vd_row(mangled, demangled, cls; params = (), ret = _VD_INT, noexcept = false)
    r = only(RepliBuild.Compiler.parse_function_signatures(
        [Dict{String,Any}("mangled" => mangled, "demangled" => demangled)]))
    r["parameters"] = Any[_vd_p("this", "Ptr{$cls}", "$cls*"), params...]
    r["return_type"] = ret
    r["is_noexcept"] = noexcept
    r["is_vararg"] = false
    r["has_object_pointer"] = true
    return r
end
const _VD_FUNCTIONS = Any[
    _vd_row("_ZNK6Animal4legsEv", "Animal::legs() const", "Animal"),
    _vd_row("_ZNK6Animal4rankEv", "Animal::rank() const", "Animal"; noexcept = true),
    _vd_row("_ZN6AnimalD1Ev", "Animal::~Animal()", "Animal"; ret = _VD_VOID),
    _vd_row("_ZNK3Dog4legsEv", "Dog::legs() const", "Dog"),
    _vd_row("_ZNK3Dog4rankEv", "Dog::rank() const", "Dog"; noexcept = true),
    _vd_row("_ZN3Dog5fetchEi", "Dog::fetch(int)", "Dog"; params = (_vd_p("n", "Cint", "int"),)),
    _vd_row("_ZN3DogD1Ev", "Dog::~Dog()", "Dog"; ret = _VD_VOID),
    _vd_row("_ZNK5Puppy4legsEv", "Puppy::legs() const", "Puppy"),
    _vd_row("_ZNK4Seen4peekEv", "Seen::peek() const", "Seen"),
]

_vd_mem(n, jt, ct, off, size) = Dict{String,Any}("name" => n, "julia_type" => jt, "c_type" => ct,
                                                 "offset" => off, "size" => size)
_vd_derived(base) = Dict{String,Any}("kind" => "struct", "byte_size" => "0x10", "members" => Any[],
    "base_classes" => Any[Dict{String,Any}("type" => base, "offset" => 0, "virtual" => false, "accessibility" => "public")])
const _VD_STRUCTS = Dict{String,Any}(
    "Animal" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x10", "members" => Any[
        _vd_mem("_vptr\$Animal", "Ptr{Cvoid}", "function_ptr(int)**", "0x0", 8), _vd_mem("legs_", "Cint", "int", "0x8", 4)]),
    "Dog" => _vd_derived("Animal"),
    "Puppy" => _vd_derived("Dog"),
)

_vd_meta() = Dict{String,Any}("functions" => deepcopy(_VD_FUNCTIONS), "struct_definitions" => deepcopy(_VD_STRUCTS),
                              "language" => "c++")

# The thunk `sym`'s body in a generated module.
function _vd_thunk(ir, sym)
    i = findfirst("func.func @$(sym)_thunk(", ir)
    i === nothing && return ""
    j = findnext("\n}\n", ir, last(i))
    return ir[first(i):(j === nothing ? lastindex(ir) : last(j))]
end

@testset "Virtual dispatch" begin

    @testset "a class with no data member of its own dispatches through its vtable" begin
        checked = 0
        for needed in (nothing, Set{String}(String(f["mangled"]) for f in _VD_FUNCTIONS))
            ir = _VD_G.generate_jlcs_ir(_VD_VT, _vd_meta(); needed_symbols = needed)
            # Each class's own slot, read through the object's vptr at 0.
            for (sym, cls, slot) in (("_ZNK6Animal4legsEv", "Animal", 2), ("_ZNK3Dog4legsEv", "Dog", 2),
                                     ("_ZNK3Dog4rankEv", "Dog", 3), ("_ZNK5Puppy4legsEv", "Puppy", 2))
                body = _vd_thunk(ir, sym)
                @test (sym, !isempty(body)) == (sym, true)
                @test (sym, occursin("\"jlcs.vcall\"(%val_1) { class_name = @$cls, vtable_offset = 0 : i64, " *
                                     "slot = $slot : i64, this_offset = 0 : i64", body)) == (sym, true)
                # ...and never by symbol.
                @test (sym, occursin("callee = @$sym", body)) == (sym, false)
                checked += 1
            end
            # A thrower gets the landing pad, a `noexcept` method a plain call.
            @test occursin("may_throw", _vd_thunk(ir, "_ZNK3Dog4legsEv"))
            @test !occursin("may_throw", _vd_thunk(ir, "_ZNK3Dog4rankEv"))
            # A non-virtual method of the same class is still a direct call.
            @test occursin("callee = @_ZN3Dog5fetchEi", _vd_thunk(ir, "_ZN3Dog5fetchEi"))
            @test !occursin("jlcs.vcall", _vd_thunk(ir, "_ZN3Dog5fetchEi"))
            # Destructors stay direct calls on the exact class.
            @test occursin("jlcs.dtor_call @_ZN3DogD1Ev(", _vd_thunk(ir, "_ZN3DogD1Ev"))
            @test !occursin("jlcs.vcall", _vd_thunk(ir, "_ZN3DogD1Ev"))
        end
        @test checked == 8
    end

    @testset "every class a vcall names has a type_info" begin
        ir = _VD_G.generate_jlcs_ir(_VD_VT, _vd_meta())
        described = Set(String(m.captures[1]) for m in eachmatch(r"jlcs\.type_info \"([^\"]+)\"", ir))
        named = Set(String(m.captures[1]) for m in eachmatch(r"class_name = @(\w+)", ir))
        @test named == Set(["Animal", "Dog", "Puppy"])
        @test issubset(named, described)
        # A class that adds no member is described by the vptr it is
        # dispatched through: an empty field list does not parse.
        @test occursin("jlcs.type_info \"Dog\", !jlcs.c_struct<\"Dog\", [!llvm.ptr], [[0 : i64]], packed = false>, \"Animal\"", ir)
        @test occursin("jlcs.type_info \"Puppy\", !jlcs.c_struct<\"Puppy\", [!llvm.ptr], [[0 : i64]], packed = false>, \"Dog\"", ir)
        @test !occursin("[], [[]]", ir)
        # A class with members is described as it was.
        @test occursin("jlcs.type_info \"Animal\", !jlcs.c_struct<\"Animal\", [!llvm.ptr, i32], [[0 : i64, 8 : i64]], packed = false>", ir)
        # An empty tag class and a declaration get neither a type_info nor a
        # vcall: the declaration's method is called by symbol, as before.
        @test !("Tag" in described) && !("Seen" in described)
        @test occursin("callee = @_ZNK4Seen4peekEv", _vd_thunk(ir, "_ZNK4Seen4peekEv"))
        # The predicate itself.
        T = _VD_G._has_type_info
        @test T(_VD_CLASSES["Animal"]) && T(_VD_CLASSES["Dog"]) && T(_VD_CLASSES["Puppy"])
        @test !T(_VD_CLASSES["Tag"]) && !T(_VD_CLASSES["Seen"])
    end

    @testset "negative check: with the data-member test, Dog's methods are called by symbol" begin
        # The old rule, as a method more specific than the real one's untyped
        # argument, so every call takes it; deleting it restores the real one.
        @eval _VD_G _has_type_info(class_info::$(_VD_DP.ClassInfo)) =
            class_info.size != 0 && !isempty(class_info.members)
        old = Base.invokelatest(which, _VD_G._has_type_info, Tuple{_VD_DP.ClassInfo})
        try
            ir = Base.invokelatest(_VD_G.generate_jlcs_ir, _VD_VT, _vd_meta())
            @test occursin("callee = @_ZNK3Dog4legsEv", _vd_thunk(ir, "_ZNK3Dog4legsEv"))
            @test !occursin("jlcs.vcall", _vd_thunk(ir, "_ZNK3Dog4legsEv"))
            @test !occursin("jlcs.type_info \"Dog\"", ir)
            @test occursin("jlcs.vcall", _vd_thunk(ir, "_ZNK6Animal4legsEv"))    # a class with members always was
        finally
            Base.delete_method(old)
        end
        @test Base.invokelatest(_VD_G._has_type_info, _VD_CLASSES["Dog"])
    end

    @testset "a declaration in another unit does not erase the class" begin
        # llvm-dwarfdump of a two-file library, trimmed (clang 23, default
        # `-g`): `struct Mid : Base { int m; int f() const override; int g()
        # const noexcept override; int extra() const; }`. The unit that defines
        # Mid's constructor describes the class. A unit that only defines two
        # of its members holds Mid as a DECLARATION listing those two. Classes
        # are keyed by name and the last DIE replaced the others, so with that
        # unit second Mid had no size and no `f`, got no type_info, and both
        # its virtual methods were called by symbol (`Mid_f(leaf)` ran
        # `Mid::f`). Which unit comes second follows the object files' names.
        definition = """
        0x0000000c: DW_TAG_compile_unit
                      DW_AT_producer	("clang version 23.1.5")
                      DW_AT_name	("/src/a.cpp")

        0x0000012a:   DW_TAG_structure_type
                        DW_AT_containing_type	(0x0000002b "Base")
                        DW_AT_calling_convention	(DW_CC_pass_by_reference)
                        DW_AT_name	("Mid")
                        DW_AT_byte_size	(0x10)

        0x00000134:     DW_TAG_inheritance
                          DW_AT_type	(0x0000002b "Base")
                          DW_AT_data_member_location	(0x00)

        0x0000013a:     DW_TAG_member
                          DW_AT_name	("m")
                          DW_AT_type	(0x0000000000000400 "int")
                          DW_AT_data_member_location	(0x0c)

        0x0000014e:     DW_TAG_subprogram
                          DW_AT_linkage_name	("_ZNK3Mid1fEv")
                          DW_AT_name	("f")
                          DW_AT_type	(0x0000000000000400 "int")
                          DW_AT_virtuality	(DW_VIRTUALITY_virtual)
                          DW_AT_vtable_elem_location	(DW_OP_constu 0x2)
                          DW_AT_declaration	(true)
                          DW_AT_external	(true)
                          DW_AT_containing_type	(0x0000012a "Mid")

        0x0000015f:       DW_TAG_formal_parameter
                            DW_AT_type	(0x000001a9 "const Mid *")
                            DW_AT_artificial	(true)

        0x00000164:       NULL

        0x00000165:     DW_TAG_subprogram
                          DW_AT_linkage_name	("_ZNK3Mid1gEv")
                          DW_AT_name	("g")
                          DW_AT_type	(0x0000000000000400 "int")
                          DW_AT_virtuality	(DW_VIRTUALITY_virtual)
                          DW_AT_vtable_elem_location	(DW_OP_constu 0x3)
                          DW_AT_declaration	(true)
                          DW_AT_external	(true)
                          DW_AT_containing_type	(0x0000012a "Mid")

        0x00000176:       DW_TAG_formal_parameter
                            DW_AT_type	(0x000001a9 "const Mid *")
                            DW_AT_artificial	(true)

        0x0000017b:       NULL

        0x0000017c:     NULL
        """
        declaration = """
        0x000003c0: DW_TAG_compile_unit
                      DW_AT_producer	("clang version 23.1.5")
                      DW_AT_name	("/src/z_late.cpp")

        0x000003d7:   DW_TAG_structure_type
                        DW_AT_name	("Mid")
                        DW_AT_declaration	(true)

        0x000003d9:     DW_TAG_subprogram
                          DW_AT_linkage_name	("_ZNK3Mid5extraEv")
                          DW_AT_name	("extra")
                          DW_AT_type	(0x00000400 "int")
                          DW_AT_declaration	(true)
                          DW_AT_external	(true)

        0x000003e2:       DW_TAG_formal_parameter
                            DW_AT_type	(0x00000404 "const Mid *")
                            DW_AT_artificial	(true)

        0x000003e7:       NULL

        0x000003e8:     DW_TAG_subprogram
                          DW_AT_linkage_name	("_ZNK3Mid1gEv")
                          DW_AT_name	("g")
                          DW_AT_type	(0x00000400 "int")
                          DW_AT_virtuality	(DW_VIRTUALITY_virtual)
                          DW_AT_vtable_elem_location	(DW_OP_constu 0x3)
                          DW_AT_declaration	(true)
                          DW_AT_external	(true)
                          DW_AT_containing_type	(0x000003d7 "Mid")

        0x000003f9:       DW_TAG_formal_parameter
                            DW_AT_type	(0x00000404 "const Mid *")
                            DW_AT_artificial	(true)

        0x000003fe:       NULL

        0x000003ff:     NULL
        """
        checked = 0
        for (order, dump) in (("definition first", definition * declaration), ("declaration first", declaration * definition))
            mid = get(_VD_DP.parse_dwarf_output(dump), "Mid", nothing)
            @test (order, mid !== nothing) == (order, true)
            mid === nothing && continue
            @test (order, mid.size) == (order, 16)
            @test (order, [m.name for m in mid.members]) == (order, ["m"])
            @test (order, mid.base_classes) == (order, ["Base"])
            @test (order, sort!([(m.name, m.mangled_name, m.slot) for m in mid.virtual_methods])) ==
                  (order, [("f", "_ZNK3Mid1fEv", 2), ("g", "_ZNK3Mid1gEv", 3)])
            @test (order, _VD_G._has_type_info(mid)) == (order, true)
            checked += 1
        end
        @test checked == 2
        # Two definitions of one name (the class in a header, seen by two
        # units): one class, each method once.
        twice = get(_VD_DP.parse_dwarf_output(definition * definition), "Mid", nothing)
        @test twice !== nothing && twice.size == 16 && length(twice.virtual_methods) == 2
    end

    @testset "a noexcept virtual goes through a thunk, never a ccall" begin
        function generate(record_abi)
            dir = mktempdir()
            write(joinpath(dir, "replibuild.toml"), """
            [project]
            name = "vdispatch"
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
            cfg = RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml"))
            meta = _vd_meta()
            meta["globals"] = Dict{String,Any}()
            meta["record_abi"] = record_abi
            res = redirect_stdout(devnull) do
                with_logger(NullLogger()) do
                    _VD_W.generate_introspective_module_cpp(cfg, _VD_LIB, meta, "Vdispatch",
                                                            _VD_W.create_type_registry(cfg), true)
                end
            end
            code = res isa Tuple ? String(first(res)) : String(res)
            manifest = RepliBuild._read_json(joinpath(RepliBuild.ConfigurationManager.get_output_path(cfg), "thunk_manifest.json"))
            return (code, Set{String}(String.(manifest["function_thunks"])))
        end
        body(code, name) = (m = match(Regex("^function $name\\(.*?\\n^end", "ms"), code); m === nothing ? "" : String(m.match))
        # The rows the build writes for every member function a record declares.
        row(sym, virtual) = Dict{String,Any}("symbol" => sym, "virtual" => virtual, "artificial" => false,
                                             "deleted" => false, "accessibility" => "public")
        rows = Dict{String,Any}(
            "Animal" => Dict{String,Any}("methods" => Any[row("_ZN6AnimalD4Ev", true), row("_ZNK6Animal4legsEv", true),
                                                          row("_ZNK6Animal4rankEv", true)]),
            "Dog" => Dict{String,Any}("methods" => Any[row("_ZN3Dog5fetchEi", false), row("_ZNK3Dog4legsEv", true),
                                                       row("_ZNK3Dog4rankEv", true)]))
        (code, thunks) = generate(rows)
        for (name, sym) in (("Animal_rank", "_ZNK6Animal4rankEv"), ("Dog_rank", "_ZNK3Dog4rankEv"))
            b = body(code, name)
            @test (name, occursin("RepliBuild.JITManager.invoke(\"_mlir_ciface_$(sym)_thunk\", Cint, this)", b)) == (name, true)
            @test (name, occursin("ccall(", b)) == (name, false)
            @test (name, sym in thunks) == (name, true)
        end
        # Metadata without the rows cannot say which methods are virtual: the
        # `noexcept` ones keep their `ccall` until the package is rebuilt.
        (old_code, old_thunks) = generate(Dict{String,Any}())
        @test occursin("ccall((:_ZNK3Dog4rankEv, LIBRARY_PATH), Cint, (Ptr{Dog},), this)", body(old_code, "Dog_rank"))
        @test !("_ZNK3Dog4rankEv" in old_thunks)
        # A thrower was a thunk either way.
        @test "_ZNK3Dog4legsEv" in thunks && "_ZNK3Dog4legsEv" in old_thunks
    end
end
