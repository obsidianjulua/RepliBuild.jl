# =============================================================================
# Which constructor and destructor a C++ wrapper binds (library-free, CI-safe)
#
# Itanium gives a constructor up to two functions and a destructor three: the
# complete-object ones (C1, D1), the base-object ones (C2, D2) and the deleting
# destructor (D0). For a class with no virtual base C1 and C2 (D1 and D2) are
# one function. For a class WITH one they are two: C2 and D2 take a VTT after
# `this` and leave the virtual base alone.
#
# Until 2026-11-17 the wrapper bound whichever came last in the metadata's
# order, which was C2 and D2 for every class:
#
#   * `VBottom_VBottom(p)` and `VLeft_VLeft(p)` ran the base-object constructor
#     with no VTT and segfaulted.
#   * `VBottom_destroy_VBottom(this, vtt)` was D2.
#   * `VBottom_delete_VBottom(this, vtt)` gave the deleting destructor a VTT
#     it does not take: every variant of a member shared one parameter list in
#     the metadata, the one of the definition the build read last.
#
# Three places, each driven here on synthetic input shaped as the build writes
# it: the DWARF parser (a base-object variant's VTT is its own), the choice
# (`_structor_bindings`), and the emitted wrapper. test_virtual_classes_live.jl
# (devtests) builds the same shapes from C++ and calls them.
# =============================================================================

using Test
using RepliBuild
using Libdl
using Logging

const _SV_W = RepliBuild.Wrapper
const _SV_C = RepliBuild.Compiler
const _SV_LIB = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))

_sv_hex(n) = string(n, base = 16)

# A subprogram DIE as GNU readelf prints clang's: `params` are (name, type
# offset, artificial); `spec` makes it a definition of the declaration there,
# and `sym` is the DIE's own DW_AT_linkage_name.
function _sv_subprogram(level, off; sym = nothing, name = nothing, spec = nothing, declaration = false,
                        fn_artificial = false, params = ())
    io = IOBuffer()
    println(io, " <$level><$(_sv_hex(off))>: Abbrev Number: 5 (DW_TAG_subprogram)")
    if !declaration
        println(io, "    <$(_sv_hex(off + 1))>   DW_AT_low_pc      : 0x$(_sv_hex(0x4000 + off))")
        println(io, "    <$(_sv_hex(off + 2))>   DW_AT_object_pointer: <0x$(_sv_hex(off + 8))>")
    end
    sym === nothing || println(io, "    <$(_sv_hex(off + 3))>   DW_AT_linkage_name: (indexed string: 0x1): $sym")
    name === nothing || println(io, "    <$(_sv_hex(off + 4))>   DW_AT_name        : $name")
    declaration && println(io, "    <$(_sv_hex(off + 5))>   DW_AT_declaration : 1")
    fn_artificial && println(io, "    <$(_sv_hex(off + 5))>   DW_AT_artificial  : 1")
    declaration && println(io, "    <$(_sv_hex(off + 5))>   DW_AT_external    : 1")
    spec === nothing || println(io, "    <$(_sv_hex(off + 6))>   DW_AT_specification: <0x$(_sv_hex(spec))>")
    for (i, (pname, ptype, artificial)) in enumerate(params)
        po = off + 8 * i
        println(io, " <$(level + 1)><$(_sv_hex(po))>: Abbrev Number: 6 (DW_TAG_formal_parameter)")
        pname === nothing || println(io, "    <$(_sv_hex(po + 1))>   DW_AT_name        : $pname")
        println(io, "    <$(_sv_hex(po + 2))>   DW_AT_type        : <0x$(_sv_hex(ptype))>")
        artificial && println(io, "    <$(_sv_hex(po + 3))>   DW_AT_artificial  : 1")
    end
    println(io, " <$(level + 1)><$(_sv_hex(off + 0x38))>: Abbrev Number: 0")
    return String(take!(io))
end

# `struct Base0 { virtual ~Base0(); }; struct Vb : virtual Base0 { Vb(int n); };`
# (Vb's destructor implicit), and `struct Pod { int v; };` whose implicit copy
# constructor a caller in the library made clang emit. Offsets: Vb 0x100, its
# declarations 0x110 (C4) and 0x150 (D4); Pod 0x200, its copy constructor's
# declaration 0x210; types from 0x900. `definitions` are the variant DIEs in
# the order the dump lists them.
const _SV_THIS_VB, _SV_INT, _SV_VTT, _SV_THIS_POD, _SV_POD_REF = 0x900, 0x910, 0x920, 0x940, 0x950
function _sv_dump(definitions; declarations_first = true)
    classes = """
     <1><100>: Abbrev Number: 2 (DW_TAG_structure_type)
        <101>   DW_AT_name        : Vb
        <102>   DW_AT_byte_size   : 32
    """ *
    _sv_subprogram(2, 0x110; sym = "_ZN2VbC4Ei", name = "Vb", declaration = true,
                   params = ((nothing, _SV_THIS_VB, true), (nothing, _SV_INT, false))) *
    _sv_subprogram(2, 0x150; sym = "_ZN2VbD4Ev", name = "~Vb", declaration = true, fn_artificial = true,
                   params = ((nothing, _SV_THIS_VB, true),)) *
    """
     <2><1f0>: Abbrev Number: 0
     <1><200>: Abbrev Number: 2 (DW_TAG_structure_type)
        <201>   DW_AT_name        : Pod
        <202>   DW_AT_byte_size   : 4
    """ *
    _sv_subprogram(2, 0x210; sym = "_ZN3PodC4ERKS_", name = "Pod", declaration = true, fn_artificial = true,
                   params = ((nothing, _SV_THIS_POD, true), (nothing, _SV_POD_REF, false))) *
    """
     <2><2f0>: Abbrev Number: 0
    """
    types = """
     <1><900>: Abbrev Number: 7 (DW_TAG_pointer_type)
        <901>   DW_AT_type        : <0x100>
     <1><910>: Abbrev Number: 8 (DW_TAG_base_type)
        <911>   DW_AT_name        : int
        <912>   DW_AT_byte_size   : 4
     <1><920>: Abbrev Number: 7 (DW_TAG_pointer_type)
        <921>   DW_AT_type        : <0x930>
     <1><930>: Abbrev Number: 9 (DW_TAG_pointer_type)
     <1><940>: Abbrev Number: 7 (DW_TAG_pointer_type)
        <941>   DW_AT_type        : <0x200>
     <1><950>: Abbrev Number: 10 (DW_TAG_reference_type)
        <951>   DW_AT_type        : <0x960>
     <1><960>: Abbrev Number: 11 (DW_TAG_const_type)
        <961>   DW_AT_type        : <0x200>
     <1><9f0>: Abbrev Number: 0
    """
    head = """
     <0><b>: Abbrev Number: 1 (DW_TAG_compile_unit)
        <c>   DW_AT_producer    : clang version 23.1.5
    """
    defs = join(definitions)
    return head * (declarations_first ? classes * defs : defs * classes) * types
end

const _SV_THIS = ("this", _SV_THIS_VB, true)
const _SV_DEFS = Dict(
    "C2" => _sv_subprogram(1, 0x300; sym = "_ZN2VbC2Ei", spec = 0x110,
                           params = (_SV_THIS, ("vtt", _SV_VTT, true), ("n", _SV_INT, false))),
    "C1" => _sv_subprogram(1, 0x340; sym = "_ZN2VbC1Ei", spec = 0x110, params = (_SV_THIS, ("n", _SV_INT, false))),
    "D1" => _sv_subprogram(1, 0x380; sym = "_ZN2VbD1Ev", spec = 0x150, params = (_SV_THIS,)),
    "D0" => _sv_subprogram(1, 0x3c0; sym = "_ZN2VbD0Ev", spec = 0x150, params = (_SV_THIS,)),
    "D2" => _sv_subprogram(1, 0x400; sym = "_ZN2VbD2Ev", spec = 0x150, params = (_SV_THIS, ("vtt", _SV_VTT, true))),
    # An implicit copy constructor: clang marks its unnamed source parameter
    # artificial too. It is declared, so it is no VTT.
    "PodC2" => _sv_subprogram(1, 0x440; sym = "_ZN3PodC2ERKS_", spec = 0x210,
                              params = (("this", _SV_THIS_POD, true), (nothing, _SV_POD_REF, true))),
)

# ── Synthetic metadata, as the build writes it ───────────────────────────────

_sv_p(n, jt, ct; artificial = false) =
    (d = Dict{String,Any}("name" => n, "julia_type" => jt, "c_type" => ct); artificial && (d["artificial"] = true); d)
const _SV_VOID = Dict{String,Any}("c_type" => "void", "julia_type" => "Cvoid", "size" => 0)
function _sv_row(mangled, demangled, params...)
    r = only(_SV_C.parse_function_signatures([Dict{String,Any}("mangled" => mangled, "demangled" => demangled)]))
    r["parameters"] = Any[params...]
    r["return_type"] = _SV_VOID
    r["is_noexcept"] = false
    r["is_vararg"] = false
    r["has_object_pointer"] = true
    return r
end
_sv_this(cls) = _sv_p("this", "Ptr{$cls}", "$cls*")
_sv_vtt(; marked = true) = _sv_p("vtt", "Ptr{Ptr{Cvoid}}", "void**"; artificial = marked)
_sv_n() = _sv_p("n", "Cint", "int")
_sv_mem(n, jt, ct, off, size) = Dict{String,Any}("name" => n, "julia_type" => jt, "c_type" => ct,
                                                 "offset" => off, "size" => size)
function _sv_rec(size, members...; base = nothing, virtual = false)
    r = Dict{String,Any}("kind" => "struct", "byte_size" => "0x" * string(size, base = 16), "members" => Any[members...])
    base === nothing || (r["base_classes"] = Any[Dict{String,Any}(
        "type" => base, "offset" => 0, "virtual" => virtual, "accessibility" => "public",
        (virtual ? ("vbase_vtable_offset" => -24,) : ())...)])
    return r
end
_sv_vptr(cls) = _sv_mem("_vptr\$$cls", "Ptr{Cvoid}", "function_ptr(int)**", "0x0", 8)
_sv_int(n, off) = _sv_mem(n, "Cint", "int", off, 4)

const _SV_STRUCTS = Dict{String,Any}(
    "Base0" => _sv_rec(16, _sv_vptr("Base0"), _sv_int("tag", "0x8")),
    # No virtual base anywhere.
    "Pod"   => _sv_rec(16, _sv_vptr("Pod"), _sv_int("v", "0x8")),
    "Homed" => _sv_rec(16, _sv_vptr("Homed"), _sv_int("v", "0x8")),
    # A virtual base, directly (Vb, Abs, Old, Named) and through a base (Deep).
    "Vb"    => _sv_rec(32, _sv_vptr("Vb"), _sv_int("n", "0x8"); base = "Base0", virtual = true),
    "Deep"  => _sv_rec(40, _sv_int("d", "0xc"); base = "Vb"),
    "Abs"   => _sv_rec(32, _sv_vptr("Abs"), _sv_int("a", "0x8"); base = "Base0", virtual = true),
    "Old"   => _sv_rec(32, _sv_vptr("Old"), _sv_int("o", "0x8"); base = "Base0", virtual = true),
    "Named" => _sv_rec(32, _sv_vptr("Named"), _sv_int("o", "0x8"); base = "Base0", virtual = true),
)

const _SV_FUNCTIONS = Any[
    # Every variant, the base-object ones listed LAST, as the build orders them.
    _sv_row("_ZN3PodC1Ei", "Pod::Pod(int)", _sv_this("Pod"), _sv_n()),
    _sv_row("_ZN3PodC2Ei", "Pod::Pod(int)", _sv_this("Pod"), _sv_n()),
    _sv_row("_ZN3PodD0Ev", "Pod::~Pod()", _sv_this("Pod")),
    _sv_row("_ZN3PodD1Ev", "Pod::~Pod()", _sv_this("Pod")),
    _sv_row("_ZN3PodD2Ev", "Pod::~Pod()", _sv_this("Pod")),
    # The library has only the base-object variants (an abstract class's
    # constructor, an inline destructor), and no virtual base: they are it.
    _sv_row("_ZN5HomedC2Ei", "Homed::Homed(int)", _sv_this("Homed"), _sv_n()),
    _sv_row("_ZN5HomedD2Ev", "Homed::~Homed()", _sv_this("Homed")),
    # A virtual base: C2 and D2 carry the VTT, marked by the build.
    _sv_row("_ZN2VbC1Ei", "Vb::Vb(int)", _sv_this("Vb"), _sv_n()),
    _sv_row("_ZN2VbC2Ei", "Vb::Vb(int)", _sv_this("Vb"), _sv_vtt(), _sv_n()),
    _sv_row("_ZN2VbD0Ev", "Vb::~Vb()", _sv_this("Vb")),
    _sv_row("_ZN2VbD1Ev", "Vb::~Vb()", _sv_this("Vb")),
    _sv_row("_ZN2VbD2Ev", "Vb::~Vb()", _sv_this("Vb"), _sv_vtt()),
    # A virtual base through a base, and only C1/D1 for it.
    _sv_row("_ZN4DeepC1Ev", "Deep::Deep()", _sv_this("Deep")),
    _sv_row("_ZN4DeepD1Ev", "Deep::~Deep()", _sv_this("Deep")),
    # A virtual base and only the base-object variants: nothing builds or ends
    # a whole Abs.
    _sv_row("_ZN3AbsC2Ei", "Abs::Abs(int)", _sv_this("Abs"), _sv_vtt(), _sv_n()),
    _sv_row("_ZN3AbsD2Ev", "Abs::~Abs()", _sv_this("Abs"), _sv_vtt()),
    # Metadata from before each variant had its own list: every variant holds
    # the list of the base-object definition, unmarked.
    _sv_row("_ZN3OldC1Ei", "Old::Old(int)", _sv_this("Old"), _sv_vtt(marked = false), _sv_n()),
    _sv_row("_ZN3OldC2Ei", "Old::Old(int)", _sv_this("Old"), _sv_vtt(marked = false), _sv_n()),
    _sv_row("_ZN3OldD0Ev", "Old::~Old()", _sv_this("Old"), _sv_vtt(marked = false)),
    _sv_row("_ZN3OldD1Ev", "Old::~Old()", _sv_this("Old"), _sv_vtt(marked = false)),
    _sv_row("_ZN3OldD2Ev", "Old::~Old()", _sv_this("Old"), _sv_vtt(marked = false)),
    # A constructor the programmer gave a `void** vtt`: declared, so it stays.
    _sv_row("_ZN5NamedC1EPPv", "Named::Named(void**)", _sv_this("Named"), _sv_vtt(marked = false)),
]

function _sv_generate(functions = _SV_FUNCTIONS; name = "structors")
    dir = mktempdir()
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "$name"
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
    meta = Dict{String,Any}("functions" => deepcopy(functions), "globals" => Dict{String,Any}(),
                            "struct_definitions" => deepcopy(_SV_STRUCTS))
    res = nothing
    report = mktemp() do path, io
        redirect_stdout(io) do
            with_logger(NullLogger()) do
                res = Base.invokelatest(_SV_W.generate_introspective_module_cpp, cfg, _SV_LIB, meta, "Structors",
                                        _SV_W.create_type_registry(cfg), true)
            end
        end
        flush(io)
        read(path, String)
    end
    code = res isa Tuple ? String(first(res)) : String(res)
    manifest = RepliBuild._read_json(joinpath(RepliBuild.ConfigurationManager.get_output_path(cfg), "thunk_manifest.json"))
    return (code, Set{String}(String.(manifest["function_thunks"])), report)
end

# The definition of `name` with exactly `sig` as its parameter list, or "".
function _sv_fn(code, name, sig)
    i = findfirst("function $name($sig)\n", code)
    i === nothing && return ""
    j = findnext("\nend", code, last(i))
    return code[first(i):(j === nothing ? lastindex(code) : last(j))]
end
_sv_sigs(code, name) = [String(m.captures[1]) for m in eachmatch(Regex("^function $name\\(([^\\n]*)\\)\$", "m"), code)]

@testset "Constructor and destructor variants" begin

    @testset "DWARF: a base-object variant's VTT is its own" begin
        # Every order clang can list the variants in, and the declarations
        # before and after the definitions. The unified key (what C1, D1 and
        # D0 are looked up by) is the declared list in each.
        orders = (["C2", "C1", "D1", "D0", "D2", "PodC2"],     # x_inherit's: D2 last, C1 last
                  ["C1", "C2", "D2", "D1", "D0", "PodC2"],     # C2 last, D0 last
                  ["PodC2", "D2", "D0", "D1", "C2", "C1"])
        checked = 0
        for order in orders, declarations_first in (true, false)
            dump = _sv_dump([_SV_DEFS[k] for k in order]; declarations_first)
            # No arity warning: a definition with a VTT is not a dropped parameter.
            rt = @test_logs min_level = Logging.Warn _SV_C.parse_dwarf_dump(dump)[1]
            names(key) = haskey(rt, key) ? [p["name"] for p in rt[key]["parameters"]] : nothing
            where_ = (order, declarations_first)
            @test (where_, names("_ZN2VbC4Ei")) == (where_, ["this", "n"])
            @test (where_, names("_ZN2VbD4Ev")) == (where_, ["this"])
            @test (where_, names("_ZN2VbC2Ei")) == (where_, ["this", "vtt", "n"])
            @test (where_, names("_ZN2VbD2Ev")) == (where_, ["this", "vtt"])
            for key in ("_ZN2VbC2Ei", "_ZN2VbD2Ev")
                haskey(rt, key) || continue
                ps = rt[key]["parameters"]
                @test (where_, key, [get(p, "artificial", false) for p in ps][1:2]) == (where_, key, [false, true])
                @test (where_, key, ps[2]["c_type"]) == (where_, key, "void**")
                checked += 1
            end
            # The unified list carries no mark and counts its own positions.
            @test (where_, [get(p, "artificial", false) for p in rt["_ZN2VbC4Ei"]["parameters"]]) == (where_, [false, false])
            @test (where_, [p["position"] for p in rt["_ZN2VbC4Ei"]["parameters"]]) == (where_, [0, 1])
            # The other variants have no entry: they are looked up by the
            # unified key, as every constructor and destructor is.
            for key in ("_ZN2VbC1Ei", "_ZN2VbD1Ev", "_ZN2VbD0Ev")
                @test (where_, key, haskey(rt, key)) == (where_, key, false)
                @test (where_, key, _SV_C._structor_dwarf_key(key, rt)) ==
                      (where_, key, replace(key, r"C1E" => "C4E", r"D[01]E" => "D4E"))
            end
            # An implicit copy constructor's source parameter is artificial in
            # its definition and declared: it stays, under the unified key only.
            @test (where_, names("_ZN3PodC4ERKS_")) == (where_, ["this", "arg1"])
            @test (where_, haskey(rt, "_ZN3PodC2ERKS_")) == (where_, false)
        end
        @test checked == 2 * length(orders) * 2
    end

    @testset "the variant a symbol names" begin
        row(m, d) = _sv_row(m, d)
        V = _SV_W._structor_variant
        @test V(row("_ZN2VbC1Ei", "Vb::Vb(int)")) == (:ctor, '1')
        @test V(row("_ZN2VbC2Ei", "Vb::Vb(int)")) == (:ctor, '2')
        @test V(row("_ZN2VbD0Ev", "Vb::~Vb()")) == (:dtor, '0')
        @test V(row("_ZN2VbD1Ev", "Vb::~Vb()")) == (:dtor, '1')
        @test V(row("_ZN2VbD2Ev", "Vb::~Vb()")) == (:dtor, '2')
        @test V(row("_ZN2ns3BoxIdEC2ERKS1_", "ns::Box<double>::Box(ns::Box<double> const&)")) == (:ctor, '2')
        # Not a constructor or destructor.
        @test V(row("_ZNK2Vb3getEv", "Vb::get() const")) === nothing
        @test V(row("_Z7make_vbi", "make_vb(int)")) === nothing
        # A `C1E` inside an identifier is not the marker: the class is
        # `C1Engine`, and its base-object constructor is still C2.
        @test V(row("_ZN8C1EngineC2Ev", "C1Engine::C1Engine()")) == (:ctor, '2')
        @test V(row("_ZN8C1EngineC1Ev", "C1Engine::C1Engine()")) == (:ctor, '1')
        # ...nor one in a parameter's name (`ns::C2`).
        @test V(row("_ZN3FooC1EN2ns2C2E", "Foo::Foo(ns::C2)")) == (:ctor, '1')
        # A default-argument function stands under the constructor's name and
        # is no variant of it.
        shim = row("_ZN18replibuild_default1cI2VbJEEEvPT_", "Vb::Vb()")
        shim["default_arguments"] = Dict{String,Any}("of" => "_ZN2VbC4Ei", "omitted" => 1)
        @test V(shim) === nothing
    end

    @testset "which classes have a virtual base" begin
        H(c) = _SV_W._has_virtual_base(c, _SV_STRUCTS)
        @test H("Vb") === true
        @test H("Deep") === true                 # through its base
        @test H("ns::Vb") === true               # a qualified class, a bare key
        @test H("Pod") === false
        @test H("Base0") === false
        @test H("Nowhere") === nothing           # no definition: the metadata cannot say
        withbase = merge(_SV_STRUCTS, Dict{String,Any}("Ext" => _sv_rec(8; base = "std::exception")))
        @test _SV_W._has_virtual_base("Ext", withbase) === nothing    # an undescribed base
    end

    @testset "the declared parameters and the VTT" begin
        P(f, kind) = (r = _SV_W._structor_parameters(f, kind); ([p["name"] for p in r[1]], [p["name"] for p in r[2]]))
        f(m) = only(x for x in _SV_FUNCTIONS if x["mangled"] == m)
        @test P(f("_ZN2VbC2Ei"), :ctor) == (["this", "n"], ["vtt"])        # marked by the build
        @test P(f("_ZN2VbD2Ev"), :dtor) == (["this"], ["vtt"])
        @test P(f("_ZN2VbC1Ei"), :ctor) == (["this", "n"], String[])
        @test P(f("_ZN2VbD1Ev"), :dtor) == (["this"], String[])
        # Unmarked (older metadata): a destructor declares nothing, and a
        # constructor's list is one longer than its demangled name's.
        @test P(f("_ZN3OldD1Ev"), :dtor) == (["this"], ["vtt"])
        @test P(f("_ZN3OldC1Ei"), :ctor) == (["this", "n"], ["vtt"])
        # A declared `void** vtt` is a parameter.
        @test P(f("_ZN5NamedC1EPPv"), :ctor) == (["this", "vtt"], String[])
    end

    @testset "the choice" begin
        (skipped, refused) = _SV_W._structor_bindings(_SV_FUNCTIONS, _SV_STRUCTS)
        @test skipped == Set(["_ZN3PodC2Ei", "_ZN3PodD2Ev", "_ZN2VbC2Ei", "_ZN2VbD2Ev", "_ZN3OldC2Ei", "_ZN3OldD2Ev"])
        @test Set(keys(refused)) == Set(["_ZN3AbsC2Ei", "_ZN3AbsD2Ev", "_ZN3OldC1Ei", "_ZN3OldD0Ev", "_ZN3OldD1Ev"])
        @test occursin("only the base-object constructor of `Abs`", refused["_ZN3AbsC2Ei"])
        @test occursin("has a virtual base", refused["_ZN3AbsC2Ei"])
        @test occursin("only the base-object destructor of `Abs`", refused["_ZN3AbsD2Ev"])
        for m in ("_ZN3OldC1Ei", "_ZN3OldD0Ev", "_ZN3OldD1Ev")
            @test (m, occursin("Rebuild the package", refused[m])) == (m, true)
        end
        # A class the metadata does not describe: its base-object destructor is
        # judged by its own list.
        orphan(params...) = Any[_sv_row("_ZN6OrphanD2Ev", "Orphan::~Orphan()", params...)]
        @test isempty(_SV_W._structor_bindings(orphan(_sv_this("Orphan")), _SV_STRUCTS)[2])
        @test haskey(_SV_W._structor_bindings(orphan(_sv_this("Orphan"), _sv_vtt()), _SV_STRUCTS)[2], "_ZN6OrphanD2Ev")
    end

    (code, thunks, report) = _sv_generate()

    @testset "the module parses" begin
        bad = Any[]
        walk(x) = x isa Expr && (x.head in (:incomplete, :error) ? push!(bad, x) : foreach(walk, x.args))
        walk(Meta.parseall(code))
        @test isempty(bad)
    end

    @testset "the complete-object constructor and destructor are the bindings" begin
        # One definition per name, with the declared parameters, on C1 and D1.
        want = (("Pod_Pod", "this::Any, n::Integer", "_ZN3PodC1Ei"), ("Pod_destroy_Pod", "this::Any", "_ZN3PodD1Ev"),
                ("Pod_delete_Pod", "this::Any", "_ZN3PodD0Ev"),
                ("Vb_Vb", "this::Any, n::Integer", "_ZN2VbC1Ei"), ("Vb_destroy_Vb", "this::Any", "_ZN2VbD1Ev"),
                ("Vb_delete_Vb", "this::Any", "_ZN2VbD0Ev"),
                ("Deep_Deep", "this::Any", "_ZN4DeepC1Ev"), ("Deep_destroy_Deep", "this::Any", "_ZN4DeepD1Ev"),
                # No virtual base and no C1/D1 in the library: C2 and D2 are it.
                ("Homed_Homed", "this::Any, n::Integer", "_ZN5HomedC2Ei"), ("Homed_destroy_Homed", "this::Any", "_ZN5HomedD2Ev"),
                # A declared `void** vtt` is kept.
                ("Named_Named", "this::Any, vtt::Any", "_ZN5NamedC1EPPv"))
        checked = 0
        for (name, sig, sym) in want
            @test (name, _sv_sigs(code, name)) == (name, [sig])
            body = _sv_fn(code, name, sig)
            @test (name, occursin("\"_mlir_ciface_$(sym)_thunk\"", body)) == (name, true)
            @test (name, sym in thunks) == (name, true)
            checked += 1
        end
        @test checked == length(want)
        # The base-object variant beside a complete-object one is in no call
        # and gets no thunk.
        for sym in ("_ZN3PodC2Ei", "_ZN3PodD2Ev", "_ZN2VbC2Ei", "_ZN2VbD2Ev", "_ZN3OldC2Ei", "_ZN3OldD2Ev")
            @test (sym, occursin(sym, code)) == (sym, false)
            @test (sym, sym in thunks) == (sym, false)
        end
    end

    @testset "no wrapper takes a VTT" begin
        vtt_sigs = [m.match for m in eachmatch(r"^function \w+\([^\n]*\bvtt\b[^\n]*\)$"m, code)]
        @test vtt_sigs == ["function Named_Named(this::Any, vtt::Any)"]     # the declared one
    end

    @testset "a base-object variant alone, with a virtual base, is a refusal" begin
        ctor = _sv_fn(code, "Abs_Abs", "this::Any, n::Integer")
        dtor = _sv_fn(code, "Abs_destroy_Abs", "this::Any")
        @test occursin("Base.error(\"ABI Safety Trap: cannot call 'Abs_Abs' (Abs::Abs(int)).", ctor)
        @test occursin("only the base-object constructor of `Abs` (`_ZN3AbsC2Ei`)", ctor)
        @test occursin("takes a VTT", ctor) && occursin("virtual base", ctor)
        @test occursin("Base.error(\"ABI Safety Trap: cannot call 'Abs_destroy_Abs' (Abs::~Abs()).", dtor)
        @test _sv_sigs(code, "Abs_Abs") == ["this::Any, n::Integer"]
        @test _sv_sigs(code, "Abs_destroy_Abs") == ["this::Any"]
        @test !("_ZN3AbsC2Ei" in thunks) && !("_ZN3AbsD2Ev" in thunks)
        @test !occursin("_mlir_ciface__ZN3Abs", code)
    end

    @testset "metadata that gives every variant the VTT says rebuild" begin
        for (name, sig) in (("Old_Old", "this::Any, n::Integer"), ("Old_destroy_Old", "this::Any"),
                            ("Old_delete_Old", "this::Any"))
            body = _sv_fn(code, name, sig)
            @test (name, _sv_sigs(code, name)) == (name, [sig])
            @test (name, occursin("ABI Safety Trap: cannot call '$name'", body)) == (name, true)
            @test (name, occursin("Rebuild the package (delete .replibuild_cache/project_hash)", body)) == (name, true)
        end
        @test !any(m -> startswith(m, "_ZN3Old"), thunks)
    end

    @testset "the wrap report names the refused classes" begin
        @test occursin("wrap: 5 constructor/destructor wrapper(s) of 2 class(es) are refusals", report)
        @test occursin(": Abs, Old", report)
    end

    @testset "negative check: without the choice, the last variant is bound again" begin
        # The old behaviour: with nothing chosen both variants are emitted and
        # dedup keeps the last, C2 and D2, the VTT in the signature. The
        # neutering method takes every call the generator makes (it is more
        # specific than the real one's untyped arguments), and deleting it
        # afterwards leaves the real one as it was.
        @eval _SV_W _structor_bindings(functions::AbstractVector, dwarf_structs::AbstractDict) =
            (Set{String}(), Dict{String,String}())
        neutered = Base.invokelatest(which, _SV_W._structor_bindings, Tuple{Vector{Any},Dict{String,Any}})
        try
            (code2, thunks2, _) = _sv_generate()
            @test "this::Any, vtt::Any, n::Integer" in _sv_sigs(code2, "Vb_Vb")
            @test occursin("_ZN2VbC2Ei", _sv_fn(code2, "Vb_Vb", "this::Any, vtt::Any, n::Integer"))
            @test "this::Any, vtt::Any" in _sv_sigs(code2, "Vb_destroy_Vb")
            @test occursin("_ZN3PodC2Ei", _sv_fn(code2, "Pod_Pod", "this::Any, n::Integer"))
            @test occursin("_ZN3PodD2Ev", _sv_fn(code2, "Pod_destroy_Pod", "this::Any"))
            @test "_ZN3AbsC2Ei" in thunks2
        finally
            Base.delete_method(neutered)
        end
        @test !isempty(Base.invokelatest(_SV_W._structor_bindings, _SV_FUNCTIONS, _SV_STRUCTS)[1])
    end
end
