# =============================================================================
# Cross-library type identity in graph mode, on synthetic metadata
# (Wrapper/TypeImports.jl; docs/updates/CROSS_LIBRARY_TYPES.md, Option 1)
#
# Several libraries built from one graph each get a wrapper, and library B's API
# uses types library A defines. B's wrapper imports the types an earlier
# library emitted as its own (`const X = Base.parentmodule(@__MODULE__).A.X`)
# and defines none of them. Who owns a type is decided from the graph and what
# each wrapper EMITTED (its type manifest), with an identity check.
#
# Four hand-built libraries: XCore → XHist → XTree, with XTree reaching an
# XHist type by declaration alone, and XLate, wrapped last, owning a class XTree
# could only forward-declare. Plus a C owner (XC) for a C++ importer. No
# toolchain; the generators' output is text. The live run on a CMake project is
# test_cross_library_types_live.jl (devtests).
# =============================================================================

using Test
using RepliBuild
using Libdl

const _XT_W = RepliBuild.Wrapper
const _XT_C = RepliBuild.Compiler
const _XT_IR = RepliBuild.JLCSIRGenerator
const _XT_LIB = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))

# ── Metadata builders ────────────────────────────────────────────────────────

_xt_p(n, jt, ct) = Dict{String,Any}("name" => n, "julia_type" => jt, "c_type" => ct)
const _XT_VOID = Dict{String,Any}("c_type" => "void", "julia_type" => "Cvoid", "size" => 0)
const _XT_INT = Dict{String,Any}("c_type" => "int", "julia_type" => "Cint", "size" => 4)
_xt_ptr(cls) = Dict{String,Any}("c_type" => "$cls*", "julia_type" => "Ptr{$cls}", "size" => 8)
_xt_val(cls, size) = Dict{String,Any}("c_type" => cls, "julia_type" => cls, "size" => size)

# A metadata row: name and class come from the demangled spelling, as at build time.
function _xt_row(mangled, demangled; params = Any[], ret = _XT_VOID, hop = nothing, noexcept = false)
    r = only(_XT_C.parse_function_signatures(
        [Dict{String,Any}("mangled" => mangled, "demangled" => demangled)]))
    r["parameters"] = Any[params...]
    r["return_type"] = ret
    r["is_noexcept"] = noexcept
    r["is_vararg"] = false
    hop === nothing || (r["has_object_pointer"] = hop)
    return r
end
_xt_dtor(cls, variant) = _xt_row("_ZN$(length(cls))$(cls)D$(variant)Ev", "$cls::~$cls()";
                                 params = (_xt_p("this", "Ptr{$cls}", "$cls*"),), hop = true)

_xt_mem(n, jt, ct, off, size) = Dict{String,Any}("name" => n, "julia_type" => jt, "c_type" => ct,
                                                 "offset" => off, "size" => size)
_xt_vptr(cls) = _xt_mem("_vptr\$$cls", "Ptr{Cvoid}", "function_ptr(int)**", "0x0", 8)
function _xt_rec(kind, size, members...; base = nothing, scopes = [""])
    r = Dict{String,Any}("kind" => kind, "byte_size" => "0x" * string(size, base = 16),
                         "members" => Any[members...])
    scopes === nothing || (r["scopes"] = scopes)
    base === nothing || (r["base_classes"] = Any[Dict{String,Any}(
        "type" => base, "offset" => 0, "virtual" => false, "accessibility" => "public")])
    return r
end
function _xt_enum(members...; scopes = [""])
    Dict{String,Any}("kind" => "enum", "underlying_type" => "unsigned int", "julia_type" => "Cuint",
                     "byte_size" => "0x4", "scopes" => scopes,
                     "enumerators" => Any[Dict{String,Any}("name" => n, "value" => v) for (n, v) in members])
end
_xt_sym(mangled) = Dict{String,Any}("mangled" => mangled, "demangled" => mangled, "type" => "T", "address" => 0)

_xt_meta(functions, structs; symbols = Any[]) =
    Dict{String,Any}("functions" => deepcopy(functions), "globals" => Dict{String,Any}(),
                     "struct_definitions" => deepcopy(structs), "record_abi" => Dict{String,Any}(),
                     "symbols" => deepcopy(symbols))

# XCore: a polymorphic TObject (homed here), a header-only Plain and Color, a
# Config in namespace `a`, a record whose scope the parser could not settle,
# and `Mismatch`, which XHist also defines with another layout.
const _XT_CORE_STRUCTS = Dict{String,Any}(
    "TObject"  => _xt_rec("class", 16, _xt_vptr("TObject"), _xt_mem("fUniqueID", "Cuint", "unsigned int", "0x8", 4)),
    "Plain"    => _xt_rec("struct", 16, _xt_mem("x", "Cint", "int", "0x0", 4), _xt_mem("y", "Cdouble", "double", "0x8", 8)),
    "Config"   => _xt_rec("struct", 4, _xt_mem("a", "Cint", "int", "0x0", 4); scopes = ["a"]),
    "NoScope"  => _xt_rec("struct", 4, _xt_mem("n", "Cint", "int", "0x0", 4); scopes = nothing),
    "Mismatch" => _xt_rec("struct", 8, _xt_mem("m", "Cdouble", "double", "0x0", 8)),
    "Only"     => _xt_rec("struct", 4, _xt_mem("o", "Cint", "int", "0x0", 4)),
    "__enum__Color" => _xt_enum("RED" => 1, "GREEN" => 2),
)
const _XT_CORE_FUNCTIONS = Any[
    _xt_dtor("TObject", 0), _xt_dtor("TObject", 1),
    _xt_row("_Z11make_objectv", "make_object()"; ret = _xt_ptr("TObject")),
    _xt_row("_Z10make_plainid", "make_plain(int, double)";
            params = (_xt_p("x", "Cint", "int"), _xt_p("y", "Cdouble", "double")), ret = _xt_val("Plain", 16)),
    _xt_row("_Z11color_value5Color", "color_value(Color)"; params = (_xt_p("c", "Color", "Color"),), ret = _XT_INT),
    _xt_row("_Z8take_cfgN1a6ConfigE", "take_cfg(a::Config)"; params = (_xt_p("c", "Config", "Config"),), ret = _XT_INT),
    _xt_row("_Z7only_fn4Only", "only_fn(Only)"; params = (_xt_p("o", "Only", "Only"),), ret = _XT_INT),
    _xt_row("_Z6no_scope7NoScope", "no_scope(NoScope)"; params = (_xt_p("n", "NoScope", "NoScope"),), ret = _XT_INT),
    _xt_row("_Z8mismatch8Mismatch", "mismatch(Mismatch)"; params = (_xt_p("m", "Mismatch", "Mismatch"),), ret = _XT_INT),
]

# XHist: THist derives from TObject (declared only here: no entry), Plain and
# Color again (same scope and layout), its own b::Config and NoScope (no
# scopes), Mismatch with another layout, Only (a different scope: a sibling for
# XTree), and functions reaching TObject by pointer and by value.
const _XT_HIST_STRUCTS = Dict{String,Any}(
    "THist"    => _xt_rec("class", 24, _xt_mem("fN", "Cint", "int", "0x10", 4); base = "TObject"),
    "Plain"    => _xt_rec("struct", 16, _xt_mem("x", "Cint", "int", "0x0", 4), _xt_mem("y", "Cdouble", "double", "0x8", 8)),
    "Config"   => _xt_rec("struct", 4, _xt_mem("a", "Cint", "int", "0x0", 4); scopes = ["b"]),
    "NoScope"  => _xt_rec("struct", 4, _xt_mem("n", "Cint", "int", "0x0", 4); scopes = nothing),
    "Only"     => _xt_rec("struct", 4, _xt_mem("o", "Cint", "int", "0x0", 4); scopes = ["h"]),
    "__enum__Color" => _xt_enum("RED" => 1, "GREEN" => 2),
)
const _XT_HIST_MISMATCH = _xt_rec("struct", 16, _xt_mem("m", "Cdouble", "double", "0x0", 8),
                                  _xt_mem("k", "Cdouble", "double", "0x8", 8))
const _XT_HIST_FUNCTIONS = Any[
    _xt_dtor("THist", 0), _xt_dtor("THist", 1),
    _xt_row("_Z7use_objP7TObject5Plain5Color", "use_obj(TObject*, Plain, Color)";
            params = (_xt_p("o", "Ptr{TObject}", "TObject*"), _xt_p("p", "Plain", "Plain"), _xt_p("c", "Color", "Color")),
            ret = _XT_INT),
    _xt_row("_Z4take7TObject", "take(TObject)"; params = (_xt_p("o", "TObject", "TObject"),), ret = _XT_INT),
    _xt_row("_Z5twice5Plain", "twice(Plain)"; params = (_xt_p("p", "Plain", "Plain"),), ret = _xt_val("Plain", 16)),
    _xt_row("_Z10spawn_histi", "spawn_hist(int)"; params = (_xt_p("n", "Cint", "int"),), ret = _xt_ptr("THist")),
    _xt_row("_Z8take_cfgN1b6ConfigE", "take_cfg(b::Config)"; params = (_xt_p("c", "Config", "Config"),), ret = _XT_INT),
]

# XTree: reaches THist by declaration (XHist is earlier) and TLater by
# declaration (XLate comes later), and uses Only, which two earlier libraries
# own.
const _XT_TREE_STRUCTS = Dict{String,Any}(
    "TTree" => _xt_rec("class", 16, _xt_vptr("TTree"), _xt_mem("fEntries", "Cint", "int", "0x8", 4)),
)
const _XT_TREE_FUNCTIONS = Any[
    _xt_dtor("TTree", 0), _xt_dtor("TTree", 1),
    _xt_row("_Z4fillP5THist", "fill(THist*)"; params = (_xt_p("h", "Ptr{THist}", "THist*"),), ret = _XT_INT),
    _xt_row("_Z5laterP6TLater", "later(TLater*)"; params = (_xt_p("l", "Ptr{TLater}", "TLater*"),), ret = _XT_INT),
    _xt_row("_Z7only_fn4Only", "only_use(Only)"; params = (_xt_p("o", "Only", "Only"),), ret = _XT_INT),
]

# XLate: owns TLater (a Managed type), wrapped after XTree's placeholder.
const _XT_LATE_STRUCTS = Dict{String,Any}(
    "TLater" => _xt_rec("class", 16, _xt_vptr("TLater"), _xt_mem("fL", "Cint", "int", "0x8", 4)),
)
const _XT_LATE_FUNCTIONS = Any[
    _xt_dtor("TLater", 0), _xt_dtor("TLater", 1),
    _xt_row("_Z10spawn_laterv", "spawn_later()"; ret = _xt_ptr("TLater")),
]

# XC, a C owner: a POD a C++ importer takes.
const _XT_C_STRUCTS = Dict{String,Any}(
    "Pt" => _xt_rec("struct", 16, _xt_mem("a", "Cint", "int", "0x0", 4), _xt_mem("b", "Cdouble", "double", "0x8", 8)),
)
const _XT_C_FUNCTIONS = Any[
    _xt_row("xc_sum", "xc_sum"; params = (_xt_p("p", "Pt", "Pt"),),
            ret = Dict{String,Any}("c_type" => "double", "julia_type" => "Cdouble", "size" => 8)),
]

function _xt_config(dir, language)
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "xt"
    root = "$(escape_string(dir))"
    [link]
    enable_lto = false
    [wrap]
    language = "$language"
    [types]
    strictness = "warn"
    allow_unknown_structs = true
    [cache]
    enabled = false
    """)
    return RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml"))
end

"""
Wrap `metadata` as module `name` the way `wrap_introspective` does in graph
mode: the pre-pass against `earlier`, then the generator with facts, then the
manifest. Returns `(text, manifest, imports, reports, metadata)`.
"""
function _xt_wrap(name, metadata, earlier; language = :cpp, subclassed = String[])
    dir = mktempdir()
    cfg = _xt_config(dir, language)
    md = deepcopy(metadata)
    (imports, reports) = _XT_W._resolve_type_imports(md, earlier; pe = false)
    _XT_W._apply_type_imports!(md, imports)
    _XT_W._drop_functions_bound_earlier!(md, earlier)
    md["graph_subclassed"] = subclassed
    md["earlier_placeholders"] = _XT_W._earlier_placeholders(md, earlier)
    facts = Dict{String,Any}()
    text = if language == :c
        _XT_W.generate_introspective_module_c(cfg, _XT_LIB, md, name, _XT_W.create_type_registry(cfg), false;
                                              emitted_facts = facts)
    else
        res = _XT_W.generate_introspective_module_cpp(cfg, _XT_LIB, md, name, _XT_W.create_type_registry(cfg), false;
                                                      emitted_facts = facts)
        String(first(res))
    end
    manifest = _XT_W._type_manifest(md, text, facts; module_name = name, language = language, library = "lib$name.so")
    return (text, manifest, imports, reports, md)
end

_xt_earlier(name, manifest, metadata) =
    _XT_W._earlier_library(name, manifest, metadata, joinpath(mktempdir(), "lib$name.so"))

const _XT_CORE = _xt_meta(_XT_CORE_FUNCTIONS, _XT_CORE_STRUCTS)
const _XT_HIST = _xt_meta(_XT_HIST_FUNCTIONS, _XT_HIST_STRUCTS)
const _XT_TREE = _xt_meta(_XT_TREE_FUNCTIONS, _XT_TREE_STRUCTS)
const _XT_LATE = _xt_meta(_XT_LATE_FUNCTIONS, _XT_LATE_STRUCTS)
const _XT_CLIB = _xt_meta(_XT_C_FUNCTIONS, _XT_C_STRUCTS)

@testset "Cross-library type identity" begin
    # XCore first: nothing earlier, so every type is its own. XHist derives from
    # TObject, so XCore learns that from the graph and gives TObject no factory
    # struct (a Managed type instead).
    (core_text, core_manifest, core_imports, core_reports, _) =
        _xt_wrap("XCore", _XT_CORE, Any[]; subclassed = ["TObject"])
    core = _xt_earlier("XCore", core_manifest, _XT_CORE)

    @testset "a first library owns everything it emits" begin
        @test isempty(core_imports) && isempty(core_reports)
        kinds = Dict(k => v["kind"] for (k, v) in core_manifest["types"])
        @test kinds == Dict("TObject" => "struct", "Plain" => "struct", "Config" => "struct", "NoScope" => "struct",
                            "Mismatch" => "struct", "Only" => "struct", "__enum__Color" => "enum")
        @test core_manifest["types"]["TObject"]["managed"] == "ManagedTObject"
        @test core_manifest["types"]["Plain"]["layout"] == [16, 8]
        @test core_manifest["types"]["Config"]["scopes"] == ["a"]
        @test core_manifest["types"]["NoScope"]["scopes"] === nothing
        @test "_Z11make_objectv" in core_manifest["bound_symbols"]
        @test !occursin("Base.parentmodule", core_text) && !occursin("IMPORTED_FROM", core_text)
        @test occursin("\nmutable struct ManagedTObject\n", core_text) && !occursin("\nmutable struct TObject\n", core_text)
        # No import, no import chunk, no sidecar.
        @test _XT_W._type_imports_chunk(_XT_CORE["struct_definitions"], _XT_W._sanitize_cpp_type_name, _XT_CORE, ".") == ""
        d = mktempdir()
        _XT_W._write_type_imports(d, Dict{String,Any}())
        @test !isfile(joinpath(d, _XT_W.TYPE_IMPORTS))
    end

    (hist_text, hist_manifest, hist_imports, hist_reports, hist_md) = _xt_wrap("XHist", _XT_HIST, Any[core])
    hist = _xt_earlier("XHist", hist_manifest, _XT_HIST)

    @testset "ownership: the four outcomes" begin
        decisions = 0
        # Declared only here, defined in XCore: imported.
        @test haskey(hist_imports, "TObject") && hist_imports["TObject"]["module"] == "XCore"; decisions += 1
        # Full in both, one scope, one layout: imported.
        @test haskey(hist_imports, "Plain") && hist_imports["Plain"]["julia_name"] == "Plain"; decisions += 1
        @test haskey(hist_imports, "__enum__Color") && hist_imports["__enum__Color"]["kind"] == "enum"; decisions += 1
        # a::Config and b::Config: two types, own plus a report.
        @test !haskey(hist_imports, "Config") && any(r -> startswith(r, "Config:") && occursin("different type", r), hist_reports); decisions += 1
        # Scopes unknown: own plus a report.
        @test !haskey(hist_imports, "NoScope") && any(r -> startswith(r, "NoScope:") && occursin("scope unknown", r), hist_reports); decisions += 1
        # Only in scope `h` here, global in XCore: own.
        @test !haskey(hist_imports, "Only"); decisions += 1
        # THist is XHist's own.
        @test !haskey(hist_imports, "THist") && hist_manifest["types"]["THist"]["kind"] == "struct"; decisions += 1
        @test decisions == 7
        @test length(hist_imports) == 3
        @test hist_imports["TObject"]["raii"] isa Vector && any(r -> occursin("~", r["demangled"]), hist_imports["TObject"]["raii"])
        @test hist_imports["Plain"]["owner_entry"]["members"][1]["name"] == "x"
        # The pre-pass left the owner's body under the key, marked.
        @test haskey(hist_md["struct_definitions"]["TObject"], "imported")
        @test hist_md["struct_definitions"]["TObject"]["imported"]["module"] == "XCore"
        @test length(hist_md["struct_definitions"]["TObject"]["members"]) == 2
        # Same scope, two layouts: a wrap error naming both libraries.
        bad = deepcopy(_XT_HIST)
        bad["struct_definitions"]["Mismatch"] = deepcopy(_XT_HIST_MISMATCH)
        err = try
            _XT_W._resolve_type_imports(bad, Any[core]; pe = false)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("Mismatch", err) && occursin("XCore", err) && occursin("two layouts", err)
        # The same two layouts, each declared in its own .c file: file-local
        # types that only share a name (C has one tag scope; libwebp's
        # decoder and demuxer each define a `MemBuffer`). Own, reported, no error.
        local_core = deepcopy(core)
        local_core.manifest["types"]["Mismatch"]["decl_files"] = ["/p/src/dec/idec_dec.c"]
        local_here = deepcopy(bad)
        local_here["struct_definitions"]["Mismatch"]["decl_files"] = ["/p/src/demux/demux.c"]
        (li, lr) = _XT_W._resolve_type_imports(local_here, Any[local_core]; pe = false)
        @test !haskey(li, "Mismatch")
        @test any(r -> startswith(r, "Mismatch:") && occursin("source file", r) && occursin("demux.c", r), lr)
        # One side file-local is enough: a header type and a .c type are two.
        (li1, _) = _XT_W._resolve_type_imports(local_here, Any[core]; pe = false)
        @test !haskey(li1, "Mismatch")
        # Declared in headers on both sides, the two layouts are still an ODR
        # error between the libraries, as before.
        hdr_core = deepcopy(core)
        hdr_core.manifest["types"]["Mismatch"]["decl_files"] = ["/p/include/m.h"]
        hdr_here = deepcopy(bad)
        hdr_here["struct_definitions"]["Mismatch"]["decl_files"] = ["/p/include/./m.h"]
        err = try
            _XT_W._resolve_type_imports(hdr_here, Any[hdr_core]; pe = false)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("two layouts", err)
        @test _XT_W._tu_local(Dict("decl_files" => ["/p/a.C", "/p/b.cpp"])) &&
              !_XT_W._tu_local(Dict("decl_files" => ["/p/a.c", "/p/a.h"])) &&
              !_XT_W._tu_local(Dict("decl_files" => String[])) && !_XT_W._tu_local(Dict{String,Any}())
        # A member defined strongly in both: an error, not on PE.
        claimed = deepcopy(_XT_HIST)
        claimed["struct_definitions"]["TObject"] = deepcopy(_XT_CORE_STRUCTS["TObject"])
        push!(claimed["functions"], _xt_dtor("TObject", 1))
        claimed["symbols"] = Any[_xt_sym("_ZN7TObjectD1Ev")]
        owner_claiming = deepcopy(core)
        owner_claiming.manifest["types"]["TObject"]["strong_members"] = ["_ZN7TObjectD1Ev"]
        err = try
            _XT_W._resolve_type_imports(claimed, Any[owner_claiming]; pe = false)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("defined strongly in both", err)
        (pe_imports, _) = _XT_W._resolve_type_imports(claimed, Any[owner_claiming]; pe = true)
        @test haskey(pe_imports, "TObject")
    end

    @testset "what the importer emits" begin
        for name in ("TObject", "Plain", "Color")
            @test (name, occursin("\nconst $name = Base.parentmodule(@__MODULE__).XCore.$name\n", hist_text)) == (name, true)
            @test (name, occursin(Regex("^(?:mutable )?struct $name\\b", "m"), hist_text)) == (name, false)
        end
        @test !occursin("ManagedTObject", hist_text)
        @test !occursin("primitive type Color", hist_text)
        sizes = match(r"^const STRUCT_SIZES = Dict\{Symbol,Int\}\(\n(.*?)\n\)"ms, hist_text)
        @test sizes !== nothing && occursin(":THist => 24", sizes.captures[1]) && occursin(":Config => 4", sizes.captures[1])
        @test !occursin(":Plain =>", hist_text) && !occursin(":TObject =>", hist_text)
        # The derived record flattens the imported base's fields.
        thist = match(r"^struct THist\n(.*?)\nend"ms, hist_text)
        @test thist !== nothing && occursin("fUniqueID::Cuint", thist.captures[1]) && occursin("fN::Cint", thist.captures[1])
        # Own types still emitted; a different-scope Config is XHist's own.
        @test occursin("\nstruct Config\n", hist_text) && occursin("\nstruct NoScope\n", hist_text)
        @test occursin(":XCore => (", hist_text) && occursin("function _check_type_imports()", hist_text)
        @test occursin("_check_type_imports()\n", hist_text)
        # Loadable, with the imports declared.
        imported = _XT_W._imported_type_names(hist_md["struct_definitions"], _XT_W._sanitize_cpp_type_name)
        @test imported == ["Color", "Plain", "TObject"]
        @test _XT_W._assert_wrapper_loadable(hist_text, "XHist"; imported = imported) === nothing
        # The manifest lists XHist's own types only.
        @test sort(collect(keys(hist_manifest["types"]))) == ["Config", "NoScope", "Only", "THist"]
    end

    @testset "a referenced-only type, a sibling, and a placeholder" begin
        (tree_text, tree_manifest, tree_imports, tree_reports, tree_md) = _xt_wrap("XTree", _XT_TREE, Any[core, hist])
        # THist reached by pointer with no definition here: imported from XHist.
        @test haskey(tree_imports, "THist") && tree_imports["THist"]["module"] == "XHist"
        @test occursin("\nconst THist = Base.parentmodule(@__MODULE__).XHist.THist\n", tree_text)
        # Only: both earlier libraries emitted it; the first is taken, reported.
        @test haskey(tree_imports, "Only") && tree_imports["Only"]["module"] == "XCore"
        @test any(r -> startswith(r, "Only:") && occursin("XCore and XHist", r), tree_reports)
        # `only_fn` is bound by XCore already: not wrapped again.
        @test !any(f -> f["mangled"] == "_Z7only_fn4Only", tree_md["functions"])
        @test !occursin("function only_use", tree_text)
        # TLater has no owner yet: a forward declaration, listed as a placeholder.
        @test !haskey(tree_imports, "TLater")
        @test occursin("\nstruct TLater end\n", tree_text)
        @test tree_manifest["types"]["TLater"]["kind"] == "placeholder"
        tree = _xt_earlier("XTree", tree_manifest, _XT_TREE)
        # XLate, the owner, comes later and converts its Managed type to XTree's placeholder.
        (late_text, late_manifest, late_imports, _, late_md) = _xt_wrap("XLate", _XT_LATE, Any[core, hist, tree])
        @test isempty(late_imports)
        @test late_md["earlier_placeholders"] == Dict("TLater" => [["XTree", "TLater"]])
        @test occursin("Base.unsafe_convert(::Type{Ptr{Base.parentmodule(@__MODULE__).XTree.TLater}}, o::ManagedTLater) = " *
                       "Ptr{Base.parentmodule(@__MODULE__).XTree.TLater}(o.handle)", late_text)
        @test late_manifest["types"]["TLater"]["kind"] == "struct"
    end

    @testset "a C owner, a C++ importer" begin
        (c_text, c_manifest, _, _, _) = _xt_wrap("XC", _XT_CLIB, Any[]; language = :c)
        @test c_manifest["language"] == "c" && c_manifest["types"]["Pt"]["kind"] == "struct"
        @test occursin("\nstruct Pt\n", c_text)
        xc = _xt_earlier("XC", c_manifest, _XT_CLIB)
        importer = _xt_meta(Any[_xt_row("_Z5shift2Pt", "shift(Pt)"; params = (_xt_p("p", "Pt", "Pt"),), ret = _xt_val("Pt", 16))],
                            Dict{String,Any}("Pt" => deepcopy(_XT_C_STRUCTS["Pt"])))
        (i_text, i_manifest, i_imports, _, _) = _xt_wrap("XI", importer, Any[xc])
        @test haskey(i_imports, "Pt") && i_imports["Pt"]["julia_name"] == "Pt"      # the owner's spelling
        @test occursin("\nconst Pt = Base.parentmodule(@__MODULE__).XC.Pt\n", i_text)
        @test !occursin("\nstruct Pt\n", i_text) && isempty(i_manifest["types"])
    end

    @testset "determinism: shuffled inputs, identical decisions and text" begin
        # Three insertion orders per Dict, so a result that follows one is caught.
        perm(xs, trial) = (v = sort(collect(xs); by = string); trial == 1 ? v : trial == 2 ? reverse(v) : circshift(v, 1))
        function permuted(d::Dict, trial)
            out = Dict{String,Any}()
            for k in perm(keys(d), trial)
                v = d[k]
                out[k] = v isa Dict ? permuted(v, trial) : deepcopy(v)
            end
            return out
        end
        reference = nothing
        for trial in 1:3
            md = permuted(_XT_HIST, trial)
            md["functions"] = deepcopy(trial == 2 ? reverse(_XT_HIST_FUNCTIONS) : circshift(_XT_HIST_FUNCTIONS, trial))
            e = _XT_W._earlier_library("XCore", permuted(core_manifest, trial), permuted(_XT_CORE, trial), core.library)
            (t, m, imps, reps, _) = _xt_wrap("XHist", md, Any[e])
            io = IOBuffer(); _XT_W._print_json_sorted(io, m); mtext = String(take!(io))
            io = IOBuffer(); _XT_W._print_json_sorted(io, _XT_W._type_imports_file(imps)); itext = String(take!(io))
            chunk = match(r"# ── Types owned by other libraries.*?\nconst IMPORTED_FROM"s, t).match
            this = (chunk, mtext, itext, reps, sort(collect(keys(imps))))
            reference === nothing ? (reference = this) : @test this == reference
        end
        @test reference !== nothing
    end

    @testset "the thunk side: the owner's definition where this library trapped" begin
        vt = RepliBuild.DWARFParser.VtableInfo(Dict{String,RepliBuild.DWARFParser.ClassInfo}(),
                                               Dict{String,UInt64}(), Dict{String,UInt64}())
        plain = deepcopy(_XT_HIST)
        without = _XT_IR.generate_jlcs_ir(vt, deepcopy(plain))
        @test without == _XT_IR.generate_jlcs_ir(vt, deepcopy(plain))
        @test !occursin("@_Z4take7TObject_thunk", without)                 # TObject by value: opaque, no thunk
        d = mktempdir()
        _XT_W._write_type_imports(d, hist_imports)
        @test isfile(joinpath(d, _XT_W.TYPE_IMPORTS))
        merged = _XT_IR.apply_type_imports!(deepcopy(plain), d)
        @test haskey(merged["struct_definitions"], "TObject")
        @test any(f -> get(f, "imported_raii", false) == true, merged["functions"])
        with = _XT_IR.generate_jlcs_ir(vt, merged)
        @test occursin("@_Z4take7TObject_thunk", with)
        @test !occursin("@_ZN7TObjectD1Ev_thunk", with)                      # the owner's destructor gets no thunk
        # No sidecar: the metadata is untouched.
        @test _XT_IR.apply_type_imports!(deepcopy(plain), mktempdir()) == plain
    end

    @testset "the guard: a loop that forgets the predicate is a refused wrap" begin
        imported = ["Plain", "TObject"]
        @test _XT_W._assert_no_imported_redefinition(hist_text, "XHist", imported) === nothing
        bad = hist_text * "\nstruct Plain\n    x::Cint\nend\n"
        err = try
            _XT_W._assert_no_imported_redefinition(bad, "XHist", imported)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("Plain (a struct)", err)
        err = try
            _XT_W._assert_no_imported_redefinition(hist_text * "\nmutable struct ManagedTObject\n    handle::Ptr{TObject}\nend\n",
                                                   "XHist", imported)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("ManagedTObject", err)
        # Neuter the predicate in every loop: the struct emitter then defines
        # Plain beside the import, and the guard refuses the text.
        Core.eval(_XT_W, :(_emits_type(key, info) = true))
        neutered = try
            first(_xt_wrap("XHist", _XT_HIST, Any[core]))
        finally
            Core.eval(_XT_W, :(_emits_type(key, info) = !(info isa AbstractDict && haskey(info, "imported"))))
        end
        @test occursin("\nstruct Plain\n", neutered) && occursin("const Plain = Base.parentmodule", neutered)
        @test_throws ErrorException _XT_W._assert_no_imported_redefinition(neutered, "XHist", imported)
        # Restored: the real predicate emits no definition.
        @test !occursin("\nstruct Plain\n", first(_xt_wrap("XHist", _XT_HIST, Any[core])))
    end

    @testset "metadata provenance" begin
        # The parser's helper: scopes from the definitions' `scope`, files given.
        def = Dict{String,Any}("kind" => "struct")
        _XT_C._record_provenance!(def, Any[Dict("scope" => "b"), Dict("scope" => "a"), Dict("scope" => "b"), Dict()],
                                  ["/p/b.h", "/p/a.h", "/p/a.h"])
        @test def["scopes"] == ["a", "b"] && def["decl_files"] == ["/p/a.h", "/p/b.h"]
        bare = Dict{String,Any}("kind" => "struct")
        _XT_C._record_provenance!(bare, Any[Dict()], String[])
        @test !haskey(bare, "scopes") && !haskey(bare, "decl_files")
        # Every record reached without a definition, by Julia spelling or, when
        # that degraded to `Ptr{Cvoid}`, by the C spelling.
        @test _XT_W._referenced_record_names(_XT_TREE) == ["Only", "THist", "TLater"]
        degraded = _xt_meta(Any[_xt_row("_Z5laterP6TLater", "later(TLater*)";
                                        params = (_xt_p("l", "Ptr{Cvoid}", "TLater*"),), ret = _XT_INT)],
                            Dict{String,Any}())
        @test _XT_W._referenced_record_names(degraded) == ["TLater"]
        @test _XT_W._reached_c_type_name("const ns::Leaf *") == "Leaf"
        @test _XT_W._reached_c_type_name("void *") === nothing
        @test _XT_W._reached_c_type_name("int (*)(int)") === nothing
    end
end
