#!/usr/bin/env julia
# test/test_cross_library_types_live.jl — one C++ type is one Julia type across
# the libraries of a graph (docs/updates/CROSS_LIBRARY_TYPES.md, Option 1).
#
# test/xlib_test/ (copied to a temp dir first; see its CMakeLists.txt): xc, a C
# library with a POD `Pt` and an enum `Mode`; xa, a C++ library linking xc,
# with a polymorphic `Shape` (factory + deleter function), a `Leaf` nothing in
# xa derives from, a header-only `Plain` and `Color`; xb, a C++ library linking
# xa, deriving from `Shape` and `Leaf`, passing `Plain`, `Color`, `Shape*`,
# `Pt` and `Mode` through its own API and embedding `Plain`. The TOML selects
# xb; xa and xc come with it.
#
# Checks: each wrapper leaves a type manifest; xb imports every type xa and xc
# own (`type_imports.json`, and `const X = Base.parentmodule(@__MODULE__).Xa.X`
# in its text) and defines none of them; `Derived` flattens `Shape`'s fields
# through the imported base; `Leaf` gets no factory struct in xa because xb
# derives from it; in a child, the types are identical across the submodules,
# the owners' values pass into xb's functions on both tiers, xa's virtual call
# reaches xb's override, and xb's records embedding an import read back right;
# wrapping again is byte-identical; the registry vendors the graph and `use`
# returns it with the same identities.
#
# Needs cmake, ninja, clang, clang++, llvm-link and libJLCS (xb's by-value
# crossings are Tier 2 thunks). The synthetic half is
# test_cross_library_types.jl (runtests).

using Test
using TOML
using RepliBuild

isdefined(@__MODULE__, :TestSupport) || include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const XL_FIXTURE = joinpath(@__DIR__, "xlib_test")

_xl_posix(p) = replace(String(p), '\\' => '/')

"""
The probe a child runs against the graph module the expression `load` returns
(after `prelude`): one PROBE line per check.
"""
xl_probe_code(load::String, prelude::String) = """
    $prelude
    M = $load
    check(name, ok, detail) = println("PROBE ", name, ": ", ok ? "PASS" : "FAIL " * string(detail))
    # One type per C++ type, whichever submodule names it.
    check("identity_plain", M.Xb.Plain === M.Xa.Plain, (M.Xb.Plain, M.Xa.Plain))
    check("identity_color", M.Xb.Color === M.Xa.Color, (M.Xb.Color, M.Xa.Color))
    check("identity_shape", M.Xb.Shape === M.Xa.Shape, (M.Xb.Shape, M.Xa.Shape))
    check("identity_leaf", M.Xb.Leaf === M.Xa.Leaf, (M.Xb.Leaf, M.Xa.Leaf))
    check("identity_pt", M.Xb.Pt === M.Xc.Pt && M.Xa.Pt === M.Xc.Pt, (M.Xb.Pt, M.Xa.Pt, M.Xc.Pt))
    check("identity_mode", M.Xb.Mode === M.Xc.Mode, (M.Xb.Mode, M.Xc.Mode))
    # The enums' values come with them, from a C owner and a C++ owner alike.
    check("enum_values", M.Xb.GREEN === M.Xa.GREEN && M.Xb.RED === M.Xa.RED && M.Xb.MODE_B === M.Xc.MODE_B &&
                         M.Xa.MODE_C === M.Xc.MODE_C, (isdefined(M.Xb, :GREEN), isdefined(M.Xb, :MODE_B)))
    # The owners' values cross into xb's functions: a factory object (its
    # unsafe_convert is keyed on xa's type), a record by value through a Tier 2
    # thunk that needed xa's layout, an enum, the C owner's POD through C++.
    p = M.Xa.make_plain(3, 4.5)
    check("plain_value", p.x == 3 && p.y == 4.5, p)
    s = M.Xa.Shape(Int32(7))
    check("use", M.Xb.use(s, p, M.Xa.GREEN) == 70 + 3 + 4 + 2 && M.Xb.use(s, p, M.Xb.GREEN) == 79,
          M.Xb.use(s, p, M.Xa.GREEN))
    q = M.Xb.twice(p)
    check("twice", q isa M.Xa.Plain && q.x == 6 && q.y == 9.0, q)
    pt = M.Xc.xc_make(1, 2.0)
    check("pt_c_owner", M.Xb.xb_pt(pt).a == 3 && M.Xa.xa_shift(pt, 10).a == 11 && M.Xc.xc_sum(pt) == 3.0,
          (M.Xb.xb_pt(pt), M.Xa.xa_shift(pt, 10)))
    check("mode", M.Xb.mode_twice(M.Xc.MODE_B) == 4 && M.Xc.xc_mode(M.Xc.MODE_C) == 30,
          (M.Xb.mode_twice(M.Xc.MODE_B), M.Xc.xc_mode(M.Xc.MODE_C)))
    # xa's virtual call on an object xb made reaches xb's override.
    d = M.Xb.spawn_derived(2, 5)
    check("virtual_dispatch", M.Xa.shape_get(d) == 25 && M.Xa.Shape_get(d) == 25, (M.Xa.shape_get(d), M.Xa.Shape_get(d)))
    sl = M.Xb.spawn_subleaf(5)
    check("subleaf_virtual", M.Xa.Leaf_value(sl) == 1015, M.Xa.Leaf_value(sl))
    # Layouts: xb's records flatten the imported bases' fields and embed the
    # imported record; the layout table lists xb's own types only.
    check("derived_fields", :id_ in fieldnames(M.Xb.Derived) && :extra_ in fieldnames(M.Xb.Derived) && !ismutabletype(M.Xb.Derived),
          fieldnames(M.Xb.Derived))
    check("subleaf_fields", :v_ in fieldnames(M.Xb.SubLeaf) && :w_ in fieldnames(M.Xb.SubLeaf), fieldnames(M.Xb.SubLeaf))
    h = M.Xb.make_holder(8)
    check("holder", h.p isa M.Xa.Plain && h.p.x == 8 && h.p.y == 4.0, h)
    check("layout_table", !haskey(M.Xb.STRUCT_SIZES, :Plain) && haskey(M.Xa.STRUCT_SIZES, :Plain) && haskey(M.Xb.STRUCT_SIZES, :Derived),
          (collect(keys(M.Xb.STRUCT_SIZES)),))
    # The D0 rule is graph-wide: xb derives from Leaf, so xa gives Leaf no
    # factory struct (it keeps its layout and a Managed type); Shape has the
    # library's deleter function and keeps its factory struct.
    check("leaf_layout_not_handle", fieldnames(M.Xa.Leaf) != (:handle,) && isdefined(M.Xa, :ManagedLeaf), fieldnames(M.Xa.Leaf))
    check("shape_factory_struct", fieldnames(M.Xa.Shape) == (:handle,) && !isdefined(M.Xb, :ManagedShape), fieldnames(M.Xa.Shape))
    check("imports_listed", Set(keys(M.Xb.IMPORTED_FROM)) == Set([:Xa, :Xc]) && Set(keys(M.Xa.IMPORTED_FROM)) == Set([:Xc]),
          (M.Xb.IMPORTED_FROM, M.Xa.IMPORTED_FROM))
    println("PROBE_DONE")
    """

const XL_PROBES = ["identity_plain", "identity_color", "identity_shape", "identity_leaf", "identity_pt",
                   "identity_mode", "enum_values", "plain_value", "use", "twice", "pt_c_owner", "mode", "virtual_dispatch",
                   "subleaf_virtual", "derived_fields", "subleaf_fields", "holder", "layout_table",
                   "leaf_layout_not_handle", "shape_factory_struct", "imports_listed"]

function xl_run(load::String; prelude::String = "")
    r = run_isolated(`-e $(xl_probe_code(load, prelude))`; timeout = 600)
    v = probe_verdicts(r.output)
    r.ok && haskey(v, "PROBE_DONE") || println(stderr, r.output)
    return v
end

_xl_norm(s) = replace(s, r"^# Generated: .*$"m => "# Generated: <n>")

if requires("Cross-library types (live)", :cmake, :clang, :clangxx, :llvm_link, :libJLCS)
if Sys.which("ninja") === nothing
    @testset "Cross-library types (live) (skipped — not found: ninja)" begin
        @test_skip false
    end
else
@testset "Cross-library types (live): one Julia type per C++ type across a graph" begin
    work = _xl_posix(mktempdir())
    proj = work * "/xl"
    cp(XL_FIXTURE, proj)
    toml = proj * "/replibuild.toml"
    write(toml, """
        [project]
        name = "xlib_fixture"
        root = "."

        [graph]
        source = "."
        targets = ["xb"]
        build_type = "Release"
        """)
    home = work * "/home"
    withenv("REPLIBUILD_HOME" => home, "REPLIBUILD_JULIA_ENV" => "none") do

    out = RepliBuild.build(toml)
    manifest = TOML.parsefile(out * "/graph.toml")
    libs = manifest["libraries"]
    @test [l["name"] for l in libs] == ["xc", "xa", "xb"]
    @test [l["module"] for l in libs] == ["Xc", "Xa", "Xb"]
    @test [l["bucket"] for l in libs] == ["c", "cpp", "cpp"]

    W = RepliBuild.Wrapper
    graph_file = RepliBuild.wrap(toml)
    @test graph_file == out * "/XlibFixture.jl"
    libdir(n) = out * "/" * dirname(only(filter(l -> l["name"] == n, libs))["library"])

    @testset "what each wrap left beside its library" begin
        checked = 0
        for n in ("xc", "xa", "xb")
            m = W._read_type_manifest(libdir(n))
            @test (n, m !== nothing && m["version"] == 1) == (n, true)
            checked += 1
        end
        @test checked == 3
        xc = W._read_type_manifest(libdir("xc"))["types"]
        @test Dict(k => v["kind"] for (k, v) in xc) == Dict("Pt" => "struct", "__enum__Mode" => "enum")
        @test xc["Pt"]["layout"] == [16, 8] && xc["Pt"]["scopes"] == [""]
        xa = W._read_type_manifest(libdir("xa"))["types"]
        @test Dict(k => v["kind"] for (k, v) in xa) ==
              Dict("Shape" => "idiomatic", "Leaf" => "struct", "Plain" => "struct", "__enum__Color" => "enum")
        @test xa["Leaf"]["managed"] == "ManagedLeaf" && xa["Shape"]["managed"] === nothing
        @test !haskey(xa, "Pt")                                             # imported, so not its own
        xb = W._read_type_manifest(libdir("xb"))["types"]
        @test Dict(k => v["kind"] for (k, v) in xb) ==
              Dict("Derived" => "struct", "SubLeaf" => "struct", "Holder" => "struct")
        @test !isfile(libdir("xc") * "/" * W.TYPE_IMPORTS)                  # the first library imports nothing
        xa_imports = RepliBuild._read_json(libdir("xa") * "/" * W.TYPE_IMPORTS)["imports"]
        # Pt by value; Mode only through xc.h, which the header walk found: an
        # owner emitted it, so xa imports it instead of emitting a second type.
        @test sort(collect(keys(xa_imports))) == ["Pt", "__enum__Mode"]
        xb_imports = RepliBuild._read_json(libdir("xb") * "/" * W.TYPE_IMPORTS)["imports"]
        @test sort(collect(keys(xb_imports))) == ["Leaf", "Plain", "Pt", "Shape", "__enum__Color", "__enum__Mode"]
        @test xb_imports["Plain"]["module"] == "Xa" && xb_imports["Pt"]["module"] == "Xc"
        @test haskey(xb_imports["Plain"]["owner_entry"], "members")
        @test any(r -> occursin("~", r["demangled"]), xb_imports["Shape"]["raii"])   # the owner's destructor rows
    end

    @testset "the importer's text" begin
        xb = read(out * "/" * libs[3]["wrapper"], String)
        for (name, owner) in (("Plain", "Xa"), ("Color", "Xa"), ("Shape", "Xa"), ("Leaf", "Xa"),
                              ("Pt", "Xc"), ("Mode", "Xc"))
            @test (name, occursin("\nconst $name = Base.parentmodule(@__MODULE__).$owner.$name\n", xb)) == (name, true)
            @test (name, occursin(Regex("^(?:mutable )?struct $name\\b", "m"), xb)) == (name, false)
            @test (name, occursin("Managed$name", xb)) == (name, false)
        end
        @test !occursin(r"^primitive type (Color|Mode)\b"m, xb)
        for (v, owner) in (("GREEN", "Xa"), ("RED", "Xa"), ("MODE_B", "Xc"))
            @test (v, occursin("\nconst $v = Base.parentmodule(@__MODULE__).$owner.$v\n", xb)) == (v, true)
        end
        @test occursin(r"^export [^\n]*\bGREEN\b"m, xb) && occursin(r"^export [^\n]*\bMODE_B\b"m, xb)
        @test !occursin(r":Plain => \d"m, xb) && occursin(r":Derived => 16"m, xb)
        derived = match(r"^struct Derived\n(.*?)\nend"ms, xb)
        @test derived !== nothing && occursin("id_::Cint", derived.captures[1]) && occursin("extra_::Cint", derived.captures[1])
        @test occursin(":Xa => (\"../../xa/lib/type_manifest.json\"", xb)
        @test occursin("_check_type_imports()", xb)
        # The owner's text is the single-library shape plus its own import from xc.
        xa = read(out * "/" * libs[2]["wrapper"], String)
        @test occursin("\nconst Pt = Base.parentmodule(@__MODULE__).Xc.Pt\n", xa)
        @test occursin("\nconst Mode = Base.parentmodule(@__MODULE__).Xc.Mode\n", xa) && !occursin("primitive type Mode", xa)
        @test occursin("\nstruct Plain\n", xa) && occursin("\nmutable struct Shape\n", xa) && occursin("\nstruct Leaf\n", xa)
        @test !occursin("Base.parentmodule", read(out * "/" * libs[1]["wrapper"], String))
    end

    @testset "live, in a child" begin
        v = xl_run("include($(repr(graph_file)))")
        checked = 0
        for name in XL_PROBES
            @test (name, get(v, name, "no PROBE line")) == (name, "PASS")
            checked += 1
        end
        @test checked == length(XL_PROBES)
    end

    @testset "wrapping again is byte-identical" begin
        before = Dict(l["wrapper"] => read(out * "/" * l["wrapper"], String) for l in libs)
        before_manifests = Dict(n => read(libdir(n) * "/" * W.TYPE_MANIFEST, String) for n in ("xc", "xa", "xb"))
        RepliBuild.wrap(toml)
        for l in libs
            @test (l["name"], _xl_norm(read(out * "/" * l["wrapper"], String)) == _xl_norm(before[l["wrapper"]])) == (l["name"], true)
        end
        for n in ("xc", "xa", "xb")
            @test (n, read(libdir(n) * "/" * W.TYPE_MANIFEST, String) == before_manifests[n]) == (n, true)
        end
    end

    @testset "registered and used as one graph" begin
        entry = RepliBuild.register(toml)
        @test entry.module_name == "XlibFixture"
        v = xl_run("RepliBuild.use(\"xlib_fixture\")"; prelude = "using RepliBuild")
        for name in ("identity_plain", "identity_pt", "use", "twice", "virtual_dispatch")
            @test (name, get(v, name, "no PROBE line")) == (name, "PASS")
        end
    end

    end # withenv
end
end # ninja
end # requires
