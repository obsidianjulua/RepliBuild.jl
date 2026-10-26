# =============================================================================
# What a C++ finalizer calls (library-free, CI-safe)
#
# Two kinds of wrapper type own a C++ object and free it from a GC finalizer:
# `Managed<X>`, and the `mutable struct` a factory function (`make_rect`) turns
# its class into. The finalizer has to free the object's storage, with the
# deallocator that matches how the library allocated it. Found in a walk of the
# user docs (2026-10-26):
#
#   * The factory struct called `<X>_destroy_<X>` by its Julia name. The
#     deleting destructor (D0) and the ones that only end the object's lifetime
#     (D1, D2) all have that name, dedup kept D2, and every collected object
#     leaked.
#   * `Managed<X>` took the first destructor in metadata order. A class with no
#     virtual destructor has no D0, so its finalizer ran D2 and leaked: 85 types
#     across the Hub's imgui, box2d, fmt, pugixml and tinyxml2.
#   * A method whose one parameter is `this` was taken for a deleter by its
#     name: imgui's `ManagedImDrawList` called `ImDrawList::_ClearFreeMemory()`,
#     which frees the list's buffers and leaves the list allocated. A static
#     member is a deleter (tinyxml2's `XMLNode::DeleteNode(XMLNode*)` returns a
#     node to its pool), and the receiver gate tells the two apart.
#   * The class a factory makes was read off the factory's name before its return
#     type: `make_rect_nx` made a struct `RectNx`, which C++ does not have.
#
# The generator runs on metadata shaped as the build writes it.
# test_cpp_finalizers_live.jl (devtests) builds the same shapes from C++ and
# counts what the finalizers free.
# =============================================================================

using Test
using RepliBuild
using Libdl

const _FZ_W = RepliBuild.Wrapper
const _FZ_LIB = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))

_fz_p(n, jt, ct) = Dict{String,Any}("name" => n, "julia_type" => jt, "c_type" => ct)
const _FZ_VOID = Dict{String,Any}("c_type" => "void", "julia_type" => "Cvoid", "size" => 0)
# A pointer return as the build records it: the class is in `c_type` only.
_fz_ret(cls) = Dict{String,Any}("c_type" => "$cls*", "julia_type" => "Ptr{Cvoid}", "size" => 8)

# A metadata row: name and class come from the demangled spelling, as at build time.
function _fz_row(mangled, demangled; params = Any[], ret = _FZ_VOID, hop = nothing)
    r = only(RepliBuild.Compiler.parse_function_signatures(
        [Dict{String,Any}("mangled" => mangled, "demangled" => demangled)]))
    r["parameters"] = Any[params...]
    r["return_type"] = ret
    r["is_noexcept"] = true
    r["is_vararg"] = false
    hop === nothing || (r["has_object_pointer"] = hop)
    return r
end
_fz_free(mangled, demangled, params...; ret = _FZ_VOID) =
    _fz_row(mangled, demangled; params = params, ret = ret, hop = false)
_fz_dtor(cls, variant) = _fz_row("_ZN$(length(cls))$(cls)D$(variant)Ev", "$cls::~$cls()";
                                 params = (_fz_p("this", "Ptr{$cls}", "$cls*"),), hop = true)

_fz_mem(n, jt, ct, off, size) = Dict{String,Any}("name" => n, "julia_type" => jt, "c_type" => ct,
                                                 "offset" => off, "size" => size)
_fz_vptr(cls) = _fz_mem("_vptr\$$cls", "Ptr{Cvoid}", "function_ptr(int)**", "0x0", 8)
function _fz_rec(kind, size, members...; base = nothing)
    r = Dict{String,Any}("kind" => kind, "byte_size" => "0x" * string(size, base = 16),
                         "members" => Any[members...])
    base === nothing || (r["base_classes"] = Any[Dict{String,Any}(
        "type" => base, "offset" => 0, "virtual" => false, "accessibility" => "public")])
    return r
end

# From `head` to the first line starting with `stop`.
function _fz_chunk(code, head, stop)
    i = findfirst(head, code)
    i === nothing && return ""
    rest = SubString(code, first(i))
    j = findfirst(stop, rest)
    return j === nothing ? String(rest) : String(rest[1:first(j) - 1])
end

const _FZ_STRUCTS = Dict{String,Any}(
    # Shape has a virtual destructor and a subclass, Rect.
    "Shape"  => _fz_rec("class", 16, _fz_vptr("Shape"), _fz_mem("id_", "Cint", "int", "0x8", 4)),
    "Rect"   => _fz_rec("class", 32, _fz_mem("w_", "Cdouble", "double", "0x10", 8),
                                     _fz_mem("h_", "Cdouble", "double", "0x18", 8); base = "Shape"),
    # No virtual destructor, so no D0.
    "Plain"  => _fz_rec("struct", 4, _fz_mem("v", "Cint", "int", "0x0", 4)),
    "Widget" => _fz_rec("struct", 4, _fz_mem("v", "Cint", "int", "0x0", 4)),
    "Leaf"   => _fz_rec("class", 16, _fz_vptr("Leaf"), _fz_mem("v_", "Cint", "int", "0x8", 4)),
    "Node"   => _fz_rec("class", 8, _fz_vptr("Node")),
    "Pooled" => _fz_rec("class", 8, _fz_vptr("Pooled")),
)

const _FZ_DD = (_fz_p("w", "Cdouble", "double"), _fz_p("h", "Cdouble", "double"))
const _FZ_FUNCTIONS = Any[
    # D2 listed before D0: what a finalizer calls must not follow the order.
    _fz_dtor("Shape", 2), _fz_dtor("Shape", 0),
    _fz_dtor("Rect", 0), _fz_dtor("Rect", 2),
    _fz_free("_Z9make_rectdd", "make_rect(double, double)", _FZ_DD...; ret = _fz_ret("Rect")),
    # Its name names no class: it makes a Rect.
    _fz_free("_Z12make_rect_nxdd", "make_rect_nx(double, double)", _FZ_DD...; ret = _fz_ret("Rect")),
    # Its name names a class it does not return: it makes a Rect.
    _fz_free("_Z8new_leafd", "new_leaf(double)", _fz_p("w", "Cdouble", "double"); ret = _fz_ret("Rect")),
    # Its name names a class derived from what it returns: it makes a Rect.
    _fz_free("_Z11create_recti", "create_rect(int)", _fz_p("n", "Cint", "int"); ret = _fz_ret("Shape")),
    # Shape has a subclass, so its D0, called directly, can skip a destructor.
    _fz_free("_Z10make_shapev", "make_shape()"; ret = _fz_ret("Shape")),
    # No D0 and no deleter function: nothing may own a Plain.
    _fz_dtor("Plain", 2),
    _fz_free("_Z10make_plaini", "make_plain(int)", _fz_p("v", "Cint", "int"); ret = _fz_ret("Plain")),
    _fz_free("_Z11spawn_plaini", "spawn_plain(int)", _fz_p("v", "Cint", "int"); ret = _fz_ret("Plain")),
    # No D0, and a deleter function listed after the destructor.
    _fz_dtor("Widget", 2),
    _fz_free("_Z13create_widgeti", "create_widget(int)", _fz_p("v", "Cint", "int"); ret = _fz_ret("Widget")),
    _fz_free("_Z14destroy_widgetP6Widget", "destroy_widget(Widget*)", _fz_p("w", "Ptr{Widget}", "Widget*")),
    # A leaf handed out by a function that is not a factory.
    _fz_dtor("Leaf", 0), _fz_dtor("Leaf", 2),
    _fz_free("_Z10spawn_leafi", "spawn_leaf(int)", _fz_p("v", "Cint", "int"); ret = _fz_ret("Leaf")),
    # A method that says free, ahead of the destructors: its one parameter is `this`.
    _fz_row("_ZN4Node10FreeMemoryEv", "Node::FreeMemory()";
            params = (_fz_p("this", "Ptr{Node}", "Node*"),), hop = true),
    _fz_dtor("Node", 0), _fz_dtor("Node", 2),
    _fz_free("_Z10spawn_nodev", "spawn_node()"; ret = _fz_ret("Node")),
    # A static member that deletes, the library's own deleter: it beats D0.
    _fz_dtor("Pooled", 0), _fz_dtor("Pooled", 2),
    _fz_row("_ZN6Pooled12DeletePooledEPS_", "Pooled::DeletePooled(Pooled*)";
            params = (_fz_p("p", "Ptr{Pooled}", "Pooled*"),), hop = false),
    _fz_free("_Z12spawn_pooledv", "spawn_pooled()"; ret = _fz_ret("Pooled")),
]

@testset "C++ finalizers free what they own" begin
    dir = mktempdir()
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "finalizers"
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
    meta = Dict{String,Any}("functions" => deepcopy(_FZ_FUNCTIONS), "globals" => Dict{String,Any}(),
                            "struct_definitions" => deepcopy(_FZ_STRUCTS))
    res = _FZ_W.generate_introspective_module_cpp(cfg, _FZ_LIB, meta, "Finalizers",
                                                  _FZ_W.create_type_registry(cfg), true)
    code = res isa Tuple ? String(first(res)) : String(res)

    @testset "the module parses" begin
        bad = Any[]
        walk(x) = x isa Expr && (x.head in (:incomplete, :error) ? push!(bad, x) : foreach(walk, x.args))
        walk(Meta.parseall(code))
        @test isempty(bad)
    end

    @testset "a factory struct's finalizer is the deleting destructor or the library's deleter" begin
        rect = _fz_chunk(code, "mutable struct Rect\n", "# Allow passing a Rect")
        @test !isempty(rect)
        @test occursin("ccall((:_ZN4RectD0Ev, LIBRARY_PATH), Cvoid, (Ptr{Cvoid},), o.handle)", rect)
        @test !occursin("Rect_destroy_Rect", rect)
        # Every factory of a Rect constructs one, whatever its name says.
        for (fac, sig) in (("make_rect", "w::Cdouble, h::Cdouble"), ("new_leaf", "w::Cdouble"),
                           ("create_rect", "n::Cint"))
            @test (fac, occursin("function Rect($sig)", rect) && occursin("= $fac(", rect)) == (fac, true)
        end

        widget = _fz_chunk(code, "mutable struct Widget\n", "# Allow passing a Widget")
        @test !isempty(widget)
        @test occursin("ccall((:_Z14destroy_widgetP6Widget, LIBRARY_PATH), Cvoid, (Ptr{Cvoid},), o.handle)", widget)
        @test !occursin("Widget_destroy_Widget", widget)
    end

    @testset "a factory makes the class it returns, or one its name names that derives from it" begin
        @test !occursin("RectNx", code)
        @test !occursin("mutable struct Leaf\n", code)    # new_leaf returns a Rect
        @test occursin("ManagedLeaf", code)
    end

    @testset "no owning type where nothing frees the object" begin
        # Shape has a subclass: its D0, called directly, skips Rect's destructor.
        @test !occursin("mutable struct Shape\n", code)
        # Plain has no D0 and no deleter function.
        @test !occursin("mutable struct Plain\n", code)
        @test occursin("\nstruct Plain\n", code)          # its layout struct stays
        for name in ("ManagedPlain", "make_plain_safe", "spawn_plain_safe")
            @test (name, occursin(name, code)) == (name, false)
        end
    end

    @testset "Managed types call a deleter function (a static member is one), else D0" begin
        want = ("Shape" => "_ZN5ShapeD0Ev", "Leaf" => "_ZN4LeafD0Ev", "Node" => "_ZN4NodeD0Ev",
                "Pooled" => "_ZN6Pooled12DeletePooledEPS_")
        checked = 0
        for (cls, sym) in want
            m = _fz_chunk(code, "mutable struct Managed$cls\n", "export Managed$cls")
            @test (cls, occursin("ccall((:$sym, LIBRARY_PATH), Cvoid", m)) == (cls, true)
            checked += 1
        end
        @test checked == length(want)
        @test !occursin("FreeMemory", _fz_chunk(code, "mutable struct ManagedNode\n", "export ManagedNode"))
        for name in ("spawn_leaf_safe", "spawn_node_safe", "spawn_pooled_safe")
            @test (name, occursin("function $name(", code)) == (name, true)
        end
    end
end
