#!/usr/bin/env julia
# test/test_typed_pointer_returns.jl — a pointer or reference return takes the
# type its declaration names (2026-11-08).
#
# Both generators returned `Ptr{Cvoid}` for every record pointer the metadata
# had not typed and for every reference: `glfwCreateWindow` (`GLFWwindow*`, whose
# parameters were already `Ptr{GLFWwindow}`), `ImGui::GetIO` (`ImGuiIO&`),
# `glfwGetMonitors` (`GLFWmonitor**`). LlamaChat's GUI cast them back by hand at
# every call site. `_typed_pointer_return` reads the declaration instead.
#
# Two things it must NOT do, both asserted here:
#   • type a pointer to a mutable record (a named union, a record carried as its
#     bytes): `unsafe_load(Ptr{T})` on a mutable `T` reads C's bytes as an object
#     reference (abi_fuzz X5), and a typed return invites that load;
#   • make a return a `Managed` owner. GeneratorCpp emits `f_safe(...)::ManagedX`
#     with a finalizer for a `Ptr{X}` return whose class has a deleter. A
#     reference never transfers ownership (`GetIO()` is the context's memory), so
#     a return typed only by its declaration gets no `_safe` wrapper; one the
#     generator typed before keeps it.
#
# No toolchain: synthetic metadata through both generators.

using Test
using RepliBuild
using Libdl

const TPR_W = RepliBuild.Wrapper

_tpr_fn(name, mangled, demangled, ret_c, ret_jl; params = Any[], noexcept = false) = Dict{String,Any}(
    "name" => name, "mangled" => mangled, "demangled" => demangled,
    "class" => "", "is_method" => false, "is_vararg" => false, "is_noexcept" => noexcept,
    "exported" => true, "has_object_pointer" => nothing,
    "parameters" => params,
    "return_type" => Dict{String,Any}("c_type" => ret_c, "julia_type" => ret_jl, "size" => 8))
_tpr_param(name, c, jl) = Dict{String,Any}("name" => name, "c_type" => c, "julia_type" => jl)
_tpr_mem(name, c, jl, off, sz) = Dict{String,Any}("name" => name, "c_type" => c, "julia_type" => jl,
                                                  "offset" => off, "size" => sz)

function _tpr_config(language)
    dir = mktempdir()
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "tpr"
    root = "$(escape_string(dir))"
    [wrap]
    language = "$language"
    [types]
    strictness = "warn"
    allow_unknown_structs = true
    [cache]
    enabled = false
    """)
    RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml"))
end
_tpr_libref() = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
_tpr_body(code, name) = let m = match(Regex("^function $(name)\\(.*?^end\$", "ms"), code)
    m === nothing ? "" : String(m.match)
end

@testset "Typed pointer returns" begin

@testset "the declaration names the type" begin
    names = Set(["Win", "IO", "Mode"])
    for (c, want) in [("Win*", "Ptr{Win}"), ("const Win*", "Ptr{Win}"), ("Win const*", "Ptr{Win}"),
                      ("Win* const", "Ptr{Win}"), ("const Win* const", "Ptr{Win}"),
                      ("struct Win*", "Ptr{Win}"), ("Win**", "Ptr{Ptr{Win}}"),
                      ("Win* const*", "Ptr{Ptr{Win}}"), ("IO&", "Ptr{IO}"), ("const IO&", "Ptr{IO}"),
                      ("IO&&", "Ptr{IO}"), ("IO*&", "Ptr{Ptr{IO}}"), ("Mode*", "Ptr{Mode}"),
                      # nothing to type: by value, void, char, function pointer, unknown, internal
                      ("IO", nothing), ("const IO", nothing), ("void*", nothing), ("char*", nothing),
                      ("const char*", nothing), ("function_ptr(void; void*)*", nothing),
                      ("Mystery*", nothing), ("_IO_FILE*", nothing)]
        @test (c, TPR_W._typed_pointer_return(c, names, identity)) == (c, want)
    end
    # the generator's sanitizer decides the spelling looked up
    @test TPR_W._typed_pointer_return("ns::Win*", Set(["ns_Win"]), s -> replace(s, "::" => "_")) == "Ptr{ns_Win}"
end

@testset "mutable records and value constants are not return types" begin
    src = """
    struct Win end
    struct IO
        x::Cint
    end
    mutable struct U
        data::NTuple{8, UInt8}
    end
    @enum Mode::Cint begin
        A = 0
    end
    const LIBRARY_PATH = "x"
    const Alias = Cuint
    """
    @test TPR_W._typed_return_names(src) == Set(["Win", "IO", "Mode"])
end

@testset "C: an opaque handle, a double pointer, a union left untyped" begin
    cfg = _tpr_config("c")
    meta = Dict{String,Any}(
        "functions" => Any[
            _tpr_fn("make_win", "make_win", "make_win", "Win*", "Ptr{Cvoid}"),
            # Win is declared only: a parameter is what makes the module emit `struct Win end`
            _tpr_fn("destroy_win", "destroy_win", "destroy_win", "void", "Cvoid";
                    params = Any[_tpr_param("w", "Win*", "Ptr{Win}")]),
            _tpr_fn("list_wins", "list_wins", "list_wins", "Win**", "Ptr{Ptr{Cvoid}}";
                    params = Any[_tpr_param("count", "int*", "Ptr{Cint}")]),
            _tpr_fn("const_win", "const_win", "const_win", "Win* const", "Ptr{Cvoid}"),
            _tpr_fn("get_uu", "get_uu", "get_uu", "U**", "Ptr{Ptr{Cvoid}}"),
            _tpr_fn("get_raw", "get_raw", "get_raw", "void*", "Ptr{Cvoid}"),
            _tpr_fn("get_name", "get_name", "get_name", "const char*", "Cstring")],
        "globals" => Dict{String,Any}(),
        "struct_definitions" => Dict{String,Any}(
            "U" => Dict{String,Any}("kind" => "union", "byte_size" => "0x8",
                                    "members" => Any[_tpr_mem("i", "int", "Cint", "0x0", 4),
                                                     _tpr_mem("d", "double", "Cdouble", "0x0", 8)])))
    code = TPR_W.generate_introspective_module_c(cfg, _tpr_libref(), meta, "TprC",
                                                 TPR_W.create_type_registry(cfg), false)
    @test occursin(r"^struct Win end"m, code)
    @test occursin(r"^mutable struct U\b"m, code)
    @test occursin(r"^function make_win\(\)::Ptr\{Win\}"m, code)
    @test occursin("ccall((:make_win, LIBRARY_PATH), Ptr{Win},", code)
    @test occursin(r"^function list_wins\(.*\)::Ptr\{Ptr\{Win\}\}"m, code)
    @test occursin(r"^function const_win\(\)::Ptr\{Win\}"m, code)
    # X5: no pointer to a mutable record. (`U*` the generator typed before this
    # change, from `struct_types`; that is unchanged here.)
    @test occursin(r"^function get_uu\(\)::Ptr\{Ptr\{Cvoid\}\}"m, code)
    @test occursin(r"^function get_raw\(\)::Ptr\{Cvoid\}"m, code)
    @test occursin(r"^function get_name\(\)::Union\{String, ?Nothing\}"m, code)   # the Cstring policy's
    @test isempty(TPR_W._undefined_ccall_types(code))
    @test !any(a -> a isa Expr && a.head in (:incomplete, :error), Meta.parseall(code).args)
end

@testset "C++: references typed, a mutable record left, no new Managed owner" begin
    cfg = _tpr_config("cpp")
    meta = Dict{String,Any}(
        "functions" => Any[
            _tpr_fn("get_io", "_Z6get_iov", "get_io()", "IO&", "Ptr{Cvoid}"),
            _tpr_fn("get_col", "_Z7get_coli", "get_col(int)", "V4 const*", "Ptr{Cvoid}";
                    params = Any[_tpr_param("idx", "int", "Cint")]),
            _tpr_fn("get_u", "_Z5get_uv", "get_u()", "U&", "Ptr{Cvoid}"),
            # `noexcept`: the `_safe` Managed wrapper is emitted on the ccall path
            _tpr_fn("widget_destroy", "_Z14widget_destroyP6Widget", "widget_destroy(Widget*)", "void", "Cvoid";
                    params = Any[_tpr_param("w", "Widget*", "Ptr{Widget}")], noexcept = true),
            # typed before this change (a `T*` to a known struct): keeps its Managed wrapper
            _tpr_fn("get_widget", "_Z10get_widgetv", "get_widget()", "Widget*", "Ptr{Cvoid}"; noexcept = true),
            # typed by its declaration only: a reference, never an owner
            _tpr_fn("widget_ref", "_Z10widget_refv", "widget_ref()", "Widget&", "Ptr{Cvoid}"; noexcept = true)],
        "globals" => Dict{String,Any}(),
        "struct_definitions" => Dict{String,Any}(
            "IO" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x8",
                                     "members" => Any[_tpr_mem("w", "float", "Cfloat", "0x0", 4),
                                                      _tpr_mem("h", "float", "Cfloat", "0x4", 4)]),
            "V4" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x10",
                                     "members" => Any[_tpr_mem("x", "float", "Cfloat", "0x0", 4),
                                                      _tpr_mem("y", "float", "Cfloat", "0x4", 4),
                                                      _tpr_mem("z", "float", "Cfloat", "0x8", 4),
                                                      _tpr_mem("w", "float", "Cfloat", "0xc", 4)]),
            "U" => Dict{String,Any}("kind" => "union", "byte_size" => "0x8",
                                    "members" => Any[_tpr_mem("i", "int", "Cint", "0x0", 4),
                                                     _tpr_mem("d", "double", "Cdouble", "0x0", 8)]),
            "Widget" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x4",
                                         "members" => Any[_tpr_mem("id", "int", "Cint", "0x0", 4)])),
        "record_abi" => Dict{String,Any}())
    res = TPR_W.generate_introspective_module_cpp(cfg, _tpr_libref(), meta, "TprCpp",
                                                  TPR_W.create_type_registry(cfg), true)
    code = res isa Tuple ? res[1] : res
    @test occursin(r"^mutable struct U\b"m, code)
    @test occursin("Ptr{IO}", _tpr_body(code, "get_io"))
    @test occursin("Ptr{V4}", _tpr_body(code, "get_col"))
    @test !occursin("Ptr{U}", _tpr_body(code, "get_u"))                  # X5
    @test occursin("Ptr{Cvoid}", _tpr_body(code, "get_u"))
    @test occursin(r"^function widget_ref\(\)::Ptr\{Widget\}"m, code)
    @test occursin(r"^function get_widget\(\)::Ptr\{Widget\}"m, code)
    # Ownership did not move: the struct-pointer return the generator always
    # typed keeps its finalizing wrapper, the reference gets none.
    @test occursin(r"^function get_widget_safe\("m, code)
    @test !occursin(r"^function widget_ref_safe\("m, code)
    @test !occursin(r"_safe\(\)::Managed(IO|V4|U)\b", code)
    @test isempty(TPR_W._undefined_ccall_types(code))
end

end
