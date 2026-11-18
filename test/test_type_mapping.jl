#!/usr/bin/env julia
# test/test_type_mapping.jl — how a C/C++ type becomes a Julia type in the
# metadata, and what both generators do with it (2026-11-17).
#
# Three defects of one audit, all in the type mapping:
#
#   * A pointer or reference to an enum was mapped to the enum BY VALUE.
#     `Compiler.cpp_to_julia_type` strips `*`/`&` to find the base type and
#     returned the enum's name for every spelling of it, so `void out(enum E
#     *)` was wrapped as `out(x::E)` with `ccall(..., (E,), x)`: C wrote
#     through the enum's value as an address. In C++ a `const Color&` argument
#     was converted to its integer and handed to a thunk that loads a pointer.
#     `E*` is `Ptr{E}` (each level) and `E&` is `Ref{E}`, in parameters,
#     members and globals; a return is typed by `_typed_pointer_return`.
#   * `unsigned __int128` was in neither type map. The C generator then wrote
#     `ccall(..., unsigned __int128, (Any,), v)`, the wrapper did not parse, and
#     one function cost the library every other one. It is `UInt128`; any C
#     scalar the map does not know is one trap stub naming the type; and a
#     128-bit integer is refused where `ccall` would not place it as C does.
#   * `char8_t`, `char16_t` and `char32_t` are base types, and were refused as
#     records with no layout. An enum defined with no enumerators (`std::byte`,
#     a library's `enum class Handle : uint32_t {}`) had no entry at all.
#
# No toolchain: the mappers, a vendored readelf dump through `parse_dwarf_dump`,
# and synthetic metadata through both generators and the thunk generator.
# test_type_mapping_live.jl builds both fixtures and calls them.

using Test
using RepliBuild
using Libdl

const _TM_C  = RepliBuild.Compiler
const _TM_W  = RepliBuild.Wrapper
const _TM_FG = RepliBuild.JLCSIRGenerator.FunctionGen
const _TM_TU = RepliBuild.JLCSIRGenerator.TypeUtils

_tm_param(name, c, jl) = Dict{String,Any}("name" => name, "c_type" => c, "julia_type" => jl)
_tm_fn(name, mangled, demangled, ret_c, ret_jl, params...; noexcept = false) = Dict{String,Any}(
    "name" => name, "mangled" => mangled, "demangled" => demangled,
    "class" => "", "is_method" => false, "is_vararg" => false, "is_noexcept" => noexcept,
    "exported" => true, "has_object_pointer" => nothing,
    "parameters" => Any[p for p in params],
    "return_type" => Dict{String,Any}("c_type" => ret_c, "julia_type" => ret_jl, "size" => 0))
_tm_mem(name, c, jl, off, sz) = Dict{String,Any}("name" => name, "c_type" => c, "julia_type" => jl,
                                                 "offset" => off, "size" => sz)
_tm_enum(underlying_c, underlying_jl, size, enumerators...) = Dict{String,Any}(
    "kind" => "enum", "underlying_type" => underlying_c, "julia_type" => underlying_jl,
    "byte_size" => "0x$(string(size; base = 16))",
    "enumerators" => Any[Dict{String,Any}("name" => String(n), "value" => v) for (n, v) in enumerators])

function _tm_config(language)
    dir = mktempdir()
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "tm"
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
_tm_libref() = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
_tm_body(code, name) = let m = match(Regex("^function $(name)\\(.*?^end\$", "ms"), code)
    m === nothing ? "" : String(m.match)
end
_tm_parses(code) = !any(a -> a isa Expr && a.head in (:incomplete, :error), Meta.parseall(code).args)

# test/fixtures/enum_types.readelf.txt is `readelf --debug-dump=info` of
# (clang++ 23 -std=c++17 -g -O0 -c, the unit's directory renamed):
#
#   enum_types_sys.h (file 1 of the unit, a toolchain header in the table below)
#     namespace std { enum class byte : unsigned char {};
#                     enum class align_val_t : unsigned long {}; }
#   enum_types.cpp (file 0)
#     enum Small { SM_A, SM_B };
#     enum class Handle : unsigned short {};
#     enum class Opaque : int;
#     struct Holder { Small *kinds; int n; };
#     Small *g_smalls = nullptr;
#     void out_small(Small *out);            int smalls_sum(const Small *xs, int n);
#     void small_ref(const Small &s, Small **pp);
#     Small *smalls_static();                Holder holder_make();
#     std::byte byte_flip(std::byte b);      int bytes_sum(const std::byte *p, int n);
#     Handle handle_next(Handle h);          void opaque_set(Opaque *o);
#     unsigned long aligned(std::align_val_t a);
const _TM_DUMP = read(joinpath(@__DIR__, "fixtures", "enum_types.readelf.txt"), String)
const _TM_FILES = [Dict{Int,String}(0 => "/src/et/enum_types.cpp", 1 => "/usr/include/c++/15/cstddef")]
const _TM_LIBRARY = ["g_smalls", "_Z9out_smallP5Small", "_Z10smalls_sumPK5Smalli", "_Z9small_refRK5SmallPPS_",
                     "_Z13smalls_staticv", "_Z11holder_makev", "_Z11handle_next6Handle",
                     "_Z10opaque_setP6Opaque"]
_tm_parse(extra...) = _TM_C.parse_dwarf_dump(_TM_DUMP; file_tables = _TM_FILES,
                                             exported = Set{String}(vcat(_TM_LIBRARY, String[extra...])))

@testset "Type mapping" begin

# ── The mapper ───────────────────────────────────────────────────────────────

@testset "a pointer or reference to an enum is an address" begin
    structs = Set(["Pt"]); enums = Set(["Small", "Color"])
    m(t) = _TM_C.cpp_to_julia_type(t, structs, enums)
    for (c, want) in [("Small", "Small"), ("const Small", "Small"),
                      ("Small*", "Ptr{Small}"), ("const Small*", "Ptr{Small}"),
                      ("Small**", "Ptr{Ptr{Small}}"), ("const const Small**", "Ptr{Ptr{Small}}"),
                      ("Color&", "Ref{Color}"), ("const Color&", "Ref{Color}"), ("Color&&", "Ref{Color}"),
                      ("Color*&", "Ref{Ptr{Color}}"), ("const const Small*&", "Ref{Ptr{Small}}"),
                      ("Small[3]", "NTuple{3, Small}"), ("Small*[2]", "NTuple{2, Ptr{Small}}"),
                      # what the mapper already did for a struct and a scalar
                      ("Pt*", "Ptr{Pt}"), ("Pt&", "Ref{Pt}"), ("int*", "Ptr{Cint}"), ("int&", "Ref{Cint}"),
                      # a name nothing knows stays an untyped address
                      ("Mystery*", "Ptr{Cvoid}")]
        @test (c, m(c)) == (c, want)
    end
    # No enum by value was ever a pointer: the name alone, however it is qualified.
    @test all(t -> !startswith(m(t), "Ptr{") && !startswith(m(t), "Ref{"), ["Small", "const Small", "Color"])
end

@testset "the 128-bit integers and the Unicode character types are scalars" begin
    for (c, want) in [("__int128", "Int128"), ("__int128_t", "Int128"),
                      ("unsigned __int128", "UInt128"), ("__int128 unsigned", "UInt128"),
                      ("__uint128_t", "UInt128"),
                      ("char8_t", "UInt8"), ("char16_t", "UInt16"), ("char32_t", "UInt32")]
        @test (c, _TM_C.cpp_to_julia_type(c)) == (c, want)       # a parameter, a member, a global
        @test (c, _TM_C.dwarf_type_to_julia(c)) == (c, want)     # a return
        @test (c, _TM_C.cpp_to_julia_type("const " * c)) == (c, want)
        @test (c, _TM_C.dwarf_type_to_julia("const " * c)) == (c, want)
        @test (c, _TM_C.cpp_to_julia_type(c * "*")) == (c, "Ptr{$want}")
        @test (c, _TM_C.dwarf_type_to_julia(c * "*")) == (c, "Ptr{$want}")
        @test (c, _TM_C.cpp_to_julia_type(c * "&")) == (c, "Ref{$want}")
    end
    @test _TM_C.get_type_size("unsigned __int128") == 16
    @test (_TM_C.get_type_size("char8_t"), _TM_C.get_type_size("char16_t"), _TM_C.get_type_size("char32_t")) == (1, 2, 4)
    # The thunk side: an integer of that width, never a record named after the type.
    @test [_TM_TU.map_cpp_type(t) for t in ("char8_t", "char16_t", "char32_t", "unsigned __int128")] ==
          ["i8", "i16", "i32", "i128"]
    for t in ("char8_t", "char16_t", "char32_t", "unsigned __int128", "const char16_t")
        @test (t, _TM_FG.by_value_crossing(t, Dict{String,Any}(), Dict{String,Any}())) == (t, (:direct, nothing))
    end
    # Narrower than `int`: extended by the caller, and these are unsigned.
    @test _TM_FG._narrow_int_extension("i8", _tm_param("c", "char8_t", "UInt8")) == "zext"
    @test _TM_FG._narrow_int_extension("i16", _tm_param("c", "char16_t", "UInt16")) == "zext"
    @test _TM_FG._narrow_int_extension("i32", _tm_param("c", "char32_t", "UInt32")) === nothing
    # A scalar the map does not know is still `Any`: the generators refuse it.
    @test _TM_C.cpp_to_julia_type("_Float16") == "Any"
    @test _TM_C.dwarf_type_to_julia("_Float16") == "Any"
end

# ── The parser ───────────────────────────────────────────────────────────────

@testset "the metadata: an enum behind a pointer, in every position" begin
    rt, structs, globals, _, _ = _tm_parse()
    ptype(sym, i) = (rt[sym]["parameters"][i]["c_type"], rt[sym]["parameters"][i]["julia_type"])
    @test ptype("_Z9out_smallP5Small", 1) == ("Small*", "Ptr{Small}")
    @test ptype("_Z10smalls_sumPK5Smalli", 1) == ("const Small*", "Ptr{Small}")
    @test ptype("_Z9small_refRK5SmallPPS_", 1) == ("const Small&", "Ref{Small}")
    @test ptype("_Z9small_refRK5SmallPPS_", 2) == ("Small**", "Ptr{Ptr{Small}}")
    @test ptype("_Z10opaque_setP6Opaque", 1) == ("Opaque*", "Ptr{Opaque}")
    # A member and a global.
    kinds = only(m for m in structs["Holder"]["members"] if m["name"] == "kinds")
    @test (kinds["c_type"], kinds["julia_type"], kinds["size"]) == ("Small*", "Ptr{Small}", 8)
    @test (globals["g_smalls"]["c_type"], globals["g_smalls"]["julia_type"]) == ("Small*", "Ptr{Small}")
    # By value it is still the enum.
    @test ptype("_Z11handle_next6Handle", 1) == ("Handle", "Handle")
    # A return is typed by the generator (`_typed_pointer_return`), from this.
    @test rt["_Z13smalls_staticv"]["c_type"] == "Small*"
    @test rt["_Z13smalls_staticv"]["julia_type"] == "Ptr{Cvoid}"
end

@testset "the metadata: an enum defined with no enumerators" begin
    _, structs, _, _, _ = _tm_parse()
    # The library's own: always, like a record with no members.
    @test haskey(structs, "__enum__Handle")
    h = structs["__enum__Handle"]
    @test (h["kind"], h["underlying_type"], h["julia_type"], h["byte_size"]) ==
          ("enum", "unsigned short", "Cushort", "0x2")
    @test isempty(h["enumerators"])
    @test !haskey(h, "toolchain")
    @test h["scopes"] == [""]
    # One with enumerators is what it was.
    @test [e["name"] for e in structs["__enum__Small"]["enumerators"]] == ["SM_A", "SM_B"]
    # An opaque declaration says neither what it is an integer of nor what it
    # holds: no entry, with or without a function that names it...
    @test !haskey(structs, "__enum__Opaque")
    # ...and it is a declaration by its attribute, whatever else the producer
    # writes on it (clang writes no underlying type there; another may).
    typed = replace(_TM_DUMP, "    <1c9>   DW_AT_declaration : 1" =>
                              "    <1c9>   DW_AT_type        : <0x61>\n    <1c9>   DW_AT_declaration : 1")
    @test typed != _TM_DUMP
    @test !haskey(_TM_C.parse_dwarf_dump(typed; file_tables = _TM_FILES)[2], "__enum__Opaque")
    # The same DIE without the attribute is a definition, and has one.
    defined = replace(typed, "    <1c9>   DW_AT_declaration : 1\n" => "")
    @test haskey(_TM_C.parse_dwarf_dump(defined; file_tables = _TM_FILES)[2], "__enum__Opaque")
end

@testset "the metadata: a toolchain enum comes back where the library names it" begin
    key = "__enum__byte"
    # Nothing exported names `std::byte`: it stays out, like a toolchain record.
    rt0, s0, _, _, _ = _tm_parse()
    @test !haskey(s0, key)
    @test !haskey(s0, "__enum__align_val_t")
    # Crossed by value by an exported function.
    rt1, s1, _, _, _ = _tm_parse("_Z9byte_flipSt4byte")
    @test haskey(s1, key)
    b = s1[key]
    @test (b["underlying_type"], b["julia_type"], b["byte_size"], b["toolchain"]) ==
          ("unsigned char", "UInt8", "0x1", true)
    @test isempty(b["enumerators"])
    @test b["scopes"] == ["std"]
    @test b["decl_files"] == ["/usr/include/c++/15/cstddef"]
    # ...and the function crossing it is typed as it is for any enum.
    f = rt1["_Z9byte_flipSt4byte"]
    @test (f["c_type"], f["julia_type"], f["size"]) == ("byte", "byte", 1)
    @test f["parameters"][1]["julia_type"] == "byte"
    # Named behind a pointer only: the type that pointer is typed with.
    rt2, s2, _, _, _ = _tm_parse("_Z9bytes_sumPKSt4bytei")
    @test haskey(s2, key) && s2[key]["toolchain"] == true
    @test rt2["_Z9bytes_sumPKSt4bytei"]["parameters"][1]["julia_type"] == "Ptr{byte}"
    # Its sibling, which only an unexported function takes, stays out; exported, it is back.
    @test !haskey(s1, "__enum__align_val_t") && !haskey(s2, "__enum__align_val_t")
    @test haskey(_tm_parse("_Z7alignedSt11align_val_t")[2], "__enum__align_val_t")
    # With no symbol table to ask (a parser test), every non-std function seeds.
    @test haskey(_TM_C.parse_dwarf_dump(_TM_DUMP; file_tables = _TM_FILES)[2], key)
    # Without the header's provenance it is a library enum like any other.
    @test !haskey(_TM_C.parse_dwarf_dump(_TM_DUMP)[2][key], "toolchain")
end

@testset "_promote_toolchain_records!: an enum is reached by value or by address" begin
    tc() = Dict{String,Dict{String,Any}}(
        "__enum__byte" => _tm_enum("unsigned char", "UInt8", 1),
        "__enum__align_val_t" => _tm_enum("unsigned long", "Culong", 8),
        "pair<std::byte, int>" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x8", "base_classes" => Any[],
            "members" => Any[_tm_mem("first", "byte", "byte", 0, 1), _tm_mem("second", "int", "Cint", 4, 4)]))
    promote(seeds; library = Dict{String,Dict{String,Any}}()) = begin
        sd = Dict{String,Dict{String,Any}}(k => deepcopy(v) for (k, v) in library)
        (sd, _TM_C._promote_toolchain_records!(sd, tc(), seeds))
    end
    @test promote(["byte"])[2] == ["__enum__byte"]
    @test promote(["std::byte"])[2] == ["__enum__byte"]
    @test promote(["const std::byte"])[2] == ["__enum__byte"]
    @test promote(["byte[16]"])[2] == ["__enum__byte"]
    @test promote(["byte*"])[2] == ["__enum__byte"]
    @test promote(["const byte*", "byte&", "byte**", "const const byte*&", "byte*[4]"])[2] == ["__enum__byte"]
    @test promote(["int", "void*", "function_ptr(void; byte)*"])[2] == String[]
    sd, got = promote(["byte"])
    @test sd["__enum__byte"]["toolchain"] == true && isempty(sd["__enum__byte"]["enumerators"])
    @test !haskey(sd, "__enum__align_val_t")
    # A toolchain record holding one is complete, and brings it along.
    @test promote(["pair<std::byte, int>"])[2] == ["__enum__byte", "pair<std::byte, int>"]
    # ...behind a pointer the record stays out, as it always did.
    @test promote(["pair<std::byte, int>*"])[2] == String[]
    # A library record reached by value is walked, one that is not is not.
    lib = Dict{String,Dict{String,Any}}("Buf" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x10",
        "members" => Any[_tm_mem("p", "byte*", "Ptr{byte}", 0, 8), _tm_mem("tag", "byte", "byte", 8, 1)]))
    @test promote(["Buf"]; library = lib)[2] == ["__enum__byte"]
    @test promote(["Buf*"]; library = lib)[2] == String[]
    # A library enum of the same name keeps its entry.
    own = Dict{String,Dict{String,Any}}("__enum__byte" => _tm_enum("int", "Cint", 4, :B0 => 0))
    sd2, got2 = promote(["byte"]; library = own)
    @test got2 == String[] && !haskey(sd2["__enum__byte"], "toolchain")
end

# ── The C generator ──────────────────────────────────────────────────────────

@testset "C: an enum pointer is typed in every position" begin
    cfg = _tm_config("c")
    meta = Dict{String,Any}(
        "functions" => Any[
            _tm_fn("out_small", "out_small", "out_small", "void", "Cvoid", _tm_param("out", "Small*", "Ptr{Small}")),
            _tm_fn("smalls_sum", "smalls_sum", "smalls_sum", "int", "Cint",
                   _tm_param("xs", "const Small*", "Ptr{Small}"), _tm_param("n", "int", "Cint")),
            _tm_fn("small_swap", "small_swap", "small_swap", "int", "Cint",
                   _tm_param("a", "Small*", "Ptr{Small}"), _tm_param("b", "Small*", "Ptr{Small}")),
            _tm_fn("next_small", "next_small", "next_small", "Small", "Small", _tm_param("s", "Small", "Small")),
            _tm_fn("smalls_static", "smalls_static", "smalls_static", "Small*", "Ptr{Cvoid}"),
            _tm_fn("smalls_handle", "smalls_handle", "smalls_handle", "Small**", "Ptr{Ptr{Cvoid}}"),
            _tm_fn("out_perm", "out_perm", "out_perm", "void", "Cvoid", _tm_param("out", "Perm*", "Ptr{Perm}")),
            _tm_fn("perms_static", "perms_static", "perms_static", "Perm*", "Ptr{Cvoid}"),
            _tm_fn("holder_make", "holder_make", "holder_make", "Holder", "Holder")],
        "globals" => Dict{String,Any}(
            "g_smalls" => Dict{String,Any}("name" => "g_smalls", "c_type" => "Small*", "julia_type" => "Ptr{Small}")),
        "struct_definitions" => Dict{String,Any}(
            "__enum__Small" => _tm_enum("unsigned int", "Cuint", 4, :SM_A => 0, :SM_B => 1, :SM_C => 2),
            # a bitmask: an integer alias, not a primitive type
            "__enum__Perm" => _tm_enum("unsigned int", "Cuint", 4, :P_R => 1, :P_W => 2, :P_X => 4, :P_ALL => 7),
            "Holder" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x10",
                "members" => Any[_tm_mem("kinds", "Small*", "Ptr{Small}", "0x0", 8),
                                 _tm_mem("n", "int", "Cint", "0x8", 4)])))
    # A pointer is a pointer under every convention: each by name.
    @testset "$abi" for abi in (:sysv, :win64, :aapcs64)
        code = _TM_W.generate_introspective_module_c(cfg, _tm_libref(), meta, "TmC",
                                                     _TM_W.create_type_registry(cfg), false; abi)
        @test occursin(r"^primitive type Small <: RepliBuild\.OpenEnums\.OpenEnum\{Cuint\} 32 end"m, code)
        @test occursin(r"^const Perm = Cuint$"m, code)
        # A parameter: relaxed like every pointer, and the ccall says what it points at,
        # so a `Ref{Small}`, a `Vector{Small}` or a `Ptr{Small}` converts and a bare
        # `Small` does not.
        @test occursin(r"^function out_small\(out::Any\)"m, code)
        @test occursin("ccall((:out_small, LIBRARY_PATH), Cvoid, (Ptr{Small},), out)", code)
        @test occursin("ccall((:smalls_sum, LIBRARY_PATH), Cint, (Ptr{Small}, Cint,), xs, n_c)", code)
        @test occursin("ccall((:small_swap, LIBRARY_PATH), Cint, (Ptr{Small}, Ptr{Small},), a, b)", code)
        @test occursin("ccall((:out_perm, LIBRARY_PATH), Cvoid, (Ptr{Perm},), out)", code)
        # By value it is the enum, as before.
        @test occursin("ccall((:next_small, LIBRARY_PATH), Small, (Small,), s)", code)
        # A return, through `_typed_pointer_return`; the bitmask alias too.
        @test occursin(r"^function smalls_static\(\)::Ptr\{Small\}"m, code)
        @test occursin(r"^function smalls_handle\(\)::Ptr\{Ptr\{Small\}\}"m, code)
        @test occursin(r"^function perms_static\(\)::Ptr\{Perm\}"m, code)
        # A struct member: the enums land before every struct, so the name is defined.
        @test occursin(r"^    kinds::Ptr\{Small\}$"m, code)
        # A global.
        @test occursin("cglobal((:g_smalls, LIBRARY_PATH), Ptr{Small})", code)
        # No enum was passed by value where a pointer is declared.
        @test !occursin("(Small,), out)", code) && !occursin("(Small, Small,)", code)
        @test isempty(_TM_W._undefined_ccall_types(code))
        @test _tm_parses(code)
        _TM_W._assert_wrapper_loadable(code, "TmC")
    end
end

@testset "C: a scalar the map does not know is one trap stub" begin
    cfg = _tm_config("c")
    meta = Dict{String,Any}(
        "functions" => Any[
            _tm_fn("plain_add", "plain_add", "plain_add", "int", "Cint",
                   _tm_param("a", "int", "Cint"), _tm_param("b", "int", "Cint")),
            _tm_fn("f16_echo", "f16_echo", "f16_echo", "_Float16", "Any", _tm_param("x", "_Float16", "Any")),
            _tm_fn("f16_bits", "f16_bits", "f16_bits", "int", "Cint", _tm_param("x", "_Float16", "Any")),
            _tm_fn("f16_make", "f16_make", "f16_make", "_Float16", "Any", _tm_param("bits", "int", "Cint")),
            # what the wrapper wrote before the map knew the type: not Julia syntax
            _tm_fn("odd_echo", "odd_echo", "odd_echo", "unsigned __int256", "Any",
                   _tm_param("v", "unsigned __int256", "Any")),
            # a record return the module binds is not this case
            _tm_fn("pt_make", "pt_make", "pt_make", "Pt", "Any"),
            # ...and one it does not bind is
            _tm_fn("lost_make", "lost_make", "lost_make", "Lost", "Any"),
            _tm_fn("u128_echo", "u128_echo", "u128_echo", "unsigned __int128", "UInt128",
                   _tm_param("v", "unsigned __int128", "UInt128")),
            _tm_fn("ld_echo", "ld_echo", "ld_echo", "long double", "NTuple{2, UInt64}",
                   _tm_param("x", "long double", "Any"))],
        "globals" => Dict{String,Any}(),
        "struct_definitions" => Dict{String,Any}(
            "Pt" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x8",
                "members" => Any[_tm_mem("x", "int", "Cint", "0x0", 4), _tm_mem("y", "int", "Cint", "0x4", 4)])))
    @testset "$abi" for abi in (:sysv, :win64, :aapcs64)
        code = _TM_W.generate_introspective_module_c(cfg, _tm_libref(), meta, "TmU",
                                                     _TM_W.create_type_registry(cfg), false; abi)
        trap(name) = (b = _tm_body(code, name); occursin("ABI Safety Trap", b) && !occursin("ccall(", b))
        # Each unmapped function is a stub that names the C type...
        @test trap("f16_echo") && occursin("takes `_Float16` x by value and returns `_Float16`", _tm_body(code, "f16_echo"))
        @test trap("f16_bits") && occursin("takes `_Float16` x by value", _tm_body(code, "f16_bits"))
        @test trap("f16_make") && occursin("returns `_Float16`", _tm_body(code, "f16_make"))
        @test trap("odd_echo") && occursin("`unsigned __int256`", _tm_body(code, "odd_echo"))
        @test trap("lost_make") && occursin("returns `Lost`", _tm_body(code, "lost_make"))
        @test occursin("maps that C type to no Julia type", _tm_body(code, "f16_echo"))
        # ...`long double` keeps its own...
        @test trap("ld_echo") && occursin("80-bit x87", _tm_body(code, "ld_echo"))
        # ...and everything else is still a call.
        @test occursin("ccall((:plain_add, LIBRARY_PATH), Cint, (Cint, Cint,), a_c, b_c)", code)
        @test occursin("ccall((:pt_make, LIBRARY_PATH), Pt, (), )", code)
        # A 128-bit integer in the first registers is a call; under Win64 its
        # RETURN is the refusal (XMM0, where ccall expects a hidden pointer).
        @test occursin("ccall((:u128_echo, LIBRARY_PATH), UInt128, (UInt128,), v)", code) == (abi !== :win64)
        @test (trap("u128_echo") && occursin("128-bit integer", _tm_body(code, "u128_echo"))) == (abi === :win64)
        # `Any` reached no foreign-call position, and no C spelling a type position.
        @test !occursin(r"ccall\([^\n]*\bAny\b", code)
        @test !occursin("__int256,", code) && !occursin("_Float16,", code)
        @test isempty(_TM_W._undefined_ccall_types(code))
        @test _tm_parses(code)
        # The guards that used to refuse the whole wrapper pass.
        _TM_W._assert_wrapper_loadable(code, "TmU")
        @test count(n -> trap(n), ["plain_add", "pt_make", "u128_echo", "f16_echo", "f16_bits", "f16_make",
                                   "odd_echo", "lost_make", "ld_echo"]) == (abi === :win64 ? 7 : 6)
    end
end

@testset "a 128-bit integer through ccall: where it is C's placement" begin
    L, U = "Clong", "UInt128"
    mis(types, ret = "Cvoid"; abi, hidden = false, gpr = [_TM_W._c_gpr_upper(t) for t in types]) =
        _TM_W._c_int128_misplaced(types, gpr, ret, hidden; abi)
    # SysV: both halves need a register. Six integer registers.
    @test isempty(mis([U]; abi = :sysv))
    @test isempty(mis([U, U, U]; abi = :sysv))
    @test isempty(mis([L, L, L, L, U]; abi = :sysv))
    @test length(mis([L, L, L, L, L, U]; abi = :sysv)) == 1          # one left: C puts it on the stack whole
    @test length(mis([L, L, L, L, L, L, U]; abi = :sysv)) == 1       # none left
    @test length(mis([L, L, L, L, L, L, L, U]; abi = :sysv)) == 1
    @test length(mis([U, U, U, U]; abi = :sysv)) == 1                # the fourth
    @test occursin("parameter 6", only(mis([L, L, L, L, L, U]; abi = :sysv)))
    # A float takes no integer register; a pointer and a narrow integer take one.
    @test isempty(mis(["Cdouble", "Cfloat", L, L, L, L, U]; abi = :sysv))
    @test length(mis(["Ptr{Cvoid}", "Cstring", "Bool", "UInt8", L, "Int128"]; abi = :sysv)) == 1
    # The return's hidden pointer takes the first.
    @test isempty(mis([L, L, L, U]; abi = :sysv, hidden = true))
    @test length(mis([L, L, L, L, U]; abi = :sysv, hidden = true)) == 1
    # A record is counted by what it can take: the caller's bound, 2 when unknown.
    @test _TM_W._c_gpr_upper("SomeRecord") == 2
    @test length(mis([L, L, L, "SomeRecord", U]; abi = :sysv)) == 1
    @test isempty(mis([L, L, L, "SomeRecord", U]; abi = :sysv, gpr = [1, 1, 1, 0, 2]))   # a MEMORY record
    # The return comes back in RAX:RDX on both sides.
    @test isempty(mis([L], U; abi = :sysv)) && isempty(mis(String[], "Int128"; abi = :sysv))
    # Win64: by reference in (as ccall passes it), XMM0 out (where ccall expects a pointer).
    @test isempty(mis([L, L, L, L, L, L, L, U]; abi = :win64))
    @test length(mis([L], U; abi = :win64)) == 1
    @test length(mis(String[], "Int128"; abi = :win64)) == 1
    @test isempty(mis([U], "Cint"; abi = :win64))
    # AAPCS64: Julia passes `i128` itself.
    @test isempty(mis([L, L, L, L, L, L, L, U], U; abi = :aapcs64))
    # The Julia types the two C spellings map to are the ones asked about.
    @test Set(_TM_W._C_INT128_TYPES) == Set([_TM_C.cpp_to_julia_type("__int128"),
                                             _TM_C.cpp_to_julia_type("unsigned __int128")])
end

@testset "C: the generator refuses the 128-bit crossings ccall would misplace" begin
    cfg = _tm_config("c")
    long(n) = _tm_param(n, "long", "Clong")
    longs(k) = [long("a$i") for i in 1:k]
    u(n) = _tm_param(n, "unsigned __int128", "UInt128")
    fnu(name, params...) = _tm_fn(name, name, name, "unsigned __int128", "UInt128", params...)
    rec(size, members...) = Dict{String,Any}("kind" => "struct", "byte_size" => "0x$(string(size; base = 16))",
                                             "members" => Any[members...])
    meta = Dict{String,Any}(
        "functions" => Any[
            fnu("after4", longs(4)..., u("v")),
            fnu("after5", longs(5)..., u("v")),
            _tm_fn("i_after5", "i_after5", "i_after5", "__int128", "Int128", longs(5)..., _tm_param("v", "__int128", "Int128")),
            fnu("after_double", _tm_param("x", "double", "Cdouble"), longs(4)..., u("v")),
            # what the parameter before it can take: an enum one register, a record
            # in memory none, two INTEGER eightbytes two, two SSE eightbytes none
            fnu("after_enum", longs(4)..., _tm_param("k", "Small", "Small"), u("v")),
            fnu("after_big", longs(4)..., _tm_param("b", "Big", "Big"), u("v")),
            fnu("after_pair", longs(3)..., _tm_param("p", "Pair2", "Pair2"), u("v")),
            fnu("after_dpair", longs(4)..., _tm_param("p", "DPair", "DPair"), u("v")),
            # a record returned through a hidden pointer takes the first register
            _tm_fn("big_ret", "big_ret", "big_ret", "Big", "Big", longs(4)..., u("v")),
            _tm_fn("hi", "hi", "hi", "unsigned long long", "Culonglong", u("v"))],
        "globals" => Dict{String,Any}(),
        "struct_definitions" => Dict{String,Any}(
            "__enum__Small" => _tm_enum("unsigned int", "Cuint", 4, :SM_A => 0, :SM_B => 1),
            "Big" => rec(24, _tm_mem("a", "long", "Clong", "0x0", 8), _tm_mem("b", "long", "Clong", "0x8", 8),
                         _tm_mem("c", "long", "Clong", "0x10", 8)),
            "Pair2" => rec(16, _tm_mem("a", "long", "Clong", "0x0", 8), _tm_mem("b", "long", "Clong", "0x8", 8)),
            "DPair" => rec(16, _tm_mem("a", "double", "Cdouble", "0x0", 8), _tm_mem("b", "double", "Cdouble", "0x8", 8))))
    names = [String(f["name"]) for f in meta["functions"]]
    returning = filter(!in(("big_ret", "hi")), names)
    want = Dict(:sysv => ["after5", "after_enum", "after_pair", "big_ret", "i_after5"],
                :win64 => sort(returning), :aapcs64 => String[])
    @testset "$abi" for abi in (:sysv, :win64, :aapcs64)
        code = _TM_W.generate_introspective_module_c(cfg, _tm_libref(), meta, "TmI",
                                                     _TM_W.create_type_registry(cfg), false; abi)
        refused = sort!([n for n in names if occursin("128-bit integer", _tm_body(code, n))])
        @test refused == want[abi]
        for n in refused
            @test (n, occursin("ABI Safety Trap", _tm_body(code, n)) && !occursin("ccall(", _tm_body(code, n))) == (n, true)
        end
        # Where both halves get registers the call is plain.
        abi === :sysv && @test occursin("(Clong, Clong, Clong, Clong, UInt128,)", _tm_body(code, "after4"))
        abi === :sysv && @test occursin("(Clong, Clong, Clong, Clong, DPair, UInt128,)", _tm_body(code, "after_dpair"))
        @test count(n -> !isempty(_tm_body(code, n)), names) == length(names)
        @test _tm_parses(code)
        _TM_W._assert_wrapper_loadable(code, "TmI")
    end
end

# ── The C++ generator and the thunks ─────────────────────────────────────────

@testset "C++: a reference to an enum is a Ref, a pointer a pointer" begin
    cfg = _tm_config("cpp")
    meta = Dict{String,Any}(
        "functions" => Any[
            _tm_fn("color_out", "_Z9color_outR5Color", "color_out(Color&)", "void", "Cvoid",
                   _tm_param("c", "Color&", "Ref{Color}")),
            _tm_fn("color_is", "_Z8color_isRK5ColorS_", "color_is(Color const&, Color)", "bool", "Bool",
                   _tm_param("c", "const Color&", "Ref{Color}"), _tm_param("d", "Color", "Color")),
            _tm_fn("level_out", "_Z9level_outP5Level", "level_out(Level*)", "void", "Cvoid",
                   _tm_param("out", "Level*", "Ptr{Level}"); noexcept = true),
            _tm_fn("color_swap", "_Z10color_swapR5ColorS0_", "color_swap(Color&, Color&)", "void", "Cvoid",
                   _tm_param("a", "Color&", "Ref{Color}"), _tm_param("b", "Color&", "Ref{Color}"); noexcept = true),
            _tm_fn("color_store", "_Z11color_storev", "color_store()", "Color*", "Ptr{Cvoid}"),
            _tm_fn("color_first", "_Z11color_firstv", "color_first()", "Color&", "Ptr{Cvoid}"),
            _tm_fn("ref_int", "_Z7ref_intRi", "ref_int(int&)", "void", "Cvoid", _tm_param("x", "int&", "Ref{Cint}")),
            _tm_fn("palette_make", "_Z12palette_makev", "palette_make()", "Palette", "Palette")],
        "globals" => Dict{String,Any}(),
        "struct_definitions" => Dict{String,Any}(
            "__enum__Color" => _tm_enum("uint8_t", "UInt8", 1, :Red => 1, :Green => 2, :Blue => 250),
            "__enum__Level" => _tm_enum("int16_t", "Int16", 2, :Low => -300, :Mid => 0, :High => 300),
            "Palette" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x10",
                "members" => Any[_tm_mem("colors", "Color*", "Ptr{Color}", "0x0", 8),
                                 _tm_mem("n", "int", "Cint", "0x8", 4),
                                 _tm_mem("floor", "Level", "Level", "0xc", 2)])),
        "record_abi" => Dict{String,Any}())
    res = _TM_W.generate_introspective_module_cpp(cfg, _tm_libref(), meta, "TmCpp",
                                                  _TM_W.create_type_registry(cfg), false)
    code = res isa Tuple ? res[1] : res
    # A reference: annotated like `ref_int(x::Ref{Cint})`, and passed as an
    # address (`pointer_arg`, the one conversion of a Tier 2 pointer position),
    # never as the enum's integer.
    @test occursin(r"^function ref_int\(x::Ref\{Cint\}\)"m, code)
    @test occursin(r"^function color_out\(c::Ref\{Color\}\)"m, code)
    @test occursin("_Z9color_outR5Color_thunk\", RepliBuild.JITManager.pointer_arg(Ptr{Color}, c))", _tm_body(code, "color_out"))
    # The reference is the address; only the by-value enum is converted to its integer.
    @test occursin(r"^function color_is\(c::Ref\{Color\}, d::Any\)"m, code)
    @test occursin("Bool, RepliBuild.JITManager.pointer_arg(Ptr{Color}, c), UInt8(d))", _tm_body(code, "color_is"))
    @test !occursin("UInt8(c)", code)
    # On the ccall path the types say the same.
    @test occursin("(Ptr{Level},), out)", _tm_body(code, "level_out"))
    @test occursin("(Ref{Color}, Ref{Color},), a, b)", _tm_body(code, "color_swap"))
    # Returns, through `_typed_pointer_return`.
    @test occursin("Ptr{Color}", _tm_body(code, "color_store"))
    @test occursin("Ptr{Color}", _tm_body(code, "color_first"))
    @test !occursin("Ptr{Cvoid}", _tm_body(code, "color_store") * _tm_body(code, "color_first"))
    # A member.
    @test occursin(r"^    colors::Ptr\{Color\}$"m, code)
    @test occursin(r"^    floor::Level$"m, code)
    @test isempty(_TM_W._undefined_ccall_types(code))
    @test _tm_parses(code)
    # The thunk agrees: the reference and the pointer are addresses.
    for f in meta["functions"][1:2]
        tc = _TM_FG.thunk_crossings(f, meta["struct_definitions"], meta["record_abi"])
        @test tc.trap === nothing
        @test first(tc.params) == (:direct, nothing)
    end
end

@testset "C++: the character types and an enum with no enumerators cross as integers" begin
    cfg = _tm_config("cpp")
    structs = Dict{String,Any}(
        # `std::byte`, put back by the parser, and a library enum of the same shape
        "__enum__byte" => merge!(_tm_enum("unsigned char", "UInt8", 1), Dict{String,Any}("toolchain" => true, "scopes" => ["std"])),
        "__enum__Handle" => _tm_enum("uint32_t", "UInt32", 4),
        "__enum__Tag" => _tm_enum("short", "Cshort", 2))
    fns = Any[
        _tm_fn("u8_echo", "_Z7u8_echoDu", "u8_echo(char8_t)", "char8_t", "UInt8", _tm_param("c", "char8_t", "UInt8")),
        _tm_fn("c16_echo", "_Z8c16_echoDs", "c16_echo(char16_t)", "char16_t", "UInt16", _tm_param("c", "char16_t", "UInt16")),
        _tm_fn("c32_echo", "_Z8c32_echoDi", "c32_echo(char32_t)", "char32_t", "UInt32", _tm_param("c", "char32_t", "UInt32")),
        _tm_fn("byte_flip", "_Z9byte_flipSt4byte", "byte_flip(std::byte)", "byte", "byte", _tm_param("b", "byte", "byte")),
        _tm_fn("bytes_fill", "_Z10bytes_fillPSt4bytei", "bytes_fill(std::byte*, int)", "void", "Cvoid",
               _tm_param("p", "byte*", "Ptr{byte}"), _tm_param("n", "int", "Cint"); noexcept = true),
        _tm_fn("handle_next", "_Z11handle_next6Handle", "handle_next(Handle)", "Handle", "Handle",
               _tm_param("h", "Handle", "Handle")),
        _tm_fn("tag_next", "_Z8tag_next3Tag", "tag_next(Tag)", "Tag", "Tag", _tm_param("t", "Tag", "Tag"))]
    meta = Dict{String,Any}("functions" => fns, "globals" => Dict{String,Any}(),
                            "struct_definitions" => structs, "record_abi" => Dict{String,Any}())
    res = _TM_W.generate_introspective_module_cpp(cfg, _tm_libref(), meta, "TmChr",
                                                  _TM_W.create_type_registry(cfg), false)
    code = res isa Tuple ? res[1] : res
    # No function is a trap, and none says "record".
    @test !occursin("ABI Safety Trap", code)
    for (name, jl) in (("u8_echo", "UInt8"), ("c16_echo", "UInt16"), ("c32_echo", "UInt32"))
        @test occursin(Regex("^function $name\\(c::$jl\\)", "m"), code)
        @test occursin("_thunk\", $jl, c)", _tm_body(code, name))
    end
    # Each empty enum is a type of its own width, never an alias of another one...
    @test occursin(r"^primitive type byte <: RepliBuild\.OpenEnums\.OpenEnum\{UInt8\} 8 end"m, code)
    @test occursin(r"^primitive type Handle <: RepliBuild\.OpenEnums\.OpenEnum\{UInt32\} 32 end"m, code)
    @test occursin(r"^primitive type Tag <: RepliBuild\.OpenEnums\.OpenEnum\{Cshort\} 16 end"m, code)
    @test !occursin(r"^const (byte|Handle|Tag) ="m, code)
    @test !occursin(r"^(mutable )?struct (byte|Handle|Tag)\b"m, code)
    # ...and crosses as its integer, from an enum value or an integer.
    @test occursin("_thunk\", byte, UInt8(b))", _tm_body(code, "byte_flip"))
    @test occursin("_thunk\", Handle, UInt32(h))", _tm_body(code, "handle_next"))
    @test occursin("(Ptr{byte}, Cint,), p, n_c)", _tm_body(code, "bytes_fill"))
    @test isempty(_TM_W._undefined_ccall_types(code))
    @test _tm_parses(code)
    # The enum section evaluates: an empty enumerator table is a tuple.
    m = Module(:TmEnums)
    Core.eval(m, :(import RepliBuild))
    blocks = [String(x.match) for x in eachmatch(r"^primitive type \w+ <: .*?^\)$"ms, code)]
    @test length(blocks) == 3
    foreach(b -> Base.include_string(m, b), blocks)
    @test sizeof(m.byte) == 1 && sizeof(m.Handle) == 4 && sizeof(m.Tag) == 2
    @test UInt8(m.byte(0xf0)) == 0xf0 && m.byte(0x0f) != m.byte(0xf0)
    @test RepliBuild.OpenEnums.enumerators(m.byte) == ()
    @test string(m.Handle(7)) == "Handle(7)"                   # no declared name to print

    # The thunks: integers at their width, the narrow ones extended by their signedness.
    for abi in (:sysv, :win64, :aapcs64)
        ir = _TM_FG.generate_function_thunks(fns, structs; may_throw = true,
                                             record_abi = Dict{String,Any}(), abi)
        decl(sym) = (m = match(Regex("func\\.func private @" * sym * "\\(([^\\n]*)\\) -> (\\S+)"), ir);
                     m === nothing ? "no declaration" : "$(m.captures[1]) -> $(m.captures[2])")
        @test (abi, decl("_Z7u8_echoDu")) == (abi, "i32 -> i8")
        @test (abi, decl("_Z8c16_echoDs")) == (abi, "i32 -> i16")
        @test (abi, decl("_Z8c32_echoDi")) == (abi, "i32 -> i32")
        @test (abi, decl("_Z9byte_flipSt4byte")) == (abi, "i32 -> i8")
        # (An enum's RETURN is spelled from its Julia base type, which this
        # generator has not changed; the argument is what was a record.)
        @test (abi, first(split(decl("_Z11handle_next6Handle"), " -> "))) == (abi, "i32")
        @test (abi, first(split(decl("_Z8tag_next3Tag"), " -> "))) == (abi, "i32")
        body(sym) = (i = findfirst("func.func @" * sym * "_thunk", ir); i === nothing ? "" :
                     ir[i.start:something(findnext("\n}\n", ir, i.start), i.start:lastindex(ir)).stop])
        exts(sym) = [x.captures[1] * " " * x.captures[2] for x in eachmatch(r"llvm\.(zext|sext) %val_\d+ : (i\d+) to i32", body(sym))]
        @test (abi, exts("_Z7u8_echoDu")) == (abi, ["zext i8"])
        @test (abi, exts("_Z8c16_echoDs")) == (abi, ["zext i16"])
        @test (abi, exts("_Z8c32_echoDi")) == (abi, String[])
        @test (abi, exts("_Z9byte_flipSt4byte")) == (abi, ["zext i8"])
        @test (abi, exts("_Z8tag_next3Tag")) == (abi, ["sext i16"])
        # None crosses as a record.
        @test (abi, occursin(r"!llvm\.struct<\"(char8_t|char16_t|char32_t|byte|Handle|Tag)\"[^\n]*\n[^\n]*_call", ir)) == (abi, false)
    end
    for f in fns
        @test (f["name"], _TM_FG.thunk_crossings(f, structs, Dict{String,Any}()).trap) == (f["name"], nothing)
    end
end

@testset "C++: a scalar the map does not know, on the ccall path, is one trap stub" begin
    # A `noexcept` or C-linkage function is a plain ccall, and its parameter was
    # declared `(Any,)` there too: the same refusal of the whole wrapper.
    cfg = _tm_config("cpp")
    odd = _tm_param("v", "unsigned __int256", "Any")
    meta = Dict{String,Any}(
        "functions" => Any[
            _tm_fn("odd_take", "_Z8odd_takeo", "odd_take(unsigned __int256)", "int", "Cint", odd; noexcept = true),
            # C linkage reaches ccall only with a `nounwind` definition
            # (`DispatchLogic._is_nounwind`); without the key it takes a thunk.
            merge!(_tm_fn("c_take", "c_take", "c_take", "int", "Cint", odd), Dict{String,Any}("nounwind" => true)),
            _tm_fn("plain", "_Z5plaini", "plain(int)", "int", "Cint", _tm_param("a", "int", "Cint"); noexcept = true)],
        "globals" => Dict{String,Any}(), "struct_definitions" => Dict{String,Any}(),
        "record_abi" => Dict{String,Any}())
    res = _TM_W.generate_introspective_module_cpp(cfg, _tm_libref(), meta, "TmAny",
                                                  _TM_W.create_type_registry(cfg), false)
    code = res isa Tuple ? res[1] : res
    for name in ("odd_take", "c_take")
        b = _tm_body(code, name)
        @test (name, occursin("ABI Safety Trap", b) && !occursin("ccall(", b)) == (name, true)
        @test (name, occursin("takes `unsigned __int256` v by value", b)) == (name, true)
    end
    @test occursin("ccall((:_Z5plaini, LIBRARY_PATH), Cint, (Cint,), a_c)", _tm_body(code, "plain"))
    @test !occursin(r"ccall\([^\n]*\bAny\b", code)
    @test _tm_parses(code)
    _TM_W._assert_wrapper_loadable(code, "TmAny")
end

@testset "C++: a 128-bit integer by value takes the thunk, signed or not" begin
    f(c, jl) = _tm_fn("w", "_Z1wx", "w(x)", "void", "Cvoid", _tm_param("v", c, jl); noexcept = true)
    safe(fn) = _TM_W.is_ccall_safe(fn, Dict{String,Any}())
    @test !safe(f("unsigned __int128", "UInt128"))
    @test !safe(f("__int128", "Int128"))
    @test !safe(_tm_fn("r", "_Z1rv", "r()", "unsigned __int128", "UInt128"; noexcept = true))
    # A pointer to one is a pointer.
    @test safe(f("unsigned __int128*", "Ptr{UInt128}"))
    @test safe(f("int", "Cint"))
    # The thunk passes `i128`, which is clang's placement on SysV and AAPCS64.
    # MinGW's callee takes a pointer and returns in XMM0: refused there, by name.
    tc(fn, abi) = _TM_FG.thunk_crossings(fn, Dict{String,Any}(), Dict{String,Any}(); abi)
    ret = _tm_fn("r", "_Z1rv", "r()", "__int128", "Int128")
    for fn in (f("unsigned __int128", "UInt128"), f("const __int128", "Int128"), ret)
        @test tc(fn, :sysv).trap === nothing
        @test tc(fn, :aapcs64).trap === nothing
        @test occursin("128-bit integer", something(tc(fn, :win64).abi_trap, ""))
        @test tc(fn, :win64).layout_trap === nothing
    end
    @test occursin("takes `unsigned __int128` by value", tc(f("unsigned __int128", "UInt128"), :win64).abi_trap)
    @test occursin("returns `__int128` by value", tc(ret, :win64).abi_trap)
    @test tc(f("unsigned __int128*", "Ptr{UInt128}"), :win64).trap === nothing
    @test tc(f("long long", "Clonglong"), :win64).trap === nothing
end

end
