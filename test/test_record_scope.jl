#!/usr/bin/env julia
# test/test_record_scope.jl — a record carries the scope it is declared in, and
# the C++ generator's STL screen reads it (2026-10-31).
#
# DWARF keys every record by its bare name, and the generator's STL screen
# matches bare names (`basic_string_view<`, `pair<`, `vector<`). So
# `boost::core::basic_string_view<char>` was screened like `std::string_view`:
# not emitted, its by-value parameters `::Any` with no Julia type to construct,
# a by-value return of it `Ptr{Cvoid}` (Boost.Charconv, Boost.JSON). The parser
# now writes each record's `scope` (the namespaces and records enclosing it,
# from the DIE tree), and a record outside `std` is never STL-internal.
#
# No toolchain: a synthetic readelf dump through `parse_dwarf_dump`, then the
# screen and the generator on synthetic metadata.

using Test
using Libdl
using RepliBuild

const _RS_C  = RepliBuild.Compiler
const _RS_W  = RepliBuild.Wrapper

_rs_mm(name, jt, ct, off, size) = Dict{String,Any}("name" => name, "julia_type" => jt, "c_type" => ct,
                                                   "offset" => off, "size" => size)

@testset "A record's scope" begin

    # boost::core::basic_string_view<char>, std::__1::pair<int, int>, a global
    # G, and a record local to a function (its subprogram sets no context, so
    # the parent depth holds a stale record: no scope, rather than G's).
    dump = """
     <0><b>: Abbrev Number: 1 (DW_TAG_compile_unit)
        <c>   DW_AT_producer    : (indexed string: 0x0): clang version 23.1.4
        <d>   DW_AT_language    : 33	(C++14)
     <1><20>: Abbrev Number: 2 (DW_TAG_namespace)
        <21>   DW_AT_name        : (indexed string: 0x1): boost
     <2><25>: Abbrev Number: 2 (DW_TAG_namespace)
        <26>   DW_AT_name        : (indexed string: 0x2): core
     <3><2a>: Abbrev Number: 3 (DW_TAG_class_type)
        <2b>   DW_AT_name        : (indexed string: 0x3): basic_string_view<char>
        <2c>   DW_AT_byte_size   : 16
     <4><30>: Abbrev Number: 4 (DW_TAG_member)
        <31>   DW_AT_name        : (indexed string: 0x4): p_
        <32>   DW_AT_type        : <0x90>
        <33>   DW_AT_data_member_location: 0
     <4><38>: Abbrev Number: 4 (DW_TAG_member)
        <39>   DW_AT_name        : (indexed string: 0x5): n_
        <3a>   DW_AT_type        : <0x98>
        <3b>   DW_AT_data_member_location: 8
     <4><40>: Abbrev Number: 0
     <3><41>: Abbrev Number: 0
     <2><42>: Abbrev Number: 0
     <1><50>: Abbrev Number: 2 (DW_TAG_namespace)
        <51>   DW_AT_name        : (indexed string: 0x6): std
     <2><55>: Abbrev Number: 2 (DW_TAG_namespace)
        <56>   DW_AT_name        : (indexed string: 0x7): __1
     <3><5a>: Abbrev Number: 3 (DW_TAG_structure_type)
        <5b>   DW_AT_name        : (indexed string: 0x8): pair<int, int>
        <5c>   DW_AT_byte_size   : 8
     <4><60>: Abbrev Number: 4 (DW_TAG_member)
        <61>   DW_AT_name        : (indexed string: 0x9): first
        <62>   DW_AT_type        : <0x80>
        <63>   DW_AT_data_member_location: 0
     <4><68>: Abbrev Number: 0
     <3><69>: Abbrev Number: 0
     <2><6a>: Abbrev Number: 0
     <1><70>: Abbrev Number: 3 (DW_TAG_structure_type)
        <71>   DW_AT_name        : (indexed string: 0xa): G
        <72>   DW_AT_byte_size   : 4
     <2><75>: Abbrev Number: 4 (DW_TAG_member)
        <76>   DW_AT_name        : (indexed string: 0xb): g
        <77>   DW_AT_type        : <0x80>
        <78>   DW_AT_data_member_location: 0
     <2><79>: Abbrev Number: 0
     <1><80>: Abbrev Number: 5 (DW_TAG_base_type)
        <81>   DW_AT_name        : (indexed string: 0xc): int
        <82>   DW_AT_encoding    : 5	(signed)
        <83>   DW_AT_byte_size   : 4
     <1><90>: Abbrev Number: 6 (DW_TAG_pointer_type)
        <91>   DW_AT_type        : <0x95>
     <1><95>: Abbrev Number: 7 (DW_TAG_const_type)
        <96>   DW_AT_type        : <0xa0>
     <1><98>: Abbrev Number: 5 (DW_TAG_base_type)
        <99>   DW_AT_name        : (indexed string: 0xd): unsigned long long
        <9a>   DW_AT_encoding    : 7	(unsigned)
        <9b>   DW_AT_byte_size   : 8
     <1><a0>: Abbrev Number: 5 (DW_TAG_base_type)
        <a1>   DW_AT_name        : (indexed string: 0xe): char
        <a2>   DW_AT_encoding    : 6	(signed char)
        <a3>   DW_AT_byte_size   : 1
     <1><b0>: Abbrev Number: 8 (DW_TAG_subprogram)
        <b1>   DW_AT_name        : (indexed string: 0xf): f
        <b2>   DW_AT_external    : 1
     <2><b5>: Abbrev Number: 3 (DW_TAG_structure_type)
        <b6>   DW_AT_name        : (indexed string: 0x10): Local
        <b7>   DW_AT_byte_size   : 4
     <3><ba>: Abbrev Number: 4 (DW_TAG_member)
        <bb>   DW_AT_name        : (indexed string: 0x11): x
        <bc>   DW_AT_type        : <0x80>
        <bd>   DW_AT_data_member_location: 0
     <3><be>: Abbrev Number: 0
     <2><bf>: Abbrev Number: 0
     <1><c0>: Abbrev Number: 0
    """

    @testset "the parser writes it from the DIE tree" begin
        _, structs, _, _ = _RS_C.parse_dwarf_dump(dump)
        @test haskey(structs, "basic_string_view<char>")
        @test get(structs["basic_string_view<char>"], "scope", nothing) == "boost::core"
        @test get(structs["pair<int, int>"], "scope", nothing) == "std::__1"
        @test get(structs["G"], "scope", nothing) == ""
        @test haskey(structs, "Local") && !haskey(structs["Local"], "scope")
        # the members still landed on their records (the namespace context
        # must not take a member)
        @test [m["name"] for m in structs["basic_string_view<char>"]["members"]] == ["p_", "n_"]
        @test [m["name"] for m in structs["G"]["members"]] == ["g"]
    end

    @testset "the screen spares a record outside std" begin
        TC = _RS_W
        sv(scope) = Dict{String,Any}("basic_string_view<char>" => Dict{String,Any}(
            "kind" => "class", "byte_size" => "0x10", "scope" => scope,
            "members" => Any[_rs_mm("p_", "Ptr{UInt8}", "const char*", "0x0", 8)]))
        @test TC._is_stl_internal_type("basic_string_view<char>")                 # the name alone
        @test !TC._is_stl_internal_record("basic_string_view<char>", sv("boost::core"))
        @test !TC._is_stl_internal_record("Ptr{basic_string_view<char>}", sv("boost::core"))
        @test !TC._is_stl_internal_record("const basic_string_view<char>&", sv("boost::core"))
        @test !TC._is_stl_internal_record("basic_string_view<char>", sv(""))      # global scope
        @test TC._is_stl_internal_record("basic_string_view<char>", sv("std::__1"))
        @test TC._is_stl_internal_record("basic_string_view<char>", sv("std"))
        @test TC._is_stl_internal_record("basic_string_view<char>", sv("__gnu_cxx"))
        # no scope (older metadata), no definition: the name decides as before
        noscope = Dict{String,Any}("basic_string_view<char>" => Dict{String,Any}("kind" => "class", "members" => Any[]))
        @test TC._is_stl_internal_record("basic_string_view<char>", noscope)
        @test TC._is_stl_internal_record("basic_string_view<char>", Dict{String,Any}())
        @test TC._is_stl_internal_record("basic_string_view<char>", nothing)
        @test !TC._is_stl_internal_record("Widget", Dict{String,Any}())
        # a std record the name never screened is still not screened
        @test !TC._is_stl_internal_record("duration<long long>", Dict{String,Any}(
            "duration<long long>" => Dict{String,Any}("kind" => "class", "scope" => "std::__1::chrono", "members" => Any[])))
    end

    @testset "the generator emits it and types the parameter" begin
        dir = mktempdir()
        write(joinpath(dir, "replibuild.toml"), """
        [project]
        name = "scopesynth"
        root = "$(escape_string(dir))"
        [wrap]
        language = "cpp"
        [types]
        strictness = "warn"
        allow_unknown_structs = true
        [cache]
        enabled = false
        """)
        cfg = RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml"))
        libref = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
        function meta(scope)
            Dict{String,Any}(
                "functions" => Any[Dict{String,Any}(
                    "name" => "ns::take", "mangled" => "_ZN2ns4takeEN5boost4core17basic_string_viewIcEE",
                    "demangled" => "ns::take(boost::core::basic_string_view<char>)",
                    "class" => "", "is_method" => false, "is_vararg" => false, "is_noexcept" => false,
                    "exported" => true, "has_object_pointer" => nothing,
                    "parameters" => Any[Dict{String,Any}("name" => "sv", "c_type" => "basic_string_view<char>",
                                                         "julia_type" => "basic_string_view<char>")],
                    "return_type" => Dict{String,Any}("c_type" => "int", "julia_type" => "Cint"))],
                "globals" => Dict{String,Any}(),
                "struct_definitions" => Dict{String,Any}("basic_string_view<char>" => Dict{String,Any}(
                    "kind" => "class", "byte_size" => "0x10", "scope" => scope,
                    "members" => Any[_rs_mm("p_", "Ptr{UInt8}", "const char*", "0x0", 8),
                                     _rs_mm("n_", "Csize_t", "size_t", "0x8", 8)])),
                "record_abi" => Dict{String,Any}("basic_string_view<char>" => Dict{String,Any}(
                    "byte_size" => "0x10", "empty" => false, "pass" => "value")))
        end
        gen(scope) = begin
            res = _RS_W.generate_introspective_module_cpp(cfg, libref, meta(scope), "ScopeSynth",
                                                          _RS_W.create_type_registry(cfg), true)
            res isa Tuple ? res[1] : res
        end
        boost = gen("boost::core")
        @test occursin(r"^struct basic_string_view_char\n"m, boost)
        @test occursin(r"^function \w*take\(sv::basic_string_view_char\)"m, boost)
        std = gen("std::__1")
        @test !occursin(r"^struct basic_string_view_char\n"m, std)
        @test occursin(r"^function \w*take\(sv::Any\)"m, std)
    end
end
