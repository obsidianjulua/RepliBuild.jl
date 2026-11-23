# =============================================================================
# C++ globals outside the global namespace (B19, 2026-11-18)
#
# A global's identity is the symbol the loader binds. `cfg::depth` is
# `_ZN3cfg5depthE`; the metadata keyed it by its bare DWARF name `depth`,
# `_drop_unresolvable_globals!` looked that up in the dynamic symbol table and
# dropped it, and a static data member (declared in its class, defined with a
# `DW_AT_specification` pointing there and no name of its own) was not
# recorded at all. Now the parser keys each exported variable by its
# `DW_AT_linkage_name` (else its name) with its scope and access
# (`Compiler._library_variables`), the wrap drops private and protected
# members and reports constants with no symbol, and GeneratorCpp names the
# accessor by the qualified name (`geo_Box_count`), with the pointer accessor
# only for a type it cannot hold by value.
#
# Library-free. The two dumps are GNU readelf's `--debug-dump=info` of
#
#     namespace geo {
#     class Counter { static int total_; public: static int pub; static constexpr int K = 3; static int get(); };
#     struct Box { static int count; protected: static int prot_; };
#     }
#     int geo::Counter::total_ = 1; int geo::Counter::pub = 2; int geo::Counter::get(){ return total_ + K; }
#     int geo::Box::count = 4; int geo::Box::prot_ = 5;
#     namespace a { int x = 7; } namespace b { int x = 8; }
#     int g_top = 9;
#
# compiled by clang 23 with `-g -gdwarf-5 -c` and `-g -gdwarf-4 -c`, with the
# producer and compilation-directory lines cut. DWARF 5 declares a static
# member as a `DW_TAG_variable` in its class, DWARF 4 as a `DW_TAG_member`
# with `DW_AT_external`; both give one table. test_cpp_globals_live.jl builds,
# wraps and calls a library of this shape.
# =============================================================================

using Test
using RepliBuild
using Libdl

const _CG_C = RepliBuild.Compiler
const _CG_W = RepliBuild.Wrapper
const _CG_LIBREF = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))

const CG_DUMP5 = raw"""Contents of the .debug_info section:

  Compilation Unit @ offset 0:
   Length:        0xca (32-bit)
   Version:       5
   Unit Type:     DW_UT_compile (1)
   Abbrev Offset: 0
   Pointer Size:  8
 <0><c>: Abbrev Number: 1 (DW_TAG_compile_unit)
    <e>   DW_AT_language    : 33	(C++14)
    <10>   DW_AT_name        : (indexed string: 0x1): sm.cpp
    <11>   DW_AT_str_offsets_base: 0x8
    <15>   DW_AT_stmt_list   : 0
    <1a>   DW_AT_low_pc      : (index: 0): 0
    <1b>   DW_AT_high_pc     : 0xf
    <1f>   DW_AT_addr_base   : 0x8
 <1><23>: Abbrev Number: 2 (DW_TAG_namespace)
    <24>   DW_AT_name        : (indexed string: 0x3): geo
 <2><25>: Abbrev Number: 3 (DW_TAG_class_type)
    <26>   DW_AT_calling_convention: 5	(pass by value)
    <27>   DW_AT_name        : (indexed string: 0xa): Counter
    <28>   DW_AT_byte_size   : 1
    <29>   DW_AT_decl_file   : 0
    <2a>   DW_AT_decl_line   : 2
 <3><2b>: Abbrev Number: 4 (DW_TAG_variable)
    <2c>   DW_AT_name        : (indexed string: 0x4): total_
    <2d>   DW_AT_type        : <0x6a>
    <31>   DW_AT_decl_file   : 0
    <32>   DW_AT_decl_line   : 2
    <33>   DW_AT_external    : 1
    <33>   DW_AT_declaration : 1
 <3><33>: Abbrev Number: 5 (DW_TAG_variable)
    <34>   DW_AT_name        : (indexed string: 0x6): pub
    <35>   DW_AT_type        : <0x6a>
    <39>   DW_AT_decl_file   : 0
    <3a>   DW_AT_decl_line   : 2
    <3b>   DW_AT_external    : 1
    <3b>   DW_AT_declaration : 1
    <3b>   DW_AT_accessibility: 1	(public)
 <3><3c>: Abbrev Number: 6 (DW_TAG_variable)
    <3d>   DW_AT_name        : (indexed string: 0x7): K
    <3e>   DW_AT_type        : <0x6e>
    <42>   DW_AT_decl_file   : 0
    <43>   DW_AT_decl_line   : 2
    <44>   DW_AT_external    : 1
    <44>   DW_AT_declaration : 1
    <44>   DW_AT_accessibility: 1	(public)
    <45>   DW_AT_const_value : 3
 <3><46>: Abbrev Number: 7 (DW_TAG_subprogram)
    <47>   DW_AT_linkage_name: (indexed string: 0x8): _ZN3geo7Counter3getEv
    <48>   DW_AT_name        : (indexed string: 0x9): get
    <49>   DW_AT_decl_file   : 0
    <4a>   DW_AT_decl_line   : 2
    <4b>   DW_AT_type        : <0x6a>
    <4f>   DW_AT_declaration : 1
    <4f>   DW_AT_external    : 1
    <4f>   DW_AT_accessibility: 1	(public)
 <3><50>: Abbrev Number: 0
 <2><51>: Abbrev Number: 8 (DW_TAG_structure_type)
    <52>   DW_AT_calling_convention: 5	(pass by value)
    <53>   DW_AT_name        : (indexed string: 0xf): Box
    <54>   DW_AT_byte_size   : 1
    <55>   DW_AT_decl_file   : 0
    <56>   DW_AT_decl_line   : 3
 <3><57>: Abbrev Number: 4 (DW_TAG_variable)
    <58>   DW_AT_name        : (indexed string: 0xd): count
    <59>   DW_AT_type        : <0x6a>
    <5d>   DW_AT_decl_file   : 0
    <5e>   DW_AT_decl_line   : 3
    <5f>   DW_AT_external    : 1
    <5f>   DW_AT_declaration : 1
 <3><5f>: Abbrev Number: 5 (DW_TAG_variable)
    <60>   DW_AT_name        : (indexed string: 0xe): prot_
    <61>   DW_AT_type        : <0x6a>
    <65>   DW_AT_decl_file   : 0
    <66>   DW_AT_decl_line   : 3
    <67>   DW_AT_external    : 1
    <67>   DW_AT_declaration : 1
    <67>   DW_AT_accessibility: 2	(protected)
 <3><68>: Abbrev Number: 0
 <2><69>: Abbrev Number: 0
 <1><6a>: Abbrev Number: 9 (DW_TAG_base_type)
    <6b>   DW_AT_name        : (indexed string: 0x5): int
    <6c>   DW_AT_encoding    : 5	(signed)
    <6d>   DW_AT_byte_size   : 4
 <1><6e>: Abbrev Number: 10 (DW_TAG_const_type)
    <6f>   DW_AT_type        : <0x6a>
 <1><73>: Abbrev Number: 11 (DW_TAG_subprogram)
    <74>   DW_AT_low_pc      : (index: 0): 0
    <75>   DW_AT_high_pc     : 0xf
    <79>   DW_AT_frame_base  : 1 byte block: 56 	(DW_OP_reg6 (rbp))
    <7b>   DW_AT_decl_line   : 5
    <7c>   DW_AT_specification: <0x46>
 <1><80>: Abbrev Number: 12 (DW_TAG_variable)
    <81>   DW_AT_specification: <0x2b>
    <85>   DW_AT_location    :	(DW_OP_addrx <0x1>)
    <88>   DW_AT_linkage_name: (indexed string: 0xb): _ZN3geo7Counter6total_E
 <1><89>: Abbrev Number: 12 (DW_TAG_variable)
    <8a>   DW_AT_specification: <0x33>
    <8e>   DW_AT_location    :	(DW_OP_addrx <0x2>)
    <91>   DW_AT_linkage_name: (indexed string: 0xc): _ZN3geo7Counter3pubE
 <1><92>: Abbrev Number: 12 (DW_TAG_variable)
    <93>   DW_AT_specification: <0x57>
    <97>   DW_AT_location    :	(DW_OP_addrx <0x3>)
    <9a>   DW_AT_linkage_name: (indexed string: 0x10): _ZN3geo3Box5countE
 <1><9b>: Abbrev Number: 12 (DW_TAG_variable)
    <9c>   DW_AT_specification: <0x5f>
    <a0>   DW_AT_location    :	(DW_OP_addrx <0x4>)
    <a3>   DW_AT_linkage_name: (indexed string: 0x11): _ZN3geo3Box5prot_E
 <1><a4>: Abbrev Number: 2 (DW_TAG_namespace)
    <a5>   DW_AT_name        : (indexed string: 0x12): a
 <2><a6>: Abbrev Number: 13 (DW_TAG_variable)
    <a7>   DW_AT_name        : (indexed string: 0x13): x
    <a8>   DW_AT_type        : <0x6a>
    <ac>   DW_AT_external    : 1
    <ac>   DW_AT_decl_file   : 0
    <ad>   DW_AT_decl_line   : 7
    <ae>   DW_AT_location    :	(DW_OP_addrx <0x5>)
    <b1>   DW_AT_linkage_name: (indexed string: 0x14): _ZN1a1xE
 <2><b2>: Abbrev Number: 0
 <1><b3>: Abbrev Number: 2 (DW_TAG_namespace)
    <b4>   DW_AT_name        : (indexed string: 0x15): b
 <2><b5>: Abbrev Number: 13 (DW_TAG_variable)
    <b6>   DW_AT_name        : (indexed string: 0x13): x
    <b7>   DW_AT_type        : <0x6a>
    <bb>   DW_AT_external    : 1
    <bb>   DW_AT_decl_file   : 0
    <bc>   DW_AT_decl_line   : 7
    <bd>   DW_AT_location    :	(DW_OP_addrx <0x6>)
    <c0>   DW_AT_linkage_name: (indexed string: 0x16): _ZN1b1xE
 <2><c1>: Abbrev Number: 0
 <1><c2>: Abbrev Number: 14 (DW_TAG_variable)
    <c3>   DW_AT_name        : (indexed string: 0x17): g_top
    <c4>   DW_AT_type        : <0x6a>
    <c8>   DW_AT_external    : 1
    <c8>   DW_AT_decl_file   : 0
    <c9>   DW_AT_decl_line   : 8
    <ca>   DW_AT_location    :	(DW_OP_addrx <0x7>)
 <1><cd>: Abbrev Number: 0
"""

const CG_DUMP4 = raw"""Contents of the .debug_info section:

  Compilation Unit @ offset 0:
   Length:        0x14b (32-bit)
   Version:       4
   Abbrev Offset: 0
   Pointer Size:  8
 <0><b>: Abbrev Number: 1 (DW_TAG_compile_unit)
    <10>   DW_AT_language    : 33	(C++14)
    <12>   DW_AT_name        : (indirect string, offset: 0x15): sm.cpp
    <16>   DW_AT_stmt_list   : 0
    <1e>   DW_AT_low_pc      : 0
    <26>   DW_AT_high_pc     : 0xf
 <1><2a>: Abbrev Number: 2 (DW_TAG_namespace)
    <2b>   DW_AT_name        : (indirect string, offset: 0x80): geo
 <2><2f>: Abbrev Number: 3 (DW_TAG_class_type)
    <30>   DW_AT_calling_convention: 5	(pass by value)
    <31>   DW_AT_name        : (indirect string, offset: 0xaf): Counter
    <35>   DW_AT_byte_size   : 1
    <36>   DW_AT_decl_file   : 1
    <37>   DW_AT_decl_line   : 2
 <3><38>: Abbrev Number: 4 (DW_TAG_member)
    <39>   DW_AT_name        : (indirect string, offset: 0x84): total_
    <3d>   DW_AT_type        : <0x8f>
    <41>   DW_AT_decl_file   : 1
    <42>   DW_AT_decl_line   : 2
    <43>   DW_AT_external    : 1
    <43>   DW_AT_declaration : 1
 <3><43>: Abbrev Number: 5 (DW_TAG_member)
    <44>   DW_AT_name        : (indirect string, offset: 0x8f): pub
    <48>   DW_AT_type        : <0x8f>
    <4c>   DW_AT_decl_file   : 1
    <4d>   DW_AT_decl_line   : 2
    <4e>   DW_AT_external    : 1
    <4e>   DW_AT_declaration : 1
    <4e>   DW_AT_accessibility: 1	(public)
 <3><4f>: Abbrev Number: 6 (DW_TAG_member)
    <50>   DW_AT_name        : (indirect string, offset: 0x93): K
    <54>   DW_AT_type        : <0x96>
    <58>   DW_AT_decl_file   : 1
    <59>   DW_AT_decl_line   : 2
    <5a>   DW_AT_external    : 1
    <5a>   DW_AT_declaration : 1
    <5a>   DW_AT_accessibility: 1	(public)
    <5b>   DW_AT_const_value : 3
 <3><5c>: Abbrev Number: 7 (DW_TAG_subprogram)
    <5d>   DW_AT_linkage_name: (indirect string, offset: 0x95): _ZN3geo7Counter3getEv
    <61>   DW_AT_name        : (indirect string, offset: 0xab): get
    <65>   DW_AT_decl_file   : 1
    <66>   DW_AT_decl_line   : 2
    <67>   DW_AT_type        : <0x8f>
    <6b>   DW_AT_declaration : 1
    <6b>   DW_AT_external    : 1
    <6b>   DW_AT_accessibility: 1	(public)
 <3><6c>: Abbrev Number: 0
 <2><6d>: Abbrev Number: 8 (DW_TAG_structure_type)
    <6e>   DW_AT_calling_convention: 5	(pass by value)
    <6f>   DW_AT_name        : (indirect string, offset: 0xf0): Box
    <73>   DW_AT_byte_size   : 1
    <74>   DW_AT_decl_file   : 1
    <75>   DW_AT_decl_line   : 3
 <3><76>: Abbrev Number: 4 (DW_TAG_member)
    <77>   DW_AT_name        : (indirect string, offset: 0xe4): count
    <7b>   DW_AT_type        : <0x8f>
    <7f>   DW_AT_decl_file   : 1
    <80>   DW_AT_decl_line   : 3
    <81>   DW_AT_external    : 1
    <81>   DW_AT_declaration : 1
 <3><81>: Abbrev Number: 5 (DW_TAG_member)
    <82>   DW_AT_name        : (indirect string, offset: 0xea): prot_
    <86>   DW_AT_type        : <0x8f>
    <8a>   DW_AT_decl_file   : 1
    <8b>   DW_AT_decl_line   : 3
    <8c>   DW_AT_external    : 1
    <8c>   DW_AT_declaration : 1
    <8c>   DW_AT_accessibility: 2	(protected)
 <3><8d>: Abbrev Number: 0
 <2><8e>: Abbrev Number: 0
 <1><8f>: Abbrev Number: 9 (DW_TAG_base_type)
    <90>   DW_AT_name        : (indirect string, offset: 0x8b): int
    <94>   DW_AT_encoding    : 5	(signed)
    <95>   DW_AT_byte_size   : 4
 <1><96>: Abbrev Number: 10 (DW_TAG_const_type)
    <97>   DW_AT_type        : <0x8f>
 <1><9b>: Abbrev Number: 11 (DW_TAG_subprogram)
    <9c>   DW_AT_low_pc      : 0
    <a4>   DW_AT_high_pc     : 0xf
    <a8>   DW_AT_frame_base  : 1 byte block: 56 	(DW_OP_reg6 (rbp))
    <aa>   DW_AT_decl_line   : 5
    <ab>   DW_AT_specification: <0x5c>
 <1><af>: Abbrev Number: 12 (DW_TAG_variable)
    <b0>   DW_AT_specification: <0x38>
    <b4>   DW_AT_location    : 9 byte block: 3 0 0 0 0 0 0 0 0 	(DW_OP_addr: 0)
    <be>   DW_AT_linkage_name: (indirect string, offset: 0xb7): _ZN3geo7Counter6total_E
 <1><c2>: Abbrev Number: 12 (DW_TAG_variable)
    <c3>   DW_AT_specification: <0x43>
    <c7>   DW_AT_location    : 9 byte block: 3 4 0 0 0 0 0 0 0 	(DW_OP_addr: 4)
    <d1>   DW_AT_linkage_name: (indirect string, offset: 0xcf): _ZN3geo7Counter3pubE
 <1><d5>: Abbrev Number: 12 (DW_TAG_variable)
    <d6>   DW_AT_specification: <0x76>
    <da>   DW_AT_location    : 9 byte block: 3 8 0 0 0 0 0 0 0 	(DW_OP_addr: 8)
    <e4>   DW_AT_linkage_name: (indirect string, offset: 0xf4): _ZN3geo3Box5countE
 <1><e8>: Abbrev Number: 12 (DW_TAG_variable)
    <e9>   DW_AT_specification: <0x81>
    <ed>   DW_AT_location    : 9 byte block: 3 c 0 0 0 0 0 0 0 	(DW_OP_addr: c)
    <f7>   DW_AT_linkage_name: (indirect string, offset: 0x107): _ZN3geo3Box5prot_E
 <1><fb>: Abbrev Number: 2 (DW_TAG_namespace)
    <fc>   DW_AT_name        : (indirect string, offset: 0x11a): a
 <2><100>: Abbrev Number: 13 (DW_TAG_variable)
    <101>   DW_AT_name        : (indirect string, offset: 0x11c): x
    <105>   DW_AT_type        : <0x8f>
    <109>   DW_AT_external    : 1
    <109>   DW_AT_decl_file   : 1
    <10a>   DW_AT_decl_line   : 7
    <10b>   DW_AT_location    : 9 byte block: 3 10 0 0 0 0 0 0 0 	(DW_OP_addr: 10)
    <115>   DW_AT_linkage_name: (indirect string, offset: 0x11e): _ZN1a1xE
 <2><119>: Abbrev Number: 0
 <1><11a>: Abbrev Number: 2 (DW_TAG_namespace)
    <11b>   DW_AT_name        : (indirect string, offset: 0x127): b
 <2><11f>: Abbrev Number: 13 (DW_TAG_variable)
    <120>   DW_AT_name        : (indirect string, offset: 0x11c): x
    <124>   DW_AT_type        : <0x8f>
    <128>   DW_AT_external    : 1
    <128>   DW_AT_decl_file   : 1
    <129>   DW_AT_decl_line   : 7
    <12a>   DW_AT_location    : 9 byte block: 3 14 0 0 0 0 0 0 0 	(DW_OP_addr: 14)
    <134>   DW_AT_linkage_name: (indirect string, offset: 0x129): _ZN1b1xE
 <2><138>: Abbrev Number: 0
 <1><139>: Abbrev Number: 14 (DW_TAG_variable)
    <13a>   DW_AT_name        : (indirect string, offset: 0x132): g_top
    <13e>   DW_AT_type        : <0x8f>
    <142>   DW_AT_external    : 1
    <142>   DW_AT_decl_file   : 1
    <143>   DW_AT_decl_line   : 8
    <144>   DW_AT_location    : 9 byte block: 3 18 0 0 0 0 0 0 0 	(DW_OP_addr: 18)
 <1><14e>: Abbrev Number: 0
"""

const CG_SYMBOLS = ["_ZN1a1xE", "_ZN1b1xE", "_ZN3geo3Box5countE", "_ZN3geo3Box5prot_E",
                    "_ZN3geo7Counter3pubE", "_ZN3geo7Counter6total_E", "g_top"]

function _cg_config(dir::String, lang::String)
    toml = joinpath(dir, "replibuild.toml")
    write(toml, """
    [project]
    name = "cppglobalsynth"
    root = "$(escape_string(dir))"

    [link]
    enable_lto = false

    [wrap]
    language = "$lang"
    """)
    return RepliBuild.ConfigurationManager.load_config(toml)
end

function _cg_generate(lang::Symbol, md)
    dir = mktempdir()
    try
        cfg = _cg_config(dir, lang === :cpp ? "cpp" : "c")
        gen = lang === :cpp ? _CG_W.generate_introspective_module_cpp :
                              _CG_W.generate_introspective_module_c
        out = redirect_stdout(devnull) do
            gen(cfg, _CG_LIBREF, md, "CppGlobalSynth", _CG_W.create_type_registry(cfg), false)
        end
        return out isa Tuple ? String(first(out)) : String(out)
    finally
        rm(dir; recursive = true, force = true)
    end
end

_cg_metadata(globals) = Dict{String,Any}("functions" => Any[], "struct_definitions" => Dict{String,Any}(),
                                         "globals" => globals)

@testset "the parser keys a global by its symbol, with its scope" begin
    tables = Dict{String,Any}()
    for (label, dump) in (("DWARF 5", CG_DUMP5), ("DWARF 4", CG_DUMP4))
        _, sd, gv, _, _ = _CG_C.parse_dwarf_dump(dump)
        tables[label] = gv
        @testset "$label" begin
            @test sort!([k for (k, g) in gv if !get(g, "constant", false)]) == CG_SYMBOLS
            entry(sym) = (gv[sym]["name"], gv[sym]["scope"], gv[sym]["symbol"])
            @test entry("_ZN1a1xE") == ("x", "a", "_ZN1a1xE")
            @test entry("_ZN1b1xE") == ("x", "b", "_ZN1b1xE")
            @test entry("_ZN3geo3Box5countE") == ("count", "geo::Box", "_ZN3geo3Box5countE")
            @test entry("_ZN3geo7Counter3pubE") == ("pub", "geo::Counter", "_ZN3geo7Counter3pubE")
            @test entry("g_top") == ("g_top", "", "g_top")
            @test gv["_ZN3geo3Box5countE"]["julia_type"] == "Cint"
            # Access: written, or the record's default (private in a class).
            @test get(gv["_ZN3geo7Counter6total_E"], "accessibility", "public") == "private"
            @test get(gv["_ZN3geo3Box5prot_E"], "accessibility", "public") == "protected"
            @test !haskey(gv["_ZN3geo7Counter3pubE"], "accessibility")
            @test !haskey(gv["_ZN3geo3Box5countE"], "accessibility")
            # A static data member says so: a namespace and a class are both
            # a scope (`geo::Box`, `cfg`), and only a member must be declared
            # inside its class (test_tls_globals.jl's accessor plan).
            @test get(gv["_ZN3geo3Box5countE"], "record_member", false) == true
            @test get(gv["_ZN3geo7Counter3pubE"], "record_member", false) == true
            @test !haskey(gv["_ZN1a1xE"], "record_member") && !haskey(gv["g_top"], "record_member")
            # A constant with no storage: named, never given a symbol.
            @test get(gv["geo::Counter::K"], "constant", false) == true
            @test gv["geo::Counter::K"]["symbol"] === nothing
        end
    end
    @test length(tables) == 2
    @test tables["DWARF 5"] == tables["DWARF 4"]
end

@testset "the wrap screens globals on the symbol" begin
    _, _, gv, _, _ = _CG_C.parse_dwarf_dump(CG_DUMP5)
    delete!(gv, "geo::Counter::K")
    md = _cg_metadata(gv)
    @test _CG_W._drop_nonpublic_globals!(md) == 2
    @test sort!(collect(keys(md["globals"]))) == ["_ZN1a1xE", "_ZN1b1xE", "_ZN3geo3Box5countE",
                                                   "_ZN3geo7Counter3pubE", "g_top"]
    # Resolved through the symbol, not the bare name.
    @test _CG_W._drop_unresolvable_globals!(md, Set(CG_SYMBOLS)) == 0
    @test _CG_W._drop_unresolvable_globals!(md, Set(["_ZN1a1xE", "g_top", "x", "count", "pub"])) == 3
    @test sort!(collect(keys(md["globals"]))) == ["_ZN1a1xE", "g_top"]
    # Metadata older than `symbol` keys a global by its symbol already.
    old = _cg_metadata(Dict{String,Any}("g_old" => Dict{String,Any}("julia_type" => "Cint")))
    @test _CG_W._drop_unresolvable_globals!(old, Set(["g_old"])) == 0
    @test _CG_W._drop_nonpublic_globals!(old) == 0
    # `[wrap] exclude_symbols` takes the qualified name too.
    _, _, gv2, _, _ = _CG_C.parse_dwarf_dump(CG_DUMP5)
    md2 = _cg_metadata(gv2)
    r = _CG_W._apply_symbol_filter!(md2, ["geo::Box::*", "_ZN1a1xE"])
    @test r.globals_dropped == 3 && isempty(r.unmatched)
    @test !any(k -> startswith(k, "_ZN3geo3Box") || k == "_ZN1a1xE", keys(md2["globals"]))
end

@testset "GeneratorCpp names an accessor by its scope and binds the symbol" begin
    _, _, gv, _, _ = _CG_C.parse_dwarf_dump(CG_DUMP5)
    delete!(gv, "geo::Counter::K")
    md = _cg_metadata(gv)
    _CG_W._drop_nonpublic_globals!(md)
    # A type the wrapper cannot hold by value, as `std::string` is spelled.
    md["globals"]["_Z8g_stringB5cxx11"] = Dict{String,Any}(
        "name" => "g_string", "scope" => "", "symbol" => "_Z8g_stringB5cxx11",
        "c_type" => "basic_string<char, std::char_traits<char>, std::allocator<char> >",
        "julia_type" => "basic_string<char, std::char_traits<char>, std::allocator<char> >")
    code = _cg_generate(:cpp, md)

    for (fname, sym) in (("a_x", "_ZN1a1xE"), ("b_x", "_ZN1b1xE"), ("geo_Box_count", "_ZN3geo3Box5countE"),
                         ("geo_Counter_pub", "_ZN3geo7Counter3pubE"), ("g_top", "g_top"))
        @test occursin("\nfunction $(fname)()::Cint\n    ptr = cglobal((:$(sym), LIBRARY_PATH), Cint)\n", code)
        @test occursin("\nfunction $(fname)_ptr()::Ptr{Cint}\n    return cglobal((:$(sym), LIBRARY_PATH), Cint)\n", code)
    end
    @test occursin("Get value of global variable `geo::Box::count`.", code)
    @test occursin("\nfunction g_string_ptr()::Ptr{Cvoid}\n    return cglobal((:_Z8g_stringB5cxx11, LIBRARY_PATH))\n", code)
    @test !occursin("function g_string()", code)
    @test !occursin("total_", code) && !occursin("prot_", code)

    # The module loads with every accessor bound and none for what was screened.
    m = Module(:CppGlobalSynthProbe)
    Base.include_string(m, code)
    mod = Base.invokelatest(getfield, m, :CppGlobalSynth)
    has(s) = Base.invokelatest(isdefined, mod, s)
    @test all(has, (:a_x, :a_x_ptr, :b_x, :geo_Box_count, :geo_Counter_pub, :g_top, :g_string_ptr))
    @test !has(:g_string) && !has(:x) && !has(:geo_Box_prot_) && !has(:geo_Counter_total_)
end

@testset "a C wrapper's globals are what they were" begin
    # The new keys (`symbol`, `scope`) change nothing GeneratorC emits.
    plain = Dict{String,Any}(
        "g_top" => Dict{String,Any}("name" => "g_top", "c_type" => "int", "julia_type" => "Cint"),
        "g_name" => Dict{String,Any}("name" => "g_name", "c_type" => "const char*", "julia_type" => "Ptr{UInt8}"))
    keyed = deepcopy(plain)
    for (k, g) in keyed
        g["symbol"] = k
        g["scope"] = ""
    end
    a = _cg_generate(:c, _cg_metadata(plain))
    b = _cg_generate(:c, _cg_metadata(keyed))
    strip_time(s) = replace(s, r"# Generated: .*\n" => "", r"\"generated_at\" => \"[^\"]*\"" => "")
    @test occursin("cglobal((:g_top, LIBRARY_PATH), Cint)", a)
    @test strip_time(a) == strip_time(b)
end
