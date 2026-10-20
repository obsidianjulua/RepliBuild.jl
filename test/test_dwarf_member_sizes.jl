# =============================================================================
# DWARF member sizes through typedefs, enums, qualifiers and arrays
# (library-free, CI-safe)
#
# `parse_dwarf_dump` sized a member with `get_type_size(c_type)`, which knows only
# spelled primitives, and fell back to the member's own DIE only for a direct
# struct/class/union or an array of primitives. A member whose type is an enum,
# a typedef of a record or enum, a `const` record, or an array of records came out
# with size 0. GeneratorCpp's layout proof then gave up and emitted the enclosing
# record as a byte blob; StructGen's packed test (sum of member sizes) and its
# fallback sizes read the same zeros.
#
# `_die_byte_size` follows the DIE chain instead. The dump below is GNU objdump
# `--dwarf=info` of this C file, compiled with clang 23 (`-g -O0 -c`), with the
# producer and compilation directory shortened:
#
#     typedef enum { E_A, E_B } E;
#     enum K { K_A = 1 };
#     typedef struct V2 { float x, y; } V2;
#     typedef struct { int a; } Anon;
#     struct Rec { E e; enum K k; V2 v; Anon an; V2 arr[2]; const V2 cv; E earr[3]; };
#     struct Rec make_rec(void) { struct Rec r = {0}; return r; }
#
# Before the fix every member of Rec came out with size 0.
# =============================================================================

using Test
using RepliBuild

const _MSZ = RepliBuild.Compiler

const MEMBER_SIZE_DUMP = raw"""

  Compilation Unit @ offset 0:
   Length:        0x161 (32-bit)
   Version:       4
   Abbrev Offset: 0
   Pointer Size:  8
 <0><b>: Abbrev Number: 1 (DW_TAG_compile_unit)
    <c>   DW_AT_producer    : (indirect string, offset: 0): clang version 23.1.3
    <10>   DW_AT_language    : 29	(C11)
    <12>   DW_AT_name        : (indirect string, offset: 0x68): rec.c
    <16>   DW_AT_stmt_list   : 0
    <1a>   DW_AT_comp_dir    : (indirect string, offset: 0x6e): C:/src
    <1e>   DW_AT_low_pc      : 0
    <26>   DW_AT_high_pc     : 0x28
 <1><2a>: Abbrev Number: 2 (DW_TAG_enumeration_type)
    <2b>   DW_AT_type        : <0x3f>
    <2f>   DW_AT_byte_size   : 4
    <30>   DW_AT_decl_file   : 1
    <31>   DW_AT_decl_line   : 1
 <2><32>: Abbrev Number: 3 (DW_TAG_enumerator)
    <33>   DW_AT_name        : (indirect string, offset: 0x12c): E_A
    <37>   DW_AT_const_value : 0
 <2><38>: Abbrev Number: 3 (DW_TAG_enumerator)
    <39>   DW_AT_name        : (indirect string, offset: 0x130): E_B
    <3d>   DW_AT_const_value : 1
 <2><3e>: Abbrev Number: 0
 <1><3f>: Abbrev Number: 4 (DW_TAG_base_type)
    <40>   DW_AT_name        : (indirect string, offset: 0x11f): unsigned int
    <44>   DW_AT_encoding    : 7	(unsigned)
    <45>   DW_AT_byte_size   : 4
 <1><46>: Abbrev Number: 5 (DW_TAG_enumeration_type)
    <47>   DW_AT_type        : <0x3f>
    <4b>   DW_AT_name        : (indirect string, offset: 0x138): K
    <4f>   DW_AT_byte_size   : 4
    <50>   DW_AT_decl_file   : 1
    <51>   DW_AT_decl_line   : 2
 <2><52>: Abbrev Number: 3 (DW_TAG_enumerator)
    <53>   DW_AT_name        : (indirect string, offset: 0x134): K_A
    <57>   DW_AT_const_value : 1
 <2><58>: Abbrev Number: 0
 <1><59>: Abbrev Number: 6 (DW_TAG_subprogram)
    <5a>   DW_AT_low_pc      : 0
    <62>   DW_AT_high_pc     : 0x28
    <66>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <68>   DW_AT_name        : (indirect string, offset: 0x13a): make_rec
    <6c>   DW_AT_decl_file   : 1
    <6d>   DW_AT_decl_line   : 14
    <6e>   DW_AT_prototyped  : 1
    <6e>   DW_AT_type        : <0x82>
    <72>   DW_AT_external    : 1
 <2><72>: Abbrev Number: 7 (DW_TAG_variable)
    <73>   DW_AT_location    : 3 byte block: 91 30 6 	(DW_OP_fbreg: 48; DW_OP_deref)
    <77>   DW_AT_name        : (indirect string, offset: 0x18a): r
    <7b>   DW_AT_decl_file   : 1
    <7c>   DW_AT_decl_line   : 14
    <7d>   DW_AT_type        : <0x82>
 <2><81>: Abbrev Number: 0
 <1><82>: Abbrev Number: 8 (DW_TAG_structure_type)
    <83>   DW_AT_name        : (indirect string, offset: 0x186): Rec
    <87>   DW_AT_byte_size   : 56
    <88>   DW_AT_decl_file   : 1
    <89>   DW_AT_decl_line   : 5
 <2><8a>: Abbrev Number: 9 (DW_TAG_member)
    <8b>   DW_AT_name        : (indirect string, offset: 0x143): e
    <8f>   DW_AT_type        : <0xdf>
    <93>   DW_AT_decl_file   : 1
    <94>   DW_AT_decl_line   : 6
    <95>   DW_AT_data_member_location: 0
 <2><96>: Abbrev Number: 9 (DW_TAG_member)
    <97>   DW_AT_name        : (indirect string, offset: 0x147): k
    <9b>   DW_AT_type        : <0x46>
    <9f>   DW_AT_decl_file   : 1
    <a0>   DW_AT_decl_line   : 7
    <a1>   DW_AT_data_member_location: 4
 <2><a2>: Abbrev Number: 9 (DW_TAG_member)
    <a3>   DW_AT_name        : (indirect string, offset: 0x149): v
    <a7>   DW_AT_type        : <0xea>
    <ab>   DW_AT_decl_file   : 1
    <ac>   DW_AT_decl_line   : 8
    <ad>   DW_AT_data_member_location: 8
 <2><ae>: Abbrev Number: 9 (DW_TAG_member)
    <af>   DW_AT_name        : (indirect string, offset: 0x158): an
    <b3>   DW_AT_type        : <0x11d>
    <b7>   DW_AT_decl_file   : 1
    <b8>   DW_AT_decl_line   : 9
    <b9>   DW_AT_data_member_location: 16
 <2><ba>: Abbrev Number: 9 (DW_TAG_member)
    <bb>   DW_AT_name        : (indirect string, offset: 0x166): arr
    <bf>   DW_AT_type        : <0x140>
    <c3>   DW_AT_decl_file   : 1
    <c4>   DW_AT_decl_line   : 10
    <c5>   DW_AT_data_member_location: 20
 <2><c6>: Abbrev Number: 9 (DW_TAG_member)
    <c7>   DW_AT_name        : (indirect string, offset: 0x17e): cv
    <cb>   DW_AT_type        : <0x153>
    <cf>   DW_AT_decl_file   : 1
    <d0>   DW_AT_decl_line   : 11
    <d1>   DW_AT_data_member_location: 36
 <2><d2>: Abbrev Number: 9 (DW_TAG_member)
    <d3>   DW_AT_name        : (indirect string, offset: 0x181): earr
    <d7>   DW_AT_type        : <0x158>
    <db>   DW_AT_decl_file   : 1
    <dc>   DW_AT_decl_line   : 12
    <dd>   DW_AT_data_member_location: 44
 <2><de>: Abbrev Number: 0
 <1><df>: Abbrev Number: 10 (DW_TAG_typedef)
    <e0>   DW_AT_type        : <0x2a>
    <e4>   DW_AT_name        : (indirect string, offset: 0x145): E
    <e8>   DW_AT_decl_file   : 1
    <e9>   DW_AT_decl_line   : 1
 <1><ea>: Abbrev Number: 10 (DW_TAG_typedef)
    <eb>   DW_AT_type        : <0xf5>
    <ef>   DW_AT_name        : (indirect string, offset: 0x155): V2
    <f3>   DW_AT_decl_file   : 1
    <f4>   DW_AT_decl_line   : 3
 <1><f5>: Abbrev Number: 8 (DW_TAG_structure_type)
    <f6>   DW_AT_name        : (indirect string, offset: 0x155): V2
    <fa>   DW_AT_byte_size   : 8
    <fb>   DW_AT_decl_file   : 1
    <fc>   DW_AT_decl_line   : 3
 <2><fd>: Abbrev Number: 9 (DW_TAG_member)
    <fe>   DW_AT_name        : (indirect string, offset: 0x14b): x
    <102>   DW_AT_type        : <0x116>
    <106>   DW_AT_decl_file   : 1
    <107>   DW_AT_decl_line   : 3
    <108>   DW_AT_data_member_location: 0
 <2><109>: Abbrev Number: 9 (DW_TAG_member)
    <10a>   DW_AT_name        : (indirect string, offset: 0x153): y
    <10e>   DW_AT_type        : <0x116>
    <112>   DW_AT_decl_file   : 1
    <113>   DW_AT_decl_line   : 3
    <114>   DW_AT_data_member_location: 4
 <2><115>: Abbrev Number: 0
 <1><116>: Abbrev Number: 4 (DW_TAG_base_type)
    <117>   DW_AT_name        : (indirect string, offset: 0x14d): float
    <11b>   DW_AT_encoding    : 4	(float)
    <11c>   DW_AT_byte_size   : 4
 <1><11d>: Abbrev Number: 10 (DW_TAG_typedef)
    <11e>   DW_AT_type        : <0x128>
    <122>   DW_AT_name        : (indirect string, offset: 0x161): Anon
    <126>   DW_AT_decl_file   : 1
    <127>   DW_AT_decl_line   : 4
 <1><128>: Abbrev Number: 11 (DW_TAG_structure_type)
    <129>   DW_AT_byte_size   : 4
    <12a>   DW_AT_decl_file   : 1
    <12b>   DW_AT_decl_line   : 4
 <2><12c>: Abbrev Number: 9 (DW_TAG_member)
    <12d>   DW_AT_name        : (indirect string, offset: 0x15b): a
    <131>   DW_AT_type        : <0x139>
    <135>   DW_AT_decl_file   : 1
    <136>   DW_AT_decl_line   : 4
    <137>   DW_AT_data_member_location: 0
 <2><138>: Abbrev Number: 0
 <1><139>: Abbrev Number: 4 (DW_TAG_base_type)
    <13a>   DW_AT_name        : (indirect string, offset: 0x15d): int
    <13e>   DW_AT_encoding    : 5	(signed)
    <13f>   DW_AT_byte_size   : 4
 <1><140>: Abbrev Number: 12 (DW_TAG_array_type)
    <141>   DW_AT_type        : <0xea>
 <2><145>: Abbrev Number: 13 (DW_TAG_subrange_type)
    <146>   DW_AT_type        : <0x14c>
    <14a>   DW_AT_count       : 2
 <2><14b>: Abbrev Number: 0
 <1><14c>: Abbrev Number: 14 (DW_TAG_base_type)
    <14d>   DW_AT_name        : (indirect string, offset: 0x16a): __ARRAY_SIZE_TYPE__
    <151>   DW_AT_byte_size   : 8
    <152>   DW_AT_encoding    : 7	(unsigned)
 <1><153>: Abbrev Number: 15 (DW_TAG_const_type)
    <154>   DW_AT_type        : <0xea>
 <1><158>: Abbrev Number: 12 (DW_TAG_array_type)
    <159>   DW_AT_type        : <0xdf>
 <2><15d>: Abbrev Number: 13 (DW_TAG_subrange_type)
    <15e>   DW_AT_type        : <0x14c>
    <162>   DW_AT_count       : 3
 <2><163>: Abbrev Number: 0
 <1><164>: Abbrev Number: 0

"""

@testset "DWARF member sizes" begin

    @testset "every member of Rec has its type's size" begin
        _, sd, _, _, _ = _MSZ.parse_dwarf_dump(MEMBER_SIZE_DUMP)
        @test haskey(sd, "Rec")
        got = Dict(String(m["name"]) => m["size"] for m in sd["Rec"]["members"])
        want = Dict("e" => 4,      # typedef of an anonymous enum
                    "k" => 4,      # enum
                    "v" => 8,      # typedef of a tagged record
                    "an" => 4,     # typedef of an anonymous record
                    "arr" => 16,   # array of a typedef'd record
                    "cv" => 8,     # const typedef'd record
                    "earr" => 12)  # array of a typedef'd enum
        checked = 0
        for (name, size) in sort(collect(want); by = first)
            @test (name, get(got, name, :missing)) == (name, size)
            checked += 1
        end
        @test checked == length(want) == length(got)
        # Members of the records themselves were sized before and still are.
        @test [m["size"] for m in sd["V2"]["members"]] == [4, 4]
        # The record is natural, now provably so from the member sizes too.
        @test sum(m["size"] for m in sd["Rec"]["members"]) == 56
    end

    @testset "_die_byte_size follows the chain" begin
        refs = Dict{String,Any}(
            "0x1" => "int",
            "0x2" => Dict{String,Any}("kind" => "typedef", "name" => "T", "target" => "0x1"),
            "0x3" => Dict{String,Any}("kind" => "struct", "name" => "S", "byte_size" => "0x18", "members" => []),
            "0x4" => Dict{String,Any}("kind" => "const", "target" => "0x3"),
            "0x5" => Dict{String,Any}("kind" => "array", "element_type" => "0x4", "dimensions" => [2, 3]),
            "0x6" => Dict{String,Any}("kind" => "pointer", "target" => "0x3"),
            "0x7" => Dict{String,Any}("kind" => "enum", "name" => "E", "byte_size" => "0x2"),
            "0x8" => Dict{String,Any}("kind" => "volatile", "target" => "0x7"),
            "0x9" => Dict{String,Any}("kind" => "typedef", "name" => "Loop", "target" => "0x9"),
            "0xa" => Dict{String,Any}("kind" => "array", "element_type" => "0x3", "dimensions" => []),
            "0xb" => "char16_t")
        sz(ref; kw...) = _MSZ._die_byte_size(ref, refs; kw...)
        @test sz("0x1") == 4
        @test sz("0x2") == 4
        @test sz("0x3") == 24
        @test sz("0x4") == 24
        @test sz("0x5") == 144
        @test sz("0x6") == 8
        @test sz("0x8") == 2
        @test sz("0x9") == 0          # a cycle ends unmeasured, it does not recurse
        @test sz("0xa") == 0          # no extent: an incomplete array
        @test sz("0xb") == 0          # a base type the name table does not know …
        @test sz("0xb"; base_sizes = Dict("0xb" => 2)) == 2   # … sized by its DIE
        @test sz("0xffff") == 0
        @test sz(nothing) == 0
    end
end
