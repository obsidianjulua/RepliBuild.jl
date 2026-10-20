# =============================================================================
# Tier 3 by-value record classification (library-free, CI-safe)
#
# Where C returns a record decides the shape of the ccall GeneratorC emits: C
# returns it in registers or through a hidden pointer, and where Julia's own
# lowering of the emitted type would not pass that pointer, the wrapper passes
# it explicitly (BRANCH 0). Getting it wrong in either direction shifts every
# argument and returns garbage, with no error.
#
# C1 (abi_fuzz, seeds 3 and 36): `{ char m1; short m2[3]; }` is 8 bytes and
# comes back in RAX on both ABIs, but the generator took each member's SIZE as
# its alignment, judged the 6-byte array at offset 2 "unaligned", and passed a
# hidden pointer the callee never reads. The same test sat in the blob-param
# trap and the blob-return guard, and none of the three knew Win64, which
# classifies by size alone.
#
# The register-class traps are per target too. A ≤16-byte record whose Julia
# stand-in cannot reproduce its member types (the byte blob of a float-bearing
# record, or a struct holding a mixed float/integer region) travels SysV in
# eightbytes the stand-in misclasses, so it is refused there. Win64 gives a
# record no register class: C and Julia both cross it by size alone, so the
# traps only cost callable functions (31 in the abi_fuzz corpus; Hub mpack's
# 31 `mpack_tag_t` stubs).
#
# A hidden return buffer must have the record's alignment, and Julia's own
# lowering aligns it to the Julia type standing in for the record. A callee
# returning a 16-aligned record stores it with movaps, so a byte blob's buffer
# (alignment 1) faulted on every call. The record's alignment comes from
# DW_AT_alignment (the parser now records it) and its members' types; where the
# stand-in is less aligned, the wrapper passes a buffer it aligned itself.
#
# Every answer is checked for BOTH conventions (`abi = :sysv` / `:win64`), on
# any host. The DWARF tables below are in the shape the parser writes: hex
# offsets, `size = 0` for record and enum members, bitfields with no offset.
# What only a real SysV callee can show (Julia's own lowering of these types)
# needs a Linux host; test_c_sret_abi.jl (devtests) runs the live calls here
# and checks this classifier against clang's lowering for both targets.
# =============================================================================

using Test
using RepliBuild
using Libdl

const _SW = RepliBuild.Wrapper

_sm(name, jt, ct, off, size) = Dict{String,Any}("name" => name, "julia_type" => jt, "c_type" => ct,
                                                "offset" => off, "size" => size)
_sbit(name, jt, ct, bits, at) = Dict{String,Any}("name" => name, "julia_type" => jt, "c_type" => ct,
                                                 "offset" => nothing, "size" => 4,
                                                 "bit_size" => bits, "data_bit_offset" => at)
_srec(size, members...; kind = "struct") =
    Dict{String,Any}("kind" => kind, "byte_size" => "0x" * string(size, base = 16), "members" => Any[members...])
_sanon(rec) = (rec["anonymous"] = true; rec)
# DW_AT_alignment, on a record or a member, as the parser records it.
_salign(d, n) = (d["alignment"] = n; d)

const _SRET_STRUCTS = Dict{String,Any}(
    # C1, seed 36: 8 bytes, `short[3]` 2-aligned at offset 2. RAX on both ABIs.
    "T4" => _srec(8, _sm("m1", "UInt8", "char", "0x0", 1),
                     _sm("m2", "NTuple{3, Cshort}", "short[3]", "0x2", 6)),
    "R5" => _srec(4, _sm("m1", "Cint", "int", "0x0", 4)),
    # 16 bytes, `char[12]` 1-aligned at 4: two INTEGER eightbytes on SysV, sret
    # on Win64 (not 1/2/4/8). The size test called it unaligned (12 → align 8).
    "IC12" => _srec(16, _sm("n", "Cint", "int", "0x0", 4),
                        _sm("s", "NTuple{12, UInt8}", "char[12]", "0x4", 12)),
    # Packed, unaligned member, 4 and 8 bytes: RAX on Win64, MEMORY on SysV.
    "P4" => _srec(4, _sm("c", "UInt8", "char", "0x0", 1),
                     _sm("s", "Cshort", "short", "0x1", 2),
                     _sm("d", "UInt8", "char", "0x3", 1)),
    "P8" => _srec(8, _sm("c", "UInt8", "char", "0x0", 1),
                     _sm("i", "Cint", "int", "0x1", 4),
                     _sm("pad", "NTuple{3, UInt8}", "char[3]", "0x5", 3)),
    # Over 16 bytes: memory on both. Big24 and IS20 are natural, so the plain
    # ccall's own lowering passes the hidden pointer (the old size test called
    # IS20 packed: `short[8]` as align 8). PK17 is packed, so BRANCH 0 does.
    "Big24" => _srec(24, _sm("a", "Clonglong", "long long", "0x0", 8),
                         _sm("b", "Clonglong", "long long", "0x8", 8),
                         _sm("c", "Cint", "int", "0x10", 4)),
    "IS20" => _srec(20, _sm("n", "Cint", "int", "0x0", 4),
                        _sm("s", "NTuple{8, Cshort}", "short[8]", "0x4", 16)),
    "PK17" => _srec(17, _sm("c", "UInt8", "char", "0x0", 1),
                        _sm("x", "Clonglong", "long long", "0x1", 8),
                        _sm("y", "Clonglong", "long long", "0x9", 8)),
    # An unaligned field inside a packed member record. `In3` is align 1, so
    # `Out6` lays out naturally and the size test (3 + 3 bytes, 3-aligned)
    # agrees with DWARF: `is_c_lto_safe` passes it. `y` is at absolute 1.
    "In3" => _srec(3, _sm("x", "UInt8", "char", "0x0", 1),
                      _sm("y", "Cshort", "short", "0x1", 2)),
    "Out6" => _srec(6, _sm("in", "In3", "In3", "0x0", 0),
                       _sm("c", "NTuple{3, UInt8}", "char[3]", "0x3", 3)),
    # An array of a packed record: element 0's `y` lands at 2 (aligned),
    # element 1's at 5. Every element has to be walked.
    "ArrIn" => _srec(7, _sm("c", "UInt8", "char", "0x0", 1),
                        _sm("v", "NTuple{2, In3}", "In3[2]", "0x1", 0)),
    # Enum members carry size 0; E2 is a 2-byte enum.
    "EnumOk" => _srec(4, _sm("c", "UInt8", "char", "0x0", 1),
                         _sm("e", "E2", "E2", "0x2", 0)),
    "EnumPacked" => _srec(3, _sm("c", "UInt8", "char", "0x0", 1),
                             _sm("e", "E2", "E2", "0x1", 0)),
    "__enum__E2" => Dict{String,Any}("kind" => "enum", "byte_size" => "0x2", "julia_type" => "Cushort",
                                     "underlying_type" => "unsigned short",
                                     "enumerators" => Any[Dict{String,Any}("name" => "E2_A", "value" => 0)]),
    # Bitfields are skipped, as clang skips them for this rule.
    "BitOnly" => _srec(4, _sm("c", "UInt8", "char", "0x0", 1), _sbit("b", "Cint", "int", 7, 9)),
    # Float-bearing bitfield records are blobs whose SysV register class is
    # unknowable: 12 bytes (sret on Win64) and 8 bytes (RAX on Win64).
    "BF12" => _srec(12, _sm("f", "Cfloat", "float", "0x0", 4),
                        _sbit("b", "Cint", "int", 3, 32),
                        _sm("c", "Cint", "int", "0x8", 4)),
    "BF8" => _srec(8, _sm("f", "Cfloat", "float", "0x0", 4), _sbit("b", "Cint", "int", 3, 32)),
    # mpack_tag_t's shape: an anonymous union of an integer and a double
    # becomes an aligned `NTuple{1, UInt64}` region, which claims INTEGER for an
    # eightbyte a float member also covers. The region and the struct holding
    # it are `abi_class_risk`. An all-float region states its class (Float64).
    "Tag16_v" => _sanon(_srec(8, _sm("i", "Clonglong", "long long", "0x0", 8),
                                 _sm("d", "Cdouble", "double", "0x0", 8); kind = "union")),
    "Tag16" => _srec(16, _sm("type", "Cint", "int", "0x0", 4),
                         _sm("v", "Tag16_v", "Tag16_v", "0x8", 8)),
    "FV8_v" => _sanon(_srec(8, _sm("f", "NTuple{2, Cfloat}", "float[2]", "0x0", 8),
                               _sm("d", "Cdouble", "double", "0x0", 8); kind = "union")),
    # Members at offset 0 of a union.
    "U3" => _srec(4, _sm("i", "Cint", "int", "0x0", 4), _sm("c", "NTuple{3, UInt8}", "char[3]", "0x0", 3);
                  kind = "union"),
    # More aligned than the Julia type standing in for them: `aligned(N)` on the
    # record, `_Alignas` or a vector on a member (the parser records both as
    # the member's alignment), `long double`, a nested aligned record. A hidden
    # return buffer must have the record's alignment.
    "V4A" => _salign(_srec(16, _sm("x", "Cfloat", "float", "0x0", 4), _sm("y", "Cfloat", "float", "0x4", 4),
                               _sm("z", "Cfloat", "float", "0x8", 4), _sm("w", "Cfloat", "float", "0xc", 4)), 16),
    "D4A" => _salign(_srec(32, _sm("a", "Cdouble", "double", "0x0", 8), _sm("b", "Cdouble", "double", "0x8", 8),
                               _sm("c", "Cdouble", "double", "0x10", 8), _sm("d", "Cdouble", "double", "0x18", 8)), 16),
    "A32" => _salign(_srec(32, _sm("d", "NTuple{2, Cdouble}", "double[2]", "0x0", 16)), 32),
    "AB16" => _srec(16, _salign(_sm("a", "Cfloat", "float", "0x0", 4), 16), _sm("b", "Cfloat", "float", "0x4", 4),
                        _sbit("c", "Cint", "int", 3, 64)),
    "HVec" => _srec(32, _salign(_sm("v", "NTuple{4, Cfloat}", "float[4]", "0x0", 16), 16),
                        _sm("k", "Cint", "int", "0x10", 4)),
    "LD16" => _srec(16, _sm("x", "NTuple{2, UInt64}", "long double", "0x0", 16)),
    "NestA" => _srec(32, _sm("a", "V4A", "V4A", "0x0", 0), _sm("t", "Cfloat", "float", "0x10", 4)),
    # A member type nothing can measure: the size bounds the alignment.
    "Mys48" => _srec(48, _sm("q", "Any", "Mystery", "0x0", 0)),
    "Mys24" => _srec(24, _sm("q", "Any", "Mystery", "0x0", 0)),
    # abi_fuzz I1: a record named with a Julia keyword is declared
    # `c_baremodule`, and every signature must spell it that way.
    "baremodule" => _srec(4, _sm("nothing", "Cint", "int", "0x0", 4)),
    # A ≤16-byte bitfield blob with no float member whose alignment nothing can
    # prove (an unmeasurable 6-byte member). SysV could pass it in memory or in
    # registers, so it is refused there; Win64 returns it by size.
    "BFM12" => _srec(12, _sbit("b", "Cint", "int", 3, 0), _sm("q", "Any", "Mystery", "0x4", 6)),
)

# DW_AT_alignment as clang emits it, through the GNU binutils parser. clang
# 23.1.3, `clang -g -O1 -c --target=x86_64-w64-windows-gnu` of the C below,
# dumped with GNU `objdump --dwarf=info` (readelf prints the same lines). The
# attribute sits on the record for `aligned(16)` and on the member for
# `_Alignas` and for a member whose type is an over-aligned typedef, whose own
# DIE carries none; a vector has none at all and is aligned to its size.
#   typedef float al_vec4[4] __attribute__((aligned(16)));
#   typedef float al_f4v __attribute__((vector_size(16)));
#   typedef struct __attribute__((aligned(16))) { float x, y, z, w; } V4A;
#   typedef struct { char c; _Alignas(16) int i; } MA32;
#   typedef struct { al_vec4 v; float s; } HV4;
#   typedef struct { al_f4v v; int k; } HVec;
#   typedef struct { V4A a; float t; } NestA;
#   V4A gv; MA32 gm; HV4 gh; HVec gx; NestA gn;
const _ALIGN_DUMP = raw"""

al.o:     file format pe-x86-64

Contents of the .debug_info section:

  Compilation Unit @ offset 0:
   Length:        0x1b2 (32-bit)
   Version:       4
   Abbrev Offset: 0
   Pointer Size:  8
 <0><b>: Abbrev Number: 1 (DW_TAG_compile_unit)
    <c>   DW_AT_producer    : (indirect string, offset: 0): clang version 23.1.3 
    <10>   DW_AT_language    : 29	(C11)
    <12>   DW_AT_name        : (indirect string, offset: 0x68): al.c
    <16>   DW_AT_stmt_list   : 0
    <1a>   DW_AT_comp_dir    : (indirect string, offset: 0x6d): <dir>
 <1><1e>: Abbrev Number: 2 (DW_TAG_variable)
    <1f>   DW_AT_name        : (indirect string, offset: 0xae): gv
    <23>   DW_AT_type        : <0x33>
    <27>   DW_AT_external    : 1
    <27>   DW_AT_decl_file   : 1
    <28>   DW_AT_decl_line   : 8
    <29>   DW_AT_location    : 9 byte block: 3 0 0 0 0 0 0 0 0 	(DW_OP_addr: 0)
 <1><33>: Abbrev Number: 3 (DW_TAG_typedef)
    <34>   DW_AT_type        : <0x3e>
    <38>   DW_AT_name        : (indirect string, offset: 0xbf): V4A
    <3c>   DW_AT_decl_file   : 1
    <3d>   DW_AT_decl_line   : 3
 <1><3e>: Abbrev Number: 4 (DW_TAG_structure_type)
    <3f>   DW_AT_byte_size   : 16
    <40>   DW_AT_decl_file   : 1
    <41>   DW_AT_decl_line   : 3
    <42>   DW_AT_alignment   : 16
 <2><43>: Abbrev Number: 5 (DW_TAG_member)
    <44>   DW_AT_name        : (indirect string, offset: 0xb1): x
    <48>   DW_AT_type        : <0x74>
    <4c>   DW_AT_decl_file   : 1
    <4d>   DW_AT_decl_line   : 3
    <4e>   DW_AT_data_member_location: 0
 <2><4f>: Abbrev Number: 5 (DW_TAG_member)
    <50>   DW_AT_name        : (indirect string, offset: 0xb9): y
    <54>   DW_AT_type        : <0x74>
    <58>   DW_AT_decl_file   : 1
    <59>   DW_AT_decl_line   : 3
    <5a>   DW_AT_data_member_location: 4
 <2><5b>: Abbrev Number: 5 (DW_TAG_member)
    <5c>   DW_AT_name        : (indirect string, offset: 0xbb): z
    <60>   DW_AT_type        : <0x74>
    <64>   DW_AT_decl_file   : 1
    <65>   DW_AT_decl_line   : 3
    <66>   DW_AT_data_member_location: 8
 <2><67>: Abbrev Number: 5 (DW_TAG_member)
    <68>   DW_AT_name        : (indirect string, offset: 0xbd): w
    <6c>   DW_AT_type        : <0x74>
    <70>   DW_AT_decl_file   : 1
    <71>   DW_AT_decl_line   : 3
    <72>   DW_AT_data_member_location: 12
 <2><73>: Abbrev Number: 0
 <1><74>: Abbrev Number: 6 (DW_TAG_base_type)
    <75>   DW_AT_name        : (indirect string, offset: 0xb3): float
    <79>   DW_AT_encoding    : 4	(float)
    <7a>   DW_AT_byte_size   : 4
 <1><7b>: Abbrev Number: 2 (DW_TAG_variable)
    <7c>   DW_AT_name        : (indirect string, offset: 0xc3): gm
    <80>   DW_AT_type        : <0x90>
    <84>   DW_AT_external    : 1
    <84>   DW_AT_decl_file   : 1
    <85>   DW_AT_decl_line   : 8
    <86>   DW_AT_location    : 9 byte block: 3 10 0 0 0 0 0 0 0 	(DW_OP_addr: 10)
 <1><90>: Abbrev Number: 3 (DW_TAG_typedef)
    <91>   DW_AT_type        : <0x9b>
    <95>   DW_AT_name        : (indirect string, offset: 0xd3): MA32
    <99>   DW_AT_decl_file   : 1
    <9a>   DW_AT_decl_line   : 4
 <1><9b>: Abbrev Number: 7 (DW_TAG_structure_type)
    <9c>   DW_AT_byte_size   : 32
    <9d>   DW_AT_decl_file   : 1
    <9e>   DW_AT_decl_line   : 4
 <2><9f>: Abbrev Number: 5 (DW_TAG_member)
    <a0>   DW_AT_name        : (indirect string, offset: 0xc6): c
    <a4>   DW_AT_type        : <0xb9>
    <a8>   DW_AT_decl_file   : 1
    <a9>   DW_AT_decl_line   : 4
    <aa>   DW_AT_data_member_location: 0
 <2><ab>: Abbrev Number: 8 (DW_TAG_member)
    <ac>   DW_AT_name        : (indirect string, offset: 0xcd): i
    <b0>   DW_AT_type        : <0xc0>
    <b4>   DW_AT_decl_file   : 1
    <b5>   DW_AT_decl_line   : 4
    <b6>   DW_AT_alignment   : 16
    <b7>   DW_AT_data_member_location: 16
 <2><b8>: Abbrev Number: 0
 <1><b9>: Abbrev Number: 6 (DW_TAG_base_type)
    <ba>   DW_AT_name        : (indirect string, offset: 0xc8): char
    <be>   DW_AT_encoding    : 6	(signed char)
    <bf>   DW_AT_byte_size   : 1
 <1><c0>: Abbrev Number: 6 (DW_TAG_base_type)
    <c1>   DW_AT_name        : (indirect string, offset: 0xcf): int
    <c5>   DW_AT_encoding    : 5	(signed)
    <c6>   DW_AT_byte_size   : 4
 <1><c7>: Abbrev Number: 2 (DW_TAG_variable)
    <c8>   DW_AT_name        : (indirect string, offset: 0xd8): gh
    <cc>   DW_AT_type        : <0xdc>
    <d0>   DW_AT_external    : 1
    <d0>   DW_AT_decl_file   : 1
    <d1>   DW_AT_decl_line   : 8
    <d2>   DW_AT_location    : 9 byte block: 3 30 0 0 0 0 0 0 0 	(DW_OP_addr: 30)
 <1><dc>: Abbrev Number: 3 (DW_TAG_typedef)
    <dd>   DW_AT_type        : <0xe7>
    <e1>   DW_AT_name        : (indirect string, offset: 0xfb): HV4
    <e5>   DW_AT_decl_file   : 1
    <e6>   DW_AT_decl_line   : 5
 <1><e7>: Abbrev Number: 7 (DW_TAG_structure_type)
    <e8>   DW_AT_byte_size   : 32
    <e9>   DW_AT_decl_file   : 1
    <ea>   DW_AT_decl_line   : 5
 <2><eb>: Abbrev Number: 8 (DW_TAG_member)
    <ec>   DW_AT_name        : (indirect string, offset: 0xdb): v
    <f0>   DW_AT_type        : <0x105>
    <f4>   DW_AT_decl_file   : 1
    <f5>   DW_AT_decl_line   : 5
    <f6>   DW_AT_alignment   : 16
    <f7>   DW_AT_data_member_location: 0
 <2><f8>: Abbrev Number: 5 (DW_TAG_member)
    <f9>   DW_AT_name        : (indirect string, offset: 0xf9): s
    <fd>   DW_AT_type        : <0x74>
    <101>   DW_AT_decl_file   : 1
    <102>   DW_AT_decl_line   : 5
    <103>   DW_AT_data_member_location: 16
 <2><104>: Abbrev Number: 0
 <1><105>: Abbrev Number: 3 (DW_TAG_typedef)
    <106>   DW_AT_type        : <0x110>
    <10a>   DW_AT_name        : (indirect string, offset: 0xf1): al_vec4
    <10e>   DW_AT_decl_file   : 1
    <10f>   DW_AT_decl_line   : 1
 <1><110>: Abbrev Number: 9 (DW_TAG_array_type)
    <111>   DW_AT_type        : <0x74>
 <2><115>: Abbrev Number: 10 (DW_TAG_subrange_type)
    <116>   DW_AT_type        : <0x11c>
    <11a>   DW_AT_count       : 4
 <2><11b>: Abbrev Number: 0
 <1><11c>: Abbrev Number: 11 (DW_TAG_base_type)
    <11d>   DW_AT_name        : (indirect string, offset: 0xdd): __ARRAY_SIZE_TYPE__
    <121>   DW_AT_byte_size   : 8
    <122>   DW_AT_encoding    : 7	(unsigned)
 <1><123>: Abbrev Number: 2 (DW_TAG_variable)
    <124>   DW_AT_name        : (indirect string, offset: 0xff): gx
    <128>   DW_AT_type        : <0x138>
    <12c>   DW_AT_external    : 1
    <12c>   DW_AT_decl_file   : 1
    <12d>   DW_AT_decl_line   : 8
    <12e>   DW_AT_location    : 9 byte block: 3 50 0 0 0 0 0 0 0 	(DW_OP_addr: 50)
 <1><138>: Abbrev Number: 3 (DW_TAG_typedef)
    <139>   DW_AT_type        : <0x143>
    <13d>   DW_AT_name        : (indirect string, offset: 0x10b): HVec
    <141>   DW_AT_decl_file   : 1
    <142>   DW_AT_decl_line   : 6
 <1><143>: Abbrev Number: 7 (DW_TAG_structure_type)
    <144>   DW_AT_byte_size   : 32
    <145>   DW_AT_decl_file   : 1
    <146>   DW_AT_decl_line   : 6
 <2><147>: Abbrev Number: 5 (DW_TAG_member)
    <148>   DW_AT_name        : (indirect string, offset: 0xdb): v
    <14c>   DW_AT_type        : <0x160>
    <150>   DW_AT_decl_file   : 1
    <151>   DW_AT_decl_line   : 6
    <152>   DW_AT_data_member_location: 0
 <2><153>: Abbrev Number: 5 (DW_TAG_member)
    <154>   DW_AT_name        : (indirect string, offset: 0x109): k
    <158>   DW_AT_type        : <0xc0>
    <15c>   DW_AT_decl_file   : 1
    <15d>   DW_AT_decl_line   : 6
    <15e>   DW_AT_data_member_location: 16
 <2><15f>: Abbrev Number: 0
 <1><160>: Abbrev Number: 3 (DW_TAG_typedef)
    <161>   DW_AT_type        : <0x16b>
    <165>   DW_AT_name        : (indirect string, offset: 0x102): al_f4v
    <169>   DW_AT_decl_file   : 1
    <16a>   DW_AT_decl_line   : 2
 <1><16b>: Abbrev Number: 12 (DW_TAG_array_type)
    <16c>   DW_AT_GNU_vector  : 1
    <16c>   DW_AT_type        : <0x74>
 <2><170>: Abbrev Number: 10 (DW_TAG_subrange_type)
    <171>   DW_AT_type        : <0x11c>
    <175>   DW_AT_count       : 4
 <2><176>: Abbrev Number: 0
 <1><177>: Abbrev Number: 2 (DW_TAG_variable)
    <178>   DW_AT_name        : (indirect string, offset: 0x110): gn
    <17c>   DW_AT_type        : <0x18c>
    <180>   DW_AT_external    : 1
    <180>   DW_AT_decl_file   : 1
    <181>   DW_AT_decl_line   : 8
    <182>   DW_AT_location    : 9 byte block: 3 70 0 0 0 0 0 0 0 	(DW_OP_addr: 70)
 <1><18c>: Abbrev Number: 3 (DW_TAG_typedef)
    <18d>   DW_AT_type        : <0x197>
    <191>   DW_AT_name        : (indirect string, offset: 0x117): NestA
    <195>   DW_AT_decl_file   : 1
    <196>   DW_AT_decl_line   : 7
 <1><197>: Abbrev Number: 7 (DW_TAG_structure_type)
    <198>   DW_AT_byte_size   : 32
    <199>   DW_AT_decl_file   : 1
    <19a>   DW_AT_decl_line   : 7
 <2><19b>: Abbrev Number: 8 (DW_TAG_member)
    <19c>   DW_AT_name        : (indirect string, offset: 0x113): a
    <1a0>   DW_AT_type        : <0x33>
    <1a4>   DW_AT_decl_file   : 1
    <1a5>   DW_AT_decl_line   : 7
    <1a6>   DW_AT_alignment   : 16
    <1a7>   DW_AT_data_member_location: 0
 <2><1a8>: Abbrev Number: 5 (DW_TAG_member)
    <1a9>   DW_AT_name        : (indirect string, offset: 0x115): t
    <1ad>   DW_AT_type        : <0x74>
    <1b1>   DW_AT_decl_file   : 1
    <1b2>   DW_AT_decl_line   : 7
    <1b3>   DW_AT_data_member_location: 16
 <2><1b4>: Abbrev Number: 0
 <1><1b5>: Abbrev Number: 0

"""

@testset "Tier 3 by-value record classification (C1)" begin

    @testset "the target comes from the host, in one place" begin
        @test _SW._c_record_abi() === (Sys.iswindows() ? :win64 : :sysv)
        @test_throws ArgumentError _SW._c_return_in_memory(8, false; abi = :aapcs64)
    end

    @testset "unaligned fields: the type's alignment, never the member's size" begin
        unaligned(k) = _SW._c_has_unaligned_field(_SRET_STRUCTS[k], _SRET_STRUCTS)
        expect = Dict("T4" => false, "R5" => false, "IC12" => false, "P4" => true, "P8" => true,
                      "Big24" => false, "IS20" => false, "PK17" => true, "In3" => true, "Out6" => true, "ArrIn" => true,
                      "EnumOk" => false, "EnumPacked" => true, "BitOnly" => false,
                      "BF12" => false, "BF8" => false, "U3" => false)
        checked = 0
        for (k, want) in sort(collect(expect))
            got = unaligned(k)
            @test (k, got) == (k, want)
            checked += 1
        end
        @test checked == length(expect)
        # ArrIn's first element is aligned; only the walk over element 1 finds `y`.
        @test _SW._c_has_unaligned_field(_srec(4, _sm("c", "UInt8", "char", "0x0", 1),
                                                  _sm("v", "NTuple{1, In3}", "In3[1]", "0x1", 0)),
                                         _SRET_STRUCTS) == false
        # An alignment nothing can derive is reported, not guessed.
        @test _SW._c_has_unaligned_field(_srec(8, _sm("q", "Any", "Mystery", "0x2", 0)),
                                         _SRET_STRUCTS) === nothing
    end

    @testset "return class per target" begin
        cls(k, abi) = let info = _SRET_STRUCTS[k]
            _SW._c_return_in_memory(_SW._parse_dwarf_size(info),
                                    _SW._c_has_unaligned_field(info, _SRET_STRUCTS); abi)
        end
        #          record      SysV   Win64
        table = [("T4",       false, false),   # C1: RAX on both
                 ("IC12",     false, true),    # INTEGER,INTEGER vs not 1/2/4/8
                 ("P4",       true,  false),   # packed 4: MEMORY vs RAX
                 ("P8",       true,  false),   # packed 8: MEMORY vs RAX
                 ("Big24",    true,  true),    # > 16
                 ("IS20",     true,  true),
                 ("PK17",     true,  true),
                 ("Out6",     true,  true),    # nested unaligned; 6 bytes
                 ("ArrIn",    true,  true),
                 ("EnumOk",   false, false),
                 ("EnumPacked", true, true),   # 3 bytes on Win64
                 ("BF12",     false, true),
                 ("BF8",      false, false),
                 ("R5",       false, false)]
        for (k, sysv, win64) in table
            @test (k, :sysv, cls(k, :sysv)) == (k, :sysv, sysv)
            @test (k, :win64, cls(k, :win64)) == (k, :win64, win64)
        end
        # Win64 never reads the alignment, SysV only at 16 bytes or less.
        @test _SW._c_return_in_memory(8, nothing; abi = :win64) == false
        @test _SW._c_return_in_memory(24, nothing; abi = :sysv) == true
        @test _SW._c_return_in_memory(8, nothing; abi = :sysv) === nothing
    end

    @testset "explicit sret only where Julia's lowering would not pass the pointer" begin
        explicit(k, abi) = let info = _SRET_STRUCTS[k]
            _SW._c_explicit_sret(_SW._parse_dwarf_size(info),
                                 _SW._c_has_unaligned_field(info, _SRET_STRUCTS); abi)
        end
        # Win64 lowers by size on both sides, so at 16 bytes or less Julia passes
        # the hidden pointer itself; SysV's unaligned rule is the one it misses.
        #          record      SysV   Win64
        table = [("T4",       false, false),
                 ("IC12",     false, false),   # Win64 sret, but Julia's own
                 ("P4",       true,  false),
                 ("P8",       true,  false),
                 ("Big24",    true,  true),    # > 16: explicit, as it always was
                 ("PK17",     true,  true),
                 ("Out6",     true,  false),
                 ("EnumPacked", true, false),
                 ("BF12",     false, false)]
        for (k, sysv, win64) in table
            @test (k, :sysv, explicit(k, :sysv)) == (k, :sysv, sysv)
            @test (k, :win64, explicit(k, :win64)) == (k, :win64, win64)
        end
        @test !_SW._c_explicit_sret(8, nothing; abi = :sysv)           # unprovable: not in memory
    end

    @testset "blob parameter: SysV MEMORY vs blob INTEGER; Win64 by size on both" begin
        mis(k, abi) = let info = _SRET_STRUCTS[k]
            _SW._c_blob_param_misclassified(_SW._parse_dwarf_size(info),
                                            _SW._c_has_unaligned_field(info, _SRET_STRUCTS); abi)
        end
        table = [("P4", true), ("P8", true), ("Out6", true), ("EnumPacked", true),
                 ("T4", false), ("IC12", false), ("BF12", false), ("Big24", false)]
        for (k, sysv) in table
            @test (k, :sysv, mis(k, :sysv)) == (k, :sysv, sysv)
            @test (k, :win64, mis(k, :win64)) == (k, :win64, false)
        end
        @test _SW._c_blob_param_misclassified(8, nothing; abi = :sysv)    # unprovable: refused
        @test !_SW._c_blob_param_misclassified(24, true; abi = :sysv)     # MEMORY on both views
        @test_throws ArgumentError _SW._c_blob_param_misclassified(8, false; abi = :aapcs64)
    end

    @testset "register class: SysV eightbytes by member type; Win64 none" begin
        # (size, the stand-in may misstate the class) → SysV, Win64
        table = [(1,  true,  true,  false),
                 (8,  true,  true,  false),   # BF8: XMM vs GPR on SysV; i64 on both Win64
                 (12, true,  true,  false),   # BF12: by reference on both Win64
                 (16, true,  true,  false),   # Tag16: by reference / sret on both Win64
                 (17, true,  false, false),   # MEMORY on both views
                 (24, true,  false, false),
                 (8,  false, false, false),   # the stand-in states the class
                 (16, false, false, false)]
        checked = 0
        for (n, risk, sysv, win64) in table
            @test (n, risk, :sysv, _SW._c_register_class_misstated(n, risk; abi = :sysv)) ==
                  (n, risk, :sysv, sysv)
            @test (n, risk, :win64, _SW._c_register_class_misstated(n, risk; abi = :win64)) ==
                  (n, risk, :win64, win64)
            checked += 1
        end
        @test checked == length(table)
        @test_throws ArgumentError _SW._c_register_class_misstated(8, true; abi = :aapcs64)
    end

    @testset "record alignment: its own, its members', their types'" begin
        expect = Dict("V4A" => 16,     # aligned(16) on the record
                      "D4A" => 16,
                      "A32" => 32,
                      "AB16" => 16,    # _Alignas(16) on a member
                      "HVec" => 16,    # a vector member
                      "LD16" => 16,    # long double, from its type
                      "NestA" => 16,   # a nested aligned record, from the record itself
                      "Big24" => 8, "IC12" => 4, "T4" => 2, "BF12" => 4, "Tag16" => 8, "U3" => 4,
                      "P4" => 2,       # packed: C says 1; more is still a correct buffer
                      "Mys48" => 16,   # unmeasurable: the largest power of two dividing the size
                      "Mys24" => 8)
        checked = 0
        for (k, want) in sort(collect(expect))
            @test (k, _SW._c_record_align(_SRET_STRUCTS[k], _SRET_STRUCTS)) == (k, want)
            checked += 1
        end
        @test checked == length(expect)
        @test _SW._c_explicit_align("32") == 32 && _SW._c_explicit_align(nothing) == 1
    end

    @testset "the parser records DW_AT_alignment and vectors" begin
        _, sd, _, _, _ = RepliBuild.Compiler.parse_dwarf_dump(_ALIGN_DUMP)
        @test sort(collect(keys(sd))) == ["HV4", "HVec", "MA32", "NestA", "V4A"]   # parsed at all
        member(rec, name) = only(filter(m -> m["name"] == name, sd[rec]["members"]))
        @test get(sd["V4A"], "alignment", nothing) == 16           # aligned(16) on the record
        @test get(member("MA32", "i"), "alignment", nothing) == 16  # _Alignas(16)
        @test get(member("HV4", "v"), "alignment", nothing) == 16   # over-aligned typedef
        @test get(member("HVec", "v"), "alignment", nothing) == 16  # vector: its size
        @test get(member("NestA", "a"), "alignment", nothing) == 16 # an aligned record
        # Natural alignment has no attribute and gets no key.
        @test !haskey(member("V4A", "x"), "alignment") && !haskey(member("MA32", "c"), "alignment")
        @test !haskey(sd["MA32"], "alignment")
        checked = 0
        for k in sort(collect(keys(sd)))
            @test (k, _SW._c_record_align(sd[k], sd)) == (k, 16)
            checked += 1
        end
        @test checked == 5
    end

    @testset "the wrapper aligns the buffer where the stand-in is less aligned" begin
        # (record alignment, stand-in alignment) → buffer alignment, 0 = Julia's own
        table = [(16, 1, 16),         # a blob of a 16-aligned record
                 (16, 4, 16),         # V4A: four Float32
                 (32, 16, 32),
                 (16, 16, 0),         # the stand-in is as aligned
                 (8, 1, 0),           # nothing on x86-64 faults below 16
                 (4, 1, 0),
                 (16, nothing, 0)]    # a mutable placeholder: not decided here
        checked = 0
        for (rec, jl, want) in table
            @test (rec, jl, _SW._c_sret_buffer_align(rec, jl)) == (rec, jl, want)
            checked += 1
        end
        @test checked == length(table)
    end

    # ── What the generator emits for each shape, per target ──────────────────
    dir = mktempdir()
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "sretsynth"
    root = "$(escape_string(dir))"
    [link]
    enable_lto = false
    [wrap]
    language = "c"
    [types]
    strictness = "warn"
    allow_unknown_structs = true
    [cache]
    enabled = false
    """)
    cfg = RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml"))
    rec_param(n, t) = Dict{String,Any}("name" => n, "julia_type" => t, "c_type" => t)
    fn(name, ret, params...) = Dict{String,Any}("name" => name, "mangled" => name, "demangled" => name,
        "parameters" => Any[params...],
        "return_type" => Dict{String,Any}("julia_type" => ret, "c_type" => ret))
    meta = Dict{String,Any}(
        "struct_definitions" => _SRET_STRUCTS,
        "globals" => Dict{String,Any}(),
        "functions" => Any[
            fn("f4", "T4", rec_param("p1", "R5")),
            fn("make_ic12", "IC12", rec_param("n", "Cint")),
            fn("make_p4", "P4"),
            fn("make_p8", "P8"),
            fn("make_big24", "Big24"),
            fn("make_is20", "IS20"),
            fn("make_pk17", "PK17"),
            fn("make_out6", "Out6"),
            fn("make_bf12", "BF12"),
            fn("make_bf8", "BF8"),
            fn("take_p8", "void", rec_param("v", "P8")),
            fn("take_t4", "void", rec_param("v", "T4")),
            fn("take_bf8", "void", rec_param("v", "BF8")),
            fn("take_bf12", "Cint", rec_param("v", "BF12")),
            fn("make_tag16", "Tag16", rec_param("n", "Clonglong")),
            fn("take_tag16", "void", rec_param("t", "Tag16")),
            fn("take_tag16_v", "void", rec_param("v", "Tag16_v")),
            fn("take_fv8_v", "void", rec_param("v", "FV8_v")),
            fn("make_v4a", "V4A", rec_param("x", "Cfloat")),
            fn("make_d4a", "D4A"),
            fn("make_a32", "A32", rec_param("d", "Cdouble")),
            fn("make_ab16", "AB16"),
            fn("make_hvec", "HVec"),
            fn("make_nesta", "NestA"),
            fn("make_mys48", "Mys48"),
            fn("make_mys24", "Mys24"),
            # A `long double` parameter: the x87 stub, never the buffer (C2).
            fn("ld_d4a", "D4A", Dict{String,Any}("name" => "x", "julia_type" => "Any", "c_type" => "long double")),
            # ...nor BRANCH 0's explicit sret, whose ccall would declare the
            # `long double` as `Any` and get the whole wrapper refused (C2).
            fn("ld_pk17", "PK17", Dict{String,Any}("name" => "x", "julia_type" => "Any", "c_type" => "long double")),
            fn("ld_p8", "P8", Dict{String,Any}("name" => "x", "julia_type" => "Any", "c_type" => "long double")),
            # A keyword-named record by value, by pointer, and as an `Any` parameter (I1).
            fn("kw_make", "baremodule"),
            fn("kw_take", "Cint", rec_param("b", "baremodule")),
            fn("kw_ptr", "Cint", rec_param("p", "Ptr{baremodule}")),
            fn("kw_any", "Cint", Dict{String,Any}("name" => "a", "julia_type" => "Any", "c_type" => "baremodule")),
            fn("make_bfm12", "BFM12"),
        ])
    libref = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
    gen(abi) = _SW.generate_introspective_module_c(cfg, libref, meta, "SretSynth",
                                                   _SW.create_type_registry(cfg), false; abi)
    body(code, name) = let i = findfirst("function $name(", code)
        i === nothing ? "" : (r = SubString(code, first(i)); j = findfirst(r"\nend\b", r);
                              j === nothing ? String(r) : String(r[1:last(j)]))
    end
    form(code, name) = let b = body(code, name)
        isempty(b) ? :missing :
        occursin("ABI Safety Trap", b) ? :trap :
        occursin("ret_ptr", b) ? :aligned :
        occursin("ret_buf", b) ? :sret : occursin("ccall((:$name, LIBRARY_PATH)", b) ? :ccall : :other
    end

    @test_throws ArgumentError gen(:aapcs64)
    sysv, win64 = gen(:sysv), gen(:win64)
    # :ccall on a record C returns through a hidden pointer means Julia's own
    # lowering passes it (Win64 by size; anything above 16 bytes).
    #          function      SysV     Win64
    table = [("f4",         :ccall,  :ccall),   # C1: was :sret on both
             ("make_ic12",  :ccall,  :ccall),   # was :sret on both; registers on SysV
             ("make_p4",    :sret,   :ccall),   # packed 4-byte return in RAX on Win64
             ("make_p8",    :sret,   :ccall),
             ("make_big24", :ccall,  :ccall),   # passes the size test; Julia's own sret
             ("make_is20",  :ccall,  :ccall),   # natural, > 16: Julia's own sret (the size test called it packed)
             ("make_pk17",  :sret,   :sret),    # packed, > 16
             ("make_out6",  :sret,   :ccall),   # SysV: nested unaligned field, size test blind
             ("make_bf12",  :trap,   :ccall),   # float blob: SysV class unknowable; Win64 sret by size
             ("take_p8",    :trap,   :ccall),   # SysV MEMORY vs blob INTEGER; Win64 i64 both
             ("take_t4",    :ccall,  :ccall),
             # The register-class traps. SysV: the stand-in claims INTEGER for an
             # eightbyte C passes in XMM. Win64: by size on both sides.
             ("make_bf8",   :trap,   :ccall),   # float blob, RAX on Win64
             ("take_bf8",   :trap,   :ccall),   # float blob, i64 on Win64
             ("take_bf12",  :trap,   :ccall),   # float blob, by reference on Win64
             ("make_tag16", :trap,   :ccall),   # mixed region (mpack_tag_t), sret on Win64
             ("take_tag16", :trap,   :ccall),   # by reference on Win64
             ("take_tag16_v", :trap, :ccall),   # the region itself, i64 on Win64
             ("take_fv8_v", :ccall,  :ccall),   # all-float region states SSE: no trap
             # Returned through a hidden pointer and more aligned than the Julia
             # type standing in: the wrapper's own buffer at the record's
             # alignment (:aligned). Julia's would be aligned to the stand-in.
             ("make_v4a",   :ccall,  :aligned), # SysV SSE,SSE in registers; Win64 sret, V4A is align 4 in Julia
             ("make_d4a",   :aligned, :aligned),# natural fields, aligned(16): c_safe, still the buffer
             ("make_a32",   :aligned, :aligned),# 32-aligned
             ("make_ab16",  :trap,   :aligned), # float blob: SysV register class; Win64 sret
             ("make_hvec",  :aligned, :aligned),# vector member
             ("make_nesta", :aligned, :aligned),# nested aligned record
             ("make_mys48", :aligned, :aligned),# unmeasurable member: aligned to 16 from the size
             ("make_mys24", :sret,   :sret),    # bound 8: Julia's Ref buffer, as before
             ("ld_d4a",     :trap,   :trap),    # long double: its x87 stub, not BRANCH 0 (C2)
             ("ld_pk17",    :trap,   :trap),    # was BRANCH 0's sret with an `Any` argument (C2)
             ("ld_p8",      :trap,   :trap),    # same on SysV, where P8 is MEMORY (C2)
             ("kw_make",    :ccall,  :ccall),   # 4-byte natural record: RAX on both (I1)
             ("kw_take",    :ccall,  :ccall),
             ("make_bfm12", :trap,   :ccall)]   # alignment unprovable: SysV refuses, Win64 by size
    checked = 0
    for (name, want_sysv, want_win64) in table
        @test (name, :sysv, form(sysv, name)) == (name, :sysv, want_sysv)
        @test (name, :win64, form(win64, name)) == (name, :win64, want_win64)
        checked += 1
    end
    @test checked == length(table)
    # SysV names what it refused; Win64 refuses nothing for a class.
    @test occursin("returns `BF8` by value", body(sysv, "make_bf8"))
    @test occursin("takes `BF12` by value", body(sysv, "take_bf12"))
    @test occursin("returns `Tag16` by value", body(sysv, "make_tag16"))
    @test occursin("takes `Tag16_v` by value", body(sysv, "take_tag16_v"))
    @test !occursin("SysV register window", win64)
    # Win64 hands the stand-in itself to ccall, which lowers it by size.
    @test occursin("ccall((:make_bf8, LIBRARY_PATH), BF8, (), )", body(win64, "make_bf8"))
    @test occursin("ccall((:take_bf12, LIBRARY_PATH), Cint, (BF12,), v)", body(win64, "take_bf12"))
    @test occursin("ccall((:make_tag16, LIBRARY_PATH), Tag16, (Clonglong,), n_c)", body(win64, "make_tag16"))
    @test occursin(r"ccall\(\(:take_tag16, LIBRARY_PATH\), \w+, \(Tag16,\), t\)", body(win64, "take_tag16"))
    # The plain ccall returns the record itself, with no hidden first argument.
    @test occursin("ccall((:f4, LIBRARY_PATH), T4, (R5,), p1)", body(sysv, "f4"))
    @test occursin("ccall((:f4, LIBRARY_PATH), T4, (R5,), p1)", body(win64, "f4"))
    @test occursin("ccall((:make_p4, LIBRARY_PATH), Cvoid, (Ptr{P4}, ), ret_buf, )", body(sysv, "make_p4"))
    @test occursin("ccall((:make_ic12, LIBRARY_PATH), IC12, (Cint,), n_c)", body(win64, "make_ic12"))
    @test occursin("ccall((:make_is20, LIBRARY_PATH), IS20, (), )", body(win64, "make_is20"))
    @test occursin("ccall((:make_pk17, LIBRARY_PATH), Cvoid, (Ptr{PK17}, ), ret_buf, )", body(win64, "make_pk17"))
    # The aligned buffer: over-allocated by align - 1, the record starts at the
    # first aligned address in it, and the callee gets that address.
    for (code, abi) in ((sysv, :sysv), (win64, :win64))
        b = body(code, "make_a32")
        @test (abi, occursin("ret_buf = Ref{NTuple{63, UInt8}}()", b)) == (abi, true)
        @test (abi, occursin("ret_ptr = Ptr{A32}((UInt(Base.unsafe_convert(Ptr{Cvoid}, ret_buf)) + 31) & ~UInt(31))", b)) == (abi, true)
        @test (abi, occursin("ccall((:make_a32, LIBRARY_PATH), Cvoid, (Ptr{A32}, Cdouble,), ret_ptr, d)", b)) == (abi, true)
        @test (abi, occursin("return unsafe_load(ret_ptr)", b)) == (abi, true)
    end
    @test occursin("Ref{NTuple{31, UInt8}}()", body(win64, "make_v4a"))
    @test occursin("& ~UInt(15)", body(win64, "make_v4a"))
    # The long double function keeps its x87 stub and passes no `Any` to ccall.
    @test occursin("long double", body(win64, "ld_d4a")) && !occursin("ccall((", body(win64, "ld_d4a"))
    for (code, abi) in ((sysv, :sysv), (win64, :win64)), f in ("ld_pk17", "ld_p8")
        @test (abi, f, occursin("long double", body(code, f)) && !occursin("ccall((", body(code, f))) ==
              (abi, f, true)
    end
    # I1: the keyword-named record is spelled as declared in every type position,
    # so the module parses at all.
    for (code, abi) in ((sysv, :sysv), (win64, :win64))
        ex = Meta.parseall(code)
        @test (abi, any(a -> a isa Expr && a.head in (:incomplete, :error), ex.args)) == (abi, false)
        @test (abi, occursin("ccall((:kw_make, LIBRARY_PATH), c_baremodule, (), )", body(code, "kw_make"))) == (abi, true)
        @test (abi, occursin("(c_baremodule,)", body(code, "kw_take"))) == (abi, true)
        @test (abi, occursin("(Ptr{c_baremodule},)", body(code, "kw_ptr"))) == (abi, true)
        @test (abi, occursin("(c_baremodule,)", body(code, "kw_any"))) == (abi, true)
    end
    @test occursin("returns `BFM12` by value", body(sysv, "make_bfm12"))
    @test occursin("ccall((:make_bfm12, LIBRARY_PATH), BFM12, (), )", body(win64, "make_bfm12"))
end
