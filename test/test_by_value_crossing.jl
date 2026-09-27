#!/usr/bin/env julia
# By-value records at the Tier-2 thunk boundary.
#
# A thunk types each by-value parameter and return from the metadata. A record
# with no entry in `struct_definitions` — an empty class (no members, so no
# entry), a type only DECLARED in this binary (constructor homing), a toolchain
# type the provenance gate leaves out (`std::string_view`), a derived class
# with no members of its own — fell to `!llvm.ptr`: its bytes were loaded as a
# pointer and passed in one register. A template or qualified spelling
# (`Box<double>`, `ns::Enum`) did too, because only bare identifiers were looked
# up. Wrong answers, NaN, the occasional segfault; never an error.
#
# Now `FunctionGen.by_value_crossing` sorts each crossing, from the struct table
# plus `record_abi` (DWARF's `DW_AT_calling_convention` and emptiness for every
# by-value record), and both the thunk and the Julia wrapper act on its verdict.
# No toolchain: the DWARF is a verbatim clang 22 readelf dump, the thunks are text.

using Test
using RepliBuild

const BVC = RepliBuild.Compiler
const BVF = RepliBuild.JLCSIRGenerator.FunctionGen
const BVW = RepliBuild.Wrapper

# clang 22.1.8, `clang++ -std=c++17 -g -O1 -fPIC -shared` of:
#   namespace ee {
#   struct Tag {};                     struct Guard { ~Guard(); };
#   struct Pair { int a; int b; };     struct P3 : Pair {};
#   struct Range { double lo, hi; inline Range(double w = 0) : lo(-w), hi(w) {} };
#   struct Unused {};
#   struct Owner { int v; ~Owner(); };        struct Handle { int id; Handle(const Handle &o); };
#   Guard::~Guard() {}  Owner::~Owner() {}  Handle::Handle(const Handle &o) : id(o.id) {}
#   int with_tag(Tag, int x);  Tag make_tag(int);  int with_guard(Guard, int x);
#   int p3_sum(P3 p);  double width(Range r, double s);  int by_ptr(Unused *u);
#   Owner make_owner(int v);  int handle_id(Handle h);
#   }
# `Range` has only inline constructors, so constructor homing leaves it a
# declaration here. `Guard`, `Owner` (user destructor) and `Handle` (user copy
# constructor) are non-trivial for calls: DW_CC_pass_by_reference.
const RECORD_DUMP = raw"""
Contents of the .debug_info section:

  Compilation Unit @ offset 0:
   Length:        0x221 (32-bit)
   Version:       5
   Unit Type:     DW_UT_compile (1)
   Abbrev Offset: 0
   Pointer Size:  8
 <0><c>: Abbrev Number: 1 (DW_TAG_compile_unit)
    <d>   DW_AT_producer    : (indexed string: 0): clang version 22.1.8
    <e>   DW_AT_language    : 33	(C++14)
    <10>   DW_AT_name        : (indexed string: 0x1): rec.cpp
    <11>   DW_AT_str_offsets_base: 0x8
    <15>   DW_AT_stmt_list   : 0
    <19>   DW_AT_comp_dir    : (indexed string: 0x2): <dir>
    <1a>   DW_AT_low_pc      : (index: 0): 0x10f0
    <1b>   DW_AT_high_pc     : 0xa3
    <1f>   DW_AT_addr_base   : 0x8
    <23>   DW_AT_loclists_base: 0xc (location list)
 <1><27>: Abbrev Number: 2 (DW_TAG_namespace)
    <28>   DW_AT_name        : (indexed string: 0x3): ee
 <2><29>: Abbrev Number: 3 (DW_TAG_structure_type)
    <2a>   DW_AT_calling_convention: 5	(pass by value)
    <2b>   DW_AT_name        : (indexed string: 0x4): Tag
    <2c>   DW_AT_byte_size   : 1
    <2d>   DW_AT_decl_file   : 0
    <2e>   DW_AT_decl_line   : 2
 <2><2f>: Abbrev Number: 4 (DW_TAG_structure_type)
    <30>   DW_AT_calling_convention: 4	(pass by ref)
    <31>   DW_AT_name        : (indexed string: 0x9): Owner
    <32>   DW_AT_byte_size   : 4
    <33>   DW_AT_decl_file   : 0
    <34>   DW_AT_decl_line   : 8
 <3><35>: Abbrev Number: 5 (DW_TAG_member)
    <36>   DW_AT_name        : (indexed string: 0x5): v
    <37>   DW_AT_type        : <0x198>
    <3b>   DW_AT_decl_file   : 0
    <3c>   DW_AT_decl_line   : 8
    <3d>   DW_AT_data_member_location: 0
 <3><3e>: Abbrev Number: 6 (DW_TAG_subprogram)
    <3f>   DW_AT_linkage_name: (indexed string: 0x7): _ZN2ee5OwnerD4Ev
    <40>   DW_AT_name        : (indexed string: 0x8): ~Owner
    <41>   DW_AT_decl_file   : 0
    <42>   DW_AT_decl_line   : 8
    <43>   DW_AT_declaration : 1
    <43>   DW_AT_external    : 1
 <4><43>: Abbrev Number: 7 (DW_TAG_formal_parameter)
    <44>   DW_AT_type        : <0x19c>
    <48>   DW_AT_artificial  : 1
 <4><48>: Abbrev Number: 0
 <3><49>: Abbrev Number: 0
 <2><4a>: Abbrev Number: 4 (DW_TAG_structure_type)
    <4b>   DW_AT_calling_convention: 4	(pass by ref)
    <4c>   DW_AT_name        : (indexed string: 0xc): Handle
    <4d>   DW_AT_byte_size   : 4
    <4e>   DW_AT_decl_file   : 0
    <4f>   DW_AT_decl_line   : 9
 <3><50>: Abbrev Number: 5 (DW_TAG_member)
    <51>   DW_AT_name        : (indexed string: 0xa): id
    <52>   DW_AT_type        : <0x198>
    <56>   DW_AT_decl_file   : 0
    <57>   DW_AT_decl_line   : 9
    <58>   DW_AT_data_member_location: 0
 <3><59>: Abbrev Number: 6 (DW_TAG_subprogram)
    <5a>   DW_AT_linkage_name: (indexed string: 0xb): _ZN2ee6HandleC4ERKS0_
    <5b>   DW_AT_name        : (indexed string: 0xc): Handle
    <5c>   DW_AT_decl_file   : 0
    <5d>   DW_AT_decl_line   : 9
    <5e>   DW_AT_declaration : 1
    <5e>   DW_AT_external    : 1
 <4><5e>: Abbrev Number: 7 (DW_TAG_formal_parameter)
    <5f>   DW_AT_type        : <0x1a1>
    <63>   DW_AT_artificial  : 1
 <4><63>: Abbrev Number: 8 (DW_TAG_formal_parameter)
    <64>   DW_AT_type        : <0x1a6>
 <4><68>: Abbrev Number: 0
 <3><69>: Abbrev Number: 0
 <2><6a>: Abbrev Number: 4 (DW_TAG_structure_type)
    <6b>   DW_AT_calling_convention: 4	(pass by ref)
    <6c>   DW_AT_name        : (indexed string: 0xf): Guard
    <6d>   DW_AT_byte_size   : 1
    <6e>   DW_AT_decl_file   : 0
    <6f>   DW_AT_decl_line   : 3
 <3><70>: Abbrev Number: 6 (DW_TAG_subprogram)
    <71>   DW_AT_linkage_name: (indexed string: 0xd): _ZN2ee5GuardD4Ev
    <72>   DW_AT_name        : (indexed string: 0xe): ~Guard
    <73>   DW_AT_decl_file   : 0
    <74>   DW_AT_decl_line   : 3
    <75>   DW_AT_declaration : 1
    <75>   DW_AT_external    : 1
 <4><75>: Abbrev Number: 7 (DW_TAG_formal_parameter)
    <76>   DW_AT_type        : <0x1b0>
    <7a>   DW_AT_artificial  : 1
 <4><7a>: Abbrev Number: 0
 <3><7b>: Abbrev Number: 0
 <2><7c>: Abbrev Number: 9 (DW_TAG_subprogram)
    <7d>   DW_AT_low_pc      : (index: 0x3): 0x1120
    <7e>   DW_AT_high_pc     : 0x4
    <82>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <84>   DW_AT_call_all_calls: 1
    <84>   DW_AT_linkage_name: (indexed string: 0x13): _ZN2ee8with_tagENS_3TagEi
    <85>   DW_AT_name        : (indexed string: 0x14): with_tag
    <86>   DW_AT_decl_file   : 0
    <87>   DW_AT_decl_line   : 13
    <88>   DW_AT_type        : <0x198>
    <8c>   DW_AT_external    : 1
 <3><8c>: Abbrev Number: 10 (DW_TAG_formal_parameter)
    <8d>   DW_AT_decl_file   : 0
    <8e>   DW_AT_decl_line   : 13
    <8f>   DW_AT_type        : <0x29>
 <3><93>: Abbrev Number: 11 (DW_TAG_formal_parameter)
    <94>   DW_AT_location    : 1 byte block: 55 	(DW_OP_reg5 (rdi))
    <96>   DW_AT_name        : (indexed string: 0x26): x
    <97>   DW_AT_decl_file   : 0
    <98>   DW_AT_decl_line   : 13
    <99>   DW_AT_type        : <0x198>
 <3><9d>: Abbrev Number: 0
 <2><9e>: Abbrev Number: 9 (DW_TAG_subprogram)
    <9f>   DW_AT_low_pc      : (index: 0x4): 0x1130
    <a0>   DW_AT_high_pc     : 0x1
    <a4>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <a6>   DW_AT_call_all_calls: 1
    <a6>   DW_AT_linkage_name: (indexed string: 0x15): _ZN2ee8make_tagEi
    <a7>   DW_AT_name        : (indexed string: 0x16): make_tag
    <a8>   DW_AT_decl_file   : 0
    <a9>   DW_AT_decl_line   : 14
    <aa>   DW_AT_type        : <0x29>
    <ae>   DW_AT_external    : 1
 <3><ae>: Abbrev Number: 12 (DW_TAG_formal_parameter)
    <af>   DW_AT_location    : 1 byte block: 55 	(DW_OP_reg5 (rdi))
    <b1>   DW_AT_decl_file   : 0
    <b2>   DW_AT_decl_line   : 14
    <b3>   DW_AT_type        : <0x198>
 <3><b7>: Abbrev Number: 0
 <2><b8>: Abbrev Number: 9 (DW_TAG_subprogram)
    <b9>   DW_AT_low_pc      : (index: 0x5): 0x1140
    <ba>   DW_AT_high_pc     : 0x3
    <be>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <c0>   DW_AT_call_all_calls: 1
    <c0>   DW_AT_linkage_name: (indexed string: 0x17): _ZN2ee10with_guardENS_5GuardEi
    <c1>   DW_AT_name        : (indexed string: 0x18): with_guard
    <c2>   DW_AT_decl_file   : 0
    <c3>   DW_AT_decl_line   : 15
    <c4>   DW_AT_type        : <0x198>
    <c8>   DW_AT_external    : 1
 <3><c8>: Abbrev Number: 12 (DW_TAG_formal_parameter)
    <c9>   DW_AT_location    : 2 byte block: 75 0 	(DW_OP_breg5 (rdi): 0)
    <cc>   DW_AT_decl_file   : 0
    <cd>   DW_AT_decl_line   : 15
    <ce>   DW_AT_type        : <0x6a>
 <3><d2>: Abbrev Number: 11 (DW_TAG_formal_parameter)
    <d3>   DW_AT_location    : 1 byte block: 54 	(DW_OP_reg4 (rsi))
    <d5>   DW_AT_name        : (indexed string: 0x26): x
    <d6>   DW_AT_decl_file   : 0
    <d7>   DW_AT_decl_line   : 15
    <d8>   DW_AT_type        : <0x198>
 <3><dc>: Abbrev Number: 0
 <2><dd>: Abbrev Number: 9 (DW_TAG_subprogram)
    <de>   DW_AT_low_pc      : (index: 0x6): 0x1150
    <df>   DW_AT_high_pc     : 0xa
    <e3>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <e5>   DW_AT_call_all_calls: 1
    <e5>   DW_AT_linkage_name: (indexed string: 0x19): _ZN2ee6p3_sumENS_2P3E
    <e6>   DW_AT_name        : (indexed string: 0x1a): p3_sum
    <e7>   DW_AT_decl_file   : 0
    <e8>   DW_AT_decl_line   : 16
    <e9>   DW_AT_type        : <0x198>
    <ed>   DW_AT_external    : 1
 <3><ed>: Abbrev Number: 13 (DW_TAG_formal_parameter)
    <ee>   DW_AT_location    : (index: 0): 0x14 (location list)
    <ef>   DW_AT_name        : (indexed string: 0x27): p
    <f0>   DW_AT_decl_file   : 0
    <f1>   DW_AT_decl_line   : 16
    <f2>   DW_AT_type        : <0x16d>
 <3><f6>: Abbrev Number: 0
 <2><f7>: Abbrev Number: 9 (DW_TAG_subprogram)
    <f8>   DW_AT_low_pc      : (index: 0x7): 0x1160
    <f9>   DW_AT_high_pc     : 0xd
    <fd>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <ff>   DW_AT_call_all_calls: 1
    <ff>   DW_AT_linkage_name: (indexed string: 0x1b): _ZN2ee5widthENS_5RangeEd
    <100>   DW_AT_name        : (indexed string: 0x1c): width
    <101>   DW_AT_decl_file   : 0
    <102>   DW_AT_decl_line   : 17
    <103>   DW_AT_type        : <0x20c>
    <107>   DW_AT_external    : 1
 <3><107>: Abbrev Number: 13 (DW_TAG_formal_parameter)
    <108>   DW_AT_location    : (index: 0x1): 0x26 (location list)
    <109>   DW_AT_name        : (indexed string: 0x2c): r
    <10a>   DW_AT_decl_file   : 0
    <10b>   DW_AT_decl_line   : 17
    <10c>   DW_AT_type        : <0x193>
 <3><110>: Abbrev Number: 11 (DW_TAG_formal_parameter)
    <111>   DW_AT_location    : 1 byte block: 63 	(DW_OP_reg19 (xmm2))
    <113>   DW_AT_name        : (indexed string: 0x2e): s
    <114>   DW_AT_decl_file   : 0
    <115>   DW_AT_decl_line   : 17
    <116>   DW_AT_type        : <0x20c>
 <3><11a>: Abbrev Number: 0
 <2><11b>: Abbrev Number: 9 (DW_TAG_subprogram)
    <11c>   DW_AT_low_pc      : (index: 0x8): 0x1170
    <11d>   DW_AT_high_pc     : 0x9
    <121>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <123>   DW_AT_call_all_calls: 1
    <123>   DW_AT_linkage_name: (indexed string: 0x1e): _ZN2ee6by_ptrEPNS_6UnusedE
    <124>   DW_AT_name        : (indexed string: 0x1f): by_ptr
    <125>   DW_AT_decl_file   : 0
    <126>   DW_AT_decl_line   : 18
    <127>   DW_AT_type        : <0x198>
    <12b>   DW_AT_external    : 1
 <3><12b>: Abbrev Number: 11 (DW_TAG_formal_parameter)
    <12c>   DW_AT_location    : 1 byte block: 55 	(DW_OP_reg5 (rdi))
    <12e>   DW_AT_name        : (indexed string: 0x2f): u
    <12f>   DW_AT_decl_file   : 0
    <130>   DW_AT_decl_line   : 18
    <131>   DW_AT_type        : <0x21f>
 <3><135>: Abbrev Number: 0
 <2><136>: Abbrev Number: 9 (DW_TAG_subprogram)
    <137>   DW_AT_low_pc      : (index: 0x9): 0x1180
    <138>   DW_AT_high_pc     : 0x6
    <13c>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <13e>   DW_AT_call_all_calls: 1
    <13e>   DW_AT_linkage_name: (indexed string: 0x20): _ZN2ee10make_ownerEi
    <13f>   DW_AT_name        : (indexed string: 0x21): make_owner
    <140>   DW_AT_decl_file   : 0
    <141>   DW_AT_decl_line   : 19
    <142>   DW_AT_type        : <0x2f>
    <146>   DW_AT_external    : 1
 <3><146>: Abbrev Number: 11 (DW_TAG_formal_parameter)
    <147>   DW_AT_location    : 1 byte block: 54 	(DW_OP_reg4 (rsi))
    <149>   DW_AT_name        : (indexed string: 0x5): v
    <14a>   DW_AT_decl_file   : 0
    <14b>   DW_AT_decl_line   : 19
    <14c>   DW_AT_type        : <0x198>
 <3><150>: Abbrev Number: 0
 <2><151>: Abbrev Number: 9 (DW_TAG_subprogram)
    <152>   DW_AT_low_pc      : (index: 0xa): 0x1190
    <153>   DW_AT_high_pc     : 0x3
    <157>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <159>   DW_AT_call_all_calls: 1
    <159>   DW_AT_linkage_name: (indexed string: 0x22): _ZN2ee9handle_idENS_6HandleE
    <15a>   DW_AT_name        : (indexed string: 0x23): handle_id
    <15b>   DW_AT_decl_file   : 0
    <15c>   DW_AT_decl_line   : 20
    <15d>   DW_AT_type        : <0x198>
    <161>   DW_AT_external    : 1
 <3><161>: Abbrev Number: 11 (DW_TAG_formal_parameter)
    <162>   DW_AT_location    : 2 byte block: 75 0 	(DW_OP_breg5 (rdi): 0)
    <165>   DW_AT_name        : (indexed string: 0x31): h
    <166>   DW_AT_decl_file   : 0
    <167>   DW_AT_decl_line   : 20
    <168>   DW_AT_type        : <0x4a>
 <3><16c>: Abbrev Number: 0
 <2><16d>: Abbrev Number: 4 (DW_TAG_structure_type)
    <16e>   DW_AT_calling_convention: 5	(pass by value)
    <16f>   DW_AT_name        : (indexed string: 0x2b): P3
    <170>   DW_AT_byte_size   : 8
    <171>   DW_AT_decl_file   : 0
    <172>   DW_AT_decl_line   : 5
 <3><173>: Abbrev Number: 14 (DW_TAG_inheritance)
    <174>   DW_AT_type        : <0x17a>
    <178>   DW_AT_data_member_location: 0
 <3><179>: Abbrev Number: 0
 <2><17a>: Abbrev Number: 4 (DW_TAG_structure_type)
    <17b>   DW_AT_calling_convention: 5	(pass by value)
    <17c>   DW_AT_name        : (indexed string: 0x2a): Pair
    <17d>   DW_AT_byte_size   : 8
    <17e>   DW_AT_decl_file   : 0
    <17f>   DW_AT_decl_line   : 4
 <3><180>: Abbrev Number: 5 (DW_TAG_member)
    <181>   DW_AT_name        : (indexed string: 0x28): a
    <182>   DW_AT_type        : <0x198>
    <186>   DW_AT_decl_file   : 0
    <187>   DW_AT_decl_line   : 4
    <188>   DW_AT_data_member_location: 0
 <3><189>: Abbrev Number: 5 (DW_TAG_member)
    <18a>   DW_AT_name        : (indexed string: 0x29): b
    <18b>   DW_AT_type        : <0x198>
    <18f>   DW_AT_decl_file   : 0
    <190>   DW_AT_decl_line   : 4
    <191>   DW_AT_data_member_location: 4
 <3><192>: Abbrev Number: 0
 <2><193>: Abbrev Number: 15 (DW_TAG_structure_type)
    <194>   DW_AT_name        : (indexed string: 0x2d): Range
    <195>   DW_AT_declaration : 1
 <2><195>: Abbrev Number: 15 (DW_TAG_structure_type)
    <196>   DW_AT_name        : (indexed string: 0x30): Unused
    <197>   DW_AT_declaration : 1
 <2><197>: Abbrev Number: 0
 <1><198>: Abbrev Number: 16 (DW_TAG_base_type)
    <199>   DW_AT_name        : (indexed string: 0x6): int
    <19a>   DW_AT_encoding    : 5	(signed)
    <19b>   DW_AT_byte_size   : 4
 <1><19c>: Abbrev Number: 17 (DW_TAG_pointer_type)
    <19d>   DW_AT_type        : <0x2f>
 <1><1a1>: Abbrev Number: 17 (DW_TAG_pointer_type)
    <1a2>   DW_AT_type        : <0x4a>
 <1><1a6>: Abbrev Number: 18 (DW_TAG_reference_type)
    <1a7>   DW_AT_type        : <0x1ab>
 <1><1ab>: Abbrev Number: 19 (DW_TAG_const_type)
    <1ac>   DW_AT_type        : <0x4a>
 <1><1b0>: Abbrev Number: 17 (DW_TAG_pointer_type)
    <1b1>   DW_AT_type        : <0x6a>
 <1><1b5>: Abbrev Number: 20 (DW_TAG_subprogram)
    <1b6>   DW_AT_low_pc      : (index: 0): 0x10f0
    <1b7>   DW_AT_high_pc     : 0x1
    <1bb>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <1bd>   DW_AT_object_pointer: <0x1c7>
    <1c1>   DW_AT_call_all_calls: 1
    <1c1>   DW_AT_decl_line   : 10
    <1c2>   DW_AT_linkage_name: (indexed string: 0x10): _ZN2ee5GuardD2Ev
    <1c3>   DW_AT_specification: <0x70>
 <2><1c7>: Abbrev Number: 21 (DW_TAG_formal_parameter)
    <1c8>   DW_AT_name        : (indexed string: 0x24): this
    <1c9>   DW_AT_type        : <0x210>
    <1cd>   DW_AT_artificial  : 1
 <2><1cd>: Abbrev Number: 0
 <1><1ce>: Abbrev Number: 20 (DW_TAG_subprogram)
    <1cf>   DW_AT_low_pc      : (index: 0x1): 0x1100
    <1d0>   DW_AT_high_pc     : 0x1
    <1d4>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <1d6>   DW_AT_object_pointer: <0x1e0>
    <1da>   DW_AT_call_all_calls: 1
    <1da>   DW_AT_decl_line   : 11
    <1db>   DW_AT_linkage_name: (indexed string: 0x11): _ZN2ee5OwnerD2Ev
    <1dc>   DW_AT_specification: <0x3e>
 <2><1e0>: Abbrev Number: 21 (DW_TAG_formal_parameter)
    <1e1>   DW_AT_name        : (indexed string: 0x24): this
    <1e2>   DW_AT_type        : <0x215>
    <1e6>   DW_AT_artificial  : 1
 <2><1e6>: Abbrev Number: 0
 <1><1e7>: Abbrev Number: 20 (DW_TAG_subprogram)
    <1e8>   DW_AT_low_pc      : (index: 0x2): 0x1110
    <1e9>   DW_AT_high_pc     : 0x5
    <1ed>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <1ef>   DW_AT_object_pointer: <0x1f9>
    <1f3>   DW_AT_call_all_calls: 1
    <1f3>   DW_AT_decl_line   : 12
    <1f4>   DW_AT_linkage_name: (indexed string: 0x12): _ZN2ee6HandleC2ERKS0_
    <1f5>   DW_AT_specification: <0x59>
 <2><1f9>: Abbrev Number: 22 (DW_TAG_formal_parameter)
    <1fa>   DW_AT_location    : 1 byte block: 55 	(DW_OP_reg5 (rdi))
    <1fc>   DW_AT_name        : (indexed string: 0x24): this
    <1fd>   DW_AT_type        : <0x21a>
    <201>   DW_AT_artificial  : 1
 <2><201>: Abbrev Number: 11 (DW_TAG_formal_parameter)
    <202>   DW_AT_location    : 1 byte block: 54 	(DW_OP_reg4 (rsi))
    <204>   DW_AT_name        : (indexed string: 0x25): o
    <205>   DW_AT_decl_file   : 0
    <206>   DW_AT_decl_line   : 12
    <207>   DW_AT_type        : <0x1a6>
 <2><20b>: Abbrev Number: 0
 <1><20c>: Abbrev Number: 16 (DW_TAG_base_type)
    <20d>   DW_AT_name        : (indexed string: 0x1d): double
    <20e>   DW_AT_encoding    : 4	(float)
    <20f>   DW_AT_byte_size   : 8
 <1><210>: Abbrev Number: 17 (DW_TAG_pointer_type)
    <211>   DW_AT_type        : <0x6a>
 <1><215>: Abbrev Number: 17 (DW_TAG_pointer_type)
    <216>   DW_AT_type        : <0x2f>
 <1><21a>: Abbrev Number: 17 (DW_TAG_pointer_type)
    <21b>   DW_AT_type        : <0x4a>
 <1><21f>: Abbrev Number: 17 (DW_TAG_pointer_type)
    <220>   DW_AT_type        : <0x195>
 <1><224>: Abbrev Number: 0

"""

@testset "By-value records at the thunk boundary" begin

    rt, sd, _, _, ra = BVC.parse_dwarf_dump(RECORD_DUMP)

    @testset "record_abi: calling convention and emptiness from DWARF" begin
        @test length(rt) == 11                      # the dump parsed at all
        @test ra["Tag"]    == Dict("byte_size" => "0x1", "pass" => "value",     "empty" => true)
        @test ra["Guard"]  == Dict("byte_size" => "0x1", "pass" => "reference", "empty" => true)
        @test ra["Owner"]  == Dict("byte_size" => "0x4", "pass" => "reference", "empty" => false)
        @test ra["Handle"] == Dict("byte_size" => "0x4", "pass" => "reference", "empty" => false)
        # A base class is bytes the callee reads: not empty.
        @test ra["P3"]     == Dict("byte_size" => "0x8", "pass" => "value",     "empty" => false)
        # A declaration has no byte_size, so no facts; `Pair` is only ever a
        # base and `Unused` only a pointee, so neither is recorded.
        @test sort(collect(keys(ra))) == ["Guard", "Handle", "Owner", "P3", "Tag"]
        # The member tables still describe only records with members.
        @test sort(collect(keys(sd))) == ["Handle", "Owner", "Pair"]
    end

    @testset "_by_value_record_name / _merge_record_facts!" begin
        @test BVC._by_value_record_name("const Tag") == "Tag"
        @test BVC._by_value_record_name("Tag const") == "Tag"
        @test BVC._by_value_record_name("fstring<const char *>") === nothing   # a `*` inside
        @test BVC._by_value_record_name("Box<const int>") == "Box<const int>"  # inner cv kept
        for t in ("Tag*", "Tag&", "Tag&&", "void", "unknown", "", "int[4]", "function_ptr(int)")
            @test BVC._by_value_record_name(t) === nothing
        end
        facts = Dict{String,Dict{String,Any}}()
        e1 = Dict{String,Any}("byte_size" => "0x1", "pass" => "value", "empty" => true)
        BVC._merge_record_facts!(facts, "T", e1)
        BVC._merge_record_facts!(facts, "T", copy(e1))              # another CU, same type
        @test !haskey(facts["T"], "conflict")
        BVC._merge_record_facts!(facts, "T", Dict{String,Any}("byte_size" => "0x10", "pass" => "value", "empty" => false))
        @test facts["T"]["conflict"] == true                       # two scopes, one name
    end

    structs = merge(Dict{String,Any}(sd), Dict{String,Any}(
        "Box<double>" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x8",
            "members" => [Dict{String,Any}("name" => "v", "c_type" => "double",
                          "julia_type" => "Cdouble", "size" => 8, "offset" => "0x0")]),
        "__enum__Color" => Dict{String,Any}("kind" => "enum", "julia_type" => "Cint",
            "byte_size" => "0x4", "enumerators" => [])))

    @testset "by_value_crossing" begin
        k(t; abi = ra) = BVF.by_value_crossing(t, structs, abi)
        for t in ("int", "double", "bool", "Pair*", "const Pair&", "Tag&&", "void",
                  "function_ptr(int; int)", "std::vector<int, std::allocator<int> >")
            @test k(t) == (:direct, nothing)
        end
        @test k("Pair") == (:layout, "Pair")
        @test k("ee::Pair") == (:layout, "Pair")              # demangler spelling
        @test k("Box<double>") == (:layout, "Box<double>")    # template spelling
        @test k("Color") == (:enum, "__enum__Color")
        @test k("ns::Color") == (:enum, "__enum__Color")
        @test k("Tag") == (:ignore, nothing)
        @test k("const Tag") == (:ignore, nothing)
        @test k("ee::Tag") == (:ignore, nothing)
        @test k("Guard") == (:indirect, nothing)
        @test k("P3") == (:opaque, nothing)                   # base but no members
        @test k("Range") == (:opaque, nothing)                # declaration only
        @test k("long double") == (:opaque, nothing)          # no scalar mapping
        @test k("basic_string_view<char, std::char_traits<char> >") == (:opaque, nothing)
        # Without record_abi (metadata from before it existed) nothing is empty.
        @test k("Tag"; abi = Dict()) == (:opaque, nothing)
        # A conflicted name is never trusted.
        @test k("Tag"; abi = Dict("Tag" => Dict{String,Any}("byte_size" => "0x1",
                   "pass" => "value", "empty" => true, "conflict" => true))) == (:opaque, nothing)
        # A C record carries no convention; only the GNU size-0 struct is empty.
        @test k("CEmpty"; abi = Dict("CEmpty" => Dict{String,Any}("byte_size" => "0x0",
                   "pass" => nothing, "empty" => true))) == (:ignore, nothing)
    end

    fn(mangled, params, ret) = Dict{String,Any}(
        "name" => mangled, "mangled" => mangled, "is_method" => false,
        "demangled" => mangled,
        "parameters" => [Dict{String,Any}("name" => "a$i", "c_type" => c) for (i, c) in enumerate(params)],
        "return_type" => Dict{String,Any}("c_type" => ret))

    @testset "thunk_crossings names the reason" begin
        @test BVF.thunk_crossings(fn("f", ["Tag", "int"], "int"), structs, ra).trap === nothing
        @test occursin("takes `P3` by value", BVF.thunk_crossings(fn("f", ["P3"], "int"), structs, ra).trap)
        # A pass-by-reference record comes back through a hidden pointer the
        # thunk does not pass.
        @test occursin("hidden pointer", BVF.thunk_crossings(fn("f", ["int"], "Guard"), structs, ra).trap)
        @test BVF.thunk_crossings(fn("f", ["Guard"], "int"), structs, ra).trap === nothing
    end

    @testset "generate_function_thunks acts on the verdict" begin
        gen(f) = BVF.generate_function_thunks([f], structs; may_throw = true, record_abi = ra)
        decl(ir, m) = strip(first(l for l in split(ir, '\n') if startswith(l, "func.func private @$m(")))

        # Empty record first: dropped from the callee's arguments (clang drops
        # it), while the slot it occupies in the Julia-side array is skipped,
        # so `x` is still read from the SECOND slot.
        ir = gen(fn("_ZN2ee8with_tagENS_3TagEi", ["Tag", "int"], "int"))
        @test decl(ir, "_ZN2ee8with_tagENS_3TagEi") == "func.func private @_ZN2ee8with_tagENS_3TagEi(i32) -> i32"
        @test occursin("fieldOffset = 8 : i64", ir)
        @test !occursin("fieldOffset = 0 : i64", ir)

        # Empty record returned: the callee is void, and so is the thunk.
        ir = gen(fn("_ZN2ee8make_tagEi", ["int"], "Tag"))
        @test decl(ir, "_ZN2ee8make_tagEi") == "func.func private @_ZN2ee8make_tagEi(i32)"
        @test occursin(r"func\.func @_ZN2ee8make_tagEi_thunk\(%args_ptr: !llvm\.ptr\)\s+attributes", ir)

        # Non-trivial for calls: the callee takes the object's address.
        ir = gen(fn("_ZN2ee10with_guardENS_5GuardEi", ["Guard", "int"], "int"))
        @test decl(ir, "_ZN2ee10with_guardENS_5GuardEi") ==
              "func.func private @_ZN2ee10with_guardENS_5GuardEi(!llvm.ptr, i32) -> i32"

        # Untypable: no thunk at all, and the .mlir says why.
        ir = gen(fn("_ZN2ee6p3_sumENS_2P3E", ["P3"], "int"))
        @test !occursin("func.func", ir)
        @test occursin("// no thunk for _ZN2ee6p3_sumENS_2P3E: takes `P3` by value", ir)

        # Template record, both directions: its layout, not a pointer in and an
        # integer blob out.
        # (`is_struct_packed` calls every padding-free struct packed, so the
        # spelling may be `packed (f64)`; the layout is what matters.)
        ir = gen(fn("_Z3boxv", ["Box<double>"], "Box<double>"))
        @test occursin(r"^func\.func private @_Z3boxv\(!llvm\.struct<[^>]*\(f64\)\s*>\) -> !llvm\.struct<[^>]*\(f64\)\s*>$",
                       decl(ir, "_Z3boxv"))

        # Qualified enum: its integer, not a pointer.
        ir = gen(fn("_Z3colN2ns5ColorE", ["ns::Color"], "ns::Color"))
        @test decl(ir, "_Z3colN2ns5ColorE") == "func.func private @_Z3colN2ns5ColorE(i32) -> i32"
    end

    @testset "_record_name_as_emitted: Base-colliding record names" begin
        recs = Set(["Pair", "Tag", "Vector"])
        @test BVW._record_name_as_emitted("Pair", recs) == "c_Pair"
        @test BVW._record_name_as_emitted("Ptr{Pair}", recs) == "Ptr{c_Pair}"
        @test BVW._record_name_as_emitted("Ref{Vector}", recs) == "Ref{c_Vector}"
        @test BVW._record_name_as_emitted("Tag", recs) == "Tag"
        @test BVW._record_name_as_emitted("Cint", recs) == "Cint"
        @test BVW._record_name_as_emitted("Any", recs) == "Any"
        @test BVW._record_name_as_emitted("Pair", Set{String}()) == "Pair"   # not a record here
    end

    @testset "method dedup prefers a callable over a trap" begin
        callable = "\"\"\"\n- Mangled symbol: `_ZN4pugi8xml_node5childEPKc`\n\"\"\"\nfunction pugi_xml_node_child(this::Any, name::Any)\n    return 1\nend\n"
        trap = "\"\"\"\n- Mangled symbol: `_ZN4pugi8xml_node5childESt17basic_string_view`\n\"\"\"\nfunction pugi_xml_node_child(this::Any, name::Any)\n    Base.error(\"ABI Safety Trap: cannot call 'pugi_xml_node_child'\")\nend\n"
        @test BVW._dedup_method_chunks([callable, trap]) == [callable]   # trap last
        @test BVW._dedup_method_chunks([trap, callable]) == [callable]   # trap first
        other = replace(callable, "return 1" => "return 2")
        @test BVW._dedup_method_chunks([callable, other]) == [other]     # last callable wins
        @test BVW._is_trap_chunk("function f()\n    Base.error(\"\"\"\n    ABI Safety Trap: …\n    \"\"\")\nend")
        @test BVW._is_trap_chunk("function f()\n    Base.error(\"\"\"\n    FFI Safety Trap: …\n    \"\"\")\nend")
        @test !BVW._is_trap_chunk(callable)
    end
end
