#!/usr/bin/env julia
# test/test_memberless_records.jl — every defined library record has a
# `struct_definitions` entry, members or not (2026-11-14; Bullet3's
# GENERATOR-memberless-class-size report).
#
# The parser gave a record with no member of its own no entry (one exception
# since 2026-11-13: a toolchain record laid out from its bases). A derived class
# that adds methods and no field — 39 of Bullet's 603 records, `btBoxShape`,
# `btSphereShape`, `btQuaternion`, `btVector4` among them — then had no
# `STRUCT_SIZES` entry to size its constructor's storage by, an empty
# `struct X end` for a Julia type, and every class deriving from it lost the
# bases above it (`btConvexInternalShape` carried a 32-byte pad where
# `btCollisionShape`'s fields belong, through the memberless `btConvexShape`).
#
# Now the entry carries `members = []` and `base_classes`. Nothing downstream is
# new: GeneratorCpp's `flatten_struct_members` turns the non-virtual bases into
# fields (`btQuaternion` is `m_floats::NTuple{4, Cfloat}`), the SysV classifier
# reads a record's class from its bases' leaves, and StructGen pads the thunk
# body to the DWARF size. Two readers had to learn the entry exists: an EMPTY
# class (a stateless functor) keeps the convention's verdict (dropped on SysV,
# a register pad elsewhere) ahead of its one-byte layout
# (`FunctionGen.by_value_crossing`), and `is_ccall_safe` sends such a crossing
# to the thunk, which drops it, where a ccall would pass the byte.
#
# No toolchain: the DWARF is a verbatim clang 23 readelf dump of
# test/memberless_test/src/memberless.cpp (`-std=c++17 -g -O1 -fPIC -shared`):
#   namespace mb {
#   struct Quad { float m[4]; };
#   struct Derived : Quad {};                            // bytes all from the base
#   struct P { bool operator()(int a, int b) const; };   // stateless functor, 1 byte
#   struct Poly { virtual ~Poly(); int id; };
#   struct PolyKid : Poly { PolyKid(int v); };           // Bullet's shape
#   struct VB : Poly, virtual Quad {};                   // a virtual base
#   float dsum(Derived); Derived dmake(float); int pcall(P, int, int);
#   int vb_first(VB); int vb_make(float); int kid_id(PolyKid*);
#   }
# test_memberless_live.jl (devtests) builds it and crosses each record.

using Test
using RepliBuild
using Libdl

const _MR_C  = RepliBuild.Compiler
const _MR_FG = RepliBuild.JLCSIRGenerator.FunctionGen
const _MR_W  = RepliBuild.Wrapper
const _MR_D  = RepliBuild.Wrapper   # DispatchLogic lives in it

const MR_DUMP = raw"""
Contents of the .debug_info section:

  Compilation Unit @ offset 0:
   Length:        0x32c (32-bit)
   Version:       5
   Unit Type:     DW_UT_compile (1)
   Abbrev Offset: 0
   Pointer Size:  8
 <0><c>: Abbrev Number: 1 (DW_TAG_compile_unit)
    <d>   DW_AT_producer    : (indexed string: 0): clang version 23.1.5
    <e>   DW_AT_language    : 33	(C++14)
    <10>   DW_AT_name        : (indexed string: 0x1): mb.cpp
    <11>   DW_AT_str_offsets_base: 0x8
    <15>   DW_AT_stmt_list   : 0
    <19>   DW_AT_comp_dir    : (indexed string: 0x2): <dir>
    <1a>   DW_AT_low_pc      : 0
    <22>   DW_AT_ranges      : (index: 0): 0x10
    <23>   DW_AT_addr_base   : 0x8
    <27>   DW_AT_rnglists_base: 0xc
    <2b>   DW_AT_loclists_base: 0xc (location list)
 <1><2f>: Abbrev Number: 2 (DW_TAG_base_type)
    <30>   DW_AT_name        : (indexed string: 0x3): int
    <31>   DW_AT_encoding    : 5	(signed)
    <32>   DW_AT_byte_size   : 4
 <1><33>: Abbrev Number: 3 (DW_TAG_namespace)
    <34>   DW_AT_name        : (indexed string: 0x4): mb
 <2><35>: Abbrev Number: 4 (DW_TAG_structure_type)
    <36>   DW_AT_calling_convention: 5	(pass by value)
    <37>   DW_AT_name        : (indexed string: 0x8): P
    <38>   DW_AT_byte_size   : 1
    <39>   DW_AT_decl_file   : 0
    <3a>   DW_AT_decl_line   : 4
 <3><3b>: Abbrev Number: 5 (DW_TAG_subprogram)
    <3c>   DW_AT_linkage_name: (indexed string: 0x5): _ZNK2mb1PclEii
    <3d>   DW_AT_name        : (indexed string: 0x6): operator()
    <3e>   DW_AT_decl_file   : 0
    <3f>   DW_AT_decl_line   : 4
    <40>   DW_AT_type        : <0x1f6>
    <44>   DW_AT_declaration : 1
    <44>   DW_AT_external    : 1
 <4><44>: Abbrev Number: 6 (DW_TAG_formal_parameter)
    <45>   DW_AT_type        : <0x1fa>
    <49>   DW_AT_artificial  : 1
 <4><49>: Abbrev Number: 7 (DW_TAG_formal_parameter)
    <4a>   DW_AT_type        : <0x2f>
 <4><4e>: Abbrev Number: 7 (DW_TAG_formal_parameter)
    <4f>   DW_AT_type        : <0x2f>
 <4><53>: Abbrev Number: 0
 <3><54>: Abbrev Number: 0
 <2><55>: Abbrev Number: 8 (DW_TAG_structure_type)
    <56>   DW_AT_containing_type: <0x55>
    <5a>   DW_AT_calling_convention: 4	(pass by ref)
    <5b>   DW_AT_name        : (indexed string: 0xe): Poly
    <5c>   DW_AT_byte_size   : 16
    <5d>   DW_AT_decl_file   : 0
    <5e>   DW_AT_decl_line   : 5
 <3><5f>: Abbrev Number: 9 (DW_TAG_member)
    <60>   DW_AT_name        : (indexed string: 0x9): _vptr$Poly
    <61>   DW_AT_type        : <0x232>
    <65>   DW_AT_data_member_location: 0
    <66>   DW_AT_artificial  : 1
 <3><66>: Abbrev Number: 10 (DW_TAG_member)
    <67>   DW_AT_name        : (indexed string: 0xb): id
    <68>   DW_AT_type        : <0x2f>
    <6c>   DW_AT_decl_file   : 0
    <6d>   DW_AT_decl_line   : 5
    <6e>   DW_AT_data_member_location: 8
 <3><6f>: Abbrev Number: 11 (DW_TAG_subprogram)
    <70>   DW_AT_linkage_name: (indexed string: 0xc): _ZN2mb4PolyD4Ev
    <71>   DW_AT_name        : (indexed string: 0xd): ~Poly
    <72>   DW_AT_decl_file   : 0
    <73>   DW_AT_decl_line   : 5
    <74>   DW_AT_virtuality  : 1	(virtual)
    <75>   DW_AT_vtable_elem_location: 2 byte block: 10 0 	(DW_OP_constu: 0)
    <78>   DW_AT_declaration : 1
    <78>   DW_AT_external    : 1
    <78>   DW_AT_containing_type: <0x55>
 <4><7c>: Abbrev Number: 6 (DW_TAG_formal_parameter)
    <7d>   DW_AT_type        : <0x242>
    <81>   DW_AT_artificial  : 1
 <4><81>: Abbrev Number: 0
 <3><82>: Abbrev Number: 12 (DW_TAG_variable)
    <83>   DW_AT_name        : (indexed string: 0x1e): __clang_vtable
    <84>   DW_AT_type        : <0x27b>
    <88>   DW_AT_external    : 1
    <88>   DW_AT_declaration : 1
    <88>   DW_AT_artificial  : 1
    <88>   DW_AT_accessibility: 3	(private)
 <3><89>: Abbrev Number: 0
 <2><8a>: Abbrev Number: 8 (DW_TAG_structure_type)
    <8b>   DW_AT_containing_type: <0x55>
    <8f>   DW_AT_calling_convention: 4	(pass by ref)
    <90>   DW_AT_name        : (indexed string: 0x13): PolyKid
    <91>   DW_AT_byte_size   : 16
    <92>   DW_AT_decl_file   : 0
    <93>   DW_AT_decl_line   : 6
 <3><94>: Abbrev Number: 13 (DW_TAG_inheritance)
    <95>   DW_AT_type        : <0x55>
    <99>   DW_AT_data_member_location: 0
 <3><9a>: Abbrev Number: 14 (DW_TAG_subprogram)
    <9b>   DW_AT_linkage_name: (indexed string: 0x12): _ZN2mb7PolyKidC4Ei
    <9c>   DW_AT_name        : (indexed string: 0x13): PolyKid
    <9d>   DW_AT_decl_file   : 0
    <9e>   DW_AT_decl_line   : 6
    <9f>   DW_AT_declaration : 1
    <9f>   DW_AT_external    : 1
 <4><9f>: Abbrev Number: 6 (DW_TAG_formal_parameter)
    <a0>   DW_AT_type        : <0x280>
    <a4>   DW_AT_artificial  : 1
 <4><a4>: Abbrev Number: 7 (DW_TAG_formal_parameter)
    <a5>   DW_AT_type        : <0x2f>
 <4><a9>: Abbrev Number: 0
 <3><aa>: Abbrev Number: 15 (DW_TAG_subprogram)
    <ab>   DW_AT_linkage_name: (indexed string: 0x1c): _ZN2mb7PolyKidD4Ev
    <ac>   DW_AT_name        : (indexed string: 0x1d): ~PolyKid
    <ad>   DW_AT_virtuality  : 1	(virtual)
    <ae>   DW_AT_vtable_elem_location: 2 byte block: 10 0 	(DW_OP_constu: 0)
    <b1>   DW_AT_declaration : 1
    <b1>   DW_AT_artificial  : 1
    <b1>   DW_AT_external    : 1
    <b1>   DW_AT_containing_type: <0x8a>
 <4><b5>: Abbrev Number: 6 (DW_TAG_formal_parameter)
    <b6>   DW_AT_type        : <0x280>
    <ba>   DW_AT_artificial  : 1
 <4><ba>: Abbrev Number: 0
 <3><bb>: Abbrev Number: 12 (DW_TAG_variable)
    <bc>   DW_AT_name        : (indexed string: 0x1e): __clang_vtable
    <bd>   DW_AT_type        : <0x27b>
    <c1>   DW_AT_external    : 1
    <c1>   DW_AT_declaration : 1
    <c1>   DW_AT_artificial  : 1
    <c1>   DW_AT_accessibility: 3	(private)
 <3><c2>: Abbrev Number: 0
 <2><c3>: Abbrev Number: 16 (DW_TAG_subprogram)
    <c4>   DW_AT_low_pc      : (index: 0x4): 0x1180
    <c5>   DW_AT_high_pc     : 0x18
    <c9>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <cb>   DW_AT_call_all_calls: 1
    <cb>   DW_AT_linkage_name: (indexed string: 0x24): _ZN2mb4dsumENS_7DerivedE
    <cc>   DW_AT_name        : (indexed string: 0x25): dsum
    <cd>   DW_AT_decl_file   : 0
    <ce>   DW_AT_decl_line   : 11
    <cf>   DW_AT_type        : <0x2b5>
    <d3>   DW_AT_external    : 1
 <3><d3>: Abbrev Number: 17 (DW_TAG_formal_parameter)
    <d4>   DW_AT_name        : (indexed string: 0x34): d
    <d5>   DW_AT_decl_file   : 0
    <d6>   DW_AT_decl_line   : 11
    <d7>   DW_AT_type        : <0x1e8>
 <3><db>: Abbrev Number: 0
 <2><dc>: Abbrev Number: 16 (DW_TAG_subprogram)
    <dd>   DW_AT_low_pc      : (index: 0x5): 0x11a0
    <de>   DW_AT_high_pc     : 0x5e
    <e2>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <e4>   DW_AT_call_all_calls: 1
    <e4>   DW_AT_linkage_name: (indexed string: 0x26): _ZN2mb5dmakeEf
    <e5>   DW_AT_name        : (indexed string: 0x27): dmake
    <e6>   DW_AT_decl_file   : 0
    <e7>   DW_AT_decl_line   : 12
    <e8>   DW_AT_type        : <0x1e8>
    <ec>   DW_AT_external    : 1
 <3><ec>: Abbrev Number: 18 (DW_TAG_formal_parameter)
    <ed>   DW_AT_location    : (index: 0x2): 0x3e (location list)
    <ee>   DW_AT_name        : (indexed string: 0x35): x
    <ef>   DW_AT_decl_file   : 0
    <f0>   DW_AT_decl_line   : 12
    <f1>   DW_AT_type        : <0x2b5>
 <3><f5>: Abbrev Number: 19 (DW_TAG_variable)
    <f6>   DW_AT_location    : 2 byte block: 91 0 	(DW_OP_fbreg: 0)
    <f9>   DW_AT_name        : (indexed string: 0x34): d
    <fa>   DW_AT_decl_file   : 0
    <fb>   DW_AT_decl_line   : 12
    <fc>   DW_AT_type        : <0x1e8>
 <3><100>: Abbrev Number: 20 (DW_TAG_lexical_block)
    <101>   DW_AT_low_pc      : (index: 0x6): 0x11b4
    <102>   DW_AT_high_pc     : 0x25
 <4><106>: Abbrev Number: 19 (DW_TAG_variable)
    <107>   DW_AT_location    : 1 byte block: 50 	(DW_OP_reg0 (rax))
    <109>   DW_AT_name        : (indexed string: 0x36): i
    <10a>   DW_AT_decl_file   : 0
    <10b>   DW_AT_decl_line   : 12
    <10c>   DW_AT_type        : <0x2f>
 <4><110>: Abbrev Number: 0
 <3><111>: Abbrev Number: 0
 <2><112>: Abbrev Number: 16 (DW_TAG_subprogram)
    <113>   DW_AT_low_pc      : (index: 0x7): 0x1200
    <114>   DW_AT_high_pc     : 0x8
    <118>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <11a>   DW_AT_call_all_calls: 1
    <11a>   DW_AT_linkage_name: (indexed string: 0x29): _ZN2mb5pcallENS_1PEii
    <11b>   DW_AT_name        : (indexed string: 0x2a): pcall
    <11c>   DW_AT_decl_file   : 0
    <11d>   DW_AT_decl_line   : 13
    <11e>   DW_AT_type        : <0x2f>
    <122>   DW_AT_external    : 1
 <3><122>: Abbrev Number: 17 (DW_TAG_formal_parameter)
    <123>   DW_AT_name        : (indexed string: 0x37): p
    <124>   DW_AT_decl_file   : 0
    <125>   DW_AT_decl_line   : 13
    <126>   DW_AT_type        : <0x35>
 <3><12a>: Abbrev Number: 21 (DW_TAG_formal_parameter)
    <12b>   DW_AT_location    : 1 byte block: 55 	(DW_OP_reg5 (rdi))
    <12d>   DW_AT_name        : (indexed string: 0x32): a
    <12e>   DW_AT_decl_file   : 0
    <12f>   DW_AT_decl_line   : 13
    <130>   DW_AT_type        : <0x2f>
 <3><134>: Abbrev Number: 21 (DW_TAG_formal_parameter)
    <135>   DW_AT_location    : 1 byte block: 54 	(DW_OP_reg4 (rsi))
    <137>   DW_AT_name        : (indexed string: 0x33): b
    <138>   DW_AT_decl_file   : 0
    <139>   DW_AT_decl_line   : 13
    <13a>   DW_AT_type        : <0x2f>
 <3><13e>: Abbrev Number: 0
 <2><13f>: Abbrev Number: 22 (DW_TAG_subprogram)
    <140>   DW_AT_low_pc      : (index: 0x8): 0x1210
    <141>   DW_AT_high_pc     : 0xd
    <145>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <147>   DW_AT_call_all_calls: 1
    <147>   DW_AT_abstract_origin: <0x154>
 <3><14b>: Abbrev Number: 23 (DW_TAG_formal_parameter)
    <14c>   DW_AT_location    : 2 byte block: 75 0 	(DW_OP_breg5 (rdi): 0)
    <14f>   DW_AT_abstract_origin: <0x15d>
 <3><153>: Abbrev Number: 0
 <2><154>: Abbrev Number: 24 (DW_TAG_subprogram)
    <155>   DW_AT_linkage_name: (indexed string: 0x14): _ZN2mb8vb_firstENS_2VBE
    <156>   DW_AT_name        : (indexed string: 0x15): vb_first
    <157>   DW_AT_decl_file   : 0
    <158>   DW_AT_decl_line   : 14
    <159>   DW_AT_type        : <0x2f>
    <15d>   DW_AT_external    : 1
    <15d>   DW_AT_inline      : 1	(inlined)
 <3><15d>: Abbrev Number: 17 (DW_TAG_formal_parameter)
    <15e>   DW_AT_name        : (indexed string: 0x16): v
    <15f>   DW_AT_decl_file   : 0
    <160>   DW_AT_decl_line   : 14
    <161>   DW_AT_type        : <0x166>
 <3><165>: Abbrev Number: 0
 <2><166>: Abbrev Number: 8 (DW_TAG_structure_type)
    <167>   DW_AT_containing_type: <0x55>
    <16b>   DW_AT_calling_convention: 4	(pass by ref)
    <16c>   DW_AT_name        : (indexed string: 0x1b): VB
    <16d>   DW_AT_byte_size   : 32
    <16e>   DW_AT_decl_file   : 0
    <16f>   DW_AT_decl_line   : 7
 <3><170>: Abbrev Number: 13 (DW_TAG_inheritance)
    <171>   DW_AT_type        : <0x55>
    <175>   DW_AT_data_member_location: 0
 <3><176>: Abbrev Number: 25 (DW_TAG_inheritance)
    <177>   DW_AT_type        : <0x18c>
    <17b>   DW_AT_data_member_location: 7 byte block: 12 6 10 18 1c 6 22 	(DW_OP_dup; DW_OP_deref; DW_OP_constu: 24; DW_OP_minus; DW_OP_deref; DW_OP_plus)
    <183>   DW_AT_virtuality  : 1	(virtual)
 <3><184>: Abbrev Number: 12 (DW_TAG_variable)
    <185>   DW_AT_name        : (indexed string: 0x1e): __clang_vtable
    <186>   DW_AT_type        : <0x27b>
    <18a>   DW_AT_external    : 1
    <18a>   DW_AT_declaration : 1
    <18a>   DW_AT_artificial  : 1
    <18a>   DW_AT_accessibility: 3	(private)
 <3><18b>: Abbrev Number: 0
 <2><18c>: Abbrev Number: 4 (DW_TAG_structure_type)
    <18d>   DW_AT_calling_convention: 5	(pass by value)
    <18e>   DW_AT_name        : (indexed string: 0x1a): Quad
    <18f>   DW_AT_byte_size   : 16
    <190>   DW_AT_decl_file   : 0
    <191>   DW_AT_decl_line   : 2
 <3><192>: Abbrev Number: 10 (DW_TAG_member)
    <193>   DW_AT_name        : (indexed string: 0x17): m
    <194>   DW_AT_type        : <0x2a9>
    <198>   DW_AT_decl_file   : 0
    <199>   DW_AT_decl_line   : 2
    <19a>   DW_AT_data_member_location: 0
 <3><19b>: Abbrev Number: 0
 <2><19c>: Abbrev Number: 16 (DW_TAG_subprogram)
    <19d>   DW_AT_low_pc      : (index: 0x9): 0x1220
    <19e>   DW_AT_high_pc     : 0x5
    <1a2>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <1a4>   DW_AT_call_all_calls: 1
    <1a4>   DW_AT_linkage_name: (indexed string: 0x2b): _ZN2mb7vb_makeEf
    <1a5>   DW_AT_name        : (indexed string: 0x2c): vb_make
    <1a6>   DW_AT_decl_file   : 0
    <1a7>   DW_AT_decl_line   : 15
    <1a8>   DW_AT_type        : <0x2f>
    <1ac>   DW_AT_external    : 1
 <3><1ac>: Abbrev Number: 21 (DW_TAG_formal_parameter)
    <1ad>   DW_AT_location    : 1 byte block: 61 	(DW_OP_reg17 (xmm0))
    <1af>   DW_AT_name        : (indexed string: 0x35): x
    <1b0>   DW_AT_decl_file   : 0
    <1b1>   DW_AT_decl_line   : 15
    <1b2>   DW_AT_type        : <0x2b5>
 <3><1b6>: Abbrev Number: 26 (DW_TAG_variable)
    <1b7>   DW_AT_location    : (index: 0x3): 0x51 (location list)
    <1b8>   DW_AT_name        : (indexed string: 0x16): v
    <1b9>   DW_AT_decl_file   : 0
    <1ba>   DW_AT_decl_line   : 15
    <1bb>   DW_AT_type        : <0x166>
 <3><1bf>: Abbrev Number: 27 (DW_TAG_inlined_subroutine)
    <1c0>   DW_AT_abstract_origin: <0x154>
    <1c4>   DW_AT_low_pc      : (index: 0x9): 0x1220
    <1c5>   DW_AT_high_pc     : 0x4
    <1c9>   DW_AT_call_file   : 0
    <1ca>   DW_AT_call_line   : 15
    <1cb>   DW_AT_call_column : 59
 <3><1cc>: Abbrev Number: 0
 <2><1cd>: Abbrev Number: 16 (DW_TAG_subprogram)
    <1ce>   DW_AT_low_pc      : (index: 0xa): 0x1228
    <1cf>   DW_AT_high_pc     : 0x4
    <1d3>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <1d5>   DW_AT_call_all_calls: 1
    <1d5>   DW_AT_linkage_name: (indexed string: 0x2d): _ZN2mb6kid_idEPNS_7PolyKidE
    <1d6>   DW_AT_name        : (indexed string: 0x2e): kid_id
    <1d7>   DW_AT_decl_file   : 0
    <1d8>   DW_AT_decl_line   : 16
    <1d9>   DW_AT_type        : <0x2f>
    <1dd>   DW_AT_external    : 1
 <3><1dd>: Abbrev Number: 21 (DW_TAG_formal_parameter)
    <1de>   DW_AT_location    : 1 byte block: 55 	(DW_OP_reg5 (rdi))
    <1e0>   DW_AT_name        : (indexed string: 0x38): k
    <1e1>   DW_AT_decl_file   : 0
    <1e2>   DW_AT_decl_line   : 16
    <1e3>   DW_AT_type        : <0x32a>
 <3><1e7>: Abbrev Number: 0
 <2><1e8>: Abbrev Number: 4 (DW_TAG_structure_type)
    <1e9>   DW_AT_calling_convention: 5	(pass by value)
    <1ea>   DW_AT_name        : (indexed string: 0x28): Derived
    <1eb>   DW_AT_byte_size   : 16
    <1ec>   DW_AT_decl_file   : 0
    <1ed>   DW_AT_decl_line   : 3
 <3><1ee>: Abbrev Number: 13 (DW_TAG_inheritance)
    <1ef>   DW_AT_type        : <0x18c>
    <1f3>   DW_AT_data_member_location: 0
 <3><1f4>: Abbrev Number: 0
 <2><1f5>: Abbrev Number: 0
 <1><1f6>: Abbrev Number: 2 (DW_TAG_base_type)
    <1f7>   DW_AT_name        : (indexed string: 0x7): bool
    <1f8>   DW_AT_encoding    : 2	(boolean)
    <1f9>   DW_AT_byte_size   : 1
 <1><1fa>: Abbrev Number: 28 (DW_TAG_pointer_type)
    <1fb>   DW_AT_type        : <0x1ff>
 <1><1ff>: Abbrev Number: 29 (DW_TAG_const_type)
    <200>   DW_AT_type        : <0x35>
 <1><204>: Abbrev Number: 30 (DW_TAG_subprogram)
    <205>   DW_AT_low_pc      : (index: 0): 0x1130
    <206>   DW_AT_high_pc     : 0x6
    <20a>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <20c>   DW_AT_object_pointer: <0x215>
    <210>   DW_AT_call_all_calls: 1
    <210>   DW_AT_decl_line   : 8
    <211>   DW_AT_specification: <0x3b>
 <2><215>: Abbrev Number: 31 (DW_TAG_formal_parameter)
    <216>   DW_AT_location    : 1 byte block: 55 	(DW_OP_reg5 (rdi))
    <218>   DW_AT_name        : (indexed string: 0x31): this
    <219>   DW_AT_type        : <0x320>
    <21d>   DW_AT_artificial  : 1
 <2><21d>: Abbrev Number: 21 (DW_TAG_formal_parameter)
    <21e>   DW_AT_location    : 1 byte block: 54 	(DW_OP_reg4 (rsi))
    <220>   DW_AT_name        : (indexed string: 0x32): a
    <221>   DW_AT_decl_file   : 0
    <222>   DW_AT_decl_line   : 8
    <223>   DW_AT_type        : <0x2f>
 <2><227>: Abbrev Number: 21 (DW_TAG_formal_parameter)
    <228>   DW_AT_location    : 1 byte block: 51 	(DW_OP_reg1 (rdx))
    <22a>   DW_AT_name        : (indexed string: 0x33): b
    <22b>   DW_AT_decl_file   : 0
    <22c>   DW_AT_decl_line   : 8
    <22d>   DW_AT_type        : <0x2f>
 <2><231>: Abbrev Number: 0
 <1><232>: Abbrev Number: 28 (DW_TAG_pointer_type)
    <233>   DW_AT_type        : <0x237>
 <1><237>: Abbrev Number: 32 (DW_TAG_pointer_type)
    <238>   DW_AT_type        : <0x23d>
    <23c>   DW_AT_name        : (indexed string: 0xa): __vtbl_ptr_type
 <1><23d>: Abbrev Number: 33 (DW_TAG_subroutine_type)
    <23e>   DW_AT_type        : <0x2f>
 <1><242>: Abbrev Number: 28 (DW_TAG_pointer_type)
    <243>   DW_AT_type        : <0x55>
 <1><247>: Abbrev Number: 34 (DW_TAG_subprogram)
    <248>   DW_AT_low_pc      : (index: 0x1): 0x1140
    <249>   DW_AT_high_pc     : 0x17
    <24d>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <24f>   DW_AT_object_pointer: <0x259>
    <253>   DW_AT_call_all_calls: 1
    <253>   DW_AT_decl_line   : 9
    <254>   DW_AT_linkage_name: (indexed string: 0x22): _ZN2mb4PolyD0Ev
    <255>   DW_AT_specification: <0x6f>
 <2><259>: Abbrev Number: 35 (DW_TAG_formal_parameter)
    <25a>   DW_AT_location    : (index: 0): 0x1c (location list)
    <25b>   DW_AT_name        : (indexed string: 0x31): this
    <25c>   DW_AT_type        : <0x325>
    <260>   DW_AT_artificial  : 1
 <2><260>: Abbrev Number: 36 (DW_TAG_call_site)
    <261>   DW_AT_call_origin : <0x26d>
    <265>   DW_AT_call_tail_call: 1
    <265>   DW_AT_call_pc     : (index: 0x2): 0x1152
 <3><266>: Abbrev Number: 37 (DW_TAG_call_site_parameter)
    <267>   DW_AT_location    : 1 byte block: 54 	(DW_OP_reg4 (rsi))
    <269>   DW_AT_call_value  : 1 byte block: 40 	(DW_OP_lit16)
 <3><26b>: Abbrev Number: 0
 <2><26c>: Abbrev Number: 0
 <1><26d>: Abbrev Number: 38 (DW_TAG_subprogram)
    <26e>   DW_AT_linkage_name: (indexed string: 0xf): _ZdlPvm
    <26f>   DW_AT_name        : (indexed string: 0x10): operator delete
    <270>   DW_AT_declaration : 1
    <270>   DW_AT_artificial  : 1
    <270>   DW_AT_external    : 1
 <2><270>: Abbrev Number: 7 (DW_TAG_formal_parameter)
    <271>   DW_AT_type        : <0x27b>
 <2><275>: Abbrev Number: 7 (DW_TAG_formal_parameter)
    <276>   DW_AT_type        : <0x27c>
 <2><27a>: Abbrev Number: 0
 <1><27b>: Abbrev Number: 39 (DW_TAG_pointer_type)
 <1><27c>: Abbrev Number: 2 (DW_TAG_base_type)
    <27d>   DW_AT_name        : (indexed string: 0x11): unsigned long
    <27e>   DW_AT_encoding    : 7	(unsigned)
    <27f>   DW_AT_byte_size   : 8
 <1><280>: Abbrev Number: 28 (DW_TAG_pointer_type)
    <281>   DW_AT_type        : <0x8a>
 <1><285>: Abbrev Number: 34 (DW_TAG_subprogram)
    <286>   DW_AT_low_pc      : (index: 0x3): 0x1160
    <287>   DW_AT_high_pc     : 0x14
    <28b>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <28d>   DW_AT_object_pointer: <0x297>
    <291>   DW_AT_call_all_calls: 1
    <291>   DW_AT_decl_line   : 10
    <292>   DW_AT_linkage_name: (indexed string: 0x23): _ZN2mb7PolyKidC2Ei
    <293>   DW_AT_specification: <0x9a>
 <2><297>: Abbrev Number: 31 (DW_TAG_formal_parameter)
    <298>   DW_AT_location    : 1 byte block: 55 	(DW_OP_reg5 (rdi))
    <29a>   DW_AT_name        : (indexed string: 0x31): this
    <29b>   DW_AT_type        : <0x32a>
    <29f>   DW_AT_artificial  : 1
 <2><29f>: Abbrev Number: 18 (DW_TAG_formal_parameter)
    <2a0>   DW_AT_location    : (index: 0x1): 0x2e (location list)
    <2a1>   DW_AT_name        : (indexed string: 0x16): v
    <2a2>   DW_AT_decl_file   : 0
    <2a3>   DW_AT_decl_line   : 10
    <2a4>   DW_AT_type        : <0x2f>
 <2><2a8>: Abbrev Number: 0
 <1><2a9>: Abbrev Number: 40 (DW_TAG_array_type)
    <2aa>   DW_AT_type        : <0x2b5>
 <2><2ae>: Abbrev Number: 41 (DW_TAG_subrange_type)
    <2af>   DW_AT_type        : <0x2b9>
    <2b3>   DW_AT_count       : 4
 <2><2b4>: Abbrev Number: 0
 <1><2b5>: Abbrev Number: 2 (DW_TAG_base_type)
    <2b6>   DW_AT_name        : (indexed string: 0x18): float
    <2b7>   DW_AT_encoding    : 4	(float)
    <2b8>   DW_AT_byte_size   : 4
 <1><2b9>: Abbrev Number: 42 (DW_TAG_base_type)
    <2ba>   DW_AT_name        : (indexed string: 0x19): __ARRAY_SIZE_TYPE__
    <2bb>   DW_AT_byte_size   : 8
    <2bc>   DW_AT_encoding    : 7	(unsigned)
 <1><2bd>: Abbrev Number: 34 (DW_TAG_subprogram)
    <2be>   DW_AT_low_pc      : (index: 0xb): 0x122c
    <2bf>   DW_AT_high_pc     : 0x1
    <2c3>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <2c5>   DW_AT_object_pointer: <0x2cf>
    <2c9>   DW_AT_call_all_calls: 1
    <2c9>   DW_AT_decl_line   : 9
    <2ca>   DW_AT_linkage_name: (indexed string: 0x2f): _ZN2mb4PolyD2Ev
    <2cb>   DW_AT_specification: <0x6f>
 <2><2cf>: Abbrev Number: 43 (DW_TAG_formal_parameter)
    <2d0>   DW_AT_name        : (indexed string: 0x31): this
    <2d1>   DW_AT_type        : <0x325>
    <2d5>   DW_AT_artificial  : 1
 <2><2d5>: Abbrev Number: 0
 <1><2d6>: Abbrev Number: 44 (DW_TAG_subprogram)
    <2d7>   DW_AT_low_pc      : (index: 0xc): 0x1230
    <2d8>   DW_AT_high_pc     : 0xa
    <2dc>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <2de>   DW_AT_object_pointer: <0x2e9>
    <2e2>   DW_AT_call_all_calls: 1
    <2e2>   DW_AT_decl_file   : 0
    <2e3>   DW_AT_decl_line   : 6
    <2e4>   DW_AT_linkage_name: (indexed string: 0x30): _ZN2mb7PolyKidD0Ev
    <2e5>   DW_AT_specification: <0xaa>
 <2><2e9>: Abbrev Number: 31 (DW_TAG_formal_parameter)
    <2ea>   DW_AT_location    : 1 byte block: 55 	(DW_OP_reg5 (rdi))
    <2ec>   DW_AT_name        : (indexed string: 0x31): this
    <2ed>   DW_AT_type        : <0x32a>
    <2f1>   DW_AT_artificial  : 1
 <2><2f1>: Abbrev Number: 36 (DW_TAG_call_site)
    <2f2>   DW_AT_call_origin : <0x26d>
    <2f6>   DW_AT_call_tail_call: 1
    <2f6>   DW_AT_call_pc     : (index: 0xd): 0x1235
 <3><2f7>: Abbrev Number: 37 (DW_TAG_call_site_parameter)
    <2f8>   DW_AT_location    : 1 byte block: 54 	(DW_OP_reg4 (rsi))
    <2fa>   DW_AT_call_value  : 1 byte block: 40 	(DW_OP_lit16)
 <3><2fc>: Abbrev Number: 37 (DW_TAG_call_site_parameter)
    <2fd>   DW_AT_location    : 1 byte block: 55 	(DW_OP_reg5 (rdi))
    <2ff>   DW_AT_call_value  : 3 byte block: a3 1 55 	(DW_OP_entry_value: (DW_OP_reg5 (rdi)))
 <3><303>: Abbrev Number: 0
 <2><304>: Abbrev Number: 0
 <1><305>: Abbrev Number: 45 (DW_TAG_variable)
    <306>   DW_AT_specification: <0x82>
    <30a>   DW_AT_alignment   : 8
    <30b>   DW_AT_location    :	(DW_OP_addrx <0xe>)
    <30e>   DW_AT_linkage_name: (indexed string: 0x1f): _ZTVN2mb4PolyE
 <1><30f>: Abbrev Number: 45 (DW_TAG_variable)
    <310>   DW_AT_specification: <0xbb>
    <314>   DW_AT_alignment   : 8
    <315>   DW_AT_location    :	(DW_OP_addrx <0xf>)
    <318>   DW_AT_linkage_name: (indexed string: 0x20): _ZTVN2mb7PolyKidE
 <1><319>: Abbrev Number: 46 (DW_TAG_variable)
    <31a>   DW_AT_specification: <0x184>
    <31e>   DW_AT_alignment   : 8
    <31f>   DW_AT_linkage_name: (indexed string: 0x21): _ZTVN2mb2VBE
 <1><320>: Abbrev Number: 28 (DW_TAG_pointer_type)
    <321>   DW_AT_type        : <0x1ff>
 <1><325>: Abbrev Number: 28 (DW_TAG_pointer_type)
    <326>   DW_AT_type        : <0x55>
 <1><32a>: Abbrev Number: 28 (DW_TAG_pointer_type)
    <32b>   DW_AT_type        : <0x8a>
 <1><32f>: Abbrev Number: 0

"""

# `nm -gp` / `nm -gpC` of the library, paired by position.
const _MR_SYMBOLS = [
    ("_ZN2mb4dsumENS_7DerivedE", "mb::dsum(mb::Derived)"),
    ("_ZN2mb5dmakeEf", "mb::dmake(float)"),
    ("_ZN2mb5pcallENS_1PEii", "mb::pcall(mb::P, int, int)"),
    ("_ZN2mb8vb_firstENS_2VBE", "mb::vb_first(mb::VB)"),
    ("_ZN2mb7vb_makeEf", "mb::vb_make(float)"),
    ("_ZN2mb6kid_idEPNS_7PolyKidE", "mb::kid_id(mb::PolyKid*)"),
    ("_ZN2mb7PolyKidC1Ei", "mb::PolyKid::PolyKid(int)"),
    ("_ZN2mb7PolyKidC2Ei", "mb::PolyKid::PolyKid(int)"),
    ("_ZN2mb7PolyKidD0Ev", "mb::PolyKid::~PolyKid()"),
    ("_ZN2mb4PolyD0Ev", "mb::Poly::~Poly()"),
    ("_ZN2mb4PolyD1Ev", "mb::Poly::~Poly()"),
    ("_ZN2mb4PolyD2Ev", "mb::Poly::~Poly()"),
    ("_ZNK2mb1PclEii", "mb::P::operator()(int, int) const"),
]

# The metadata `extract_compilation_metadata` derives, without a binary: the
# symbol table's functions with the DWARF signatures merged in.
function _mr_metadata(dump)
    rt, sd, gv, td, ra = _MR_C.parse_dwarf_dump(dump)
    syms = [Dict{String,Any}("mangled" => m, "demangled" => d) for (m, d) in _MR_SYMBOLS]
    fns = _MR_C.parse_function_signatures(syms, Set(k for k in keys(sd) if !startswith(k, "__enum__")), Set{String}())
    for f in fns
        m = f["mangled"]
        dk = haskey(rt, m) ? m : _MR_C._structor_dwarf_key(m, rt)
        f["is_noexcept"] = false
        f["is_vararg"] = false
        dk === nothing && continue
        di = rt[dk]
        f["return_type"] = Dict{String,Any}("c_type" => get(di, "c_type", "void"),
                                            "julia_type" => get(di, "julia_type", "Cvoid"),
                                            "size" => get(di, "size", 0))
        haskey(di, "has_object_pointer") && (f["has_object_pointer"] = di["has_object_pointer"])
        haskey(di, "parameters") && !isempty(di["parameters"]) && (f["parameters"] = di["parameters"])
    end
    md = Dict{String,Any}("functions" => fns, "globals" => gv, "struct_definitions" => sd,
                          "record_abi" => ra, "typedef_table" => td)
    return (md, rt, sd, ra, fns)
end

_mr_fn(fns, name) = only(f for f in fns if f["name"] == name)

# From `head` to the first `stop` after it.
function _mr_chunk(code, head, stop)
    i = findfirst(head, code)
    i === nothing && return ""
    rest = SubString(code, first(i))
    j = findfirst(stop, rest)
    return j === nothing ? String(rest) : String(rest[1:first(j) - 1])
end

@testset "A defined library record has an entry, members or not" begin
    (md, rt, sd, ra, fns) = _mr_metadata(MR_DUMP)

    @testset "the parser: an entry with no members and the bases" begin
        @test length(rt) == 11                      # the dump parsed at all
        @test sort(collect(keys(sd))) == ["Derived", "P", "Poly", "PolyKid", "Quad", "VB"]
        checked = 0
        for k in ("Derived", "P", "PolyKid", "VB")
            @test (k, sd[k]["members"]) == (k, Any[])
            @test (k, sd[k]["kind"]) == (k, "struct")
            checked += 1
        end
        @test checked == 4
        @test sd["Derived"]["byte_size"] == "0x10"
        @test [(b["type"], b["offset"], b["virtual"]) for b in sd["Derived"]["base_classes"]] == [("Quad", 0, false)]
        @test sd["P"]["byte_size"] == "0x1"
        @test !haskey(sd["P"], "base_classes")
        @test sd["PolyKid"]["byte_size"] == "0x10"
        @test [(b["type"], b["virtual"]) for b in sd["PolyKid"]["base_classes"]] == [("Poly", false)]
        @test sd["VB"]["byte_size"] == "0x20"
        # Bases are stored sorted by (offset, type): both at 0, so Poly first.
        @test [(b["type"], b["virtual"]) for b in sd["VB"]["base_classes"]] == [("Poly", false), ("Quad", true)]
        # A record with members is as it was.
        @test [m["name"] for m in sd["Quad"]["members"]] == ["m"]
        @test [m["name"] for m in sd["Poly"]["members"]] == ["_vptr\$Poly", "id"]
        # `record_abi` says what it said: emptiness is about bytes a callee reads.
        # (`P` declares `operator()`, which is its `methods` row; set aside.)
        @test delete!(Dict{String,Any}(ra["P"]), "methods") == Dict("byte_size" => "0x1", "pass" => "value", "empty" => true)
        @test ra["Derived"] == Dict("byte_size" => "0x10", "pass" => "value", "empty" => false)
        @test ra["VB"]["pass"] == "reference"
    end

    @testset "the rule, and what it leaves out" begin
        f = _MR_C._memberless_record_entry
        @test f("X", Dict{String,Any}("byte_size" => "0x10"))
        @test !f("X", Dict{String,Any}())                                   # a declaration
        @test !f("X", Dict{String,Any}("byte_size" => "0x0"))               # GNU C's empty struct
        @test !f("X", Dict{String,Any}("byte_size" => "0x4", "anonymous" => true))
        @test !f("(lambda at a.cpp:3:7)", Dict{String,Any}("byte_size" => "0x1"))
    end

    @testset "negative: without the rule the entries are gone" begin
        @eval _MR_C _memberless_record_entry(::AbstractString, ::AbstractDict)::Bool = false
        try
            _, sd0, _, _, _ = Base.invokelatest(_MR_C.parse_dwarf_dump, MR_DUMP)
            @test sort(collect(keys(sd0))) == ["Poly", "Quad"]
        finally
            # The definition in Compiler.jl, restored for the rest of the process.
            @eval _MR_C function _memberless_record_entry(name::AbstractString, type_info::AbstractDict)::Bool
                return something(_definition_byte_size(type_info), 0) > 0 &&
                       get(type_info, "anonymous", false) != true &&
                       !startswith(name, "(lambda")
            end
        end
        @test Base.invokelatest(_MR_C._memberless_record_entry, "X", Dict{String,Any}("byte_size" => "0x1"))
    end

    @testset "by_value_crossing: a layout from the bases, an empty class still dropped" begin
        k(t; abi = RepliBuild.host_record_abi(), ret = false) = _MR_FG.by_value_crossing(t, sd, ra; abi, ret)
        @test k("Derived") == (:layout, "Derived")
        @test k("mb::Derived") == (:layout, "Derived")
        @test k("PolyKid") == (:layout, "PolyKid")
        @test k("VB") == (:layout, "VB")
        @test _MR_FG.record_passes_by_reference("VB", "VB", ra)
        # The convention's answer for an empty record comes before its entry,
        # every convention by name.
        empty = Dict(:sysv => (:ignore, nothing), :win64 => (:regpad, "i8"), :aapcs64 => (:regpad, "i64"))
        for (abi, want) in sort!(collect(empty); by = first)
            @test (abi, k("P"; abi)) == (abi, want)
            @test (abi, k("const P"; abi)) == (abi, want)
            @test (abi, k("P"; abi, ret = true)) == (abi, abi === :win64 ? (:regpad, "i8") : (:ignore, nothing))
        end
        # Metadata with no `record_abi` (from before 2026-10) has no memberless
        # entries either; given one, the layout is all it can say.
        @test _MR_FG.by_value_crossing("P", sd, Dict()) == (:layout, "P")
        for n in ("dsum", "dmake", "pcall", "vb_first"), abi in (:sysv, :win64, :aapcs64)
            @test (n, abi, _MR_FG.thunk_crossings(_mr_fn(fns, n), sd, ra; abi).trap) == (n, abi, nothing)
        end
        # `VB` crosses by address (DW_CC_pass_by_reference) on every convention,
        # so no convention is asked for its class; asked, AAPCS64 cannot read
        # one past the virtual base, and the first arm64 CI run trapped it.
        @test occursin("virtual base", something(_MR_FG.sysv_trap_reason("VB", sd; ret = false, abi = :aapcs64), ""))
    end

    @testset "is_ccall_safe sends an empty record to the thunk" begin
        pc = deepcopy(_mr_fn(fns, "pcall"))
        pc["is_noexcept"] = true
        @test !_MR_D.is_ccall_safe(pc, sd; record_abi = ra)
        # The same function with the empty record gone is a plain ccall.
        pc["parameters"] = pc["parameters"][2:end]
        @test _MR_D.is_ccall_safe(pc, sd; record_abi = ra)
    end

    @testset "the thunks: the bases' SysV class, a padded body, the empty one dropped" begin
        gen(f; abi = :sysv) = _MR_FG.generate_function_thunks([f], sd; may_throw = true, record_abi = ra, abi)
        decl(ir, m) = strip(first(l for l in split(ir, '\n') if startswith(l, "func.func private @$m(")))
        ir = gen(_mr_fn(fns, "dsum"))
        @test decl(ir, "_ZN2mb4dsumENS_7DerivedE") ==
              "func.func private @_ZN2mb4dsumENS_7DerivedE(!llvm.struct<\"Derived\", (!llvm.array<16 x i8>)>) -> f32"
        @test occursin("sysv_args = [\"regs:sse,sse\"]", ir)
        ir = gen(_mr_fn(fns, "dmake"))
        @test decl(ir, "_ZN2mb5dmakeEf") == "func.func private @_ZN2mb5dmakeEf(f32) -> !llvm.struct<\"Derived\", (!llvm.array<16 x i8>)>"
        @test occursin("sysv_ret = \"regs:sse,sse\"", ir)
        ir = gen(_mr_fn(fns, "pcall"))
        @test decl(ir, "_ZN2mb5pcallENS_1PEii") == "func.func private @_ZN2mb5pcallENS_1PEii(i32, i32) -> i32"
        ir = gen(_mr_fn(fns, "pcall"); abi = :win64)
        @test decl(ir, "_ZN2mb5pcallENS_1PEii") == "func.func private @_ZN2mb5pcallENS_1PEii(i8, i32, i32) -> i32"
        ir = gen(_mr_fn(fns, "vb_first"))
        @test decl(ir, "_ZN2mb8vb_firstENS_2VBE") == "func.func private @_ZN2mb8vb_firstENS_2VBE(!llvm.ptr) -> i32"
    end

    @testset "the wrapper: fields from the bases, a sized blob, STRUCT_SIZES" begin
        dir = mktempdir()
        write(joinpath(dir, "replibuild.toml"), """
        [project]
        name = "memberless"
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
        lib = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
        res = _MR_W.generate_introspective_module_cpp(cfg, lib, deepcopy(md), "Memberless",
                                                      _MR_W.create_type_registry(cfg), true)
        code = res isa Tuple ? String(first(res)) : String(res)

        bad = Any[]
        walk(x) = x isa Expr && (x.head in (:incomplete, :error) ? push!(bad, x) : foreach(walk, x.args))
        walk(Meta.parseall(code))
        @test isempty(bad)

        @test occursin("\nstruct Derived\n    m::NTuple{4, Cfloat}\nend\n", code)
        @test occursin("\nstruct P\n    _data::NTuple{1, UInt8}\nend\n", code)
        @test occursin("\nstruct PolyKid\n    _vptr_Poly::Ptr{Cvoid}\n    id::Cint\n    _pad_tail::NTuple{4, UInt8}\nend\n", code)
        # The virtual base is nobody's to flatten: Poly's fields, then its bytes.
        @test occursin("\nstruct VB\n    _vptr_Poly::Ptr{Cvoid}\n    id::Cint\n    _pad_tail::NTuple{20, UInt8}\nend\n", code)
        @test !occursin("struct Derived end", code) && !occursin("Placeholder", code)

        sizes = _mr_chunk(code, "const STRUCT_SIZES", ")\n")
        for (n, s) in ((":Derived", 16), (":P", 1), (":PolyKid", 16), (":VB", 32), (":Quad", 16), (":Poly", 16))
            @test (n, occursin("    $n => $s,\n", sizes)) == (n, true)
        end

        # Bound, typed by the emitted record, not trapped.
        @test occursin("function mb_dsum(d::Derived)", code)
        @test occursin("function mb_dmake(x::Cfloat)", code)
        @test occursin("function mb_pcall(p::P, a::Integer, b::Integer)", code)
        @test occursin("function mb_PolyKid_PolyKid(this::Any, v::Integer)", code)
        for f in ("mb_dsum", "mb_dmake", "mb_pcall", "mb_vb_first")
            @test (f, occursin("ABI Safety Trap", _mr_chunk(code, "function $f(", "\nend"))) == (f, false)
        end
    end
end
