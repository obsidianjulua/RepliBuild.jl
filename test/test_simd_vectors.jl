# =============================================================================
# SIMD vectors by value (library-free, CI-safe)
#
# DWARF spells a vector (`__attribute__((vector_size))`, `__m128`) as the array
# of its lanes, marked DW_AT_GNU_vector, and `resolve_type` keeps that spelling
# (`float[4]`). x86-64 SysV classifies a vector by its size, never lane by lane:
# `<1 x float>` is INTEGER, `<2 x int>` SSE, `<1 x double>` MEMORY, `<4 x float>`
# one SSE register (SSE, SSEUP). Until 2026-10-31 the flag stopped at a member's
# alignment and both tiers read every vector as its lanes: `{<4 x float>}` went
# in XMM0 and XMM1 where C uses XMM0 alone, `{<1 x float>}` in XMM0 where C uses
# RDI, a bare vector argument crossed a thunk as `!llvm.array` (one register per
# lane), and a function returning a vector made GeneratorC emit
# `ccall(…, float[4], …)`, which Julia evaluates when the module loads, so the
# whole wrapper failed to load on both hosts.
#
# The parser now records `vector_size` on the member, parameter and return;
# `FunctionGen.sysv_record_abi` classifies a vector member by its size and
# refuses the 16-byte one no call statement states (`:unstated`);
# `thunk_crossings` refuses a bare vector on every host; GeneratorC traps a
# vector crossing ccall would misplace (`sysv_vector_misread`) on SysV and a
# vector return on both conventions. test_sysv_abi_oracle.jl (devtests) checks
# the classes against LLVM's ABI library and clang on records of each size.
#
# The fixture is clang 23.1.4's DWARF for the C unit below
# (`clang -g -O1 -c --target=x86_64-w64-windows-gnu`), dumped with GNU
# `objdump --dwarf=info` (readelf prints the same lines), with the object path
# and DW_AT_comp_dir replaced. The layout of every record is the same on SysV.
#   typedef float v1f __attribute__((vector_size(4)));
#   typedef int v2i __attribute__((vector_size(8)));
#   typedef double v1d __attribute__((vector_size(8)));
#   typedef float v2f __attribute__((vector_size(8)));
#   typedef float v4f __attribute__((vector_size(16)));
#   typedef float v8f __attribute__((vector_size(32)));
#   typedef struct { v1f v; } RV1F;
#   typedef struct { v2i v; } RV2I;
#   typedef struct { v1d v; } RV1D;
#   typedef struct { v2f v; int i; } RV2FI;
#   typedef struct { v4f v; } RV4F;
#   typedef struct { v8f v; } RV8F;
#   typedef struct { v4f lanes[2]; int k; } RArr;
#   typedef struct { RV1F in; } RNest;
#   typedef struct { float x, y; } RPlain;
#   float take_v4f(v4f x) { return x[3]; }
#   v4f make_v4f(float a) { v4f r = {a, a, a, a}; return r; }
#   float take_rv1f(RV1F r) { return r.v[0]; }
#   int take_rv2i(RV2I r) { return r.v[1]; }
#   double take_rv1d(RV1D r) { return r.v[0]; }
#   int take_rv2fi(RV2FI r) { return r.i; }
#   RV4F make_rv4f(float a) { RV4F r = {{a, a, a, a}}; return r; }
#   float take_rv8f(RV8F r) { return r.v[7]; }
#   int take_rarr(RArr r) { return r.k; }
#   RNest make_rnest(float a) { RNest r = {{{a}}}; return r; }
#   float take_rplain(RPlain r) { return r.x; }
# =============================================================================

using Test
using RepliBuild
using Libdl

const _VFG = RepliBuild.JLCSIRGenerator.FunctionGen

const _VEC_DUMP = raw"""
vec.o:     file format pe-x86-64

Contents of the .debug_info section:

  Compilation Unit @ offset 0:
   Length:        0x3d9 (32-bit)
   Version:       4
   Abbrev Offset: 0
   Pointer Size:  8
 <0><b>: Abbrev Number: 1 (DW_TAG_compile_unit)
    <c>   DW_AT_producer    : (indirect string, offset: 0): clang version 23.1.4 (https://github.com/msys2/MINGW-packages 39c2da431174df09576b662535bdaa688a9bf8e0)
    <10>   DW_AT_language    : 29	(C11)
    <12>   DW_AT_name        : (indirect string, offset: 0x68): vec.c
    <16>   DW_AT_stmt_list   : 0
    <1a>   DW_AT_comp_dir    : (indirect string, offset: 0x6e): /src
    <1e>   DW_AT_low_pc      : 0
    <26>   DW_AT_high_pc     : 0xa5
 <1><2a>: Abbrev Number: 2 (DW_TAG_subprogram)
    <2b>   DW_AT_low_pc      : 0
    <33>   DW_AT_high_pc     : 0x8
    <37>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <39>   DW_AT_GNU_all_call_sites: 1
    <39>   DW_AT_name        : (indirect string, offset: 0x11b): take_v4f
    <3d>   DW_AT_decl_file   : 1
    <3e>   DW_AT_decl_line   : 16
    <3f>   DW_AT_prototyped  : 1
    <3f>   DW_AT_type        : <0x20a>
    <43>   DW_AT_external    : 1
 <2><43>: Abbrev Number: 3 (DW_TAG_formal_parameter)
    <44>   DW_AT_location    : 0 (location list)
    <48>   DW_AT_name        : (indirect string, offset: 0x1cd): x
    <4c>   DW_AT_decl_file   : 1
    <4d>   DW_AT_decl_line   : 16
    <4e>   DW_AT_type        : <0x211>
 <2><52>: Abbrev Number: 0
 <1><53>: Abbrev Number: 2 (DW_TAG_subprogram)
    <54>   DW_AT_low_pc      : 0x10
    <5c>   DW_AT_high_pc     : 0x5
    <60>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <62>   DW_AT_GNU_all_call_sites: 1
    <62>   DW_AT_name        : (indirect string, offset: 0x12a): make_v4f
    <66>   DW_AT_decl_file   : 1
    <67>   DW_AT_decl_line   : 17
    <68>   DW_AT_prototyped  : 1
    <68>   DW_AT_type        : <0x211>
    <6c>   DW_AT_external    : 1
 <2><6c>: Abbrev Number: 3 (DW_TAG_formal_parameter)
    <6d>   DW_AT_location    : 0x23 (location list)
    <71>   DW_AT_name        : (indirect string, offset: 0x1cf): a
    <75>   DW_AT_decl_file   : 1
    <76>   DW_AT_decl_line   : 17
    <77>   DW_AT_type        : <0x20a>
 <2><7b>: Abbrev Number: 4 (DW_TAG_variable)
    <7c>   DW_AT_location    : 0x5c (location list)
    <80>   DW_AT_name        : (indirect string, offset: 0x1d1): r
    <84>   DW_AT_decl_file   : 1
    <85>   DW_AT_decl_line   : 17
    <86>   DW_AT_type        : <0x211>
 <2><8a>: Abbrev Number: 0
 <1><8b>: Abbrev Number: 2 (DW_TAG_subprogram)
    <8c>   DW_AT_low_pc      : 0x20
    <94>   DW_AT_high_pc     : 0x5
    <98>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <9a>   DW_AT_GNU_all_call_sites: 1
    <9a>   DW_AT_name        : (indirect string, offset: 0x14b): take_rv1f
    <9e>   DW_AT_decl_file   : 1
    <9f>   DW_AT_decl_line   : 18
    <a0>   DW_AT_prototyped  : 1
    <a0>   DW_AT_type        : <0x20a>
    <a4>   DW_AT_external    : 1
 <2><a4>: Abbrev Number: 5 (DW_TAG_formal_parameter)
    <a5>   DW_AT_location    : 1 byte block: 52 	(DW_OP_reg2 (rcx))
    <a7>   DW_AT_name        : (indirect string, offset: 0x1d1): r
    <ab>   DW_AT_decl_file   : 1
    <ac>   DW_AT_decl_line   : 18
    <ad>   DW_AT_type        : <0x275>
 <2><b1>: Abbrev Number: 0
 <1><b2>: Abbrev Number: 2 (DW_TAG_subprogram)
    <b3>   DW_AT_low_pc      : 0x30
    <bb>   DW_AT_high_pc     : 0x8
    <bf>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <c1>   DW_AT_GNU_all_call_sites: 1
    <c1>   DW_AT_name        : (indirect string, offset: 0x155): take_rv2i
    <c5>   DW_AT_decl_file   : 1
    <c6>   DW_AT_decl_line   : 19
    <c7>   DW_AT_prototyped  : 1
    <c7>   DW_AT_type        : <0x22f>
    <cb>   DW_AT_external    : 1
 <2><cb>: Abbrev Number: 5 (DW_TAG_formal_parameter)
    <cc>   DW_AT_location    : 1 byte block: 52 	(DW_OP_reg2 (rcx))
    <ce>   DW_AT_name        : (indirect string, offset: 0x1d1): r
    <d2>   DW_AT_decl_file   : 1
    <d3>   DW_AT_decl_line   : 19
    <d4>   DW_AT_type        : <0x2a8>
 <2><d8>: Abbrev Number: 0
 <1><d9>: Abbrev Number: 2 (DW_TAG_subprogram)
    <da>   DW_AT_low_pc      : 0x40
    <e2>   DW_AT_high_pc     : 0x6
    <e6>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <e8>   DW_AT_GNU_all_call_sites: 1
    <e8>   DW_AT_name        : (indirect string, offset: 0x163): take_rv1d
    <ec>   DW_AT_decl_file   : 1
    <ed>   DW_AT_decl_line   : 20
    <ee>   DW_AT_prototyped  : 1
    <ee>   DW_AT_type        : <0x236>
    <f2>   DW_AT_external    : 1
 <2><f2>: Abbrev Number: 5 (DW_TAG_formal_parameter)
    <f3>   DW_AT_location    : 1 byte block: 52 	(DW_OP_reg2 (rcx))
    <f5>   DW_AT_name        : (indirect string, offset: 0x1d1): r
    <f9>   DW_AT_decl_file   : 1
    <fa>   DW_AT_decl_line   : 20
    <fb>   DW_AT_type        : <0x2db>
 <2><ff>: Abbrev Number: 0
 <1><100>: Abbrev Number: 2 (DW_TAG_subprogram)
    <101>   DW_AT_low_pc      : 0x50
    <109>   DW_AT_high_pc     : 0x4
    <10d>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <10f>   DW_AT_GNU_all_call_sites: 1
    <10f>   DW_AT_name        : (indirect string, offset: 0x174): take_rv2fi
    <113>   DW_AT_decl_file   : 1
    <114>   DW_AT_decl_line   : 21
    <115>   DW_AT_prototyped  : 1
    <115>   DW_AT_type        : <0x22f>
    <119>   DW_AT_external    : 1
 <2><119>: Abbrev Number: 5 (DW_TAG_formal_parameter)
    <11a>   DW_AT_location    : 2 byte block: 72 0 	(DW_OP_breg2 (rcx): 0)
    <11d>   DW_AT_name        : (indirect string, offset: 0x1d1): r
    <121>   DW_AT_decl_file   : 1
    <122>   DW_AT_decl_line   : 21
    <123>   DW_AT_type        : <0x30e>
 <2><127>: Abbrev Number: 0
 <1><128>: Abbrev Number: 2 (DW_TAG_subprogram)
    <129>   DW_AT_low_pc      : 0x60
    <131>   DW_AT_high_pc     : 0xb
    <135>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <137>   DW_AT_GNU_all_call_sites: 1
    <137>   DW_AT_name        : (indirect string, offset: 0x17f): make_rv4f
    <13b>   DW_AT_decl_file   : 1
    <13c>   DW_AT_decl_line   : 22
    <13d>   DW_AT_prototyped  : 1
    <13d>   DW_AT_type        : <0x23d>
    <141>   DW_AT_external    : 1
 <2><141>: Abbrev Number: 3 (DW_TAG_formal_parameter)
    <142>   DW_AT_location    : 0x7f (location list)
    <146>   DW_AT_name        : (indirect string, offset: 0x1cf): a
    <14a>   DW_AT_decl_file   : 1
    <14b>   DW_AT_decl_line   : 22
    <14c>   DW_AT_type        : <0x20a>
 <2><150>: Abbrev Number: 6 (DW_TAG_variable)
    <151>   DW_AT_location    : 2 byte block: 72 0 	(DW_OP_breg2 (rcx): 0)
    <154>   DW_AT_name        : (indirect string, offset: 0x1d1): r
    <158>   DW_AT_decl_file   : 1
    <159>   DW_AT_decl_line   : 22
    <15a>   DW_AT_type        : <0x23d>
 <2><15e>: Abbrev Number: 0
 <1><15f>: Abbrev Number: 2 (DW_TAG_subprogram)
    <160>   DW_AT_low_pc      : 0x70
    <168>   DW_AT_high_pc     : 0x6
    <16c>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <16e>   DW_AT_GNU_all_call_sites: 1
    <16e>   DW_AT_name        : (indirect string, offset: 0x190): take_rv8f
    <172>   DW_AT_decl_file   : 1
    <173>   DW_AT_decl_line   : 23
    <174>   DW_AT_prototyped  : 1
    <174>   DW_AT_type        : <0x20a>
    <178>   DW_AT_external    : 1
 <2><178>: Abbrev Number: 5 (DW_TAG_formal_parameter)
    <179>   DW_AT_location    : 2 byte block: 72 0 	(DW_OP_breg2 (rcx): 0)
    <17c>   DW_AT_name        : (indirect string, offset: 0x1d1): r
    <180>   DW_AT_decl_file   : 1
    <181>   DW_AT_decl_line   : 23
    <182>   DW_AT_type        : <0x34d>
 <2><186>: Abbrev Number: 0
 <1><187>: Abbrev Number: 2 (DW_TAG_subprogram)
    <188>   DW_AT_low_pc      : 0x80
    <190>   DW_AT_high_pc     : 0x4
    <194>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <196>   DW_AT_GNU_all_call_sites: 1
    <196>   DW_AT_name        : (indirect string, offset: 0x19a): take_rarr
    <19a>   DW_AT_decl_file   : 1
    <19b>   DW_AT_decl_line   : 24
    <19c>   DW_AT_prototyped  : 1
    <19c>   DW_AT_type        : <0x22f>
    <1a0>   DW_AT_external    : 1
 <2><1a0>: Abbrev Number: 5 (DW_TAG_formal_parameter)
    <1a1>   DW_AT_location    : 2 byte block: 72 0 	(DW_OP_breg2 (rcx): 0)
    <1a4>   DW_AT_name        : (indirect string, offset: 0x1d1): r
    <1a8>   DW_AT_decl_file   : 1
    <1a9>   DW_AT_decl_line   : 24
    <1aa>   DW_AT_type        : <0x380>
 <2><1ae>: Abbrev Number: 0
 <1><1af>: Abbrev Number: 2 (DW_TAG_subprogram)
    <1b0>   DW_AT_low_pc      : 0x90
    <1b8>   DW_AT_high_pc     : 0x5
    <1bc>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <1be>   DW_AT_GNU_all_call_sites: 1
    <1be>   DW_AT_name        : (indirect string, offset: 0x1a4): make_rnest
    <1c2>   DW_AT_decl_file   : 1
    <1c3>   DW_AT_decl_line   : 25
    <1c4>   DW_AT_prototyped  : 1
    <1c4>   DW_AT_type        : <0x259>
    <1c8>   DW_AT_external    : 1
 <2><1c8>: Abbrev Number: 5 (DW_TAG_formal_parameter)
    <1c9>   DW_AT_location    : 1 byte block: 61 	(DW_OP_reg17 (xmm0))
    <1cb>   DW_AT_name        : (indirect string, offset: 0x1cf): a
    <1cf>   DW_AT_decl_file   : 1
    <1d0>   DW_AT_decl_line   : 25
    <1d1>   DW_AT_type        : <0x20a>
 <2><1d5>: Abbrev Number: 7 (DW_TAG_variable)
    <1d6>   DW_AT_name        : (indirect string, offset: 0x1d1): r
    <1da>   DW_AT_decl_file   : 1
    <1db>   DW_AT_decl_line   : 25
    <1dc>   DW_AT_type        : <0x259>
 <2><1e0>: Abbrev Number: 0
 <1><1e1>: Abbrev Number: 2 (DW_TAG_subprogram)
    <1e2>   DW_AT_low_pc      : 0xa0
    <1ea>   DW_AT_high_pc     : 0x5
    <1ee>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <1f0>   DW_AT_GNU_all_call_sites: 1
    <1f0>   DW_AT_name        : (indirect string, offset: 0x1c1): take_rplain
    <1f4>   DW_AT_decl_file   : 1
    <1f5>   DW_AT_decl_line   : 26
    <1f6>   DW_AT_prototyped  : 1
    <1f6>   DW_AT_type        : <0x20a>
    <1fa>   DW_AT_external    : 1
 <2><1fa>: Abbrev Number: 3 (DW_TAG_formal_parameter)
    <1fb>   DW_AT_location    : 0xb8 (location list)
    <1ff>   DW_AT_name        : (indirect string, offset: 0x1d1): r
    <203>   DW_AT_decl_file   : 1
    <204>   DW_AT_decl_line   : 26
    <205>   DW_AT_type        : <0x3b4>
 <2><209>: Abbrev Number: 0
 <1><20a>: Abbrev Number: 8 (DW_TAG_base_type)
    <20b>   DW_AT_name        : (indirect string, offset: 0x124): float
    <20f>   DW_AT_encoding    : 4	(float)
    <210>   DW_AT_byte_size   : 4
 <1><211>: Abbrev Number: 9 (DW_TAG_typedef)
    <212>   DW_AT_type        : <0x21c>
    <216>   DW_AT_name        : (indirect string, offset: 0x147): v4f
    <21a>   DW_AT_decl_file   : 1
    <21b>   DW_AT_decl_line   : 5
 <1><21c>: Abbrev Number: 10 (DW_TAG_array_type)
    <21d>   DW_AT_GNU_vector  : 1
    <21d>   DW_AT_type        : <0x20a>
 <2><221>: Abbrev Number: 11 (DW_TAG_subrange_type)
    <222>   DW_AT_type        : <0x228>
    <226>   DW_AT_count       : 4
 <2><227>: Abbrev Number: 0
 <1><228>: Abbrev Number: 12 (DW_TAG_base_type)
    <229>   DW_AT_name        : (indirect string, offset: 0x133): __ARRAY_SIZE_TYPE__
    <22d>   DW_AT_byte_size   : 8
    <22e>   DW_AT_encoding    : 7	(unsigned)
 <1><22f>: Abbrev Number: 8 (DW_TAG_base_type)
    <230>   DW_AT_name        : (indirect string, offset: 0x15f): int
    <234>   DW_AT_encoding    : 5	(signed)
    <235>   DW_AT_byte_size   : 4
 <1><236>: Abbrev Number: 8 (DW_TAG_base_type)
    <237>   DW_AT_name        : (indirect string, offset: 0x16d): double
    <23b>   DW_AT_encoding    : 4	(float)
    <23c>   DW_AT_byte_size   : 8
 <1><23d>: Abbrev Number: 9 (DW_TAG_typedef)
    <23e>   DW_AT_type        : <0x248>
    <242>   DW_AT_name        : (indirect string, offset: 0x18b): RV4F
    <246>   DW_AT_decl_file   : 1
    <247>   DW_AT_decl_line   : 11
 <1><248>: Abbrev Number: 13 (DW_TAG_structure_type)
    <249>   DW_AT_byte_size   : 16
    <24a>   DW_AT_decl_file   : 1
    <24b>   DW_AT_decl_line   : 11
 <2><24c>: Abbrev Number: 14 (DW_TAG_member)
    <24d>   DW_AT_name        : (indirect string, offset: 0x189): v
    <251>   DW_AT_type        : <0x211>
    <255>   DW_AT_decl_file   : 1
    <256>   DW_AT_decl_line   : 11
    <257>   DW_AT_data_member_location: 0
 <2><258>: Abbrev Number: 0
 <1><259>: Abbrev Number: 9 (DW_TAG_typedef)
    <25a>   DW_AT_type        : <0x264>
    <25e>   DW_AT_name        : (indirect string, offset: 0x1bb): RNest
    <262>   DW_AT_decl_file   : 1
    <263>   DW_AT_decl_line   : 14
 <1><264>: Abbrev Number: 13 (DW_TAG_structure_type)
    <265>   DW_AT_byte_size   : 4
    <266>   DW_AT_decl_file   : 1
    <267>   DW_AT_decl_line   : 14
 <2><268>: Abbrev Number: 14 (DW_TAG_member)
    <269>   DW_AT_name        : (indirect string, offset: 0x1af): in
    <26d>   DW_AT_type        : <0x275>
    <271>   DW_AT_decl_file   : 1
    <272>   DW_AT_decl_line   : 14
    <273>   DW_AT_data_member_location: 0
 <2><274>: Abbrev Number: 0
 <1><275>: Abbrev Number: 9 (DW_TAG_typedef)
    <276>   DW_AT_type        : <0x280>
    <27a>   DW_AT_name        : (indirect string, offset: 0x1b6): RV1F
    <27e>   DW_AT_decl_file   : 1
    <27f>   DW_AT_decl_line   : 7
 <1><280>: Abbrev Number: 13 (DW_TAG_structure_type)
    <281>   DW_AT_byte_size   : 4
    <282>   DW_AT_decl_file   : 1
    <283>   DW_AT_decl_line   : 7
 <2><284>: Abbrev Number: 14 (DW_TAG_member)
    <285>   DW_AT_name        : (indirect string, offset: 0x189): v
    <289>   DW_AT_type        : <0x291>
    <28d>   DW_AT_decl_file   : 1
    <28e>   DW_AT_decl_line   : 7
    <28f>   DW_AT_data_member_location: 0
 <2><290>: Abbrev Number: 0
 <1><291>: Abbrev Number: 9 (DW_TAG_typedef)
    <292>   DW_AT_type        : <0x29c>
    <296>   DW_AT_name        : (indirect string, offset: 0x1b2): v1f
    <29a>   DW_AT_decl_file   : 1
    <29b>   DW_AT_decl_line   : 1
 <1><29c>: Abbrev Number: 10 (DW_TAG_array_type)
    <29d>   DW_AT_GNU_vector  : 1
    <29d>   DW_AT_type        : <0x20a>
 <2><2a1>: Abbrev Number: 11 (DW_TAG_subrange_type)
    <2a2>   DW_AT_type        : <0x228>
    <2a6>   DW_AT_count       : 1
 <2><2a7>: Abbrev Number: 0
 <1><2a8>: Abbrev Number: 9 (DW_TAG_typedef)
    <2a9>   DW_AT_type        : <0x2b3>
    <2ad>   DW_AT_name        : (indirect string, offset: 0x1d7): RV2I
    <2b1>   DW_AT_decl_file   : 1
    <2b2>   DW_AT_decl_line   : 8
 <1><2b3>: Abbrev Number: 13 (DW_TAG_structure_type)
    <2b4>   DW_AT_byte_size   : 8
    <2b5>   DW_AT_decl_file   : 1
    <2b6>   DW_AT_decl_line   : 8
 <2><2b7>: Abbrev Number: 14 (DW_TAG_member)
    <2b8>   DW_AT_name        : (indirect string, offset: 0x189): v
    <2bc>   DW_AT_type        : <0x2c4>
    <2c0>   DW_AT_decl_file   : 1
    <2c1>   DW_AT_decl_line   : 8
    <2c2>   DW_AT_data_member_location: 0
 <2><2c3>: Abbrev Number: 0
 <1><2c4>: Abbrev Number: 9 (DW_TAG_typedef)
    <2c5>   DW_AT_type        : <0x2cf>
    <2c9>   DW_AT_name        : (indirect string, offset: 0x1d3): v2i
    <2cd>   DW_AT_decl_file   : 1
    <2ce>   DW_AT_decl_line   : 2
 <1><2cf>: Abbrev Number: 10 (DW_TAG_array_type)
    <2d0>   DW_AT_GNU_vector  : 1
    <2d0>   DW_AT_type        : <0x22f>
 <2><2d4>: Abbrev Number: 11 (DW_TAG_subrange_type)
    <2d5>   DW_AT_type        : <0x228>
    <2d9>   DW_AT_count       : 2
 <2><2da>: Abbrev Number: 0
 <1><2db>: Abbrev Number: 9 (DW_TAG_typedef)
    <2dc>   DW_AT_type        : <0x2e6>
    <2e0>   DW_AT_name        : (indirect string, offset: 0x1e0): RV1D
    <2e4>   DW_AT_decl_file   : 1
    <2e5>   DW_AT_decl_line   : 9
 <1><2e6>: Abbrev Number: 13 (DW_TAG_structure_type)
    <2e7>   DW_AT_byte_size   : 8
    <2e8>   DW_AT_decl_file   : 1
    <2e9>   DW_AT_decl_line   : 9
 <2><2ea>: Abbrev Number: 14 (DW_TAG_member)
    <2eb>   DW_AT_name        : (indirect string, offset: 0x189): v
    <2ef>   DW_AT_type        : <0x2f7>
    <2f3>   DW_AT_decl_file   : 1
    <2f4>   DW_AT_decl_line   : 9
    <2f5>   DW_AT_data_member_location: 0
 <2><2f6>: Abbrev Number: 0
 <1><2f7>: Abbrev Number: 9 (DW_TAG_typedef)
    <2f8>   DW_AT_type        : <0x302>
    <2fc>   DW_AT_name        : (indirect string, offset: 0x1dc): v1d
    <300>   DW_AT_decl_file   : 1
    <301>   DW_AT_decl_line   : 3
 <1><302>: Abbrev Number: 10 (DW_TAG_array_type)
    <303>   DW_AT_GNU_vector  : 1
    <303>   DW_AT_type        : <0x236>
 <2><307>: Abbrev Number: 11 (DW_TAG_subrange_type)
    <308>   DW_AT_type        : <0x228>
    <30c>   DW_AT_count       : 1
 <2><30d>: Abbrev Number: 0
 <1><30e>: Abbrev Number: 9 (DW_TAG_typedef)
    <30f>   DW_AT_type        : <0x319>
    <313>   DW_AT_name        : (indirect string, offset: 0x1eb): RV2FI
    <317>   DW_AT_decl_file   : 1
    <318>   DW_AT_decl_line   : 10
 <1><319>: Abbrev Number: 13 (DW_TAG_structure_type)
    <31a>   DW_AT_byte_size   : 16
    <31b>   DW_AT_decl_file   : 1
    <31c>   DW_AT_decl_line   : 10
 <2><31d>: Abbrev Number: 14 (DW_TAG_member)
    <31e>   DW_AT_name        : (indirect string, offset: 0x189): v
    <322>   DW_AT_type        : <0x336>
    <326>   DW_AT_decl_file   : 1
    <327>   DW_AT_decl_line   : 10
    <328>   DW_AT_data_member_location: 0
 <2><329>: Abbrev Number: 14 (DW_TAG_member)
    <32a>   DW_AT_name        : (indirect string, offset: 0x1e9): i
    <32e>   DW_AT_type        : <0x22f>
    <332>   DW_AT_decl_file   : 1
    <333>   DW_AT_decl_line   : 10
    <334>   DW_AT_data_member_location: 8
 <2><335>: Abbrev Number: 0
 <1><336>: Abbrev Number: 9 (DW_TAG_typedef)
    <337>   DW_AT_type        : <0x341>
    <33b>   DW_AT_name        : (indirect string, offset: 0x1e5): v2f
    <33f>   DW_AT_decl_file   : 1
    <340>   DW_AT_decl_line   : 4
 <1><341>: Abbrev Number: 10 (DW_TAG_array_type)
    <342>   DW_AT_GNU_vector  : 1
    <342>   DW_AT_type        : <0x20a>
 <2><346>: Abbrev Number: 11 (DW_TAG_subrange_type)
    <347>   DW_AT_type        : <0x228>
    <34b>   DW_AT_count       : 2
 <2><34c>: Abbrev Number: 0
 <1><34d>: Abbrev Number: 9 (DW_TAG_typedef)
    <34e>   DW_AT_type        : <0x358>
    <352>   DW_AT_name        : (indirect string, offset: 0x1f5): RV8F
    <356>   DW_AT_decl_file   : 1
    <357>   DW_AT_decl_line   : 12
 <1><358>: Abbrev Number: 13 (DW_TAG_structure_type)
    <359>   DW_AT_byte_size   : 32
    <35a>   DW_AT_decl_file   : 1
    <35b>   DW_AT_decl_line   : 12
 <2><35c>: Abbrev Number: 14 (DW_TAG_member)
    <35d>   DW_AT_name        : (indirect string, offset: 0x189): v
    <361>   DW_AT_type        : <0x369>
    <365>   DW_AT_decl_file   : 1
    <366>   DW_AT_decl_line   : 12
    <367>   DW_AT_data_member_location: 0
 <2><368>: Abbrev Number: 0
 <1><369>: Abbrev Number: 9 (DW_TAG_typedef)
    <36a>   DW_AT_type        : <0x374>
    <36e>   DW_AT_name        : (indirect string, offset: 0x1f1): v8f
    <372>   DW_AT_decl_file   : 1
    <373>   DW_AT_decl_line   : 6
 <1><374>: Abbrev Number: 10 (DW_TAG_array_type)
    <375>   DW_AT_GNU_vector  : 1
    <375>   DW_AT_type        : <0x20a>
 <2><379>: Abbrev Number: 11 (DW_TAG_subrange_type)
    <37a>   DW_AT_type        : <0x228>
    <37e>   DW_AT_count       : 8
 <2><37f>: Abbrev Number: 0
 <1><380>: Abbrev Number: 9 (DW_TAG_typedef)
    <381>   DW_AT_type        : <0x38b>
    <385>   DW_AT_name        : (indirect string, offset: 0x202): RArr
    <389>   DW_AT_decl_file   : 1
    <38a>   DW_AT_decl_line   : 13
 <1><38b>: Abbrev Number: 13 (DW_TAG_structure_type)
    <38c>   DW_AT_byte_size   : 48
    <38d>   DW_AT_decl_file   : 1
    <38e>   DW_AT_decl_line   : 13
 <2><38f>: Abbrev Number: 14 (DW_TAG_member)
    <390>   DW_AT_name        : (indirect string, offset: 0x1fa): lanes
    <394>   DW_AT_type        : <0x3a8>
    <398>   DW_AT_decl_file   : 1
    <399>   DW_AT_decl_line   : 13
    <39a>   DW_AT_data_member_location: 0
 <2><39b>: Abbrev Number: 14 (DW_TAG_member)
    <39c>   DW_AT_name        : (indirect string, offset: 0x200): k
    <3a0>   DW_AT_type        : <0x22f>
    <3a4>   DW_AT_decl_file   : 1
    <3a5>   DW_AT_decl_line   : 13
    <3a6>   DW_AT_data_member_location: 32
 <2><3a7>: Abbrev Number: 0
 <1><3a8>: Abbrev Number: 15 (DW_TAG_array_type)
    <3a9>   DW_AT_type        : <0x211>
 <2><3ad>: Abbrev Number: 11 (DW_TAG_subrange_type)
    <3ae>   DW_AT_type        : <0x228>
    <3b2>   DW_AT_count       : 2
 <2><3b3>: Abbrev Number: 0
 <1><3b4>: Abbrev Number: 9 (DW_TAG_typedef)
    <3b5>   DW_AT_type        : <0x3bf>
    <3b9>   DW_AT_name        : (indirect string, offset: 0x209): RPlain
    <3bd>   DW_AT_decl_file   : 1
    <3be>   DW_AT_decl_line   : 15
 <1><3bf>: Abbrev Number: 13 (DW_TAG_structure_type)
    <3c0>   DW_AT_byte_size   : 8
    <3c1>   DW_AT_decl_file   : 1
    <3c2>   DW_AT_decl_line   : 15
 <2><3c3>: Abbrev Number: 14 (DW_TAG_member)
    <3c4>   DW_AT_name        : (indirect string, offset: 0x1cd): x
    <3c8>   DW_AT_type        : <0x20a>
    <3cc>   DW_AT_decl_file   : 1
    <3cd>   DW_AT_decl_line   : 15
    <3ce>   DW_AT_data_member_location: 0
 <2><3cf>: Abbrev Number: 14 (DW_TAG_member)
    <3d0>   DW_AT_name        : (indirect string, offset: 0x207): y
    <3d4>   DW_AT_type        : <0x20a>
    <3d8>   DW_AT_decl_file   : 1
    <3d9>   DW_AT_decl_line   : 15
    <3da>   DW_AT_data_member_location: 4
 <2><3db>: Abbrev Number: 0
 <1><3dc>: Abbrev Number: 0

"""

const _VRT, _VSD, _, _, _ = RepliBuild.Compiler.parse_dwarf_dump(_VEC_DUMP)

# The metadata `functions` entry a build writes for a fixture function: the
# return type and parameters the parser read, merged the way
# `Compiler.extract_compilation_metadata` merges them (`vector_size` included).
function _vfunc(name; byval = nothing)
    r = _VRT[name]
    ret = Dict{String,Any}("c_type" => r["c_type"], "julia_type" => r["julia_type"], "size" => r["size"])
    haskey(r, "vector_size") && (ret["vector_size"] = r["vector_size"])
    f = Dict{String,Any}("name" => name, "mangled" => name, "demangled" => name, "is_vararg" => false,
                         "parameters" => Any[copy(p) for p in r["parameters"]], "return_type" => ret)
    byval === nothing ||
        (f["byval_params"] = Any[Dict{String,Any}("record" => n, "align" => a) for (n, a) in byval])
    return f
end

_vmember(rec, name) = only(filter(m -> m["name"] == name, _VSD[rec]["members"]))

@testset "SIMD vectors by value" begin

    @testset "the parser records vector_size" begin
        # A member: the vector, its lanes as the c_type, its alignment its size.
        v = _vmember("RV4F", "v")
        @test (v["c_type"], v["size"], v["vector_size"], v["alignment"]) == ("float[4]", 16, 16, 16)
        @test _vmember("RV1F", "v")["vector_size"] == 4
        @test _vmember("RV1D", "v")["vector_size"] == 8
        # An array of vectors holds vectors: each is 16 bytes and 16-aligned,
        # which puts the record's byval slot at 16 (clang: `align 16`).
        lanes = _vmember("RArr", "lanes")
        @test (lanes["size"], lanes["vector_size"], lanes["alignment"]) == (32, 16, 16)
        # Not a vector: plain floats, a record member, a record parameter.
        @test !haskey(_vmember("RPlain", "x"), "vector_size")
        @test !haskey(_vmember("RNest", "in"), "vector_size")
        @test !haskey(only(_VRT["take_rv1f"]["parameters"]), "vector_size")
        # A bare vector parameter and return.
        @test only(_VRT["take_v4f"]["parameters"])["vector_size"] == 16
        @test _VRT["make_v4f"]["vector_size"] == 16
        @test !haskey(_VRT["take_v4f"], "vector_size")
    end

    @testset "SysV classifies a vector by its size" begin
        # record => (C's statement, or its kind when it has none; the lanes' classes)
        want = Dict(
            "RV1F"   => ("regs:int",     [:sse]),        # <1 x float>: INTEGER, as gcc passes it
            "RV2I"   => ("regs:sse",     [:int]),        # <2 x int>: SSE
            "RV1D"   => ("mem:8",        [:sse]),        # <1 x double>: MEMORY
            "RV2FI"  => ("regs:sse,int", [:sse, :int]),  # <2 x float>: SSE either way
            "RNest"  => ("regs:int",     [:sse]),        # a nested record's vector
            "RPlain" => ("regs:sse",     [:sse]),        # no vector at all
            "RArr"   => ("mem:16",       Symbol[]),      # 48 bytes, MEMORY in a 16-aligned slot
            "RV4F"   => (:unstated,      [:sse, :sse]),  # one SSE register (SSE, SSEUP)
            "RV8F"   => (:unknown,       Symbol[]))      # YMM when built for AVX, else memory
        checked = 0
        for (rec, (c, lanes)) in want
            v = _VFG.sysv_record_abi(rec, _VSD)
            got = c isa Symbol ? v.kind : _VFG.sysv_statement(rec, _VSD)
            @test (rec, got) == (rec, c)
            @test (rec, _VFG.sysv_record_abi(rec, _VSD; vectors = false).classes) == (rec, lanes)
            checked += 1
        end
        @test checked == length(want)

        # A vector no statement states is refused on SysV, with the reason, and
        # never on Win64, which passes a record by size.
        @test _VFG.sysv_statement("RV4F", _VSD) == ""
        why = _VFG.sysv_trap_reason("RV4F", _VSD; ret = false, abi = :sysv)
        @test why !== nothing && occursin("16-byte SIMD vector", why)
        @test occursin("AVX", _VFG.sysv_trap_reason("RV8F", _VSD; ret = true, abi = :sysv))
        @test _VFG.sysv_trap_reason("RV4F", _VSD; ret = true, abi = :win64) === nothing
        @test _VFG.sysv_trap_reason("RV1F", _VSD; ret = false, abi = :sysv) === nothing

        # Where ccall's lane-by-lane reading of the NTuple stand-in differs
        # from C's: these, and no other record.
        misread = sort!([k for k in keys(_VSD) if _VFG.sysv_vector_misread(k, _VSD) !== nothing])
        @test misread == ["RNest", "RV1D", "RV1F", "RV2I", "RV4F", "RV8F"]
        @test occursin("in registers (INTEGER)", _VFG.sysv_vector_misread("RV1F", _VSD))
    end

    @testset "a bare vector never crosses a thunk" begin
        rabi = Dict{String,Any}()
        for name in ("take_v4f", "make_v4f")
            c = _VFG.thunk_crossings(_vfunc(name), _VSD, rabi)
            @test (name, c.abi_trap !== nothing && occursin("SIMD vector", c.abi_trap)) == (name, true)
            @test c.trap !== nothing
        end
        # A record holding a vector is the convention's question, not this one.
        @test _VFG.thunk_crossings(_vfunc("take_rplain"), _VSD, rabi).trap === nothing
        # And the thunk generator emits no thunk for either.
        meta = Dict{String,Any}("language" => "c", "struct_definitions" => _VSD, "record_abi" => rabi,
                                "functions" => Any[_vfunc("take_v4f"), _vfunc("make_v4f"), _vfunc("take_rplain")])
        vt = RepliBuild.DWARFParser.VtableInfo(Dict{String,RepliBuild.DWARFParser.ClassInfo}(),
                                               Dict{String,UInt64}(), Dict{String,UInt64}())
        ir = RepliBuild.JLCSIRGenerator.generate_jlcs_ir(vt, meta)
        @test occursin("// no thunk for take_v4f:", ir)
        @test occursin("// no thunk for make_v4f:", ir)
        @test !occursin("@take_v4f(", ir) && !occursin("@make_v4f(", ir)   # nor a callee declaration
        @test occursin("@take_rplain_thunk", ir)
    end

    @testset "GeneratorC traps what ccall would misplace" begin
        dir = mktempdir()
        write(joinpath(dir, "replibuild.toml"), """
        [project]
        name = "vec"
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
        W = RepliBuild.Wrapper
        names = ["take_v4f", "make_v4f", "take_rv1f", "take_rv2i", "take_rv1d", "take_rv2fi",
                 "make_rv4f", "take_rv8f", "take_rarr", "make_rnest", "take_rplain"]
        # The compiler's slot for RArr (16-aligned), so SysV passes it in a carrier.
        fns = Any[n == "take_rarr" ? _vfunc(n; byval = [("RArr", 16)]) : _vfunc(n) for n in names]
        meta = Dict{String,Any}("functions" => fns, "struct_definitions" => _VSD, "globals" => Dict{String,Any}())
        libref = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
        gen(abi) = W.generate_introspective_module_c(cfg, libref, meta, "Vec", W.create_type_registry(cfg), false; abi)
        body(code, name) = let i = findfirst("function $name(", code)
            i === nothing ? "" : (r = SubString(code, first(i)); j = findfirst(r"\nend\b", r);
                                  j === nothing ? String(r) : String(r[1:last(j)]))
        end
        form(code, name) = let b = body(code, name)
            isempty(b) ? :missing :
            occursin("SIMD vector", b) && occursin("ABI Safety Trap", b) ? :vector :
            occursin("ABI Safety Trap", b) ? :trap :
            occursin("_rb_byval16(", b) ? :carrier :
            occursin("ccall(", b) ? :ccall : :other
        end
        sysv, win64 = gen(:sysv), gen(:win64)
        #         function        SysV      Win64
        table = [("take_v4f",    :vector,  :ccall),   # one XMM vs two; Win64 passes it by reference either way
                 ("make_v4f",    :vector,  :vector),  # no Julia type: `float[4]` failed the module's load
                 ("take_rv1f",   :vector,  :ccall),   # RDI vs XMM0; Win64 by size
                 ("take_rv2i",   :vector,  :ccall),
                 ("take_rv1d",   :vector,  :ccall),   # memory vs XMM0
                 ("make_rv4f",   :vector,  :ccall),
                 ("take_rv8f",   :vector,  :ccall),
                 ("make_rnest",  :vector,  :ccall),
                 ("take_rv2fi",  :ccall,   :ccall),   # SSE, INTEGER on both readings
                 ("take_rarr",   :carrier, :ccall),   # MEMORY on both readings
                 ("take_rplain", :ccall,   :ccall)]
        checked = 0
        for (fn, s, w) in table
            @test (fn, form(sysv, fn), form(win64, fn)) == (fn, s, w)
            checked += 1
        end
        @test checked == length(names)
        # Both wrappers parse, and no ccall names a C array spelling as a type.
        for code in (sysv, win64)
            @test !any(e -> e isa Expr && e.head in (:incomplete, :error), Meta.parseall(code).args)
            @test !occursin(r"ccall\([^\n]*\bfloat\[", code)
        end
    end
end
