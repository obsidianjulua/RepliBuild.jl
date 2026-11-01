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
#   typedef float al4[4] __attribute__((aligned(16)));
#   typedef struct { al4 a; } RAl;
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
#   float take_ral(RAl r) { return r.a[3]; }
#
# `RAl` is an array aligned to its own size, which is how a vector looked in
# metadata from before `vector_size`. The parser marks it `vector_size: null`,
# and the last testset strips every `vector_size` to stand for that metadata.
# =============================================================================

using Test
using RepliBuild
using Libdl

const _VFG = RepliBuild.JLCSIRGenerator.FunctionGen

const _VEC_DUMP = raw"""
vec.o:     file format pe-x86-64

Contents of the .debug_info section:

  Compilation Unit @ offset 0:
   Length:        0x435 (32-bit)
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
    <26>   DW_AT_high_pc     : 0xb6
 <1><2a>: Abbrev Number: 2 (DW_TAG_subprogram)
    <2b>   DW_AT_low_pc      : 0
    <33>   DW_AT_high_pc     : 0x8
    <37>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <39>   DW_AT_GNU_all_call_sites: 1
    <39>   DW_AT_name        : (indirect string, offset: 0x11b): take_v4f
    <3d>   DW_AT_decl_file   : 1
    <3e>   DW_AT_decl_line   : 18
    <3f>   DW_AT_prototyped  : 1
    <3f>   DW_AT_type        : <0x232>
    <43>   DW_AT_external    : 1
 <2><43>: Abbrev Number: 3 (DW_TAG_formal_parameter)
    <44>   DW_AT_location    : 0 (location list)
    <48>   DW_AT_name        : (indirect string, offset: 0x1d6): x
    <4c>   DW_AT_decl_file   : 1
    <4d>   DW_AT_decl_line   : 18
    <4e>   DW_AT_type        : <0x239>
 <2><52>: Abbrev Number: 0
 <1><53>: Abbrev Number: 2 (DW_TAG_subprogram)
    <54>   DW_AT_low_pc      : 0x10
    <5c>   DW_AT_high_pc     : 0x5
    <60>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <62>   DW_AT_GNU_all_call_sites: 1
    <62>   DW_AT_name        : (indirect string, offset: 0x12a): make_v4f
    <66>   DW_AT_decl_file   : 1
    <67>   DW_AT_decl_line   : 19
    <68>   DW_AT_prototyped  : 1
    <68>   DW_AT_type        : <0x239>
    <6c>   DW_AT_external    : 1
 <2><6c>: Abbrev Number: 3 (DW_TAG_formal_parameter)
    <6d>   DW_AT_location    : 0x23 (location list)
    <71>   DW_AT_name        : (indirect string, offset: 0x1d8): a
    <75>   DW_AT_decl_file   : 1
    <76>   DW_AT_decl_line   : 19
    <77>   DW_AT_type        : <0x232>
 <2><7b>: Abbrev Number: 4 (DW_TAG_variable)
    <7c>   DW_AT_location    : 0x5c (location list)
    <80>   DW_AT_name        : (indirect string, offset: 0x1da): r
    <84>   DW_AT_decl_file   : 1
    <85>   DW_AT_decl_line   : 19
    <86>   DW_AT_type        : <0x239>
 <2><8a>: Abbrev Number: 0
 <1><8b>: Abbrev Number: 2 (DW_TAG_subprogram)
    <8c>   DW_AT_low_pc      : 0x20
    <94>   DW_AT_high_pc     : 0x5
    <98>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <9a>   DW_AT_GNU_all_call_sites: 1
    <9a>   DW_AT_name        : (indirect string, offset: 0x14b): take_rv1f
    <9e>   DW_AT_decl_file   : 1
    <9f>   DW_AT_decl_line   : 20
    <a0>   DW_AT_prototyped  : 1
    <a0>   DW_AT_type        : <0x232>
    <a4>   DW_AT_external    : 1
 <2><a4>: Abbrev Number: 5 (DW_TAG_formal_parameter)
    <a5>   DW_AT_location    : 1 byte block: 52 	(DW_OP_reg2 (rcx))
    <a7>   DW_AT_name        : (indirect string, offset: 0x1da): r
    <ab>   DW_AT_decl_file   : 1
    <ac>   DW_AT_decl_line   : 20
    <ad>   DW_AT_type        : <0x29d>
 <2><b1>: Abbrev Number: 0
 <1><b2>: Abbrev Number: 2 (DW_TAG_subprogram)
    <b3>   DW_AT_low_pc      : 0x30
    <bb>   DW_AT_high_pc     : 0x8
    <bf>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <c1>   DW_AT_GNU_all_call_sites: 1
    <c1>   DW_AT_name        : (indirect string, offset: 0x155): take_rv2i
    <c5>   DW_AT_decl_file   : 1
    <c6>   DW_AT_decl_line   : 21
    <c7>   DW_AT_prototyped  : 1
    <c7>   DW_AT_type        : <0x257>
    <cb>   DW_AT_external    : 1
 <2><cb>: Abbrev Number: 5 (DW_TAG_formal_parameter)
    <cc>   DW_AT_location    : 1 byte block: 52 	(DW_OP_reg2 (rcx))
    <ce>   DW_AT_name        : (indirect string, offset: 0x1da): r
    <d2>   DW_AT_decl_file   : 1
    <d3>   DW_AT_decl_line   : 21
    <d4>   DW_AT_type        : <0x2d0>
 <2><d8>: Abbrev Number: 0
 <1><d9>: Abbrev Number: 2 (DW_TAG_subprogram)
    <da>   DW_AT_low_pc      : 0x40
    <e2>   DW_AT_high_pc     : 0x6
    <e6>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <e8>   DW_AT_GNU_all_call_sites: 1
    <e8>   DW_AT_name        : (indirect string, offset: 0x163): take_rv1d
    <ec>   DW_AT_decl_file   : 1
    <ed>   DW_AT_decl_line   : 22
    <ee>   DW_AT_prototyped  : 1
    <ee>   DW_AT_type        : <0x25e>
    <f2>   DW_AT_external    : 1
 <2><f2>: Abbrev Number: 5 (DW_TAG_formal_parameter)
    <f3>   DW_AT_location    : 1 byte block: 52 	(DW_OP_reg2 (rcx))
    <f5>   DW_AT_name        : (indirect string, offset: 0x1da): r
    <f9>   DW_AT_decl_file   : 1
    <fa>   DW_AT_decl_line   : 22
    <fb>   DW_AT_type        : <0x303>
 <2><ff>: Abbrev Number: 0
 <1><100>: Abbrev Number: 2 (DW_TAG_subprogram)
    <101>   DW_AT_low_pc      : 0x50
    <109>   DW_AT_high_pc     : 0x4
    <10d>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <10f>   DW_AT_GNU_all_call_sites: 1
    <10f>   DW_AT_name        : (indirect string, offset: 0x174): take_rv2fi
    <113>   DW_AT_decl_file   : 1
    <114>   DW_AT_decl_line   : 23
    <115>   DW_AT_prototyped  : 1
    <115>   DW_AT_type        : <0x257>
    <119>   DW_AT_external    : 1
 <2><119>: Abbrev Number: 5 (DW_TAG_formal_parameter)
    <11a>   DW_AT_location    : 2 byte block: 72 0 	(DW_OP_breg2 (rcx): 0)
    <11d>   DW_AT_name        : (indirect string, offset: 0x1da): r
    <121>   DW_AT_decl_file   : 1
    <122>   DW_AT_decl_line   : 23
    <123>   DW_AT_type        : <0x336>
 <2><127>: Abbrev Number: 0
 <1><128>: Abbrev Number: 2 (DW_TAG_subprogram)
    <129>   DW_AT_low_pc      : 0x60
    <131>   DW_AT_high_pc     : 0xb
    <135>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <137>   DW_AT_GNU_all_call_sites: 1
    <137>   DW_AT_name        : (indirect string, offset: 0x17f): make_rv4f
    <13b>   DW_AT_decl_file   : 1
    <13c>   DW_AT_decl_line   : 24
    <13d>   DW_AT_prototyped  : 1
    <13d>   DW_AT_type        : <0x265>
    <141>   DW_AT_external    : 1
 <2><141>: Abbrev Number: 3 (DW_TAG_formal_parameter)
    <142>   DW_AT_location    : 0x7f (location list)
    <146>   DW_AT_name        : (indirect string, offset: 0x1d8): a
    <14a>   DW_AT_decl_file   : 1
    <14b>   DW_AT_decl_line   : 24
    <14c>   DW_AT_type        : <0x232>
 <2><150>: Abbrev Number: 6 (DW_TAG_variable)
    <151>   DW_AT_location    : 2 byte block: 72 0 	(DW_OP_breg2 (rcx): 0)
    <154>   DW_AT_name        : (indirect string, offset: 0x1da): r
    <158>   DW_AT_decl_file   : 1
    <159>   DW_AT_decl_line   : 24
    <15a>   DW_AT_type        : <0x265>
 <2><15e>: Abbrev Number: 0
 <1><15f>: Abbrev Number: 2 (DW_TAG_subprogram)
    <160>   DW_AT_low_pc      : 0x70
    <168>   DW_AT_high_pc     : 0x6
    <16c>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <16e>   DW_AT_GNU_all_call_sites: 1
    <16e>   DW_AT_name        : (indirect string, offset: 0x190): take_rv8f
    <172>   DW_AT_decl_file   : 1
    <173>   DW_AT_decl_line   : 25
    <174>   DW_AT_prototyped  : 1
    <174>   DW_AT_type        : <0x232>
    <178>   DW_AT_external    : 1
 <2><178>: Abbrev Number: 5 (DW_TAG_formal_parameter)
    <179>   DW_AT_location    : 2 byte block: 72 0 	(DW_OP_breg2 (rcx): 0)
    <17c>   DW_AT_name        : (indirect string, offset: 0x1da): r
    <180>   DW_AT_decl_file   : 1
    <181>   DW_AT_decl_line   : 25
    <182>   DW_AT_type        : <0x375>
 <2><186>: Abbrev Number: 0
 <1><187>: Abbrev Number: 2 (DW_TAG_subprogram)
    <188>   DW_AT_low_pc      : 0x80
    <190>   DW_AT_high_pc     : 0x4
    <194>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <196>   DW_AT_GNU_all_call_sites: 1
    <196>   DW_AT_name        : (indirect string, offset: 0x19a): take_rarr
    <19a>   DW_AT_decl_file   : 1
    <19b>   DW_AT_decl_line   : 26
    <19c>   DW_AT_prototyped  : 1
    <19c>   DW_AT_type        : <0x257>
    <1a0>   DW_AT_external    : 1
 <2><1a0>: Abbrev Number: 5 (DW_TAG_formal_parameter)
    <1a1>   DW_AT_location    : 2 byte block: 72 0 	(DW_OP_breg2 (rcx): 0)
    <1a4>   DW_AT_name        : (indirect string, offset: 0x1da): r
    <1a8>   DW_AT_decl_file   : 1
    <1a9>   DW_AT_decl_line   : 26
    <1aa>   DW_AT_type        : <0x3a8>
 <2><1ae>: Abbrev Number: 0
 <1><1af>: Abbrev Number: 2 (DW_TAG_subprogram)
    <1b0>   DW_AT_low_pc      : 0x90
    <1b8>   DW_AT_high_pc     : 0x5
    <1bc>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <1be>   DW_AT_GNU_all_call_sites: 1
    <1be>   DW_AT_name        : (indirect string, offset: 0x1a4): make_rnest
    <1c2>   DW_AT_decl_file   : 1
    <1c3>   DW_AT_decl_line   : 27
    <1c4>   DW_AT_prototyped  : 1
    <1c4>   DW_AT_type        : <0x281>
    <1c8>   DW_AT_external    : 1
 <2><1c8>: Abbrev Number: 5 (DW_TAG_formal_parameter)
    <1c9>   DW_AT_location    : 1 byte block: 61 	(DW_OP_reg17 (xmm0))
    <1cb>   DW_AT_name        : (indirect string, offset: 0x1d8): a
    <1cf>   DW_AT_decl_file   : 1
    <1d0>   DW_AT_decl_line   : 27
    <1d1>   DW_AT_type        : <0x232>
 <2><1d5>: Abbrev Number: 7 (DW_TAG_variable)
    <1d6>   DW_AT_name        : (indirect string, offset: 0x1da): r
    <1da>   DW_AT_decl_file   : 1
    <1db>   DW_AT_decl_line   : 27
    <1dc>   DW_AT_type        : <0x281>
 <2><1e0>: Abbrev Number: 0
 <1><1e1>: Abbrev Number: 2 (DW_TAG_subprogram)
    <1e2>   DW_AT_low_pc      : 0xa0
    <1ea>   DW_AT_high_pc     : 0x5
    <1ee>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <1f0>   DW_AT_GNU_all_call_sites: 1
    <1f0>   DW_AT_name        : (indirect string, offset: 0x1c1): take_rplain
    <1f4>   DW_AT_decl_file   : 1
    <1f5>   DW_AT_decl_line   : 28
    <1f6>   DW_AT_prototyped  : 1
    <1f6>   DW_AT_type        : <0x232>
    <1fa>   DW_AT_external    : 1
 <2><1fa>: Abbrev Number: 3 (DW_TAG_formal_parameter)
    <1fb>   DW_AT_location    : 0xb8 (location list)
    <1ff>   DW_AT_name        : (indirect string, offset: 0x1da): r
    <203>   DW_AT_decl_file   : 1
    <204>   DW_AT_decl_line   : 28
    <205>   DW_AT_type        : <0x3dc>
 <2><209>: Abbrev Number: 0
 <1><20a>: Abbrev Number: 2 (DW_TAG_subprogram)
    <20b>   DW_AT_low_pc      : 0xb0
    <213>   DW_AT_high_pc     : 0x6
    <217>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <219>   DW_AT_GNU_all_call_sites: 1
    <219>   DW_AT_name        : (indirect string, offset: 0x1cd): take_ral
    <21d>   DW_AT_decl_file   : 1
    <21e>   DW_AT_decl_line   : 29
    <21f>   DW_AT_prototyped  : 1
    <21f>   DW_AT_type        : <0x232>
    <223>   DW_AT_external    : 1
 <2><223>: Abbrev Number: 5 (DW_TAG_formal_parameter)
    <224>   DW_AT_location    : 2 byte block: 72 0 	(DW_OP_breg2 (rcx): 0)
    <227>   DW_AT_name        : (indirect string, offset: 0x1da): r
    <22b>   DW_AT_decl_file   : 1
    <22c>   DW_AT_decl_line   : 29
    <22d>   DW_AT_type        : <0x404>
 <2><231>: Abbrev Number: 0
 <1><232>: Abbrev Number: 8 (DW_TAG_base_type)
    <233>   DW_AT_name        : (indirect string, offset: 0x124): float
    <237>   DW_AT_encoding    : 4	(float)
    <238>   DW_AT_byte_size   : 4
 <1><239>: Abbrev Number: 9 (DW_TAG_typedef)
    <23a>   DW_AT_type        : <0x244>
    <23e>   DW_AT_name        : (indirect string, offset: 0x147): v4f
    <242>   DW_AT_decl_file   : 1
    <243>   DW_AT_decl_line   : 5
 <1><244>: Abbrev Number: 10 (DW_TAG_array_type)
    <245>   DW_AT_GNU_vector  : 1
    <245>   DW_AT_type        : <0x232>
 <2><249>: Abbrev Number: 11 (DW_TAG_subrange_type)
    <24a>   DW_AT_type        : <0x250>
    <24e>   DW_AT_count       : 4
 <2><24f>: Abbrev Number: 0
 <1><250>: Abbrev Number: 12 (DW_TAG_base_type)
    <251>   DW_AT_name        : (indirect string, offset: 0x133): __ARRAY_SIZE_TYPE__
    <255>   DW_AT_byte_size   : 8
    <256>   DW_AT_encoding    : 7	(unsigned)
 <1><257>: Abbrev Number: 8 (DW_TAG_base_type)
    <258>   DW_AT_name        : (indirect string, offset: 0x15f): int
    <25c>   DW_AT_encoding    : 5	(signed)
    <25d>   DW_AT_byte_size   : 4
 <1><25e>: Abbrev Number: 8 (DW_TAG_base_type)
    <25f>   DW_AT_name        : (indirect string, offset: 0x16d): double
    <263>   DW_AT_encoding    : 4	(float)
    <264>   DW_AT_byte_size   : 8
 <1><265>: Abbrev Number: 9 (DW_TAG_typedef)
    <266>   DW_AT_type        : <0x270>
    <26a>   DW_AT_name        : (indirect string, offset: 0x18b): RV4F
    <26e>   DW_AT_decl_file   : 1
    <26f>   DW_AT_decl_line   : 11
 <1><270>: Abbrev Number: 13 (DW_TAG_structure_type)
    <271>   DW_AT_byte_size   : 16
    <272>   DW_AT_decl_file   : 1
    <273>   DW_AT_decl_line   : 11
 <2><274>: Abbrev Number: 14 (DW_TAG_member)
    <275>   DW_AT_name        : (indirect string, offset: 0x189): v
    <279>   DW_AT_type        : <0x239>
    <27d>   DW_AT_decl_file   : 1
    <27e>   DW_AT_decl_line   : 11
    <27f>   DW_AT_data_member_location: 0
 <2><280>: Abbrev Number: 0
 <1><281>: Abbrev Number: 9 (DW_TAG_typedef)
    <282>   DW_AT_type        : <0x28c>
    <286>   DW_AT_name        : (indirect string, offset: 0x1bb): RNest
    <28a>   DW_AT_decl_file   : 1
    <28b>   DW_AT_decl_line   : 14
 <1><28c>: Abbrev Number: 13 (DW_TAG_structure_type)
    <28d>   DW_AT_byte_size   : 4
    <28e>   DW_AT_decl_file   : 1
    <28f>   DW_AT_decl_line   : 14
 <2><290>: Abbrev Number: 14 (DW_TAG_member)
    <291>   DW_AT_name        : (indirect string, offset: 0x1af): in
    <295>   DW_AT_type        : <0x29d>
    <299>   DW_AT_decl_file   : 1
    <29a>   DW_AT_decl_line   : 14
    <29b>   DW_AT_data_member_location: 0
 <2><29c>: Abbrev Number: 0
 <1><29d>: Abbrev Number: 9 (DW_TAG_typedef)
    <29e>   DW_AT_type        : <0x2a8>
    <2a2>   DW_AT_name        : (indirect string, offset: 0x1b6): RV1F
    <2a6>   DW_AT_decl_file   : 1
    <2a7>   DW_AT_decl_line   : 7
 <1><2a8>: Abbrev Number: 13 (DW_TAG_structure_type)
    <2a9>   DW_AT_byte_size   : 4
    <2aa>   DW_AT_decl_file   : 1
    <2ab>   DW_AT_decl_line   : 7
 <2><2ac>: Abbrev Number: 14 (DW_TAG_member)
    <2ad>   DW_AT_name        : (indirect string, offset: 0x189): v
    <2b1>   DW_AT_type        : <0x2b9>
    <2b5>   DW_AT_decl_file   : 1
    <2b6>   DW_AT_decl_line   : 7
    <2b7>   DW_AT_data_member_location: 0
 <2><2b8>: Abbrev Number: 0
 <1><2b9>: Abbrev Number: 9 (DW_TAG_typedef)
    <2ba>   DW_AT_type        : <0x2c4>
    <2be>   DW_AT_name        : (indirect string, offset: 0x1b2): v1f
    <2c2>   DW_AT_decl_file   : 1
    <2c3>   DW_AT_decl_line   : 1
 <1><2c4>: Abbrev Number: 10 (DW_TAG_array_type)
    <2c5>   DW_AT_GNU_vector  : 1
    <2c5>   DW_AT_type        : <0x232>
 <2><2c9>: Abbrev Number: 11 (DW_TAG_subrange_type)
    <2ca>   DW_AT_type        : <0x250>
    <2ce>   DW_AT_count       : 1
 <2><2cf>: Abbrev Number: 0
 <1><2d0>: Abbrev Number: 9 (DW_TAG_typedef)
    <2d1>   DW_AT_type        : <0x2db>
    <2d5>   DW_AT_name        : (indirect string, offset: 0x1e0): RV2I
    <2d9>   DW_AT_decl_file   : 1
    <2da>   DW_AT_decl_line   : 8
 <1><2db>: Abbrev Number: 13 (DW_TAG_structure_type)
    <2dc>   DW_AT_byte_size   : 8
    <2dd>   DW_AT_decl_file   : 1
    <2de>   DW_AT_decl_line   : 8
 <2><2df>: Abbrev Number: 14 (DW_TAG_member)
    <2e0>   DW_AT_name        : (indirect string, offset: 0x189): v
    <2e4>   DW_AT_type        : <0x2ec>
    <2e8>   DW_AT_decl_file   : 1
    <2e9>   DW_AT_decl_line   : 8
    <2ea>   DW_AT_data_member_location: 0
 <2><2eb>: Abbrev Number: 0
 <1><2ec>: Abbrev Number: 9 (DW_TAG_typedef)
    <2ed>   DW_AT_type        : <0x2f7>
    <2f1>   DW_AT_name        : (indirect string, offset: 0x1dc): v2i
    <2f5>   DW_AT_decl_file   : 1
    <2f6>   DW_AT_decl_line   : 2
 <1><2f7>: Abbrev Number: 10 (DW_TAG_array_type)
    <2f8>   DW_AT_GNU_vector  : 1
    <2f8>   DW_AT_type        : <0x257>
 <2><2fc>: Abbrev Number: 11 (DW_TAG_subrange_type)
    <2fd>   DW_AT_type        : <0x250>
    <301>   DW_AT_count       : 2
 <2><302>: Abbrev Number: 0
 <1><303>: Abbrev Number: 9 (DW_TAG_typedef)
    <304>   DW_AT_type        : <0x30e>
    <308>   DW_AT_name        : (indirect string, offset: 0x1e9): RV1D
    <30c>   DW_AT_decl_file   : 1
    <30d>   DW_AT_decl_line   : 9
 <1><30e>: Abbrev Number: 13 (DW_TAG_structure_type)
    <30f>   DW_AT_byte_size   : 8
    <310>   DW_AT_decl_file   : 1
    <311>   DW_AT_decl_line   : 9
 <2><312>: Abbrev Number: 14 (DW_TAG_member)
    <313>   DW_AT_name        : (indirect string, offset: 0x189): v
    <317>   DW_AT_type        : <0x31f>
    <31b>   DW_AT_decl_file   : 1
    <31c>   DW_AT_decl_line   : 9
    <31d>   DW_AT_data_member_location: 0
 <2><31e>: Abbrev Number: 0
 <1><31f>: Abbrev Number: 9 (DW_TAG_typedef)
    <320>   DW_AT_type        : <0x32a>
    <324>   DW_AT_name        : (indirect string, offset: 0x1e5): v1d
    <328>   DW_AT_decl_file   : 1
    <329>   DW_AT_decl_line   : 3
 <1><32a>: Abbrev Number: 10 (DW_TAG_array_type)
    <32b>   DW_AT_GNU_vector  : 1
    <32b>   DW_AT_type        : <0x25e>
 <2><32f>: Abbrev Number: 11 (DW_TAG_subrange_type)
    <330>   DW_AT_type        : <0x250>
    <334>   DW_AT_count       : 1
 <2><335>: Abbrev Number: 0
 <1><336>: Abbrev Number: 9 (DW_TAG_typedef)
    <337>   DW_AT_type        : <0x341>
    <33b>   DW_AT_name        : (indirect string, offset: 0x1f4): RV2FI
    <33f>   DW_AT_decl_file   : 1
    <340>   DW_AT_decl_line   : 10
 <1><341>: Abbrev Number: 13 (DW_TAG_structure_type)
    <342>   DW_AT_byte_size   : 16
    <343>   DW_AT_decl_file   : 1
    <344>   DW_AT_decl_line   : 10
 <2><345>: Abbrev Number: 14 (DW_TAG_member)
    <346>   DW_AT_name        : (indirect string, offset: 0x189): v
    <34a>   DW_AT_type        : <0x35e>
    <34e>   DW_AT_decl_file   : 1
    <34f>   DW_AT_decl_line   : 10
    <350>   DW_AT_data_member_location: 0
 <2><351>: Abbrev Number: 14 (DW_TAG_member)
    <352>   DW_AT_name        : (indirect string, offset: 0x1f2): i
    <356>   DW_AT_type        : <0x257>
    <35a>   DW_AT_decl_file   : 1
    <35b>   DW_AT_decl_line   : 10
    <35c>   DW_AT_data_member_location: 8
 <2><35d>: Abbrev Number: 0
 <1><35e>: Abbrev Number: 9 (DW_TAG_typedef)
    <35f>   DW_AT_type        : <0x369>
    <363>   DW_AT_name        : (indirect string, offset: 0x1ee): v2f
    <367>   DW_AT_decl_file   : 1
    <368>   DW_AT_decl_line   : 4
 <1><369>: Abbrev Number: 10 (DW_TAG_array_type)
    <36a>   DW_AT_GNU_vector  : 1
    <36a>   DW_AT_type        : <0x232>
 <2><36e>: Abbrev Number: 11 (DW_TAG_subrange_type)
    <36f>   DW_AT_type        : <0x250>
    <373>   DW_AT_count       : 2
 <2><374>: Abbrev Number: 0
 <1><375>: Abbrev Number: 9 (DW_TAG_typedef)
    <376>   DW_AT_type        : <0x380>
    <37a>   DW_AT_name        : (indirect string, offset: 0x1fe): RV8F
    <37e>   DW_AT_decl_file   : 1
    <37f>   DW_AT_decl_line   : 12
 <1><380>: Abbrev Number: 13 (DW_TAG_structure_type)
    <381>   DW_AT_byte_size   : 32
    <382>   DW_AT_decl_file   : 1
    <383>   DW_AT_decl_line   : 12
 <2><384>: Abbrev Number: 14 (DW_TAG_member)
    <385>   DW_AT_name        : (indirect string, offset: 0x189): v
    <389>   DW_AT_type        : <0x391>
    <38d>   DW_AT_decl_file   : 1
    <38e>   DW_AT_decl_line   : 12
    <38f>   DW_AT_data_member_location: 0
 <2><390>: Abbrev Number: 0
 <1><391>: Abbrev Number: 9 (DW_TAG_typedef)
    <392>   DW_AT_type        : <0x39c>
    <396>   DW_AT_name        : (indirect string, offset: 0x1fa): v8f
    <39a>   DW_AT_decl_file   : 1
    <39b>   DW_AT_decl_line   : 6
 <1><39c>: Abbrev Number: 10 (DW_TAG_array_type)
    <39d>   DW_AT_GNU_vector  : 1
    <39d>   DW_AT_type        : <0x232>
 <2><3a1>: Abbrev Number: 11 (DW_TAG_subrange_type)
    <3a2>   DW_AT_type        : <0x250>
    <3a6>   DW_AT_count       : 8
 <2><3a7>: Abbrev Number: 0
 <1><3a8>: Abbrev Number: 9 (DW_TAG_typedef)
    <3a9>   DW_AT_type        : <0x3b3>
    <3ad>   DW_AT_name        : (indirect string, offset: 0x20b): RArr
    <3b1>   DW_AT_decl_file   : 1
    <3b2>   DW_AT_decl_line   : 13
 <1><3b3>: Abbrev Number: 13 (DW_TAG_structure_type)
    <3b4>   DW_AT_byte_size   : 48
    <3b5>   DW_AT_decl_file   : 1
    <3b6>   DW_AT_decl_line   : 13
 <2><3b7>: Abbrev Number: 14 (DW_TAG_member)
    <3b8>   DW_AT_name        : (indirect string, offset: 0x203): lanes
    <3bc>   DW_AT_type        : <0x3d0>
    <3c0>   DW_AT_decl_file   : 1
    <3c1>   DW_AT_decl_line   : 13
    <3c2>   DW_AT_data_member_location: 0
 <2><3c3>: Abbrev Number: 14 (DW_TAG_member)
    <3c4>   DW_AT_name        : (indirect string, offset: 0x209): k
    <3c8>   DW_AT_type        : <0x257>
    <3cc>   DW_AT_decl_file   : 1
    <3cd>   DW_AT_decl_line   : 13
    <3ce>   DW_AT_data_member_location: 32
 <2><3cf>: Abbrev Number: 0
 <1><3d0>: Abbrev Number: 15 (DW_TAG_array_type)
    <3d1>   DW_AT_type        : <0x239>
 <2><3d5>: Abbrev Number: 11 (DW_TAG_subrange_type)
    <3d6>   DW_AT_type        : <0x250>
    <3da>   DW_AT_count       : 2
 <2><3db>: Abbrev Number: 0
 <1><3dc>: Abbrev Number: 9 (DW_TAG_typedef)
    <3dd>   DW_AT_type        : <0x3e7>
    <3e1>   DW_AT_name        : (indirect string, offset: 0x212): RPlain
    <3e5>   DW_AT_decl_file   : 1
    <3e6>   DW_AT_decl_line   : 15
 <1><3e7>: Abbrev Number: 13 (DW_TAG_structure_type)
    <3e8>   DW_AT_byte_size   : 8
    <3e9>   DW_AT_decl_file   : 1
    <3ea>   DW_AT_decl_line   : 15
 <2><3eb>: Abbrev Number: 14 (DW_TAG_member)
    <3ec>   DW_AT_name        : (indirect string, offset: 0x1d6): x
    <3f0>   DW_AT_type        : <0x232>
    <3f4>   DW_AT_decl_file   : 1
    <3f5>   DW_AT_decl_line   : 15
    <3f6>   DW_AT_data_member_location: 0
 <2><3f7>: Abbrev Number: 14 (DW_TAG_member)
    <3f8>   DW_AT_name        : (indirect string, offset: 0x210): y
    <3fc>   DW_AT_type        : <0x232>
    <400>   DW_AT_decl_file   : 1
    <401>   DW_AT_decl_line   : 15
    <402>   DW_AT_data_member_location: 4
 <2><403>: Abbrev Number: 0
 <1><404>: Abbrev Number: 9 (DW_TAG_typedef)
    <405>   DW_AT_type        : <0x40f>
    <409>   DW_AT_name        : (indirect string, offset: 0x21d): RAl
    <40d>   DW_AT_decl_file   : 1
    <40e>   DW_AT_decl_line   : 17
 <1><40f>: Abbrev Number: 13 (DW_TAG_structure_type)
    <410>   DW_AT_byte_size   : 16
    <411>   DW_AT_decl_file   : 1
    <412>   DW_AT_decl_line   : 17
 <2><413>: Abbrev Number: 16 (DW_TAG_member)
    <414>   DW_AT_name        : (indirect string, offset: 0x1d8): a
    <418>   DW_AT_type        : <0x421>
    <41c>   DW_AT_decl_file   : 1
    <41d>   DW_AT_decl_line   : 17
    <41e>   DW_AT_alignment   : 16
    <41f>   DW_AT_data_member_location: 0
 <2><420>: Abbrev Number: 0
 <1><421>: Abbrev Number: 9 (DW_TAG_typedef)
    <422>   DW_AT_type        : <0x42c>
    <426>   DW_AT_name        : (indirect string, offset: 0x219): al4
    <42a>   DW_AT_decl_file   : 1
    <42b>   DW_AT_decl_line   : 16
 <1><42c>: Abbrev Number: 15 (DW_TAG_array_type)
    <42d>   DW_AT_type        : <0x232>
 <2><431>: Abbrev Number: 11 (DW_TAG_subrange_type)
    <432>   DW_AT_type        : <0x250>
    <436>   DW_AT_count       : 4
 <2><437>: Abbrev Number: 0
 <1><438>: Abbrev Number: 0

"""

const _VRT, _VSD, _, _, _ = RepliBuild.Compiler.parse_dwarf_dump(_VEC_DUMP)

# The metadata `functions` entry a build writes for a fixture function: the
# return type and parameters the parser read, merged the way
# `Compiler.extract_compilation_metadata` merges them (`vector_size` included).
function _vfunc(name; byval = nothing, rt = _VRT)
    r = rt[name]
    ret = Dict{String,Any}("c_type" => r["c_type"], "julia_type" => r["julia_type"], "size" => r["size"])
    haskey(r, "vector_size") && (ret["vector_size"] = r["vector_size"])
    f = Dict{String,Any}("name" => name, "mangled" => name, "demangled" => name, "is_vararg" => false,
                         "parameters" => Any[copy(p) for p in r["parameters"]], "return_type" => ret)
    byval === nothing ||
        (f["byval_params"] = Any[Dict{String,Any}("record" => n, "align" => a) for (n, a) in byval])
    return f
end

_vmember(rec, name) = only(filter(m -> m["name"] == name, _VSD[rec]["members"]))

# Metadata from before `vector_size`: every key gone, `null` markers included.
_vstrip!(d::AbstractDict) = (delete!(d, "vector_size"); foreach(_vstrip!, values(d)); d)
_vstrip!(v::AbstractVector) = (foreach(_vstrip!, v); v)
_vstrip!(x) = x

# GeneratorC's wrapper for `fns` over `structs`, under `abi`, and what each
# function became: a vector trap, another trap, a call through BRANCH 0's
# aligned buffer or the byval carrier, or a plain call.
const _VDIR = mktempdir()
write(joinpath(_VDIR, "replibuild.toml"), """
[project]
name = "vec"
root = "$(escape_string(_VDIR))"
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
function _vgen(fns, structs, abi)
    cfg = RepliBuild.ConfigurationManager.load_config(joinpath(_VDIR, "replibuild.toml"))
    W = RepliBuild.Wrapper
    meta = Dict{String,Any}("functions" => fns, "struct_definitions" => structs, "globals" => Dict{String,Any}())
    libref = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
    return W.generate_introspective_module_c(cfg, libref, meta, "Vec", W.create_type_registry(cfg), false; abi)
end
_vbody(code, name) = let i = findfirst("function $name(", code)
    i === nothing ? "" : (r = SubString(code, first(i)); j = findfirst(r"\nend\b", r);
                          j === nothing ? String(r) : String(r[1:last(j)]))
end
_vform(code, name) = let b = _vbody(code, name)
    isempty(b) ? :missing :
    occursin("SIMD vector", b) && occursin("ABI Safety Trap", b) ? :vector :
    occursin("ABI Safety Trap", b) ? :trap :
    occursin("ret_ptr", b) ? :aligned :
    occursin("_rb_byval16(", b) ? :carrier :
    occursin("ccall(", b) ? :ccall : :other
end
_vparses(code) = !any(e -> e isa Expr && e.head in (:incomplete, :error), Meta.parseall(code).args)

const _VNAMES = ["take_v4f", "make_v4f", "take_rv1f", "take_rv2i", "take_rv1d", "take_rv2fi",
                 "make_rv4f", "take_rv8f", "take_rarr", "make_rnest", "take_rplain", "take_ral"]
# The compiler's slot for RArr (16-aligned), so SysV passes it in a carrier.
_vfns(rt) = Any[n == "take_rarr" ? _vfunc(n; byval = [("RArr", 16)], rt) : _vfunc(n; rt) for n in _VNAMES]

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
        # An array aligned to its own size is marked as looked at, not a vector.
        al = _vmember("RAl", "a")
        @test (al["c_type"], al["size"], al["alignment"]) == ("float[4]", 16, 16)
        @test haskey(al, "vector_size") && al["vector_size"] === nothing
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
            "RAl"    => ("regs:sse,sse", [:sse, :sse]),  # an aligned array is its lanes
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
        sysv, win64 = _vgen(_vfns(_VRT), _VSD, :sysv), _vgen(_vfns(_VRT), _VSD, :win64)
        #         function        SysV      Win64
        table = [("take_v4f",    :vector,  :ccall),   # one XMM vs two; Win64 passes it by reference either way
                 ("make_v4f",    :vector,  :vector),  # no Julia type: `float[4]` failed the module's load
                 ("take_rv1f",   :vector,  :ccall),   # RDI vs XMM0; Win64 by size
                 ("take_rv2i",   :vector,  :ccall),
                 ("take_rv1d",   :vector,  :ccall),   # memory vs XMM0
                 ("make_rv4f",   :vector,  :aligned), # Win64: a 16-aligned buffer for the hidden pointer
                 ("take_rv8f",   :vector,  :ccall),
                 ("make_rnest",  :vector,  :ccall),
                 ("take_rv2fi",  :ccall,   :ccall),   # SSE, INTEGER on both readings
                 ("take_rarr",   :carrier, :ccall),   # MEMORY on both readings
                 ("take_rplain", :ccall,   :ccall),
                 ("take_ral",    :ccall,   :ccall)]   # an aligned array is not a vector
        checked = 0
        for (fn, s, w) in table
            @test (fn, _vform(sysv, fn), _vform(win64, fn)) == (fn, s, w)
            checked += 1
        end
        @test checked == length(_VNAMES)
        # Both wrappers parse, and no ccall names a C array spelling as a type.
        for code in (sysv, win64)
            @test _vparses(code)
            @test !occursin(r"ccall\([^\n]*\bfloat\[", code)
        end
    end

    @testset "metadata from before vector_size fails safe" begin
        # A re-wrap of a package built before `vector_size` reads metadata
        # with none: no key on a vector, and no `null` on an aligned array.
        old_sd, old_rt = _vstrip!(deepcopy(_VSD)), _vstrip!(deepcopy(_VRT))
        @test !occursin("vector_size", repr(old_sd)) && !occursin("vector_size", repr(old_rt))

        # A bare vector still reads as one from its spelling, exactly: C and
        # C++ never pass or return an array by value.
        @test _VFG.crosses_as_vector(only(old_rt["take_v4f"]["parameters"]))
        @test _VFG.crosses_as_vector(Dict("c_type" => old_rt["make_v4f"]["c_type"]))
        @test !_VFG.crosses_as_vector(only(old_rt["take_rv1f"]["parameters"]))
        @test !_VFG.crosses_as_vector(Dict("c_type" => "float*"))
        for name in ("take_v4f", "make_v4f")
            @test _VFG.thunk_crossings(_vfunc(name; rt = old_rt), old_sd, Dict{String,Any}()).abi_trap !== nothing
        end

        # A member that is an array aligned to its own size may be a vector or
        # an aligned array (RAl), which only a rebuild tells apart: refused on
        # SysV, naming the rebuild. Win64 crosses a record by size and asks
        # nothing. A record with no such member is classified as before.
        checked = 0
        for rec in ("RV1F", "RV2I", "RV1D", "RV2FI", "RV4F", "RV8F", "RNest", "RAl")
            v = _VFG.sysv_record_abi(rec, old_sd)
            @test (rec, v.kind, occursin("rebuild", v.reason)) == (rec, :unknown, true)
            @test _VFG.sysv_trap_reason(rec, old_sd; ret = false, abi = :win64) === nothing
            checked += 1
        end
        @test checked == 8
        @test _VFG.sysv_statement("RPlain", old_sd) == "regs:sse"

        # GeneratorC: every crossing that may hold a vector is refused on SysV;
        # on Win64 only the vector return, as with new metadata.
        sysv, win64 = _vgen(_vfns(old_rt), old_sd, :sysv), _vgen(_vfns(old_rt), old_sd, :win64)
        for fn in _VNAMES
            s = fn in ("take_rarr",) ? :carrier : fn == "take_rplain" ? :ccall : :vector
            w = fn == "make_v4f" ? :vector : fn == "make_rv4f" ? :aligned : :ccall
            @test (fn, _vform(sysv, fn), _vform(win64, fn)) == (fn, s, w)
        end
        @test occursin("rebuild", _vbody(sysv, "take_ral"))
        for code in (sysv, win64)
            @test _vparses(code)
            @test !occursin(r"ccall\([^\n]*\bfloat\[", code)
        end
    end
end
