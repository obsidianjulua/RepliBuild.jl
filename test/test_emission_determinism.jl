#!/usr/bin/env julia
# Emitted metadata and wrapper text do not depend on Dict order, the Julia
# release or DIE offsets (2026-10-22).
#
# Found by a dry run of an unattended llama.cpp bump: nine upstream commits with
# no public header change, 0 API changes, and still ~1000 changed wrapper lines.
# Re-running the same source on Julia 1.12.7 and 1.13.2 separated the causes:
#
#   * DWARF extraction iterated `type_refs`, a Dict keyed by DIE-offset strings,
#     and the last definition written won. Its order is the strings' hash order,
#     which changes with the Julia release (1.12 and 1.13 hash strings
#     differently) and with every offset shift. Six pimpl `impl`s shared one
#     entry, C and C++ compile units flipped `_Bool`/`bool` and `_anon1`/`data`.
#   * GeneratorCpp emitted `Managed*` blocks in a Dict's order, and callback
#     docstrings took their `@cfunction` example from a name-similarity score
#     whose ties followed Dict order. The examples were often another callback's.
#   * IR file names carried `Base.hash` of the source directory, so a Julia
#     switch recompiled every TU and left a second IR set in `build/`.
#   * Lambda closure types were emitted under names built from the lambda's
#     absolute path and line (John chose to stop emitting them).
#
# Every DWARF case is a synthetic dump laid out per compile unit, so the same
# units can be reassembled in another order and at other offsets: the tables
# must not move. No toolchain; the generators' output is text.

using Test
using RepliBuild
using Libdl

const _ED_C = RepliBuild.Compiler
const _ED_W = RepliBuild.Wrapper
const _ED_F = RepliBuild.JLCSIRGenerator.FunctionGen
const _ED_S = RepliBuild.JLCSIRGenerator.StructGen
const _ED_LIB = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))

# ── Synthetic compile units ─────────────────────────────────────────────────
#
# Offsets are written `@<hex>`, relative to the unit's start. `_ed_assemble`
# places the units in a given order at a given base, which is what a link order
# change or an upstream edit does to a real dump: every DIE keeps its content
# and gets another offset. Shapes are copied from GNU objdump on clang 23.1.3
# output (`__clang_vtable` is a location-less member, a lambda's closure class
# is nested in its function and declared in the project's file, libc++'s
# `__func<(lambda at …)>` holds it as `__func_`).
#
# Every unit starts with a subprogram: pass 2 of the parser gives a
# subroutine type's parameters to the last subroutine type of the dump when
# they come before the first subprogram, a separate defect this test does not
# exercise.

const _ED_CU_C = (files = Dict(0 => "/proj/src/c_part.c", 1 => "/proj/include/shared.h"), tpl = """
 <0><@0>: Abbrev Number: 1 (DW_TAG_compile_unit)
    <@1>   DW_AT_producer    : (indirect string, offset: 0x0): clang version 23.1.3
    <@2>   DW_AT_language    : 29	(C11)
    <@3>   DW_AT_name        : (indirect string, offset: 0x1): c_part.c
 <1><@10>: Abbrev Number: 2 (DW_TAG_subprogram)
    <@11>   DW_AT_low_pc      : 0x1000
    <@12>   DW_AT_high_pc     : 0x10
    <@13>   DW_AT_name        : (indirect string, offset: 0x2): run_with
    <@14>   DW_AT_decl_file   : 0
    <@15>   DW_AT_type        : <0x@80>
    <@16>   DW_AT_external    : 1
 <2><@20>: Abbrev Number: 3 (DW_TAG_formal_parameter)
    <@21>   DW_AT_name        : (indirect string, offset: 0x3): cb
    <@22>   DW_AT_type        : <0x@90>
 <2><@28>: Abbrev Number: 3 (DW_TAG_formal_parameter)
    <@29>   DW_AT_name        : (indirect string, offset: 0x4): again
    <@2a>   DW_AT_type        : <0x@a0>
 <2><@30>: Abbrev Number: 3 (DW_TAG_formal_parameter)
    <@31>   DW_AT_name        : (indirect string, offset: 0x5): raw
    <@32>   DW_AT_type        : <0x@b0>
 <2><@38>: Abbrev Number: 0
 <1><@80>: Abbrev Number: 4 (DW_TAG_base_type)
    <@81>   DW_AT_name        : (indirect string, offset: 0x6): int
    <@82>   DW_AT_encoding    : 5	(signed)
    <@83>   DW_AT_byte_size   : 4
 <1><@88>: Abbrev Number: 4 (DW_TAG_base_type)
    <@89>   DW_AT_name        : (indirect string, offset: 0x7): _Bool
    <@8a>   DW_AT_encoding    : 2	(boolean)
    <@8b>   DW_AT_byte_size   : 1
 <1><@90>: Abbrev Number: 5 (DW_TAG_typedef)
    <@91>   DW_AT_type        : <0x@b0>
    <@92>   DW_AT_name        : (indirect string, offset: 0x8): progress_cb
    <@93>   DW_AT_decl_file   : 1
 <1><@a0>: Abbrev Number: 5 (DW_TAG_typedef)
    <@a1>   DW_AT_type        : <0x@90>
    <@a2>   DW_AT_name        : (indirect string, offset: 0x9): alias_cb
    <@a3>   DW_AT_decl_file   : 1
 <1><@b0>: Abbrev Number: 6 (DW_TAG_pointer_type)
    <@b1>   DW_AT_type        : <0x@c0>
 <1><@c0>: Abbrev Number: 7 (DW_TAG_subroutine_type)
    <@c1>   DW_AT_type        : <0x@80>
    <@c2>   DW_AT_prototyped  : 1
 <2><@c8>: Abbrev Number: 8 (DW_TAG_formal_parameter)
    <@c9>   DW_AT_type        : <0x@80>
 <2><@cc>: Abbrev Number: 0
 <1><@100>: Abbrev Number: 9 (DW_TAG_structure_type)
    <@101>   DW_AT_name        : (indirect string, offset: 0xa): traits
    <@102>   DW_AT_byte_size   : 8
    <@103>   DW_AT_decl_file   : 1
 <2><@108>: Abbrev Number: 10 (DW_TAG_member)
    <@109>   DW_AT_name        : (indirect string, offset: 0xb): size
    <@10a>   DW_AT_type        : <0x@80>
    <@10b>   DW_AT_decl_file   : 1
    <@10c>   DW_AT_data_member_location: 0
 <2><@110>: Abbrev Number: 10 (DW_TAG_member)
    <@111>   DW_AT_name        : (indirect string, offset: 0xc): is_quantized
    <@112>   DW_AT_type        : <0x@88>
    <@113>   DW_AT_decl_file   : 1
    <@114>   DW_AT_data_member_location: 4
 <2><@118>: Abbrev Number: 0
 <1><@200>: Abbrev Number: 9 (DW_TAG_structure_type)
    <@201>   DW_AT_name        : (indirect string, offset: 0xd): block
    <@202>   DW_AT_byte_size   : 8
    <@203>   DW_AT_decl_file   : 1
 <2><@208>: Abbrev Number: 11 (DW_TAG_union_type)
    <@209>   DW_AT_byte_size   : 4
    <@20a>   DW_AT_decl_file   : 1
 <3><@210>: Abbrev Number: 10 (DW_TAG_member)
    <@211>   DW_AT_name        : (indirect string, offset: 0xe): dm
    <@212>   DW_AT_type        : <0x@80>
    <@213>   DW_AT_decl_file   : 1
    <@214>   DW_AT_data_member_location: 0
 <3><@218>: Abbrev Number: 0
 <2><@220>: Abbrev Number: 12 (DW_TAG_member)
    <@221>   DW_AT_type        : <0x@208>
    <@222>   DW_AT_decl_file   : 1
    <@223>   DW_AT_data_member_location: 0
 <2><@228>: Abbrev Number: 10 (DW_TAG_member)
    <@229>   DW_AT_name        : (indirect string, offset: 0xf): qs
    <@22a>   DW_AT_type        : <0x@80>
    <@22b>   DW_AT_decl_file   : 1
    <@22c>   DW_AT_data_member_location: 4
 <2><@230>: Abbrev Number: 0
 <1><@300>: Abbrev Number: 13 (DW_TAG_variable)
    <@301>   DW_AT_name        : (indirect string, offset: 0x10): inline_flag
    <@302>   DW_AT_type        : <0x@88>
    <@303>   DW_AT_external    : 1
    <@304>   DW_AT_decl_file   : 1
 <1><@310>: Abbrev Number: 0
""")

# The C++ view of the same shared types, a nested `impl`, a project record with
# an anonymous union member, a lambda stored in libc++'s `std::function` (which
# captures a second lambda), a class with its vtable home, an enum and a member
# typedef whose bare names other units reuse for other types.
const _ED_CU_A = (files = Dict(0 => "/proj/src/cpp_a.cpp", 1 => "/proj/include/shared.h",
                               2 => "/usr/include/c++/v1/__functional/function.h",
                               3 => "/proj/src/model.h"), tpl = """
 <0><@0>: Abbrev Number: 1 (DW_TAG_compile_unit)
    <@1>   DW_AT_producer    : (indirect string, offset: 0x0): clang version 23.1.3
    <@2>   DW_AT_language    : 33	(C++14)
    <@3>   DW_AT_name        : (indirect string, offset: 0x1): cpp_a.cpp
 <1><@10>: Abbrev Number: 2 (DW_TAG_subprogram)
    <@11>   DW_AT_low_pc      : 0x2000
    <@12>   DW_AT_high_pc     : 0x10
    <@13>   DW_AT_linkage_name: (indirect string, offset: 0x2): _Z7a_entryv
    <@14>   DW_AT_name        : (indirect string, offset: 0x3): a_entry
    <@15>   DW_AT_decl_file   : 0
    <@16>   DW_AT_type        : <0x@80>
    <@17>   DW_AT_external    : 1
 <2><@20>: Abbrev Number: 14 (DW_TAG_class_type)
    <@21>   DW_AT_calling_convention: 5	(pass by value)
    <@22>   DW_AT_byte_size   : 8
    <@23>   DW_AT_decl_file   : 0
 <3><@28>: Abbrev Number: 10 (DW_TAG_member)
    <@29>   DW_AT_name        : (indirect string, offset: 0x4): k
    <@2a>   DW_AT_type        : <0x@80>
    <@2b>   DW_AT_decl_file   : 0
    <@2c>   DW_AT_data_member_location: 0
 <3><@30>: Abbrev Number: 10 (DW_TAG_member)
    <@31>   DW_AT_name        : (indirect string, offset: 0x5): inner
    <@32>   DW_AT_type        : <0x@40>
    <@33>   DW_AT_decl_file   : 0
    <@34>   DW_AT_data_member_location: 4
 <3><@38>: Abbrev Number: 0
 <2><@40>: Abbrev Number: 14 (DW_TAG_class_type)
    <@41>   DW_AT_calling_convention: 5	(pass by value)
    <@42>   DW_AT_byte_size   : 4
    <@43>   DW_AT_decl_file   : 0
 <3><@48>: Abbrev Number: 10 (DW_TAG_member)
    <@49>   DW_AT_name        : (indirect string, offset: 0x6): j
    <@4a>   DW_AT_type        : <0x@80>
    <@4b>   DW_AT_decl_file   : 0
    <@4c>   DW_AT_data_member_location: 0
 <3><@50>: Abbrev Number: 0
 <2><@58>: Abbrev Number: 0
 <1><@80>: Abbrev Number: 4 (DW_TAG_base_type)
    <@81>   DW_AT_name        : (indirect string, offset: 0x7): int
    <@82>   DW_AT_encoding    : 5	(signed)
    <@83>   DW_AT_byte_size   : 4
 <1><@88>: Abbrev Number: 4 (DW_TAG_base_type)
    <@89>   DW_AT_name        : (indirect string, offset: 0x8): bool
    <@8a>   DW_AT_encoding    : 2	(boolean)
    <@8b>   DW_AT_byte_size   : 1
 <1><@90>: Abbrev Number: 4 (DW_TAG_base_type)
    <@91>   DW_AT_name        : (indirect string, offset: 0x9): float
    <@92>   DW_AT_encoding    : 4	(float)
    <@93>   DW_AT_byte_size   : 4
 <1><@98>: Abbrev Number: 6 (DW_TAG_pointer_type)
 <1><@100>: Abbrev Number: 15 (DW_TAG_class_type)
    <@101>   DW_AT_calling_convention: 4	(pass by ref)
    <@102>   DW_AT_name        : (indirect string, offset: 0xa): __func<(lambda at /proj/src/cpp_a.cpp:10:5), int (int)>
    <@103>   DW_AT_byte_size   : 16
    <@104>   DW_AT_decl_file   : 2
 <2><@108>: Abbrev Number: 10 (DW_TAG_member)
    <@109>   DW_AT_name        : (indirect string, offset: 0xb): __func_
    <@10a>   DW_AT_type        : <0x@20>
    <@10b>   DW_AT_decl_file   : 2
    <@10c>   DW_AT_data_member_location: 8
 <2><@110>: Abbrev Number: 0
 <1><@200>: Abbrev Number: 9 (DW_TAG_structure_type)
    <@201>   DW_AT_name        : (indirect string, offset: 0xc): Holder
    <@202>   DW_AT_byte_size   : 8
    <@203>   DW_AT_decl_file   : 0
 <2><@208>: Abbrev Number: 10 (DW_TAG_member)
    <@209>   DW_AT_name        : (indirect string, offset: 0xd): u
    <@20a>   DW_AT_type        : <0x@210>
    <@20b>   DW_AT_decl_file   : 0
    <@20c>   DW_AT_data_member_location: 0
 <2><@210>: Abbrev Number: 11 (DW_TAG_union_type)
    <@211>   DW_AT_byte_size   : 4
    <@212>   DW_AT_decl_file   : 0
 <3><@218>: Abbrev Number: 10 (DW_TAG_member)
    <@219>   DW_AT_name        : (indirect string, offset: 0xe): i
    <@21a>   DW_AT_type        : <0x@80>
    <@21b>   DW_AT_decl_file   : 0
    <@21c>   DW_AT_data_member_location: 0
 <3><@220>: Abbrev Number: 10 (DW_TAG_member)
    <@221>   DW_AT_name        : (indirect string, offset: 0xf): g
    <@222>   DW_AT_type        : <0x@90>
    <@223>   DW_AT_decl_file   : 0
    <@224>   DW_AT_data_member_location: 0
 <3><@228>: Abbrev Number: 0
 <2><@230>: Abbrev Number: 10 (DW_TAG_member)
    <@231>   DW_AT_name        : (indirect string, offset: 0x10): n
    <@232>   DW_AT_type        : <0x@80>
    <@233>   DW_AT_decl_file   : 0
    <@234>   DW_AT_data_member_location: 4
 <2><@238>: Abbrev Number: 0
 <1><@300>: Abbrev Number: 9 (DW_TAG_structure_type)
    <@301>   DW_AT_name        : (indirect string, offset: 0x11): traits
    <@302>   DW_AT_byte_size   : 8
    <@303>   DW_AT_decl_file   : 1
 <2><@308>: Abbrev Number: 10 (DW_TAG_member)
    <@309>   DW_AT_name        : (indirect string, offset: 0x12): size
    <@30a>   DW_AT_type        : <0x@80>
    <@30b>   DW_AT_decl_file   : 1
    <@30c>   DW_AT_data_member_location: 0
 <2><@310>: Abbrev Number: 10 (DW_TAG_member)
    <@311>   DW_AT_name        : (indirect string, offset: 0x13): is_quantized
    <@312>   DW_AT_type        : <0x@88>
    <@313>   DW_AT_decl_file   : 1
    <@314>   DW_AT_data_member_location: 4
 <2><@318>: Abbrev Number: 0
 <1><@400>: Abbrev Number: 9 (DW_TAG_structure_type)
    <@401>   DW_AT_name        : (indirect string, offset: 0x14): block
    <@402>   DW_AT_byte_size   : 8
    <@403>   DW_AT_decl_file   : 1
 <2><@408>: Abbrev Number: 11 (DW_TAG_union_type)
    <@409>   DW_AT_byte_size   : 4
    <@40a>   DW_AT_decl_file   : 1
 <3><@410>: Abbrev Number: 10 (DW_TAG_member)
    <@411>   DW_AT_name        : (indirect string, offset: 0x15): dm
    <@412>   DW_AT_type        : <0x@80>
    <@413>   DW_AT_decl_file   : 1
    <@414>   DW_AT_data_member_location: 0
 <3><@418>: Abbrev Number: 0
 <2><@420>: Abbrev Number: 10 (DW_TAG_member)
    <@421>   DW_AT_name        : (indirect string, offset: 0x16): data
    <@422>   DW_AT_type        : <0x@408>
    <@423>   DW_AT_decl_file   : 1
    <@424>   DW_AT_data_member_location: 0
 <2><@428>: Abbrev Number: 10 (DW_TAG_member)
    <@429>   DW_AT_name        : (indirect string, offset: 0x17): qs
    <@42a>   DW_AT_type        : <0x@80>
    <@42b>   DW_AT_decl_file   : 1
    <@42c>   DW_AT_data_member_location: 4
 <2><@430>: Abbrev Number: 0
 <1><@500>: Abbrev Number: 15 (DW_TAG_class_type)
    <@501>   DW_AT_name        : (indirect string, offset: 0x18): impl
    <@502>   DW_AT_byte_size   : 8
    <@503>   DW_AT_decl_file   : 3
 <2><@508>: Abbrev Number: 10 (DW_TAG_member)
    <@509>   DW_AT_name        : (indirect string, offset: 0x19): n_layer
    <@50a>   DW_AT_type        : <0x@80>
    <@50b>   DW_AT_decl_file   : 3
    <@50c>   DW_AT_data_member_location: 0
 <2><@510>: Abbrev Number: 10 (DW_TAG_member)
    <@511>   DW_AT_name        : (indirect string, offset: 0x1a): n_head
    <@512>   DW_AT_type        : <0x@80>
    <@513>   DW_AT_decl_file   : 3
    <@514>   DW_AT_data_member_location: 4
 <2><@518>: Abbrev Number: 0
 <1><@600>: Abbrev Number: 15 (DW_TAG_class_type)
    <@601>   DW_AT_name        : (indirect string, offset: 0x1b): Dyn
    <@602>   DW_AT_byte_size   : 16
    <@603>   DW_AT_decl_file   : 3
 <2><@608>: Abbrev Number: 10 (DW_TAG_member)
    <@609>   DW_AT_name        : (indirect string, offset: 0x1c): _vptr\$Dyn
    <@60a>   DW_AT_type        : <0x@98>
    <@60b>   DW_AT_artificial  : 1
    <@60c>   DW_AT_data_member_location: 0
 <2><@610>: Abbrev Number: 10 (DW_TAG_member)
    <@611>   DW_AT_name        : (indirect string, offset: 0x1d): x
    <@612>   DW_AT_type        : <0x@80>
    <@613>   DW_AT_decl_file   : 3
    <@614>   DW_AT_data_member_location: 8
 <2><@618>: Abbrev Number: 32 (DW_TAG_member)
    <@619>   DW_AT_name        : (indirect string, offset: 0x1e): __clang_vtable
    <@61a>   DW_AT_type        : <0x@98>
    <@61b>   DW_AT_external    : 1
    <@61c>   DW_AT_declaration : 1
    <@61d>   DW_AT_artificial  : 1
 <2><@620>: Abbrev Number: 0
 <1><@700>: Abbrev Number: 16 (DW_TAG_enumeration_type)
    <@701>   DW_AT_type        : <0x@80>
    <@702>   DW_AT_name        : (indirect string, offset: 0x1f): Mode
    <@703>   DW_AT_byte_size   : 4
    <@704>   DW_AT_decl_file   : 3
 <2><@708>: Abbrev Number: 17 (DW_TAG_enumerator)
    <@709>   DW_AT_name        : (indirect string, offset: 0x20): MODE_X
    <@70a>   DW_AT_const_value : 0
 <2><@70c>: Abbrev Number: 17 (DW_TAG_enumerator)
    <@70d>   DW_AT_name        : (indirect string, offset: 0x21): MODE_Y
    <@70e>   DW_AT_const_value : 1
 <2><@710>: Abbrev Number: 0
 <1><@780>: Abbrev Number: 5 (DW_TAG_typedef)
    <@781>   DW_AT_type        : <0x@80>
    <@782>   DW_AT_name        : (indirect string, offset: 0x22): value_type
    <@783>   DW_AT_decl_file   : 3
 <1><@800>: Abbrev Number: 13 (DW_TAG_variable)
    <@801>   DW_AT_name        : (indirect string, offset: 0x23): inline_flag
    <@802>   DW_AT_type        : <0x@88>
    <@803>   DW_AT_external    : 1
    <@804>   DW_AT_decl_file   : 1
 <1><@810>: Abbrev Number: 0
""")

# A second C++ unit: the shared `traits` again (same spelling as unit A), a
# different `impl`, a different `Mode`, another `value_type`, and `Dyn` outside
# its vtable home (no `__clang_vtable`).
const _ED_CU_B = (files = Dict(0 => "/proj/src/cpp_b.cpp", 1 => "/proj/include/shared.h",
                               2 => "/proj/src/vocab.h", 3 => "/proj/src/model.h"), tpl = """
 <0><@0>: Abbrev Number: 1 (DW_TAG_compile_unit)
    <@1>   DW_AT_producer    : (indirect string, offset: 0x0): clang version 23.1.3
    <@2>   DW_AT_language    : 33	(C++14)
    <@3>   DW_AT_name        : (indirect string, offset: 0x1): cpp_b.cpp
 <1><@10>: Abbrev Number: 2 (DW_TAG_subprogram)
    <@11>   DW_AT_low_pc      : 0x3000
    <@12>   DW_AT_high_pc     : 0x10
    <@13>   DW_AT_linkage_name: (indirect string, offset: 0x2): _Z7b_entryv
    <@14>   DW_AT_name        : (indirect string, offset: 0x3): b_entry
    <@15>   DW_AT_decl_file   : 0
    <@16>   DW_AT_type        : <0x@80>
    <@17>   DW_AT_external    : 1
 <1><@80>: Abbrev Number: 4 (DW_TAG_base_type)
    <@81>   DW_AT_name        : (indirect string, offset: 0x4): int
    <@82>   DW_AT_encoding    : 5	(signed)
    <@83>   DW_AT_byte_size   : 4
 <1><@88>: Abbrev Number: 4 (DW_TAG_base_type)
    <@89>   DW_AT_name        : (indirect string, offset: 0x5): bool
    <@8a>   DW_AT_encoding    : 2	(boolean)
    <@8b>   DW_AT_byte_size   : 1
 <1><@90>: Abbrev Number: 4 (DW_TAG_base_type)
    <@91>   DW_AT_name        : (indirect string, offset: 0x6): double
    <@92>   DW_AT_encoding    : 4	(float)
    <@93>   DW_AT_byte_size   : 8
 <1><@98>: Abbrev Number: 6 (DW_TAG_pointer_type)
 <1><@300>: Abbrev Number: 9 (DW_TAG_structure_type)
    <@301>   DW_AT_name        : (indirect string, offset: 0x7): traits
    <@302>   DW_AT_byte_size   : 8
    <@303>   DW_AT_decl_file   : 1
 <2><@308>: Abbrev Number: 10 (DW_TAG_member)
    <@309>   DW_AT_name        : (indirect string, offset: 0x8): size
    <@30a>   DW_AT_type        : <0x@80>
    <@30b>   DW_AT_decl_file   : 1
    <@30c>   DW_AT_data_member_location: 0
 <2><@310>: Abbrev Number: 10 (DW_TAG_member)
    <@311>   DW_AT_name        : (indirect string, offset: 0x9): is_quantized
    <@312>   DW_AT_type        : <0x@88>
    <@313>   DW_AT_decl_file   : 1
    <@314>   DW_AT_data_member_location: 4
 <2><@318>: Abbrev Number: 0
 <1><@500>: Abbrev Number: 15 (DW_TAG_class_type)
    <@501>   DW_AT_name        : (indirect string, offset: 0xa): impl
    <@502>   DW_AT_byte_size   : 16
    <@503>   DW_AT_decl_file   : 2
 <2><@508>: Abbrev Number: 10 (DW_TAG_member)
    <@509>   DW_AT_name        : (indirect string, offset: 0xb): scores
    <@50a>   DW_AT_type        : <0x@90>
    <@50b>   DW_AT_decl_file   : 2
    <@50c>   DW_AT_data_member_location: 0
 <2><@510>: Abbrev Number: 10 (DW_TAG_member)
    <@511>   DW_AT_name        : (indirect string, offset: 0xc): n_vocab
    <@512>   DW_AT_type        : <0x@80>
    <@513>   DW_AT_decl_file   : 2
    <@514>   DW_AT_data_member_location: 8
 <2><@518>: Abbrev Number: 0
 <1><@600>: Abbrev Number: 15 (DW_TAG_class_type)
    <@601>   DW_AT_name        : (indirect string, offset: 0xd): Dyn
    <@602>   DW_AT_byte_size   : 16
    <@603>   DW_AT_decl_file   : 3
 <2><@608>: Abbrev Number: 10 (DW_TAG_member)
    <@609>   DW_AT_name        : (indirect string, offset: 0xe): _vptr\$Dyn
    <@60a>   DW_AT_type        : <0x@98>
    <@60b>   DW_AT_artificial  : 1
    <@60c>   DW_AT_data_member_location: 0
 <2><@610>: Abbrev Number: 10 (DW_TAG_member)
    <@611>   DW_AT_name        : (indirect string, offset: 0xf): x
    <@612>   DW_AT_type        : <0x@80>
    <@613>   DW_AT_decl_file   : 3
    <@614>   DW_AT_data_member_location: 8
 <2><@618>: Abbrev Number: 0
 <1><@700>: Abbrev Number: 16 (DW_TAG_enumeration_type)
    <@701>   DW_AT_type        : <0x@80>
    <@702>   DW_AT_name        : (indirect string, offset: 0x10): Mode
    <@703>   DW_AT_byte_size   : 4
    <@704>   DW_AT_decl_file   : 2
 <2><@708>: Abbrev Number: 17 (DW_TAG_enumerator)
    <@709>   DW_AT_name        : (indirect string, offset: 0x11): MODE_P
    <@70a>   DW_AT_const_value : 0
 <2><@70c>: Abbrev Number: 17 (DW_TAG_enumerator)
    <@70d>   DW_AT_name        : (indirect string, offset: 0x12): MODE_Q
    <@70e>   DW_AT_const_value : 1
 <2><@710>: Abbrev Number: 17 (DW_TAG_enumerator)
    <@711>   DW_AT_name        : (indirect string, offset: 0x13): MODE_R
    <@712>   DW_AT_const_value : 2
 <2><@714>: Abbrev Number: 0
 <1><@780>: Abbrev Number: 5 (DW_TAG_typedef)
    <@781>   DW_AT_type        : <0x@90>
    <@782>   DW_AT_name        : (indirect string, offset: 0x14): value_type
    <@783>   DW_AT_decl_file   : 2
 <1><@800>: Abbrev Number: 13 (DW_TAG_variable)
    <@801>   DW_AT_name        : (indirect string, offset: 0x15): inline_flag
    <@802>   DW_AT_type        : <0x@88>
    <@803>   DW_AT_external    : 1
    <@804>   DW_AT_decl_file   : 1
 <1><@810>: Abbrev Number: 0
""")

const _ED_CUS = (c = _ED_CU_C, a = _ED_CU_A, b = _ED_CU_B)

# Lay the named units out in `order` from offset `start`, one per 0x1000.
function _ed_assemble(order; start::Integer = 0x1000)
    io = IOBuffer()
    println(io, "Contents of the .debug_info section:\n")
    tables = Dict{Int,String}[]
    for (i, name) in enumerate(order)
        cu = getfield(_ED_CUS, name)
        base = Int(start) + (i - 1) * 0x1000
        println(io, "  Compilation Unit @ offset 0x", string(base; base = 16), ":")
        print(io, replace(cu.tpl, r"@([0-9a-f]+)" =>
                          s -> string(base + parse(Int, s[2:end]; base = 16); base = 16)))
        push!(tables, cu.files)
    end
    return (String(take!(io)), tables)
end

_ed_parse(order; start::Integer = 0x1000) =
    let (dump, tables) = _ed_assemble(order; start)
        _ED_C.parse_dwarf_dump(dump; file_tables = tables)
    end

# Everything parse_dwarf_dump returns, as one comparable text.
_ed_text(parsed) = _ED_C._canonical_text(Any[parsed...])

@testset "Emission does not depend on Dict order, the Julia release or DIE offsets" begin

    # ── DWARF extraction ────────────────────────────────────────────────────
    # The layouts a real link order change and a real upstream edit produce.
    # Under the old last-writer-wins loop these disagree: the winner was the
    # last definition in the hash order of the offset strings.
    layouts = [((:c, :a, :b), 0x1000), ((:c, :a, :b), 0x7a5c0), ((:b, :a, :c), 0x1000),
               ((:a, :c, :b), 0x2340), ((:b, :c, :a), 0x14b30), ((:a, :b, :c), 0xffd000)]
    parsed = [_ed_parse(collect(o); start = s) for (o, s) in layouts]
    rt, sd, gv, td, ra = parsed[1]

    @testset "one answer for every unit order and every offset" begin
        texts = map(_ed_text, parsed)
        @test length(texts) == length(layouts)
        @test all(==(texts[1]), texts)
    end

    @testset "spellings of one type: kept by content, not marked" begin
        # C's `_Bool` and C++'s `bool` (twice); the first in canonical order wins.
        @test [m["c_type"] for m in sd["traits"]["members"]] == ["int", "_Bool"]
        @test !haskey(sd["traits"], "conflict")
        # C's anonymous member and C++'s named one are the same bytes.
        @test [m["name"] for m in sd["block"]["members"]] == ["_anon1", "qs"]
        @test !haskey(sd["block"], "conflict")
        # A vtable home's location-less `__clang_vtable` is not layout.
        @test !haskey(sd["Dyn"], "conflict")
        @test "_vptr\$Dyn" in [m["name"] for m in sd["Dyn"]["members"]]
        # Globals and typedefs: one by content as well.
        @test gv["inline_flag"]["c_type"] == "_Bool"
        @test td["value_type"] == "Cdouble"
    end

    @testset "distinct types sharing a bare name are marked" begin
        @test sd["impl"]["conflict"] == true
        @test sd["impl"]["byte_size"] == "0x10"          # canonical: "0x10" < "0x8"
        @test sd["__enum__Mode"]["conflict"] == true
        @test [e["name"] for e in sd["__enum__Mode"]["enumerators"]] == ["MODE_P", "MODE_Q", "MODE_R"]
    end

    @testset "a lambda's closure goes with the toolchain record that names it" begin
        # `__func<(lambda at …)>` is in a libc++ header; the closure, declared
        # in the project's file, used to be emitted as
        # `_func_lambda_at_proj_src_cpp_a_cpp_10_5_int_int_func`, and the
        # lambda it captures as `…_func_inner`.
        @test !any(k -> occursin("lambda", k) || occursin("func", k), keys(sd))
        # The gate stays narrow: a project record's anonymous union is kept.
        @test haskey(sd, "Holder")
        @test haskey(sd, "Holder_u")
        @test get(sd["Holder_u"], "anonymous", false) == true
        @test [m["name"] for m in sd["Holder_u"]["members"]] == ["i", "g"]
        @test haskey(sd, "block_anon1")
        # Without file tables there is no provenance and nothing is dropped, as before.
        _, sd_nofiles, _, _, _ = _ED_C.parse_dwarf_dump(first(_ed_assemble([:a])))
        @test any(k -> occursin("lambda", k), keys(sd_nofiles))
    end

    @testset "a function-pointer parameter records the typedef it is declared with" begin
        ps = Dict(p["name"] => p for p in rt["run_with"]["parameters"])
        @test ps["cb"]["function_pointer_typedef"] == "progress_cb"
        # Through `typedef progress_cb alias_cb;`: the typedef of the pointer.
        @test ps["again"]["function_pointer_typedef"] == "progress_cb"
        # `int (*raw)(int)`: no typedef, no key.
        @test !haskey(ps["raw"], "function_pointer_typedef")
        @test startswith(ps["raw"]["c_type"], "function_ptr(")
    end

    @testset "_die_order is dump order" begin
        refs = Dict{String,Any}("0x10" => 1, "0x9" => 2, "last_tag_offset" => "0x9",
                                "0xff00" => 3, "0x10000" => 4)
        @test _ED_C._die_order(refs) == ["0x9", "0x10", "0xff00", "0x10000"]
    end

    @testset "_canonical_definition does not follow the order it is given" begin
        mk(ct, nm) = Dict{String,Any}("kind" => "struct", "byte_size" => "0x8",
            "members" => Any[Dict{String,Any}("name" => nm, "c_type" => ct, "julia_type" => "Bool",
                                              "offset" => "0x0", "size" => 1)])
        defs = [mk("bool", "flag"), mk("_Bool", "flag"), mk("bool", "data"), mk("_Bool", "_anon1")]
        picks = Set(_ED_C._canonical_text(_ED_C._canonical_definition(deepcopy(p)))
                    for p in (defs, reverse(defs), defs[[2, 4, 1, 3]], defs[[3, 1, 4, 2]]))
        @test length(picks) == 1
    end

    # ── Generators ──────────────────────────────────────────────────────────
    function _ed_config(lang)
        dir = mktempdir()
        write(joinpath(dir, "replibuild.toml"), """
        [project]
        name = "determinism"
        root = "$(escape_string(dir))"
        [link]
        enable_lto = false
        [wrap]
        language = "$lang"
        [types]
        strictness = "warn"
        allow_unknown_structs = true
        [cache]
        enabled = false
        """)
        return RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml"))
    end
    function _ed_generate(lang, md)
        cfg = _ed_config(lang)
        gen = lang == "cpp" ? _ED_W.generate_introspective_module_cpp : _ED_W.generate_introspective_module_c
        out = gen(cfg, _ED_LIB, deepcopy(md), "Determinism", _ED_W.create_type_registry(cfg), true)
        code = out isa Tuple ? String(first(out)) : String(out)
        # The generation time is the one line that may differ.
        return join(filter(l -> !occursin(r"Generated: |generated_at", l), split(code, '\n')), '\n')
    end
    _ed_row(mangled, demangled, params, ret) = begin
        row = only(_ED_C.parse_function_signatures(
            [Dict{String,Any}("mangled" => mangled, "demangled" => demangled)]))
        row["parameters"] = params
        row["return_type"] = ret
        row["is_noexcept"] = true
        row
    end
    _ed_p(n, jt, ct; kw...) = Dict{String,Any}("name" => n, "julia_type" => jt, "c_type" => ct,
                                                (String(k) => v for (k, v) in kw)...)
    _ed_void = Dict{String,Any}("c_type" => "void", "julia_type" => "Cvoid", "size" => 0)
    _ed_int = Dict{String,Any}("c_type" => "int", "julia_type" => "Cint", "size" => 4)
    _ed_rec(members...) = Dict{String,Any}("kind" => "struct", "byte_size" => "0x4",
        "members" => Any[Dict{String,Any}("name" => m, "c_type" => "int", "julia_type" => "Cint",
                                          "offset" => "0x0", "size" => 4) for m in members])

    classes = ["Zeta", "Alpha", "Mid", "Omega", "Beta", "Kappa", "Delta", "Gamma", "Lambda", "Epsilon"]

    # Callbacks: `handler` is declared with `progress_cb`, `other` with no
    # typedef. `handler` (typedef) is the decoy the old scoring picked for both:
    # its name is the parameter's, and the function's name contains it.
    callback_fn = _ed_row("register_handler", "register_handler",
        Any[_ed_p("handler", "Ptr{Cvoid}", "function_ptr(int)*";
                  function_pointer_signature = "function_ptr(int)*",
                  function_pointer_typedef = "progress_cb"),
            _ed_p("other", "Ptr{Cvoid}", "function_ptr(void)*";
                  function_pointer_signature = "function_ptr(void)*")], _ed_void)
    fp_typedefs = Dict{String,Any}(
        "handler"     => Dict{String,Any}("return_type" => "double", "parameters" => Any["double"]),
        "progress_cb" => Dict{String,Any}("return_type" => "int", "parameters" => Any["int", "void*"]),
        "register"    => Dict{String,Any}("return_type" => "float", "parameters" => Any["float"]))

    function _ed_metadata(lang)
        structs = Dict{String,Any}(c => _ed_rec("v") for c in classes)
        for (i, e) in enumerate(["Color", "Shape", "Axis", "Unit", "Phase", "Kind"])
            structs["__enum__" * e] = Dict{String,Any}("kind" => "enum", "underlying_type" => "int",
                "julia_type" => "Cint", "byte_size" => "0x4",
                "enumerators" => Any[Dict{String,Any}("name" => "$(uppercase(e))_A", "value" => 0),
                                     Dict{String,Any}("name" => "$(uppercase(e))_B", "value" => i)])
        end
        fns = Any[callback_fn]
        if lang == "cpp"
            # A deleting destructor each: a class whose destructors only end its
            # lifetime gets no Managed type (test_cpp_finalizers.jl).
            for c in classes
                push!(fns, _ed_row("_ZN$(length(c))$(c)D0Ev", "$c::~$c()",
                                   Any[_ed_p("this", "Ptr{$c}", "$c*")], _ed_void))
            end
        end
        Dict{String,Any}(
            "functions" => fns,
            "struct_definitions" => structs,
            "globals" => Dict{String,Any}(g => Dict{String,Any}("name" => g, "c_type" => "int", "julia_type" => "Cint")
                                          for g in ["g_zeta", "g_alpha", "g_mid", "g_omega", "g_beta", "g_kappa"]),
            "header_enums" => Dict{String,Any}(h => Any[("$(h)_ONE", 1), ("$(h)_TWO", 2)]
                                               for h in ["hz", "ha", "hm", "ho", "hb", "hk"]),
            "function_pointer_typedefs" => fp_typedefs,
            "function_pointer_parameters" => true,
            "timestamp" => "2026-10-22T00:00:00")
    end

    # Every Dict rebuilt: keys inserted in reverse, at another capacity. That is
    # what reading the same metadata on another Julia release does to it.
    _ed_reshuffle(x) = x
    _ed_reshuffle(x::AbstractVector) = Any[_ed_reshuffle(v) for v in x]
    function _ed_reshuffle(d::AbstractDict)
        out = Dict{String,Any}()
        sizehint!(out, 7 * length(d) + 31)
        for k in reverse(sort!(collect(keys(d)); by = string))
            out[k] = _ed_reshuffle(d[k])
        end
        out
    end

    for lang in ("c", "cpp")
        md = _ed_metadata(lang)
        code = _ed_generate(lang, md)

        @testset "$lang: wrapper text does not follow metadata key order" begin
            @test _ed_generate(lang, _ed_reshuffle(md)) == code
        end

        @testset "$lang: globals and header enums are emitted sorted" begin
            gpos = [something(findfirst("function $g()", code), 0:0)[1] for g in sort(collect(keys(md["globals"])))]
            @test all(>(0), gpos)
            @test issorted(gpos)
            hpos = [something(findfirst("@enum $h::", code), 0:0)[1] for h in sort(collect(keys(md["header_enums"])))]
            @test all(>(0), hpos)
            @test issorted(hpos)
        end

        @testset "$lang: a callback's example comes from its own typedef, or from DWARF" begin
            @test occursin("**Callback `handler`**", code)
            @test occursin("@cfunction(my_callback, Cint, (Cint, Ptr{Cvoid},))", code)
            # Never the decoy's signature. `other` has no typedef, and its
            # example is its own DWARF signature's.
            @test !occursin("Cdouble, (Cdouble,)", code)
            @test !occursin("Cfloat, (Cfloat,)", code)
            @test occursin("**Callback `other`**", code)
            @test occursin("@cfunction(my_callback, Cvoid, ())", code)
        end
    end

    @testset "cpp: Managed finalizer types are emitted sorted" begin
        code = _ed_generate("cpp", _ed_metadata("cpp"))
        managed = [m.captures[1] for m in eachmatch(r"\nmutable struct Managed(\w+)\n", code)]
        @test sort(managed) == sort(classes)
        @test managed == sort(classes)
    end

    @testset "a callback without typedef metadata has no typedef signature" begin
        p = _ed_p("cb", "Ptr{Cvoid}", "function_ptr(int)*"; function_pointer_typedef = "progress_cb")
        @test _ED_W._callback_typedef_signature(p, Dict{String,Any}("function_pointer_typedefs" => fp_typedefs)) ==
              "function_ptr(int; int, void*)"
        @test _ED_W._callback_typedef_signature(p, Dict{String,Any}()) === nothing
        @test _ED_W._callback_typedef_signature(_ed_p("cb", "Ptr{Cvoid}", "function_ptr(int)*"),
                                                Dict{String,Any}("function_pointer_typedefs" => fp_typedefs)) === nothing
    end

    # ── Fuzzy record lookups, thunk side and wrapper side ───────────────────
    @testset "fuzzy record lookups pick the smallest key, whatever the Dict order" begin
        spellings = ["Box<int>>>", "Box<int> > >", "Box<int>  >", "Box<int>> >", "Box<int> >>"]
        expect = minimum(spellings)
        answers = Set{Any}()
        for cap in (0, 5, 17, 64, 333, 1024), order in (spellings, reverse(spellings))
            d = Dict{String,Any}()
            sizehint!(d, cap)
            for k in order; d[k] = Dict{String,Any}(); end
            push!(answers, _ED_F._fuzzy_struct_lookup("Box<int>", d))
            push!(answers, _ED_W._fuzzy_dwarf_lookup("Box<int>", d))
        end
        @test answers == Set{Any}([expect])
    end

    @testset "StructGen: alias order and enum-or-record do not follow the Dict" begin
        rec(n) = Dict{String,Any}("kind" => "struct", "byte_size" => "0x8",
            "members" => Any[Dict{String,Any}("name" => "a", "c_type" => "int", "offset" => "0x0", "size" => 4),
                             Dict{String,Any}("name" => "b", "c_type" => "int", "offset" => "0x4", "size" => 4)])
        names = ["mode", "Shape", "zed", "alpha", "Mid"]
        texts = Set{Any}()
        for cap in (0, 9, 100, 777), order in (names, reverse(names))
            d = Dict{String,Any}()
            sizehint!(d, cap)
            for n in order
                d[n] = rec(n)
                # An enum sharing two of the bare names.
                n in ("mode", "Shape") && (d["__enum__" * n] = Dict{String,Any}(
                    "kind" => "enum", "underlying_type" => "int", "byte_size" => "0x4"))
            end
            push!(texts, _ED_S.generate_struct_definitions(d))
        end
        @test length(texts) == 1
        aliases = first(first(texts))
        # The enum's entry describes the shared alias, for both spellings of the
        # name relative to `__enum__` in sort order ('S' < '_' < 'm').
        @test occursin("!Struct_mode = !llvm.struct<\"mode\", (i32)>", aliases)
        @test occursin("!Struct_Shape = !llvm.struct<\"Shape\", (i32)>", aliases)
    end

    # ── Persisted keys ──────────────────────────────────────────────────────
    @testset "IR file names are SHA-256 of the source directory" begin
        # Golden values: SHA-256 is the same on every Julia release, which is
        # the point. `Base.hash` gave 1.12 and 1.13 different names.
        @test _ED_C._ir_dir_key("/rb/src") == "4d551ce2"
        src = abspath(joinpath(mktempdir(), "src", "ggml.c"))
        @test _ED_C._ir_output_path("/b", src) ==
              joinpath("/b", "ggml.c-" * _ED_C._ir_dir_key(dirname(src)) * ".ll")
        if Sys.iswindows()
            # One directory, two spellings, one name.
            @test _ED_C._ir_dir_key("C:\\rb\\src") == _ED_C._ir_dir_key("c:/rb/src")
        else
            @test _ED_C._ir_dir_key("/RB/src") != _ED_C._ir_dir_key("/rb/src")
        end
        # The templates TU's `.key` fingerprint, persisted beside its IR.
        @test _ED_C._file_compile_fingerprint("abc", "replibuild_templates.cpp") ==
              "b95d6f3b96794a060c8948621b63b79b9246586272365d64642c2accd4d2b904"
    end
end
