#!/usr/bin/env julia
# Attribute-packed and `#pragma pack` C++ structs: every layer holds the C layout.
#
# A packed `{char; int}` is 5 bytes with the int at offset 1. GeneratorCpp
# emitted it as Julia fields at natural alignment (8 bytes, int at 4) and the
# thunk converted between that and the C layout — so by-value calls worked
# while every read through a pointer, every array of them, and every struct
# embedding one was wrong. A `#pragma pack(2)` struct has padding, so it was not
# "packed" to StructGen at all: it degraded to an `!llvm.array<14 x i8>`, which
# the SysV classifier never saw as a struct, and broke by value too.
#
# Now: the wrapper proves a struct's Julia layout against DWARF and emits a
# C-layout byte blob when it cannot (`_prove_julia_layout`); the thunk reads
# Julia's value at DWARF offsets and returns a packed value as it is; StructGen
# models a struct natural alignment cannot place as a PACKED body with explicit
# pads, so the classifier sees the misaligned field. No toolchain: verbatim
# clang 22 readelf and text.

using Test
using RepliBuild

const PLC = RepliBuild.Compiler
const PLF = RepliBuild.JLCSIRGenerator.FunctionGen
const PLS = RepliBuild.JLCSIRGenerator.StructGen
const PLW = RepliBuild.Wrapper

# clang 22.1.8, `clang++ -std=c++17 -g -O1 -fPIC -shared` of:
#   namespace pk {
#   struct Pk { char c; int i; } __attribute__((packed));          // 5B: 0, 1
#   #pragma pack(push, 2)
#   struct P2 { char c; int i; double d; };                          // 14B: 0, 2, 6
#   #pragma pack(pop)
#   struct Holder { Pk p; double d; };                               // 16B: 0, 8
#   struct Tagged { char tag; char data[3]; int v; };                // 8B: 0, 1, 4
#   Pk mk_Pk(int);  double sum_Pk(Pk);  P2 mk_P2(int);
#   double sum_Holder(Holder);  int tagged_sum(Tagged);
#   }
const PACKED_DUMP = raw"""
Contents of the .debug_info section:

  Compilation Unit @ offset 0:
   Length:        0x14e (32-bit)
   Version:       5
   Unit Type:     DW_UT_compile (1)
   Abbrev Offset: 0
   Pointer Size:  8
 <0><c>: Abbrev Number: 1 (DW_TAG_compile_unit)
    <d>   DW_AT_producer    : (indexed string: 0): clang version 22.1.8
    <e>   DW_AT_language    : 33	(C++14)
    <10>   DW_AT_name        : (indexed string: 0x1): pk.cpp
    <11>   DW_AT_str_offsets_base: 0x8
    <15>   DW_AT_stmt_list   : 0
    <19>   DW_AT_comp_dir    : (indexed string: 0x2): <dir>
    <1a>   DW_AT_low_pc      : (index: 0): 0x10f0
    <1b>   DW_AT_high_pc     : 0x87
    <1f>   DW_AT_addr_base   : 0x8
    <23>   DW_AT_loclists_base: 0xc (location list)
 <1><27>: Abbrev Number: 2 (DW_TAG_base_type)
    <28>   DW_AT_name        : (indexed string: 0x3): char
    <29>   DW_AT_encoding    : 6	(signed char)
    <2a>   DW_AT_byte_size   : 1
 <1><2b>: Abbrev Number: 3 (DW_TAG_namespace)
    <2c>   DW_AT_name        : (indexed string: 0x4): pk
 <2><2d>: Abbrev Number: 4 (DW_TAG_subprogram)
    <2e>   DW_AT_low_pc      : (index: 0): 0x10f0
    <2f>   DW_AT_high_pc     : 0xc
    <33>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <35>   DW_AT_call_all_calls: 1
    <35>   DW_AT_linkage_name: (indexed string: 0x5): _ZN2pk5mk_PkEi
    <36>   DW_AT_name        : (indexed string: 0x6): mk_Pk
    <37>   DW_AT_decl_file   : 0
    <38>   DW_AT_decl_line   : 8
    <39>   DW_AT_type        : <0xc1>
    <3d>   DW_AT_external    : 1
 <3><3d>: Abbrev Number: 5 (DW_TAG_formal_parameter)
    <3e>   DW_AT_name        : (indexed string: 0x16): a
    <3f>   DW_AT_decl_file   : 0
    <40>   DW_AT_decl_line   : 8
    <41>   DW_AT_type        : <0x139>
 <3><45>: Abbrev Number: 6 (DW_TAG_variable)
    <46>   DW_AT_name        : (indexed string: 0x17): p
    <47>   DW_AT_decl_file   : 0
    <48>   DW_AT_decl_line   : 8
    <49>   DW_AT_type        : <0xc1>
 <3><4d>: Abbrev Number: 0
 <2><4e>: Abbrev Number: 4 (DW_TAG_subprogram)
    <4f>   DW_AT_low_pc      : (index: 0x1): 0x1100
    <50>   DW_AT_high_pc     : 0xe
    <54>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <56>   DW_AT_call_all_calls: 1
    <56>   DW_AT_linkage_name: (indexed string: 0xb): _ZN2pk6sum_PkENS_2PkE
    <57>   DW_AT_name        : (indexed string: 0xc): sum_Pk
    <58>   DW_AT_decl_file   : 0
    <59>   DW_AT_decl_line   : 9
    <5a>   DW_AT_type        : <0x13d>
    <5e>   DW_AT_external    : 1
 <3><5e>: Abbrev Number: 7 (DW_TAG_formal_parameter)
    <5f>   DW_AT_location    : 2 byte block: 91 8 	(DW_OP_fbreg: 8)
    <62>   DW_AT_name        : (indexed string: 0x18): s
    <63>   DW_AT_decl_file   : 0
    <64>   DW_AT_decl_line   : 9
    <65>   DW_AT_type        : <0xc1>
 <3><69>: Abbrev Number: 0
 <2><6a>: Abbrev Number: 4 (DW_TAG_subprogram)
    <6b>   DW_AT_low_pc      : (index: 0x2): 0x1110
    <6c>   DW_AT_high_pc     : 0x1e
    <70>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <72>   DW_AT_call_all_calls: 1
    <72>   DW_AT_linkage_name: (indexed string: 0xe): _ZN2pk5mk_P2Ei
    <73>   DW_AT_name        : (indexed string: 0xf): mk_P2
    <74>   DW_AT_decl_file   : 0
    <75>   DW_AT_decl_line   : 10
    <76>   DW_AT_type        : <0xda>
    <7a>   DW_AT_external    : 1
 <3><7a>: Abbrev Number: 5 (DW_TAG_formal_parameter)
    <7b>   DW_AT_name        : (indexed string: 0x16): a
    <7c>   DW_AT_decl_file   : 0
    <7d>   DW_AT_decl_line   : 10
    <7e>   DW_AT_type        : <0x139>
 <3><82>: Abbrev Number: 6 (DW_TAG_variable)
    <83>   DW_AT_name        : (indexed string: 0x17): p
    <84>   DW_AT_decl_file   : 0
    <85>   DW_AT_decl_line   : 10
    <86>   DW_AT_type        : <0xda>
 <3><8a>: Abbrev Number: 0
 <2><8b>: Abbrev Number: 4 (DW_TAG_subprogram)
    <8c>   DW_AT_low_pc      : (index: 0x3): 0x1130
    <8d>   DW_AT_high_pc     : 0x14
    <91>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <93>   DW_AT_call_all_calls: 1
    <93>   DW_AT_linkage_name: (indexed string: 0x12): _ZN2pk10sum_HolderENS_6HolderE
    <94>   DW_AT_name        : (indexed string: 0x13): sum_Holder
    <95>   DW_AT_decl_file   : 0
    <96>   DW_AT_decl_line   : 11
    <97>   DW_AT_type        : <0x13d>
    <9b>   DW_AT_external    : 1
 <3><9b>: Abbrev Number: 7 (DW_TAG_formal_parameter)
    <9c>   DW_AT_location    : 2 byte block: 91 8 	(DW_OP_fbreg: 8)
    <9f>   DW_AT_name        : (indexed string: 0x19): h
    <a0>   DW_AT_decl_file   : 0
    <a1>   DW_AT_decl_line   : 11
    <a2>   DW_AT_type        : <0xfd>
 <3><a6>: Abbrev Number: 0
 <2><a7>: Abbrev Number: 4 (DW_TAG_subprogram)
    <a8>   DW_AT_low_pc      : (index: 0x4): 0x1150
    <a9>   DW_AT_high_pc     : 0x27
    <ad>   DW_AT_frame_base  : 1 byte block: 57 	(DW_OP_reg7 (rsp))
    <af>   DW_AT_call_all_calls: 1
    <af>   DW_AT_linkage_name: (indexed string: 0x14): _ZN2pk10tagged_sumENS_6TaggedE
    <b0>   DW_AT_name        : (indexed string: 0x15): tagged_sum
    <b1>   DW_AT_decl_file   : 0
    <b2>   DW_AT_decl_line   : 12
    <b3>   DW_AT_type        : <0x139>
    <b7>   DW_AT_external    : 1
 <3><b7>: Abbrev Number: 8 (DW_TAG_formal_parameter)
    <b8>   DW_AT_location    : (index: 0): 0x10 (location list)
    <b9>   DW_AT_name        : (indexed string: 0x1b): t
    <ba>   DW_AT_decl_file   : 0
    <bb>   DW_AT_decl_line   : 12
    <bc>   DW_AT_type        : <0x116>
 <3><c0>: Abbrev Number: 0
 <2><c1>: Abbrev Number: 9 (DW_TAG_structure_type)
    <c2>   DW_AT_calling_convention: 5	(pass by value)
    <c3>   DW_AT_name        : (indexed string: 0xa): Pk
    <c4>   DW_AT_byte_size   : 5
    <c5>   DW_AT_decl_file   : 0
    <c6>   DW_AT_decl_line   : 2
 <3><c7>: Abbrev Number: 10 (DW_TAG_member)
    <c8>   DW_AT_name        : (indexed string: 0x7): c
    <c9>   DW_AT_type        : <0x27>
    <cd>   DW_AT_decl_file   : 0
    <ce>   DW_AT_decl_line   : 2
    <cf>   DW_AT_data_member_location: 0
 <3><d0>: Abbrev Number: 10 (DW_TAG_member)
    <d1>   DW_AT_name        : (indexed string: 0x8): i
    <d2>   DW_AT_type        : <0x139>
    <d6>   DW_AT_decl_file   : 0
    <d7>   DW_AT_decl_line   : 2
    <d8>   DW_AT_data_member_location: 1
 <3><d9>: Abbrev Number: 0
 <2><da>: Abbrev Number: 11 (DW_TAG_structure_type)
    <db>   DW_AT_calling_convention: 5	(pass by value)
    <dc>   DW_AT_name        : (indexed string: 0x11): P2
    <dd>   DW_AT_byte_size   : 14
    <de>   DW_AT_decl_file   : 0
    <df>   DW_AT_decl_line   : 4
    <e0>   DW_AT_alignment   : 2
 <3><e1>: Abbrev Number: 10 (DW_TAG_member)
    <e2>   DW_AT_name        : (indexed string: 0x7): c
    <e3>   DW_AT_type        : <0x27>
    <e7>   DW_AT_decl_file   : 0
    <e8>   DW_AT_decl_line   : 4
    <e9>   DW_AT_data_member_location: 0
 <3><ea>: Abbrev Number: 10 (DW_TAG_member)
    <eb>   DW_AT_name        : (indexed string: 0x8): i
    <ec>   DW_AT_type        : <0x139>
    <f0>   DW_AT_decl_file   : 0
    <f1>   DW_AT_decl_line   : 4
    <f2>   DW_AT_data_member_location: 2
 <3><f3>: Abbrev Number: 10 (DW_TAG_member)
    <f4>   DW_AT_name        : (indexed string: 0x10): d
    <f5>   DW_AT_type        : <0x13d>
    <f9>   DW_AT_decl_file   : 0
    <fa>   DW_AT_decl_line   : 4
    <fb>   DW_AT_data_member_location: 6
 <3><fc>: Abbrev Number: 0
 <2><fd>: Abbrev Number: 9 (DW_TAG_structure_type)
    <fe>   DW_AT_calling_convention: 5	(pass by value)
    <ff>   DW_AT_name        : (indexed string: 0x1a): Holder
    <100>   DW_AT_byte_size   : 16
    <101>   DW_AT_decl_file   : 0
    <102>   DW_AT_decl_line   : 6
 <3><103>: Abbrev Number: 10 (DW_TAG_member)
    <104>   DW_AT_name        : (indexed string: 0x17): p
    <105>   DW_AT_type        : <0xc1>
    <109>   DW_AT_decl_file   : 0
    <10a>   DW_AT_decl_line   : 6
    <10b>   DW_AT_data_member_location: 0
 <3><10c>: Abbrev Number: 10 (DW_TAG_member)
    <10d>   DW_AT_name        : (indexed string: 0x10): d
    <10e>   DW_AT_type        : <0x13d>
    <112>   DW_AT_decl_file   : 0
    <113>   DW_AT_decl_line   : 6
    <114>   DW_AT_data_member_location: 8
 <3><115>: Abbrev Number: 0
 <2><116>: Abbrev Number: 9 (DW_TAG_structure_type)
    <117>   DW_AT_calling_convention: 5	(pass by value)
    <118>   DW_AT_name        : (indexed string: 0x20): Tagged
    <119>   DW_AT_byte_size   : 8
    <11a>   DW_AT_decl_file   : 0
    <11b>   DW_AT_decl_line   : 7
 <3><11c>: Abbrev Number: 10 (DW_TAG_member)
    <11d>   DW_AT_name        : (indexed string: 0x1c): tag
    <11e>   DW_AT_type        : <0x27>
    <122>   DW_AT_decl_file   : 0
    <123>   DW_AT_decl_line   : 7
    <124>   DW_AT_data_member_location: 0
 <3><125>: Abbrev Number: 10 (DW_TAG_member)
    <126>   DW_AT_name        : (indexed string: 0x1d): data
    <127>   DW_AT_type        : <0x141>
    <12b>   DW_AT_decl_file   : 0
    <12c>   DW_AT_decl_line   : 7
    <12d>   DW_AT_data_member_location: 1
 <3><12e>: Abbrev Number: 10 (DW_TAG_member)
    <12f>   DW_AT_name        : (indexed string: 0x1f): v
    <130>   DW_AT_type        : <0x139>
    <134>   DW_AT_decl_file   : 0
    <135>   DW_AT_decl_line   : 7
    <136>   DW_AT_data_member_location: 4
 <3><137>: Abbrev Number: 0
 <2><138>: Abbrev Number: 0
 <1><139>: Abbrev Number: 2 (DW_TAG_base_type)
    <13a>   DW_AT_name        : (indexed string: 0x9): int
    <13b>   DW_AT_encoding    : 5	(signed)
    <13c>   DW_AT_byte_size   : 4
 <1><13d>: Abbrev Number: 2 (DW_TAG_base_type)
    <13e>   DW_AT_name        : (indexed string: 0xd): double
    <13f>   DW_AT_encoding    : 4	(float)
    <140>   DW_AT_byte_size   : 8
 <1><141>: Abbrev Number: 12 (DW_TAG_array_type)
    <142>   DW_AT_type        : <0x27>
 <2><146>: Abbrev Number: 13 (DW_TAG_subrange_type)
    <147>   DW_AT_type        : <0x14d>
    <14b>   DW_AT_count       : 3
 <2><14c>: Abbrev Number: 0
 <1><14d>: Abbrev Number: 14 (DW_TAG_base_type)
    <14e>   DW_AT_name        : (indexed string: 0x1e): __ARRAY_SIZE_TYPE__
    <14f>   DW_AT_byte_size   : 8
    <150>   DW_AT_encoding    : 7	(unsigned)
 <1><151>: Abbrev Number: 0

"""

@testset "Packed C++ structs hold the C layout" begin
    rt, sd, _, _, ra = PLC.parse_dwarf_dump(PACKED_DUMP)
    @test sort(collect(keys(sd))) == ["Holder", "P2", "Pk", "Tagged"]   # parsed at all

    @testset "_prove_julia_layout" begin
        L = Dict{String,Tuple{Int,Int}}(); M = Set{String}()
        prove(plan, n) = PLW._prove_julia_layout(plan, n, L, M)
        # Julia puts Cint at 4; C put it at 1.
        @test prove([("UInt8", 0), ("Cint", 1)], 5) === :mismatch
        # #pragma pack(2): a 1-byte pad cannot move a Cint to 2.
        @test prove([("UInt8", 0), ("NTuple{1, UInt8}", nothing), ("Cint", 2), ("Cdouble", 6)], 14) === :mismatch
        # Padding-free with a byte array: every field lands.
        @test prove([("UInt8", 0), ("NTuple{3, UInt8}", 1), ("Cint", 4)], 8) == (8, 4)
        # Holder, once Pk is a 5-byte blob (alignment 1): exact.
        L["Pk"] = (5, 1)
        @test prove([("Pk", 0), ("NTuple{3, UInt8}", nothing), ("Cdouble", 8)], 16) == (16, 8)
        # A named union is `mutable`: as a field it is a pointer, not 16 bytes.
        push!(M, "U")
        @test prove([("U", 0)], 16) === :mismatch
        # Something unmeasurable keeps the previous behaviour.
        @test prove([("Mystery", 0)], 8) === :unknown
    end

    @testset "StructGen: a struct natural alignment cannot place is a packed body" begin
        body = PLS.get_struct_definition_string("P2", sd["P2"], sd)
        @test body == "!llvm.struct<\"P2\", packed (i8, !llvm.array<1 x i8>, i32, f64)>"
        @test PLS._mlir_layout(body, sd) == (14, 1)
        # Still a byte region when no body can model it (overlap).
        ov = Dict{String,Any}("kind" => "struct", "byte_size" => "0x10", "members" => [
            Dict{String,Any}("name" => "a", "c_type" => "long", "offset" => "0x0", "size" => 8),
            Dict{String,Any}("name" => "b", "c_type" => "long", "offset" => "0x0", "size" => 8),
            Dict{String,Any}("name" => "c", "c_type" => "long", "offset" => "0x8", "size" => 8)])
        @test PLS.get_struct_definition_string("Ov", ov, Dict{String,Any}("Ov" => ov)) == "!llvm.array<16 x i8>"
    end

    fn(m, params, ret) = Dict{String,Any}("name" => m, "mangled" => m, "is_method" => false,
        "demangled" => m,
        "parameters" => [Dict{String,Any}("name" => "a$i", "c_type" => c) for (i, c) in enumerate(params)],
        "return_type" => Dict{String,Any}("c_type" => ret))
    gen(f) = PLF.generate_function_thunks([f], sd; may_throw = true, record_abi = ra)

    @testset "thunks read the Julia value at DWARF offsets" begin
        ir = gen(fn("_ZN2pk6sum_PkENS_2PkE", ["Pk"], "double"))
        @test occursin("juliaOffsets = [0 : i64, 1 : i64]", ir)
        ir = gen(fn("_ZN2pk10tagged_sumENS_6TaggedE", ["Tagged"], "int"))
        @test occursin("juliaOffsets = [0 : i64, 1 : i64, 4 : i64]", ir)   # was 0, 3, 8
    end

    @testset "a truly packed return goes back as the packed value" begin
        ir = gen(fn("_ZN2pk5mk_PkEi", ["int"], "Pk"))
        @test !occursin("jlcs.marshal_ret", ir)
        @test occursin(r"func\.func @_ZN2pk5mk_PkEi_thunk\(%args_ptr: !llvm\.ptr\) -> !llvm\.struct<packed \(i8, i32\)\s*>", ir)
        # #pragma pack(2): a packed body the classifier can see, not an array.
        ir = gen(fn("_ZN2pk5mk_P2Ei", ["int"], "P2"))
        @test occursin("-> !llvm.struct<\"P2\", packed (i8, !llvm.array<1 x i8>, i32, f64)>", ir)
        @test !occursin("!llvm.array<14 x i8>", ir)
        # Padding-free and naturally aligned: unchanged (aligned == packed).
        ir = gen(fn("_ZN2pk10mk_TaggedEi", ["int"], "Tagged"))
        @test occursin("jlcs.marshal_ret", ir)
    end
end
