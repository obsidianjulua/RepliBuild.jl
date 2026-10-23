#!/usr/bin/env julia
# A by-value crossing of a name several types share is a trap (2026-10-23).
#
# Types are keyed by their bare DWARF name, and `_canonical_definition` keeps one
# definition per name, marking the entry `conflict` when the definitions are
# distinct types (a pimpl `impl` per class). Every reference by that name gets
# the kept layout, and a by-value crossing got it too: `by_value_crossing` asked
# the struct table first, so the thunk laid out the kept record, GeneratorCpp's
# ccall passed its Julia struct, and GeneratorC never asked at all. A function
# taking one `impl` by value would have been called with the other's layout.
# Measured on 26 Hub packages and 13 devtests fixtures before this change: none
# did yet.
#
# Now a crossing of a conflicted record, or of a record holding one by value at
# any depth, is a trap in the thunk generator, both Julia generators and the
# shared varargs path. An enum crosses as an integer of its width, so it traps
# only when its definitions differ in width, which the parser now records as
# `conflict_sizes`. Synthetic DWARF for two units, text output, no toolchain.

using Test
using RepliBuild
using Libdl

const _CN_C = RepliBuild.Compiler
const _CN_F = RepliBuild.JLCSIRGenerator.FunctionGen
const _CN_W = RepliBuild.Wrapper
const _CN_LIB = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))

# ── Two units, each with its own `impl`, `Mode` and `Kind` ───────────────────
#
# Offsets are `@<hex>` from the unit's start (as in test_emission_determinism).
# Unit A: `impl` {int n_layer; int n_head} (8 bytes), `holder` {impl i; int k},
# `Mode` {MODE_X, MODE_Y} (4 bytes), `Kind : unsigned char` (1 byte), and the
# shared `Pair`. Unit B: `impl` {double scores; int n_vocab} (16 bytes), `Mode`
# {MODE_P, MODE_Q, MODE_R} (4 bytes), `Kind : int` (4 bytes), `Pair` again.

const _CN_UNIT_A = """
 <0><@0>: Abbrev Number: 1 (DW_TAG_compile_unit)
    <@1>   DW_AT_producer    : (indirect string, offset: 0x0): clang version 23.1.3
    <@2>   DW_AT_language    : 33	(C++14)
    <@3>   DW_AT_name        : (indirect string, offset: 0x1): a.cpp
 <1><@10>: Abbrev Number: 2 (DW_TAG_subprogram)
    <@11>   DW_AT_low_pc      : 0x1000
    <@12>   DW_AT_high_pc     : 0x10
    <@13>   DW_AT_linkage_name: (indirect string, offset: 0x2): _Z10takes_impl4impl
    <@14>   DW_AT_name        : (indirect string, offset: 0x3): takes_impl
    <@15>   DW_AT_type        : <0x@80>
    <@16>   DW_AT_external    : 1
 <2><@18>: Abbrev Number: 3 (DW_TAG_formal_parameter)
    <@19>   DW_AT_name        : (indirect string, offset: 0x4): v
    <@1a>   DW_AT_type        : <0x@100>
 <2><@1c>: Abbrev Number: 0
 <1><@20>: Abbrev Number: 2 (DW_TAG_subprogram)
    <@21>   DW_AT_low_pc      : 0x1010
    <@22>   DW_AT_high_pc     : 0x10
    <@23>   DW_AT_linkage_name: (indirect string, offset: 0x5): _Z12takes_holder6holder
    <@24>   DW_AT_name        : (indirect string, offset: 0x6): takes_holder
    <@25>   DW_AT_type        : <0x@80>
    <@26>   DW_AT_external    : 1
 <2><@28>: Abbrev Number: 3 (DW_TAG_formal_parameter)
    <@29>   DW_AT_name        : (indirect string, offset: 0x7): h
    <@2a>   DW_AT_type        : <0x@120>
 <2><@2c>: Abbrev Number: 0
 <1><@30>: Abbrev Number: 2 (DW_TAG_subprogram)
    <@31>   DW_AT_low_pc      : 0x1020
    <@32>   DW_AT_high_pc     : 0x10
    <@33>   DW_AT_linkage_name: (indirect string, offset: 0x8): _Z10takes_pair4Pair
    <@34>   DW_AT_name        : (indirect string, offset: 0x9): takes_pair
    <@35>   DW_AT_type        : <0x@80>
    <@36>   DW_AT_external    : 1
 <2><@38>: Abbrev Number: 3 (DW_TAG_formal_parameter)
    <@39>   DW_AT_name        : (indirect string, offset: 0xa): p
    <@3a>   DW_AT_type        : <0x@140>
 <2><@3c>: Abbrev Number: 0
 <1><@40>: Abbrev Number: 2 (DW_TAG_subprogram)
    <@41>   DW_AT_low_pc      : 0x1030
    <@42>   DW_AT_high_pc     : 0x10
    <@43>   DW_AT_linkage_name: (indirect string, offset: 0xb): _Z10takes_mode4Mode
    <@44>   DW_AT_name        : (indirect string, offset: 0xc): takes_mode
    <@45>   DW_AT_type        : <0x@80>
    <@46>   DW_AT_external    : 1
 <2><@48>: Abbrev Number: 3 (DW_TAG_formal_parameter)
    <@49>   DW_AT_name        : (indirect string, offset: 0xd): m
    <@4a>   DW_AT_type        : <0x@160>
 <2><@4c>: Abbrev Number: 0
 <1><@50>: Abbrev Number: 2 (DW_TAG_subprogram)
    <@51>   DW_AT_low_pc      : 0x1040
    <@52>   DW_AT_high_pc     : 0x10
    <@53>   DW_AT_linkage_name: (indirect string, offset: 0xe): _Z14takes_impl_ptrP4impl
    <@54>   DW_AT_name        : (indirect string, offset: 0xf): takes_impl_ptr
    <@55>   DW_AT_type        : <0x@80>
    <@56>   DW_AT_external    : 1
 <2><@58>: Abbrev Number: 3 (DW_TAG_formal_parameter)
    <@59>   DW_AT_name        : (indirect string, offset: 0xa): p
    <@5a>   DW_AT_type        : <0x@98>
 <2><@5c>: Abbrev Number: 0
 <1><@80>: Abbrev Number: 4 (DW_TAG_base_type)
    <@81>   DW_AT_name        : (indirect string, offset: 0x10): int
    <@82>   DW_AT_encoding    : 5	(signed)
    <@83>   DW_AT_byte_size   : 4
 <1><@88>: Abbrev Number: 4 (DW_TAG_base_type)
    <@89>   DW_AT_name        : (indirect string, offset: 0x11): unsigned char
    <@8a>   DW_AT_encoding    : 8	(unsigned char)
    <@8b>   DW_AT_byte_size   : 1
 <1><@98>: Abbrev Number: 5 (DW_TAG_pointer_type)
    <@99>   DW_AT_type        : <0x@100>
 <1><@100>: Abbrev Number: 6 (DW_TAG_class_type)
    <@101>   DW_AT_calling_convention: 5	(pass by value)
    <@102>   DW_AT_name        : (indirect string, offset: 0x12): impl
    <@103>   DW_AT_byte_size   : 8
 <2><@108>: Abbrev Number: 7 (DW_TAG_member)
    <@109>   DW_AT_name        : (indirect string, offset: 0x13): n_layer
    <@10a>   DW_AT_type        : <0x@80>
    <@10b>   DW_AT_data_member_location: 0
 <2><@110>: Abbrev Number: 7 (DW_TAG_member)
    <@111>   DW_AT_name        : (indirect string, offset: 0x14): n_head
    <@112>   DW_AT_type        : <0x@80>
    <@113>   DW_AT_data_member_location: 4
 <2><@118>: Abbrev Number: 0
 <1><@120>: Abbrev Number: 8 (DW_TAG_structure_type)
    <@121>   DW_AT_calling_convention: 5	(pass by value)
    <@122>   DW_AT_name        : (indirect string, offset: 0x15): holder
    <@123>   DW_AT_byte_size   : 12
 <2><@128>: Abbrev Number: 7 (DW_TAG_member)
    <@129>   DW_AT_name        : (indirect string, offset: 0x16): i
    <@12a>   DW_AT_type        : <0x@100>
    <@12b>   DW_AT_data_member_location: 0
 <2><@130>: Abbrev Number: 7 (DW_TAG_member)
    <@131>   DW_AT_name        : (indirect string, offset: 0x17): k
    <@132>   DW_AT_type        : <0x@80>
    <@133>   DW_AT_data_member_location: 8
 <2><@138>: Abbrev Number: 0
 <1><@140>: Abbrev Number: 8 (DW_TAG_structure_type)
    <@141>   DW_AT_calling_convention: 5	(pass by value)
    <@142>   DW_AT_name        : (indirect string, offset: 0x18): Pair
    <@143>   DW_AT_byte_size   : 8
 <2><@148>: Abbrev Number: 7 (DW_TAG_member)
    <@149>   DW_AT_name        : (indirect string, offset: 0x19): a
    <@14a>   DW_AT_type        : <0x@80>
    <@14b>   DW_AT_data_member_location: 0
 <2><@150>: Abbrev Number: 7 (DW_TAG_member)
    <@151>   DW_AT_name        : (indirect string, offset: 0x1a): b
    <@152>   DW_AT_type        : <0x@80>
    <@153>   DW_AT_data_member_location: 4
 <2><@158>: Abbrev Number: 0
 <1><@160>: Abbrev Number: 9 (DW_TAG_enumeration_type)
    <@161>   DW_AT_type        : <0x@80>
    <@162>   DW_AT_name        : (indirect string, offset: 0x1b): Mode
    <@163>   DW_AT_byte_size   : 4
 <2><@168>: Abbrev Number: 10 (DW_TAG_enumerator)
    <@169>   DW_AT_name        : (indirect string, offset: 0x1c): MODE_X
    <@16a>   DW_AT_const_value : 0
 <2><@16c>: Abbrev Number: 10 (DW_TAG_enumerator)
    <@16d>   DW_AT_name        : (indirect string, offset: 0x1d): MODE_Y
    <@16e>   DW_AT_const_value : 1
 <2><@170>: Abbrev Number: 0
 <1><@180>: Abbrev Number: 9 (DW_TAG_enumeration_type)
    <@181>   DW_AT_type        : <0x@88>
    <@182>   DW_AT_name        : (indirect string, offset: 0x1e): Kind
    <@183>   DW_AT_byte_size   : 1
 <2><@188>: Abbrev Number: 10 (DW_TAG_enumerator)
    <@189>   DW_AT_name        : (indirect string, offset: 0x1f): K_A
    <@18a>   DW_AT_const_value : 0
 <2><@18c>: Abbrev Number: 10 (DW_TAG_enumerator)
    <@18d>   DW_AT_name        : (indirect string, offset: 0x20): K_B
    <@18e>   DW_AT_const_value : 1
 <2><@190>: Abbrev Number: 0
 <1><@1a0>: Abbrev Number: 0
"""

const _CN_UNIT_B = """
 <0><@0>: Abbrev Number: 1 (DW_TAG_compile_unit)
    <@1>   DW_AT_producer    : (indirect string, offset: 0x0): clang version 23.1.3
    <@2>   DW_AT_language    : 33	(C++14)
    <@3>   DW_AT_name        : (indirect string, offset: 0x1): b.cpp
 <1><@10>: Abbrev Number: 2 (DW_TAG_subprogram)
    <@11>   DW_AT_low_pc      : 0x2000
    <@12>   DW_AT_high_pc     : 0x10
    <@13>   DW_AT_linkage_name: (indirect string, offset: 0x2): _Z11vocab_score4impl
    <@14>   DW_AT_name        : (indirect string, offset: 0x3): vocab_score
    <@15>   DW_AT_type        : <0x@88>
    <@16>   DW_AT_external    : 1
 <2><@18>: Abbrev Number: 3 (DW_TAG_formal_parameter)
    <@19>   DW_AT_name        : (indirect string, offset: 0x4): v
    <@1a>   DW_AT_type        : <0x@100>
 <2><@1c>: Abbrev Number: 0
 <1><@20>: Abbrev Number: 2 (DW_TAG_subprogram)
    <@21>   DW_AT_low_pc      : 0x2010
    <@22>   DW_AT_high_pc     : 0x10
    <@23>   DW_AT_linkage_name: (indirect string, offset: 0x5): _Z9make_kindi
    <@24>   DW_AT_name        : (indirect string, offset: 0x6): make_kind
    <@25>   DW_AT_type        : <0x@180>
    <@26>   DW_AT_external    : 1
 <2><@28>: Abbrev Number: 3 (DW_TAG_formal_parameter)
    <@29>   DW_AT_name        : (indirect string, offset: 0x7): x
    <@2a>   DW_AT_type        : <0x@80>
 <2><@2c>: Abbrev Number: 0
 <1><@30>: Abbrev Number: 2 (DW_TAG_subprogram)
    <@31>   DW_AT_low_pc      : 0x2020
    <@32>   DW_AT_high_pc     : 0x10
    <@33>   DW_AT_linkage_name: (indirect string, offset: 0x8): _Z10pair_twice4Pair
    <@34>   DW_AT_name        : (indirect string, offset: 0x9): pair_twice
    <@35>   DW_AT_type        : <0x@80>
    <@36>   DW_AT_external    : 1
 <2><@38>: Abbrev Number: 3 (DW_TAG_formal_parameter)
    <@39>   DW_AT_name        : (indirect string, offset: 0xa): p
    <@3a>   DW_AT_type        : <0x@140>
 <2><@3c>: Abbrev Number: 0
 <1><@80>: Abbrev Number: 4 (DW_TAG_base_type)
    <@81>   DW_AT_name        : (indirect string, offset: 0xb): int
    <@82>   DW_AT_encoding    : 5	(signed)
    <@83>   DW_AT_byte_size   : 4
 <1><@88>: Abbrev Number: 4 (DW_TAG_base_type)
    <@89>   DW_AT_name        : (indirect string, offset: 0xc): double
    <@8a>   DW_AT_encoding    : 4	(float)
    <@8b>   DW_AT_byte_size   : 8
 <1><@100>: Abbrev Number: 6 (DW_TAG_class_type)
    <@101>   DW_AT_calling_convention: 5	(pass by value)
    <@102>   DW_AT_name        : (indirect string, offset: 0xd): impl
    <@103>   DW_AT_byte_size   : 16
 <2><@108>: Abbrev Number: 7 (DW_TAG_member)
    <@109>   DW_AT_name        : (indirect string, offset: 0xe): scores
    <@10a>   DW_AT_type        : <0x@88>
    <@10b>   DW_AT_data_member_location: 0
 <2><@110>: Abbrev Number: 7 (DW_TAG_member)
    <@111>   DW_AT_name        : (indirect string, offset: 0xf): n_vocab
    <@112>   DW_AT_type        : <0x@80>
    <@113>   DW_AT_data_member_location: 8
 <2><@118>: Abbrev Number: 0
 <1><@140>: Abbrev Number: 8 (DW_TAG_structure_type)
    <@141>   DW_AT_calling_convention: 5	(pass by value)
    <@142>   DW_AT_name        : (indirect string, offset: 0x10): Pair
    <@143>   DW_AT_byte_size   : 8
 <2><@148>: Abbrev Number: 7 (DW_TAG_member)
    <@149>   DW_AT_name        : (indirect string, offset: 0x11): a
    <@14a>   DW_AT_type        : <0x@80>
    <@14b>   DW_AT_data_member_location: 0
 <2><@150>: Abbrev Number: 7 (DW_TAG_member)
    <@151>   DW_AT_name        : (indirect string, offset: 0x12): b
    <@152>   DW_AT_type        : <0x@80>
    <@153>   DW_AT_data_member_location: 4
 <2><@158>: Abbrev Number: 0
 <1><@160>: Abbrev Number: 9 (DW_TAG_enumeration_type)
    <@161>   DW_AT_type        : <0x@80>
    <@162>   DW_AT_name        : (indirect string, offset: 0x13): Mode
    <@163>   DW_AT_byte_size   : 4
 <2><@168>: Abbrev Number: 10 (DW_TAG_enumerator)
    <@169>   DW_AT_name        : (indirect string, offset: 0x14): MODE_P
    <@16a>   DW_AT_const_value : 0
 <2><@16c>: Abbrev Number: 10 (DW_TAG_enumerator)
    <@16d>   DW_AT_name        : (indirect string, offset: 0x15): MODE_Q
    <@16e>   DW_AT_const_value : 1
 <2><@170>: Abbrev Number: 10 (DW_TAG_enumerator)
    <@171>   DW_AT_name        : (indirect string, offset: 0x16): MODE_R
    <@172>   DW_AT_const_value : 2
 <2><@174>: Abbrev Number: 0
 <1><@180>: Abbrev Number: 9 (DW_TAG_enumeration_type)
    <@181>   DW_AT_type        : <0x@80>
    <@182>   DW_AT_name        : (indirect string, offset: 0x17): Kind
    <@183>   DW_AT_byte_size   : 4
 <2><@188>: Abbrev Number: 10 (DW_TAG_enumerator)
    <@189>   DW_AT_name        : (indirect string, offset: 0x18): K_P
    <@18a>   DW_AT_const_value : 0
 <2><@18c>: Abbrev Number: 10 (DW_TAG_enumerator)
    <@18d>   DW_AT_name        : (indirect string, offset: 0x19): K_Q
    <@18e>   DW_AT_const_value : 7
 <2><@190>: Abbrev Number: 0
 <1><@1a0>: Abbrev Number: 0
"""

function _cn_dump()
    io = IOBuffer()
    println(io, "Contents of the .debug_info section:\n")
    for (base, tpl) in ((0x1000, _CN_UNIT_A), (0x2000, _CN_UNIT_B))
        println(io, "  Compilation Unit @ offset 0x", string(base; base = 16), ":")
        print(io, replace(tpl, r"@([0-9a-f]+)" =>
                          s -> string(base + parse(Int, s[2:end]; base = 16); base = 16)))
    end
    return String(take!(io))
end

@testset "A by-value crossing of a name several types share is a trap" begin

    rt, sd, _, _, ra = _CN_C.parse_dwarf_dump(_cn_dump())

    @testset "the parser keeps every definition's size" begin
        @test length(rt) == 8                                   # both units parsed
        @test sd["impl"]["conflict"] == true
        @test sd["impl"]["conflict_sizes"] == [8, 16]
        # The same width with other enumerators, and two widths.
        @test sd["__enum__Mode"]["conflict"] == true
        @test sd["__enum__Mode"]["conflict_sizes"] == [4]
        @test sd["__enum__Kind"]["conflict"] == true
        @test sd["__enum__Kind"]["conflict_sizes"] == [1, 4]
        # One type in two units, and a record defined once: nothing to record.
        for k in ("Pair", "holder")
            @test !haskey(sd[k], "conflict")
            @test !haskey(sd[k], "conflict_sizes")
        end
        @test [m["c_type"] for m in sd["holder"]["members"]] == ["impl", "int"]
    end

    # The parsed tables plus shapes a dump this small does not need to spell.
    mem(n, ct, off) = Dict{String,Any}("name" => n, "c_type" => ct, "julia_type" => "Any",
                                       "offset" => off, "size" => 8)
    rec(ms...; bases = Any[], extra...) = Dict{String,Any}("kind" => "struct", "byte_size" => "0x18",
        "members" => Any[ms...], "base_classes" => bases, (String(k) => v for (k, v) in extra)...)
    structs = merge(Dict{String,Any}(sd), Dict{String,Any}(
        "derived"    => rec(mem("x", "int", "0x10"); bases = Any[Dict{String,Any}("type" => "impl", "offset" => 0)]),
        "arr"        => rec(mem("two", "impl[2]", "0x0")),
        "nest"       => rec(mem("h", "const holder", "0x0")),
        "by_ptr"     => rec(mem("p", "impl*", "0x0"), mem("r", "const impl&", "0x8")),
        "static_only"=> rec(mem("s", "impl", nothing)),
        "with_mode"  => rec(mem("m", "Mode", "0x0")),
        "with_kind"  => rec(mem("k", "Kind", "0x0")),
        "loop_a"     => rec(mem("b", "loop_b", "0x0")),
        "loop_b"     => rec(mem("a", "loop_a", "0x0")),
        # Conflicted and naming itself as its base, as llama.cpp's `graph` does.
        "graph"      => rec(; bases = Any[Dict{String,Any}("type" => "graph", "offset" => 0)], conflict = true,
                            conflict_sizes = [336, 344]),
        # An enum from metadata written before `conflict_sizes` existed.
        "__enum__Old" => Dict{String,Any}("kind" => "enum", "byte_size" => "0x4", "julia_type" => "Cint",
                                          "enumerators" => Any[], "conflict" => true)))

    @testset "by_value_crossing" begin
        k(t) = _CN_F.by_value_crossing(t, structs, ra)
        @test k("impl") == (:conflict, "impl")
        @test k("const impl") == (:conflict, "impl")
        @test k("model::impl") == (:conflict, "impl")           # the demangler's spelling
        @test k("impl*") == (:direct, nothing)                  # by address: not this
        @test k("const impl&") == (:direct, nothing)
        # Held by value at any depth: a member, a const member, an array, a base.
        @test k("holder") == (:conflict, "impl")
        @test k("nest") == (:conflict, "impl")
        @test k("arr") == (:conflict, "impl")
        @test k("derived") == (:conflict, "impl")
        @test k("graph") == (:conflict, "graph")
        @test k("with_kind") == (:conflict, "__enum__Kind")
        # Nothing conflicted is held, or nothing is held by value.
        @test k("Pair") == (:layout, "Pair")
        @test k("by_ptr") == (:layout, "by_ptr")
        @test k("static_only") == (:layout, "static_only")
        @test k("with_mode") == (:layout, "with_mode")
        @test k("loop_a") == (:layout, "loop_a")                # the walk ends
        # Every `Mode` is 4 bytes, so one i32 is right for all of them.
        @test k("Mode") == (:enum, "__enum__Mode")
        @test k("Kind") == (:conflict, "__enum__Kind")
        @test k("Old") == (:conflict, "__enum__Old")
    end

    row(m) = Dict{String,Any}("name" => m, "mangled" => m, "demangled" => m, "is_method" => false,
        "parameters" => rt[m]["parameters"],
        "return_type" => Dict{String,Any}("c_type" => rt[m]["c_type"], "julia_type" => rt[m]["julia_type"],
                                          "size" => get(rt[m], "size", 0)))

    @testset "thunk_crossings and the thunk generator" begin
        tc(m) = _CN_F.thunk_crossings(row(m), structs, ra)
        @test tc("_Z10takes_impl4impl").trap ==
              "takes `impl` by value, and distinct records in this library are named `impl` (8 and 16 bytes)"
        @test tc("_Z10takes_impl4impl").conflict_trap == tc("_Z10takes_impl4impl").trap
        @test tc("_Z10takes_impl4impl").layout_trap === nothing
        @test tc("_Z12takes_holder6holder").trap ==
              "takes `holder` by value, which holds `impl` by value, and distinct records in this library " *
              "are named `impl` (8 and 16 bytes)"
        @test tc("_Z9make_kindi").trap ==
              "returns `Kind` by value, and enums of different widths in this library are named `Kind` (1 and 4 bytes)"
        for m in ("_Z10takes_pair4Pair", "_Z10takes_mode4Mode", "_Z14takes_impl_ptrP4impl", "_Z10pair_twice4Pair")
            @test (m, tc(m).trap) == (m, nothing)
        end

        ir = _CN_F.generate_function_thunks([row(m) for m in sort(collect(keys(rt)))], structs;
                                            may_throw = true, record_abi = ra)
        for m in ("_Z10takes_impl4impl", "_Z12takes_holder6holder", "_Z11vocab_score4impl", "_Z9make_kindi")
            @test occursin("// no thunk for $m: ", ir)
            @test !occursin("@$(m)_thunk(", ir)
        end
        for m in ("_Z10takes_pair4Pair", "_Z10takes_mode4Mode", "_Z14takes_impl_ptrP4impl", "_Z10pair_twice4Pair")
            @test occursin("func.func @$(m)_thunk(", ir)
        end
    end

    @testset "a conflicted crossing never takes Tier 3" begin
        nx(m) = merge(row(m), Dict{String,Any}("is_noexcept" => true))
        @test !_CN_W.is_ccall_safe(nx("_Z10takes_impl4impl"), structs; record_abi = ra)
        @test !_CN_W.is_ccall_safe(nx("_Z12takes_holder6holder"), structs; record_abi = ra)
        @test !_CN_W.is_ccall_safe(nx("_Z9make_kindi"), structs; record_abi = ra)
        @test _CN_W.is_ccall_safe(nx("_Z10takes_pair4Pair"), structs; record_abi = ra)
        @test _CN_W.is_ccall_safe(nx("_Z10takes_mode4Mode"), structs; record_abi = ra)
        @test _CN_W.is_ccall_safe(nx("_Z14takes_impl_ptrP4impl"), structs; record_abi = ra)
    end

    # ── Both generators ─────────────────────────────────────────────────────
    function _cn_config(lang)
        dir = mktempdir()
        write(joinpath(dir, "replibuild.toml"), """
        [project]
        name = "conflicts"
        root = "$(escape_string(dir))"
        [link]
        enable_lto = false
        [wrap]
        language = "$lang"
        [wrap.varargs]
        log_impl = [["Cint"]]
        [types]
        strictness = "warn"
        allow_unknown_structs = true
        [cache]
        enabled = false
        """)
        return RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml"))
    end
    function _cn_generate(lang, md)
        cfg = _cn_config(lang)
        gen = lang == "cpp" ? _CN_W.generate_introspective_module_cpp : _CN_W.generate_introspective_module_c
        out = gen(cfg, _CN_LIB, deepcopy(md), "Conflicts", _CN_W.create_type_registry(cfg), true)
        return out isa Tuple ? String(first(out)) : String(out)
    end
    # Rows as `extract_compilation_metadata` writes them. C has no mangling, so
    # its rows carry the bare name; `vocab_score` and `pair_twice` may throw.
    demangled = Dict("_Z10takes_impl4impl" => "takes_impl(impl)", "_Z12takes_holder6holder" => "takes_holder(holder)",
                     "_Z10takes_pair4Pair" => "takes_pair(Pair)", "_Z10takes_mode4Mode" => "takes_mode(Mode)",
                     "_Z14takes_impl_ptrP4impl" => "takes_impl_ptr(impl*)", "_Z11vocab_score4impl" => "vocab_score(impl)",
                     "_Z9make_kindi" => "make_kind(int)", "_Z10pair_twice4Pair" => "pair_twice(Pair)")
    function _cn_row(lang, mangled, dem, params, ret; noexcept = true, vararg = false)
        sym = lang == "c" ? String(first(split(dem, '('))) : mangled
        r = only(_CN_C.parse_function_signatures([Dict{String,Any}("mangled" => sym,
                                                                   "demangled" => lang == "c" ? sym : dem)]))
        r["parameters"] = params
        r["return_type"] = ret
        r["is_noexcept"] = noexcept
        r["is_vararg"] = vararg
        return r
    end
    _cn_row(lang, m) = _cn_row(lang, m, demangled[m], row(m)["parameters"], row(m)["return_type"];
                               noexcept = !(m in ("_Z11vocab_score4impl", "_Z10pair_twice4Pair")))
    _cn_p(n, jt, ct) = Dict{String,Any}("name" => n, "julia_type" => jt, "c_type" => ct)
    _cn_int = Dict{String,Any}("c_type" => "int", "julia_type" => "Cint", "size" => 4)

    trapped = ["takes_impl", "takes_holder", "vocab_score", "make_kind"]
    called = ["takes_pair", "takes_mode", "takes_impl_ptr", "pair_twice"]
    for lang in ("c", "cpp")
        rows = Any[_cn_row(lang, m) for m in sort(collect(keys(rt)))]
        push!(rows, _cn_row(lang, "_Z8log_impl4implPKcz", "log_impl(impl, const char*, ...)",
                            Any[_cn_p("v", "impl", "impl"), _cn_p("fmt", "Cstring", "const char*"),
                                _cn_p("varargs...", "Any", "...")], _cn_int; vararg = true))
        # Declared nowhere in the dump: its layout is missing, not conflicted.
        lang == "cpp" && push!(rows, _cn_row(lang, "_Z10takes_decl4Decl", "takes_decl(Decl)",
                                             Any[_cn_p("d", "Any", "Decl")], _cn_int))
        md = Dict{String,Any}("functions" => rows, "struct_definitions" => Dict{String,Any}(sd), "record_abi" => ra,
                              "globals" => Dict{String,Any}(), "timestamp" => "2026-10-23T00:00:00")
        code = _cn_generate(lang, md)
        # One chunk per function: from its `function` line to the next top-level `end`.
        body(f) = (m = match(Regex("\\nfunction $f\\(.*?\\n(.*?)\\nend\\n", "s"), code);
                   m === nothing ? "" : String(m.captures[1]))

        @testset "$lang: the conflicted crossings are traps" begin
            checked = 0
            for f in trapped
                b = body(f)
                @test (f, occursin("Base.error(\"ABI Safety Trap: cannot call '$f'", b)) == (f, true)
                @test (f, occursin("RepliBuild keys types by their bare DWARF name", b)) == (f, true)
                @test (f, occursin("ccall", b) || occursin("JITManager", b)) == (f, false)
                @test _CN_W._is_trap_chunk("function $f()\n$b\nend")
                checked += 1
            end
            @test checked == length(trapped)
            @test occursin("It takes `impl` by value, and distinct records in this library are named `impl` " *
                           "(8 and 16 bytes).", body("takes_impl"))
            @test occursin("which holds `impl` by value", body("takes_holder"))
            @test occursin("enums of different widths in this library are named `Kind` (1 and 4 bytes)",
                           body("make_kind"))
            # The varargs path `continue`s past both generators' own trap.
            for f in ("log_impl", "log_impl_Cint")
                @test occursin("Base.error(\"ABI Safety Trap: cannot call '$f'", body(f))
                @test !occursin("@ccall", body(f))
            end
        end

        @testset "$lang: everything else is still called" begin
            checked = 0
            for f in called
                b = body(f)
                @test (f, occursin("ccall", b) || occursin("JITManager.invoke", b)) == (f, true)
                @test (f, occursin("Safety Trap", b)) == (f, false)
                checked += 1
            end
            @test checked == length(called)
            # Tier 3 where nothing else sent it to the thunk (C has no thunks).
            for f in ("takes_pair", "takes_mode", "takes_impl_ptr")
                @test occursin("ccall((:", body(f))
            end
        end

        lang == "cpp" && @testset "cpp: a layout missing from DWARF keeps its message" begin
            want = "ABI Safety Trap: cannot call 'takes_decl' (takes_decl(Decl)). It takes `Decl` by value, " *
                   "and that record's layout is not in this library's DWARF, so no call could pass it " *
                   "correctly. A type only declared here (header-only, constructor homing) is recovered by " *
                   "building with `-Xclang -fno-use-ctor-homing` or `-fstandalone-debug` in [compile] flags; " *
                   "otherwise keep the function out with [wrap] exclude_symbols."
            @test occursin("Base.error($(repr(want)))", body("takes_decl"))
        end
    end
end
