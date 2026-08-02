#!/usr/bin/env julia
# DWARF DIE attribution — parameter/context boundary tests.
#
# These drive Compiler.parse_dwarf_dump directly with synthetic readelf dumps:
# no compiler, no binary, no fixture build. The parser walks a flat line stream
# and reconstructs the DIE tree from the `<depth><offset>` headers, so every
# context bug in it is a boundary bug — a context that outlives the DIE that
# opened it. Synthetic dumps pin those boundaries exactly, which a built
# fixture cannot: the fixture only fails once the mis-attribution happens to
# land on a function someone calls.
#
# Regression: free_opaque (test/c_abomination_test) came out with a phantom
# second parameter `next::Ptr{SelfReferential}` — the first MEMBER of a struct
# defined after it — and a two-argument ccall against a one-argument C
# function.

using Test
using RepliBuild

const DWARF_COMPILER = RepliBuild.Compiler

dwarf_params_of(rt, key) = [(p["name"], p["c_type"]) for p in get(rt[key], "parameters", [])]

@testset "DWARF DIE attribution" begin

    # ── The c_abomination shape, reduced to its essentials ──────────────────
    #
    # Verbatim structure from readelf --debug-dump=info on the fixture:
    #   * free_opaque is the LAST subprogram in the CU (nothing resets the
    #     parser's function context after it)
    #   * a DW_TAG_subroutine_type with an UNNAMED formal_parameter follows it
    #     (function-pointer typedef — nothing to extract, so the parameter
    #     context stays open)
    #   * then a struct whose first member is a named pointer
    #
    # The member's DW_AT_name + DW_AT_type must land on the struct, never on
    # the subroutine parameter, and never on free_opaque.
    dump_abomination = """
     <0><b>: Abbrev Number: 1 (DW_TAG_compile_unit)
        <c>   DW_AT_producer    : (indexed string: 0x0): clang version 18.1.7jl
        <d>   DW_AT_language    : 29	(C11)
     <1><153>: Abbrev Number: 14 (DW_TAG_subprogram)
        <154>   DW_AT_low_pc      : (index: 0x6): 0x12f0
        <15b>   DW_AT_name        : (indexed string: 0x26): free_opaque
        <15d>   DW_AT_decl_line   : 53
        <15e>   DW_AT_external    : 1
     <2><15e>: Abbrev Number: 12 (DW_TAG_formal_parameter)
        <15f>   DW_AT_location    : 2 byte block: 91 78 	(DW_OP_fbreg: -8)
        <162>   DW_AT_name        : (indexed string: 0x2c): state
        <165>   DW_AT_type        : <0x41>
     <2><169>: Abbrev Number: 0
     <1><2a3>: Abbrev Number: 7 (DW_TAG_pointer_type)
        <2a4>   DW_AT_type        : <0x2a8>
     <1><2a8>: Abbrev Number: 19 (DW_TAG_subroutine_type)
        <2a9>   DW_AT_type        : <0x77>
        <2ad>   DW_AT_prototyped  : 1
     <2><2ad>: Abbrev Number: 20 (DW_TAG_formal_parameter)
        <2ae>   DW_AT_type        : <0x23a>
     <2><2b2>: Abbrev Number: 0
     <1><2b3>: Abbrev Number: 7 (DW_TAG_pointer_type)
        <2b4>   DW_AT_type        : <0x2b8>
     <1><2b8>: Abbrev Number: 21 (DW_TAG_structure_type)
        <2b9>   DW_AT_name        : (indexed string: 0x31): SelfReferential
        <2ba>   DW_AT_byte_size   : 24
     <2><2bd>: Abbrev Number: 10 (DW_TAG_member)
        <2be>   DW_AT_name        : (indexed string: 0x2e): next
        <2bf>   DW_AT_type        : <0x2b3>
        <2c5>   DW_AT_data_member_location: 0
     <2><2c6>: Abbrev Number: 10 (DW_TAG_member)
        <2c7>   DW_AT_name        : (indexed string: 0x2f): process
        <2c8>   DW_AT_type        : <0x2d9>
        <2ce>   DW_AT_data_member_location: 8
     <2><2d8>: Abbrev Number: 0
     <1><30a>: Abbrev Number: 0
    """

    @testset "struct member does not leak onto a preceding subprogram" begin
        rt, structs, _, _ = DWARF_COMPILER.parse_dwarf_dump(dump_abomination)

        @test haskey(rt, "free_opaque")
        # The bug: 2 parameters, the second being SelfReferential's `next` member.
        @test length(rt["free_opaque"]["parameters"]) == 1
        @test dwarf_params_of(rt, "free_opaque")[1][1] == "state"
        @test !any(p["name"] == "next" for p in rt["free_opaque"]["parameters"])

        # And the member still belongs to the struct it was declared in.
        @test haskey(structs, "SelfReferential")
        member_names = [m["name"] for m in structs["SelfReferential"]["members"]]
        @test "next" in member_names
        @test "process" in member_names
    end

    @testset "parameter context closes at its DIE terminator" begin
        # Two adjacent void functions, the first with a parameter, the second
        # with none. If the parameter context survives its `Abbrev Number: 0`,
        # the second function inherits an argument it does not have.
        dump = """
         <0><b>: Abbrev Number: 1 (DW_TAG_compile_unit)
            <c>   DW_AT_producer    : (indexed string: 0x0): clang version 18.1.7jl
         <1><30>: Abbrev Number: 14 (DW_TAG_subprogram)
            <34>   DW_AT_name        : (indexed string: 0x1): takes_one
            <36>   DW_AT_external    : 1
         <2><40>: Abbrev Number: 12 (DW_TAG_formal_parameter)
            <44>   DW_AT_name        : (indexed string: 0x2): only
            <48>   DW_AT_type        : <0x67>
         <2><50>: Abbrev Number: 0
         <1><60>: Abbrev Number: 14 (DW_TAG_subprogram)
            <64>   DW_AT_name        : (indexed string: 0x3): takes_none
            <66>   DW_AT_external    : 1
         <1><70>: Abbrev Number: 0
        """
        rt, _, _, _ = DWARF_COMPILER.parse_dwarf_dump(dump)

        @test length(rt["takes_one"]["parameters"]) == 1
        @test rt["takes_none"]["parameters"] == []
    end

    @testset "recorded parameters are not mutated after the function closes" begin
        # The parameter array used to be stored live in the return_types entry,
        # so anything pushed later retro-mutated an already-finished function.
        # A named parameter under a LATER subprogram must not reach an earlier one.
        dump = """
         <0><b>: Abbrev Number: 1 (DW_TAG_compile_unit)
            <c>   DW_AT_producer    : (indexed string: 0x0): clang version 18.1.7jl
         <1><30>: Abbrev Number: 14 (DW_TAG_subprogram)
            <34>   DW_AT_name        : (indexed string: 0x1): first
            <36>   DW_AT_external    : 1
         <2><40>: Abbrev Number: 12 (DW_TAG_formal_parameter)
            <44>   DW_AT_name        : (indexed string: 0x2): a
            <48>   DW_AT_type        : <0x67>
         <2><50>: Abbrev Number: 0
         <1><60>: Abbrev Number: 14 (DW_TAG_subprogram)
            <64>   DW_AT_name        : (indexed string: 0x3): second
            <66>   DW_AT_external    : 1
         <2><70>: Abbrev Number: 12 (DW_TAG_formal_parameter)
            <74>   DW_AT_name        : (indexed string: 0x4): b
            <78>   DW_AT_type        : <0x67>
         <2><80>: Abbrev Number: 0
         <1><90>: Abbrev Number: 0
        """
        rt, _, _, _ = DWARF_COMPILER.parse_dwarf_dump(dump)

        @test [p["name"] for p in rt["first"]["parameters"]] == ["a"]
        @test [p["name"] for p in rt["second"]["parameters"]] == ["b"]
    end

    @testset "function-pointer parameters still resolve" begin
        # The fix tightened parameter attribution to DIRECT children of the
        # subprogram. A parameter whose type is a pointer to a
        # DW_TAG_subroutine_type must still come through — the subroutine's own
        # parameters live outside the function subtree and must not be counted
        # as the function's.
        dump = """
         <0><b>: Abbrev Number: 1 (DW_TAG_compile_unit)
            <c>   DW_AT_producer    : (indexed string: 0x0): clang version 18.1.7jl
         <1><30>: Abbrev Number: 14 (DW_TAG_subprogram)
            <32>   DW_AT_type        : <0x67>
            <34>   DW_AT_name        : (indexed string: 0x1): execute_outer
            <36>   DW_AT_external    : 1
         <2><40>: Abbrev Number: 12 (DW_TAG_formal_parameter)
            <44>   DW_AT_name        : (indexed string: 0x2): f
            <48>   DW_AT_type        : <0xa0>
         <2><50>: Abbrev Number: 12 (DW_TAG_formal_parameter)
            <54>   DW_AT_name        : (indexed string: 0x3): a
            <58>   DW_AT_type        : <0x67>
         <2><5a>: Abbrev Number: 0
         <1><67>: Abbrev Number: 2 (DW_TAG_base_type)
            <68>   DW_AT_byte_size   : 4
            <69>   DW_AT_encoding    : 5	(signed)
            <6a>   DW_AT_name        : (indexed string: 0x4): int
         <1><a0>: Abbrev Number: 7 (DW_TAG_pointer_type)
            <a1>   DW_AT_type        : <0xb0>
         <1><b0>: Abbrev Number: 19 (DW_TAG_subroutine_type)
            <b1>   DW_AT_type        : <0x67>
            <b5>   DW_AT_prototyped  : 1
         <2><b6>: Abbrev Number: 20 (DW_TAG_formal_parameter)
            <b7>   DW_AT_type        : <0x67>
         <2><ba>: Abbrev Number: 0
         <1><c0>: Abbrev Number: 0
        """
        rt, _, _, _ = DWARF_COMPILER.parse_dwarf_dump(dump)

        @test [p["name"] for p in rt["execute_outer"]["parameters"]] == ["f", "a"]
    end

    # ── The arity guard ─────────────────────────────────────────────────────

    @testset "arity guard rejects phantom parameters" begin
        # A signature claiming more parameters than the DIE tree has is a
        # wrong-argument ccall; the guard must abort rather than emit it.
        rt = Dict{String,Dict{String,Any}}(
            "free_opaque" => Dict{String,Any}("parameters" => [
                Dict("name" => "state"), Dict("name" => "next")
            ])
        )
        err = try
            DWARF_COMPILER.check_param_arity!(rt, Dict("free_opaque" => 1))
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("free_opaque", err.msg)
        @test occursin("phantom", err.msg)
    end

    @testset "arity guard accepts a matching signature" begin
        rt = Dict{String,Dict{String,Any}}(
            "free_opaque" => Dict{String,Any}("parameters" => [Dict("name" => "state")])
        )
        @test DWARF_COMPILER.check_param_arity!(rt, Dict("free_opaque" => 1)) === nothing
    end

    @testset "arity guard warns (does not abort) on unextracted parameters" begin
        # Declaration-only DIEs carry unnamed parameters that nothing can be
        # extracted from; the definition DIE supplies the real list. Loud, but
        # not fatal.
        rt = Dict{String,Dict{String,Any}}(
            "decl_only" => Dict{String,Any}("parameters" => [])
        )
        @test_logs (:warn,) DWARF_COMPILER.check_param_arity!(rt, Dict("decl_only" => 2))
    end

    @testset "arity guard is silent for functions with no subprogram DIE" begin
        rt = Dict{String,Dict{String,Any}}(
            "from_symbol_table" => Dict{String,Any}("parameters" => [])
        )
        @test DWARF_COMPILER.check_param_arity!(rt, Dict{String,Int}()) === nothing
    end

end
