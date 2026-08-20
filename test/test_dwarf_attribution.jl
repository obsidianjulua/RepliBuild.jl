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

    # ── By-value aggregate returns reached through a typedef ────────────────
    #
    # `typedef struct mpack_tag_t mpack_tag_t;` is the ordinary C spelling, so
    # a subprogram's DW_AT_type points at the TYPEDEF, not at the structure.
    # The in-loop return mapping tested the DIE kind of what it pointed at
    # (`kind in ["struct","class"]`), missed, and fell through to the name-only
    # mapper, which answers "Any" with size 0. Downstream the function is
    # dropped rather than wrapped -- an emitted one would trip
    # `_assert_no_any_ccall_return` and refuse the whole wrapper -- so the
    # symbol silently had no binding at all.
    #
    # The asymmetry is what makes it unmistakable, and it is asserted below:
    # the SAME type as a PARAMETER always resolved, because that path runs the
    # struct/enum-aware mapper. Measured on the Hub before the fix: 2724
    # `Any` returns across 21 packages, incl. every libcurl function returning
    # CURLcode and every mpack_tag_* returning mpack_tag_t.
    dump_byval = """
     <0><b>: Abbrev Number: 1 (DW_TAG_compile_unit)
        <c>   DW_AT_producer    : (indexed string: 0x0): clang version 18.1.7jl
        <d>   DW_AT_language    : 29	(C11)
     <1><100>: Abbrev Number: 21 (DW_TAG_structure_type)
        <101>   DW_AT_name        : (indexed string: 0x1): tag_s
        <102>   DW_AT_byte_size   : 16
     <2><103>: Abbrev Number: 10 (DW_TAG_member)
        <104>   DW_AT_name        : (indexed string: 0x2): kind
        <105>   DW_AT_type        : <0x300>
        <106>   DW_AT_data_member_location: 0
     <2><107>: Abbrev Number: 10 (DW_TAG_member)
        <108>   DW_AT_name        : (indexed string: 0x3): value
        <109>   DW_AT_type        : <0x300>
        <10a>   DW_AT_data_member_location: 8
     <2><10b>: Abbrev Number: 0
     <1><200>: Abbrev Number: 34 (DW_TAG_typedef)
        <201>   DW_AT_name        : (indexed string: 0x4): tag_t
        <202>   DW_AT_type        : <0x100>
     <1><300>: Abbrev Number: 5 (DW_TAG_base_type)
        <301>   DW_AT_name        : (indexed string: 0x5): long int
        <302>   DW_AT_byte_size   : 8
        <303>   DW_AT_encoding    : 5	(signed)
     <1><310>: Abbrev Number: 5 (DW_TAG_base_type)
        <311>   DW_AT_name        : (indexed string: 0x6): char
        <312>   DW_AT_byte_size   : 1
        <313>   DW_AT_encoding    : 6	(signed char)
     <1><320>: Abbrev Number: 7 (DW_TAG_pointer_type)
        <321>   DW_AT_type        : <0x310>
     <1><400>: Abbrev Number: 14 (DW_TAG_subprogram)
        <401>   DW_AT_name        : (indexed string: 0x7): tag_make
        <402>   DW_AT_type        : <0x200>
        <403>   DW_AT_external    : 1
     <1><410>: Abbrev Number: 0
     <1><420>: Abbrev Number: 14 (DW_TAG_subprogram)
        <421>   DW_AT_name        : (indexed string: 0x8): tag_make_direct
        <422>   DW_AT_type        : <0x100>
        <423>   DW_AT_external    : 1
     <1><430>: Abbrev Number: 0
     <1><440>: Abbrev Number: 14 (DW_TAG_subprogram)
        <441>   DW_AT_name        : (indexed string: 0x9): tag_cmp
        <442>   DW_AT_type        : <0x300>
        <443>   DW_AT_external    : 1
     <2><444>: Abbrev Number: 12 (DW_TAG_formal_parameter)
        <445>   DW_AT_name        : (indexed string: 0xa): left
        <446>   DW_AT_type        : <0x200>
     <2><447>: Abbrev Number: 0
     <1><450>: Abbrev Number: 14 (DW_TAG_subprogram)
        <451>   DW_AT_name        : (indexed string: 0xb): tag_name
        <452>   DW_AT_type        : <0x320>
        <453>   DW_AT_external    : 1
     <1><460>: Abbrev Number: 0
     <1><470>: Abbrev Number: 0
    """

    @testset "by-value aggregate return through a typedef resolves" begin
        rt, structs, _, _ = DWARF_COMPILER.parse_dwarf_dump(dump_byval)

        @test haskey(structs, "tag_s")

        # THE BUG: return reached through a typedef.
        @test haskey(rt, "tag_make")
        @test rt["tag_make"]["c_type"] == "tag_s"
        @test rt["tag_make"]["julia_type"] != "Any"
        @test rt["tag_make"]["julia_type"] == "tag_s"
        # Size must come from the aggregate's own DIE. It was 0, and a 0-sized
        # by-value struct return is exactly what the generator cannot classify.
        @test rt["tag_make"]["size"] == 16

        # The direct (non-typedef) spelling always worked — pinned so a future
        # rewrite cannot fix one path by breaking the other.
        @test rt["tag_make_direct"]["julia_type"] == "tag_s"
        @test rt["tag_make_direct"]["size"] == 16

        # THE ASYMMETRY that identified the bug: the same type as a PARAMETER
        # resolved correctly the whole time.
        @test rt["tag_cmp"]["julia_type"] == "Clong"
        params = dwarf_params_of(rt, "tag_cmp")
        @test length(params) == 1
        @test params[1][2] == "tag_s"
        @test rt["tag_cmp"]["parameters"][1]["julia_type"] == "tag_s"

        # NEGATIVE CONTROL — the repair must never touch a return that already
        # mapped. `char*` stays Cstring: the whole Cstring return policy
        # (`_assert_cstring_policy`, the `_ptr` siblings, the
        # `Union{String,Nothing}` wrappers) is built on this mapping, and a
        # wholesale swap to the parameter path's mapper would have changed it
        # to Ptr{UInt8} and silently dismantled the policy.
        @test rt["tag_name"]["julia_type"] == "Cstring"
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
