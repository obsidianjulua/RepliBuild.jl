#!/usr/bin/env julia
# test/vi_test/verify.jl — Virtual inheritance (diamond) end-to-end
#
#   VBase (virtual base: t=7, virtual tag())
#   Left : virtual VBase (l=100)      Right : virtual VBase (r=200)
#   Diamond : Left, Right (d=300, overrides tag() -> t+1000)
#
# The load-bearing fact: VBase sits at +16 in a standalone Left but +32 in a
# Diamond (tail-padding reuse puts d at 28) — only the object's vtable knows
# which. The chain verified here:
#   1. Extraction: virtual edges carry vbase_vtable_offset (-24), NOT a fake
#      static offset; Diamond's Left/Right stay static (0/16)
#   2. type_info: vbaseNames/vbaseVtableOffsets emitted, verifier-clean
#   3. Wrapper: dynamic upcasts — the SAME helper resolves different offsets
#      for standalone-Left vs Left-inside-Diamond (the canary), and all
#      three views of a Diamond land on the ONE shared VBase
#   4. Live dispatch: VBase_tag through the vbase vtable reaches Diamond's
#      override; the vcall producer needs no changes for any of this

using Test
using JSON
import RepliBuild

const SCRIPT_DIR = @__DIR__

wrapper_path = joinpath(SCRIPT_DIR, "julia", "ViTest.jl")
isfile(wrapper_path) || error("Wrapper not found at $wrapper_path — run build+wrap first")

include(wrapper_path)
using .ViTest

@testset "Virtual Inheritance Pipeline" begin

    # ── 1. Extraction metadata ──────────────────────────────────────────────
    @testset "metadata: vbase vtable coordinates" begin
        meta = JSON.parsefile(joinpath(SCRIPT_DIR, "julia", "compilation_metadata.json"))
        structs = get(meta, "struct_definitions", Dict())
        @test haskey(structs, "Left") && haskey(structs, "Diamond")

        lbases = get(structs["Left"], "base_classes", [])
        @test length(lbases) == 1
        @test get(lbases[1], "type", "") == "VBase"
        @test get(lbases[1], "virtual", false) == true
        @test get(lbases[1], "vbase_vtable_offset", 0) == -24
        # No fake static offset (the pre-fix constant regex matched the "7"
        # of readelf's "7 byte block:" — pinned dead here)
        @test get(lbases[1], "offset", -1) == 0

        dbases = get(structs["Diamond"], "base_classes", [])
        @test [get(b, "type", "") for b in dbases] == ["Left", "Right"]
        @test [get(b, "offset", -1) for b in dbases] == [0, 16]
        @test all(b -> get(b, "virtual", false) == false, dbases)
    end

    # ── 2. DWARFParser + type_info vbase table ──────────────────────────────
    @testset "DWARFParser + type_info emission" begin
        so_path = joinpath(SCRIPT_DIR, "julia", "libvi_test.so")
        vt = RepliBuild.DWARFParser.parse_vtables(so_path)
        ci = vt.classes["Left"]
        @test ci.base_classes == ["VBase"]
        @test ci.virtual_bases == [true]
        @test ci.vbase_vtable_offsets == [-24]

        ir = RepliBuild.JLCSIRGenerator.generate_type_info_ir("Left", ci, UInt64(0))
        @test occursin("vbaseNames = [\"VBase\"]", ir)
        @test occursin("vbaseVtableOffsets = [-24 : i64]", ir)
        @test !occursin("baseNames", split(ir, "vbase")[1])  # not in the static table

        if isfile(RepliBuild.MLIRNative.libJLCS)
            ctx = RepliBuild.MLIRNative.create_context()
            try
                @test RepliBuild.MLIRNative.parse_module(ctx, "module {\n$(ir)\n}") != C_NULL
            finally
                RepliBuild.MLIRNative.destroy_context(ctx)
            end
        end
    end

    # ── 3. Dynamic upcasts: the canary ──────────────────────────────────────
    @testset "dynamic vbase upcasts" begin
        for fn in (:Left_as_VBase, :Right_as_VBase, :Diamond_as_VBase,
                   :Diamond_as_Left, :Diamond_as_Right)
            @test isdefined(ViTest, fn)
        end

        pl = ViTest.make_left()
        pl = pl isa Ptr ? Ptr{Cvoid}(pl) : Ptr{Cvoid}(pl.handle)
        pdl = ViTest.make_diamond_as_left()
        pdl = pdl isa Ptr ? Ptr{Cvoid}(pdl) : Ptr{Cvoid}(pdl.handle)
        try
            # THE canary: the same helper, two dynamic types, two offsets.
            # Any static-offset implementation gets one of these wrong.
            @test Int(ViTest.Left_as_VBase(pl))  - Int(pl)  == 16   # standalone Left
            @test Int(ViTest.Left_as_VBase(pdl)) - Int(pdl) == 32   # Left-in-Diamond

            # Dispatch through the vbase vtable: standalone gets VBase::tag,
            # the Diamond-backed one gets the OVERRIDE — from identical calls
            @test ViTest.VBase_tag(ViTest.Left_as_VBase(pl))  == 7
            @test ViTest.VBase_tag(ViTest.Left_as_VBase(pdl)) == 1007
        finally
            ViTest.free_left(pl)
            ViTest.free_left(pdl)   # virtual dtor: deletes the Diamond correctly
        end
    end

    # ── 4. Diamond single-copy + full dispatch matrix ───────────────────────
    @testset "diamond single-copy + dispatch" begin
        pd = ViTest.make_diamond()
        pd = pd isa Ptr ? Ptr{Cvoid}(pd) : Ptr{Cvoid}(pd.handle)
        try
            # All three views resolve to the ONE shared VBase subobject
            vb = ViTest.Diamond_as_VBase(pd)
            @test vb == ViTest.Left_as_VBase(pd)
            @test vb == ViTest.Right_as_VBase(ViTest.Diamond_as_Right(pd))
            @test Int(vb) - Int(pd) == 32

            # Virtual dispatch from every angle reaches the override
            @test ViTest.VBase_tag(vb) == 1007
            @test ViTest.Diamond_tag(pd) == 1007      # re-homed primary slot
            # Non-virtual + non-vbase methods unchanged
            @test ViTest.Left_lval(pd) == 100
            @test ViTest.Right_rval(ViTest.Diamond_as_Right(pd)) == 200
            @test ViTest.Diamond_dval(pd) == 100 + 200 + 7 + 300
        finally
            ViTest.free_diamond(pd)
        end
    end
end

println("VI_VERIFY_DONE")
