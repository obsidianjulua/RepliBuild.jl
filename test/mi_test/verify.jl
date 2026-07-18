#!/usr/bin/env julia
# test/mi_test/verify.jl — Verify multiple-inheritance handling end-to-end
#
# Layout under test (Itanium x86_64):
#   Derived = Base1 subobject @0 {vptr, a=111}, Base2 subobject @16 {vptr, b=222}
# A Base2 method receives a BASE-relative `this`: called with an unadjusted
# Derived* it reads Base1's `a` through Base2 code. The chain verified here:
#   1. Extraction: base_classes carry real subobject offsets (both parsers)
#   2. type_info IR: baseNames/baseOffsets table emitted and verifier-clean
#   3. Wrapper: Derived_as_Base2 upcast helper applies +16
#   4. Live calls: Base2 methods return Base2's data via the upcast pointer
#      (and the pinned wrong-`this` read is observable on a non-virtual call)

using Test
using JSON
import RepliBuild

const SCRIPT_DIR = @__DIR__

wrapper_path = joinpath(SCRIPT_DIR, "julia", "MiTest.jl")
if !isfile(wrapper_path)
    error("Wrapper not found at $wrapper_path — run build+wrap first")
end

include(wrapper_path)
using .MiTest

@testset "Multiple Inheritance Pipeline" begin

    # ── 1. Extraction metadata: base offsets present and correct ────────────
    @testset "compilation_metadata base offsets" begin
        meta_path = joinpath(SCRIPT_DIR, "julia", "compilation_metadata.json")
        @test isfile(meta_path)
        meta = JSON.parsefile(meta_path)
        structs = get(meta, "struct_definitions", Dict())
        @test haskey(structs, "Derived")
        # Nested-type member attribution: members FOLLOWING a nested enum
        # definition must stay attributed to the enclosing class (the parser
        # used to re-point at the enum and drop them — found via box2d
        # b2Shape::m_radius, which follows the nested Shape::Type enum)
        @test haskey(structs, "NestedEnumHolder")
        neh_members = Dict(m["name"] => m["offset"] for m in get(structs["NestedEnumHolder"], "members", []))
        @test haskey(neh_members, "before")
        @test haskey(neh_members, "after")
        @test haskey(neh_members, "kind")
        if haskey(neh_members, "after")
            @test parse(Int, neh_members["kind"]) == 0
            @test parse(Int, neh_members["after"]) == 4
            @test parse(Int, neh_members["before"]) == 8
        end

        bases = get(structs["Derived"], "base_classes", [])
        @test length(bases) == 2
        # Extractor sorts by subobject offset: primary base first
        @test get(bases[1], "type", "") == "Base1"
        @test get(bases[1], "offset", -1) == 0
        @test get(bases[2], "type", "") == "Base2"
        @test get(bases[2], "offset", -1) == 16
        @test all(b -> get(b, "virtual", false) == false, bases)
    end

    # ── 2. DWARFParser path + type_info base table ──────────────────────────
    @testset "DWARFParser offsets + type_info emission" begin
        so_path = joinpath(SCRIPT_DIR, "julia", "libmi_test.so")
        @test isfile(so_path)
        vtinfo = RepliBuild.DWARFParser.parse_vtables(so_path)
        @test haskey(vtinfo.classes, "Derived")
        ci = vtinfo.classes["Derived"]
        @test ci.base_classes == ["Base1", "Base2"]
        @test ci.base_offsets == [0, 16]
        @test all(v -> v == false, ci.virtual_bases)

        ir = RepliBuild.JLCSIRGenerator.generate_type_info_ir("Derived", ci, UInt64(0))
        @test occursin("baseNames = [\"Base1\", \"Base2\"]", ir)
        @test occursin("baseOffsets = [0 : i64, 16 : i64]", ir)

        # The emitted op must survive the dialect verifier (libJLCS required)
        if isfile(RepliBuild.MLIRNative.libJLCS)
            ctx = RepliBuild.MLIRNative.create_context()
            try
                mod = RepliBuild.MLIRNative.parse_module(ctx, "module {\n$(ir)\n}")
                @test mod != C_NULL
            finally
                RepliBuild.MLIRNative.destroy_context(ctx)
            end
        end
    end

    # ── 3. Upcast helpers ───────────────────────────────────────────────────
    @testset "upcast helpers" begin
        @test isdefined(MiTest, :Derived_as_Base1)
        @test isdefined(MiTest, :Derived_as_Base2)
        probe = Ptr{Cvoid}(UInt(0x1000))
        @test MiTest.Derived_as_Base1(probe) == probe
        @test Int(MiTest.Derived_as_Base2(probe)) - Int(probe) == 16
    end

    # ── 4. Live calls through the real object ───────────────────────────────
    @testset "live MI calls" begin
        d = MiTest.make_derived()
        p = d isa Ptr ? Ptr{Cvoid}(d) : Ptr{Cvoid}(d.handle)
        @test p != C_NULL

        try
            b2 = MiTest.Derived_as_Base2(p)

            # Primary base: virtual, not overridden — vcall through the
            # primary vtable reaches Base1::get_a
            @test MiTest.Base1_get_a(p) == 111

            # Non-virtual Base2 method (direct symbol call — vcall producer
            # only routes virtuals): with the upcast pointer it reads b; with
            # the raw derived pointer it reads a THROUGH Base2's code
            # (2*222=444 vs 2*111=222). The wrong-`this` variant is pinned
            # deliberately: it is what every call did before upcasts existed.
            @test MiTest.Base2_double_b(b2) == 444
            @test MiTest.Base2_double_b(p) == 222

            # OVERRIDE DISPATCH — the vcall producer's whole point. Derived
            # overrides Base2::get_b (returns b + 1000). Calling the BASE's
            # wrapper on the upcast pointer must reach the override through
            # the secondary vtable's this-adjusting thunk; static dispatch
            # (the pre-producer behavior) returned bare b (222).
            @test MiTest.Base2_get_b(b2) == 1222

            # The override through its own class's wrapper: primary-vtable
            # slot with a derived-relative `this`
            @test MiTest.Derived_get_b(p) == 1222

            # Mutate through the secondary base (virtual, not overridden —
            # secondary vtable entry points straight at Base2::set_b),
            # observe through both dispatch paths
            MiTest.Base2_set_b(b2, Int32(999))
            @test MiTest.Base2_get_b(b2) == 1999
            @test MiTest.Derived_get_sum(p) == 111 + 999 + 3
        finally
            MiTest.free_derived(p)
        end

        # A Base2* that is REALLY a Derived, from the C++ side — the caller
        # has no derived-type knowledge at all; only the vtable knows.
        b2v = MiTest.make_derived_as_base2()
        pb = b2v isa Ptr ? Ptr{Cvoid}(b2v) : Ptr{Cvoid}(b2v.handle)
        @test pb != C_NULL
        try
            @test MiTest.Base2_get_b(pb) == 1222     # override, not 222
            @test MiTest.Base2_double_b(pb) == 444   # non-virtual: static, base-relative
        finally
            # Polymorphic deletion: deleting-dtor in the vtable adjusts back
            MiTest.free_base2(pb)
        end
    end
end

println("MI_VERIFY_DONE")
