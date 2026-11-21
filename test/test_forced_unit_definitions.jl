# =============================================================================
# What the forced unit defines, and what a failed relink leaves (2026-11-21,
# toolchain-free)
#
# Found by the Hub's fmt and clipper2 on the 4.6.0 release sweep: neither
# BUILT. The forced unit (the inline constructors, member functions and
# default-argument calls a build defines since 4.5.1) includes the header each
# forced record is declared in. fmt declares records in `format-inl.h`, its
# implementation file: without `FMT_HEADER_ONLY` the functions in it are not
# inline, and `src/format.cc` is the one unit meant to include it. clipper2's
# recipe lists `clipper.export.h`, which defines the C entry points, in
# `[wrap] shim_headers`. Included a second time, each function was defined
# twice, and the relink failed with "multiple definition of".
#
# Every compile of the unit succeeded; nothing read what it DEFINED. Now the
# unit's IR is read for definitions with external linkage that are not its
# own (`_foreign_strong_definitions`), the header that brings them is dropped
# like one that does not compile, and a relink that still fails falls back to
# the library without the unit (`_forced_unit_refused`).
#
# This file checks the two readers. test_defining_headers_live.jl builds.
# =============================================================================

using Test
using RepliBuild

const _FUD = RepliBuild.Compiler

# One line per linkage a compiler writes, as clang 23 prints them.
const _FUD_IR = """
; ModuleID = 'replibuild_ctors.cpp'
\$_ZN5ShapeC2Ed = comdat any
@gadget_count = dso_local global i32 0, align 4
@_ZL19replibuild_member_0 = internal constant { i64, i64 } { i64 ptrtoint (ptr @_ZNK5Shape4areaEv to i64), i64 0 }, align 8
@.str = private unnamed_addr constant [4 x i8] c"abc\\00", align 1
@extern_var = external global i32, align 4
@weak_var = weak_odr dso_local global i32 0, comdat, align 4
@"quoted name" = dso_local constant i32 3, align 4
@_ZTV5Shape = linkonce_odr dso_local unnamed_addr constant { [3 x ptr] } zeroinitializer, comdat, align 8
define dso_local void @_ZN3fmt3v1211assert_failEPKciS2_(ptr noundef %0, i32 noundef %1, ptr noundef %2) #0 {
define linkonce_odr dso_local void @_ZN5ShapeC2Ed(ptr noundef nonnull align 8 %0, double noundef %1) unnamed_addr #1 comdat align 2 {
define weak_odr dso_local void @_ZN16replibuild_force4ctorI5ShapeJdEEEvPT_DpT0_(ptr noundef %0, double noundef %1) #2 comdat {
define internal void @_ZL6helperv() #0 {
define available_externally noundef i32 @avail() #0 {
define dso_local noundef i32 @_ZN18replibuild_default2d1Ev(ptr noundef %0) #0 {
define hidden %"struct.weak" @returns_a_record_named_weak() #0 {
define dso_local i32 @export_version() #0 {
define linkonce_odr hidden noundef double @_ZNK5Shape4areaEv(ptr noundef nonnull align 8 %0) #1 comdat align 2 {
declare void @declared_only(ptr noundef) #3
declare noundef i32 @_Z12gadget_totalRK6Gadget(ptr noundef nonnull align 4) #3
"""

@testset "The forced unit: what it defines, and a relink that fails" begin

    @testset "_strong_definitions: external linkage, not mergeable" begin
        dir = mktempdir()
        ir = joinpath(dir, "unit.ll")
        write(ir, _FUD_IR)
        strong = _FUD._strong_definitions(ir)
        @test strong == sort(["_ZN3fmt3v1211assert_failEPKciS2_", "_ZN18replibuild_default2d1Ev",
                              "returns_a_record_named_weak", "export_version",
                              "gadget_count", "quoted name"])
        # Not one of: an inline function, a template instantiation, a static,
        # an available_externally copy, a declaration, a vtable, a string.
        for name in ("_ZN5ShapeC2Ed", "_ZN16replibuild_force4ctorI5ShapeJdEEEvPT_DpT0_", "_ZL6helperv", "avail",
                     "_ZNK5Shape4areaEv", "declared_only", "_Z12gadget_totalRK6Gadget", "_ZL19replibuild_member_0",
                     ".str", "extern_var", "weak_var", "_ZTV5Shape")
            @test (name, name in strong) == (name, false)
        end
        # The unit's own entry points are not foreign; everything else strong is.
        foreign = _FUD._foreign_strong_definitions(ir)
        @test foreign == sort(["_ZN3fmt3v1211assert_failEPKciS2_", "returns_a_record_named_weak",
                               "export_version", "gadget_count", "quoted name"])
        @test _FUD._is_forced_unit_own("_ZN18replibuild_default2d1Ev")
        @test !_FUD._is_forced_unit_own("_ZN3fmt3v1211assert_failEPKciS2_")

        # A unit that only instantiates and takes addresses defines nothing foreign.
        clean = joinpath(dir, "clean.ll")
        write(clean, join(filter(l -> !occursin("assert_fail", l) && !occursin("named_weak", l) &&
                                      !occursin("export_version", l) && !occursin("gadget_count = ", l) &&
                                      !occursin("quoted name", l), split(_FUD_IR, '\n')), '\n'))
        @test isempty(_FUD._foreign_strong_definitions(clean))
        @test _FUD._strong_definitions(clean) == ["_ZN18replibuild_default2d1Ev"]
    end

    @testset "_forced_unit_link_reason: the line that says why" begin
        e = ErrorException("Library creation failed: /usr/bin/ld: /tmp/replibuild_ctors-9e21b8.o: in function `Version':\n" *
                           "/x/clipper.export.h:553: multiple definition of `Version'; /tmp/export_tu-4e27d7.o:/x/clipper.export.h:553: first defined here\n" *
                           "clang++: error: linker command failed with exit code 1 (use -v to see invocation)")
        why = _FUD._forced_unit_link_reason(e)
        @test occursin("multiple definition of `Version'", why)
        @test !occursin('\n', why)
        @test occursin("undefined reference", _FUD._forced_unit_link_reason(
            ErrorException("Library creation failed: ld: a.o: undefined reference to `f()'\nclang++: error: linker command failed")))
        # With nothing it recognises it still answers with a line.
        @test _FUD._forced_unit_link_reason(ErrorException("no such file")) == "no such file"
    end

    @testset "_forced_unit_refused: every kept row is skipped with the reason" begin
        row(c, s) = Dict{String,Any}("class" => c, "symbol" => s, "headers" => ["/x/$c.h"])
        plan = Dict{String,Any}("forced" => Any[row("Shape", "_ZN5ShapeC4Ed"), row("Widget", "_ZN6WidgetC4Ei")],
                                "skipped" => Any[Dict{String,Any}("class" => "Old", "symbol" => "_ZN3OldC4Ev", "reason" => "deleted")])
        mplan = Dict{String,Any}("forced" => Any[row("Shape", "_ZNK5Shape4areaEv")], "skipped" => Any[])
        dplan = Dict{String,Any}("candidates" => Any[], "bound" => Any[Dict{String,Any}("symbol" => "_ZN5Shape5scaleEdd", "shim" => "_ZN18replibuild_default2d1E")])
        (p, m, d) = _FUD._forced_unit_refused(plan, mplan, dplan, "multiple definition of `Version'")
        @test isempty(p["forced"]) && isempty(m["forced"])
        @test [String(f["class"]) for f in p["skipped"]] == ["Old", "Shape", "Widget"]
        @test p["skipped"][1]["reason"] == "deleted"      # an earlier skip keeps its own reason
        for f in vcat(p["skipped"][2:end], m["skipped"])
            @test f["reason"] == "the library did not link with the forced unit: multiple definition of `Version'"
            @test sort(collect(keys(f))) == ["class", "reason", "symbol"]
        end
        @test isempty(d["bound"]) && isempty(d["candidates"])
        # The plans handed in are not modified.
        @test length(plan["forced"]) == 2 && length(plan["skipped"]) == 1 && length(dplan["bound"]) == 1
    end
end
