#!/usr/bin/env julia
# noexcept routing — which C++ functions may be called through a plain ccall.
#
# A C++ exception thrown through a Tier-3 `ccall` has no landing pad and aborts
# the process (`terminate called after throwing …`, exit 134). So a function may
# take Tier 3 only if no exception can leave it. The routing used to decide that
# from `_scan_noexcept_functions` alone, which regex-matches `name(…) noexcept`
# in source text and returns BARE names. Every function sharing a bare name
# with a noexcept one was routed to ccall: `B::get` beside `A::get() noexcept`,
# every constructor of a class with a `noexcept` move constructor, and
# `noexcept(false)`, which the regex also matched. Hub fmt shipped eight such
# functions, e.g. `fmt::file::dup2(int)`, which throws `system_error` on EBADF.
#
# Now a name is only a candidate. The proof is LLVM's `nounwind` on the
# function's own mangled DEFINITION in the IR the binary is linked from
# (`_nounwind_definitions`). No toolchain needed: the IR here is text.

using Test
using RepliBuild

const NXC = RepliBuild.Compiler

# Verbatim shapes from clang 22 -O2 output of the audit fixture
# (A::get() const noexcept, B::get(int) that throws, Res(Res&&) noexcept,
# Res(int) that throws), plus the edge cases the scanner must survive.
const IR = raw"""
; ModuleID = 'audit_cpp.cpp'
define dso_local noundef i32 @_ZNK5audit1A3getEv(ptr noundef nonnull align 4 dereferenceable(4) %0) local_unnamed_addr #0 align 2 !dbg !10 {
  ret i32 1
}
define dso_local noundef i32 @_ZN5audit1B3getEi(ptr noundef nonnull align 4 dereferenceable(4) %0, i32 noundef %1) local_unnamed_addr #1 align 2 personality ptr @__gxx_personality_v0 !dbg !20 {
  ret i32 0
}
define dso_local void @_ZN5audit3ResC2EOS0_(ptr noundef nonnull align 4 dereferenceable(4) %0, ptr noundef nonnull align 4 dereferenceable(4) %1) unnamed_addr #0 align 2 {
  ret void
}
define dso_local void @_ZN5audit3ResC2Ei(ptr noundef nonnull align 4 dereferenceable(4) %0, i32 noundef %1) unnamed_addr #1 align 2 personality ptr @__gxx_personality_v0 {
  ret void
}
define internal void @"quoted\5Cname"(i32 %0) nounwind {
  ret void
}
define dso_local void @_Z6unwindPFvvE(ptr noundef %0) #1 {
  ret void
}
declare void @_Z9declared0v() #0
attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="none" "no-trapping-math"="true" }
attributes #1 = { mustprogress uwtable "frame-pointer"="none" "target-cpu"="x86-64" }
"""

@testset "noexcept routing" begin

    nounwind = mktempdir() do dir
        f = joinpath(dir, "m.ll")
        write(f, IR)
        NXC._nounwind_definitions([f])
    end

    @testset "_nounwind_definitions reads the IR's own answer" begin
        @test "_ZNK5audit1A3getEv" in nounwind       # declared noexcept, group #0
        @test "_ZN5audit3ResC2EOS0_" in nounwind     # noexcept move ctor
        @test !("_ZN5audit1B3getEi" in nounwind)     # throws — group #1 has no nounwind
        @test !("_ZN5audit3ResC2Ei" in nounwind)     # throwing ctor of the same class
        # Inline function attribute + LLVM's \XX hex escape in a quoted name.
        @test "quoted\\name" in nounwind
        # A `ptr` parameter carries no parens of its own; a parameter list that
        # does (`dereferenceable(4)` above) must not end the scan early.
        @test !("_Z6unwindPFvvE" in nounwind)
        # Declarations are not definitions: nothing is known about their bodies.
        @test !("_Z9declared0v" in nounwind)
        @test length(nounwind) == 3
        # Missing files are skipped, not fatal.
        @test isempty(NXC._nounwind_definitions(["/nonexistent/x.ll"]))
    end

    fn(name, mangled) = Dict{String,Any}("name" => name, "mangled" => mangled)

    @testset "a bare-name candidate is not enough" begin
        funcs = [fn("get", "_ZNK5audit1A3getEv"),    # A::get() const noexcept
                 fn("get", "_ZN5audit1B3getEi"),     # B::get(int) — throws, same bare name
                 fn("Res", "_ZN5audit3ResC2EOS0_"),  # Res(Res&&) noexcept
                 fn("Res", "_ZN5audit3ResC2Ei")]     # Res(int) — throws
        # What the scanner returns for this source: bare names only.
        NXC._mark_noexcept!(funcs, Set(["get", "Res"]), nounwind)
        @test [f["is_noexcept"] for f in funcs] == [true, false, true, false]
    end

    @testset "nounwind alone does not widen Tier 3" begin
        # The optimizer proves plenty of functions non-throwing that nobody
        # declared noexcept. Routing those to ccall is a separate change: the
        # Tier-3 emitter has type gaps the thunks tolerate.
        funcs = [fn("get", "_ZNK5audit1A3getEv")]
        NXC._mark_noexcept!(funcs, Set{String}(), nounwind)
        @test funcs[1]["is_noexcept"] == false
    end

    @testset "no IR means nothing is noexcept" begin
        # Ingest mode: no module to read, so every C++ function goes to Tier 2.
        funcs = [fn("get", "_ZNK5audit1A3getEv")]
        NXC._mark_noexcept!(funcs, Set(["get"]), nothing)
        @test funcs[1]["is_noexcept"] == false
    end

    @testset "wired into metadata extraction" begin
        src = read(joinpath(@__DIR__, "..", "src", "Builder", "Compiler.jl"), String)
        body = match(r"function extract_compilation_metadata\(.*?\nend\n"s, src)
        @test body !== nothing
        @test occursin("_mark_noexcept!(functions, candidates, nounwind)", body.match)
        cp = match(r"function compile_project\(.*?\nend\n"s, src)
        @test occursin("_nounwind_definitions(", cp.match)
    end
end
