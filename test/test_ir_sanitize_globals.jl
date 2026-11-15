# test_ir_sanitize_globals.jl — sanitize_ir_for_julia's per-line rewrites
# must not assume a type or a name is one token.
#
# The LTO copy of a C library turns every exported global definition into an
# external declaration. The rewrite read the type with a regex whose last
# alternative was `\S+`, so a type that spells its elements with spaces was cut
# at the first one: wasm3's `@preopen = external global [5` and
# `@c_operationsFC = external constant [18`, which no LLVM parses, so
# `<lib>_lto.bc` was never written. The keyword strips for LLVM 19 attributes
# (`range(…)`, `captures(…)`, …) matched right after `@`, so a C function named
# `range` lost its name at its definition and every call. Both are checked
# line by line and by parsing the sanitized module in Julia's own libLLVM.
# No toolchain.

using Test
using RepliBuild

const _SG_C = RepliBuild.Compiler
const _sg = _SG_C.sanitize_ir_for_julia

@testset "IR sanitizer: global types and names" begin

@testset "externalization reads the whole type" begin
    cases = [
        # wasm3's two, verbatim shapes
        "@preopen = local_unnamed_addr global [5 x { i32, [4 x i8], ptr, ptr }] [{ i32, [4 x i8], ptr, ptr } { i32 0, [4 x i8] undef, ptr null, ptr null }], align 16" =>
            "@preopen = external global [5 x { i32, [4 x i8], ptr, ptr }]",
        "@c_operationsFC = constant [18 x { i8, i8, [6 x i8], [4 x ptr], ptr }] zeroinitializer, align 16" =>
            "@c_operationsFC = external constant [18 x { i8, i8, [6 x i8], [4 x ptr], ptr }]",
        # nested arrays
        "@b = dso_local constant [2 x [3 x i32]] [[3 x i32] [i32 1, i32 2, i32 3], [3 x i32] [i32 4, i32 5, i32 6]], align 16" =>
            "@b = external constant [2 x [3 x i32]]",
        # an array of packed structs
        "@pk = global [2 x <{ i32, i8 }>] zeroinitializer, align 1" =>
            "@pk = external global [2 x <{ i32, i8 }>]",
        # a quoted type name with a space in it
        "@q = global %\"class.std::pair<int, int>\" zeroinitializer, align 4" =>
            "@q = external global %\"class.std::pair<int, int>\"",
        # shapes the old rewrite handled keep working
        "@c = global <{ i32, i8 }> <{ i32 1, i8 2 }>, align 1" => "@c = external global <{ i32, i8 }>",
        "@d = global %struct.S { i32 1, ptr null }, align 8"    => "@d = external global %struct.S",
        "@e = dso_local global ptr null, align 8"                => "@e = external global ptr",
        "@f = global <4 x float> zeroinitializer, align 16"     => "@f = external global <4 x float>",
        "@g = global { i32, { i8, i8 } } zeroinitializer"       => "@g = external global { i32, { i8, i8 } }",
        "@i = global i32 0, align 4"                            => "@i = external global i32",
        # what a declaration must keep to name the same object
        "@t = thread_local global i32 7, align 4"               => "@t = external thread_local global i32",
        "@tl = dso_local thread_local(initialexec) global i64 0" => "@tl = external thread_local(initialexec) global i64",
        "@a = addrspace(1) global i32 0"                        => "@a = external addrspace(1) global i32",
        "@ap = global ptr addrspace(1) null"                    => "@ap = external global ptr addrspace(1)",
    ]
    checked = 0
    for (src, want) in cases
        @test _sg(src) == want
        checked += 1
    end
    @test checked == length(cases)
end

@testset "lines that are not exported definitions pass through" begin
    for l in [
        "@x = external global i32, align 4",
        "@y = external global [5 x { i32, [4 x i8], ptr, ptr }]",
        "@h = hidden global [2 x [2 x i8]] zeroinitializer, align 1",
        "@p = internal constant [3 x { i8, i8 }] zeroinitializer",
        "@.str = private unnamed_addr constant [4 x i8] c\"abc\\00\", align 1",
        "@al = alias i32, ptr @global",
    ]
        @test _sg(l) == l
    end
end

@testset "a type that cannot be delimited keeps its definition, with a warning" begin
    bad = "@broken = global [5 x { i32, ptr ] zeroinitializer"
    @test (@test_logs (:warn, r"@broken") _sg(bad)) == bad
end

@testset "attribute strips leave names alone" begin
    for name in ("range", "captures", "inrange", "initializes", "allocptr",
                 "dead_on_unwind", "noinline", "optnone", "ptrtoaddr")
        call = "  %r = call i32 @$name(i32 %x)"
        @test _sg(call) == call
    end
    # the attributes themselves still go
    @test _sg("  %v = load i32, ptr %p, align 4, !range !3") == "  %v = load i32, ptr %p, align 4"
    @test _sg("define void @f(ptr captures(none) %p) {") == "define void @f(ptr %p) {"
    @test _sg("  %c = call range(i32 0, 10) i32 @g()") == "  %c = call i32 @g()"
    @test _sg("  %a = ptrtoaddr ptr %p to i64") == "  %a = ptrtoint ptr %p to i64"
end

@testset "the sanitized module parses in Julia's libLLVM" begin
    ir = """
    %struct.S = type { i32, ptr }
    %"class.std::pair<int, int>" = type { i32, i32 }

    @preopen = local_unnamed_addr global [2 x { i32, [4 x i8], ptr, ptr }] zeroinitializer, align 16
    @ops = constant [1 x { i8, i8, [6 x i8], [4 x ptr], ptr }] zeroinitializer, align 16
    @grid = dso_local global [2 x [3 x i32]] [[3 x i32] [i32 1, i32 2, i32 3], [3 x i32] [i32 4, i32 5, i32 6]], align 16
    @pk = global [2 x <{ i32, i8 }>] zeroinitializer, align 1
    @s = global %struct.S zeroinitializer, align 8
    @pr = global %"class.std::pair<int, int>" zeroinitializer, align 4
    @t = thread_local global i32 7, align 4

    define i32 @range(i32 %x) {
      ret i32 %x
    }

    define i32 @captures(i32 %x) {
      ret i32 %x
    }

    define i32 @get(i32 %i) {
      %a = call i32 @range(i32 %i)
      %b = call i32 @captures(i32 %a)
      %g = load i32, ptr @grid, align 4
      %t = load i32, ptr @t, align 4
      %s = add i32 %b, %g
      %r = add i32 %s, %t
      ret i32 %r
    }
    """
    out = _sg(ir)
    for decl in ("@preopen = external global [2 x { i32, [4 x i8], ptr, ptr }]",
                 "@ops = external constant [1 x { i8, i8, [6 x i8], [4 x ptr], ptr }]",
                 "@grid = external global [2 x [3 x i32]]",
                 "@t = external thread_local global i32",
                 "define i32 @range(i32 %x) {",
                 "%a = call i32 @range(i32 %i)")
        @test occursin(decl, out)
    end
    dir = mktempdir()
    ll = joinpath(dir, "m.ll"); bc = joinpath(dir, "m.bc")
    write(ll, out)
    @test _SG_C._assemble_bitcode_libllvm(ll, bc)
    @test isfile(bc)
end

end
