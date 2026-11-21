# =============================================================================
# Records with const-qualified members through Tier 2, live (2026-11-21)
#
# test_const_members.jl checks what the generator emits. This builds
# test/const_member_test/names.cpp with the system compiler, wraps it and
# calls it.
#
# The shape is llama.cpp's `LLM_TN::operator()`, which returns `LLM_TN_IMPL`
# by value: five const members tiling 24 bytes. On the 4.6.0 release sweep
# that one thunk failed to lower and took llama.cpp's whole thunk module with
# it. Here: a 24-byte record of const members returned and passed by value
# (MEMORY class on SysV), an 8-byte one (one INTEGER register), and one with
# padding, a volatile member and a const double (INTEGER, SSE).
#
# And the same qualifiers in a signature: `double scaled(const double x,
# const int n)`. A definition's DWARF keeps them, and the thunk typed every
# such parameter and return as a pointer: the double went to an integer
# register and `scaled(2.5, 4)` was 0.0, with no error. A wrong number, so it
# is only found by a callee the system compiler built.
#
# The callees are compiled by clang++, so the values are the compiler's own
# reading of the ABI. The calls run in a child: before the fix the library's
# JIT did not initialise, and a wrong crossing is a crash.
#
# Needs the C++ toolchain and libJLCS.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())

const CM_DIR = joinpath(@__DIR__, "const_member_test")

function cm_build()
    for d in ("build", "julia", ".replibuild_cache", ".debug")
        p = joinpath(CM_DIR, d)
        ispath(p) && RepliBuild._rm_tree(p)
    end
    toml_path = joinpath(CM_DIR, "replibuild.toml")
    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "const_member"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(CM_DIR))"

    [compile]
    flags = ["-std=c++17", "-O1", "-fPIC"]
    source_files = ["$(escape_string(joinpath(CM_DIR, "names.cpp")))"]
    include_dirs = ["$(escape_string(CM_DIR))"]

    [link]
    enable_lto = false
    optimization_level = "0"

    [binary]
    type = "shared"

    [wrap]
    language = "cpp"

    [types]
    strictness = "warn"
    allow_unknown_structs = true

    [cache]
    enabled = false
    """)
    withenv("REPLIBUILD_JULIA_ENV" => "none") do
        redirect_stdout(devnull) do
            RepliBuild.build(toml_path)
        end
        RepliBuild.wrap(toml_path)
    end
    return joinpath(CM_DIR, "julia", "ConstMember.jl")
end

if requires("records with const members through Tier 2, live", :libJLCS, :clangxx)
@testset "Records with const members through Tier 2, live" begin
    wrapper = cm_build()
    @test isfile(wrapper)

    @testset "the library's JIT initialises, and the records cross both ways" begin
        r = run_isolated(`$(joinpath(CM_DIR, "probe.jl")) $wrapper`; timeout = 600)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        @test !occursin("Failed to initialize JIT", r.output)
        @test !occursin("MarshalRetOp", r.output)
        for name in ("module", "name_returned", "name_operator", "name_passed", "pair", "mixed", "byte_body", "qualified_scalars", "throws")
            @test (name, startswith(get(v, name, "no PROBE line"), "PASS")) == (name, true)
        end
    end
    @testset "in the thunk module: fields for const members, bytes for what has no type" begin
        # `.debug/mlir` files are named by content and never rewritten:
        # cm_build emptied the directory, and the child above wrote this one.
        dir = joinpath(CM_DIR, ".debug", "mlir")
        mods = isdir(dir) ? filter(f -> endswith(f, ".mlir"), readdir(dir; join = true)) : String[]
        @test length(mods) == 1
        ir = isempty(mods) ? "" : read(first(mods), String)
        @test occursin("!Struct_Name = !jlcs.c_struct<\"Name\", [!Struct_Kind, !Struct_Kind, !llvm.ptr, i32, i32]", ir)
        @test occursin("!Struct_Pair = !jlcs.c_struct<\"Pair\", [i32, i32]", ir)
        @test !occursin(r"^!Struct_(?:Name|Pair|Mixed) = !llvm\.array<"m, ir)
        # `Wide` is the byte region, and no marshal op touches one.
        @test occursin(r"^!Struct_Wide = !llvm\.array<24 x i8>"m, ir)
        @test occursin("func.func private @_Z8wide_sum4Wide(", ir)      # it has a thunk
        # What a marshal op takes and gives is a struct: `: (operand) -> result`.
        sigs = [String(m.captures[1]) for m in eachmatch(r"jlcs\.marshal_(?:ret|arg) [^\n]* : (\([^\n]*)", ir)]
        @test length(sigs) >= 5
        @test !any(sig -> occursin(r"^\(!llvm\.array<", sig) || occursin(r"-> !llvm\.array<", sig), sigs)
    end
end
end # requires
