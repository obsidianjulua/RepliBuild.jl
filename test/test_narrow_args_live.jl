# =============================================================================
# Integer arguments narrower than `int`, through a thunk, live (2026-11-16)
#
# Jolt's `BodyManager::GetNumActiveBodies(EBodyType)` (`enum class EBodyType :
# uint8`) returned 0 for 45 awake bodies, and the PhysicsSystem form returned
# garbage and then segfaulted. The thunk passed the enum as a one-byte record,
# `!llvm.struct<"EBodyType", (i8)>`, which is coerced to a whole register read
# from one byte of storage; the callee, compiled from the enum, takes
# `i8 zeroext` and reads the argument 32 bits wide.
#
# An enum is an integer in C, and an integer narrower than `int` is extended
# to 32 bits by the caller (clang's callees rely on it). The thunk now loads
# the argument at its width, extends it by its signedness, and declares the
# callee taking the `i32`.
#
# This builds test/narrow_arg_test with the system compiler, whose callees
# index arrays and do 32-bit arithmetic with their narrow arguments, wraps it,
# and calls each from a child. On the old lowering the child segfaults.
# Needs clang++ and libJLCS.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const NARROW_ARG_DIR = joinpath(@__DIR__, "narrow_arg_test")

if requires("narrow integer arguments through a thunk, live", :libJLCS, :clangxx)
@testset "Narrow integer arguments through a thunk, live" begin
    toml_path = joinpath(NARROW_ARG_DIR, "replibuild.toml")
    src = joinpath(NARROW_ARG_DIR, "narrow_arg.cpp")

    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "narrowarg"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(NARROW_ARG_DIR))"

    [compile]
    flags = ["-std=c++17", "-fPIC", "-O2", "-fstandalone-debug"]
    source_files = ["$(escape_string(src))"]
    include_dirs = ["$(escape_string(NARROW_ARG_DIR))"]

    [link]
    enable_lto = false
    optimization_level = "2"

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
    redirect_stdout(devnull) do
        RepliBuild.build(toml_path)
    end
    RepliBuild.wrap(toml_path)
    wrapper = joinpath(NARROW_ARG_DIR, "julia", "Narrowarg.jl")
    @test isfile(wrapper)

    @testset "every value, from a child" begin
        r = run_isolated(`$(joinpath(NARROW_ARG_DIR, "probe_narrow_arg.jl")) $wrapper`)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        names = ["enum_uint8_indexes_an_array", "enum_uint8_zero_extended", "enum_int8_sign_extended",
                 "enum_uint16_and_int", "plain_narrow_integers", "narrow_among_wide"]
        for n in names
            @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
        end
        @test count(n -> haskey(v, n), names) == length(names)
    end
    @testset "the thunk module: an integer, extended, never a one-member record" begin
        # The module the child's engine was given, as it wrote it for the
        # debugger (`.debug/mlir/`); the newest is this run's.
        dir = joinpath(NARROW_ARG_DIR, ".debug", "mlir")
        files = isdir(dir) ? filter(f -> endswith(f, ".mlir"), readdir(dir; join = true)) : String[]
        @test !isempty(files)
        ir = isempty(files) ? "" : read(argmax(mtime, files), String)
        decl(sym) = (m = match(Regex("func\\.func private @" * sym * "\\(([^\\n]*)\\) ->"), ir);
                     m === nothing ? "no declaration" : String(m.captures[1]))
        # The callee is declared taking the extended i32, for an enum of any
        # narrow width and for a plain narrow integer.
        @test decl("_ZNK2na5Table5countENS_4KindE") == "!llvm.ptr, i32"
        @test decl("_ZN2na11level_valueENS_5LevelE") == "i32"
        @test decl("_ZN2na12wide16_valueENS_6Wide16E") == "i32"
        @test decl("_ZN2na10full_valueENS_4FullE") == "i32"
        @test decl("_ZN2na8u8_valueEh") == "i32"
        @test decl("_ZN2na10bool_valueEb") == "i32"
        @test decl("_ZN2na3mixElNS_4KindEatbdNS_5LevelE") == "i64, i32, i32, i32, i32, f64, i32"
        # No enum crosses as a record: none in a declaration or a call (the
        # module's header still aliases each enum for use as a record MEMBER).
        crossing = filter(l -> occursin("func.func", l) || occursin("_call", l) || occursin("jlcs.vcall", l), split(ir, '\n'))
        @test !isempty(crossing)
        @test !any(l -> occursin(r"!llvm\.struct<\"(Kind|Level|Wide16|Full)\"", l), crossing)
        # Each extension follows the type's signedness.
        body(sym) = (i = findfirst("func.func @" * sym * "_thunk", ir); i === nothing ? "" :
                     ir[i.start:something(findnext("\n}\n", ir, i.start), i.start:lastindex(ir)).stop])
        exts(sym) = [m.captures[1] * " " * m.captures[2] for m in eachmatch(r"llvm\.(zext|sext) %val_\d+ : (i\d+) to i32", body(sym))]
        @test exts("_ZN2na10kind_indexENS_4KindE") == ["zext i8"]
        @test exts("_ZN2na11level_valueENS_5LevelE") == ["sext i8"]
        @test exts("_ZN2na12wide16_valueENS_6Wide16E") == ["zext i16"]
        @test exts("_ZN2na10full_valueENS_4FullE") == String[]          # 32 bits already
        @test exts("_ZN2na8s8_valueEa") == ["sext i8"]
        @test exts("_ZN2na9s16_valueEs") == ["sext i16"]
        @test exts("_ZN2na10bool_valueEb") == ["zext i1"]
        @test exts("_ZN2na3mixElNS_4KindEatbdNS_5LevelE") == ["zext i8", "sext i8", "zext i16", "zext i1", "sext i8"]
    end

end
end # requires
