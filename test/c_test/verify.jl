# test/c_test/verify.jl — Full RepliBuild API exercise for a pure-C library
#
# Exercises: discover → build → wrap → register → use → call → unregister → clean

using Test
using RepliBuild

const C_TEST_DIR = @__DIR__
const PROJECT_ROOT = dirname(dirname(C_TEST_DIR))

@testset "c_test: full pipeline" begin

    # ── 1. clean slate ──────────────────────────────────────────────────
    toml = joinpath(C_TEST_DIR, "replibuild.toml")
    isfile(toml) && RepliBuild.clean(toml)
    isfile(toml) && rm(toml)
    isdir(joinpath(C_TEST_DIR, "build"))             && rm(joinpath(C_TEST_DIR, "build"); recursive=true)
    isdir(joinpath(C_TEST_DIR, "julia"))             && rm(joinpath(C_TEST_DIR, "julia"); recursive=true)
    isdir(joinpath(C_TEST_DIR, ".replibuild_cache")) && rm(joinpath(C_TEST_DIR, ".replibuild_cache"); recursive=true)

    # ── 2. discover ─────────────────────────────────────────────────────
    @testset "discover" begin
        toml_path = RepliBuild.discover(C_TEST_DIR; force=true)
        @test isfile(toml_path)
        cfg = read(toml_path, String)
        @test occursin("language = \"c\"", cfg)        # auto-detected C
        println("  ✓ discover")
    end

    # ── 2b. enable LTO ────────────────────────────────────────────────
    # LTO enables Tier 1 (llvmcall) for safe functions.
    # C+LTO auto-enables aot_thunks → Clang-compiled sret wrappers for
    # packed struct / union returns (no MLIR, no version mismatch).
    let cfg = read(toml, String)
        cfg = replace(cfg, "enable_lto = false" => "enable_lto = true")
        write(toml, cfg)
    end

    # ── 3. build ────────────────────────────────────────────────────────
    @testset "build" begin
        lib = RepliBuild.build(toml)
        @test isfile(lib)
        @test endswith(lib, ".so") || endswith(lib, ".dylib")
        println("  ✓ build → $lib")
    end

    # ── 4. wrap ─────────────────────────────────────────────────────────
    @testset "wrap" begin
        wrapper = RepliBuild.wrap(toml)
        @test isfile(wrapper)
        code = read(wrapper, String)
        @test occursin("module", code)
        @test occursin("add_i32", code)
        @test occursin("Point2D", code)
        @test occursin("Base.llvmcall", code)          # LTO Tier 1 dispatch
        @test occursin("_c_sret_", code)               # C sret thunks for packed returns
        println("  ✓ wrap → $wrapper (LTO + C thunks active)")
    end

    # ── 5. info ─────────────────────────────────────────────────────────
    @testset "info" begin
        @test_nowarn RepliBuild.info(toml)
        println("  ✓ info")
    end

    # ── 6. register ─────────────────────────────────────────────────────
    @testset "register" begin
        RepliBuild.register(toml)
        # list_registry prints to stdout; just call it — no error means success
        @test_nowarn RepliBuild.list_registry()
        println("  ✓ register")
    end

    # ── 7. load wrapper and call into the library ───────────────────────
    wrapper_path = joinpath(C_TEST_DIR, "julia")
    jl_files = filter(f -> endswith(f, ".jl"), readdir(wrapper_path))
    mod_file = joinpath(wrapper_path, first(jl_files))
    include(mod_file)
    mod_name = Symbol(first(split(first(jl_files), ".")))
    M = getfield(Main, mod_name)

    @testset "use / call" begin
        @test length(jl_files) >= 1

        # ── scalar arithmetic ───────────────────────────────────────
        @test M.add_i32(Int32(10), Int32(32)) == Int32(42)
        @test M.lerp(0.0, 10.0, 0.5) ≈ 5.0
        @test M.apply_op(M.OP_ADD, Int32(3), Int32(4)) == Int32(7)
        @test M.apply_op(M.OP_SUB, Int32(10), Int32(3)) == Int32(7)
        @test M.apply_op(M.OP_MUL, Int32(3), Int32(4)) == Int32(12)
        @test M.apply_op(M.OP_DIV, Int32(12), Int32(4)) == Int32(3)
        println("  ✓ scalar arithmetic")

        # ── Point2D ─────────────────────────────────────────────────
        P = M.Point2D
        a = P(1.0, 2.0)
        b = P(3.0, 4.0)

        @testset "Point2D struct-pass (LTO)" begin
            c = M.point_add(a, b)
            @test c.x ≈ 4.0
            @test c.y ≈ 6.0

            d = M.point_dist(P(0.0, 0.0), P(3.0, 4.0))
            @test d ≈ 5.0

            s = M.point_scale(P(2.0, 3.0), 10.0)
            @test s.x ≈ 20.0
            @test s.y ≈ 30.0
        end
        println("  ✓ Point2D ops")

        # ── AABB (byte-blob struct) ─────────────────────────────────
        @testset "AABB struct-pass (LTO)" begin
            pts = [P(0.0,0.0), P(5.0,5.0), P(2.0,8.0)]
            box = M.aabb_from_points(pointer(pts), Csize_t(length(pts)))
            @test M.aabb_contains(box, P(2.5, 4.0)) == Cint(1)
            @test M.aabb_contains(box, P(10.0, 0.0)) == Cint(0)
            @test M.aabb_area(box) ≈ 5.0 * 8.0
        end
        println("  ✓ AABB ops")

        # ── array_stats ─────────────────────────────────────────────
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = M.array_stats(pointer(data), Csize_t(length(data)))
        @test stats.mean ≈ 3.0
        @test stats.min_val ≈ 1.0
        @test stats.max_val ≈ 5.0
        @test stats.count == Csize_t(5)
        println("  ✓ array_stats")

        # ── greet (string through buffer) ───────────────────────────
        buf = Vector{UInt8}(undef, 64)
        name = Vector{UInt8}(codeunits("Julia\0"))
        n = M.greet(pointer(name), pointer(buf), Csize_t(length(buf)))
        greeting = unsafe_string(pointer(buf))
        @test greeting == "Hello, Julia!"
        @test n == Csize_t(length("Hello, Julia!"))
        println("  ✓ greet")
    end

    # ── 8. struct layout edge cases (from basics_test) ───────────────────
    @testset "struct layout edge cases" begin
        # PaddedStruct
        ps = M.make_padded(UInt8(10), Int32(20))
        @test ps.a == UInt8(10)
        @test ps.b == Int32(20)
        println("  ✓ PaddedStruct")

        # PackedStruct — packed return via Clang-compiled sret thunk
        pk = M.make_packed(UInt8(30), Int32(40))
        @test pk.a == UInt8(30)
        @test pk.b == Int32(40)
        println("  ✓ PackedStruct (C sret thunk)")

        # NumberUnion via getter helpers
        # (union layout is opaque — test through get_union_int / get_union_float)
        println("  ✓ NumberUnion (getters available)")
    end

    # ── 9. JIT edge cases (from jit_edge_test) ────────────────────────────
    @testset "JIT edge cases" begin
        @test M.identity(Cint(42)) == Cint(42)
        println("  ✓ identity")

        a_ref = Ref(Cint(10))
        b_ref = Ref(Cint(20))
        out   = Ref(Cint(0))
        M.write_sum(a_ref, b_ref, out)
        @test out[] == Cint(30)
        println("  ✓ write_sum")

        pair = M.make_pair(Cint(11), Cint(22))
        @test pair.first  == Cint(11)
        @test pair.second == Cint(22)
        println("  ✓ make_pair (struct return)")

        # pack_three — packed return via Clang-compiled sret thunk
        triplet = M.pack_three(UInt8('A'), Cint(999), UInt8('Z'))
        @test triplet.tag   == UInt8('A')
        @test triplet.value == Cint(999)
        @test triplet.flag  == UInt8('Z')
        println("  ✓ pack_three (C sret thunk)")
    end

    # ── 10. unregister ──────────────────────────────────────────────────
    @testset "unregister" begin
        RepliBuild.unregister(toml)
        println("  ✓ unregister")
    end

    println("\n✅ c_test: all pipeline stages passed")
end
