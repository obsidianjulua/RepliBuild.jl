#!/usr/bin/env julia
# test/test_registry.jl — Registry System Tests
#
# Tests the PackageRegistry module: register, unregister, list_registry,
# TOML hashing, build artifact caching, and environment check caching.
#
# Usage:  julia --project=. test/test_registry.jl
#
# These tests use REPLIBUILD_HOME pointed at a temp directory for isolation.

using Test
using RepliBuild
using TOML
using Dates

const TEST_DIR = @__DIR__
const PROJECT_ROOT = dirname(TEST_DIR)

# Access internals via the PackageRegistry submodule
const PR = RepliBuild.PackageRegistry
const CM = RepliBuild.ConfigurationManager
const ED = RepliBuild.EnvironmentDoctor

# ── Helpers ──────────────────────────────────────────────────────────────────

"""Set up an isolated registry home in a temp directory and return its path."""
function with_temp_registry(f::Function)
    mktempdir() do tmpdir
        old_home = get(ENV, "REPLIBUILD_HOME", nothing)
        ENV["REPLIBUILD_HOME"] = tmpdir
        try
            f(tmpdir)
        finally
            if isnothing(old_home)
                delete!(ENV, "REPLIBUILD_HOME")
            else
                ENV["REPLIBUILD_HOME"] = old_home
            end
        end
    end
end

"""Capture stdout from a zero-arg function that prints directly."""
function capture_stdout(f::Function)::String
    old_stdout = stdout
    rd, wr = redirect_stdout()
    try
        f()
    finally
        redirect_stdout(old_stdout)
        close(wr)
    end
    return read(rd, String)
end
function make_test_toml(dir::String; name="test_pkg", lto=false, extra="")
    toml_path = joinpath(dir, "replibuild.toml")
    content = """
    [project]
    name = "$name"
    uuid = "00000000-0000-0000-0000-000000000001"
    root = "$dir"

    [compile]
    source_files = []
    include_dirs = []
    flags = ["-std=c++17", "-fPIC"]

    [link]
    enable_lto = $lto
    optimization_level = "0"

    [binary]
    type = "shared"

    [wrap]
    style = "module"

    [cache]
    enabled = true
    $extra
    """
    write(toml_path, content)
    return toml_path
end

# =============================================================================
# 1. REGISTRY UNIT TESTS
# =============================================================================

@testset "Registry Unit Tests" begin

    @testset "register and verify entry" begin
        with_temp_registry() do home
            mktempdir() do proj
                toml = make_test_toml(proj; name="my_lib")
                entry = PR.register(toml)

                @test entry.name == "my_lib"
                @test !isempty(entry.hash)
                @test entry.verified == false
                @test isfile(entry.toml_path)
                @test startswith(entry.toml_path, joinpath(home, "registry"))

                # Index should contain the entry
                index = PR._load_index()
                @test haskey(index.entries, "my_lib")
                @test index.entries["my_lib"].hash == entry.hash
            end
        end
    end

    @testset "register with explicit name override" begin
        with_temp_registry() do home
            mktempdir() do proj
                toml = make_test_toml(proj; name="original_name")
                entry = PR.register(toml; name="custom_name")

                @test entry.name == "custom_name"
                index = PR._load_index()
                @test haskey(index.entries, "custom_name")
                @test !haskey(index.entries, "original_name")
            end
        end
    end

    @testset "unregister removes entry and stored TOML" begin
        with_temp_registry() do home
            mktempdir() do proj
                toml = make_test_toml(proj; name="ephemeral")
                entry = PR.register(toml)

                stored_path = entry.toml_path
                @test isfile(stored_path)

                PR.unregister("ephemeral")

                index = PR._load_index()
                @test !haskey(index.entries, "ephemeral")
                @test !isfile(stored_path)
            end
        end
    end

    @testset "unregister non-existent package warns" begin
        with_temp_registry() do home
            @test_logs (:warn, r"not found") PR.unregister("ghost_pkg")
        end
    end

    @testset "register → unregister → re-register cycle" begin
        with_temp_registry() do home
            mktempdir() do proj
                toml = make_test_toml(proj; name="cycled")

                e1 = PR.register(toml)
                PR.unregister("cycled")

                index = PR._load_index()
                @test !haskey(index.entries, "cycled")

                e2 = PR.register(toml)
                @test e2.hash == e1.hash  # same content → same hash

                index = PR._load_index()
                @test haskey(index.entries, "cycled")
            end
        end
    end

    @testset "content-addressed dedup (same content, different path)" begin
        with_temp_registry() do home
            mktempdir() do proj1
                mktempdir() do proj2
                    toml1 = make_test_toml(proj1; name="dedup_a")
                    toml2 = make_test_toml(proj2; name="dedup_b")

                    e1 = PR.register(toml1)
                    e2 = PR.register(toml2)

                    # Different names but hashes depend on content (which differs by name/root)
                    @test e1.name == "dedup_a"
                    @test e2.name == "dedup_b"

                    index = PR._load_index()
                    @test length(index.entries) == 2
                end
            end
        end
    end

    @testset "register identical content is idempotent" begin
        with_temp_registry() do home
            mktempdir() do proj
                toml = make_test_toml(proj; name="idem")

                e1 = PR.register(toml)
                e2 = PR.register(toml)

                @test e1.hash == e2.hash
                @test e1.toml_path == e2.toml_path

                index = PR._load_index()
                @test length(index.entries) == 1
            end
        end
    end

    @testset "list_registry shows registered packages" begin
        with_temp_registry() do home
            mktempdir() do proj
                toml = make_test_toml(proj; name="visible_pkg")
                PR.register(toml)

                output = capture_stdout(PR.list_registry)
                @test contains(output, "visible_pkg")
                @test contains(output, "local")
            end
        end
    end

    @testset "list_registry empty registry" begin
        with_temp_registry() do home
            output = capture_stdout(PR.list_registry)
            @test contains(output, "no packages registered")
        end
    end

    @testset "register multiple packages then list" begin
        with_temp_registry() do home
            mktempdir() do proj1
                mktempdir() do proj2
                    make_test_toml(proj1; name="alpha")
                    make_test_toml(proj2; name="beta")

                    PR.register(joinpath(proj1, "replibuild.toml"))
                    PR.register(joinpath(proj2, "replibuild.toml"))

                    index = PR._load_index()
                    @test length(index.entries) == 2
                    @test haskey(index.entries, "alpha")
                    @test haskey(index.entries, "beta")

                    output = capture_stdout(PR.list_registry)
                    @test contains(output, "alpha")
                    @test contains(output, "beta")
                end
            end
        end
    end

    @testset "error: register missing TOML file" begin
        with_temp_registry() do home
            @test_throws ErrorException PR.register("/nonexistent/path.toml")
        end
    end

    @testset "error: register TOML without project name and no override" begin
        with_temp_registry() do home
            mktempdir() do proj
                toml_path = joinpath(proj, "replibuild.toml")
                write(toml_path, """
                [compile]
                source_files = []
                """)
                @test_throws ErrorException PR.register(toml_path)
            end
        end
    end

    @testset "register preserves git dependency info" begin
        with_temp_registry() do home
            mktempdir() do proj
                toml_path = joinpath(proj, "replibuild.toml")
                write(toml_path, """
                [project]
                name = "with_deps"
                root = "."

                [dependencies.mylib]
                type = "git"
                url = "https://github.com/example/mylib.git"
                tag = "v1.0.0"

                [compile]
                source_files = []

                [link]
                enable_lto = false
                """)

                entry = PR.register(toml_path)
                @test contains(entry.source_url, "mylib.git")
                @test entry.source_tag == "v1.0.0"
            end
        end
    end

    @testset "index persistence across load/save" begin
        with_temp_registry() do home
            mktempdir() do proj
                toml = make_test_toml(proj; name="persist_test")
                PR.register(toml)

                # Load index from disk (simulating a fresh session)
                idx1 = PR._load_index()
                @test haskey(idx1.entries, "persist_test")

                # Save and reload
                PR._save_index(idx1)
                idx2 = PR._load_index()
                @test idx2.entries["persist_test"].hash == idx1.entries["persist_test"].hash
                @test idx2.entries["persist_test"].name == "persist_test"
            end
        end
    end

end

# =============================================================================
# 2. TOML HASHING & CACHING TESTS
# =============================================================================

@testset "TOML Hashing & Caching" begin

    @testset "hash_toml produces stable hashes" begin
        mktempdir() do dir
            toml = make_test_toml(dir; name="hash_stable")
            h1 = PR.hash_toml(toml)
            h2 = PR.hash_toml(toml)
            @test h1 == h2
            @test length(h1) == 64  # SHA256 hex
        end
    end

    @testset "hash_toml normalizes formatting" begin
        mktempdir() do dir
            # Write two files with same content but different formatting
            toml1 = joinpath(dir, "a.toml")
            toml2 = joinpath(dir, "b.toml")

            write(toml1, """
            [project]
            name = "fmt_test"
            root = "."

            [compile]
            flags = ["-O2"]
            """)

            # Same content, different whitespace/ordering within sections
            write(toml2, """
            [project]
            name="fmt_test"
            root="."
            [compile]
            flags=["-O2"]
            """)

            h1 = PR.hash_toml(toml1)
            h2 = PR.hash_toml(toml2)
            @test h1 == h2  # normalized (parsed + reserialized)
        end
    end

    @testset "hash_toml differs for different content" begin
        mktempdir() do dir
            t1 = make_test_toml(dir; name="pkg_a")
            h1 = PR.hash_toml(t1)

            # Overwrite with different content
            t2 = make_test_toml(dir; name="pkg_b")
            h2 = PR.hash_toml(t2)

            @test h1 != h2
        end
    end

    @testset "hash_toml error on missing file" begin
        @test_throws ErrorException PR.hash_toml("/nonexistent.toml")
    end

    @testset "build artifact cache round-trip" begin
        with_temp_registry() do home
            mktempdir() do output_dir
                # Simulate build artifacts
                write(joinpath(output_dir, "libtest.so"), "fake library")
                write(joinpath(output_dir, "compilation_metadata.json"), "{}")
                write(joinpath(output_dir, "TestWrapper.jl"), "module TestWrapper end")

                fake_hash = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"

                # Store
                PR._store_build(fake_hash, output_dir)
                @test PR._has_cached_build(fake_hash)

                # Verify files are in cache
                cache_dir = PR._cached_build_dir(fake_hash)
                @test isfile(joinpath(cache_dir, "libtest.so"))
                @test isfile(joinpath(cache_dir, "compilation_metadata.json"))
                @test isfile(joinpath(cache_dir, "TestWrapper.jl"))
            end
        end
    end

    @testset "no cached build for unknown hash" begin
        with_temp_registry() do home
            @test !PR._has_cached_build("0000000000000000000000000000000000000000000000000000000000000000")
        end
    end

    @testset "environment check caching" begin
        with_temp_registry() do home
            # Initially no cache
            @test isnothing(PR._cached_env_check())

            # Create a fake status and cache it
            status = ED.ToolchainStatus(ED.ToolStatus[], true, true, false)
            PR._cache_env_check(status)

            # Read it back
            cached = PR._cached_env_check()
            @test !isnothing(cached)
            @test cached.ready == true
            @test cached.tier1_ready == true
            @test cached.tier2_ready == false
        end
    end

    @testset "environment cache respects TTL" begin
        with_temp_registry() do home
            # Write a cache entry with an old timestamp
            cache_path = joinpath(home, "toolchain.toml")
            old_time = now() - Dates.Hour(25)
            data = Dict{String, Any}(
                "cached_at" => string(old_time),
                "ready" => true,
                "tier1_ready" => true,
                "tier2_ready" => false,
                "llvm_path" => "",
                "llvm_version" => "",
            )
            open(cache_path, "w") do io
                TOML.print(io, data)
            end

            # Should be invalidated (>24h)
            @test isnothing(PR._cached_env_check())
        end
    end

end

# =============================================================================
# 3. REGISTRY INTEGRATION TEST — register → build → wrap → use → unregister
# =============================================================================

@testset "Registry Integration (basics_test)" begin
    basics_dir = joinpath(TEST_DIR, "basics_test")
    if !isdir(basics_dir)
        @warn "basics_test/ not found — skipping registry integration"
        return
    end

    # Make sure basics_test has a TOML (may need discover)
    basics_toml = joinpath(basics_dir, "replibuild.toml")
    if !isfile(basics_toml)
        RepliBuild.discover(basics_dir; force=true)
    end

    with_temp_registry() do home
        @testset "register basics_test" begin
            entry = PR.register(basics_toml; name="basics_test_reg")
            @test entry.name == "basics_test_reg"
            @test isfile(entry.toml_path)
            @test !entry.verified
        end

        @testset "list shows basics_test" begin
            output = capture_stdout(PR.list_registry)
            @test contains(output, "basics_test_reg")
        end

        @testset "unregister basics_test" begin
            PR.unregister("basics_test_reg")
            index = PR._load_index()
            @test !haskey(index.entries, "basics_test_reg")

            output = capture_stdout(PR.list_registry)
            @test contains(output, "no packages registered")
        end
    end
end

println("\n✓ All registry tests passed")
