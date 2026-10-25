#!/usr/bin/env julia
# test/test_registry.jl — Registry System Tests
#
# Tests the PackageRegistry module: register, unregister, list_registry,
# TOML hashing, build artifact caching, vendoring, the Julia package around a
# vendored build, and environment check caching.
#
# Usage:  julia --project=. test/test_registry.jl
#
# These tests use REPLIBUILD_HOME pointed at a temp directory, and temp Julia
# environments, for isolation. Pkg runs offline in them: only installed
# versions resolve, so nothing is downloaded.

using Test
using RepliBuild
using TOML
using Dates
using UUIDs
import Pkg

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const TEST_DIR = @__DIR__
const PROJECT_ROOT = dirname(TEST_DIR)

# Access internals via the PackageRegistry submodule
const PR = RepliBuild.PackageRegistry
const CM = RepliBuild.ConfigurationManager
const ED = RepliBuild.EnvironmentDoctor

# ── Helpers ──────────────────────────────────────────────────────────────────

"""Record a project without building it or touching a Julia environment."""
register_only(toml::String; kw...) = PR.register(toml; build=false, env=nothing, kw...)

"""
A project with a fake build in `julia/`: what `build` + `wrap` leave there, for
testing what `register` does after building without a toolchain.
"""
function make_fake_built_project(dir::String; name="fakewrap", module_name="Fakewrap")
    mkpath(joinpath(dir, "julia"))
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "$name"
    root = "."

    [compile]
    source_files = []

    [wrap]
    language = "c"
    """)
    write(joinpath(dir, "julia", "$module_name.jl"), """
    module $module_name
    using Libdl
    import RepliBuild
    ok() = 42
    vendored_from() = @__DIR__
    end
    """)
    write(joinpath(dir, "julia", "lib$name." * Base.Libc.Libdl.dlext), "not a library")
    write(joinpath(dir, "julia", "compilation_metadata.json"), "{}")
    return joinpath(dir, "replibuild.toml")
end

"""Vendor the fake build as `register` does once it has built (no toolchain)."""
function vendor_fake!(entry)
    cfg = CM.load_config(entry.toml_path)
    h = PR.hash_config(cfg)   # `use` hashes the same: this project has no dependencies to resolve
    PR._store_build(h, CM.get_output_path(cfg))
    return PR._publish!(entry, h, CM.get_module_name(cfg); verbose=false)
end

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
    # escape_string: `root` is a real path, and TOML basic strings take escapes.
    # A Windows tempdir (`C:\\Users\\...`) otherwise reaches the parser as `\\U`
    # and dies as "invalid unicode scalar".
    root = "$(escape_string(dir))"

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
                entry = register_only(toml)

                @test entry.name == "my_lib"
                @test !isempty(entry.hash)
                @test entry.verified == false
                # The entry points at the project itself, not a copy.
                @test entry.toml_path == abspath(toml)
                @test entry.project_root == proj
                @test isempty(entry.build_hash) && isempty(entry.julia_env)

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
                entry = register_only(toml; name="custom_name")

                @test entry.name == "custom_name"
                index = PR._load_index()
                @test haskey(index.entries, "custom_name")
                @test !haskey(index.entries, "original_name")
            end
        end
    end

    @testset "unregister removes the entry and keeps the project's TOML" begin
        with_temp_registry() do home
            mktempdir() do proj
                toml = make_test_toml(proj; name="ephemeral")
                entry = register_only(toml)
                @test entry.toml_path == abspath(toml)

                PR.unregister("ephemeral")

                index = PR._load_index()
                @test !haskey(index.entries, "ephemeral")
                @test isfile(toml)        # the project's own file, never the registry's to delete
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

                e1 = register_only(toml)
                PR.unregister("cycled")

                index = PR._load_index()
                @test !haskey(index.entries, "cycled")

                e2 = register_only(toml)
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

                    e1 = register_only(toml1)
                    e2 = register_only(toml2)

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

                e1 = register_only(toml)
                e2 = register_only(toml)

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
                register_only(toml)

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

                    register_only(joinpath(proj1, "replibuild.toml"))
                    register_only(joinpath(proj2, "replibuild.toml"))

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
            @test_throws ErrorException register_only("/nonexistent/path.toml")
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
                @test_throws ErrorException register_only(toml_path)
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

                entry = register_only(toml_path)
                @test contains(entry.source_url, "mylib.git")
                @test entry.source_tag == "v1.0.0"
            end
        end
    end

    @testset "index persistence across load/save" begin
        with_temp_registry() do home
            mktempdir() do proj
                toml = make_test_toml(proj; name="persist_test")
                register_only(toml)

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

    @testset "generator fingerprint feeds the config hash" begin
        # hash_config mixes this in so cached builds are invalidated when the
        # codegen that produced them changes (2026-07-10 audit: use() served
        # wrappers from months-old generators forever).
        fp = PR._generator_fingerprint()
        @test startswith(fp, "replibuild-")
        @test occursin(string(pkgversion(RepliBuild)), fp)
        @test fp == PR._generator_fingerprint()  # stable within a session
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

    # The dirty digest covers the generator's inputs only. On a dev checkout
    # CLAUDE.md and CHANGELOG.md are dirty most of the time, and hashing the
    # whole tree re-vendored every build after every note. A scratch git repo
    # stands in for the checkout, so this test never touches the real one.
    @testset "dirty digest follows src/ and Project.toml, not docs" begin
        mktempdir() do repo
            run(pipeline(`git -C $repo init -q`; stdout=devnull, stderr=devnull))
            run(`git -C $repo config user.email t@example.com`)
            run(`git -C $repo config user.name t`)
            run(`git -C $repo config commit.gpgsign false`)
            mkpath(joinpath(repo, "src")); mkpath(joinpath(repo, "docs"))
            write(joinpath(repo, "src", "a.jl"), "a = 1\n")
            write(joinpath(repo, "Project.toml"), "name = \"X\"\nversion = \"0.1.0\"\n")
            write(joinpath(repo, "CLAUDE.md"), "notes\n")
            write(joinpath(repo, "docs", "d.md"), "doc\n")
            run(`git -C $repo add -A`)
            run(pipeline(`git -C $repo commit -q -m init`; stdout=devnull, stderr=devnull))
            @test PR._get_git_dirty_digest(repo) == ""
            # Docs and notes: not the generator's.
            write(joinpath(repo, "CLAUDE.md"), "notes, edited\n")
            write(joinpath(repo, "docs", "d.md"), "doc, edited\n")
            write(joinpath(repo, "scratch.txt"), "untracked\n")
            @test PR._get_git_dirty_digest(repo) == ""
            # A source edit is.
            write(joinpath(repo, "src", "a.jl"), "a = 2\n")
            d1 = PR._get_git_dirty_digest(repo)
            @test startswith(d1, "+dirty-")
            # And so is an untracked source file, and a Project.toml edit; each
            # changes the digest.
            write(joinpath(repo, "src", "b.jl"), "b = 1\n")
            d2 = PR._get_git_dirty_digest(repo)
            @test startswith(d2, "+dirty-") && d2 != d1
            write(joinpath(repo, "Project.toml"), "name = \"X\"\nversion = \"0.2.0\"\n")
            d3 = PR._get_git_dirty_digest(repo)
            @test startswith(d3, "+dirty-") && d3 != d2
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

@testset "Registry Integration (c_test)" begin
    ctest_dir = joinpath(TEST_DIR, "c_test")
    if !isdir(ctest_dir)
        @warn "c_test/ not found — skipping registry integration"
        return
    end

    # Make sure c_test has a TOML (may need discover)
    ctest_toml = joinpath(ctest_dir, "replibuild.toml")
    if !isfile(ctest_toml)
        RepliBuild.discover(ctest_dir; force=true)
    end

    with_temp_registry() do home
        @testset "register c_test" begin
            entry = register_only(ctest_toml; name="c_test_reg")
            @test entry.name == "c_test_reg"
            @test isfile(entry.toml_path)
            @test !entry.verified
        end

        @testset "list shows c_test" begin
            output = capture_stdout(PR.list_registry)
            @test contains(output, "c_test_reg")
        end

        @testset "unregister c_test" begin
            PR.unregister("c_test_reg")
            index = PR._load_index()
            @test !haskey(index.entries, "c_test_reg")

            output = capture_stdout(PR.list_registry)
            @test contains(output, "no packages registered")
        end
    end
end

@testset "registry cache and wrapper filename" begin
    @testset "nested headers change hash_config" begin
        mktempdir() do dir
            inc = joinpath(dir, "include", "vecx")
            mkpath(inc)
            write(joinpath(inc, "cfg.h"), "#define VECX_SCALE 2\n")
            write(joinpath(dir, "replibuild.toml"), """
            [project]
            name = "vecx"
            root = "$(replace(dir, '\\' => '/'))"
            [compile]
            source_files = []
            include_dirs = ["$(replace(joinpath(dir, "include"), '\\' => '/'))"]
            [wrap]
            language = "c"
            """)
            cfg = CM.load_config(joinpath(dir, "replibuild.toml"))
            h1 = PR.hash_config(cfg)
            write(joinpath(inc, "cfg.h"), "#define VECX_SCALE 3\n")
            @test PR.hash_config(cfg) != h1
        end
    end

    @testset "wrapper filename follows get_module_name" begin
        mktempdir() do dir
            write(joinpath(dir, "replibuild.toml"), """
            [project]
            name = "vec2d"
            root = "$(replace(dir, '\\' => '/'))"
            [compile]
            source_files = []
            [wrap]
            language = "c"
            """)
            cfg = CM.load_config(joinpath(dir, "replibuild.toml"))
            @test CM.get_module_name(cfg) == "Vec2d"
            @test CM.get_module_name(cfg) * ".jl" != "Vec2D.jl"
        end
    end

    @testset "relative root is the TOML's directory" begin
        mktempdir() do dir
            write(joinpath(dir, "replibuild.toml"), """
            [project]
            name = "relroot"
            root = "."
            [compile]
            source_files = []
            """)
            cfg = CM.load_config(joinpath(dir, "replibuild.toml"))
            @test cfg.project.root == abspath(dir)
        end
    end
end

# ── a registered project builds from its own tree ────────────────────────────
# The registry used to copy a project's sources into itself and build the copy,
# because the stored TOML's relative root pointed there. A missed input built a
# different library (audit F16: Hub expat compiled against the system's
# expat_config.h), two projects with the same relative paths overwrote each
# other, and edits never reached `use`. The entry now points at the project.
# Pure file operations: no toolchain.
@testset "a registered project builds from its own tree" begin
    with_temp_registry() do home
        mktempdir() do sb
            proj = joinpath(sb, "proj")
            for d in ("config", joinpath("include", "lib"), "src")
                mkpath(joinpath(proj, d))
            end
            write(joinpath(proj, "config", "lib_config.h"), "#define LIB_CFG 1\n")
            write(joinpath(proj, "include", "lib", "lib.h"), "int lib(void);\n")
            write(joinpath(proj, "src", "lib.c"), "int lib(void){return 1;}\n")
            toml = joinpath(proj, "replibuild.toml")
            write(toml, """
            [project]
            name = "ownlib"
            root = "."

            [compile]
            source_files = ["src/lib.c"]
            include_dirs = ["config", "include"]

            [wrap]
            language = "c"
            """)

            entry = register_only(toml)
            @test entry.toml_path == abspath(toml)
            @test entry.project_root == proj
            cfg = CM.load_config(entry.toml_path)
            @test cfg.project.root == abspath(proj)
            @test isfile(joinpath(cfg.project.root, "config", "lib_config.h"))
            @test all(isdir, cfg.compile.include_dirs)
            # Nothing of the project is copied into the registry.
            copied = [f for (_, _, fs) in walkdir(home) for f in fs if f in ("lib_config.h", "lib.h", "lib.c")]
            @test isempty(copied)
        end
    end
end

@testset "twin projects stay apart" begin
    # Two projects with the same relative paths: with copies in one shared
    # registry folder, `use("twin_a")` loaded twin_b's library.
    with_temp_registry() do home
        mktempdir() do sb
            for (name, n) in (("twin_a", 1), ("twin_b", 2))
                proj = joinpath(sb, name)
                mkpath(joinpath(proj, "src"))
                write(joinpath(proj, "src", "lib.c"), "int f(void){return $n;}\n")
                write(joinpath(proj, "replibuild.toml"), """
                [project]
                name = "$name"
                root = "."

                [compile]
                source_files = ["src/lib.c"]

                [wrap]
                language = "c"
                """)
                register_only(joinpath(proj, "replibuild.toml"))
            end
            entries = PR._load_index().entries
            for (name, n) in (("twin_a", 1), ("twin_b", 2))
                cfg = CM.load_config(entries[name].toml_path)
                @test read(only(cfg.compile.source_files), String) == "int f(void){return $n;}\n"
            end
        end
    end
end

@testset "a bare discover registers nothing" begin
    # There is no build yet to vendor; `discover(...; build=true, wrap=true)`
    # registers once it has built and wrapped.
    with_temp_registry() do home
        mktempdir() do proj
            mkpath(joinpath(proj, "src"))
            write(joinpath(proj, "src", "tiny.c"), "int tiny(void){return 1;}\n")
            RepliBuild.discover(proj)
            @test isfile(joinpath(proj, "replibuild.toml"))
            @test isempty(PR._load_index().entries)
        end
    end
end

# ── vendoring: use() and the Julia package load the vendored build ───────────
@testset "use loads the vendored build, and still does when the project is gone" begin
    with_temp_registry() do home
        mktempdir() do sb
            proj = joinpath(sb, "fakewrap")
            toml = make_fake_built_project(proj)
            entry = vendor_fake!(register_only(toml))
            build_dir = PR._cached_build_dir(entry.build_hash)
            @test isfile(joinpath(build_dir, "Fakewrap.jl"))
            @test entry.verified && entry.module_name == "Fakewrap"
            @test PR._load_index().entries["fakewrap"].build_hash == entry.build_hash
            # The package written around it includes that build and nothing else.
            stub = read(joinpath(PR._packages_dir(), "Fakewrap", "src", "Fakewrap.jl"), String)
            @test occursin(repr(joinpath(build_dir, "Fakewrap.jl")), stub)

            m = PR.use("fakewrap"; verbose=false)
            @test Base.invokelatest(() -> m.ok()) == 42
            @test Base.invokelatest(() -> m.vendored_from()) == build_dir

            out = capture_stdout(PR.list_registry)
            @test occursin("vendored $(entry.build_hash[1:12])", out)

            # The project moves away: what was vendored still loads, with a warning.
            rm(proj; recursive=true)
            m2 = @test_logs (:warn, r"project is gone") PR.use("fakewrap"; verbose=false)
            @test Base.invokelatest(() -> m2.ok()) == 42
        end
    end
end

@testset "register adds the vendored build to a Julia environment" begin
    with_temp_registry() do home
        mktempdir() do sb
            proj = joinpath(sb, "fakewrap")
            envdir = joinpath(sb, "env")
            toml = make_fake_built_project(proj)
            entry = withenv("JULIA_PKG_OFFLINE" => "true") do
                vendor_fake!(PR.register(toml; build=false, env=envdir))
            end
            @test entry.julia_env == envdir
            pkg = joinpath(PR._packages_dir(), "Fakewrap")
            @test PR._env_package_path(envdir, "Fakewrap") == normpath(pkg)
            project = TOML.parsefile(joinpath(pkg, "Project.toml"))
            @test project["name"] == "Fakewrap"
            @test sort(collect(keys(project["deps"]))) == ["Libdl", "RepliBuild"]

            # `using` in a child whose only environments are that one and the stdlib.
            code = "using Fakewrap; print(\"LOADED \", Fakewrap.ok(), \" \", Fakewrap.vendored_from())"
            r = withenv("JULIA_LOAD_PATH" => join(["@", "@stdlib"], Sys.iswindows() ? ";" : ":"),
                        "JULIA_PKG_OFFLINE" => "true") do
                run_isolated(`--startup-file=no -e $code`; project=envdir, timeout=900)
            end
            @test r.ok
            @test occursin("LOADED 42 " * PR._cached_build_dir(entry.build_hash), r.output)

            # unregister takes it all back out, and leaves the project alone.
            withenv("JULIA_PKG_OFFLINE" => "true") do
                PR.unregister("fakewrap")
            end
            @test PR._env_package_path(envdir, "Fakewrap") === nothing
            @test !isdir(pkg)
            @test !isdir(PR._cached_build_dir(entry.build_hash))
            @test isfile(toml)
        end
    end
end

@testset "register adds to environments, never to a package's Project.toml" begin
    with_temp_registry() do home
        mktempdir() do sb
            pkgproj = joinpath(sb, "SomePkg")
            mkpath(pkgproj)
            write(joinpath(pkgproj, "Project.toml"), "name = \"SomePkg\"\nuuid = \"$(uuid4())\"\n")
            before = read(joinpath(pkgproj, "Project.toml"), String)
            entry = PR.register(make_fake_built_project(joinpath(sb, "fakewrap")); build=false, env=pkgproj)
            @test_logs (:warn, r"belongs to a package") match_mode=:any vendor_fake!(entry)
            @test read(joinpath(pkgproj, "Project.toml"), String) == before
        end
    end
end

@testset "which Julia environment register uses" begin
    depot_env = joinpath(first(DEPOT_PATH), "environments", "v$(VERSION.major).$(VERSION.minor)")
    @test PR._julia_env_dir(:default) == depot_env
    @test PR._julia_env_dir(:none) === nothing
    @test PR._julia_env_dir(nothing) === nothing
    @test PR._julia_env_dir(:active) == dirname(Base.active_project())
    mktempdir() do d
        @test PR._julia_env_dir(joinpath(d, "Project.toml")) == d
        @test PR._julia_env_dir(d) == d
    end
    @test_throws ErrorException PR._julia_env_dir(:elsewhere)
    withenv("REPLIBUILD_JULIA_ENV" => nothing) do
        @test PR._default_julia_env() === :default
    end
    withenv("REPLIBUILD_JULIA_ENV" => "none") do
        @test PR._default_julia_env() === :none
    end
    withenv("REPLIBUILD_JULIA_ENV" => "some/env") do
        @test PR._default_julia_env() == "some/env"
    end
end

@testset "a vendored package keeps its identity" begin
    mktempdir() do d
        with_uuid = joinpath(d, "a.toml")
        write(with_uuid, "[project]\nname = \"a\"\nuuid = \"62ced8e5-6545-4ef7-b0ee-9fbee98eda80\"\nversion = \"5.4.7\"\n")
        no_uuid = joinpath(d, "b.toml")
        write(no_uuid, "[project]\nname = \"b\"\n")
        @test PR._package_uuid(with_uuid, "a") == UUID("62ced8e5-6545-4ef7-b0ee-9fbee98eda80")
        # Derived from the name, never random: the package must not change identity.
        @test PR._package_uuid(no_uuid, "b") == PR._package_uuid(no_uuid, "b")
        @test PR._package_uuid(no_uuid, "b") != PR._package_uuid(no_uuid, "c")
        @test PR._package_version(with_uuid) == v"5.4.7"
        @test PR._package_version(no_uuid) == v"0.1.0"
    end
end

@testset "list_registry does not claim a build it cannot see" begin
    # It used to test builds/<TOML hash>, but builds are stored under
    # hash_config, so every entry read "not built".
    with_temp_registry() do home
        mktempdir() do proj
            register_only(make_test_toml(proj; name="listed"))
            out = capture_stdout(PR.list_registry)
            @test occursin("not built yet", out)
            @test occursin(proj, out)
        end
    end
end

# scaffold_package wrote `RepliBuild = "2"` into the package it generates, so
# Pkg resolved that package against RepliBuild 2.x, the newest version that
# compat admitted. The entry now comes from `_replibuild_compat`, like the
# registry's own packages: at least the RepliBuild that wrote it.
@testset "scaffold_package's Project.toml admits the RepliBuild that wrote it" begin
    mktempdir() do dir
        pkg = RepliBuild.scaffold_package("ScaffoldProbe"; path=dir, from_registry=false)
        proj = TOML.parsefile(joinpath(pkg, "Project.toml"))
        @test proj["deps"]["RepliBuild"] == string(Base.PkgId(RepliBuild).uuid)
        spec = Pkg.Versions.semver_spec(proj["compat"]["RepliBuild"])
        @test RepliBuild.VERSION in spec
        @test !(v"2.5.3" in spec)                       # the 2.x the old entry resolved
        @test !(VersionNumber(RepliBuild.VERSION.major + 1) in spec)
        # The registry's packages take their entry from the same place.
        @test proj["compat"]["RepliBuild"] == PR._replibuild_compat()
    end
end
