#!/usr/bin/env julia
# test/test_toml_preservation.jl — user-intent TOML sections survive re-discovery
#
# Regression guard for the stl_test outage (2026-07-17): discover(force=true)
# regenerated replibuild.toml from scratch, and generate_config emits the
# user-intent keys ([types].templates etc.) empty — every forced re-discovery
# silently destroyed them, which killed the entire STL wrapper section of
# stl_test for six weeks. These tests pin the preservation helpers directly
# (no toolchain needed); the devtests stl_test integration run covers the
# full pipeline.

using Test
using TOML
using RepliBuild

const DISC = RepliBuild.Discovery

@testset "TOML user-intent preservation" begin

    mktempdir() do dir
        toml = joinpath(dir, "replibuild.toml")

        # ── collect: picks up exactly the whitelisted, non-empty keys ──────
        write(toml, """
        [project]
        name = "demo"

        [types]
        templates = ["std::vector<int>"]
        template_headers = ["<vector>"]
        strictness = "warn"

        [wrap]
        shim_headers = ["demo.h"]

        [wrap.cstring_owned]
        demo_print = "demo_free"

        [wrap.varargs]
        demo_log = [["Cint"], ["Cint", "Cdouble"]]
        """)
        kept = DISC._collect_preserved_sections(toml)
        @test kept !== nothing
        @test kept[("types", "templates")] == ["std::vector<int>"]
        @test kept[("types", "template_headers")] == ["<vector>"]
        @test kept[("wrap", "shim_headers")] == ["demo.h"]
        @test kept[("wrap", "cstring_owned")] == Dict("demo_print" => "demo_free")
        @test haskey(kept, ("wrap", "varargs"))
        # Non-whitelisted keys are NOT preserved (regeneration owns them)
        @test !haskey(kept, ("types", "strictness"))
        @test !haskey(kept, ("project", "name"))

        # ── restore: merges into a freshly regenerated (empty-keys) toml ───
        write(toml, """
        [project]
        name = "demo"

        [types]
        strictness = "warn"

        [compile]
        flags = ["-O2"]
        """)
        DISC._restore_preserved_sections(toml, kept)
        doc = TOML.parsefile(toml)
        @test doc["types"]["templates"] == ["std::vector<int>"]
        @test doc["types"]["template_headers"] == ["<vector>"]
        @test doc["wrap"]["shim_headers"] == ["demo.h"]
        @test doc["wrap"]["cstring_owned"]["demo_print"] == "demo_free"
        # Regenerated content is untouched
        @test doc["types"]["strictness"] == "warn"
        @test doc["compile"]["flags"] == ["-O2"]

        # ── regenerated NON-empty value wins over the preserved one ────────
        write(toml, """
        [types]
        templates = ["std::deque<float>"]
        """)
        DISC._restore_preserved_sections(toml, kept)
        doc = TOML.parsefile(toml)
        @test doc["types"]["templates"] == ["std::deque<float>"]   # not clobbered
        @test doc["types"]["template_headers"] == ["<vector>"]     # empty slot filled

        # ── degenerate inputs are no-ops, never throw ──────────────────────
        @test DISC._restore_preserved_sections(toml, nothing) === nothing
        empty_toml = joinpath(dir, "empty.toml")
        write(empty_toml, "[project]\nname = \"x\"\n")
        @test DISC._collect_preserved_sections(empty_toml) === nothing
        @test DISC._collect_preserved_sections(joinpath(dir, "nope.toml")) === nothing
        garbage = joinpath(dir, "garbage.toml")
        write(garbage, "not [ valid toml ===")
        @test DISC._collect_preserved_sections(garbage) === nothing
    end

    # ── empty whitelisted keys are skipped (nothing to preserve) ───────────
    mktempdir() do dir
        toml = joinpath(dir, "replibuild.toml")
        write(toml, """
        [types]
        templates = []
        """)
        @test DISC._collect_preserved_sections(toml) === nothing
    end
end

println("✅ TOML preservation tests passed")
