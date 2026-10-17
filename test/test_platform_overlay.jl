# test_platform_overlay.jl — `[platform.<host>]` tables override the base config.
#
# One replibuild.toml has to build on Linux and Windows, and the two disagree on
# link libraries (no `dl`/`rt` under mingw), platform defines (`LUA_USE_LINUX`,
# `_GLFW_X11`) and which platform sources a dependency excludes. Without an
# overlay every Hub config could be right for one host only.
#
# The merge is driven with both host names explicitly, so neither arm depends on
# which machine runs the suite; one testset then goes through load_config on the
# real host. No toolchain.

using Test
using RepliBuild
import TOML

const _PO_CM = RepliBuild.ConfigurationManager

const _PO_DOC = Dict{String,Any}(
    "project" => Dict{String,Any}("name" => "overlayprobe"),
    "compile" => Dict{String,Any}("flags" => ["-O2", "-DLUA_USE_LINUX"]),
    "link"    => Dict{String,Any}("link_libraries" => ["m", "dl"], "optimization_level" => "2"),
    "dependencies" => Dict{String,Any}(
        "up" => Dict{String,Any}("type" => "git", "url" => "https://example.invalid/up.git",
                                 "exclude" => ["win32_"])),
    "platform" => Dict{String,Any}(
        "windows" => Dict{String,Any}(
            "compile" => Dict{String,Any}("flags" => ["-O2"]),
            "link"    => Dict{String,Any}("link_libraries" => String[]),
            "dependencies" => Dict{String,Any}(
                "up" => Dict{String,Any}("exclude" => ["x11_", "posix_"])))))

@testset "platform overlay" begin
    @testset "merge semantics, both hosts" begin
        win = _PO_CM.apply_platform_overlay(_PO_DOC, "probe.toml"; platform="windows")
        @test !haskey(win, "platform")
        @test win["compile"]["flags"] == ["-O2"]                  # arrays replace
        @test win["link"]["link_libraries"] == String[]           # ... including to empty
        @test win["link"]["optimization_level"] == "2"            # untouched keys survive
        @test win["dependencies"]["up"]["exclude"] == ["x11_", "posix_"]
        @test win["dependencies"]["up"]["url"] == "https://example.invalid/up.git"  # tables merge

        lin = _PO_CM.apply_platform_overlay(_PO_DOC, "probe.toml"; platform="linux")
        @test !haskey(lin, "platform")
        @test lin["compile"]["flags"] == ["-O2", "-DLUA_USE_LINUX"]   # no linux overlay: base
        @test lin["link"]["link_libraries"] == ["m", "dl"]
        @test lin["dependencies"]["up"]["exclude"] == ["win32_"]

        # The input is not mutated: the windows merge must not leak into the base.
        @test _PO_DOC["link"]["link_libraries"] == ["m", "dl"]
        @test _PO_DOC["dependencies"]["up"]["exclude"] == ["win32_"]

        # An unrecognised host sees the base.
        @test _PO_CM.apply_platform_overlay(_PO_DOC, "probe.toml"; platform=nothing)["link"]["link_libraries"] == ["m", "dl"]
    end

    @testset "no platform table is the identity" begin
        plain = Dict{String,Any}("project" => Dict{String,Any}("name" => "p"))
        @test _PO_CM.apply_platform_overlay(plain, "p.toml"; platform="windows") === plain
    end

    @testset "malformed overlays are hard errors" begin
        typo = Dict{String,Any}("platform" => Dict{String,Any}("window" => Dict{String,Any}()))
        @test_throws ErrorException _PO_CM.apply_platform_overlay(typo, "t.toml"; platform="windows")
        flat = Dict{String,Any}("platform" => Dict{String,Any}("windows" => "nope"))
        @test_throws ErrorException _PO_CM.apply_platform_overlay(flat, "t.toml"; platform="windows")
        notable = Dict{String,Any}("platform" => ["windows"])
        @test_throws ErrorException _PO_CM.apply_platform_overlay(notable, "t.toml"; platform="windows")
    end

    @testset "load_config applies the running host's overlay" begin
        mktempdir() do dir
            toml = joinpath(dir, "replibuild.toml")
            doc = deepcopy(_PO_DOC)
            doc["project"]["root"] = dir
            delete!(doc, "dependencies")      # keep load_config off the network
            open(io -> TOML.print(io, doc), toml, "w")
            cfg = _PO_CM.load_config(toml)
            host = _PO_CM.host_platform_key()
            if host == "windows"
                @test cfg.link.link_libraries == String[]
                @test !("-DLUA_USE_LINUX" in cfg.compile.flags)
            else
                @test cfg.link.link_libraries == ["m", "dl"]
                @test "-DLUA_USE_LINUX" in cfg.compile.flags
            end
            @info "platform overlay exercised on host" host
        end
    end

    @testset "forced re-discovery keeps the overlays" begin
        for name in _PO_CM.PLATFORM_KEYS
            @test ("platform", name) in RepliBuild.Discovery.PRESERVED_TOML_KEYS
        end
    end
end
