#!/usr/bin/env julia
# test/test_llvm_tool_table.jl — the LLVM tool dict is the names `get_tool`
# looks up, not a catalogue of the install.
#
# `discover_llvm_tools` used to `isfile` 51 binaries at toolchain init,
# including clang-format, clang-tidy, mlir-*, FileCheck, bugpoint, and a
# hardcoded `clang-20` that is already stale on LLVM 22. Grepping every
# `get_tool("...")` and `execute("...")` literal in src/ shows four LLVM
# names actually asked for (clang, clang++, llvm-as, llvm-config). The rest
# are backticks or `resolve_tool` / PATH. cmake/nm/pkg-config are not LLVM
# tools.
#
# Pruning is safe because `BuildBridge.execute` already falls back to the
# bare command when the table misses (`isempty(tool_path) ? command :
# tool_path`) for any name starting with `clang` or `llvm-`. That fallback
# is the whole reason a pruned tool still runs. Two modules also shared
# the name `discover_llvm_tools` for a producer and its consumer; the
# consumer is `resolve_required_tools`.
#
# The list and the fallback are textual because a new probe that is not
# on this machine would pass a keys-of-the-live-dict check, and a Linux
# CI box without llvm-dis would skip a live execute. The live probe runs
# when a pruned `llvm-`/`clang*` binary is on PATH, so the fallback is
# also executed, not just read.

using Test
using RepliBuild

const _SRC = joinpath(@__DIR__, "..", "src")
const _ENV_SRC = joinpath(_SRC, "Builder", "LLVMEnvironment.jl")
const _BB_SRC  = joinpath(_SRC, "Builder", "BuildBridge.jl")

const _TABLE_TOOLS = ["clang", "clang++", "llvm-as", "llvm-config"]

"""Quoted names inside the `essential_tools = [...]` array, in source order."""
function _probed_tool_names(src::AbstractString)
    m = match(r"essential_tools\s*=\s*\[(.*?)\]"s, src)
    m === nothing && return nothing
    return [mm.captures[1] for mm in eachmatch(r"\"([^\"]+)\"", m.captures[1])]
end

@testset "LLVM tool table probes only what get_tool looks up" begin
    @test isfile(_ENV_SRC)
    @test isfile(_BB_SRC)
    env_src = read(_ENV_SRC, String)
    bb_src  = read(_BB_SRC, String)

    probed = _probed_tool_names(env_src)
    @test probed !== nothing
    @test probed == _TABLE_TOOLS

    @testset "BuildBridge.execute falls back to PATH when the table misses" begin
        # The fallback is the contract the prune relies on. Pin the source
        # so a refactor that drops it fails here rather than at the first
        # execute of a non-table clang/llvm- name.
        @test occursin("isempty(tool_path) ? command : tool_path", bb_src)
        @test occursin("startswith(command, \"clang\") || startswith(command, \"llvm-\")", bb_src)

        probe = nothing
        for name in ("llvm-dis", "llvm-link", "clang-format")
            if Sys.which(name) !== nothing
                probe = name
                break
            end
        end
        if probe === nothing
            @info "no pruned LLVM tool on PATH — skipping execute table-miss fallback probe"
        else
            @test isempty(RepliBuild.LLVMEnvironment.get_tool(probe))
            out, ec = RepliBuild.BuildBridge.execute(probe, ["--version"])
            @test ec == 0
            @test !isempty(out)
        end
    end

    @testset "the two discovery functions no longer share a name" begin
        @test isdefined(RepliBuild.LLVMEnvironment, :discover_llvm_tools)
        @test isdefined(RepliBuild.BuildBridge, :resolve_required_tools)
        @test !isdefined(RepliBuild.BuildBridge, :discover_llvm_tools)
        @test occursin("export", bb_src)
        @test occursin("resolve_required_tools", bb_src)
        @test !occursin(r"function\s+discover_llvm_tools", bb_src)
    end
end
