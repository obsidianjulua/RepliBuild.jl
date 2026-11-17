# Which LLVM the doctor reports on (2026-11-17)
#
# `check_environment` compares libJLCS's build stamp with "today's llvm-config".
# `src/mlir/build.sh` builds libJLCS against the bare `llvm-config` on PATH, and
# the doctor looked for `llvm-config-<minimum>` BEFORE the bare name. A host
# with the distribution's LLVM at exactly the minimum beside a newer one on PATH
# therefore had a libJLCS built against the newer one judged against the older:
# GitHub's Ubuntu 26.04 image ships `/usr/bin/llvm-config-21`, the workflow puts
# apt.llvm.org's 23 on PATH, and every x86-64 cell of tier2-linux failed
# "Tier 2 is live" with "built against LLVM 23.1.6, but llvm-config is 21.1.8".
#
# The probe order is now the bare name, then the versioned names newest first,
# and the first one that meets the minimum is taken. Toolchain-free: the pick is
# driven with a fabricated PATH, as a table and (off Windows) as scripts.

using Test
using RepliBuild

const _DP = RepliBuild.EnvironmentDoctor
const _DP_MIN = RepliBuild.LLVMEnvironment.MIN_LLVM_VERSION
const _DP_MAX = RepliBuild.LLVMEnvironment.MAX_PROBED_LLVM_VERSION

# A PATH as a table: tool name => version it answers.
_dp_pick(names, path::Dict{String,String}) =
    _DP._pick_llvm_tool(names; find = n -> haskey(path, n) ? (true, "/fake/" * n) : (false, ""),
                        version = p -> path[basename(p)])

@testset "The doctor reports on the LLVM libJLCS is built against" begin

    @testset "the probe order: the bare name, then versioned names newest first" begin
        for base in ("llvm-config", "clang++", "mlir-tblgen")
            names = _DP._versioned(base)
            @test first(names) == base
            @test names[2:end] == [base * "-$v" for v in _DP_MAX:-1:_DP_MIN]
            @test last(names) == base * "-$(_DP_MIN)"
            @test allunique(names)
        end
    end

    @testset "the pick" begin
        names = _DP._versioned("llvm-config")
        newer = "$(_DP_MIN + 2).1.6"
        floor = "$(_DP_MIN).1.8"
        old = "$(_DP_MIN - 2).1.0"

        # Ubuntu 26.04 as GitHub ships it: the distribution's LLVM at the
        # minimum under its versioned name, a newer one first on PATH.
        ubuntu = Dict("llvm-config" => newer, "llvm-config-$(_DP_MIN)" => floor)
        @test _dp_pick(names, ubuntu) == ("/fake/llvm-config", newer, true)
        # The order this replaces answers with the other install.
        @test _dp_pick(["llvm-config-$(_DP_MIN)", "llvm-config"], ubuntu) ==
              ("/fake/llvm-config-$(_DP_MIN)", floor, true)

        # apt.llvm.org with its bin directory off PATH: only a versioned name,
        # and not the minimum's. It used to be "not found".
        @test _dp_pick(names, Dict("llvm-config-$(_DP_MIN + 2)" => newer)) ==
              ("/fake/llvm-config-$(_DP_MIN + 2)", newer, true)
        # Two versioned installs: the newest.
        @test _dp_pick(names, Dict("llvm-config-$(_DP_MIN)" => floor, "llvm-config-$(_DP_MIN + 2)" => newer)) ==
              ("/fake/llvm-config-$(_DP_MIN + 2)", newer, true)
        # A default that is too old beside a supported versioned one: the
        # supported one, not "too old".
        @test _dp_pick(names, Dict("llvm-config" => old, "llvm-config-$(_DP_MIN + 1)" => "$(_DP_MIN + 1).1.0")) ==
              ("/fake/llvm-config-$(_DP_MIN + 1)", "$(_DP_MIN + 1).1.0", true)
        # Only too old: found, and said to be too old.
        @test _dp_pick(names, Dict("llvm-config" => old)) == ("/fake/llvm-config", old, false)
        # A version that does not parse is not a supported install.
        @test _dp_pick(names, Dict("llvm-config" => "unknown")) == ("/fake/llvm-config", "unknown", false)
        # Nothing on PATH.
        @test _dp_pick(names, Dict{String,String}()) === nothing
    end

    # The checks themselves, over scripts on a fabricated PATH. Windows cannot
    # run a shebang script, and has no versioned names to confuse.
    if !Sys.iswindows()
        @testset "the checks read the PATH's own llvm-config and clang++" begin
            mktempdir() do dir
                function tool(name, version)
                    p = joinpath(dir, name)
                    write(p, "#!/bin/sh\necho 'version $version'\n")
                    chmod(p, 0o755)
                end
                tool("llvm-config", "$(_DP_MIN + 2).1.6")
                tool("llvm-config-$(_DP_MIN)", "$(_DP_MIN).1.8")
                tool("clang++", "$(_DP_MIN + 2).1.6")
                tool("clang++-$(_DP_MIN)", "$(_DP_MIN).1.8")
                (cfg, clang) = withenv("PATH" => dir) do
                    (_DP._check_llvm_config(), _DP._check_clang())
                end
                @test cfg.found && cfg.meets_requirement
                @test cfg.version == "$(_DP_MIN + 2).1.6"
                @test cfg.path == joinpath(dir, "llvm-config")
                @test clang.found && clang.meets_requirement
                @test clang.version == "$(_DP_MIN + 2).1.6"
                @test clang.path == joinpath(dir, "clang++")
                # Nothing there: not found, with the advice.
                (none, _) = withenv("PATH" => mktempdir()) do
                    (_DP._check_llvm_config(), nothing)
                end
                @test !none.found && !isempty(none.advice)
            end
        end
    end
end
