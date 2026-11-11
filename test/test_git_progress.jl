# test/test_git_progress.jl — a git dependency's clone, fetch and submodule
# fetch report progress at a terminal (2026-11-10).
#
# `_clone_dep` ran `git clone --quiet`, so a large dependency sat at
# "cloning <url>..." with no sign of life. When stderr is a terminal the network
# steps now run with `--progress`, and `_git_progress` turns git's stream into a
# ProgressMeter bar per local phase. Elsewhere they run `--quiet` as before.
#
# The transcript below is git's own output for a file:// clone, captured through
# a pipe, with a `warning:` and a failure appended. The live testsets clone over
# file:// with progress forced on (needs `git` only; no network).

using Test
using RepliBuild

const _GP = RepliBuild.DependencyResolver

_gp_file_url(p) = "file://" * (Sys.iswindows() ? "/" * replace(abspath(p), '\\' => '/') : abspath(p))
# What a terminal would show once each `\r` returns to the start of the line.
_gp_lines(s) = filter(!isempty, strip.(split(replace(s, "\e[K" => ""), r"[\r\n]")))

const _GP_TRANSCRIPT = string(
    "Cloning into 'c1'...\n",
    "remote: Enumerating objects: 36, done.        \n",
    "remote: Counting objects:  50% (18/36)        \rremote: Counting objects: 100% (36/36), done.        \n",
    "remote: Compressing objects: 100% (35/35), done.        \n",
    "Receiving objects:  50% (18/36)\rremote: Total 36 (delta 1), reused 0 (delta 0), pack-reused 0 (from 0)        \n",
    "Receiving objects: 100% (36/36)\rReceiving objects: 100% (36/36), 587.75 KiB | 58.78 MiB/s, done.\n",
    "Resolving deltas:   0% (0/1)\rResolving deltas: 100% (1/1)\rResolving deltas: 100% (1/1), done.\n",
    "Submodule 'libs/sub' (file:///tmp/sub) registered for path 'libs/sub'\n",
    "warning: kept for the user\n",
    "Receiving objects:  40% (2/5)\r",
    "fatal: early EOF\n")

@testset "git dependency progress" begin
    @testset "a transcript: a bar per local phase, chatter dropped, messages kept" begin
        out = IOBuffer()
        _GP._git_progress(IOBuffer(_GP_TRANSCRIPT); out)
        lines = _gp_lines(String(take!(out)))
        # Every line is a bar or one of git's messages for the user.
        bars = filter(l -> occursin('%', l) && occursin('|', l), lines)
        @test filter(l -> !(l in bars), lines) == ["warning: kept for the user", "fatal: early EOF"]
        # Each local phase is drawn, and finished at 100% exactly once.
        @test count(l -> startswith(l, "receiving objects 100%"), lines) == 1
        @test count(l -> startswith(l, "resolving deltas 100%"), lines) == 1
        @test any(l -> startswith(l, "receiving objects  50%"), lines)
        # The server's phases and the narration --quiet suppressed are gone.
        @test !any(l -> occursin(r"remote:|counting|compressing|Cloning into|registered for path", l), lines)
        # A phase git abandoned stays where it stopped, and the failure follows it.
        @test lines[end] == "fatal: early EOF"
        @test startswith(lines[end-1], "receiving objects  40%")
    end

    if Sys.which("git") === nothing
        @warn "git not found — skipping the live git progress tests"
    else
        mktempdir() do sb
            up = joinpath(sb, "up")
            mkpath(up)
            git(args...) = run(`git -C $up -c user.email=t@example.com -c user.name=tester -c commit.gpgsign=false -c core.autocrlf=false $(collect(args))`)
            git("init", "-q")
            for i in 1:20
                write(joinpath(up, "f$i.bin"), rand(UInt8, 20_000))
            end
            git("add", "-A"); git("commit", "-q", "-m", "init")

            @testset "a clone with progress forced on" begin
                dst = joinpath(sb, "clone")
                out = IOBuffer()
                _GP._run_git(`git clone --quiet -- $(_gp_file_url(up)) $dst`,
                             `git clone --progress -- $(_gp_file_url(up)) $dst`; show = true, out)
                @test isfile(joinpath(dst, "f20.bin"))
                lines = _gp_lines(String(take!(out)))
                @test count(l -> startswith(l, "receiving objects 100%"), lines) == 1
                @test !any(l -> occursin(r"remote:|Cloning into", l), lines)
            end

            @testset "a failed clone still raises, and git's reason reaches the user" begin
                out = IOBuffer()
                bad = _gp_file_url(joinpath(sb, "nope"))
                @test_throws ProcessFailedException _GP._run_git(`git clone --quiet -- $bad $(joinpath(sb, "c2"))`,
                    `git clone --progress -- $bad $(joinpath(sb, "c2"))`; show = true, out)
                @test occursin("fatal:", String(take!(out)))
            end

            @testset "off a terminal the quiet command runs" begin
                # A test's stderr is not a terminal, so the default is the quiet path.
                @test _GP._git_progress_enabled() == (stderr isa Base.TTY)
                dst = joinpath(sb, "quiet")
                out = IOBuffer()
                _GP._run_git(`git clone --quiet -- $(_gp_file_url(up)) $dst`,
                             `git clone --progress -- $(_gp_file_url(up)) $dst`; show = false, out)
                @test isfile(joinpath(dst, "f20.bin"))
                @test isempty(take!(out))
            end
        end
    end
end
