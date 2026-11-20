# The doctor and a Rust project (2026-11-19)
#
# `check_environment` reported on the C and C++ toolchains only, and `build`
# refused a `[rust]` project on a machine where those were incomplete, though
# a crate is compiled and linked by cargo and rustc and starts neither. The
# doctor now probes the Rust toolchain the build will start, and `build` asks
# a Rust project about that and about the readers every wrapper needs, and
# nothing else. Toolchain-free: the probes are driven with stand-ins.

using Test
using RepliBuild

const _DR = RepliBuild.EnvironmentDoctor
const _DR_SPAWN_FAILED = RepliBuild.BuildBridge.SPAWN_FAILED

_dr_ok(name; section = "") = _DR.ToolStatus(name, false, true, "/fake/" * name, "1.0.0", true, name; section)
_dr_bad(name; advice = String[]) = _DR.ToolStatus(name, false, false, "", "", false, "not found"; advice)
_dr_status(tools) = _DR.ToolchainStatus(tools, false, false, false)

@testset "The doctor and a Rust project" begin

    @testset "cargo and rustc are asked what they are, as the build asks" begin
        answers = Dict("cargo" => ("cargo 1.97.1 (c980f4866 2026-06-30)\n", "", 0),
                       "rustc" => ("rustc 1.97.1 (abc 2026-06-30)\nbinary: rustc\nhost: x86_64-unknown-linux-gnu\n" *
                                   "release: 1.97.1\nLLVM version: 22.1.6\n", "", 0))
        asked = Cmd[]
        capture(cmd) = (push!(asked, cmd); answers[basename(first(cmd.exec))])
        cargo = _DR._check_rust_tool("cargo"; tool = n -> "/opt/rust/bin/" * n, capture)
        rustc = _DR._check_rust_tool("rustc"; tool = n -> "/opt/rust/bin/" * n, capture)
        @test [c.exec for c in asked] == [["/opt/rust/bin/cargo", "-V"], ["/opt/rust/bin/rustc", "-vV"]]
        @test (cargo.found, cargo.meets_requirement, cargo.version, cargo.path) == (true, true, "1.97.1", "/opt/rust/bin/cargo")
        @test (rustc.found, rustc.version) == (true, "1.97.1")
        # rustc carries an LLVM of its own, which is not Julia's or the system's: the line says which.
        @test occursin("host x86_64-unknown-linux-gnu", _DR._plain(rustc.message)) && occursin("its own LLVM 22.1.6", _DR._plain(rustc.message))
        # Optional: a machine without Rust warns and does not fail.
        @test !cargo.required && !rustc.required && cargo.section == _DR.SEC_RUST

        # Not installed: the program never started.
        absent = _DR._check_rust_tool("cargo"; tool = identity, capture = _ -> ("", "no such file", _DR_SPAWN_FAILED))
        @test (absent.found, absent.message, _DR.verdict(absent)) == (false, "not found", :warn)
        @test any(a -> occursin("rustup", a), absent.advice) && any(a -> occursin("REPLIBUILD_TOOL_CARGO", a), absent.advice)
        @test any(a -> occursin("Only a [rust] project needs this", a), absent.advice)
        # Installed and broken: what it said, not "not found".
        broken = _DR._check_rust_tool("rustc"; tool = n -> "/x/" * n, capture = _ -> ("", "error: no default toolchain\nmore", 1))
        @test !broken.found && occursin("exited with 1: error: no default toolchain", broken.message)
    end

    @testset "the demangler is asked to read one of rustc's names" begin
        good = _DR._check_rust_demangler(; demangle = s -> "lean[1a2b3c]::add\n")
        @test good.found && good.meets_requirement && good.name == "demangler"
        # An old binutils answers with the name it was given.
        old = _DR._check_rust_demangler(; demangle = s -> s * "\n")
        @test !old.meets_requirement && occursin(_DR._RUST_V0_SAMPLE, old.message)
        @test any(a -> occursin("2.36", a), old.advice)
        gone = _DR._check_rust_demangler(; demangle = s -> error("no c++filt"))
        @test !gone.found && occursin("could not be run", gone.message)
        # The sample is a v0 name of this crate family, the kind every Rust build here has.
        @test RepliBuild.RustABI.v0_root(_DR._RUST_V0_SAMPLE).crate == "lean"
    end

    @testset "what a Rust project needs, and nothing of C or C++" begin
        needs = [_dr_ok(n) for n in _DR.RUST_PROBES]
        @test _DR.rust_ready(_dr_status(needs))
        # No C compiler, no sysroot, no LLVM, no dialect library: still ready.
        without_c = vcat(needs, [_dr_bad("clang (JLL)"), _dr_bad("sysroot"), _dr_bad("llvm-config"), _dr_bad("clang++"), _dr_bad("libJLCS")])
        @test _DR.rust_ready(_dr_status(without_c))
        # Each of its own probes is needed.
        for missing_one in _DR.RUST_PROBES
            tools = [n == missing_one ? _dr_bad(n; advice = ["fix " * n]) : _dr_ok(n) for n in _DR.RUST_PROBES]
            @test !_DR.rust_ready(_dr_status(tools))
            msg = _DR.rust_problem(_dr_status(tools))
            @test occursin("A [rust] project cannot be built here", msg) && occursin("1 probe(s) failed", msg)
            @test occursin("✗ $missing_one: not found", msg) && occursin("fix " * missing_one, msg)
            @test !any(n -> n != missing_one && occursin("✗ $n", msg), _DR.RUST_PROBES)
        end
        # A status read back from the registry's cache has no probes: not ready, and it says what was not asked.
        empty = _dr_status(_DR.ToolStatus[])
        @test !_DR.rust_ready(empty)
        @test occursin("cargo: not probed", _DR.rust_problem(empty))
    end

    @testset "the summary says what Rust needs" begin
        lines(tools) = join(_DR._plain.(_DR._summary_lines(_dr_status(tools))), "\n")
        ready = vcat([_dr_ok(n) for n in _DR.RUST_PROBES], [_dr_ok("libJLCS")])
        @test occursin("Rust crates — cargo and rustc found", lines(ready))
        @test !occursin("dialect library", lines(ready))
        no_dialect = [_dr_ok(n) for n in _DR.RUST_PROBES]
        @test occursin("need the dialect library", lines(no_dialect))
        no_rust = [n in ("cargo", "rustc") ? _dr_bad(n) : _dr_ok(n) for n in _DR.RUST_PROBES]
        @test occursin("Rust crates unavailable — a [rust] project needs cargo and rustc; nothing else does", lines(no_rust))
        no_reader = [n == "DWARF dumper" ? _dr_bad(n) : _dr_ok(n) for n in _DR.RUST_PROBES]
        @test occursin("cargo and rustc are here, and a probe a wrapper needs failed", lines(no_reader))
        # A status built without the Rust probes says nothing about Rust.
        @test !occursin("Rust", lines([_dr_ok("libJLCS")]))
    end

    @testset "the report has the Rust probes" begin
        status = _DR.check_environment(verbose = false)
        names = [t.name for t in status]
        @test "cargo" in names && "rustc" in names && "demangler" in names
        @test all(t -> t.section == _DR.SEC_RUST && !t.required, [t for t in status if t.name in ("cargo", "rustc", "demangler")])
        # Missing Rust never makes the C and C++ verdicts worse.
        @test status.ready == status.tier1_ready
    end
end
