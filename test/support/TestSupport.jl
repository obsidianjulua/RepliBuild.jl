# test/support/TestSupport.jl — the two things toolchain tests used to hand-roll:
# "skip when a prerequisite is missing" and "run this in a process that may
# crash or hang".
#
# Included by the test files themselves, not by the suites, so a file run on its
# own behaves exactly as it does inside devtests:
#
#     isdefined(@__MODULE__, :TestSupport) ||
#         include(joinpath(@__DIR__, "support", "TestSupport.jl"))
#     using .TestSupport
#
# The `isdefined` guard lets every file include it into one suite process
# without replacing the module each time.

module TestSupport

using Test
import RepliBuild

export have, requires, run_isolated, probe_verdicts

const REPO_ROOT = dirname(dirname(@__DIR__))

# ── Prerequisites ────────────────────────────────────────────────────────────

const _TOOLS = Dict(:clang => "clang", :clangxx => "clang++", :cmake => "cmake",
                    :git => "git", :nm => "nm")

"""
    have(need::Symbol) -> Bool

`:libJLCS` (the Tier-2 dialect library is built), `:llvm` (a system LLVM that
`LLVMEnvironment.get_toolchain` finds: BuildBridge starts every tool it names
through it, and the runtests CI runners have none), `:llvm_link` (the llvm-link a
C++ link runs: BuildBridge starts it by bare name on the `PATH` that
`LLVMEnvironment.with_llvm_env` sets, which can reach a bin, such as
apt.llvm.org's `/usr/lib/llvm-N/bin`, that the user's `PATH` does not), or a tool
on `PATH`:
`:clang`, `:clangxx`, `:cmake`, `:git`, `:nm`. An unknown name is an error rather
than `false`, because a misspelt prerequisite would otherwise skip its file
forever and look like a machine without the tool.
"""
function have(need::Symbol)
    need === :libJLCS && return isfile(RepliBuild.MLIRNative.libJLCS)
    need === :llvm && return _system_llvm_found()
    need === :llvm_link && return _system_llvm_found() &&
        RepliBuild.LLVMEnvironment.with_llvm_env(() -> Sys.which("llvm-link") !== nothing)
    haskey(_TOOLS, need) ||
        error("TestSupport.have: unknown prerequisite :$need " *
              "(known: :libJLCS, :llvm, :llvm_link, $(join(sort!([":$k" for k in keys(_TOOLS)]), ", ")))")
    return Sys.which(_TOOLS[need]) !== nothing
end

# `get_toolchain` raises (`error`, so an ErrorException) when it finds no system
# LLVM; anything else it raises is a bug and propagates.
function _system_llvm_found()
    try
        RepliBuild.LLVMEnvironment.get_toolchain()
        return true
    catch e
        e isa ErrorException || rethrow()
        return false
    end
end

"""
    requires(what, needs::Symbol...) -> Bool

`true` when every prerequisite is present. Otherwise records a skipped testset
named after `what` and the missing prerequisites, and returns `false`.

This replaces `exit(0)`. Inside an `include`, `exit(0)` ends the whole suite with
a success status: every file after it silently never runs and the run still
looks green. Standalone the two are indistinguishable. A file wraps its body in

    if requires("JLCS producers", :libJLCS, :clangxx)
    …
    end # requires

and runtests.jl refuses an include-time `exit` in any file a suite includes.
"""
function requires(what::AbstractString, needs::Symbol...)
    absent = [n for n in needs if !have(n)]
    isempty(absent) && return true
    names = join(string.(absent), ", ")
    @info "Skipping $what — not found: $names"
    @testset "$what (skipped — not found: $names)" begin
        @test_skip false
    end
    return false
end

# ── Isolated processes ───────────────────────────────────────────────────────

"""
    run_isolated(args::Cmd; timeout=600, project=REPO_ROOT, echo=true)
        -> (; ok, exitcode, signal, timedout, output)

Run `julia --project=<project> <args…>` in a fresh process with stdout and stderr
captured together, and SIGKILL it after `timeout` seconds. `args` is a script and
its arguments, or `-e <code>`.

For anything whose failure mode is a crash or a hang (an ABI break, a JIT
deadlock, a pipe that never drains): the failure takes the child, not the suite.

- SIGKILL, not SIGTERM: a julia wedged in native code ignores SIGTERM.
- Output goes through a file, not a pipe. A chatty child cannot block on a full
  pipe buffer while this process waits for it to exit, which is the deadlock the
  C-bucket clang call had (`test_c_bucket_pipe.jl`).
- On failure the captured output is echoed to stderr, since otherwise it is lost.
  Pass `echo=false` when a non-zero exit is an expected outcome being classified.
"""
function run_isolated(args::Cmd; timeout::Real = 600, project::AbstractString = REPO_ROOT,
                      echo::Bool = true)
    log = tempname()
    cmd = `$(Base.julia_cmd()) --project=$project $args`
    proc = open(log, "w") do io
        run(pipeline(ignorestatus(cmd); stdout = io, stderr = io); wait = false)
    end
    timedout = timedwait(() -> process_exited(proc), float(timeout); pollint = 0.2) === :timed_out
    timedout && kill(proc, Base.SIGKILL)
    wait(proc)
    output = read(log, String)
    rm(log; force = true)
    ok = !timedout && proc.exitcode == 0 && proc.termsignal == 0
    if !ok && echo
        println(stderr, "── isolated run failed (exit=$(proc.exitcode), signal=$(proc.termsignal), ",
                "timed out=$timedout): $cmd\n", output)
    end
    return (; ok, exitcode = proc.exitcode, signal = proc.termsignal, timedout, output)
end

"""
    probe_verdicts(output) -> Dict{String,String}

Parse the probe protocol: one `PROBE <name>: <verdict>` line per case, then
`PROBE_DONE` when the script ran to its end. Returns `name => verdict`
(`"PASS …"` or `"FAIL <detail>"`), plus `"PROBE_DONE" => ""`.

Assert with `startswith(get(v, name, "no PROBE line"), "PASS")`, so a failure
prints the probe's own detail instead of a regex miss over the whole output.
"""
function probe_verdicts(output::AbstractString)
    v = Dict{String,String}()
    for line in eachline(IOBuffer(output))
        if line == "PROBE_DONE"
            v["PROBE_DONE"] = ""
            continue
        end
        m = match(r"^PROBE (\S+): (.*)$", line)
        m === nothing || (v[m.captures[1]] = m.captures[2])
    end
    return v
end

end # module TestSupport
