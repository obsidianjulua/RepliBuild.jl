#!/usr/bin/env julia
# =============================================================================
# hub_sweep — rebuild, wrap and test the Hub's packages, one table at the end
# =============================================================================
#
# A HAND-RUN DEVELOPER TOOL, NOT A TEST. It rebuilds Hub packages in place,
# which no test may do, and is wired into no suite. CLAUDE.md ("RepliBuild-Hub")
# says when a sweep is due: before a release, for the package a change
# exercises, or when John asks. It takes hours on the whole Hub.
#
#     julia --project=. test/tools/hub_sweep.jl [<hub>] [options]
#
# `<hub>` is the Hub checkout (default: `RepliBuild-Hub` beside this repo; pass
# it from a worktree). Every directory of `<hub>/packages` with a
# `replibuild.toml` is a package, indexed or not. Per package, in order:
#
#   save      `julia/` is copied to `<out>/before/<pkg>/julia` before anything
#             touches it. A rebuild cleans first, so a build that then fails
#             leaves a package with no wrapper; the copy is what puts it back.
#   rebuild   `clean(toml; deps = false)`, `build`, `wrap`: from the sources
#             already there. Never `discover`: Hub recipes are hand-written.
#             A rebuild that fails gets its saved `julia/` back, and the
#             failed state is kept under `<out>/failed/<pkg>/`.
#   verify    every `test_deep*.jl`, then `test.jl`.
#   check     the wrapper says it was written by this RepliBuild
#             (`BUILD_GENERATOR`); a recipe with `aot_thunks = true`, or a
#             package that had a compiled thunks library before, has one now
#             (a wrap that cannot compile its thunks still succeeds as a JIT
#             wrapper, so nothing else would say); and no source was fetched
#             again.
#
# NOTHING HERE PULLS SOURCES THAT HAVE NOT CHANGED (John, 2026-11-23). The
# rebuild keeps the dependency checkouts, and RepliBuild fetches only when a
# recipe's pin is new. Two things hold that:
#   - A test script that calls `clean` is not run, and its package fails. A
#     test script builds what is missing, wraps, and asserts; this tool is
#     what rebuilds. (With RepliBuild 4.6.0 and older a bare `clean` also
#     deleted the checkout, and the next build cloned it again.)
#   - Each checkout is looked at before and after the package's steps. One
#     that was cloned or fetched again under the same pin is a failure.
#
# Each step is a child Julia with a time limit and its own log. One line per
# step goes to `<out>/status.tsv`, the table to `<out>/summary.md`, and the
# exit status is 1 when any package failed.
#
# Options:
#   --out DIR          where logs, saved wrappers and tables go
#                      (default `.claude/hub-sweep-<date>` in this checkout)
#   --only a,b         these packages only     --skip a,b   leave these out
#   --verify           rebuild nothing: run the verify steps on what is there
#   --rewrap           `wrap` only (after an engine change), then verify
#   --jobs N           packages at a time (default 2; a build is parallel itself)
#   --timeout S        seconds per step (default 3600); `--timeout pkg=S` for one
#   --arg pkg=VALUE    an argument for that package's `test_deep*.jl`
#                      (llamacpp takes the path of its model file)
#   --resume           skip packages `<out>/status.tsv` already has a verdict for
#   --restore          put the saved `julia/` folders back, and stop
#   --plan             print what would run, and stop
#
# oneDNN is left out unless `--only` names it (John, 2026-11-21: its build
# needs most of a 64 GB machine).
#
# The children get this process's environment, plus `REPLIBUILD_JULIA_ENV=none`
# so that a package's `register` adds nothing to a Julia environment. Which
# `REPLIBUILD_*` variables were set is written to `<out>/environment.txt`.
# =============================================================================

using RepliBuild
using Dates
using SHA
using TOML

const SWEEP_DEFAULT_SKIP = ["onednn"]
const SWEEP_REPO = normpath(joinpath(@__DIR__, "..", ".."))

# What the tool runs to rebuild (or re-wrap) one package: a child of its own,
# so a crash or a hang ends that package and not the sweep.
const SWEEP_CHILD = raw"""
using RepliBuild
toml = abspath(ARGS[1]); mode = ARGS[2]
say(k, v) = (println("SWEEP ", k, "=", v); flush(stdout))
say("replibuild", string(RepliBuild.VERSION, " at ", pkgdir(RepliBuild)))
say("julia", VERSION)
t0 = time()
if mode == "rebuild"
    try
        RepliBuild.clean(toml; deps = false)
        say("clean", "ok")
    catch e
        say("clean", "FAILED " * first(split(sprint(showerror, e), '\n')))
        # Without a clean, at least force the metadata to be extracted again.
        ph = joinpath(dirname(toml), ".replibuild_cache", "project_hash")
        isfile(ph) && rm(ph)
    end
    lib = RepliBuild.build(toml)
    say("build_s", round(time() - t0; digits = 1))
    say("lib", lib)
end
t1 = time()
w = RepliBuild.wrap(toml)
say("wrap_s", round(time() - t1; digits = 1))
say("wrapper", w)
say("done", "ok")
"""

Base.@kwdef mutable struct SweepOptions
    hub::String = joinpath(dirname(SWEEP_REPO), "RepliBuild-Hub")
    out::String = joinpath(SWEEP_REPO, ".claude", "hub-sweep-" * string(Dates.today()))
    only::Vector{String} = String[]
    skip::Vector{String} = String[]
    mode::Symbol = :rebuild            # :rebuild, :rewrap or :verify
    jobs::Int = 2
    timeout::Int = 3600
    timeouts::Dict{String,Int} = Dict{String,Int}()
    args::Dict{String,Vector{String}} = Dict{String,Vector{String}}()
    resume::Bool = false
    restore::Bool = false
    plan::Bool = false
    help::Bool = false
end

sweep_usage() = join(Iterators.takewhile(l -> startswith(l, "#"),
                                         Iterators.drop(eachline(@__FILE__), 1)), "\n")

function sweep_parse_args(args::Vector{String})::SweepOptions
    o = SweepOptions()
    names(v) = String[strip(x) for x in split(v, ',') if !isempty(strip(x))]
    i = 1
    value() = (i += 1; i <= length(args) || error("$(args[i-1]) needs a value"); args[i])
    seen_hub = false
    while i <= length(args)
        a = args[i]
        if a == "--out";               o.out = abspath(value())
        elseif a == "--only";          append!(o.only, names(value()))
        elseif a == "--skip";          append!(o.skip, names(value()))
        elseif a == "--verify";        o.mode = :verify
        elseif a == "--rewrap";        o.mode = :rewrap
        elseif a == "--jobs";          o.jobs = parse(Int, value())
        elseif a == "--resume";        o.resume = true
        elseif a == "--restore";       o.restore = true
        elseif a == "--plan";          o.plan = true
        elseif a == "--timeout"
            v = value()
            if occursin('=', v)
                (p, s) = split(v, '='; limit = 2); o.timeouts[String(p)] = parse(Int, s)
            else
                o.timeout = parse(Int, v)
            end
        elseif a == "--arg"
            v = value()
            occursin('=', v) || error("--arg takes pkg=VALUE, got `$v`")
            (p, s) = split(v, '='; limit = 2)
            push!(get!(o.args, String(p), String[]), String(s))
        elseif a in ("-h", "--help");  o.help = true
        elseif startswith(a, "-")
            error("unknown option `$a` (--help lists them)")
        else
            seen_hub && error("two Hub paths given: `$(o.hub)` and `$a`")
            o.hub = abspath(a); seen_hub = true
        end
        i += 1
    end
    o.jobs >= 1 || error("--jobs must be at least 1")
    return o
end

"The directory that holds the packages: `<hub>/packages`, or `<hub>` itself."
sweep_packages_dir(hub::AbstractString) =
    isdir(joinpath(hub, "packages")) ? joinpath(hub, "packages") : String(hub)

"Every package of the Hub: a directory with a `replibuild.toml`."
function sweep_packages(hub::AbstractString)::Vector{String}
    d = sweep_packages_dir(hub)
    isdir(d) || error("no Hub at `$hub`")
    return sort!(String[p for p in readdir(d) if isfile(joinpath(d, p, "replibuild.toml"))])
end

"""
The packages this run covers. A name that is not a package is an error: a typo
must not turn into a sweep of nothing.
"""
function sweep_select(o::SweepOptions, all::Vector{String})::Vector{String}
    for n in vcat(o.only, o.skip)
        n in all || error("`$n` is not a package of $(sweep_packages_dir(o.hub))")
    end
    isempty(o.only) || return String[p for p in all if p in o.only && !(p in o.skip)]
    return String[p for p in all if !(p in o.skip) && !(p in SWEEP_DEFAULT_SKIP)]
end

"""
Does this script call `clean` (bare, or qualified as in `RepliBuild.clean`,
with any arguments)? Read off the parsed code, so a comment or a string that
mentions it does not count. A script that does is never run by a sweep: a test
script does not rebuild, and with RepliBuild 4.6.0 and older a bare `clean`
deleted the dependency checkout as well.
"""
function sweep_cleans(path::AbstractString)::Bool
    isfile(path) || return false
    ast = try
        Meta.parseall(read(path, String))
    catch
        return false
    end
    found = Ref(false)
    function walk(ex)
        (found[] || !(ex isa Expr)) && return
        if ex.head == :call && !isempty(ex.args)
            f = ex.args[1]
            name = f isa Symbol ? f :
                   (f isa Expr && f.head == :. && f.args[end] isa QuoteNode) ? f.args[end].value : nothing
            name === :clean && (found[] = true; return)
        end
        foreach(walk, ex.args)
    end
    walk(ast)
    return found[]
end

"""
What identifies each dependency checkout of a package: the pin its marker
records (`<name>.resolved`: url, tag, commit), and a signature that moves when
the checkout is cloned or fetched again. A clone makes a new `.git` (another
inode, a new `config`); a fetch rewrites `.git/FETCH_HEAD`; nothing a build
does locally (`rev-parse`, a checkout) touches either. An unpacked archive has
no `.git` and is replaced whole when it is fetched.
"""
function sweep_sources(pkgdir::AbstractString)::Dict{String,Tuple{String,String}}
    deps = joinpath(pkgdir, ".replibuild_cache", "deps")
    out = Dict{String,Tuple{String,String}}()
    isdir(deps) || return out
    for name in readdir(deps)
        d = joinpath(deps, name)
        isdir(d) || continue
        marker = joinpath(deps, name * ".resolved")
        pin = isfile(marker) ? read(marker, String) : ""
        git = joinpath(d, ".git")
        stamp(f) = isfile(f) ? string(mtime(f)) : "-"
        sig = isdir(git) ?
              string(stat(git).inode, " ", stamp(joinpath(git, "config")), " ", stamp(joinpath(git, "FETCH_HEAD"))) :
              string(stat(d).inode)
        out[name] = (pin, sig)
    end
    return out
end

"""
The table's word for a package's sources, and the problem when one was pulled:
cloned or fetched again while its pin stayed what it was. A checkout that was
not there before is new, and one whose pin moved was meant to be fetched.
"""
function sweep_sources_verdict(before::AbstractDict, after::AbstractDict)
    isempty(after) && return ("", String[])
    pulled = String[]; moved = String[]; fresh = String[]
    for name in sort!(collect(keys(after)))
        (pin, sig) = after[name]
        if !haskey(before, name)
            push!(fresh, name)
        elseif sig != before[name][2]
            push!(pin == before[name][1] ? pulled : moved, name)
        end
    end
    cell = !isempty(pulled) ? "PULLED" : !isempty(moved) ? "new pin" : !isempty(fresh) ? "fetched (new)" : "kept"
    problems = isempty(pulled) ? String[] :
               ["sources fetched again though their pin did not change: " * join(pulled, ", ")]
    return (cell, problems)
end

"Does the recipe ask for a compiled thunks library, in any table of it?"
function sweep_wants_thunks(toml_path::AbstractString)::Bool
    t = try
        TOML.parsefile(toml_path)
    catch
        return false
    end
    asks(d) = d isa AbstractDict &&
              any(((k, v),) -> (k == "aot_thunks" && v === true) || asks(v), collect(d))
    return asks(t)
end

function sweep_files(dir::AbstractString, keep)::Vector{String}
    out = String[]
    isdir(dir) || return out
    for (root, _, files) in walkdir(dir), f in files
        keep(f) && push!(out, joinpath(root, f))
    end
    return sort!(out)
end

sweep_wrappers(julia_dir) = sweep_files(julia_dir, f -> endswith(f, ".jl"))
sweep_thunks_libraries(julia_dir) = sweep_files(julia_dir, f -> occursin(r"_thunks\.(so|dll|dylib)$", f))

"Which RepliBuild versions the wrappers under `julia_dir` say wrote them."
function sweep_generators(julia_dir::AbstractString)::Vector{String}
    found = String[]
    for w in sweep_wrappers(julia_dir)
        for l in eachline(w)        # the first stamp of each wrapper
            m = match(r"^const BUILD_GENERATOR\s*=\s*\"([^\"]*)\"", l)
            m === nothing && continue
            push!(found, String(m.captures[1]))
            break
        end
    end
    return sort!(unique!(found))
end

"A digest of every wrapper under `julia_dir`: it moves when one is written again."
function sweep_wrapper_digest(julia_dir::AbstractString)::String
    ctx = SHA.SHA256_CTX()
    for w in sweep_wrappers(julia_dir)
        SHA.update!(ctx, codeunits(relpath(w, julia_dir)))
        SHA.update!(ctx, read(w))
    end
    return bytes2hex(SHA.digest!(ctx))
end

sweep_tree_count(dir) = isdir(dir) ? sum(length(files) for (_, _, files) in walkdir(dir); init = 0) : 0

"""
Copy `<pkg>/julia` to `<save>/julia` once. `:none` when the package has no
wrapper folder, `:kept` when a copy from an earlier run of this sweep is there
(it is the state from before the sweep, and must not be replaced by a later one).
"""
function sweep_backup!(pkgdir::AbstractString, save::AbstractString)::Symbol
    src = joinpath(pkgdir, "julia"); dst = joinpath(save, "julia")
    isdir(src) || return :none
    isdir(dst) && return :kept
    mkpath(save)
    partial = dst * ".partial"
    ispath(partial) && rm(partial; recursive = true)
    cp(src, partial; follow_symlinks = false)
    sweep_tree_count(partial) == sweep_tree_count(src) ||
        error("the copy of $src is incomplete; nothing was changed")
    mv(partial, dst)
    return :saved
end

"""
Put the saved `julia/` back. What is there now moves to `<aside>/julia` first,
so a failed build's output is kept for reading. `false` when nothing was saved.
"""
function sweep_restore!(pkgdir::AbstractString, save::AbstractString, aside::AbstractString)::Bool
    src = joinpath(save, "julia"); dst = joinpath(pkgdir, "julia")
    isdir(src) || return false
    if ispath(dst)
        mkpath(aside)
        kept = joinpath(aside, "julia")
        ispath(kept) && (kept = kept * "." * string(time_ns()))
        mv(dst, kept)
    end
    cp(src, dst; follow_symlinks = false)
    return true
end

"""
Pass, Fail, Error, Broken and Total over every top-level `Test Summary` of a
log, or `nothing` when it has none (the script never reached its tests).
"""
function sweep_test_totals(text::AbstractString)
    lines = split(replace(text, r"\e\[[0-9;]*m" => ""), '\n')
    tot = Dict("Pass" => 0, "Fail" => 0, "Error" => 0, "Broken" => 0, "Total" => 0)
    found = false
    for i in 1:length(lines)-1
        startswith(lines[i], "Test Summary:") || continue
        head = split(lines[i], '|'; limit = 2)
        row = split(lines[i+1], '|')
        (length(head) == 2 && length(row) >= 2) || continue
        for (name, v) in zip(split(head[2]), split(row[end]))
            n = tryparse(Int, v)
            (n === nothing || !haskey(tot, name)) && continue
            tot[name] += n
        end
        found = true
    end
    return found ? tot : nothing
end

"The first line of a log that says what went wrong, for the table."
function sweep_first_error(log::AbstractString)::String
    isfile(log) || return ""
    for l in eachline(log)
        s = strip(replace(l, r"\e\[[0-9;]*m" => ""))
        occursin(r"^(ERROR|error)\b|\berror:|Test Failed|Got exception|signal \(\d+|FAILED ", s) || continue
        return length(s) > 160 ? first(s, 160) * "…" : String(s)
    end
    return ""
end

# ── Running a step ───────────────────────────────────────────────────────────

const SWEEP_RUNNING = Base.Process[]
const SWEEP_LOCK = ReentrantLock()

function sweep_kill(p::Base.Process)
    process_running(p) || return
    if Sys.isunix()
        # The child leads its own process group (`detach`), so its compilers go too.
        ccall(:kill, Cint, (Cint, Cint), -getpid(p), 15)
        sleep(2)
        process_running(p) && ccall(:kill, Cint, (Cint, Cint), -getpid(p), 9)
    else
        kill(p)
    end
end

"""
Run `cmd` in `dir` with its output in `log`, for at most `limit` seconds.
Returns `(status, seconds)`: 0 is success, 124 a step that ran out of time and
was killed, 128 + n a step a signal ended.
"""
function sweep_run(cmd::Cmd, dir::AbstractString, log::AbstractString, limit::Real)
    env = copy(ENV)
    get!(env, "REPLIBUILD_JULIA_ENV", "none")
    c = setenv(cmd, env; dir = dir)
    Sys.isunix() && (c = detach(c))
    t0 = time()
    status = open(log, "w") do io
        p = run(pipeline(c; stdout = io, stderr = io); wait = false)
        lock(SWEEP_LOCK) do; push!(SWEEP_RUNNING, p); end
        try
            while process_running(p)
                if time() - t0 > limit
                    println(io, "\nhub_sweep: killed after $(round(Int, limit)) s")
                    flush(io)
                    sweep_kill(p)
                    wait(p)
                    return 124
                end
                sleep(0.25)
            end
            wait(p)
            return p.termsignal != 0 ? 128 + p.termsignal : p.exitcode
        finally
            lock(SWEEP_LOCK) do; filter!(q -> q !== p, SWEEP_RUNNING); end
        end
    end
    return (status, time() - t0)
end

sweep_julia() = `$(Base.julia_cmd()) --startup-file=no --project=$(dirname(Base.active_project()))`

# ── One package ──────────────────────────────────────────────────────────────

struct SweepStep
    name::String
    status::Int
    seconds::Float64
    note::String
end

mutable struct SweepResult
    pkg::String
    rebuilt_by::String                  # "rebuild", "wrap" or "" (verify only)
    steps::Vector{SweepStep}
    generators_before::Vector{String}
    generators_after::Vector{String}
    thunks::String                      # "", "yes", "MISSING"
    sources::String                     # "", "kept", "fetched (new)", "new pin", "PULLED"
    problems::Vector{String}
    restored::Bool
end

sweep_ok(r::SweepResult) = isempty(r.problems)

"What a sweep of `pkgdir` would run, without running it."
function sweep_plan(o::SweepOptions, pkgdir::AbstractString)
    rebuild = o.mode == :verify ? "" : o.mode == :rewrap ? "wrap" : "rebuild"
    scripts = sort!(String[f for f in readdir(pkgdir) if occursin(r"^test_deep.*\.jl$", f)])
    isfile(joinpath(pkgdir, "test.jl")) && push!(scripts, "test.jl")
    refused = filter(f -> sweep_cleans(joinpath(pkgdir, f)), scripts)
    return (; rebuild, scripts = filter(f -> !(f in refused), scripts), refused,
            wants_thunks = sweep_wants_thunks(joinpath(pkgdir, "replibuild.toml")))
end

function sweep_package(o::SweepOptions, pkg::String, record)::SweepResult
    pkgdir = joinpath(sweep_packages_dir(o.hub), pkg)
    julia_dir = joinpath(pkgdir, "julia")
    save = joinpath(o.out, "before", pkg)
    logs = joinpath(o.out, "logs")
    limit = get(o.timeouts, pkg, o.timeout)
    plan = sweep_plan(o, pkgdir)
    r = SweepResult(pkg, plan.rebuild, SweepStep[], sweep_generators(julia_dir), String[], "", "", String[], false)
    had_thunks = !isempty(sweep_thunks_libraries(julia_dir))
    sources_before = sweep_sources(pkgdir)
    for f in plan.refused
        push!(r.problems, "$f calls `clean`: not run (a test script must not clean or pull)")
    end

    function step!(name, cmd)
        (status, secs) = sweep_run(cmd, pkgdir, joinpath(logs, "$pkg.$name.log"), limit)
        totals = sweep_test_totals(read(joinpath(logs, "$pkg.$name.log"), String))
        note = totals === nothing ? "" :
               string(totals["Pass"], "/", totals["Total"],
                      totals["Broken"] > 0 ? " ($(totals["Broken"]) broken)" : "")
        s = SweepStep(name, status, secs, note)
        push!(r.steps, s); record(pkg, s)
        if status != 0
            why = status == 124 ? "ran out of time ($(round(Int, limit)) s)" :
                  status > 128 ? "ended by signal $(status - 128)" : "exit $status"
            first_line = sweep_first_error(joinpath(logs, "$pkg.$name.log"))
            push!(r.problems, "$name: $why" * (isempty(note) ? "" : ", $note") *
                              (isempty(first_line) ? "" : " — " * first_line))
        end
        return status
    end

    if plan.rebuild != ""
        sweep_backup!(pkgdir, save)
        step!(plan.rebuild, `$(sweep_julia()) -e $SWEEP_CHILD -- replibuild.toml $(plan.rebuild)`)
        if isempty(sweep_wrappers(julia_dir))
            # The build took the wrapper and gave none back.
            r.restored = sweep_restore!(pkgdir, save, joinpath(o.out, "failed", pkg))
            push!(r.problems, "no wrapper after the rebuild" *
                              (r.restored ? "; the saved `julia/` is back" : "; nothing was saved to put back"))
            isempty(r.steps) || r.steps[end].status != 0 ||
                push!(r.problems, "the step reported success")
        else
            version = string(RepliBuild.VERSION)
            written = sweep_generators(julia_dir)
            written == [version] ||
                push!(r.problems, "wrapper written by $(isempty(written) ? "nothing that says" : join(written, ", ")), not $version")
        end
    end

    if !r.restored && !isempty(sweep_wrappers(julia_dir))
        for f in plan.scripts
            extra = startswith(f, "test_deep") ? get(o.args, pkg, String[]) : String[]
            step!(splitext(f)[1], `$(sweep_julia()) $f $extra`)
        end
    elseif plan.rebuild == ""
        push!(r.problems, "no wrapper to verify")
    end
    isempty(plan.scripts) && isempty(plan.refused) && record(pkg, SweepStep("note", 0, 0.0, "no test script"))

    # Judged at the end, after every step: a test script wraps again, and any
    # step could have gone back to the network.
    if !r.restored && !isempty(sweep_wrappers(julia_dir))
        r.generators_after = sweep_generators(julia_dir)
        if plan.wants_thunks || had_thunks
            have = !isempty(sweep_thunks_libraries(julia_dir))
            r.thunks = have ? "yes" : "MISSING"
            have || push!(r.problems, "no compiled thunks library (" *
                          (plan.wants_thunks ? "the recipe sets aot_thunks" : "it had one before") * ")")
        end
    end
    (r.sources, pulled) = sweep_sources_verdict(sources_before, sweep_sources(pkgdir))
    append!(r.problems, pulled)
    return r
end

# ── The table ────────────────────────────────────────────────────────────────

function sweep_summary(results::Vector{SweepResult}, left_out::Vector{String})::String
    io = IOBuffer()
    bad = count(!sweep_ok, results)
    println(io, "# Hub sweep, ", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM"))
    println(io)
    println(io, length(results), " package(s): ", length(results) - bad, " clean, ", bad, " with a problem.",
            isempty(left_out) ? "" : " Left out: " * join(left_out, ", ") * ".")
    println(io)
    println(io, "| Package | Rebuild s | Generator | Sources | Thunks library | Tests | Verdict |")
    println(io, "|:--|--:|:--|:--|:--|:--|:--|")
    for r in results
        build = filter(s -> s.name in ("rebuild", "wrap"), r.steps)
        secs = isempty(build) ? "" : string(round(Int, sum(s.seconds for s in build)))
        # Before → after, or the one there is (a package never built, or one put back).
        gen = isempty(r.generators_after) ? join(r.generators_before, ", ") :
              (isempty(r.generators_before) || r.generators_before == r.generators_after) ?
              join(r.generators_after, ", ") :
              join(r.generators_before, ", ") * " → " * join(r.generators_after, ", ")
        tests = join(("$(s.name) $(s.note)" for s in r.steps if !isempty(s.note)), "; ")
        println(io, "| ", r.pkg, " | ", secs, " | ", gen, " | ", r.sources, " | ", r.thunks, " | ",
                tests, " | ", sweep_ok(r) ? "ok" : "**FAIL**", " |")
    end
    if bad > 0
        println(io, "\n## Problems\n")
        for r in results, p in r.problems
            println(io, "- **", r.pkg, "**: ", p)
        end
    end
    return String(take!(io))
end

function sweep_environment(o::SweepOptions, pkgs::Vector{String})::String
    git(dir, args...) = try
        readchomp(pipeline(`git -C $dir $args`; stderr = devnull))
    catch
        "unknown"
    end
    lines = String[
        "started: $(Dates.now())",
        "RepliBuild: $(RepliBuild.VERSION) at $(pkgdir(RepliBuild)), HEAD $(git(pkgdir(RepliBuild), "rev-parse", "--short", "HEAD"))",
        "RepliBuild uncommitted files: $(length(split(git(pkgdir(RepliBuild), "status", "--porcelain"), '\n'; keepempty = false)))",
        "Julia: $VERSION",
        "Hub: $(o.hub), HEAD $(git(o.hub, "rev-parse", "--short", "HEAD"))",
        "mode: $(o.mode), jobs $(o.jobs), timeout $(o.timeout) s",
        "packages ($(length(pkgs))): $(join(pkgs, ' '))",
    ]
    for k in sort!(collect(keys(ENV)))
        startswith(k, "REPLIBUILD_") && push!(lines, "$k=$(ENV[k])")
    end
    return join(lines, "\n") * "\n"
end

"Packages `<out>/status.tsv` already has a verdict line for."
function sweep_done(out::AbstractString)::Set{String}
    done = Set{String}()
    f = joinpath(out, "status.tsv")
    isfile(f) || return done
    for l in eachline(f)
        c = split(l, '\t')
        length(c) >= 2 && c[2] == "verdict" && push!(done, String(c[1]))
    end
    return done
end

function sweep_main(args::Vector{String})::Int
    o = sweep_parse_args(args)
    o.help && (println(sweep_usage()); return 0)
    every = sweep_packages(o.hub)
    pkgs = sweep_select(o, every)
    left_out = isempty(o.only) ? String[p for p in SWEEP_DEFAULT_SKIP if p in every] : String[]
    isempty(pkgs) && error("no package selected")

    if o.plan
        for p in pkgs
            plan = sweep_plan(o, joinpath(sweep_packages_dir(o.hub), p))
            println(rpad(p, 18), isempty(plan.rebuild) ? "" : plan.rebuild * ", then ", join(plan.scripts, ", "),
                    isempty(plan.refused) ? "" : "   [NOT RUN, calls clean: " * join(plan.refused, ", ") * "]",
                    plan.wants_thunks ? "   [thunks library expected]" : "")
        end
        isempty(left_out) || println("left out: ", join(left_out, ", "))
        return 0
    end

    if o.restore
        for p in pkgs
            back = sweep_restore!(joinpath(sweep_packages_dir(o.hub), p), joinpath(o.out, "before", p),
                                  joinpath(o.out, "replaced", p))
            println(rpad(p, 18), back ? " restored" : " nothing saved under $(o.out)")
        end
        return 0
    end

    mkpath(joinpath(o.out, "logs"))
    if o.resume
        done = sweep_done(o.out)
        pkgs = String[p for p in pkgs if !(p in done)]
        isempty(pkgs) && (println("every selected package already has a verdict in $(o.out)"); return 0)
    end
    write(joinpath(o.out, "environment.txt"), sweep_environment(o, pkgs))
    status = joinpath(o.out, "status.tsv")
    function record(pkg, s::SweepStep)
        lock(SWEEP_LOCK) do
            open(status, "a") do io
                println(io, join((pkg, s.name, s.status, round(Int, s.seconds), s.note), '\t'))
            end
            println(rpad(pkg, 18), rpad(s.name, 18), s.status == 0 ? "ok  " : "FAIL",
                    lpad(round(Int, s.seconds), 6), " s  ", s.note)
        end
    end
    println("hub_sweep: $(length(pkgs)) package(s), $(o.mode), into $(o.out)")
    results = try
        asyncmap(pkgs; ntasks = o.jobs) do p
            r = try
                sweep_package(o, p, record)
            catch e
                e isa InterruptException && rethrow()
                x = SweepResult(p, "", SweepStep[], String[], String[], "", "", String[], false)
                push!(x.problems, "the sweep itself failed here: " * first(split(sprint(showerror, e), '\n')))
                x
            end
            record(p, SweepStep("verdict", sweep_ok(r) ? 0 : 1, sum((s.seconds for s in r.steps); init = 0.0),
                                sweep_ok(r) ? "ok" : join(r.problems, " | ")))
            r
        end
    catch
        lock(SWEEP_LOCK) do; foreach(sweep_kill, copy(SWEEP_RUNNING)); end
        rethrow()
    end
    sort!(results; by = r -> r.pkg)
    text = sweep_summary(results, left_out)
    write(joinpath(o.out, "summary.md"), text)
    println(); print(text)
    println("\nlogs and saved wrappers: ", o.out)
    return all(sweep_ok, results) ? 0 : 1
end

const _SWEEP_AS_SCRIPT = abspath(PROGRAM_FILE) == abspath(@__FILE__)
_SWEEP_AS_SCRIPT && exit(sweep_main(String.(ARGS)))
