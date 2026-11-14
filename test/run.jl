#!/usr/bin/env julia
# test/run.jl — run a suite, an area of it, or what a diff touches.
#
#     julia --project=. test/run.jl dev                   # every dev job
#     julia --project=. test/run.jl dev --area=c,build    # the jobs that cover those areas
#     julia --project=. test/run.jl dev --changed         # what the diff against origin/main touches
#     julia --project=. test/run.jl dev --changed=HEAD~3  # …against another base
#     julia --project=. test/run.jl unit                  # the toolchain-free tests, one process
#     julia --project=. test/run.jl unit --changed        # the unit files for the diff
#     julia --project=. test/run.jl dev --files=test_c_sret_abi.jl,test_abi_nested.jl
#     julia --project=. test/run.jl dev --list            # print the selection, run nothing
#
# Options: --processes (a child per job, in parallel: --jobs=N, default
# min(4, threads ÷ 3), or --serial), --tail (print each child's log as it
# finishes), --timeout=S (per child), --no-deps (do not add the fixture jobs a
# selected file needs), --list.
#
# Every job is a test file or a fixture build (`fixture:<name>`, see
# test/support/fixture.jl) from test/suites.jl. By default the selected jobs run
# in THIS process, in order, under one testset, as the suites always ran: a
# child must compile RepliBuild's build and wrap paths again before its first
# build, about fifty seconds, so children are not faster (48 of them, four at a
# time: 597s against 614s in one process, 2026-11-11). `--processes` buys
# isolation and a log per job (test/.logs/<job>.log, one terminal line per job,
# the `Test Failed` excerpts at the end): two jobs locking one fixture
# directory never run at once, a job runs after the jobs it names and is
# skipped when one of those failed. REPLIBUILD_SKIP_TESTS names fixtures to skip
# (with the jobs that need them, in --processes mode). Exit status 1 on failure.

using Dates

include(joinpath(@__DIR__, "suites.jl"))

const ROOT = dirname(@__DIR__)
const LOGS = joinpath(@__DIR__, ".logs")

struct Job
    id::String
    script::String            # relative to ROOT
    args::Vector{String}
    locks::Vector{String}
    after::Vector{String}
    areas::Vector{Symbol}
    timeout::Int
end

# ── Arguments ────────────────────────────────────────────────────────────────

function parse_args(argv)
    opts = Dict{String,Any}("suite" => nothing, "area" => Symbol[], "changed" => nothing, "files" => String[],
                            "jobs" => max(1, min(4, Sys.CPU_THREADS ÷ 3)), "list" => false, "inprocess" => nothing,
                            "deps" => true, "tail" => false, "timeout" => nothing)
    for a in argv
        if a in ("pkg", "unit", "dev", "all")
            opts["suite"] = Symbol(a)
        elseif startswith(a, "--area=")
            opts["area"] = Symbol.(split(a[8:end], ','; keepempty = false))
        elseif a == "--changed"
            opts["changed"] = "origin/main"
        elseif startswith(a, "--changed=")
            opts["changed"] = String(a[11:end])
        elseif startswith(a, "--files=")
            opts["files"] = String.(split(a[9:end], ','; keepempty = false))
        elseif startswith(a, "--jobs=")
            opts["jobs"] = parse(Int, a[8:end])
        elseif a == "--serial"
            opts["jobs"] = 1
        elseif a == "--inprocess"
            opts["inprocess"] = true
        elseif a == "--processes"
            opts["inprocess"] = false
        elseif a == "--no-deps"
            opts["deps"] = false
        elseif a == "--list"
            opts["list"] = true
        elseif a == "--tail"
            opts["tail"] = true
        elseif startswith(a, "--timeout=")
            opts["timeout"] = parse(Int, a[11:end])
        elseif a in ("-h", "--help")
            println(read(@__FILE__, String)[1:findfirst("using Dates", read(@__FILE__, String))[1]-1])
            exit(0)
        else
            error("run.jl: unknown argument `$a` (try --help)")
        end
    end
    opts["suite"] === nothing && error("run.jl: name a suite: pkg, unit, dev or all (try --help)")
    return opts
end

# ── The jobs of a suite ──────────────────────────────────────────────────────

function suite_jobs(suite::Symbol)
    jobs = Job[]
    suite === :pkg && return [Job("runtests.jl", "test/runtests.jl", String[], String[], String[], Symbol[:build], 900)]
    if suite === :dev
        for f in FIXTURES
            push!(jobs, Job("fixture:" * f.name, "test/support/fixture.jl", [f.name], [f.name], String[], f.areas, 2400))
        end
    end
    for t in TESTS
        t.suite === suite || continue
        push!(jobs, Job(t.file, joinpath("test", t.file), String[], t.locks, t.after, t.areas, t.timeout))
    end
    return jobs
end

# ── Selection ────────────────────────────────────────────────────────────────

"Repository-relative paths the diff against `base` touches, plus uncommitted and untracked files."
function changed_paths(base::AbstractString)
    paths = Set{String}()
    git(args) = readlines(Cmd(`git -C $ROOT $args`; ignorestatus = true))
    mb = strip(join(git(`merge-base $base HEAD`), ""))
    isempty(mb) && error("run.jl: `git merge-base $base HEAD` found nothing; fetch the base or pass --changed=<ref>")
    foreach(l -> isempty(l) || push!(paths, l), git(`diff --name-only $mb HEAD`))
    for l in git(`status --porcelain --untracked-files=all`)
        length(l) > 3 || continue
        p = strip(l[4:end])
        occursin(" -> ", p) && (p = strip(split(p, " -> ")[end]))
        push!(paths, p)
    end
    return sort!(collect(paths))
end

"""
The jobs a change selects: every job sharing an area with a changed source
path, every changed test file, every job locking a changed fixture directory,
and everything when a path everything depends on moved.
"""
function select_changed(jobs::Vector{Job}, paths::Vector{String})
    areas = Set{Symbol}()
    picked = Set{String}()
    for p in paths
        a = areas_of_path(p)
        a === :all && return jobs, "everything: `$p` changed"
        union!(areas, a)
        if startswith(p, "test/")
            rel = p[6:end]
            for j in jobs
                (j.id == rel || j.script == p) && push!(picked, j.id)
                any(l -> startswith(rel, l * "/"), j.locks) && push!(picked, j.id)
            end
        end
    end
    sel = Job[j for j in jobs if j.id in picked || !isempty(intersect(Set(j.areas), areas))]
    why = isempty(areas) ? "no source area touched" : "areas " * join(sort(string.(collect(areas))), ", ")
    return sel, why
end

"Add the jobs the selected ones run `after`, transitively."
function with_deps(sel::Vector{Job}, all::Vector{Job})
    byid = Dict(j.id => j for j in all)
    out = Dict(j.id => j for j in sel)
    queue = collect(sel)
    while !isempty(queue)
        j = pop!(queue)
        for d in j.after
            haskey(out, d) && continue
            haskey(byid, d) || error("run.jl: `$(j.id)` runs after `$d`, which no suite defines (test/suites.jl)")
            out[d] = byid[d]
            push!(queue, byid[d])
        end
    end
    order = Dict(j.id => i for (i, j) in enumerate(all))
    return sort!(collect(values(out)); by = j -> order[j.id])
end

# ── Running ──────────────────────────────────────────────────────────────────

logfile(j::Job) = joinpath(LOGS, replace(j.id, r"[^A-Za-z0-9_.-]" => "_") * ".log")

# Every child gets its share of the machine's threads, not all of them: a
# build runs its compiles under `Threads.@threads`, and four children each
# believing they own every core ran slower than one process (11m32s against
# 10m14s on 12 threads, 2026-11-11). `Base.julia_cmd()` carries the parent's
# thread count, so it is overridden on the command line.
function child_cmd(j::Job, njobs::Int)
    home = mktempdir()
    threads = max(1, Sys.CPU_THREADS ÷ max(1, njobs))
    cmd = `$(Base.julia_cmd()) --project=$ROOT --color=no --threads=$threads $(joinpath(ROOT, j.script)) $(j.args)`
    return addenv(cmd, "REPLIBUILD_HOME" => home, "REPLIBUILD_JULIA_ENV" => "none",
                  "REPLIBUILD_RUNNER" => "1", "JULIA_NUM_THREADS" => string(threads))
end

fmt(s::Real) = s >= 60 ? string(round(Int, s ÷ 60), "m", lpad(round(Int, s % 60), 2, '0'), "s") :
               string(round(s; digits = 1), "s")

function run_parallel(jobs::Vector{Job}; njobs::Int, tail::Bool, timeout)
    mkpath(LOGS)
    skip = Set(split(get(ENV, "REPLIBUILD_SKIP_TESTS", ""), ','; keepempty = false))
    status = Dict{String,Symbol}()          # :pass | :fail | :skip | :timeout
    times = Dict{String,Float64}()
    reason = Dict{String,String}()
    # Longest first, by the last run's times (summary.tsv): the long graph and
    # fixture jobs then start early instead of trailing alone at the end.
    prior = Dict{String,Float64}()
    let tsv = joinpath(LOGS, "summary.tsv")
        if isfile(tsv)
            for l in Iterators.drop(eachline(tsv), 1)
                f = split(l, '\t')
                length(f) >= 3 && (prior[String(f[1])] = something(tryparse(Float64, f[3]), 0.0))
            end
        end
    end
    pending = sort(jobs; by = j -> -get(prior, j.id, 120.0))
    running = Dict{String,Any}()            # id => (proc, job, t0)
    held = Set{String}()
    width = maximum(length(j.id) for j in jobs; init = 10)
    report(j, verdict, t, note = "") =
        println(rpad(verdict, 6), " ", rpad(j.id, width), " ", lpad(fmt(t), 7), isempty(note) ? "" : "  " * note)

    for j in jobs
        fx = startswith(j.id, "fixture:") ? j.id[9:end] : ""
        if fx in skip
            status[j.id] = :skip
            reason[j.id] = "REPLIBUILD_SKIP_TESTS"
        end
    end
    t_start = time()
    while !isempty(pending) || !isempty(running)
        # Start what is ready.
        i = 1
        while i <= length(pending) && length(running) < njobs
            j = pending[i]
            if haskey(status, j.id)
                deleteat!(pending, i)
                report(j, "skip", 0.0, reason[j.id])
                continue
            end
            blocked = any(d -> get(status, d, nothing) in (:fail, :skip, :timeout), j.after)
            if blocked
                d = first(filter(d -> get(status, d, nothing) in (:fail, :skip, :timeout), j.after))
                status[j.id] = :skip
                reason[j.id] = "after $d, which $(status[d] === :skip ? "was skipped" : "failed")"
                deleteat!(pending, i)
                report(j, "skip", 0.0, reason[j.id])
                continue
            end
            ready = all(d -> get(status, d, nothing) === :pass, j.after) && isempty(intersect(Set(j.locks), held))
            if ready
                log = logfile(j)
                proc = run(pipeline(child_cmd(j, njobs); stdout = log, stderr = log); wait = false)
                running[j.id] = (proc, j, time())
                union!(held, j.locks)
                deleteat!(pending, i)
            else
                i += 1
            end
        end
        isempty(running) && !isempty(pending) && break   # nothing can start: a cycle or an unknown dep
        # Collect what finished.
        for (id, (proc, j, t0)) in collect(running)
            limit = something(timeout, j.timeout)
            if process_running(proc)
                if time() - t0 > limit
                    kill(proc, Base.SIGKILL)
                    wait(proc)
                    status[id] = :timeout
                    times[id] = time() - t0
                    report(j, "TIME", times[id], "killed after $(limit)s → $(relpath(logfile(j), ROOT))")
                else
                    continue
                end
            else
                times[id] = time() - t0
                if success(proc)
                    status[id] = :pass
                    report(j, "ok", times[id])
                else
                    status[id] = :fail
                    report(j, "FAIL", times[id], "→ " * relpath(logfile(j), ROOT))
                end
            end
            tail && (println(); println(read(logfile(j), String)); println())
            delete!(running, id)
            setdiff!(held, j.locks)
        end
        isempty(running) || sleep(0.5)
    end
    for j in pending
        status[j.id] = :skip
        reason[j.id] = "never became ready (a cycle in `after`, or a lock never released)"
        report(j, "skip", 0.0, reason[j.id])
    end
    return (; status, times, reason, wall = time() - t_start)
end

"The `Test Failed` and error excerpts of a child's log, at most `maxlines` lines."
function failure_excerpt(path::AbstractString; maxlines = 60)
    isfile(path) || return "(no log)"
    lines = readlines(path)
    out = String[]
    i = 1
    while i <= length(lines) && length(out) < maxlines
        l = lines[i]
        if occursin(r"Test Failed at|Error During Test at|^ERROR:|LoadError|signal \(11\)|Segmentation fault", l)
            append!(out, lines[i:min(i + 4, end)])
            push!(out, "")
            i += 5
        else
            i += 1
        end
    end
    isempty(out) && (out = lines[max(1, end - 15):end])
    return join(out, "\n")
end

function summarize(jobs, r)
    n(s) = count(j -> get(r.status, j.id, :skip) === s, jobs)
    println()
    println("── $(length(jobs)) job(s) in $(fmt(r.wall)): $(n(:pass)) ok, $(n(:fail)) failed, ",
            "$(n(:timeout)) timed out, $(n(:skip)) skipped")
    open(joinpath(LOGS, "summary.tsv"), "w") do io
        println(io, "job\tstatus\tseconds\tnote")
        for j in jobs
            println(io, j.id, "\t", get(r.status, j.id, :skip), "\t", round(get(r.times, j.id, 0.0); digits = 1),
                    "\t", get(r.reason, j.id, ""))
        end
    end
    bad = [j for j in jobs if get(r.status, j.id, :skip) in (:fail, :timeout)]
    for j in bad
        println()
        println("══ ", j.id, " (", r.status[j.id], ") ", relpath(logfile(j), ROOT))
        println(failure_excerpt(logfile(j)))
    end
    return isempty(bad)
end

"""
Include the selected jobs in this process, in order, under one testset. A
fixture job calls `run_fixture` (test/support/fixture.jl). This is how the old
single-process suites ran, and it is the fastest way: every child pays Julia's
compilation of the build and wrap paths again, about fifty seconds before its
first build, so 48 children four at a time took as long as one process (597s
against 614s) and a 16-job selection took 2m30s either way (2026-11-11). Unlike
the old suites, a failing file does not end the run: the jobs are nested
testsets, so every selected job runs and the summary names the failures.
"""
function run_inprocess(jobs::Vector{Job}, suite::Symbol)
    @eval Main begin
        using Test
        using RepliBuild
    end
    ENV["REPLIBUILD_HOME"] = mktempdir()
    ENV["REPLIBUILD_JULIA_ENV"] = "none"
    any(j -> startswith(j.id, "fixture:"), jobs) && Base.include(Main, joinpath(@__DIR__, "support", "fixture.jl"))
    t0 = time()
    @eval Main Test.@testset $(string(suite)) begin
        for j in $jobs
            t = time()
            if startswith(j.id, "fixture:")
                Main.run_fixture(j.id[9:end])
            else
                Base.include(Main, joinpath($ROOT, j.script))
            end
            println(rpad("ran", 6), " ", j.id, " ", lpad($fmt(time() - t), 7))
        end
    end
    println("── $(length(jobs)) job(s) in $(fmt(time() - t0)); the testset summary above has the verdicts")
    return true
end

# ── Main ─────────────────────────────────────────────────────────────────────

function main(argv)
    opts = parse_args(argv)
    suite = opts["suite"]
    all_jobs = suite === :all ? vcat(suite_jobs(:pkg), suite_jobs(:unit), suite_jobs(:dev)) : suite_jobs(suite)
    sel = all_jobs
    why = "the whole $(suite) suite"
    if opts["changed"] !== nothing
        paths = changed_paths(opts["changed"])
        sel, why = select_changed(all_jobs, paths)
        why = "$(length(paths)) changed path(s) against $(opts["changed"]): $why"
    end
    if !isempty(opts["area"])
        want = Set(opts["area"])
        sel = Job[j for j in sel if !isempty(intersect(Set(j.areas), want))]
        why *= "; areas " * join(string.(opts["area"]), ", ")
    end
    if !isempty(opts["files"])
        want = Set(opts["files"])
        sel = Job[j for j in sel if j.id in want || basename(j.id) in want]
        missing = setdiff(want, Set(vcat([j.id for j in sel], [basename(j.id) for j in sel])))
        isempty(missing) || error("run.jl: not in the $(suite) suite: " * join(sort(collect(missing)), ", "))
        why *= "; files " * join(opts["files"], ", ")
    end
    opts["deps"] && (sel = with_deps(sel, all_jobs))
    println("run.jl: $(length(sel)) of $(length(all_jobs)) $(suite) job(s) — $why")
    if opts["list"]
        foreach(j -> println("  ", j.id, isempty(j.after) ? "" : "  (after " * join(j.after, ", ") * ")"), sel)
        return 0
    end
    isempty(sel) && return 0
    inprocess = something(opts["inprocess"], true)   # `--processes` opts out; see `run_inprocess`
    if inprocess
        return run_inprocess(sel, suite) ? 0 : 1
    end
    r = run_parallel(sel; njobs = opts["jobs"], tail = opts["tail"], timeout = opts["timeout"])
    return summarize(sel, r) ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
