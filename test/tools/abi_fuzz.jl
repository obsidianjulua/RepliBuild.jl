#!/usr/bin/env julia
# =============================================================================
# abi_fuzz — differential ABI testing for RepliBuild, with the C compiler as
# the oracle
# =============================================================================
#
# A HAND-RUN DEVELOPER TOOL, NOT A TEST. It is wired into no suite. It finds
# wrapper bugs by generating C shapes instead of wrapping more libraries.
#
#     julia --project=. test/tools/abi_fuzz.jl --seeds=1:100 --dir=/short/path
#     julia --project=. test/tools/abi_fuzz.jl --seeds=1:100 --ident
#     julia --project=. test/tools/abi_fuzz.jl --seeds=1:30 --neuter=resolve_exact_layout
#     julia --project=. test/tools/abi_fuzz.jl --target=3/c/call/f2
#     julia --project=. test/tools/abi_fuzz.jl --repro=<dir>/repros/<case>
#
# Each seed (`Random.Xoshiro`) generates one small C translation unit:
# structs and unions of every C scalar, pointers, fixed arrays, nested records,
# enums, C11 anonymous members, bitfields, `__attribute__((packed))` and
# `#pragma pack(n)`, sized mostly 1–32 bytes so they straddle Win64's
# 1/2/4/8 rule and SysV's 16-byte cutoff; functions that take records by
# value and by pointer among scalars (some with enough arguments to spill past
# the argument registers) and return a record or a scalar; and an exported
# table of `sizeof`/`offsetof` for every record and member. A companion unit
# holds a second enum with the same member set (C forbids that in one unit).
#
# The unit is wrapped twice in fresh projects with
# `RepliBuild.discover(dir, build=true, wrap=true)`:
#   c    the `.c` unit, wrapped as C: Tier 3, ccall.
#   cpp  the same text as `.cpp` (C++ linkage), so every function routes to a
#        Tier 2 thunk. Skipped, with a note, when libJLCS does not load.
# Tier 1 is never enabled. Building happens in this process; every check runs
# in a child (`run_isolated`, SIGKILL on timeout) that prints the probe
# protocol: one `PROBE <seed>/<mode>/<check>: <verdict>` per case, then
# `PROBE_DONE`.
#
# THE CHECKS (verdicts: PASS, FAIL, GAP = coverage gap, XFAIL = expected
# refusal, BLOCKED = could not run because `load` failed, SAC = Windows Smart
# App Control refused the freshly built DLL, which says nothing about
# RepliBuild):
#
#   load            the wrapper loads; every C function is found in it, either
#                   callable or a refusal stub. A `long double` refusal is
#                   expected (CLAUDE.md lists it as an ABI trap on both tiers);
#                   any other refusal or missing function is a GAP.
#   layout/<R>      `sizeof` of the emitted type, `struct_size(:R)`,
#                   `member_offset(:R, :m)` and the field offsets of a named
#                   Julia struct equal the C table, which the child reads
#                   through `cglobal` (no call involved).
#   access/<R>      no call either: a value made from a C image reads back every
#                   member through the wrapper's accessors (fields,
#                   `getproperty`, `get_*`), and setting every member through
#                   them (`with`, `setproperty`, `set_*`) reproduces the image.
#   call/<f>        every argument member is set to a distinct value through the
#                   wrapper's field access, the call is made, and each value is
#                   compared with what the callee stored, as C read it
#                   (`x.a`, `x.b.c[2]`), in an exported array. A mismatch names
#                   the argument and the member.
#   ret/<f>         each member of the returned record (or the scalar) equals
#                   the formula on (seed, function, member) the callee used.
#   exit            the child exited cleanly after its last check.
#
# A crash or a timeout is a FAIL that carries the child's output; the child is
# rerun without the crashed check so it cannot hide the rest. The run asserts
# that every check planned from the generated unit came back with a verdict.
#
# GAPS are listed apart from FAILs, one `GAP <unit>/<kind>: <detail>` per case:
#   trap        a refusal stub other than `long double` (an ABI Safety Trap)
#   unwrapped   a C function with no Julia function, or one making no call
#   no-getter   a member the wrapper offers no way to read
#   no-setter   a member it offers no way to set (the call check then writes
#               that member's bytes at its C offset, so the crossing is still
#               tested)
#   no-size / no-offset   `struct_size` / `member_offset` cannot answer
#   no-with     the wrapper defines no `with` (the default constructor is used)
#   param-type  a parameter declared narrower than its C type admits (`::UInt8`
#               for a C `char` rejects a `Cchar`)
#
# A FAIL is delta-debugged (`--minimise=N`, default 3 failure groups, or one
# `--target` per process): drop functions, then parameters, records and members,
# then simplify types, while the same check still fails (a crash and a wrong
# value count as the same failure). The minimised `.c`, the seed and the check
# go to `<dir>/repros/` (outside the repository, or under the gitignored
# `.claude/`).
#
# OPTIONS
#   --seeds=1:100 | 3,17,42   seeds (default 1:20)
#   --modes=c,cpp             default both
#   --ident                   identifier-stress units: Julia keywords, Base and
#                             Core names, names the wrapper itself defines,
#                             all-underscore and leading-underscore names, and
#                             names that differ only in case
#   --dir=PATH                work directory (default $ABI_FUZZ_DIR, else
#                             <tempdir>/abi_fuzz). Keep it SHORT on Windows:
#                             MAX_PATH breaks deep build trees.
#   --jobs=N                  children in parallel (default 4)
#   --timeout=S               per child, seconds (default 900)
#   --minimise=N              failures to minimise (default 3; 0 = off)
#   --budget=N                builds per minimisation (default 120)
#   --neuter=NAME             negative check: put a known bug back in-process
#                             (see abi_fuzz/neuter.jl for the names)
#   --repro=DIR               rebuild and recheck a saved repro
#   --target=SEED/MODE/CHECK  regenerate one unit, confirm that check fails, and
#                             minimise it (`3/c/call/f2`, `7n/cpp/load`; an `n`
#                             after the seed is identifier mode)
#   --keep                    keep the build directories of passing units
#
# Results: <dir>/results-<run>.tsv (one row per check), <dir>/gaps-<run>.tsv,
# and a summary on stdout. Exit status 0 = no FAIL, 1 = FAILs found,
# 2 = the harness lost a check.
#
# The registry is redirected to <dir>/_rbhome (REPLIBUILD_HOME), because
# `discover` registers every project it sees.
#
# Smart App Control (Windows): with it on, the loader can refuse a DLL this
# tool just built. Such a unit is SAC, counted per mode in the summary, never a
# FAIL; the evidence is the loader's message or a Code Integrity event 3077
# naming the DLL. The minimiser treats a blocked candidate as unknown. Nothing
# retries, rebuilds or alters a blocked file.
#
# Portable by construction: nothing here knows the ABI. The table comes from
# the compiler that built the unit, so on Linux this checks SysV and on
# Windows Win64 with the same code.
# =============================================================================

using Dates
using Serialization
using Printf
using Libdl
using Logging

const TOOLS = @__DIR__
const TEST_DIR = dirname(TOOLS)
const CHILD = joinpath(TOOLS, "abi_fuzz", "child.jl")

# ── Options ──────────────────────────────────────────────────────────────────

function parse_seeds(s)
    out = Int[]
    for part in split(s, ',')
        if occursin(':', part)
            a, b = parse.(Int, split(part, ':'))
            append!(out, a:b)
        else
            push!(out, parse(Int, part))
        end
    end
    return out
end

function options(args)
    o = Dict{String,String}("seeds" => "1:20", "modes" => "c,cpp", "jobs" => "4", "timeout" => "900",
                            "minimise" => "3", "budget" => "120", "neuter" => "", "repro" => "", "target" => "",
                            "dir" => get(ENV, "ABI_FUZZ_DIR", joinpath(tempdir(), "abi_fuzz")))
    flags = Set{String}()
    for a in args
        m = match(r"^--([a-z-]+)(?:=(.*))?$", a)
        m === nothing && error("abi_fuzz: unknown argument `$a`")
        k = m.captures[1]
        if m.captures[2] === nothing
            k in ("ident", "keep") || error("abi_fuzz: --$k needs a value")
            push!(flags, k)
        else
            haskey(o, k) || error("abi_fuzz: unknown option --$k")
            o[k] = m.captures[2]
        end
    end
    return (seeds = parse_seeds(o["seeds"]), modes = Symbol.(split(o["modes"], ',')),
            ident = "ident" in flags, keep = "keep" in flags, dir = abspath(o["dir"]),
            jobs = parse(Int, o["jobs"]), timeout = parse(Int, o["timeout"]),
            minimise = parse(Int, o["minimise"]), budget = parse(Int, o["budget"]),
            neuter = o["neuter"], repro = o["repro"], target = o["target"])
end

const _AS_SCRIPT = abspath(PROGRAM_FILE) == abspath(@__FILE__)
const OPT = options(_AS_SCRIPT ? ARGS : String.(split(get(ENV, "ABI_FUZZ_ARGS", ""))))
mkpath(OPT.dir)
ENV["REPLIBUILD_HOME"] = joinpath(OPT.dir, "_rbhome")
# `discover(build=true, wrap=true)` registers each unit, and `register` adds the
# package to the default Julia environment unless told otherwise. One
# `--seeds=1:60` run put 180 path deps into `@v1.13` (2026-10-24).
ENV["REPLIBUILD_JULIA_ENV"] = "none"

import RepliBuild
isdefined(@__MODULE__, :TestSupport) || include(joinpath(TEST_DIR, "support", "TestSupport.jl"))
using .TestSupport: run_isolated
include(joinpath(TOOLS, "abi_fuzz", "model.jl"))
using .AbiFuzzModel
const AF = AbiFuzzModel
include(joinpath(TOOLS, "abi_fuzz", "neuter.jl"))

const PROJECT = dirname(Base.active_project())

isempty(OPT.neuter) || apply_neuter(RepliBuild, OPT.neuter; side = :parent)

# ── Smart App Control ────────────────────────────────────────────────────────
#
# On a Windows machine with Smart App Control on, the loader can refuse a DLL
# this tool just built: SAC blocks unsigned files Microsoft's cloud cannot
# clear, so which builds are refused is not predictable. Such a unit says
# nothing about RepliBuild. It gets the status SAC (counted per mode, never a
# FAIL), its other checks are BLOCKED, and the minimiser treats a blocked
# candidate as unknown: neither still failing nor passing. Nothing here retries,
# rebuilds or alters a blocked file.
#
# Evidence: the loader's message ("An Application Control policy has blocked
# this file") or a Code Integrity event 3077 naming the DLL.

const RUN_START = now()
const SAC_MSG = "Application Control policy has blocked"

_normpath_ci(p) = replace(lowercase(replace(String(p), '/' => '\\')),
                          r"^\\device\\harddiskvolume\d+" => "", r"^[a-z]:" => "")

"Code Integrity 3077 events since this run started: (time, path) pairs. Windows only."
function sac_events()
    Sys.iswindows() || return Tuple{String,String}[]
    ps = something(Sys.which("powershell"), Sys.which("powershell.exe"), "")
    isempty(ps) && return Tuple{String,String}[]
    since = Dates.format(RUN_START - Dates.Minute(2), "yyyy-mm-ddTHH:MM:SS")
    script = "Get-WinEvent -FilterHashtable @{LogName='Microsoft-Windows-CodeIntegrity/Operational'; " *
             "Id=3077; StartTime=[datetime]'$since'} -ErrorAction SilentlyContinue | ForEach-Object { " *
             "\$_.TimeCreated.ToString('s') + ' ' + (\$_.Message -replace '\\s+', ' ') }"
    text = try
        read(pipeline(`$ps -NoProfile -NonInteractive -Command $script`; stderr = devnull), String)
    catch
        ""
    end
    out = Tuple{String,String}[]
    for m in eachmatch(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d) .*?attempted to load (\S+) that", text)
        push!(out, (String(m.captures[1]), String(m.captures[2])))
    end
    return out
end

"""
    sac_block(text, dlls) -> "" | description

Whether `text` (a build log or a child's output) shows one of `dlls` refused by
Smart App Control, and the evidence for it.
"""
function sac_block(text::AbstractString, dlls)
    Sys.iswindows() || return ""
    bymsg = occursin(SAC_MSG, text)
    ev = sac_events()
    for d in dlls
        hit = findfirst(e -> _normpath_ci(e[2]) == _normpath_ci(d), ev)
        hit === nothing || return "blocked by Smart App Control: $(basename(d)) (Code Integrity event 3077 at $(ev[hit][1]))"
    end
    if bymsg
        m = match(r"could not load library \"([^\"]+)\"", text)
        return "blocked by Smart App Control: $(m === nothing ? "?" : basename(m.captures[1])) " *
               "(the loader's message; no 3077 event matched)"
    end
    return ""
end

# Every DLL a unit's load can involve: the library itself, and libJLCS for C++.
unit_dlls(dir, lang) = vcat([joinpath(dir, "julia", f) for f in (isdir(joinpath(dir, "julia")) ?
                             readdir(joinpath(dir, "julia")) : String[]) if endswith(f, ".dll")],
                            lang === :cpp ? [RepliBuild.MLIRNative.libJLCS] : String[])

readlog(dir) = (p = joinpath(dir, "build.log"); isfile(p) ? read(p, String) : "")

"Mark a unit's verdicts as blocked by SAC."
function sac_verdicts(checks, why)
    v = Dict(c => "BLOCKED: Smart App Control blocked the library" for c in checks)
    v["load"] = "SAC " * why
    return v
end

# ── Building ─────────────────────────────────────────────────────────────────

modechar(lang) = lang === :cpp ? "x" : "c"

"""
    build_unit(u, lang, dir) -> (plan, "") | (nothing, why)

Write the unit's sources into a fresh `dir` and run discover/build/wrap there,
in this process, with the build's output in `dir/build.log`.
"""
function build_unit(u::AF.Unit, lang::Symbol, dir::String; tag = "")
    rm(dir; force = true, recursive = true)
    mkpath(dir)
    (mf, tf) = AF.src_names(u, lang)
    write(joinpath(dir, mf), AF.emit_main(u))
    any(f -> f.tu == 2, u.funcs) && write(joinpath(dir, tf), AF.emit_twin(u))
    err = ""
    logio = open(joinpath(dir, "build.log"), "w")
    try
        redirect_stdio(stdout = logio, stderr = logio) do
            Logging.with_logger(Logging.ConsoleLogger(logio)) do
                RepliBuild.discover(dir; build = true, wrap = true)
            end
        end
    catch e
        err = sprint(showerror, e)
        println(logio, "\nabi_fuzz: build/wrap threw:\n", sprint(showerror, e, catch_backtrace()))
    finally
        close(logio)
    end
    if !isempty(err)
        why = valid_c(u, lang, dir)
        return (nothing, isempty(why) ? "build/wrap threw: " * first(err, 600) :
                                        "HARNESS: the generated unit is not valid C: " * why)
    end
    jdir = joinpath(dir, "julia")
    wrappers = filter(f -> endswith(f, ".jl"), readdir(jdir))
    libs = filter(f -> startswith(f, "lib") && any(endswith(f, e) for e in (".dll", ".so", ".dylib")) &&
                       !occursin("_thunks", f), readdir(jdir))
    (length(wrappers) == 1 && length(libs) == 1) ||
        return (nothing, "build/wrap produced $(length(wrappers)) wrappers and $(length(libs)) libraries in $jdir")
    wrapper = joinpath(jdir, only(wrappers))
    m = match(r"(?m)^module\s+(\S+)", read(wrapper, String))
    m === nothing && return (nothing, "no `module` line in $wrapper")
    meta = RepliBuild.JSON.parse(read(joinpath(jdir, "compilation_metadata.json"), String))
    mangled = Dict{String,String}()
    for f in get(meta, "functions", [])
        nm = String(get(f, "name", ""))
        isempty(nm) && continue
        mangled[nm] = String(get(f, "mangled", nm))
    end
    plan = AF.Plan(u, lang, tag, joinpath(jdir, only(libs)), wrapper, String(m.captures[1]), mangled,
                   OPT.neuter, AF.planned_checks(u), String[], String[])
    return (plan, "")
end

# Is the unit valid C/C++ according to the system compiler? Only asked when a
# build fails, to tell a generator bug from a RepliBuild one.
function valid_c(u, lang, dir)
    cc = Sys.which(lang === :cpp ? "clang++" : "clang")
    cc === nothing && return ""
    (mf, tf) = AF.src_names(u, lang)
    for f in (mf, tf)
        p = joinpath(dir, f)
        isfile(p) || continue
        out = IOBuffer()
        std = lang === :cpp ? "-std=c++17" : "-std=gnu17"
        ok = success(pipeline(ignorestatus(`$cc $std -fsyntax-only -w $p`); stdout = out, stderr = out))
        ok || return first(String(take!(out)), 600)
    end
    return ""
end

# ── Running the checks ───────────────────────────────────────────────────────

"""
    run_checks(plan, planfile) -> (verdicts, gaps, log)

Run the child until every planned check has a verdict. A crash or timeout is
blamed on the check the child announced last (`RUNNING`), which is then skipped
on the rerun. `log` keeps every child's output.
"""
function run_checks(plan::AF.Plan, planfile::String; timeout = OPT.timeout)
    verdicts = Dict{String,String}()
    gaps = Tuple{String,String}[]
    logs = String[]
    skip = copy(plan.skip)
    planned = isempty(plan.only) ? plan.checks : intersect(plan.checks, vcat(plan.only, ["load", "exit"]))
    for attempt in 1:(length(planned) + 2)
        p = AF.Plan(plan.unit, plan.lang, plan.tag, plan.libpath, plan.wrapper, plan.modname, plan.mangled,
                    plan.neuter, plan.checks, plan.only, skip)
        serialize(planfile, p)
        t = @elapsed r = run_isolated(`$CHILD $planfile`; timeout = timeout, project = PROJECT, echo = false)
        push!(logs, "── child attempt $attempt ($(round(t; digits = 1))s, exit=$(r.exitcode), " *
                    "signal=$(r.signal), timed out=$(r.timedout))\n" * r.output)
        last_running = ""
        done = false
        prefix = plan.tag * "/"
        for line in eachline(IOBuffer(r.output))
            if line == "PROBE_DONE"
                done = true
            elseif (m = match(r"^PROBE (\S+): (.*)$", line)) !== nothing
                nm = m.captures[1]
                startswith(nm, prefix) && (verdicts[nm[length(prefix)+1:end]] = m.captures[2])
            elseif (m = match(r"^GAP (\S+): (.*)$", line)) !== nothing
                push!(gaps, (m.captures[1][length(prefix)+1:end], m.captures[2]))
            elseif (m = match(r"^RUNNING (\S+)$", line)) !== nothing
                last_running = m.captures[1][length(prefix)+1:end]
            end
        end
        if done
            verdicts["exit"] = r.ok ? "PASS" :
                "FAIL child ended abnormally after its last check (exit=$(r.exitcode), signal=$(r.signal)): " *
                tail(r.output)
            break
        end
        what = r.timedout ? "timed out after $(timeout)s" : "crashed (exit=$(r.exitcode), signal=$(r.signal))"
        detail = "FAIL $what: " * tail(r.output)
        if last_running == "" || last_running == "load"
            verdicts["load"] = get(verdicts, "load", "") == "" || startswith(verdicts["load"], "PASS") ?
                               detail : verdicts["load"]
            for c in planned
                haskey(verdicts, c) || (verdicts[c] = "BLOCKED: the child died during load")
            end
            break
        end
        if startswith(last_running, "fn/")
            f = last_running[4:end]
            for c in ("call/" * f, "ret/" * f)
                c in planned && !haskey(verdicts, c) && (verdicts[c] = detail)
                push!(skip, c)
            end
        else
            haskey(verdicts, last_running) || (verdicts[last_running] = detail)
            push!(skip, last_running)
        end
    end
    loadfail = status(get(verdicts, "load", "")) == "FAIL"
    for c in planned
        haskey(verdicts, c) && continue
        verdicts[c] = loadfail ? "BLOCKED: the wrapper did not load" :
                                 "FAIL harness: the child never reported this check"
    end
    return (verdicts, unique!(gaps), join(logs, "\n"))
end

"The fault line and the frames under it when there is one, else the last lines."
function tail(out; n = 12)
    ls = [l for l in split(out, '\n') if !startswith(l, "PROBE ") && !startswith(l, "RUNNING ") &&
          !startswith(l, "GAP ") && !isempty(strip(l))]
    i = findfirst(l -> occursin(r"EXCEPTION_|signal \(\d+\)|Unhandled|fatal|Assertion|ERROR:", l), ls)
    pick = i === nothing ? ls[max(1, end - n + 1):end] : ls[i:min(end, i + 4)]
    return replace(join(pick, " | "), '\t' => ' ')
end

status(v) = String(rstrip(first(split(v, ' '; limit = 2)), ':'))
fclass(v) = status(v) != "FAIL" ? status(v) :
            occursin(r"crashed|timed out|ended abnormally", v) ? "crash" :
            startswith(v, "FAIL threw") ? "threw" :
            startswith(v, "FAIL wrapper did not load") ? "noload" :
            startswith(v, "FAIL build/wrap threw") ? "nobuild" : "mismatch"
# A wrong value and a crash are one family for the minimiser: dropping an
# argument can turn a crash into a wrong value without changing the bug.
family(c) = c in ("crash", "mismatch") ? "abi" : c

"""
What the minimiser holds fixed while it reduces. A wrong value or a crash is
one failure; a build or load failure is the same failure only with the same
message (module names, numbers and paths aside), or dropping every function
would "reproduce" a refused wrap with RepliBuild's empty-surface error.
"""
function failkey(v::AbstractString)
    c = fclass(v)
    c in ("nobuild", "noload", "threw") || return family(c)
    m = replace(String(v), r"^FAIL (build/wrap threw|wrapper did not load|threw): " => "")
    m = replace(m, r"[A-Za-z]:\\[^\s\"']+" => "<path>", r"lib[\w${}]+\.(?:dll|so|dylib)" => "<lib>",
                r"'[^']*'" => "'#'", r"\b[ST]\d+n?[cx]\w*" => "#", r"\d+" => "#")
    return c * ": " * first(strip(first(split(m, '\n'))), 90)
end

"""
A short, unit-independent description of a FAIL, for grouping: the exception
type for a throw, which property disagreed for a layout or access check.
"""
function signature(check::AbstractString, v::AbstractString)
    c = fclass(v)
    kind = first(split(check, '/'))
    c == "crash" && return occursin("timed out", v) ? "timeout" : "crash"
    if c == "threw"
        m = match(r"^FAIL threw: (\w+(?:: [^(|]{0,40})?)", v)
        return "threw " * (m === nothing ? "?" : strip(m.captures[1]))
    end
    c in ("noload", "nobuild") && return c
    if kind == "layout"
        return "layout: " * join(sort(unique(m.match for m in eachmatch(r"sizeof|struct_size|member_offset|fieldoffset", v))), "+")
    elseif kind == "access"
        parts = String[]
        occursin(r"get \S+ = ", v) && push!(parts, "getter value")
        occursin(r"get \S+: ", v) && push!(parts, "getter threw")
        occursin("set ", v) && push!(parts, "setter value")
        occursin("setter threw", v) && push!(parts, "setter threw")
        occursin("cannot view", v) && push!(parts, "size")
        return "access: " * join(parts, "+")
    elseif kind == "call"
        occursin("callee did not run", v) && return "call: wrong callee"
        return "call: recorded values differ" * (occursin("setter:", v) ? " (setter)" : "")
    elseif kind == "ret"
        occursin("returned a ", v) && return "ret: wrong type"
        occursin("bytes right: getter", v) && !occursin("bytes wrong", v) && return "ret: getter"
        return "ret: returned values differ"
    end
    return c
end

# ── Minimising ───────────────────────────────────────────────────────────────

"Records and enums no function or record still references are dropped, and the rest renumbered."
function compact(u::AF.Unit)
    u = deepcopy(u)
    used_r = Set{Int}(); used_e = Set{Int}()
    mark(t::AF.CT) = if t isa AF.RecT
        if !(t.r in used_r)
            push!(used_r, t.r)
            for m in u.records[t.r].members
                mark(m.ty)
            end
        end
    elseif t isa AF.EnumT
        push!(used_e, t.e)
    elseif t isa AF.ArrT
        mark(t.elem)
    end
    for f in u.funcs
        f.ret === nothing || mark(f.ret)
        foreach(p -> mark(p.ty), f.params)
    end
    for (i, e) in enumerate(u.enums)
        e.twin != 0 && any(f -> f.name in ("af_twin_a", "af_twin_b"), u.funcs) && push!(used_e, i)
    end
    rmap = Dict{Int,Int}(); emap = Dict{Int,Int}()
    recs = AF.Record[]; ens = AF.EnumDef[]
    for (i, r) in enumerate(u.records)
        i in used_r && (push!(recs, r); rmap[i] = length(recs))
    end
    for (i, e) in enumerate(u.enums)
        i in used_e && (push!(ens, e); emap[i] = length(ens))
    end
    remap(t::AF.CT) = t isa AF.RecT ? AF.RecT(rmap[t.r]) : t isa AF.EnumT ? AF.EnumT(emap[t.e]) :
                      t isa AF.ArrT ? AF.ArrT(remap(t.elem), t.n) : t
    for r in recs, m in r.members
        m.ty = remap(m.ty)
    end
    for f in u.funcs
        f.ret === nothing || (f.ret = remap(f.ret))
        for p in f.params
            p.ty = remap(p.ty)
        end
    end
    return AF.Unit(u.seed, u.ident, recs, ens, u.funcs)
end

# Content-based labels, so a candidate keeps its description when an earlier
# reduction shifts the indices.
rlabel(r::AF.Record) = isempty(r.name) ? "anonymous record {" * join((m.name for m in r.members), ",") * "}" : r.name
mlabel(r::AF.Record, mi) = (m = r.members[mi]; isempty(m.name) ?
    (m.bits >= 0 ? "unnamed bitfield :$(m.bits) (#$mi)" : "anonymous member (#$mi)") : m.name)

# Candidate reductions of `u` for a failure in `check`, smallest effect last.
function candidates(u::AF.Unit, check::String)
    out = Tuple{String,AF.Unit}[]
    kind, target = occursin('/', check) ? split(check, '/'; limit = 2) : (check, "")
    keepf(f) = (kind in ("call", "ret") && f.name == target)
    # every other function at once, then one at a time
    others = [i for (i, f) in enumerate(u.funcs) if !keepf(f)]
    if !isempty(others)
        v = deepcopy(u); v.funcs = [f for f in v.funcs if keepf(f)]; push!(out, ("keep only the failing function", v))
        for i in others
            v = deepcopy(u); deleteat!(v.funcs, i); push!(out, ("drop function $(u.funcs[i].name)", v))
        end
    end
    # parameters of the failing function
    for (fi, f) in enumerate(u.funcs), pi in length(f.params):-1:1
        v = deepcopy(u); deleteat!(v.funcs[fi].params, pi); push!(out, ("drop $(f.name) parameter $(f.params[pi].name)", v))
    end
    # a record return becomes a scalar, a scalar return goes
    for (fi, f) in enumerate(u.funcs)
        f.ret === nothing && continue
        kind == "ret" && f.name == target && continue
        v = deepcopy(u); v.funcs[fi].ret = nothing; push!(out, ("$(f.name) returns void", v))
    end
    # records: members, then attributes, then member types
    for (ri, r) in enumerate(u.records)
        kind in ("layout", "access") && r.name == target && length(r.members) <= 1 && continue
        for mi in length(r.members):-1:1
            length(r.members) <= 1 && break
            v = deepcopy(u)
            deleteat!(v.records[ri].members, mi)
            rr = v.records[ri]
            rr.isunion && (rr.active = rr.active == mi ? 1 : rr.active - (rr.active > mi))
            isempty(AF.leaves(v, ri)) && continue
            push!(out, ("drop member $(mlabel(r, mi)) of $(rlabel(r))", v))
        end
        if r.packed
            v = deepcopy(u); v.records[ri].packed = false; push!(out, ("$(r.name) not packed", v))
        end
        if r.pack > 0
            v = deepcopy(u); v.records[ri].pack = 0; push!(out, ("$(r.name) without #pragma pack", v))
        end
        if r.isunion && !r.anon
            v = deepcopy(u); v.records[ri].isunion = false; v.records[ri].active = 0; push!(out, ("$(r.name) a struct, not a union", v))
        end
        if !r.anon && (r.tagonly || !isempty(r.tag))
            v = deepcopy(u); v.records[ri].tagonly = false; v.records[ri].tag = ""
            v.records[ri].name = r.name; push!(out, ("$(r.name) an untagged typedef", v))
        end
        for (mi, m) in enumerate(r.members)
            simpler = AF.CT[]
            t = m.ty
            if t isa AF.ArrT
                t.n > 1 && push!(simpler, AF.ArrT(t.elem, 1))
                push!(simpler, t.elem)
            elseif t isa AF.RecT && !isempty(m.name)
                push!(simpler, AF.Prim(:int))
            elseif t isa AF.EnumT
                push!(simpler, AF.Prim(:int))
            elseif t isa AF.Prim && !(t.k in (:int, :char)) && m.bits < 0
                push!(simpler, AF.Prim(:int), AF.Prim(:char))
            end
            for s in simpler
                v = deepcopy(u); v.records[ri].members[mi].ty = s; push!(out, ("member $(mlabel(r, mi)) of $(rlabel(r)) becomes $s", v))
            end
            if m.bits > 0 && !(t isa AF.Prim && t.k == :bool)
                v = deepcopy(u); v.records[ri].members[mi].bits = -1; push!(out, ("member $(mlabel(r, mi)) of $(rlabel(r)) not a bitfield", v))
            end
        end
    end
    for (fi, f) in enumerate(u.funcs), (pi, p) in enumerate(f.params)
        p.ty isa AF.Prim && p.ty.k != :int && p.ty.k != :ldouble &&
            (v = deepcopy(u); v.funcs[fi].params[pi].ty = AF.Prim(:int); push!(out, ("$(f.name) parameter $(p.name) an int", v)))
        AF.isptrparam(p.ty) && (v = deepcopy(u); v.funcs[fi].params[pi].ty = (p.ty::AF.ArrT).elem; push!(out, ("$(f.name) parameter $(p.name) by value", v)))
    end
    return [(w, compact(v)) for (w, v) in out]
end

"""
    minimise(u, lang, check, verdict) -> (unit, verdict, builds)

Delta-debug `u` while `check` still fails the same way (`fclass`).
"""
function minimise(u::AF.Unit, lang::Symbol, check::String, verdict::String; budget = OPT.budget, label = "")
    want = failkey(verdict)
    builds = 0
    k = 0
    mkpath(joinpath(OPT.dir, "min"))
    mlog = open(joinpath(OPT.dir, "min", label * ".log"), "w")
    bestdir = ""
    function fails(v::AF.Unit, what::String)
        builds += 1
        k += 1
        dir = joinpath(OPT.dir, "min", string(label, "_", k))
        (plan, why) = build_unit(v, lang, dir; tag = "min")
        if plan === nothing
            # a blocked candidate is unknown: neither still failing nor passing
            sac = sac_block(readlog(dir), unit_dlls(dir, lang))
            if !isempty(sac)
                println(mlog, k, "\t", what, "\tunknown\tSAC ", sac); flush(mlog)
                return (false, "SAC " * sac)
            end
            ok = failkey("FAIL " * why) == want
            println(mlog, k, "\t", what, "\t", ok ? "KEEP" : "drop", "\tFAIL ", first(why, 300)); flush(mlog)
            ok || rm(dir; force = true, recursive = true)
            return (ok, "FAIL " * why)
        end
        plan = AF.Plan(plan.unit, plan.lang, plan.tag, plan.libpath, plan.wrapper, plan.modname, plan.mangled,
                       plan.neuter, plan.checks, [check], String[])
        (vs, _, clog) = run_checks(plan, joinpath(dir, "plan.jls"))
        if status(get(vs, "load", "")) != "PASS"
            sac = sac_block(clog, unit_dlls(dir, lang))
            if !isempty(sac)
                println(mlog, k, "\t", what, "\tunknown\tSAC ", sac); flush(mlog)
                return (false, "SAC " * sac)
            end
        end
        got = get(vs, check, "")
        ok = status(got) == "FAIL" && failkey(got) == want
        println(mlog, k, "\t", what, "\t", ok ? "KEEP" : "drop", "\t", first(got, 300))
        occursin("FAIL harness", got) && println(mlog, "    child output:\n", clog)
        flush(mlog)
        if ok
            isempty(bestdir) || rm(bestdir; force = true, recursive = true)
            bestdir = dir
        else
            rm(dir; force = true, recursive = true)
        end
        return (ok, got)
    end
    best = compact(u)
    bestv = verdict
    # A candidate that did not reproduce is not retried after later reductions
    # (descriptions are content-based, so they survive index shifts); one full
    # pass at the end catches reductions that only an earlier one enabled.
    dropped = Set{String}()
    final_pass = false
    while builds < budget
        progress = false
        for (what, c) in candidates(best, check)
            builds >= budget && break
            what in dropped && continue
            # the check's subject must survive the reduction
            kind, target = occursin('/', check) ? split(check, '/'; limit = 2) : (check, "")
            kind in ("call", "ret") && !any(f -> f.name == target, c.funcs) && continue
            kind == "call" && all(f -> f.name != target || isempty(f.params), c.funcs) && continue
            kind in ("layout", "access") && !any(r -> !r.anon && r.name == target, c.records) && continue
            (ok, got) = fails(c, what)
            if ok
                best = c; bestv = got
                progress = true
                @info "abi_fuzz minimise $label: reduced (build $builds): $(length(best.funcs)) functions, " *
                      "$(count(r -> !r.anon, best.records)) records, " *
                      "$(sum(length(r.members) for r in best.records; init = 0)) members"
                break
            end
            push!(dropped, what)
        end
        progress && continue
        (final_pass || isempty(dropped)) && break
        final_pass = true
        empty!(dropped)
    end
    close(mlog)
    return (best, bestv, builds)
end

function save_repro(u::AF.Unit, lang::Symbol, check::String, original::String, verdict::String, builds::Int)
    case = string(AF.unit_tag(u), "-", lang, "-", replace(check, r"[^A-Za-z0-9_]" => "_"))
    dir = joinpath(OPT.dir, "repros", case)
    rm(dir; force = true, recursive = true)
    mkpath(dir)
    (mf, tf) = AF.src_names(u, lang)
    write(joinpath(dir, mf), AF.emit_main(u))
    any(f -> f.tu == 2, u.funcs) && write(joinpath(dir, tf), AF.emit_twin(u))
    serialize(joinpath(dir, "unit.jls"), (u, lang, check))
    write(joinpath(dir, "README.md"), """
    # abi_fuzz repro: seed $(u.seed)$(u.ident ? " (identifier mode)" : ""), $(lang), check `$(check)`

    Generated $(now()) by `test/tools/abi_fuzz.jl` on $(Sys.MACHINE), Julia $(VERSION),
    RepliBuild $(pkgversion(RepliBuild))$(isempty(OPT.neuter) ? "" : ", NEUTERED: $(OPT.neuter)").

    Original verdict:

        $(original)

    Minimised verdict ($(builds) builds):

        $(verdict)

    Source: `$(mf)`$(isfile(joinpath(dir, tf)) ? " and `$(tf)`" : ""). Recheck with

        julia --project=. test/tools/abi_fuzz.jl --repro=$(dir)
    """)
    return dir
end

# ── Driver ───────────────────────────────────────────────────────────────────

struct Job
    seed::Int
    lang::Symbol
    tag::String
    dir::String
    unit::AF.Unit
end

function libjlcs_loads()
    p = RepliBuild.MLIRNative.libJLCS
    isfile(p) || return "libJLCS not built ($p)"
    h = Libdl.dlopen(p; throw_error = false)
    h === nothing && return "libJLCS does not load ($p)"
    return ""
end

"""
    run_target(target)

`--target=SEED/MODE/CHECK`: regenerate that unit, check that the named check
still fails, minimise it and save the repro. Several of these can run at once,
one process each, which is how a long list of failures gets minimised in
reasonable time.
"""
function run_target(target::AbstractString)
    m = match(r"^(\d+)(n?)/(c|cpp)/(.+)$", target)
    m === nothing && error("--target wants SEED/MODE/CHECK, e.g. 3/c/call/f2 or 7n/cpp/load; got `$target`")
    seed = parse(Int, m.captures[1]); ident = m.captures[2] == "n"
    lang = Symbol(m.captures[3]); check = String(m.captures[4])
    u = AF.generate(seed; ident = ident)
    check in AF.planned_checks(u) || error("unit $(AF.unit_tag(u)) plans no check `$check`")
    tag = string(AF.unit_tag(u), "/", lang)
    label = string("t", AF.unit_tag(u), modechar(lang), "_", replace(check, r"[^A-Za-z0-9_]" => "_"))
    dir = joinpath(OPT.dir, label)
    (plan, why) = build_unit(u, lang, dir; tag = tag)
    v = if plan === nothing
        sac = sac_block(readlog(dir), unit_dlls(dir, lang))
        isempty(sac) ? "FAIL " * why : "SAC " * sac
    else
        plan = AF.Plan(plan.unit, plan.lang, plan.tag, plan.libpath, plan.wrapper, plan.modname, plan.mangled,
                       plan.neuter, plan.checks, [check], String[])
        (vs, _, clog) = run_checks(plan, joinpath(dir, "plan.jls"))
        sac = status(get(vs, "load", "")) == "PASS" ? "" : sac_block(clog, unit_dlls(dir, lang))
        isempty(sac) ? get(vs, check, "") : "SAC " * sac
    end
    println("abi_fuzz target $target: ", first(v, 400)); flush(stdout)
    if status(v) == "SAC"
        println("  blocked by Smart App Control: not minimised")
        return 0
    end
    if status(v) != "FAIL"
        println("  does not fail: nothing to minimise")
        return 0
    end
    t = @elapsed (u2, v2, nb) = minimise(u, lang, check, v; label = label)
    d = save_repro(u2, lang, check, v, v2, nb)
    println(@sprintf("  minimised in %d builds, %.0fs -> %s", nb, t, d))
    println("  minimised verdict: ", first(v2, 400))
    OPT.keep || rm(dir; force = true, recursive = true)
    return 1
end

function main()
    isempty(OPT.target) || exit(run_target(OPT.target))
    runid = Dates.format(now(), "yyyymmdd-HHMMSS")
    modes = copy(OPT.modes)
    notes = String[]
    if :cpp in modes
        why = libjlcs_loads()
        if !isempty(why)
            push!(notes, "cpp mode skipped: $why")
            filter!(!=(:cpp), modes)
        end
    end
    jobs = Job[]
    if !isempty(OPT.repro)
        (u, lang, check) = deserialize(joinpath(OPT.repro, "unit.jls"))
        push!(jobs, Job(u.seed, lang, string(AF.unit_tag(u), "/", lang), joinpath(OPT.dir, "r" * AF.unit_tag(u) * modechar(lang)), u))
    else
        for seed in OPT.seeds
            u = AF.generate(seed; ident = OPT.ident)
            for lang in modes
                push!(jobs, Job(seed, lang, string(AF.unit_tag(u), "/", lang),
                                joinpath(OPT.dir, string("s", AF.unit_tag(u), modechar(lang))), u))
            end
        end
    end
    println("abi_fuzz run $runid: $(length(jobs)) units ($(join(string.(modes), ", "))) on Julia $(VERSION), ",
            "RepliBuild $(pkgversion(RepliBuild)), $(Sys.MACHINE)",
            isempty(OPT.neuter) ? "" : ", NEUTER $(OPT.neuter)")
    foreach(n -> println("  note: ", n), notes)

    # Phase 1: build and wrap, in this process
    plans = Dict{String,AF.Plan}()
    results = Dict{String,Dict{String,String}}()
    gaps = Dict{String,Vector{Tuple{String,String}}}()
    for (i, j) in enumerate(jobs)
        t = @elapsed (plan, why) = build_unit(j.unit, j.lang, j.dir; tag = j.tag)
        if plan === nothing
            sac = sac_block(readlog(j.dir), unit_dlls(j.dir, j.lang))
            if isempty(sac)
                v = Dict(c => "BLOCKED: build/wrap failed" for c in AF.planned_checks(j.unit))
                v["load"] = "FAIL " * why
            else
                v = sac_verdicts(AF.planned_checks(j.unit), sac)
                why = "SAC " * sac
            end
            results[j.tag] = v
            gaps[j.tag] = Tuple{String,String}[]
            println(@sprintf("  [%3d/%d] %-10s build FAILED (%.1fs): %s", i, length(jobs), j.tag, t, first(why, 200))); flush(stdout)
        else
            plans[j.tag] = plan
            println(@sprintf("  [%3d/%d] %-10s built (%.1fs)", i, length(jobs), j.tag, t)); flush(stdout)
        end
    end

    # Phase 2: check, in children
    lk = ReentrantLock()
    asyncmap(collect(values(plans)); ntasks = OPT.jobs) do plan
        dir = dirname(dirname(plan.wrapper))
        t = @elapsed (v, g, log) = run_checks(plan, joinpath(dir, "plan.jls"))
        write(joinpath(dir, "child.log"), log)
        if status(get(v, "load", "")) != "PASS"
            sac = sac_block(log, unit_dlls(dir, plan.lang))
            isempty(sac) || (v = sac_verdicts(plan.checks, sac); g = Tuple{String,String}[])
        end
        lock(lk) do
            results[plan.tag] = v
            gaps[plan.tag] = g
            nfail = count(x -> status(x) == "FAIL", values(v))
            println(@sprintf("  checked %-10s %3d checks, %d FAIL (%.1fs)", plan.tag, length(v), nfail, t)); flush(stdout)
        end
    end

    # Completeness: every planned check has exactly one verdict
    planned = sum(length(AF.planned_checks(j.unit)) for j in jobs)
    got = sum(length(results[j.tag]) for j in jobs)
    missing_ = [(j.tag, c) for j in jobs for c in AF.planned_checks(j.unit) if !haskey(results[j.tag], c)]
    extra = [(j.tag, c) for j in jobs for c in keys(results[j.tag]) if !(c in AF.planned_checks(j.unit))]

    # Report
    rows = String[]
    counts = Dict{Tuple{Symbol,String},Int}()
    for j in jobs, c in AF.planned_checks(j.unit)
        v = get(results[j.tag], c, "MISSING")
        push!(rows, join([j.tag, c, status(v), replace(v, '\t' => ' ')], '\t'))
        counts[(j.lang, status(v))] = get(counts, (j.lang, status(v)), 0) + 1
    end
    resfile = joinpath(OPT.dir, "results-$runid.tsv")
    write(resfile, "unit\tcheck\tstatus\tverdict\n" * join(rows, '\n') * "\n")
    gaprows = [join([tag, k, d], '\t') for (tag, g) in gaps for (k, d) in g]
    gapfile = joinpath(OPT.dir, "gaps-$runid.tsv")
    write(gapfile, "unit\tkind\tdetail\n" * join(sort(gaprows), '\n') * "\n")

    println("\n══ abi_fuzz summary ($runid) ══")
    for lang in unique(j.lang for j in jobs)
        n = sum(get(counts, (lang, s), 0) for s in ("PASS", "FAIL", "GAP", "XFAIL", "BLOCKED", "SAC", "MISSING"))
        nsac = count(j -> j.lang == lang && status(get(results[j.tag], "load", "")) == "SAC", jobs)
        println(@sprintf("  %-4s %5d checks: %d PASS, %d FAIL, %d GAP, %d XFAIL, %d BLOCKED; %d of %d units blocked by Smart App Control",
                         lang, n, get(counts, (lang, "PASS"), 0), get(counts, (lang, "FAIL"), 0),
                         get(counts, (lang, "GAP"), 0), get(counts, (lang, "XFAIL"), 0),
                         get(counts, (lang, "BLOCKED"), 0), nsac, count(j -> j.lang == lang, jobs)))
    end
    println("  checks planned $planned, verdicts $got",
            isempty(missing_) && isempty(extra) ? " ✓" : "  ✗ missing $(length(missing_)), unplanned $(length(extra))")
    fails = [(j, c, results[j.tag][c]) for j in jobs for c in AF.planned_checks(j.unit)
             if status(get(results[j.tag], c, "")) == "FAIL"]
    groups = Dict{Tuple{Symbol,String},Vector{Tuple{Job,String,String}}}()
    for (j, c, v) in fails
        push!(get!(groups, (j.lang, signature(c, v)), Tuple{Job,String,String}[]), (j, c, v))
    end
    if !isempty(fails)
        println("\n  FAIL ($(length(fails)) in $(length(groups)) groups; every one is in the results file):")
        for ((lang, sig), xs) in sort(collect(groups); by = x -> (string(x[1][1]), -length(x[2])))
            units = unique(j.tag for (j, _, _) in xs)
            println("    [$lang] $sig — $(length(xs)) checks in $(length(units)) units ($(join(first(units, 6), ", ")))")
            (j, c, v) = xs[1]
            println("        e.g. ", j.tag, " ", c, ": ", first(v, 260))
        end
    end
    # gaps, grouped on their kind and the shape of their detail
    gsum = Dict{String,Vector{String}}()
    for (tag, g) in gaps, (k, d) in g
        key = string(occursin('/', tag) ? split(tag, '/')[end] : tag, " ", k, ": ",
                     replace(d, r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z0-9_.\[\]]+" => "<member>",
                                r"0x[0-9a-f]+" => "<hex>", r"\b[a-zA-Z]+[0-9]+(_[0-9]+)*\b" => "<name>"))
        push!(get!(gsum, first(key, 220), String[]), tag)
    end
    if !isempty(gsum)
        println("\n  GAPS (kind: pattern — count, first units):")
        for (key, tags) in sort(collect(gsum); by = x -> -length(x[2]))
            println("    ", key, " — ", length(tags), " (", join(first(unique(tags), 4), ", "), ")")
        end
    end
    println("\n  results: $resfile\n  gaps:    $gapfile")

    # Phase 3: minimise distinct failures
    if OPT.minimise > 0 && !isempty(fails)
        # one representative per group, smallest unit first
        todo = [first(sort(xs; by = x -> length(AF.planned_checks(x[1].unit)))) for (_, xs) in
                sort(collect(groups); by = x -> -length(x[2]))]
        println("\n  minimising $(min(length(todo), OPT.minimise)) of $(length(todo)) failure groups:")
        for (j, c, v) in first(todo, OPT.minimise)
            label = string("m", AF.unit_tag(j.unit), modechar(j.lang), "_", replace(c, r"[^A-Za-z0-9_]" => "_"))
            t = @elapsed (u2, v2, nb) = minimise(j.unit, j.lang, c, v; label = label)
            d = save_repro(u2, j.lang, c, v, v2, nb)
            println(@sprintf("    %s %s: %d builds, %.0fs -> %s", j.tag, c, nb, t, d)); flush(stdout)
        end
    end

    # tidy: passing units' build trees
    if !OPT.keep
        for j in jobs
            all(v -> status(v) in ("PASS", "XFAIL", "GAP"), values(results[j.tag])) &&
                rm(j.dir; force = true, recursive = true)
        end
    end
    !isempty(missing_) || !isempty(extra) ? exit(2) : (isempty(fails) ? exit(0) : exit(1))
end

_AS_SCRIPT && main()
