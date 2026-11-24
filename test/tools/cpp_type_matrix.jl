#!/usr/bin/env julia
# =============================================================================
# cpp_type_matrix — which C++ standard-library types a wrapper can make, pass
# and read back
# =============================================================================
#
# A HAND-RUN DEVELOPER TOOL, NOT A TEST. It reports where the C++ front end
# stands, refusals included, and is wired into no suite. It reads no Hub.
#
#     julia --project=. test/tools/cpp_type_matrix.jl [options]
#
# `test/tools/cpp_type_matrix/<family>/` holds one small library per type
# family (`lib.cpp`: free functions that make, take and return the type), the
# `[types] templates` it needs (`family.toml`) and a probe (`probe.jl`). The
# tool:
#
#   1. stages each family as a project in a scratch folder, laid out like a Hub
#      (`packages/tm_<family>/`), with a hand-written recipe;
#   2. builds and wraps them with `hub_sweep.jl` (a child per family, logs, a
#      time limit);
#   3. runs `cpp_thunk_coverage.jl` over that folder: what each wrapper binds,
#      refuses or drops, read off its parsed code;
#   4. runs each `probe.jl` in a child. A probe calls the wrapper from Julia
#      the way a user would and prints one line per case:
#
#          PROBE <type> | <cell> : <VERDICT> <detail>
#
#      Cells: `make` (a value of the type made from Julia, with no function
#      of the library making it), `pass by value`, `pass by reference`,
#      `return by value`, `return by reference`; anything else a probe names
#      (`thrown`, `ended by Julia`) is listed under "other".
#      Verdicts:
#        WORKS      called from Julia with what a user has, and the answer is right
#        OPAQUE     it crosses and the library reads it right, and the wrapper
#                   gives Julia no way to read or build what is inside: raw
#                   bytes or a pointer, a handle
#        NOT TAKEN  a plain Julia value of the natural type (a Vector for a
#                   std::vector) is a MethodError there: the wrapper's own type
#                   is needed
#        REFUSED    the wrapper refuses the call and says why (an ABI Safety Trap)
#        NO WAY     the function is bound, and nothing a wrapper gives can be
#                   passed to it, or what it returns cannot be used at all
#        WRONG      it ran and the answer is not the library's
#        ERROR      it threw something that is not a stated refusal
#        ABSENT     the wrapper has no such function
#        CRASH      the process died in the call, or never came back (the tool
#                   fills this in)
#   5. writes the matrix, `<out>/matrix.md`.
#
# Options:
#   --out DIR       where the scratch Hub, logs and the matrix go
#                   (default `.claude/cpp-type-matrix-<date>` in this checkout)
#   --only a,b      these families only
#   --jobs N        builds and probes at a time (default 6)
#   --probes-only   skip the build: probe what `--out` already holds
#   --flags "..."   compile flags in place of the default
#                   (`-std=c++20 -O1 -fPIC -fstandalone-debug`)
#
# The verdicts are the probe author's reading of each wrapper, so a probe is
# evidence only as far as it tries: read `probe.jl` before believing a NO WAY.
# =============================================================================

using RepliBuild
using Dates
using TOML

const TM_DIR = joinpath(@__DIR__, "cpp_type_matrix")
const TM_REPO = normpath(joinpath(@__DIR__, "..", ".."))
const TM_CELLS = ["make", "pass by value", "pass by reference", "return by value", "return by reference"]
const TM_VERDICTS = ["WORKS", "OPAQUE", "NOT TAKEN", "REFUSED", "NO WAY", "WRONG", "ERROR", "ABSENT", "CRASH"]
# What each word means, for the file a reader opens without this one.
const TM_LEGEND = [
    "WORKS" => "called from Julia with what a wrapper's user has, and the answer is the library's",
    "OPAQUE" => "it crosses and the library reads it right, and the wrapper gives Julia no way to read or build what is inside: raw bytes or a pointer",
    "NOT TAKEN" => "a plain Julia value of the natural type is a `MethodError` there; the wrapper's own type is needed",
    "REFUSED" => "the wrapper refuses the call and says why",
    "NO WAY" => "the function is bound, and nothing a wrapper gives can be passed to it, or what it returns cannot be used",
    "WRONG" => "it ran and the answer is not the library's",
    "ERROR" => "it threw something that is not a stated refusal",
    "ABSENT" => "the wrapper has no such function",
    "CRASH" => "the process died in the call, or never came back",
]
const TM_DEFAULT_FLAGS = ["-std=c++20", "-O1", "-fPIC", "-fstandalone-debug"]

isdefined(@__MODULE__, :sweep_main) || include(joinpath(@__DIR__, "hub_sweep.jl"))

tm_families() = sort!(String[d for d in readdir(TM_DIR) if isfile(joinpath(TM_DIR, d, "lib.cpp"))])

"Write family `name` as a project under `<hub>/packages/tm_<name>/`."
function tm_stage(hub::AbstractString, name::AbstractString, flags::Vector{String})
    src = joinpath(TM_DIR, name)
    dst = joinpath(hub, "packages", "tm_" * name)
    mkpath(dst)
    for f in readdir(src)
        (f == "probe.jl" || endswith(f, ".cpp") || endswith(f, ".hpp") || endswith(f, ".h")) || continue
        # Only when it differs: a rewritten source moves the project's hash.
        (isfile(joinpath(dst, f)) && read(joinpath(dst, f)) == read(joinpath(src, f))) && continue
        cp(joinpath(src, f), joinpath(dst, f); force = true)
    end
    cp(joinpath(TM_DIR, "probe_support.jl"), joinpath(dst, "probe_support.jl"); force = true)
    fam = isfile(joinpath(src, "family.toml")) ? TOML.parsefile(joinpath(src, "family.toml")) : Dict{String,Any}()
    recipe = Dict{String,Any}(
        "project" => Dict("name" => "tm_" * name, "root" => "."),
        "compile" => Dict("flags" => flags, "source_files" => ["lib.cpp"], "include_dirs" => ["."]),
        "link" => Dict("enable_lto" => false, "optimization_level" => "0"),
        "binary" => Dict("type" => "shared"),
        "wrap" => Dict("language" => "cpp"),
        "types" => Dict{String,Any}("strictness" => "warn", "allow_unknown_structs" => true),
        "cache" => Dict("enabled" => false),
    )
    templates = String.(get(fam, "templates", String[]))
    if !isempty(templates)
        recipe["types"]["templates"] = templates
        recipe["types"]["template_headers"] = String.(get(fam, "template_headers", String[]))
    end
    # Sorted, and written only when it differs: the same recipe is the same file.
    text = sprint(io -> TOML.print(io, recipe; sorted = true))
    toml = joinpath(dst, "replibuild.toml")
    (isfile(toml) && read(toml, String) == text) || write(toml, text)
    return dst
end

"`type | cell => (verdict, detail)` from a probe's output, in the order printed."
function tm_parse_probe(text::AbstractString)
    out = Pair{Tuple{String,String},Tuple{String,String}}[]
    for l in eachline(IOBuffer(text))
        m = match(r"^PROBE (.+?) \| (.+?) : (WORKS|OPAQUE|NOT TAKEN|REFUSED|NO WAY|WRONG|ERROR|ABSENT|CRASH)\s*(.*)$", l)
        m === nothing && continue
        push!(out, (String(m.captures[1]), String(m.captures[2])) => (String(m.captures[3]), String(m.captures[4])))
    end
    return out
end

"""
Run one family's probe to its end. A probe that dies mid-case is run again from
the case after the one that killed it, which is recorded as CRASH.
"""
function tm_run_probe(pkgdir::AbstractString, log::AbstractString, limit::Real)
    results = Pair{Tuple{String,String},Tuple{String,String}}[]
    skip = 0
    text = ""
    for attempt in 1:40
        (status, _) = sweep_run(`$(sweep_julia()) probe.jl $skip`, pkgdir, log * ".$attempt", limit)
        text = read(log * ".$attempt", String)
        got = tm_parse_probe(text)
        append!(results, got)
        occursin("PROBE_DONE", text) && break
        # It died: the case it had announced and not finished is the crash.
        started = [m.captures[1] for m in eachmatch(r"^PROBE_START (.+)$"m, text)]
        isempty(started) && (push!(results, ("(probe)", "(load)") => ("ERROR", "the probe did not start: " * first(strip(text), 200))); break)
        last_case = String(started[end])
        parts = split(last_case, " | ")
        push!(results, (String(parts[1]), String(get(parts, 2, "?"))) =>
              ("CRASH", status > 128 ? "signal $(status - 128)" : status == 124 ? "ran out of time" : "exit $status"))
        skip += length(started)
    end
    return results
end

function tm_matrix(all_results::Dict{String,Any}, build_failed::Dict{String,String}, flags, coverage::AbstractString)::String
    io = IOBuffer()
    println(io, "# C++ standard-library types: what a wrapper can make, pass and read back")
    println(io)
    println(io, "Measured ", Dates.format(Dates.now(), "yyyy-mm-dd"), " with RepliBuild ", RepliBuild.VERSION,
            " on Julia ", VERSION, " by `test/tools/cpp_type_matrix.jl`, on ", Sys.MACHINE,
            ". Probe libraries compiled with `", join(flags, ' '), "`; thunks compiled when the wrapper loads.")
    println(io)
    counts = Dict{String,Int}()
    rows = String[]
    for fam in sort!(collect(keys(all_results)))
        res = all_results[fam]
        for (_, (v, _)) in res
            counts[v] = get(counts, v, 0) + 1
        end
        for t in unique!(String[k[1] for (k, _) in res])
            mine = [(ce, v) for ((ty, ce), (v, _)) in res if ty == t]
            cells = String[join(unique!(String[v for (ce, v) in mine if ce == c]), ", ") for c in TM_CELLS]
            other = unique!(String[ce * ": " * v for (ce, v) in mine if !(ce in TM_CELLS)])
            push!(rows, "| `" * t * "` | " * join(cells, " | ") * " | " * join(other, "; ") * " |")
        end
    end
    println(io, sum(values(counts); init = 0), " cases: ",
            join(("$(counts[v]) $(lowercase(v))" for v in TM_VERDICTS if haskey(counts, v)), ", "), ".")
    println(io)
    println(io, "| Type | ", join(TM_CELLS, " | "), " | other |")
    println(io, "|:--|", repeat(":--|", length(TM_CELLS) + 1))
    foreach(r -> println(io, r), rows)
    println(io)
    println(io, "`make` is a value made from Julia with no function of the library making it. A cell with two words had two cases; every case is listed below with what was tried.")
    println(io)
    for (v, meaning) in TM_LEGEND
        println(io, "- **", v, "**: ", meaning, ".")
    end
    if !isempty(build_failed)
        println(io, "\n## Families that did not build or wrap\n")
        for f in sort!(collect(keys(build_failed)))
            println(io, "- **", f, "**: ", build_failed[f])
        end
    end
    println(io, "\n## Every case\n")
    for fam in sort!(collect(keys(all_results)))
        println(io, "### ", fam, "\n")
        for ((t, c), (v, d)) in all_results[fam]
            println(io, "- `", t, "`, ", c, ": **", v, "**", isempty(d) ? "" : " — " * d)
        end
        println(io)
    end
    if !isempty(strip(coverage))
        println(io, "## What the coverage tool says of the same wrappers\n")
        println(io, "`cpp_thunk_coverage.jl` counts a function as bound when the wrapper's code invokes its thunk. Whether a user has anything to pass it is the table above.\n")
        println(io, "```")
        print(io, coverage)
        println(io, "```")
    end
    return String(take!(io))
end

function tm_main(args::Vector{String})::Int
    out = joinpath(TM_REPO, ".claude", "cpp-type-matrix-" * string(Dates.today()))
    only = String[]; jobs = 6; probes_only = false; flags = copy(TM_DEFAULT_FLAGS)
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--out";            i += 1; out = abspath(args[i])
        elseif a == "--only";       i += 1; append!(only, String.(strip.(split(args[i], ','))))
        elseif a == "--jobs";       i += 1; jobs = parse(Int, args[i])
        elseif a == "--flags";      i += 1; flags = String.(split(args[i]))
        elseif a == "--probes-only"; probes_only = true
        elseif a in ("-h", "--help")
            println(join(Iterators.takewhile(l -> startswith(l, "#"), Iterators.drop(eachline(@__FILE__), 1)), "\n")); return 0
        else
            error("unknown option `$a`")
        end
        i += 1
    end
    fams = tm_families()
    for n in only
        n in fams || error("`$n` is not a family of $TM_DIR")
    end
    isempty(only) || (fams = filter(in(only), fams))
    hub = joinpath(out, "hub")
    mkpath(joinpath(out, "logs"))
    foreach(f -> tm_stage(hub, f, flags), fams)
    # The coverage tool reads a Hub by its index.
    staged = sort!(String[d for d in readdir(joinpath(hub, "packages")) if startswith(d, "tm_")])
    open(joinpath(hub, "index.toml"), "w") do io
        TOML.print(io, Dict(p => Dict("language" => "cpp") for p in staged); sorted = true)
    end

    build_failed = Dict{String,String}()
    if !probes_only
        haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = joinpath(out, "home"))
        sweep_out = joinpath(out, "build")
        ispath(joinpath(sweep_out, "status.tsv")) && rm(joinpath(sweep_out, "status.tsv"))
        redirect_stdout(devnull) do
            sweep_main(vcat([hub, "--out", sweep_out, "--jobs", string(jobs), "--timeout", "1200", "--only",
                             join(("tm_" * f for f in fams), ",")]))
        end
        for l in eachline(joinpath(sweep_out, "status.tsv"))
            c = split(l, '\t')
            (length(c) >= 5 && c[2] == "verdict" && c[3] != "0") || continue
            build_failed[replace(String(c[1]), r"^tm_" => "")] = String(c[5])
        end
        println("built ", length(fams) - length(build_failed), " of ", length(fams), " families")
    end

    coverage = try
        read(pipeline(`$(sweep_julia()) $(joinpath(@__DIR__, "cpp_thunk_coverage.jl")) $hub --all --list`; stderr = devnull), String)
    catch e
        "the coverage tool did not run: " * first(sprint(showerror, e), 200) * "\n"
    end
    write(joinpath(out, "coverage.txt"), coverage)

    all_results = Dict{String,Any}()
    probed = String[f for f in fams if isfile(joinpath(hub, "packages", "tm_" * f, "probe.jl")) &&
                                      !isempty(sweep_wrappers(joinpath(hub, "packages", "tm_" * f, "julia")))]
    asyncmap(probed; ntasks = jobs) do f
        all_results[f] = tm_run_probe(joinpath(hub, "packages", "tm_" * f), joinpath(out, "logs", "probe.$f.log"), 240)
        println("probed ", f, ": ", length(all_results[f]), " case(s)")
    end
    text = tm_matrix(all_results, build_failed, flags, coverage)
    write(joinpath(out, "matrix.md"), text)
    println("\n", first(split(text, "\n## Every case")), "\nmatrix: ", joinpath(out, "matrix.md"))
    return 0
end

const _TM_AS_SCRIPT = abspath(PROGRAM_FILE) == abspath(@__FILE__)
_TM_AS_SCRIPT && exit(tm_main(String.(ARGS)))
