#!/usr/bin/env julia
# =============================================================================
# Regenerate test/fixtures/receiver_gate_corpus.json from the Hub
# =============================================================================
#
# THIS IS A DEVELOPER TOOL, NOT A TEST. It is wired into no suite, and it is the
# ONLY file under test/ permitted to read RepliBuild-Hub. Run it by hand after a
# Hub C++ rebuild that changes class/name metadata, then commit the fixture:
#
#     julia --project=. test/gen_receiver_corpus.jl
#
# WHY A FIXTURE INSTEAD OF READING THE HUB DIRECTLY. `test_symbol_hygiene.jl`
# §"Both receiver gates agree" used to sweep `~/Desktop/Projects/RepliBuild-Hub`
# live. That made a CI test — one that requires no toolchain — depend on a
# working tree in another repo. It was `isdir`-guarded, so it did not fail
# elsewhere; it went **vacuously green**, which is worse. The sweep is the
# assertion that would have caught the ImGui incident, the adjustor thunks and
# the box2d destructors, and it only ever ran on one machine.
#
# WHY THE REDUCTION IS LOSSLESS. Both gates are pure functions of three inputs:
#
#   FunctionGen._has_receiver(func, structs)
#       reads func["class"], func["name"], and reaches `structs` only through
#       `_fuzzy_struct_lookup`, which calls `haskey` and `keys` — never a value.
#   Wrapper._cpp_this_param(class, name, struct_types)
#       reads the two strings and tests `in` against the name set.
#
# So the whole 8.5 MB corpus reduces to: per package, the set of struct NAMES
# plus the distinct (class, name) pairs. `mangled` is carried for one
# representative of each pair so a failure message can name a real symbol.
#
# Duplicate (class, name) rows are dropped because the verdict is a pure
# function of the pair — a repeat cannot disagree when the first did not. That
# claim is not taken on faith: this script runs BOTH gates over the full Hub
# sweep and over the reduced corpus and refuses to write unless every verdict
# matches.

using RepliBuild
import JSON
import Dates

const FG   = RepliBuild.JLCSIRGenerator.FunctionGen
const W    = RepliBuild.Wrapper
const HUB  = joinpath(homedir(), "Desktop", "Projects", "RepliBuild-Hub", "packages")
const OUT  = joinpath(@__DIR__, "fixtures", "receiver_gate_corpus.json")

# Both gates, against one (class, name) and one aggregate-name set.
function verdicts(cls::String, nm::String, structs, stypes)
    fg  = FG._has_receiver(Dict("class" => cls, "name" => nm), structs)
    cpp = W._cpp_this_param(cls, nm, stypes) !== nothing
    return (fg, cpp)
end

function main()
    isdir(HUB) || error("Hub not found at $HUB — this tool needs it; the test does not.")

    packages = Any[]
    full_sweep = 0        # every method row in the Hub
    reduced_rows = 0      # distinct (class, name) pairs kept

    for pkg in sort(readdir(HUB))
        mf = joinpath(HUB, pkg, "julia", "compilation_metadata.json")
        isfile(mf) || continue
        # No try/catch: a corpus that fails to load must be loud. The live
        # version of this sweep was once silently skipped by a bare catch
        # swallowing `UndefVarError: JSON`, and proved nothing while passing.
        meta = JSON.parsefile(mf)
        get(meta, "language", "") in ("c++", "cpp", "cxx") || continue

        structs = get(meta, "struct_definitions", Dict())
        stypes  = Set{String}(String(k) for k in keys(structs))

        # class => name => representative mangled
        seen = Dict{Tuple{String,String},String}()
        for f in get(meta, "functions", [])
            get(f, "is_method", false) || continue
            cls = String(get(f, "class", ""))
            isempty(cls) && continue
            nm  = String(get(f, "name", ""))
            full_sweep += 1
            get!(seen, (cls, nm), String(get(f, "mangled", "?")))
        end
        isempty(seen) && continue

        names = sort!(collect(stypes))
        rows  = sort!([[k[1], k[2], v] for (k, v) in seen])
        reduced_rows += length(rows)

        push!(packages, (name = pkg, structs = names, methods = rows))
        println(rpad(pkg, 12), " structs=", lpad(length(names), 5),
                " methods=", lpad(length(rows), 6))
    end

    isempty(packages) && error("no C++ Hub packages with metadata found under $HUB")

    # ---- Faithfulness proof: reduced corpus must reproduce the live sweep ----
    # Reconstruct `structs` the way the test will (names only, dummy values) and
    # confirm both gates answer identically to the real metadata Dict.
    mismatches = String[]
    for p in packages
        real_meta = JSON.parsefile(joinpath(HUB, p.name, "julia", "compilation_metadata.json"))
        real_structs = get(real_meta, "struct_definitions", Dict())
        real_stypes  = Set{String}(String(k) for k in keys(real_structs))

        fixture_structs = Dict{String,Any}(n => nothing for n in p.structs)
        fixture_stypes  = Set{String}(p.structs)

        for row in p.methods
            cls, nm, mangled = row[1], row[2], row[3]
            a = verdicts(cls, nm, real_structs,    real_stypes)
            b = verdicts(cls, nm, fixture_structs, fixture_stypes)
            a == b || push!(mismatches, "$(p.name) :: $mangled real=$a fixture=$b")
        end
    end
    if !isempty(mismatches)
        error("Reduction is NOT faithful — $(length(mismatches)) verdict mismatch(es).\n" *
              "  First: " * join(first(mismatches, 5), "\n         ") *
              "\nA gate must be reading struct VALUES, not just keys. Do not commit this fixture.")
    end

    # ---- Emit ----
    # Hand-emitted so key order is fixed and the file diffs cleanly; `JSON.json`
    # is used per-scalar so escaping stays correct.
    mkpath(dirname(OUT))
    open(OUT, "w") do io
        println(io, "{")
        println(io, "  \"_comment\": ", JSON.json(
            "Reduced receiver-gate decision table. Regenerate with " *
            "test/gen_receiver_corpus.jl; do not hand-edit. Both gates are pure " *
            "functions of (class, name) plus the aggregate-name set, so only " *
            "those are stored."), ",")
        println(io, "  \"_generated\": ", JSON.json(string(Dates.today())), ",")
        println(io, "  \"_source\": ", JSON.json("RepliBuild-Hub/packages"), ",")
        println(io, "  \"packages\": [")
        for (i, p) in enumerate(packages)
            println(io, "    {")
            println(io, "      \"name\": ", JSON.json(p.name), ",")
            println(io, "      \"structs\": [")
            for (j, n) in enumerate(p.structs)
                println(io, "        ", JSON.json(n), j == length(p.structs) ? "" : ",")
            end
            println(io, "      ],")
            println(io, "      \"methods\": [")
            for (j, r) in enumerate(p.methods)
                println(io, "        [", JSON.json(r[1]), ", ", JSON.json(r[2]), ", ",
                        JSON.json(r[3]), "]", j == length(p.methods) ? "" : ",")
            end
            println(io, "      ]")
            println(io, "    }", i == length(packages) ? "" : ",")
        end
        println(io, "  ]")
        println(io, "}")
    end

    println("-"^62)
    println("Hub method rows swept : ", full_sweep)
    println("distinct rows written : ", reduced_rows)
    println("verdict mismatches    : 0  (reduction proven faithful)")
    println("wrote ", OUT, "  (", round(filesize(OUT) / 1024, digits=1), " KB)")
end

main()
