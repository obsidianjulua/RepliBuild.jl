#!/usr/bin/env julia
# Regenerate test/fixtures/thunk_symbols.json — the vendored `nm` output for the
# MI/VI fixtures that test_symbol_hygiene.jl drives its Itanium-thunk predicate
# over.
#
# Hand-run, wired into no suite (the wiring guard only scans `test_*.jl`).
# Run it after a change that moves fixture symbols, then commit the fixture:
#
#     julia --project=. test/devtests.jl      # build the fixtures first
#     julia --project=. test/gen_thunk_symbols.jl
#
# Why vendored at all: reading the `.so` directly made the assertions skip
# silently wherever the fixtures were unbuilt — a fresh clone reported 89/89
# where this box reported 97/97, both green. Same reason the receiver-gate
# corpus is vendored; see test/gen_receiver_corpus.jl.

import JSON

const FIXTURES = ("mi_test", "vi_test")
const OUT = joinpath(@__DIR__, "fixtures", "thunk_symbols.json")

out = Dict{String,Any}()
for f in FIXTURES
    so = joinpath(@__DIR__, f, "julia", "lib$(f).so")
    isfile(so) || error("fixture not built: $so\nRun test/devtests.jl first.")
    syms = String[]
    for line in split(read(`nm -g --defined-only $so`, String), '\n')
        p = split(strip(line))
        length(p) >= 3 && p[2] in ("T", "W", "t", "w") && push!(syms, String(p[3]))
    end
    isempty(syms) && error("no code symbols in $so — is it stripped?")
    # The whole point of the fixture is that it still EXERCISES the thunk class.
    # Writing a list with none would pin a guard that proves nothing.
    any(s -> startswith(s, "_ZTh") || startswith(s, "_ZTv"), syms) ||
        error("$f defines no _ZTh/_ZTv thunks — refusing to write a vacuous fixture")
    out[f] = sort(unique(syms))
end

open(OUT, "w") do io
    JSON.print(io, Dict("generated_from" => "nm -g --defined-only on " * join(FIXTURES, ", "),
                        "fixtures" => out), 2)
end
println("wrote $(sum(length(v) for v in values(out))) symbols across $(length(out)) fixtures → $OUT")
