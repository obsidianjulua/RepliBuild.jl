# Subprocess probe for test_field_ctor_shadow_live.jl: loads a wrapper of
# test/field_ctor_shadow_test and calls the struct `pair`'s field constructor
# and the function the wrap moved to `pair_`. Prints one
# "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe.jl <wrapper> c|cpp

wrapper = ARGS[1]
lang = ARGS[2]

function probe(f, name)
    verdict = try
        p = String[]
        f(p)
        isempty(p) ? "PASS" : "FAIL " * join(unique(p)[1:min(end, 6)], "; ")
    catch e
        "FAIL threw " * first(replace(sprint(showerror, e), '\n' => ' '), 300)
    end
    println("PROBE $name: ", verdict)
end

# A module that fails to load ends the probe here, with no PROBE_DONE.
const G = include(wrapper)

probe("field_constructor_builds_the_struct") do p
    for v in (G.pair(1, 2), G.pair(Cint(1), Cint(2)), G.pair(Int8(1), UInt16(2)))
        (v isa G.pair && (v.a, v.b) == (1, 2)) || push!(p, "pair(…) = $(repr(v))")
    end
    # ... and it is the struct the library reads.
    G.pair_sum(G.pair(3, 4)) == 7 || push!(p, "pair_sum(pair(3, 4)) = $(G.pair_sum(G.pair(3, 4)))")
    G.make_pair(5, 6) == G.pair(5, 6) || push!(p, "make_pair(5, 6) != pair(5, 6)")
end

probe("function_under_moved_name") do p
    G.pair_(1, 2) == 12 || push!(p, "pair_(1, 2) = $(G.pair_(1, 2))")
    G.pair_(Cint(1), Cint(2)) == 12 || push!(p, "pair_(Cint(1), Cint(2))")
end

probe("nothing_else_moved") do p
    G.triple(4) == 12 || push!(p, "triple(4) = $(G.triple(4))")
    G.flags(1, 2) == 3 || push!(p, "flags(1, 2) = $(G.flags(1, 2))")
    G.num(5) == -5 || push!(p, "num(5) = $(G.num(5))")
    G.make_flags() isa G.flags || push!(p, "make_flags()")
    G.make_num() isa G.num || push!(p, "make_num()")
    if lang == "c"
        G.packed_s(1, 2) == 3 || push!(p, "packed_s(1, 2) = $(G.packed_s(1, 2))")
    end
    for n in (:triple_, :flags_, :num_, :packed_s_)
        isdefined(G, n) && push!(p, "$n is bound")
    end
end

println("PROBE_DONE")
