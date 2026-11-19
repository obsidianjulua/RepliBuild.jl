# Subprocess probe for test_type_name_collisions_live.jl: loads the wrapper of
# test/type_name_collision_test (a global and two functions named like the
# structs they read) and calls each one under the name the wrap moved it to.
# Prints one "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe.jl <wrapper>

wrapper = ARGS[1]

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

zeroed(v, fields) = all(f -> getfield(v, f) == 0, fields)

probe("type_constructors_are_zeroed") do p
    c = G.conf()
    (c isa G.conf && zeroed(c, (:level, :flags))) || push!(p, "conf() = $c")
    q = G.pool()
    (q isa G.pool && zeroed(q, (:size, :used))) || push!(p, "pool() = $q")
    t = G.token()
    (t isa G.token && zeroed(t, (:kind, :len))) || push!(p, "token() = $t")
end

probe("global_value_under_moved_name") do p
    v = G.conf_()
    v isa Ptr{G.conf} || push!(p, "conf_() isa $(typeof(v))")
    v == G.conf_default_ptr() || push!(p, "conf_() is not &conf_default")
    unsafe_load(v).level == 3 || push!(p, "conf_()->level = $(unsafe_load(v).level)")
    G.conf_level(v) == 3 || push!(p, "conf_level(conf_()) = $(G.conf_level(v))")
    G.conf_default().flags == 1 || push!(p, "conf_default().flags = $(G.conf_default().flags)")
end

probe("global_pointer_keeps_its_name") do p
    pp = G.conf_ptr()
    pp isa Ptr{Ptr{G.conf}} || push!(p, "conf_ptr() isa $(typeof(pp))")
    unsafe_load(pp) == G.conf_() || push!(p, "*conf_ptr() != conf_()")
    isdefined(G, :conf__ptr) && push!(p, "conf__ptr is bound")
end

probe("functions_under_moved_names") do p
    q = G.pool_()
    (q isa Ptr{G.pool} && unsafe_load(q).size == 5 && unsafe_load(q).used == 2) || push!(p, "pool_() = $q")
    t = G.token_()
    (t isa G.token && t.kind == 7 && t.len == 2) || push!(p, "token_() = $t")
end

probe("nothing_else_moved") do p
    G.counter() == 9 || push!(p, "counter() = $(G.counter())")
    unsafe_load(G.counter_ptr()) == 9 || push!(p, "counter_ptr")
    for n in (:counter_, :conf_level_, :conf_default_)
        isdefined(G, n) && push!(p, "$n is bound")
    end
end

println("PROBE_DONE")
