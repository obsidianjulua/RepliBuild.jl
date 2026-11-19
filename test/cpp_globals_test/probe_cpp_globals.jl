# Subprocess probe for test_cpp_globals_live.jl: reads and writes C++ globals
# outside the global namespace through the wrapper's accessors. In a child so
# a wrong symbol cannot take the suite down. Prints one
# "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_cpp_globals.jl <wrapper_path>

wrapper, = ARGS
const M = include(wrapper)

has(s) = Base.invokelatest(isdefined, M, s)

function probe(f, name)
    verdict = try
        p = String[]
        Base.invokelatest(f, p)
        isempty(p) ? "PASS" : "FAIL " * join(p, "; ")
    catch e
        "FAIL threw " * first(sprint(showerror, e), 300)
    end
    println("PROBE $name: ", verdict)
end

# A namespace member, read and written through its own symbol.
probe("namespaced") do p
    (d = M.cfg_depth()) == 4 || push!(p, "cfg_depth() = $d")
    (s = M.cfg_scale()) == 1.25 || push!(p, "cfg_scale() = $s")
    unsafe_store!(M.cfg_depth_ptr(), Cint(6))
    (d = M.cfg_depth()) == 6 || push!(p, "cfg_depth() after a store = $d")
    (g = M.g_top()) == 9 || push!(p, "g_top() = $g")
end

# One bare name in two scopes is two accessors.
probe("same_name_two_scopes") do p
    (M.a_x(), M.b_x()) == (7, 8) || push!(p, "(a_x(), b_x()) = $((M.a_x(), M.b_x()))")
    M.a_x_ptr() != M.b_x_ptr() || push!(p, "a::x and b::x share an address")
end

# A static data member follows the class's own functions.
probe("static_member") do p
    c0 = M.Stat_count()
    M.Stat_bump()
    (c = M.Stat_count()) == c0 + 1 || push!(p, "Stat_count() $c0 -> $c across Stat::bump()")
    (f = M.Stat_factor()) == 2.5 || push!(p, "Stat_factor() = $f")
end

# A static member of a namespaced class counts what C++ does.
probe("live_count") do p
    n0 = M.geo_Vec3_live_count()
    v = M.geo_vec3_new()
    (n = M.geo_Vec3_live_count()) == n0 + 1 || push!(p, "live_count $n0 -> $n across vec3_new()")
    M.geo_vec3_free(v)
    (n = M.geo_Vec3_live_count()) == n0 || push!(p, "live_count $n after vec3_free()")
end

# No Julia value type for a std::string: the pointer, never a value.
probe("string_pointer_only") do p
    has(:g_string) && push!(p, "g_string has a value accessor")
    has(:g_string_ptr) || push!(p, "no g_string_ptr")
    has(:g_string_ptr) && M.g_string_ptr() == C_NULL && push!(p, "g_string_ptr() is NULL")
end

# Private and protected members, and a constant with no symbol, get nothing.
probe("not_offered") do p
    for n in names(M; all = true)
        s = string(n)
        (occursin("total_", s) || occursin("prot_", s) || occursin("LIMIT", s) || s == "Stat_K") &&
            push!(p, "$s is bound")
    end
    (t = M.geo_Counter_total()) == 11 || push!(p, "Counter::total() = $t")
end

println("PROBE_DONE")
