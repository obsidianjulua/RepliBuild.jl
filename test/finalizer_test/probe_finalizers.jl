# Subprocess probe for test_cpp_finalizers_live.jl: makes objects through the
# wrapper types that own them, drops them, collects, and reads the library's
# counters. Prints one "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_finalizers.jl <wrapper_path>

wrapper, = ARGS
const M = include(wrapper)

function probe(f, name)
    verdict = try
        p = String[]
        f(p)
        isempty(p) ? "PASS" : "FAIL " * join(p, "; ")
    catch e
        "FAIL threw " * first(sprint(showerror, e), 300)
    end
    println("PROBE $name: ", verdict)
end

# Make `n` objects in a frame of their own, so none is reachable afterwards.
@noinline churn(make, n) = (for _ in 1:n; make(); end; nothing)
collect_all() = (for _ in 1:4; GC.gc(true); end; yield())

# make_rect's struct, and its method proxy.
@noinline function rect_areas(n)
    t = 0.0
    for _ in 1:n
        t += M.area(M.Rect(2.0, 3.0))
    end
    return t
end

probe("rect_finalizer_deletes") do p
    (s = rect_areas(20)) == 120.0 || push!(p, "area sum $s")
    collect_all()
    d, f = M.rect_dtors(), M.rect_frees()
    f > 0 || push!(p, "no Rect was freed (destructor bodies: $d)")
    f == d || push!(p, "$d destructor bodies but $f frees")
end

probe("widget_finalizer_calls_destroy_widget") do p
    churn(() -> M.Widget(Cint(7)), 10)            # a factory struct takes the C type exactly
    collect_all()
    M.widget_frees() > 0 || push!(p, "destroy_widget never ran")
end

probe("managed_leaf_deletes") do p
    churn(() -> M.spawn_leaf_safe(3), 10)
    collect_all()
    d, f = M.leaf_dtors(), M.leaf_frees()
    f > 0 || push!(p, "no Leaf was freed (destructor bodies: $d)")
    f == d || push!(p, "$d destructor bodies but $f frees")
end

# Node::FreeMemory says free and takes only `this`: before the fix it could be
# the finalizer, and nothing was freed.
probe("managed_node_deletes") do p
    churn(() -> M.spawn_node_safe(), 10)
    collect_all()
    d, f = M.node_dtors(), M.node_frees()
    f > 0 || push!(p, "no Node was freed (destructor bodies: $d)")
    f == d || push!(p, "$d destructor bodies but $f frees")
end

probe("managed_pooled_calls_its_static_deleter") do p
    churn(() -> M.spawn_pooled_safe(), 10)
    collect_all()
    M.pooled_deletes() > 0 || push!(p, "Pooled::DeletePooled never ran")
end

probe("nothing_owns_a_plain") do p
    for name in (:ManagedPlain, :make_plain_safe, :spawn_plain_safe)
        isdefined(M, name) && push!(p, "$name is defined")
    end
    ismutabletype(M.Plain) && push!(p, "Plain is a handle type, not its layout")
end

probe("no_invented_class") do p
    isdefined(M, :RectNx) && push!(p, "RectNx is defined")
end

println("PROBE_DONE")
