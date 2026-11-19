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
        # `make_rect_nx` takes the same arguments, so the factory is named.
        t += M.area(M.Rect(Val(:make_rect), 2.0, 3.0))
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

# Twig's destructor is an alias of Leaf's D2 (-O1 and up). Its owner must free
# through Twig's own D0: Leaf's body runs once per Twig and Leaf's operator
# delete frees it, so both counters advance together.
probe("managed_twig_deletes_through_its_own_destructor") do p
    d0, f0 = M.leaf_dtors(), M.leaf_frees()
    churn(() -> M.spawn_twig_safe(5), 10)
    collect_all()
    d, f = M.leaf_dtors() - d0, M.leaf_frees() - f0
    f > 0 || push!(p, "no Twig was freed (destructor bodies: $d)")
    f == d || push!(p, "$d destructor bodies but $f frees")
end

# Sprout's only destructor is D0. Bound as `Sprout_delete_Sprout`, it runs
# Leaf's body once and Leaf's operator delete once, and the storage is gone:
# nothing to free afterwards. Before, it was `Sprout_destroy_Sprout`, named
# like the non-freeing destructors, and a caller who freed after it double-freed.
probe("d0_only_class_binds_delete_which_frees_once") do p
    isdefined(M, :Sprout_destroy_Sprout) && push!(p, "Sprout_destroy_Sprout is defined")
    isdefined(M, :Sprout_delete_Sprout) || (push!(p, "Sprout_delete_Sprout is not defined"); return)
    d0, f0 = M.leaf_dtors(), M.leaf_frees()
    s = M.spawn_sprout(4)
    M.Sprout_delete_Sprout(s)
    d, f = M.leaf_dtors() - d0, M.leaf_frees() - f0
    d == 1 || push!(p, "$d destructor bodies ran")
    f == 1 || push!(p, "$f frees")
end

# Leaf has all three variants: `destroy_` ends the object (D2) and leaves the
# storage, `delete_` (D0) frees it.
probe("destroy_and_delete_are_two_bindings") do p
    d0, f0 = M.leaf_dtors(), M.leaf_frees()
    l = M.spawn_leaf(6)
    M.Leaf_destroy_Leaf(l)
    (M.leaf_dtors() - d0, M.leaf_frees() - f0) == (1, 0) || push!(p, "destroy_ freed or did not destroy")
    M.Leaf_delete_Leaf(M.spawn_leaf(7))
    (M.leaf_dtors() - d0, M.leaf_frees() - f0) == (2, 1) || push!(p, "delete_ did not free exactly once")
    # `l` was destroyed but not freed: give its storage back through the
    # class's operator delete, which counts it, so the library's books balance.
    ccall((:_ZN4LeafdlEPv, M.LIBRARY_PATH), Cvoid, (Ptr{Cvoid},), l)
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
