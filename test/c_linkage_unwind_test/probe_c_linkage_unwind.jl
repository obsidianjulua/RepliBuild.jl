# Subprocess probe for test_c_linkage_unwind_live.jl: calls the fixture's
# `extern "C"` functions through the wrapper, the throwing ones with the
# argument that makes them throw. Through a plain ccall the first throw ends
# this process in std::terminate (SIGABRT), which is what the parent would
# see. Prints one "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_c_linkage_unwind.jl <wrapper_path>

using RepliBuild
wrapper, = ARGS
const M = include(wrapper)
const CxxE = RepliBuild.JITManager.CxxException

function probe(f, name)
    verdict = try
        r = f()
        r === true ? "PASS" : "FAIL got " * first(repr(r), 300)
    catch e
        "FAIL threw " * first(sprint(showerror, e), 300)
    end
    println("PROBE $name: ", verdict)
    flush(stdout)
end

# What a call raised, or `:returned`.
raised(f) = try f(); :returned catch e; e end
cxx(f, text) = (e = raised(f); (e isa CxxE && occursin(text, e.message)) || e)

probe("throw_arrives_as_CxxException") do
    cxx(() -> M.cl_checked(-1), "cl_checked: negative")
end
probe("returns_through_the_thunk") do
    M.cl_checked(5) == 105 || M.cl_checked(5)
end
probe("void_return") do
    M.cl_checked_void(1) === nothing || return M.cl_checked_void(1)
    cxx(() -> M.cl_checked_void(-1), "cl_checked_void: negative")
end
probe("cstring_return") do
    (M.cl_name(0) == "zero" && M.cl_name(7) == "other") || return (M.cl_name(0), M.cl_name(7))
    cxx(() -> M.cl_name(-1), "cl_name: negative")
end
probe("record_return") do
    p = M.cl_point(1.0, 2.0)
    (p.x == 2.0 && p.y == 4.0) || return p
    cxx(() -> M.cl_point(-1.0, 2.0), "cl_point: negative")
end
probe("function_pointer_argument") do
    double_it = @cfunction(x -> Cint(2x), Cint, (Cint,))
    M.cl_apply(double_it, 4) == 9 || return M.cl_apply(double_it, 4)
    cxx(() -> M.cl_apply(double_it, -4), "cl_apply: negative")
end
probe("throw_of_a_non_exception") do
    M.cl_throws_int(3) == 3 || return M.cl_throws_int(3)
    e = raised(() -> M.cl_throws_int(-1))
    e isa CxxE || e
end
probe("proven_nounwind_keeps_ccall") do
    (M.cl_add(2, 3) == 5 && M.cl_plain_c(4) == 12) || return (M.cl_add(2, 3), M.cl_plain_c(4))
    tiers = (M.dispatch_tier(:cl_add), M.dispatch_tier(:cl_plain_c), M.dispatch_tier(:cl_calls),
             M.dispatch_tier(:cl_checked), M.dispatch_tier(:cl_apply))
    tiers == (:tier3, :tier3, :tier3, :tier2, :tier2) || tiers
end
probe("alive_after_many_throws") do
    before = M.cl_calls()
    thrown = 0
    for i in 1:200
        raised(() -> M.cl_checked(-i)) isa CxxE && (thrown += 1)
        M.cl_checked(i) == i + 100 || return "cl_checked($i) after a throw"
    end
    (thrown == 200 && M.cl_calls() - before == 200) || (thrown, M.cl_calls() - before)
end

println("PROBE_DONE")
