# Subprocess probe for test_alias_signatures_live.jl: calls each alias with
# arguments its result depends on, and each symbol that has no debug
# information, which must refuse. Prints one "PROBE <case>: PASS|FAIL …" line
# each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_alias_signatures.jl <wrapper_path> <c|cpp>

wrapper, kind = ARGS
const M = include(wrapper)

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

# The refusal: an ErrorException that says there is no debug information.
refused(f, args...) = try
    r = f(args...)
    "returned $(repr(r))"
catch e
    (e isa ErrorException && occursin("no debug information", e.msg)) || sprint(showerror, e)
end

# Leave different values in the argument registers between calls: the old
# wrapper called an alias with no arguments, and the callee read what was there.
@noinline churn(a, b, c) = (a * 31 + b) ⊻ c

if kind == "c"
    probe("alias_takes_its_argument") do
        for x in (5, -7, 1000, 0)
            churn(x + 99, x - 3, 12345)
            (got = M.al_name(x)) == x + 1 || return "al_name($x) = $got"
            M.al_real(x) == x + 1 || return "al_real($x)"
        end
        true
    end
    probe("weak_alias_floating_and_integer") do
        (M.al_scaled(1.5, 4) == 6.0 && M.al_scaled(-2.0, 3) == -6.0) || (M.al_scaled(1.5, 4), M.al_scaled(-2.0, 3))
    end
    probe("alias_returns_a_record") do
        p = M.al_pair(3.0, 1.0)
        (p.a == 4.0 && p.b == 2.0) || p
    end
    probe("alias_with_a_stack_argument") do
        M.al_wide(1, 2, 3, 4, 5, 6, 7) == 140 || M.al_wide(1, 2, 3, 4, 5, 6, 7)
    end
    probe("no_prototype_still_called") do
        M.al_old_style() == 1978 || M.al_old_style()
    end
    probe("no_debug_information_is_refused") do
        for args in ((), (1,), (1, 2))
            (r = refused(M.al_undescribed, args...)) === true || return r
        end
        true
    end
else
    probe("c_linkage_alias") do
        for x in (5, -7, 1000)
            churn(x + 99, x - 3, 12345)
            (got = M.ax_name(x)) == x + 1 || return "ax_name($x) = $got"
        end
        true
    end
    probe("mangled_alias") do
        (M.ax_mangled(1.5, 4) == 6.0 && M.ax_mangled_real(1.5, 4) == 6.0) || M.ax_mangled(1.5, 4)
    end
    probe("no_debug_information_is_refused") do
        for f in (M.ax_undescribed, M.ax_undescribed_cxx), args in ((), (1,))
            (r = refused(f, args...)) === true || return r
        end
        true
    end
    probe("alias_of_a_static_function_is_refused") do
        for args in ((), (6, 7))
            (r = refused(M.ax_via_static, args...)) === true || return r
        end
        true
    end
end

println("PROBE_DONE")
