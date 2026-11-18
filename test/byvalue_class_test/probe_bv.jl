# Subprocess probe for test_byvalue_class_live.jl. A thunk module that calls a
# symbol it never declares kills the process when the wrapper loads, and a
# temporary made the wrong way frees memory twice, so this runs in a child.
# Prints one "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
#
# Run: julia --project=<RepliBuild> probe_bv.jl <wrapper_path> jit
#      julia --project=<RepliBuild> probe_bv.jl <replibuild.toml> aot
# With `aot` the child builds and wraps first: compiling the thunks is wrap's
# last stage, and it translates the same module the JIT is given.

using RepliBuild

target, mode = ARGS
wrapper = if mode == "aot"
    RepliBuild.build(target)
    RepliBuild.wrap(target)
else
    target
end
const M = include(wrapper)
const JM = RepliBuild.JITManager

function probe(f, name)
    verdict = try
        p = String[]
        f(p)
        isempty(p) ? "PASS" : "FAIL " * join(p, "; ")
    catch e
        "FAIL threw " * first(sprint(showerror, e), 300)
    end
    println("PROBE $name: ", verdict)
    flush(stdout)
end
check(p, what, got, want) = got == want || push!(p, "$what: got $got, want $want")
# The message of the error `f()` raises, "" when it returns.
refusal(f) = try
    f()
    ""
catch e
    sprint(showerror, e)
end
function check_refused(p, what, msg, parts...)
    for part in ("cannot be passed by value", parts...)
        occursin(part, msg) || push!(p, "$what: no `$part` in the error: $(first(msg, 240))")
    end
end

# The wrapper loaded at all: its thunks came from the engine (or, wrapped with
# `aot_thunks`, from the thunks library and no engine was built).
if mode == "aot"
    probe("thunks_loaded") do p
        unresolved = [sym for (slot, sym) in M._THUNK_SLOTS if slot[] == C_NULL]
        check(p, "thunks library open", M.THUNKS_HANDLE[] != C_NULL, true)
        check(p, "unresolved thunk slots", unresolved, String[])
        check(p, "JIT engines", length(JM.GLOBAL_JIT.engines), 0)
    end
else
    probe("thunks_loaded") do p
        check(p, "JIT engines", length(JM.GLOBAL_JIT.engines), 1)
        for eng in JM.GLOBAL_JIT.engines
            check(p, "engine init error", eng.init_error === nothing ? "none" : sprint(showerror, eng.init_error), "none")
            check(p, "engine built", eng.jit_engine !== nothing, true)
        end
    end
end

# A copyable class: the callee gets a copy, which is ended; the caller's own
# object is untouched and can be passed again.
probe("copy_keeps_the_callers") do p
    r = M.make_res(5)
    l0 = M.live()
    M.reset_tally()
    check(p, "take_res, twice", (M.take_res(r), M.take_res(r)), (5, 5))
    check(p, "live", M.live() - l0, 0)
    check(p, "tally (2 copies, 2 destructors)", M.tally(), 202)
end

# A move-only class: one move, one destructor, and the resource ends once.
# Passed as its bytes, the caller's object was freed with nothing moved.
probe("move_consumes_once") do p
    u = M.make_uniq(7)
    l0 = M.live()
    M.reset_tally()
    check(p, "consume_uniq", M.consume_uniq(u), 7)
    check(p, "live dropped by", l0 - M.live(), 1)
    check(p, "tally (1 move, 1 destructor)", M.tally(), 11)
end

probe("no_copy_no_move_is_refused") do p
    x = M.make_pinned(3)
    l0 = M.live()
    check_refused(p, "take_pinned", refusal(() -> M.take_pinned(x)),
                  "`Pinned`", "copy constructor is deleted", "no move constructor")
    check(p, "live", M.live() - l0, 0)
end

probe("private_copy_is_refused") do p
    g = M.make_guarded(6)
    check_refused(p, "take_guarded", refusal(() -> M.take_guarded(g)), "copy constructor is private")
end

probe("trivial_copy_is_bytes") do p
    M.reset_tally()
    check(p, "take_plain", M.take_plain(M.Plain(Cint(9))), 9)
    check(p, "tally (1 destructor)", M.tally(), 1)
end

probe("defaulted_copy_is_bytes") do p
    d = M.make_dflt(11)
    M.reset_tally()
    check(p, "take_dflt", M.take_dflt(d), 11)
    check(p, "tally (1 destructor)", M.tally(), 1)
end

probe("implicit_nontrivial_copy_is_refused") do p
    h = M.make_holder(4)
    check_refused(p, "take_holder", refusal(() -> M.take_holder(h)), "implicit and not trivial")
end

# The object holds its own address. Its copy constructor sets that; its bytes
# carry the address of wherever it was built.
probe("copy_constructor_runs") do p
    t = M.make_tracked(13)
    M.reset_tally()
    check(p, "take_tracked", M.take_tracked(t), 13)
    check(p, "tally (1 copy, 1 destructor)", M.tally(), 101)
end

probe("virtual_destructor") do p
    x = M.make_poly(8)
    M.reset_tally()
    check(p, "take_poly", M.take_poly(x), 8)
    check(p, "tally (1 copy, 1 destructor)", M.tally(), 101)
end

# The temporary is aligned as the object: the callee sees its own address.
probe("aligned_temporary") do p
    w = M.make_wide(33)
    M.reset_tally()
    check(p, "take_wide", M.take_wide(w), 33)
    check(p, "tally (1 copy, 1 destructor)", M.tally(), 101)
end

probe("nested_class") do p
    i = M.make_inner(31)
    l0 = M.live()
    M.reset_tally()
    check(p, "take_inner, twice", (M.take_inner(i), M.take_inner(i)), (31, 31))
    check(p, "live", M.live() - l0, 0)
    check(p, "tally", M.tally(), 202)
end

probe("namespaced_copy") do p
    r = M.ns_make_nres(21)
    l0 = M.live()
    M.reset_tally()
    check(p, "ns::take_nres, twice", (M.ns_take_nres(r), M.ns_take_nres(r)), (21, 21))
    check(p, "live", M.live() - l0, 0)
    check(p, "tally", M.tally(), 202)
end

probe("namespaced_move") do p
    u = M.ns_make_nuniq(22)
    l0 = M.live()
    M.reset_tally()
    check(p, "ns::consume_nuniq", M.ns_consume_nuniq(u), 22)
    check(p, "live dropped by", l0 - M.live(), 1)
    check(p, "tally", M.tally(), 11)
end

probe("template_class") do p
    b = M.ns_make_box(17)
    l0 = M.live()
    M.reset_tally()
    check(p, "ns::take_box, twice", (M.ns_take_box(b), M.ns_take_box(b)), (17, 17))
    check(p, "live", M.live() - l0, 0)
    check(p, "tally", M.tally(), 202)
end

# `one::Twin` and `two::Twin` are one name to a parameter.
probe("shared_name_is_refused") do p
    t = M.make_twin(2)
    check_refused(p, "take_twin", refusal(() -> M.take_twin(t)), "named `Twin`")
end

# Two temporaries in one call, a scalar between them: a copy and a move, each
# ended once.
probe("two_temporaries") do p
    r = M.make_res(5)
    u = M.make_uniq(7)
    l0 = M.live()
    M.reset_tally()
    check(p, "take_both", M.take_both(r, 10, u), 57)
    check(p, "live dropped by", l0 - M.live(), 1)
    check(p, "tally (1 copy, 1 move, 2 destructors)", M.tally(), 112)
end

println("PROBE_DONE")
