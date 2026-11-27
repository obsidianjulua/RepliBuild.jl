# Subprocess probe for test_cxx_objects_live.jl: calls the wrapper of
# ownership_test from Julia, as a user would. A wrong ending here is a double
# free and a wrong move a use after free, so this runs in a child. Prints one
# "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_owned.jl <wrapper_path>

using RepliBuild

wrapper, = ARGS
const M = include(wrapper)
const Owned = RepliBuild.CxxObjects.Owned

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
# The error a call raises, or `nothing` when it was made.
raised(f) = try f(); nothing catch e; e end
argerr(p, what, f, needle) = (e = raised(f); (e isa ArgumentError && occursin(needle, e.msg)) ||
                              push!(p, "$what: got $(e === nothing ? "no error" : sprint(showerror, e)), want an ArgumentError with `$needle`"))

live() = M.own_live_count()
# Everything unreachable is collected and its finalizer run.
collect_all() = (for _ in 1:4; GC.gc(true); end; nothing)

# A Shape to call the member functions on: its constructor builds it in
# storage the probe holds for the whole run.
const SHAPE = Ref{M.Shape}()
M.own_Shape_Shape(SHAPE)

# ── What comes back ──────────────────────────────────────────────────────────

# A type that has to be ended is the object, held where C++ built it; a type
# with nothing to end is the plain value it was.
probe("returns_are_held_or_plain") do p
    v = M.own_iota(4)
    check(p, "a vector", v isa Owned{NTuple{24, UInt8}}, true)
    check(p, "a map", M.own_Shape_index(SHAPE, 3) isa Owned{NTuple{48, UInt8}}, true)
    check(p, "a list", M.own_Shape_chain_abi_cxx11(SHAPE, 3) isa Owned{NTuple{24, UInt8}}, true)
    check(p, "a unique_ptr", M.own_Shape_boxed(SHAPE, 1) isa Owned{NTuple{8, UInt8}}, true)
    check(p, "a class with a destructor", M.own_Shape_tracked(SHAPE, 1) isa Owned{M.Tracked}, true)
    check(p, "a class that cannot be copied", M.own_Shape_token(SHAPE, 1) isa Owned{M.Token}, true)
    check(p, "a copy constructor and nothing to end", M.own_Shape_copied(SHAPE) isa M.Copied, true)
    check(p, "plain data", M.own_Shape_plain(SHAPE) isa M.Plain, true)
    check(p, "its fields", (M.own_Shape_plain(SHAPE).a, M.own_Shape_plain(SHAPE).b), (7, 2.5))
    check(p, "copied's value", M.own_Shape_copied(SHAPE).v, (1, 2, 3, 4))
end
collect_all()

# The library reads each one right where it is. A `std::map` and a `std::list`
# point into themselves: moved by their bytes, the last key was found wrong and
# a walk to the end did not stop.
probe("held_objects_read_right") do p
    check(p, "sum of a vector", M.own_sum(M.own_iota(100)), 4950)
    check(p, "the last key of a map", M.own_map_last(M.own_Shape_index(SHAPE, 40)), 40)
    check(p, "a walk to the end of a list", M.own_list_sum(M.own_Shape_chain_abi_cxx11(SHAPE, 100)), 5050)
    check(p, "an empty list", M.own_list_sum(M.own_Shape_chain_abi_cxx11(SHAPE, 0)), 0)
    check(p, "a class through its own method", M.own_Tracked_value(M.own_Shape_tracked(SHAPE, 17)), 17)
    check(p, "a unique_ptr by reference", M.own_boxed_value(M.own_Shape_boxed(SHAPE, 23)), 23)
    # `Ref(x)`, the manual's way while a container was its bytes, still passes the object.
    v = M.own_iota(10)
    check(p, "Ref of a vector", M.own_sum(Ref(v)), 45)
    check(p, "Ref of a unique_ptr", M.own_boxed_value(Ref(M.own_Shape_boxed(SHAPE, 24))), 24)
    check(p, "Ref of a class", M.own_peek(Ref(M.own_Shape_tracked(SHAPE, 25))), 25)
end
collect_all()

# A reference is to the caller's object: the library writes through it.
probe("a_reference_is_the_callers_object") do p
    v = M.own_iota(3)                                   # 0 1 2
    M.own_append(v, 10)
    M.own_append(v, 20)
    check(p, "after two appends", M.own_sum(v), 33)
    for i in 1:1000; M.own_append(v, 1); end            # grows: the object reallocates, the holder does not move
    check(p, "after it grew", M.own_sum(v), 1033)
end
collect_all()

# An overload set that differs in what the reference is to keeps two methods,
# and each takes its own object.
probe("overloads_take_their_own_object") do p
    check(p, "methods of peek", length(methods(M.own_peek)), 2)
    check(p, "peek(Tracked)", M.own_peek(M.own_Shape_tracked(SHAPE, 31)), 31)
    check(p, "peek(vector)", M.own_peek(M.own_iota(7)), 7)
end
collect_all()

# ── Ended once ───────────────────────────────────────────────────────────────

probe("released_once") do p
    base = live()
    t = M.own_Shape_tracked(SHAPE, 5)
    check(p, "alive after the call", live(), base + 1)
    RepliBuild.release!(t)
    check(p, "ended by release!", live(), base)
    RepliBuild.release!(t)
    finalize(t)
    check(p, "released twice, then collected", live(), base)
    check(p, "isalive", RepliBuild.isalive(t), false)
    argerr(p, "a method on it", () -> M.own_Tracked_value(t), "was released")
    argerr(p, "by value", () -> M.own_Shape_see(SHAPE, t), "was released")
end

probe("collected_once") do p
    base = live()
    made = 300
    (() -> (for i in 1:made; M.own_Shape_tracked(SHAPE, i); M.own_Shape_boxed(SHAPE, i); M.own_Shape_token(SHAPE, i); end; nothing))()
    check(p, "alive before collection", live() >= base + 3 * made - 6, true)
    collect_all()
    left = live() - base
    (0 <= left <= 6) || push!(p, "after collection: $left of $(3 * made) still alive")   # a conservative root may hold the last few
end
collect_all()
const BASE = live()                                     # whatever a conservative root still holds

# A call that throws before it returns built nothing: there is nothing to end.
probe("a_return_that_throws_leaves_nothing") do p
    base = live()
    e = raised(() -> M.own_Shape_broken(SHAPE, 1))
    check(p, "the C++ exception is a Julia error", e !== nothing && occursin("broken", sprint(showerror, e)), true)
    collect_all()
    check(p, "live objects", live(), base)
end

# ── By value ─────────────────────────────────────────────────────────────────

# C++ copies where it can: the call gets its own copy and the caller keeps
# the object, even when the callee moves out of what it was given.
probe("by_value_is_a_copy") do p
    v = M.own_iota(100)
    check(p, "take reads it", M.own_Shape_take(SHAPE, v), 4950)
    check(p, "keep moves out of its copy", M.own_Shape_keep(SHAPE, v), 100)
    check(p, "what the library kept", M.own_Shape_kept(SHAPE), 4950)
    check(p, "the caller's vector is whole", M.own_sum(v), 4950)
    check(p, "still alive", RepliBuild.isalive(v), true)
    check(p, "and passes again", M.own_Shape_take(SHAPE, v), 4950)
    # A class with a layout here: the thunk's copy constructor reads the object where it is.
    base = live()
    t = M.own_Shape_tracked(SHAPE, 9)
    check(p, "see", M.own_Shape_see(SHAPE, t), 9)
    check(p, "the copy was ended", live(), base + 1)
    check(p, "the caller's object is whole", M.own_Tracked_value(t), 9)
    # `copy(x)` is the copy constructor too, and a second object.
    c = copy(t)
    check(p, "a copy is one more", live(), base + 2)
    RepliBuild.release!(t)
    check(p, "the copy outlives the original", M.own_Tracked_value(c), 9)
    RepliBuild.release!(c)
    check(p, "both ended", live(), base)
end
collect_all()

# What C++ cannot copy is moved: the call takes it, and it is spent.
probe("uncopyable_is_moved_and_spent") do p
    base = live()
    b = M.own_Shape_boxed(SHAPE, 77)
    check(p, "made", live(), base + 1)
    check(p, "adopt", M.own_Shape_adopt(SHAPE, b), 77)
    check(p, "the call ended it", live(), base)
    check(p, "isalive", RepliBuild.isalive(b), false)
    argerr(p, "passing it again", () -> M.own_Shape_adopt(SHAPE, b), "was moved into a C++ call")
    argerr(p, "by reference", () -> M.own_boxed_value(b), "was moved into a C++ call")
    argerr(p, "copy", () -> copy(M.own_Shape_boxed(SHAPE, 1)), "cannot be copied")
    # A class with a layout here that C++ can only move: the thunk moves, the object is spent.
    collect_all()
    base = live()
    k = M.own_Shape_token(SHAPE, 55)
    check(p, "redeem", M.own_Shape_redeem(SHAPE, k), 55)
    check(p, "the token is ended", live(), base)
    argerr(p, "redeeming it again", () -> M.own_Shape_redeem(SHAPE, k), "was moved into a C++ call")
    finalize(b); finalize(k)
    check(p, "collected afterwards: not ended again", live(), base)
end
collect_all()

# `move(x)` gives the object where a copy would have been made.
probe("move_gives_the_object") do p
    v = M.own_iota(100)
    check(p, "keep(move(v))", M.own_Shape_keep(SHAPE, RepliBuild.move(v)), 100)
    check(p, "what the library kept", M.own_Shape_kept(SHAPE), 4950)
    check(p, "isalive", RepliBuild.isalive(v), false)
    argerr(p, "using it", () -> M.own_sum(v), "was moved into a C++ call")
    base = live()
    t = M.own_Shape_tracked(SHAPE, 3)
    check(p, "see(move(t))", M.own_Shape_see(SHAPE, RepliBuild.move(t)), 3)
    check(p, "ended by the call", live(), base)
    argerr(p, "using it", () -> M.own_Tracked_value(t), "was moved into a C++ call")
end
collect_all()

# A call that throws still ends what it was given, as C++ unwinds a by-value argument.
probe("a_throw_settles_the_argument") do p
    v = M.own_iota(5)
    e = raised(() -> M.own_Shape_fail(SHAPE, v))
    check(p, "the C++ exception", e !== nothing && occursin("refused 5", sprint(showerror, e)), true)
    check(p, "the caller's vector is whole", M.own_sum(v), 10)
    e = raised(() -> M.own_Shape_fail(SHAPE, RepliBuild.move(v)))
    check(p, "thrown again", e !== nothing && occursin("refused 5", sprint(showerror, e)), true)
    check(p, "a moved one is spent all the same", RepliBuild.isalive(v), false)
end
collect_all()

# What is not that type is refused before any call.
probe("another_type_is_refused") do p
    argerr(p, "a list where a vector is taken", () -> M.own_Shape_take(SHAPE, M.own_Shape_chain_abi_cxx11(SHAPE, 3)), "by value")
    e = raised(() -> M.own_Shape_take(SHAPE, 3))
    check(p, "an integer", e isa MethodError, true)
    e = raised(() -> M.own_Shape_see(SHAPE, M.own_iota(3)))
    check(p, "a vector where a class is taken", e isa MethodError, true)
    # The plain bytes, and a pointer to an object, still pass: a copy is made from them.
    v = M.own_iota(10)
    check(p, "the bytes", M.own_Shape_take(SHAPE, v[]), 45)
    check(p, "a pointer", GC.@preserve(v, M.own_Shape_take(SHAPE, RepliBuild.CxxObjects.address(v))), 45)
    check(p, "the caller's vector is whole", M.own_sum(v), 45)
end
collect_all()

# After all of it, nothing the probe made is still alive: every copy, every
# temporary and every moved object was ended, once.
probe("nothing_left_alive") do p
    collect_all()
    left = live() - BASE
    (-6 <= left <= 6) || push!(p, "live objects moved by $left since the by-value cases began")
    # ...and exactly none where the collector is not involved.
    base = live()
    t = M.own_Shape_tracked(SHAPE, 1); M.own_Shape_see(SHAPE, t); c = copy(t)
    b = M.own_Shape_boxed(SHAPE, 2); M.own_Shape_adopt(SHAPE, b)
    k = M.own_Shape_token(SHAPE, 3); M.own_Shape_redeem(SHAPE, k)
    foreach(RepliBuild.release!, (t, c, b, k))
    check(p, "made, copied, moved and released", live(), base)
end

println("PROBE_DONE")
