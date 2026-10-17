# Subprocess probe for test_aot_thunks.jl: load the AOT-thunk wrapper and call
# through it. Prints one `PROBE <name>: PASS|FAIL …` line per case, then
# PROBE_DONE.
#
# Each line is flushed as soon as it is written. A C++ throw that escapes a
# thunk ends the process, Julia block-buffers redirected stdout, and the
# verdicts printed before the crash are the diagnosis. The cases most likely to
# take the process down (exceptions) come last for the same reason.
#
# Run: julia --project=<RepliBuild> probe_aot.jl <wrapper_path>

wrapper = abspath(ARGS[1])   # `include` would resolve a relative path against this file
include(wrapper)
using .AotTest
import RepliBuild

const JM = RepliBuild.JITManager

function probe(f, name)
    verdict = try
        (ok, detail) = f()
        (ok ? "PASS" : "FAIL") * (isempty(detail) ? "" : " " * detail)
    catch e
        "FAIL threw " * replace(sprint(showerror, e), '\n' => " | ")
    end
    println("PROBE $name: ", verdict)
    flush(stdout)
end

# The exception a call raised, or `nothing` when it returned.
thrown(f) = try
    f()
    nothing
catch e
    e
end

cxx_error(e, needles...) =
    e isa JM.CxxException && all(n -> occursin(n, e.message), needles)

jit_untouched() =
    !(@atomic JM.GLOBAL_JIT.initialized) && isempty(JM.GLOBAL_JIT.engines)

# ── Loading: the thunks library opened and every slot resolved ─────────────

probe("aot_loaded") do
    slots = AotTest._THUNK_SLOTS
    unresolved = [sym for (slot, sym) in slots if slot[] == C_NULL]
    ok = AotTest.THUNKS_HANDLE[] != C_NULL && !isempty(slots) && isempty(unresolved)
    (ok, "slots=$(length(slots)) unresolved=$(unresolved)")
end

probe("dispatch_tier") do
    tiers = Dict(f => AotTest.dispatch_tier(f) for f in
                 (:Accumulator_add, :Accumulator_as_vec2, :Shape_area, :vec2_scale,
                  :vec3_cross, :pair_swap, :safe_ratio, :aot_new_accumulator))
    ok = all(tiers[f] === :tier2 for f in keys(tiers) if f !== :aot_new_accumulator) &&
         tiers[:aot_new_accumulator] === :tier3
    (ok, string(tiers))
end

# ── Plain member functions ─────────────────────────────────────────────────

probe("member_fn") do
    a = aot_new_accumulator(Int32(10))
    try
        r1 = Accumulator_add(a, Int32(5))
        r2 = Accumulator_add(a, Int32(-3))
        t = Accumulator_total(a)
        (r1 == 15 && r2 == 12 && t == 12, "add=$r1,$r2 total=$t")
    finally
        aot_free_accumulator(a)
    end
end

# A member returning a 16-byte record: sret on Win64, with `this` alongside.
probe("member_byvalue_return") do
    a = aot_new_accumulator(Int32(12))
    try
        v = Accumulator_as_vec2(a)
        (v == Vec2(12.0, -12.0), "v=$v")
    finally
        aot_free_accumulator(a)
    end
end

# ── By-value records, argument and return ──────────────────────────────────

probe("byvalue_pair32") do
    p = pair_swap(Pair32(Int32(1), Int32(-2)))
    (p == Pair32(Int32(-2), Int32(1)), "p=$p")
end

probe("byvalue_vec2") do
    v = vec2_scale(Vec2(1.5, -2.0), 2.0)
    d = vec2_dot(Vec2(1.0, 2.0), Vec2(3.0, 4.0))
    (v == Vec2(3.0, -4.0) && d == 11.0, "scale=$v dot=$d")
end

probe("byvalue_vec3") do
    c = vec3_cross(Vec3(1.0, 2.0, 3.0), Vec3(4.0, 5.0, 6.0))
    (c == Vec3(-3.0, 6.0, -3.0), "cross=$c")
end

# ── Virtual dispatch ───────────────────────────────────────────────────────
# One wrapper, two dynamic types, two answers. Calling the statically named
# Shape::area would give -1.0 for the Square too.

probe("virtual_override") do
    sq = aot_new_square_as_shape(3.0)
    base = aot_new_shape()
    try
        got = (Shape_area(sq), Shape_sides(sq), Shape_area(base), Shape_sides(base),
               Square_area(sq), Shape_area_times(sq, 2.0))
        (got == (9.0, Int32(4), -1.0, Int32(0), 9.0, 18.0), "got=$got")
    finally
        aot_free_shape(sq)
        aot_free_shape(base)
    end
end

# Every call above went through the thunks library. Had any of them reached
# for the JIT, an engine would exist now.
probe("no_jit") do
    (jit_untouched(),
     "initialized=$(@atomic JM.GLOBAL_JIT.initialized) engines=$(length(JM.GLOBAL_JIT.engines))")
end

# ── Exceptions crossing the thunk ──────────────────────────────────────────
# One return shape each; on Windows each is caught by a different libJLCS
# guard. Every case also checks the call still works when it does not throw.

probe("throw_i32") do
    a = aot_new_accumulator(Int32(1))
    try
        e = thrown(() -> Accumulator_checked_add(a, Int32(-4)))
        unchanged = Accumulator_total(a)
        ok_after = Accumulator_checked_add(a, Int32(4))
        (cxx_error(e, "checked_add", "-4") && unchanged == 1 && ok_after == 5,
         "e=$(repr(e)) total=$unchanged after=$ok_after")
    finally
        aot_free_accumulator(a)
    end
end

probe("throw_f64") do
    e = thrown(() -> safe_ratio(1.0, 0.0))
    r = safe_ratio(6.0, 3.0)
    (cxx_error(e, "safe_ratio") && r == 2.0, "e=$(repr(e)) r=$r")
end

probe("throw_ptr") do
    e = thrown(() -> make_square(-1.0))
    p = make_square(2.0)
    try
        area = Square_area(p)
        (cxx_error(e, "make_square") && area == 4.0, "e=$(repr(e)) area=$area")
    finally
        aot_free_shape(Ptr{Cvoid}(p))
    end
end

probe("throw_sret") do
    e = thrown(() -> vec2_unit(Vec2(0.0, 0.0)))
    u = vec2_unit(Vec2(3.0, 4.0))
    (cxx_error(e, "vec2_unit") && isapprox(u.x, 0.6) && isapprox(u.y, 0.8),
     "e=$(repr(e)) u=$u")
end

probe("throw_non_std") do
    e = thrown(() -> throw_int(Int32(7)))
    r = throw_int(Int32(0))
    (cxx_error(e, "unknown") && r == 0, "e=$(repr(e)) r=$r")
end

probe("throw_void") do
    a = aot_new_accumulator(Int32(3))
    try
        e = thrown(() -> Accumulator_set_total(a, Int32(-7)))
        unchanged = Accumulator_total(a)
        Accumulator_set_total(a, Int32(100))
        after = Accumulator_total(a)
        (cxx_error(e, "set_total", "-7") && unchanged == 3 && after == 100,
         "e=$(repr(e)) total=$unchanged after=$after")
    finally
        aot_free_accumulator(a)
    end
end

# Nothing left pending, and still no JIT after every throw.
probe("clean_after_throws") do
    pending = RepliBuild.MLIRNative.has_pending_exception()
    (!pending && jit_untouched(), "pending=$pending jit_untouched=$(jit_untouched())")
end

println("PROBE_DONE")
flush(stdout)
