# Subprocess probe for test_default_args_live.jl: makes the calls that leave
# default arguments out and checks, through the library's own readers, that
# the omitted parameters took the values the header declares. In a child so a
# wrong binding cannot take the suite down. Prints one
# "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_default_arg.jl <wrapper_path>

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

function with_storage(f, sym::Symbol)
    mem = Libc.malloc(M.struct_size(sym))
    try
        f(mem)
    finally
        Libc.free(mem)
    end
end
with_hinge(f) = with_storage(:Hinge) do h
    M.da_Hinge_Hinge(h, Ref(M.Vec(3f0, 4f0)), Ref(M.Vec(5f0, 6f0)))
    f(h)
end

# An out-of-line constructor called without its defaulted argument.
probe("constructor_default") do p
    with_storage(:Hinge) do h
        M.da_Hinge_Hinge(h, Ref(M.Vec(3f0, 4f0)))
        (M.da_Hinge_ax(h), M.da_Hinge_bx(h), M.da_Hinge_by(h)) == (3f0, 1f0, 2f0) ||
            push!(p, "a.x, b = $((M.da_Hinge_ax(h), M.da_Hinge_bx(h), M.da_Hinge_by(h)))")
    end
end

# A header-only constructor with a default: forced, and its short call too
# (btDefaultMotionState(start)).
probe("inline_constructor_default") do p
    with_storage(:State) do s
        M.da_State_State(s, Ref(M.Vec(9f0, 1f0)))
        (M.da_state_start_x(s), M.da_state_offset_y(s)) == (9f0, 8f0) ||
            push!(p, "start.x, offset.y = $((M.da_state_start_x(s), M.da_state_offset_y(s)))")
        M.da_State_State(s, Ref(M.Vec(9f0, 1f0)), Ref(M.Vec(0f0, 2f0)))
        M.da_state_offset_y(s) == 2f0 || push!(p, "the full call: offset.y = $(M.da_state_offset_y(s))")
    end
end

# setLimit with two, three, four and five arguments (btHingeConstraint's).
probe("method_trailing_defaults") do p
    with_hinge() do h
        got() = (M.da_Hinge_lo(h), M.da_Hinge_hi(h), M.da_Hinge_soft(h), M.da_Hinge_bias(h), M.da_Hinge_relax(h))
        M.da_Hinge_setLimit(h, -1f0, 1f0)
        got() == (-1f0, 1f0, 0.9f0, 0.3f0, 1f0) || push!(p, "setLimit(lo, hi): $(got())")
        M.da_Hinge_setLimit(h, -2f0, 2f0, 0.5f0)
        got() == (-2f0, 2f0, 0.5f0, 0.3f0, 1f0) || push!(p, "setLimit(lo, hi, soft): $(got())")
        M.da_Hinge_setLimit(h, -3f0, 3f0, 0.6f0, 0.7f0)
        got() == (-3f0, 3f0, 0.6f0, 0.7f0, 1f0) || push!(p, "setLimit(lo, hi, soft, bias): $(got())")
        M.da_Hinge_setLimit(h, -4f0, 4f0, 0.1f0, 0.2f0, 0.3f0)
        got() == (-4f0, 4f0, 0.1f0, 0.2f0, 0.3f0) || push!(p, "the full call: $(got())")
    end
end

# An overload set: a Float32 takes the member with defaults, a Float64 the
# member that takes a double, as C++ resolves them.
probe("overload_beside_defaults") do p
    with_hinge() do h
        (r = M.da_Hinge_step(h, 3f0)) == 128 || push!(p, "step(3f0) = $r")              # 1*100 + 25 + 3
        (r = M.da_Hinge_step(h, 3f0, 2)) == 228 || push!(p, "step(3f0, 2) = $r")
        (r = M.da_Hinge_step(h, 3f0, 2, 0.5f0)) == 253 || push!(p, "step(3f0, 2, 0.5f0) = $r")
        (r = M.da_Hinge_step(h, 3.0)) == -3 || push!(p, "step(3.0) = $r")
    end
end

# A record returned by value through the short call.
probe("by_value_return") do p
    with_hinge() do h
        v = M.da_Hinge_scaled(h, 2f0)
        (v.x, v.y) == (6.5f0, 8.5f0) || push!(p, "scaled(2) = $v")
        w = M.da_Hinge_scaled(h, 2f0, 1f0)
        (w.x, w.y) == (7f0, 9f0) || push!(p, "scaled(2, 1) = $w")
    end
end

# What must NOT exist: a short call where the header has no default, the
# one-argument count (that call is `count(int)`'s own), a static member's
# short call, a protected member's.
probe("no_default_no_binding") do p
    with_hinge() do h
        arities(f) = sort([length(m.sig.parameters) - 1 for m in methods(f)])
        arities(M.da_Hinge_none) == [3] || push!(p, "none: arities $(arities(M.da_Hinge_none))")
        arities(M.da_Hinge_count) == [2, 3] || push!(p, "count: arities $(arities(M.da_Hinge_count))")
        (r = M.da_Hinge_count(h, 4)) == -4 || push!(p, "count(4) = $r, not count(int)'s")
        (r = M.da_Hinge_count(h, 4, 7)) == 47 || push!(p, "count(4, 7) = $r")
        arities(M.da_Hinge_make) == [2] || push!(p, "make: arities $(arities(M.da_Hinge_make))")
        arities(M.da_Hinge_prot) == [3] || push!(p, "prot: arities $(arities(M.da_Hinge_prot))")
    end
end

println("PROBE_DONE")
