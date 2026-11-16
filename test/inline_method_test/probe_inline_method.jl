# Subprocess probe for test_inline_methods_live.jl: calls the member functions
# the build's forced unit defined, on a Body the library's own constructor
# built, and checks each against the library's own readers or the member's
# arithmetic. In a child so a wrong binding cannot take the suite down. Prints
# one "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_inline_method.jl <wrapper_path>

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

# A Body of mass 2 on caller storage, built by the constructor the library defines.
function with_body(f)
    mem = Libc.malloc(M.struct_size(:Body))
    try
        M.im_Body_Body(mem, 2.0f0)
        f(mem)
    finally
        Libc.free(mem)
    end
end

# The Bullet pair: setLinearVelocity then getLinearVelocity round-trips, and
# the library's own reader sees what the forced setter wrote.
probe("set_then_get_round_trips") do p
    with_body() do b
        M.im_Body_setVelocity(b, Ref(M.Vec3(1f0, 2f0, 3f0)))
        (x = M.im_body_vel_x(b)) == 1f0 || push!(p, "library reads vel.x = $x")
        v = unsafe_load(Ptr{M.Vec3}(M.im_Body_getVelocity(b)))
        v == M.Vec3(1f0, 2f0, 3f0) || push!(p, "getVelocity = $v")
    end
end

# applyCentralImpulse on a 2 kg body changes velocity by impulse / 2; the
# member calls the always_inline invMass().
probe("impulse_over_mass") do p
    with_body() do b
        M.im_Body_setVelocity(b, Ref(M.Vec3(1f0, 0f0, 0f0)))
        M.im_Body_applyImpulse(b, Ref(M.Vec3(4f0, 0f0, 0f0)))
        (x = M.im_body_vel_x(b)) == 3f0 || push!(p, "vel.x after impulse 4 on mass 2 = $x")
        (i = M.im_Body_invMass(b)) == 0.5f0 || push!(p, "invMass = $i")
        (m = M.im_Body_mass(b)) == 2f0 || push!(p, "mass = $m")
    end
end

# bool in, bool out.
probe("bool_setter_and_getter") do p
    with_body() do b
        M.im_Body_isActive(b) == true || push!(p, "a new Body is not active")
        M.im_Body_setActive(b, false)
        M.im_Body_isActive(b) == false || push!(p, "still active after setActive(false)")
    end
end

# Two overloads of one name that differ in their parameters are two methods.
probe("overloads_by_parameters") do p
    with_body() do b
        M.im_Body_scale(b, 5)
        (s = M.im_Body_getScale(b)) == 5f0 || push!(p, "scale(int): $s")
        M.im_Body_scale(b, 1.5f0, 0.25f0)
        (s = M.im_Body_getScale(b)) == 1.75f0 || push!(p, "scale(float, float): $s")
    end
end

# A static member takes no receiver; an operator is bound under its name.
probe("static_and_operator") do p
    (d = M.im_Body_defaultMass()) == 2f0 || push!(p, "defaultMass = $d")
    with_body() do b
        M.im_Body_operatorplusassign(b, Ref(M.Vec3(1f0, 2f0, 3f0)))
        M.im_Body_operatorplusassign(b, Ref(M.Vec3(1f0, 2f0, 3f0)))
        pos = unsafe_load(Ptr{M.Vec3}(M.im_Body_position(b)))
        pos == M.Vec3(2f0, 4f0, 6f0) || push!(p, "position after two += : $pos")
    end
end

# What the library defined itself is bound as before, and what the unit must
# not have produced: the member declared with no body, the protected and the
# private one, the member template, and a class template specialization's
# members (those are `[types] templates`' to instantiate).
probe("defined_untouched_and_refused_unbound") do p
    with_body() do b
        (k = M.im_Body_outOfLine(b, 4)) == 12 || push!(p, "outOfLine(4) = $k")
        (k = M.im_body_kind(b)) == 7 || push!(p, "virtual kind through the library = $k")
    end
    for name in (:im_Body_declaredOnly, :im_Body_protectedInline, :im_Body_privateInline, :im_Body_as,
                 :im_Arr_int_size, :im_Arr_int_at)
        isdefined(M, name) && push!(p, "$name is bound")
    end
end

println("PROBE_DONE")
