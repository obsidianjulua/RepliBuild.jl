# Subprocess probe for the convenience-overload ownership guard.
# Asserts the struct-by-value footgun is gone: no f(x::Struct) overload may
# exist for a Ptr{Struct}-taking C function, because that overload passed
# Ref(local copy) to the ccall — undefined behavior whenever the callee frees,
# mutates-and-retains, or stores the pointer (crash-proven: the generated
# cJSON_Delete(::cJSON) aborted with a glibc double-free). The Vector
# convenience path for input arrays must survive, with Cstring returns
# aligned to the base wrapper's String policy.
# Prints "PROBE <name>: PASS/FAIL" lines; the parent test parses them.
# Run: julia --project=<RepliBuild> probe_convenience.jl <wrapper_path>

wrapper = ARGS[1]
include(wrapper)
using .Gripkit

# No by-value overload on Ptr{Grip}-taking functions (ownership unknowable)
let bad = [String(f) for f in (:grip_free, :grip_value)
           if any(m -> Gripkit.Grip in m.sig.parameters, methods(getfield(Gripkit, f)))]
    println("PROBE no_byvalue_overload: ", isempty(bad) ? "PASS" : "FAIL byvalue methods on: $(join(bad, ", "))")
end

# Handing a by-value struct to the free()-taking function must refuse loudly
# (MethodError — no Struct→Ptr conversion), never reach free(). Pre-fix this
# was a glibc double-free abort, which kills this subprocess.
p = Gripkit.grip_new(9, 1.0)
g = unsafe_load(p)
refused = try
    Gripkit.grip_free(g)
    false
catch e
    true
end
println("PROBE byvalue_call_refused: ", refused ? "PASS" : "FAIL (by-value call reached free())")
Gripkit.grip_free(p)

# The base pointer path is the supported lifecycle and must round-trip
p2 = Gripkit.grip_new(7, 2.5)
let ok = Gripkit.grip_value(p2) == 7
    println("PROBE pointer_lifecycle: ", ok ? "PASS" : "FAIL value=$(Gripkit.grip_value(p2))")
end
Gripkit.grip_free(p2)

# Survivor: input-array params still get the Vector convenience overload
v = [1.0, 2.0, 3.0]
let has_method = any(m -> m.sig == Tuple{typeof(Gripkit.sum_xs), Vector{Float64}, Cint},
                     methods(Gripkit.sum_xs)),
    via_vec = Gripkit.sum_xs(v, Cint(3)),
    via_base = Gripkit.sum_xs(v, 3)
    ok = has_method && via_vec == 6.0 && via_base == 6.0
    println("PROBE vector_convenience: ", ok ? "PASS" : "FAIL method=$(has_method) vec=$(via_vec) base=$(via_base)")
end

# Survivor alignment: the Vector overload of a char*-returning function must
# return String (base wrapper policy), not a raw Cstring
let via_vec = Gripkit.describe_values(v, Cint(3)),
    via_base = Gripkit.describe_values(v, 3)
    ok = via_vec isa String && via_base isa String && occursin("sum=6.0", via_vec) && via_vec == via_base
    println("PROBE cstring_policy_aligned: ", ok ? "PASS" : "FAIL vec=$(repr(via_vec)) base=$(repr(via_base))")
end

println("PROBE_DONE")
