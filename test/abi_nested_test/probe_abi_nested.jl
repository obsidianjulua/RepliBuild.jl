# Subprocess probe for the nested-struct ABI trace test.
# Loads the generated wrapper and exercises every by-value crossing class.
# Prints "PROBE <name>: PASS/FAIL ..." lines; the parent test parses them.
# Run: julia --project=<RepliBuild> probe_abi_nested.jl <wrapper_path>

wrapper = ARGS[1]
include(wrapper)
using .AbiNested

approx(a, b) = isapprox(a, b; atol=1e-4)

# Nested structs must come out field-resolved, not byte blobs
let ok = fieldnames(XForm) == (:p, :q) && fieldnames(Mass) == (:m, :c, :i)
    println("PROBE fields_resolved: ", ok ? "PASS" : "FAIL",
            " XForm=", fieldnames(XForm), " Mass=", fieldnames(Mass))
end

# 16B all-float nested struct RETURNED by value (SSE,SSE)
t = make_xform(1.5f0, 2.5f0, 3.5f0, 4.5f0)
let ok = approx(t.p.x, 1.5) && approx(t.p.y, 2.5) && approx(t.q.x, 3.5) && approx(t.q.y, 4.5)
    println("PROBE xform_return: ", ok ? "PASS" : "FAIL", " t=", t)
end

# 16B all-float nested struct PASSED by value
let v = xform_p(t)
    println("PROBE xform_byvalue_arg: ", (approx(v.x, 1.5) && approx(v.y, 2.5)) ? "PASS" : "FAIL", " v=", v)
end

# 16B float/nested/float sandwich
md = make_mass(1.0f0, 2.0f0, 3.0f0, 4.0f0)
let ok = approx(mass_total(md), 10.0)
    println("PROBE mass_roundtrip: ", ok ? "PASS" : "FAIL", " total=", mass_total(md))
end

# 12B nested float (odd size, still register class)
d = make_disc(0.0f0, 0.0f0, 2.0f0)
let ok = approx(disc_area(d), 4 * 3.14159265)
    println("PROBE disc_roundtrip: ", ok ? "PASS" : "FAIL", " area=", disc_area(d))
end

# >16B array-of-struct member: MEMORY class both directions (control)
p = make_poly()
let ok = approx(poly_sum(p), 21.0)
    println("PROBE poly_memory_class: ", ok ? "PASS" : "FAIL", " sum=", poly_sum(p))
end

# Nested int struct: INTEGER class (control — safe even as blob)
ni = make_nestint()
let ok = nestint_sum(ni) == 1000
    println("PROBE nestint_roundtrip: ", ok ? "PASS" : "FAIL", " sum=", nestint_sum(ni))
end

# Packed float struct: must stay opaque AND refuse by-value crossing loudly
s = try
    make_packedfv()   # return side: sret branch handles MEMORY-class packed returns
catch e
    nothing
end
if s === nothing
    println("PROBE packed_byvalue_guard: PASS (return refused loudly)")
else
    refused = try
        packedfv_sum(s)
        false
    catch e
        occursin("ABI", sprint(showerror, e)) || occursin("opaque", sprint(showerror, e))
    end
    println("PROBE packed_byvalue_guard: ", refused ? "PASS (arg refused loudly)" : "FAIL (silent crossing)")
end

# Float param ergonomics: Int and Float64 into a Cfloat slot
let ok = try
        approx(scale_vec(Vec2(1f0, 2f0), 2), 6.0) &&
        approx(scale_vec(Vec2(1f0, 2f0), 2.5), 7.5)
    catch e
        false
    end
    println("PROBE float_param_loosening: ", ok ? "PASS" : "FAIL")
end

# with(): immutable field-update helper for def-struct workflows
let ok = try
        t2 = AbiNested.with(t; q = Vec2(9f0, 9f0))
        approx(t2.q.x, 9.0) && approx(t2.p.x, 1.5)
    catch e
        false
    end
    println("PROBE with_helper: ", ok ? "PASS" : "FAIL")
end

println("PROBE_DONE")
