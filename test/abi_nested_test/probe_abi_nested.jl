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

# The SysV XMM bug this test exists to catch: a 16-byte all-float struct
# emitted as NTuple{16,UInt8}. On Win64 that blob can still round-trip
# because the aggregate is sret/memory, so this check is the one that
# cannot pass by accident on either ABI.
let blob = :_data in fieldnames(XForm) || XForm <: NTuple
    println("PROBE xform_not_blob: ", blob ? "FAIL" : "PASS",
            " typeof=", XForm, " fields=", fieldnames(XForm))
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

# Packed float struct: must stay opaque (a byte blob), and its by-value crossing
# is per target. SysV: 9 bytes with a float at offset 1 is MEMORY class while
# the blob classifies INTEGER, and the float puts it in the SSE trap besides, so
# the argument must be refused loudly. Win64 passes any 9-byte record by
# reference to a copy and returns it through a hidden pointer, the blob
# included, so both directions must work. The target is the host's, decided
# here independently of the generator's own `_c_record_abi`.
let opaque = :_data in fieldnames(PackedFV)
    s = try
        make_packedfv()   # SysV: explicit sret (MEMORY class); Win64: hidden pointer by size
    catch e
        nothing
    end
    if Sys.iswindows()
        v = s === nothing ? nothing : try packedfv_sum(s) catch e; sprint(showerror, e) end
        ok = opaque && v isa AbstractFloat && approx(v, 11.0)
        println("PROBE packed_float_byvalue: ", ok ? "PASS (Win64: crosses by reference)" : "FAIL",
                " opaque=", opaque, " s=", s, " sum=", v)
    elseif s === nothing
        println("PROBE packed_float_byvalue: ", opaque ? "PASS (return refused loudly)" : "FAIL (not opaque)")
    else
        refused = try
            packedfv_sum(s)
            false
        catch e
            occursin("ABI", sprint(showerror, e)) || occursin("opaque", sprint(showerror, e))
        end
        println("PROBE packed_float_byvalue: ", !opaque ? "FAIL (not opaque)" :
                refused ? "PASS (arg refused loudly)" : "FAIL (silent crossing)")
    end
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
