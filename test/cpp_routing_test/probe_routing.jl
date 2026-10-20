# Subprocess probe for test_cpp_record_routing.jl: calls every function in
# cpp_routing_test, checking each returned record's bytes at their C offsets and
# that each record passed back in reads correctly (every *_sum weights each
# member differently). Values are read and built as bytes, so the probe does not
# care whether a record's Julia type has fields or is a byte blob. Prints one
# "PROBE <function>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_routing.jl <wrapper_path>

wrapper = ARGS[1]
include(wrapper)
using .CppRouting

bytes(x) = (r = Ref(x); GC.@preserve r copy(unsafe_wrap(Array, Ptr{UInt8}(pointer_from_objref(r)), sizeof(x))))
field(T, b::Vector{UInt8}, off) = GC.@preserve b unsafe_load(Ptr{T}(pointer(b) + off))

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

probe("cs3") do p
    r = make_cs3(7)
    b = bytes(r)
    sizeof(r) == 8 || push!(p, "sizeof = $(sizeof(r))")
    (b[1], field(Int16, b, 2), field(Int16, b, 4), field(Int16, b, 6)) == (0x08, 14, 21, -7) ||
        push!(p, "make_cs3 bytes = $b")
    (s = cs3_sum(r)) == 8 + 140 + 2100 - 7000 || push!(p, "cs3_sum = $s")
end

probe("mass") do p
    r = make_mass(1.5f0)
    b = bytes(r)
    sizeof(r) == 16 || push!(p, "sizeof = $(sizeof(r))")
    Tuple(field(Float32, b, o) for o in (0, 4, 8, 12)) == (1.5f0, 2.5f0, 3.5f0, 6.0f0) ||
        push!(p, "make_mass bytes = $b")
    (s = mass_sum(r)) == 1.5f0 + 25f0 + 350f0 + 6000f0 || push!(p, "mass_sum = $s")
end

probe("enumrec") do p
    r = make_enumrec(9)
    b = bytes(r)
    sizeof(r) == 4 || push!(p, "sizeof = $(sizeof(r))")
    (b[1], field(Int16, b, 2)) == (0x09, 2) || push!(p, "make_enumrec bytes = $b")
    (s = enumrec_sum(r)) == 9 + 200 || push!(p, "enumrec_sum = $s")
end

probe("pk5") do p
    r = make_pk5(Int32(0x01020304))
    b = bytes(r)
    sizeof(r) == 5 || push!(p, "sizeof = $(sizeof(r))")
    (b[1], field(Int32, b, 1)) == (0x21, Int32(0x01020304)) || push!(p, "make_pk5 bytes = $b")
    (s = pk5_sum(r)) == 0x21 + 2 * 0x01020304 || push!(p, "pk5_sum = $s")
end

probe("top6") do p
    r = make_top6(Int16(1000))
    b = bytes(r)
    sizeof(r) == 6 || push!(p, "sizeof = $(sizeof(r))")
    (b[1], field(Int16, b, 1), b[4], b[5], b[6]) == (0x31, 1000, 0x04, 0x05, 0x06) ||
        push!(p, "make_top6 bytes = $b")
    (s = top6_sum(r)) == 0x31 + 10000 + 1500 || push!(p, "top6_sum = $s")
end

println("PROBE_DONE")
