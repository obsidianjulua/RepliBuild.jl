# Subprocess probe for test_simd_union_live.jl: passes and returns records
# that hold a 16-byte SIMD vector in a union, by value, through the wrapper's
# thunks, and checks every lane against the library's own arithmetic. In a
# child so a wrong register class cannot take the suite down. Prints one
# "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_simd_union.jl <wrapper_path>

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

# A record's 16 bytes from four lanes, and back. The Julia struct is the C
# layout, so its bytes are the lanes'.
bytes16(lanes::NTuple{4,T}) where {T} = reinterpret(NTuple{16,UInt8}, lanes)
function record(::Type{R}, lanes::NTuple{4,T}) where {R,T}
    r = Ref{R}()
    GC.@preserve r unsafe_store!(Ptr{NTuple{4,T}}(Base.unsafe_convert(Ptr{R}, r)), lanes)
    return r[]
end
function lanes(::Type{T}, x::R) where {T,R}
    r = Ref(x)
    return GC.@preserve r unsafe_load(Ptr{NTuple{4,T}}(Base.unsafe_convert(Ptr{R}, r)))
end
vf(a, b, c, d) = record(M.Vec4f, Float32.((a, b, c, d)))
vu(a, b, c, d) = record(M.Vec4u, UInt32.((a, b, c, d)))

# Two SSE eightbytes in, two out: every lane of both arguments matters.
probe("float_union_by_value_both_ways") do p
    sizeof(M.Vec4f) == 16 || push!(p, "sizeof(Vec4f) = $(sizeof(M.Vec4f))")
    r = lanes(Float32, M.v_add(vf(1, 2, 3, 4), vf(10, 20, 30, 40)))
    r == (11f0, 22f0, 33f0, 44f0) || push!(p, "v_add = $r")
    for i in 0:3
        (x = M.v_lane(vf(5, 6, 7, 8), i)) == Float32(5 + i) || push!(p, "v_lane($i) = $x")
    end
    m = lanes(Float32, M.v_make(1.5f0, -2.5f0, 3.25f0, 1f6))
    m == (1.5f0, -2.5f0, 3.25f0, 1f6) || push!(p, "v_make = $m")
end

# Beside an int, a double and a float: the records take four of the eight
# SSE registers between other SSE arguments, and the int a general one.
probe("float_union_among_other_arguments") do p
    x = M.v_mixed(100, vf(1, 0, 0, 0), 20.0, vf(0, 0, 0, 3), 0.5f0)
    x == 124.5f0 || push!(p, "v_mixed = $x")
end

# The integer union is two GENERAL registers, not SSE.
probe("integer_union_by_value") do p
    r = lanes(UInt32, M.u_or(vu(0x1, 0x20, 0x300, 0x4000), vu(0x10000, 0x200000, 0x3000000, 0x40000000)))
    r == (0x00010001, 0x00200020, 0x03000300, 0x40004000) || push!(p, "u_or = $r")
    for i in 0:3
        (x = M.u_lane(vu(7, 8, 9, 0xdeadbeef), i)) == (7, 8, 9, 0xdeadbeef)[i+1] || push!(p, "u_lane($i) = $x")
    end
end

# The union one record down (Jolt's Quat holds a Vec4).
probe("nested_record_by_value") do p
    q = record(M.Quatf, Float32.((1, 2, 3, 4)))
    r = lanes(Float32, M.q_conj(q))
    r == (-1f0, -2f0, -3f0, 4f0) || push!(p, "q_conj = $r")
    (w = M.q_w(q)) == 4f0 || push!(p, "q_w = $w")
end

# Member functions: `this` in a general register beside the record's SSE pair.
probe("member_functions") do p
    b = Ref(M.Body(vf(0, 0, 0, 0)))
    M.Body_setPos(b, vf(1, 2, 3, 4))
    g = lanes(Float32, M.Body_getPos(b))
    g == (1f0, 2f0, 3f0, 4f0) || push!(p, "getPos after setPos = $g")
    s = lanes(Float32, M.Body_scaled(b, vf(2, 2, 2, 2), 0.5f0))
    s == (1f0, 2f0, 3f0, 4f0) || push!(p, "scaled = $s")
end

# A vector ALONE in its record is one SSE register, which nothing states yet:
# refused by name, never called wrong.
probe("lone_vector_still_refused") do p
    err = try M.lone_x(record(M.Lone, Float32.((1, 2, 3, 4)))); nothing catch e; e end
    (err isa ErrorException && occursin("ABI Safety Trap", err.msg) && occursin("one SSE register", err.msg)) ||
        push!(p, "lone_x did not trap: $(err)")
end

println("PROBE_DONE")
