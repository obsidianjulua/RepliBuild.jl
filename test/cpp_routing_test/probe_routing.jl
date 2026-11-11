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

# Byte regions by value (abi_fuzz X2): built from their bytes, since a union
# and a bitfield record are a `mutable struct` of their C bytes in Julia.
probe("u4") do p
    (s = u4_sum(U4((0x01, 0x02, 0x03, 0x04)))) == 1 + 4 + 9 + 16 || push!(p, "u4_sum = $s")
end

probe("u12") do p
    le(x) = ntuple(k -> UInt8((x >> (8 * (k - 1))) & 0xff), 4)
    v = U12((le(5)..., le(6)..., le(7)...))
    (s = u12_sum(v)) == 5 + 12 + 21 || push!(p, "u12_sum = $s")
end

probe("bf8") do p
    # a = 5 in bits 0-2, b = 9 in bits 3-9, n = 3 at byte 4.
    v = BF8((UInt8(5 | (9 << 3)), 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00))
    (s = bf8_sum(v)) == 5 + 90 + 300 || push!(p, "bf8_sum = $s")
end

# Byte regions returned by value: X1 allocates the record on the Julia side and
# X2 makes the thunk receive it the way C returns it (U4 and BF8 in a register,
# U12 in two on SysV and through a hidden pointer on Win64). Each goes back in
# by value too. The one field of such a `mutable struct` is its C bytes.
rbytes(x) = collect(getfield(x, 1))

probe("u4ret") do p
    r = make_u4(Int32(0x04030201))
    (b = rbytes(r)) == [0x01, 0x02, 0x03, 0x04] || push!(p, "make_u4 bytes = $b")
    (s = u4_sum(r)) == 1 + 4 + 9 + 16 || push!(p, "u4_sum(make_u4) = $s")
end

probe("u12ret") do p
    r = make_u12(Int32(5))
    b = rbytes(r)
    Tuple(field(Int32, b, o) for o in (0, 4, 8)) == (5, 6, 7) || push!(p, "make_u12 bytes = $b")
    (s = u12_sum(r)) == 5 + 12 + 21 || push!(p, "u12_sum(make_u12) = $s")
end

probe("bf8ret") do p
    # a = 5 and b = 9 in bits 0-9 (the rest of that unit is padding), n at byte 4.
    r = make_bf8(Int32(3))
    b = rbytes(r)
    ((UInt16(b[1]) | UInt16(b[2]) << 8) & 0x03ff, field(Int32, b, 4)) == (UInt16(5 | 9 << 3), 3) ||
        push!(p, "make_bf8 bytes = $b")
    (s = bf8_sum(r)) == 5 + 90 + 300 || push!(p, "bf8_sum(make_bf8) = $s")
end

# A tiled record holding a bitfield (abi_fuzz X9): m1 in bytes 0-3, m2's 7 bits
# in byte 4, whatever the record's size (12 on MinGW, 5 on Itanium).
probe("pb12") do p
    b = zeros(UInt8, sizeof(PB12))
    b[1:4] .= reinterpret(UInt8, [Int32(3)])
    b[5] = 0x09
    (s = pb12_sum(PB12(Tuple(b)))) == 3 + 9000 || push!(p, "pb12_sum = $s")
    r = make_pb12(Int32(5))
    rb = rbytes(r)
    (field(Int32, rb, 0), rb[5] & 0x7f) == (5, 0x0b) || push!(p, "make_pb12 bytes = $rb")
    (s = pb12_sum(r)) == 5 + 11000 || push!(p, "pb12_sum(make_pb12) = $s")
end

# A `#pragma pack` record holding a pointer-aligned one (abi_fuzz X8): `r` at 4.
probe("pk28") do p
    r = make_pk28(Int32(11))
    b = bytes(r)
    sizeof(r) == 28 || push!(p, "sizeof = $(sizeof(r))")
    (b[1], field(Int32, b, 4), field(Int32, b, 12), field(UInt64, b, 20)) == (0x41, 11, 7, 0x1234) ||
        push!(p, "make_pk28 bytes = $b")
    (s = pk28_sum(r)) == 0x41 + 22 + 21 + 0x1234 || push!(p, "pk28_sum = $s")
end

# Over-aligned through a member (2026-11-10): MA32 and NestMA are 32-aligned
# by a member's `alignas` alone. Passed after Big24, each lives on the stack at
# 32; a ccall put it at 24, where Big24's last eightbyte and padding are.
const PAD_SUM = 1 + 2 * 2 + 3 * 3
ma32_want(x) = PAD_SUM + x + 10 * (x + 1) + 100 * (x + 2) + 1000 * (4x)

probe("big24") do p
    (s = big24_sum(Big24(1, 2, 3))) == PAD_SUM || push!(p, "big24_sum = $s")
end

probe("ma32") do p
    r = make_ma32(1.5)
    b = bytes(r)
    sizeof(r) == 32 || push!(p, "sizeof = $(sizeof(r))")
    Tuple(field(Float64, b, o) for o in (0, 8, 16, 24)) == (1.5, 2.5, 3.5, 6.0) ||
        push!(p, "make_ma32 bytes = $b")
    (s = ma32_sum(Big24(1, 2, 3), r)) == ma32_want(1.5) || push!(p, "ma32_sum = $s")
end

probe("nestma") do p
    r = make_nestma(2.0)
    b = bytes(r)
    sizeof(r) == 32 || push!(p, "sizeof = $(sizeof(r))")
    Tuple(field(Float64, b, o) for o in (0, 8, 16, 24)) == (2.0, 3.0, 4.0, 8.0) ||
        push!(p, "make_nestma bytes = $b")
    (s = nestma_sum(Big24(1, 2, 3), r)) == ma32_want(2.0) || push!(p, "nestma_sum = $s")
end

# 16-aligned by its `__int128`, so Tier 2 too: returned, then passed back
# behind Big24. Low half k * 3, high half k + 1, then k.
probe("i128") do p
    r = make_i128(5)
    b = bytes(r)
    sizeof(r) == 32 || push!(p, "sizeof = $(sizeof(r))")
    (field(UInt64, b, 0), field(Int64, b, 8), field(Int64, b, 16)) == (15, 6, 5) ||
        push!(p, "make_i128 bytes = $b")
    (s = i128_sum(Big24(1, 2, 3), r)) == PAD_SUM + 60 + 1500 + 5000 || push!(p, "i128_sum = $s")
end

# Scalars at their C width (2026-11-03): a Julia integer narrower than the C
# parameter used to be boxed as is, and the thunk read past it.
probe("scalar_widths") do p
    want = -1 + 200_000 + 3_000_000 + 2_000_000_000
    (s = scalar_widths(Int8(-1), UInt8(200), Int16(3), Int8(2))) == want || push!(p, "narrow arguments = $s")
    (s = scalar_widths(Int32(-1), UInt32(200), Csize_t(3), KIND_B)) == want || push!(p, "exact arguments = $s")
    (s = scalar_widths(Int64(-1), Int64(200), Int64(3), Int64(1))) == want - 1_000_000_000 || push!(p, "wide arguments = $s")
end

println("PROBE_DONE")
