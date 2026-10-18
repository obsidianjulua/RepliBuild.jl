# Subprocess probe for test_c_sret_abi.jl: one call per function in
# c_sret_test, checking every argument member the callee read (it records them
# in c1_rec[]) and every member of the record it returns. Prints one
# "PROBE <function>: PASS|FAIL …" line each, then PROBE_DONE. A wrong return
# class is a crash or a plausible wrong value, so this runs in a child.
# Run: julia --project=<RepliBuild> probe_c_sret.jl <wrapper_path>

wrapper = ARGS[1]
include(wrapper)
using .CSret
import RepliBuild

# Which answer this host should give where the two conventions differ.
const ABI = RepliBuild.Wrapper._c_record_abi()

const REC = cglobal((:c1_rec, CSret.LIBRARY_PATH), UInt64)
const REC_N = cglobal((:c1_rec_n, CSret.LIBRARY_PATH), Cint)
rec(i) = unsafe_load(REC, i + 1)
function reset_rec()
    for i in 1:16
        unsafe_store!(REC, 0xdeadbeefdeadbeef, i)
    end
    unsafe_store!(REC_N, Cint(-1))
end
# What the callee's REC() stores for a signed value: widened by sign.
s64(x) = reinterpret(UInt64, Int64(x))
# A field of a byte blob, read at its C offset.
field(T, bytes::NTuple, off) = let r = Ref(bytes)
    GC.@preserve r unsafe_load(Ptr{T}(pointer_from_objref(r) + off))
end
hx(x) = "0x" * string(x; base = 16)

# Compare what the callee read with what was passed: `want` is i => value.
function args_read(p, n, want)
    got_n = unsafe_load(REC_N)
    got_n == n || push!(p, "c1_rec_n = $got_n want $n (the callee did not run as declared)")
    for (i, w) in want
        rec(i) == w || push!(p, "arg $i read $(hx(rec(i))) want $(hx(w))")
    end
end

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

# On SysV a crossing the blob cannot reproduce must refuse before the call.
function refused(p, f)
    try
        f()
        push!(p, "called; SysV must refuse this crossing")
    catch e
        occursin("ABI Safety Trap", sprint(showerror, e)) ||
            push!(p, "threw something other than the ABI trap: " * first(sprint(showerror, e), 200))
    end
end

probe("f4") do p                            # C1, abi_fuzz seed 36
    reset_rec()
    r = f4(R5(Int32(0x79aa7901)))
    args_read(p, 1, [0 => s64(0x79aa7901)])
    r.m1 == 0x01 || push!(p, "ret.m1 = $(r.m1)")
    r.m2 == (Int16(22018), Int16(29443), Int16(23812)) || push!(p, "ret.m2 = $(r.m2)")
end

probe("f2") do p                            # C1, abi_fuzz seed 3
    reset_rec()
    t = CSret.with(T4(); m1 = 0x05, m2 = (Int16(-3), Int16(300), Int16(-30000)))
    q = CSret.with(R4(); m1 = Int32(-123456789), m2 = 0x0123456789abcdef)
    r = f2(t, q)
    args_read(p, 6, [0 => s64(5), 1 => s64(-3), 2 => s64(300), 3 => s64(-30000),
                     4 => s64(-123456789), 5 => UInt64(0x0123456789abcdef)])
    r.m1 == 0x01 || push!(p, "ret.m1 = $(r.m1)")
    r.m2 == (Int16(11778), Int16(-26365), Int16(11268)) || push!(p, "ret.m2 = $(r.m2)")
end

probe("make_ic12") do p                     # 16 B: Win64 sret, SysV INTEGER,INTEGER
    reset_rec()
    r = make_ic12(1000, 0x41)
    args_read(p, 2, [0 => s64(1000), 1 => s64(0x41)])
    r.n == 3000 || push!(p, "ret.n = $(r.n)")
    r.s == ntuple(i -> UInt8(0x40 + i), 12) || push!(p, "ret.s = $(r.s)")
end

probe("ic12_sum") do p                      # the same record as an argument
    reset_rec()
    s = ic12_sum(IC12(Int32(7), ntuple(i -> UInt8(i), 12)))
    s == 7 + 78 || push!(p, "ic12_sum = $s want 85")
end

probe("make_p4") do p                       # packed 4 B: Win64 RAX, SysV MEMORY
    reset_rec()
    r = make_p4(Int16(0x1234))
    args_read(p, 1, [0 => s64(0x1234)])
    d = getfield(r, :_data)
    (d[1], field(Int16, d, 1), d[4]) == (0x11, Int16(0x1235), 0x22) ||
        push!(p, "ret bytes = $d")
end

probe("make_p8") do p                       # packed 8 B: Win64 RAX, SysV MEMORY
    reset_rec()
    r = make_p8(Int32(0x01020304))
    args_read(p, 1, [0 => s64(0x01020304)])
    d = getfield(r, :_data)
    (d[1], field(Int32, d, 1), d[6:8]) == (0x33, Int32(0x01020304 ⊻ 0x5a5a5a5a), (0x07, 0x08, 0x09)) ||
        push!(p, "ret bytes = $d")
end

probe("take_p8") do p                       # Win64: i64 on both views; SysV: refused
    reset_rec()
    i = Int32(0x7eadbeef)
    ib = reinterpret(NTuple{4,UInt8}, i)
    v = P8((0x21, ib..., 0x41, 0x42, 0x43))
    if ABI === :sysv
        refused(p, () -> take_p8(v))
    else
        s = take_p8(v)
        args_read(p, 5, [0 => s64(0x21), 1 => s64(i), 2 => s64(0x41), 3 => s64(0x42), 4 => s64(0x43)])
        s == i + Int32(0x21) || push!(p, "take_p8 = $s")
    end
end

probe("make_big24") do p                    # 24 B: memory on both
    reset_rec()
    r = make_big24(-5, 1 << 40, 77)
    args_read(p, 3, [0 => s64(-5), 1 => s64(1 << 40), 2 => s64(77)])
    (r.a, r.b, r.c) == (-4, (1 << 40) + 2, Int32(80)) || push!(p, "ret = $r")
end

probe("make_pk17") do p                     # packed 17 B: the explicit sret on both
    reset_rec()
    x = 0x0102030405060708
    r = make_pk17(x)
    args_read(p, 1, [0 => UInt64(x)])
    d = getfield(r, :_data)
    (d[1], field(Int64, d, 1), field(Int64, d, 9)) == (0x55, Int64(x) + 1, Int64(x) - 1) ||
        push!(p, "ret bytes = $d")
end

probe("make_is20") do p                     # natural 20 B: a plain ccall
    reset_rec()
    r = make_is20(-300)
    args_read(p, 1, [0 => s64(-300)])
    r.n == -299 || push!(p, "ret.n = $(r.n)")
    r.s == ntuple(i -> Int16(-300 * i), 8) || push!(p, "ret.s = $(r.s)")
end

probe("make_out6") do p                     # unaligned field inside a packed member
    reset_rec()
    r = make_out6(Int16(1000))
    args_read(p, 1, [0 => s64(1000)])
    d = getfield(r, :_data)
    (d[1], field(Int16, d, 1), d[4:6]) == (0x44, Int16(1005), (0x01, 0x02, 0x03)) ||
        push!(p, "ret bytes = $d")
end

probe("make_bf12") do p                     # float blob, 12 B: Win64 sret; SysV refused
    reset_rec()
    if ABI === :sysv
        refused(p, () -> make_bf12(1.5f0, -3, 99))
    else
        r = make_bf12(1.5f0, -3, 99)
        args_read(p, 3, [0 => UInt64(reinterpret(UInt32, 1.5f0)), 1 => s64(-3), 2 => s64(99)])
        d = getfield(r, :_data)
        b3 = Int(d[5] & 0x07)
        b3 >= 4 && (b3 -= 8)                # a signed 3-bit field
        (field(Float32, d, 0), b3, field(Int32, d, 8)) == (3.0f0, -3, Int32(100)) ||
            push!(p, "ret bytes = $d")
    end
end

println("PROBE_DONE")
