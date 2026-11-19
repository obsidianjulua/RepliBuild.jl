# Subprocess probe for test_char_bitfield_live.jl, C library: a `char` taken by
# value, signed bitfields, an open enum taken by value. One
# "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Every expectation is the host's: `char` is signed on x86-64 and unsigned on
# AArch64, so a byte is spelled `reinterpret(Cchar, 0xfb)`, never `Cchar(-5)`.
# Run: julia --project=<RepliBuild> probe_char_bitfield_c.jl <wrapper_path>

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
check(p, what, ok) = ok || push!(p, what)
throws_inexact(f) = try f(); false catch e; e isa InexactError end

byte(b) = reinterpret(Cchar, UInt8(b))   # the host's `char` holding byte b

probe("char_takes_cchar_and_bytes") do p
    # The callee returns `(int)c`: the value C sees, extended as C extends it.
    for (arg, b) in ((byte(0xfb), 0xfb), (0xfb, 0xfb), (-5, 0xfb), (65, 0x41), (0x41, 0x41),
                     (Int16(-128), 0x80), (255, 0xff), (true, 0x01))
        got = M.cb_char_value(arg)
        check(p, "cb_char_value($(repr(arg))) = $got", got == Int(byte(b)))
    end
    check(p, "256 accepted", throws_inexact(() -> M.cb_char_value(256)))
    check(p, "-129 accepted", throws_inexact(() -> M.cb_char_value(-129)))
end

probe("char_return_is_cchar") do p
    r = M.cb_char_next(byte(0x41))
    check(p, "cb_char_next = $(repr(r))", r === byte(0x42))
    r = M.cb_char_next(byte(0xfe))
    check(p, "cb_char_next(0xfe) = $(repr(r))", r === byte(0xff))
end

probe("signed_char_is_int8") do p
    check(p, "cb_schar_value(Int8(-5))", M.cb_schar_value(Int8(-5)) == -5)
    check(p, "cb_schar_value(-5)", M.cb_schar_value(-5) == -5)
    check(p, "cb_schar_value(200) accepted", throws_inexact(() -> M.cb_schar_value(200)))
end

probe("char_into_a_record") do p
    o = M.cb_onebyte(byte(0xf7))
    check(p, "tag = $(repr(o.tag))", (o.tag % UInt8) == 0xf7)
end

probe("signed_bitfields_read") do p
    r = Ref(M.Bits())
    M.cb_bits_fill(r)
    b = r[]
    check(p, "a = $(repr(M.get_a(b)))", M.get_a(b) === UInt32(5))         # unsigned: as before
    check(p, "u = $(repr(M.get_u(b)))", M.get_u(b) === UInt32(17))
    check(p, "s = $(repr(M.get_s(b)))", M.get_s(b) === Cint(-3))
    check(p, "w = $(repr(M.get_w(b)))", M.get_w(b) === Clonglong(-123456789012))
    check(p, "t = $(repr(M.get_t(b)))", M.get_t(b) === Int8(-8))
end

probe("signed_bitfields_write") do p
    r = Ref(M.Bits()); M.cb_bits_fill(r); b = r[]
    for v in (-64, -1, 0, 63)
        b2 = M.set_s(b, v)
        check(p, "set_s($v): C reads $(M.cb_bits_s(b2))", M.cb_bits_s(b2) == v)
        check(p, "set_s($v): get_s", M.get_s(b2) == v)
        check(p, "set_s($v) moved a", M.get_a(b2) == 5 && M.get_u(b2) == 17)
    end
    check(p, "set_s(64) accepted", throws_inexact(() -> M.set_s(b, 64)))
    check(p, "set_s(-65) accepted", throws_inexact(() -> M.set_s(b, -65)))
    for v in (-(Int64(1) << 39), (Int64(1) << 39) - 1, -2)
        b2 = M.set_w(b, v)
        check(p, "set_w($v): C reads $(M.cb_bits_w(b2))", M.cb_bits_w(b2) == v)
        check(p, "set_w($v) moved t", M.get_t(b2) == -8 && M.get_s(b2) == -3)
    end
    check(p, "set_w(2^39) accepted", throws_inexact(() -> M.set_w(b, Int64(1) << 39)))
    check(p, "set_t(7)", M.cb_bits_t(M.set_t(b, 7)) == 7)
    check(p, "set_t(8) accepted", throws_inexact(() -> M.set_t(b, 8)))
    # An unsigned field keeps its masking store.
    check(p, "set_u(33)", M.get_u(M.set_u(b, 33)) == 1)
end

probe("enum_bitfield_follows_underlying") do p
    r = Ref(M.ModeBits()); M.cb_modebits_fill(r); b = r[]
    check(p, "m = $(repr(M.get_m(b)))", M.get_m(b) == -1 && M.get_m(b) isa Signed)
    check(p, "k = $(repr(M.get_k(b)))", M.get_k(b) === UInt32(2))
end

probe("enum_param_takes_integer") do p
    check(p, "enum", M.cb_mode_value(M.MODE_HIGH) == 10)
    check(p, "Int", M.cb_mode_value(1) == 10)
    check(p, "negative Int", M.cb_mode_value(-1) == -10)
    check(p, "undeclared value", M.cb_mode_value(Int8(7)) == 70)
    check(p, "too wide accepted", throws_inexact(() -> M.cb_mode_value(Int64(1) << 40)))
end

println("PROBE_DONE")
