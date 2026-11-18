# Subprocess probe for test_char_bitfield_live.jl, C++ library: a `char` by
# value through a Tier 2 thunk and through ccall, `char` in an overload set,
# a signed bitfield in a C++ record. One "PROBE <case>: PASS|FAIL …" line
# each, then PROBE_DONE.
# Every expectation is the host's: `char` is signed on x86-64 and unsigned on
# AArch64, so a byte is spelled `reinterpret(Cchar, 0xfb)`, never `Cchar(-5)`.
# Run: julia --project=<RepliBuild> probe_char_bitfield_cpp.jl <wrapper_path>

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

byte(b) = reinterpret(Cchar, UInt8(b))
@noinline churn(n) = sum(ntuple(i -> UInt64(0xfedcba9876543210) ⊻ UInt64(i * n), 24))

# The callees return `(int)c`, which clang compiles to a plain move of the
# argument register: the extension the caller did is what comes back.
for (fname, f) in (("thunk", M.cbx_char_value), ("ccall", M.cbx_char_value_nx))
    probe("char_value_$fname") do p
        for r in 1:50, (arg, b) in ((byte(0xfb), 0xfb), (-5, 0xfb), (0xfb, 0xfb), (65, 0x41),
                                    (0x41, 0x41), (255, 0xff), (-128, 0x80))
            churn(r)
            got = f(arg)
            check(p, "$(repr(arg)) -> $got", got == Int(byte(b)))
        end
        check(p, "256 accepted", throws_inexact(() -> f(256)))
    end
end

probe("char_return") do p
    r = M.cbx_char_next(byte(0x41))
    check(p, "char_next = $(repr(r))", r === byte(0x42))
end

probe("char_in_an_overload_set") do p
    # `ovl(char)` is the method of the host's `char`.
    check(p, "ovl(char)", M.cbx_ovl(byte(0x41)) == 3065)
    check(p, "ovl(int)", M.cbx_ovl(Cint(7)) == (Clong === Cint ? M.cbx_ovl(Clong(7)) : 1007))
end

probe("signed_bitfield_in_a_cpp_record") do p
    f = M.Flags()
    M.cbx_flags_fill(Ref(f))
    check(p, "on = $(repr(M.get_on(f)))", M.get_on(f) === UInt32(1))
    check(p, "level = $(repr(M.get_level(f)))", M.get_level(f) === Cint(-5))
    check(p, "rest = $(repr(M.get_rest(f)))", M.get_rest(f) === UInt32(1000))
    for v in (-8, -1, 0, 7)
        M.set_level!(f, v)
        check(p, "set_level!($v): C reads", M.cbx_flags_level(Ref(f)) == v)
        check(p, "set_level!($v) moved rest", M.get_rest(f) == 1000 && M.get_on(f) == 1)
    end
    check(p, "set_level!(8) accepted", throws_inexact(() -> M.set_level!(f, 8)))
    check(p, "set_level!(-9) accepted", throws_inexact(() -> M.set_level!(f, -9)))
end

println("PROBE_DONE")
