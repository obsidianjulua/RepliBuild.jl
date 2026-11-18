# Subprocess probe for test_type_mapping_live.jl: calls every function of
# test/typemap_c_test through its wrapper. Before the fix `out_small` wrote
# through an enum's value as an address and this process died; with a 128-bit
# integer beyond the registers a plain ccall read a wrong value.
# Prints one "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_typemap_c.jl <wrapper_path>

wrapper, = ARGS
const M = include(wrapper)

function probe(f, name)
    verdict = try
        p = String[]
        f(p)
        isempty(p) ? "PASS" : "FAIL " * join(unique(p)[1:min(end, 4)], "; ")
    catch e
        "FAIL threw " * first(sprint(showerror, e), 300)
    end
    println("PROBE $name: ", verdict)
    flush(stdout)
end

# Leave different bytes on the stack and in the registers between calls.
@noinline churn(n) = sum(ntuple(i -> UInt64(0xfedcba9876543210) ⊻ UInt64(i * n), 24))
const ROUNDS = 50

# The value a call gives, or `:refused` when the wrapper is an ABI trap stub.
function right_or_refused(f, args...)
    try
        return f(args...)
    catch e
        (e isa ErrorException && occursin("ABI Safety Trap", e.msg)) || rethrow()
        return :refused
    end
end

probe("enum_out_param") do p
    # A Ref, a Vector and a Ptr all convert to the declared `Ptr{Small}`.
    r = Ref(M.SM_A)
    M.out_small(r)
    r[] == M.SM_C || push!(p, "Ref: $(r[])")
    v = [M.SM_A, M.SM_B]
    M.out_small(v)
    v == [M.SM_C, M.SM_B] || push!(p, "Vector: $v")
    b = Ref(M.SM_B)
    GC.@preserve b M.out_small(Base.unsafe_convert(Ptr{M.Small}, b))
    b[] == M.SM_C || push!(p, "Ptr: $(b[])")
    # The enum is a primitive type of its C width, so the Ref is what C writes.
    sizeof(M.Small) == 4 || push!(p, "sizeof(Small) = $(sizeof(M.Small))")
end

probe("enum_bare_value_refused") do p
    # A value where a pointer is declared never reaches C.
    for (what, call) in (("out_small(SM_A)", () -> M.out_small(M.SM_A)),
                         ("small_swap(SM_A, SM_B)", () -> M.small_swap(M.SM_A, M.SM_B)),
                         ("smalls_sum(SM_A, 1)", () -> M.smalls_sum(M.SM_A, 1)),
                         ("out_small(1)", () -> M.out_small(1)),
                         ("out_small(Ref(1.0))", () -> M.out_small(Ref(1.0))))
        got = try
            call()
            "returned"
        catch e
            e isa MethodError ? "MethodError" : "threw $(typeof(e))"
        end
        got == "MethodError" || push!(p, "$what $got")
    end
end

probe("enum_array") do p
    xs = fill(M.SM_A, 7)
    M.smalls_fill(xs, 7)
    Int.(Integer.(xs)) == [0, 1, 2, 0, 1, 2, 0] || push!(p, "fill: $xs")
    (s = M.smalls_sum(xs, 7)) == 6 || push!(p, "sum = $s")
    (s = M.smalls_sum([M.SM_C, M.SM_C], 2)) == 4 || push!(p, "sum of two SM_C = $s")
end

probe("enum_swap") do p
    for r in 1:ROUNDS
        churn(r)
        a = Ref(M.SM_A); b = Ref(M.SM_C)
        (rc = M.small_swap(a, b)) == 1 || push!(p, "returned $rc")
        (a[], b[]) == (M.SM_C, M.SM_A) || push!(p, "swapped to $((a[], b[]))")
    end
end

probe("enum_pointer_return") do p
    q = M.smalls_static()
    q isa Ptr{M.Small} || push!(p, "smalls_static() isa $(typeof(q))")
    (unsafe_load(Ptr{M.Small}(q), 1), unsafe_load(Ptr{M.Small}(q), 2)) == (M.SM_C, M.SM_B) || push!(p, "wrong elements")
    h = M.smalls_handle()
    h isa Ptr{Ptr{M.Small}} || push!(p, "smalls_handle() isa $(typeof(h))")
    unsafe_load(Ptr{Ptr{Cvoid}}(h)) == Ptr{Cvoid}(q) || push!(p, "handle does not point at the store")
end

probe("enum_pointer_member") do p
    fieldtype(M.Holder, :kinds) == Ptr{M.Small} || push!(p, "Holder.kinds isa $(fieldtype(M.Holder, :kinds))")
    h = M.holder_make()
    h.n == 2 || push!(p, "n = $(h.n)")
    unsafe_load(Ptr{M.Small}(h.kinds), 2) == M.SM_B || push!(p, "kinds[2] wrong")
    sizeof(M.Holder) == 16 || push!(p, "sizeof(Holder) = $(sizeof(M.Holder))")
end

probe("enum_pointer_global") do p
    g = M.g_smalls()
    g isa Ptr{M.Small} || push!(p, "g_smalls() isa $(typeof(g))")
    Ptr{Cvoid}(g) == Ptr{Cvoid}(M.smalls_static()) || push!(p, "g_smalls is not the store")
end

probe("bitmask_enum_pointer") do p
    # A bitmask enum is an integer alias: the Ref is that integer.
    r = Ref{M.Perm}(0)
    M.out_perm(r)
    r[] == (M.P_R | M.P_X) || push!(p, "out_perm gave $(r[])")
    sizeof(M.Perm) == 4 || push!(p, "sizeof(Perm) = $(sizeof(M.Perm))")
end

const BIG = (UInt128(0xfedcba9876543210) << 64) | 0x0123456789abcdef

probe("u128_right_or_refused") do p
    # On every host: the value C computes, or a refusal. Never another value.
    cases = (
        ("u128_echo", () -> M.u128_echo(BIG), BIG),
        ("u128_add", () -> M.u128_add(BIG, UInt128(1)), BIG + 1),
        ("u128_make", () -> M.u128_make(0xfedcba9876543210, 0x0123456789abcdef), BIG),
        ("u128_after4", () -> M.u128_after4(1, 2, 3, 4, BIG), BIG + 10),
        ("u128_after_double", () -> M.u128_after_double(7.0, 1, 2, 3, 4, BIG), BIG + 17),
        ("u128_after5", () -> M.u128_after5(1, 2, 3, 4, 5, BIG), BIG + 15),
        ("u128_after7", () -> M.u128_after7(1, 2, 3, 4, 5, 6, 7, BIG), BIG + 28),
        ("i128_after5", () -> M.i128_after5(1, 2, 3, 4, 5, Int128(-7) << 70), (Int128(-7) << 70) - 15),
        ("u128_then_long", () -> M.u128_then_long(1, 2, 3, 4, 5, BIG, 9), BIG + 15 + 9000),
    )
    called = 0
    for r in 1:ROUNDS, (name, call, want) in cases
        churn(r)
        got = right_or_refused(call)
        got === :refused && continue
        called += 1
        got == want || push!(p, "$name = $(repr(got))")
    end
    # Not every one may be refused: the simplest is callable wherever a
    # 128-bit return is (and where it is not, the pointer case below runs).
    println("NOTE u128 calls made: ", called)
end

probe("u128_pointer_and_halves") do p
    # A pointer to one is an ordinary pointer, on every host.
    r = Ref{UInt128}(0)
    M.u128_out(r)
    r[] == BIG || push!(p, "u128_out gave $(repr(r[]))")
    (hi = right_or_refused(M.u128_hi, BIG)) in (:refused, 0xfedcba9876543210) || push!(p, "u128_hi = $(repr(hi))")
    (lo = right_or_refused(M.u128_lo, BIG)) in (:refused, 0x0123456789abcdef) || push!(p, "u128_lo = $(repr(lo))")
    hi === :refused && push!(p, "u128_hi refused: a parameter in the first registers is callable on every host")
end

probe("unmapped_scalar_is_one_stub") do p
    # The library still loads, and the mapped functions are calls.
    (s = M.plain_add(2, 3)) == 5 || push!(p, "plain_add = $s")
    for (what, call) in (("f16_echo", () -> M.f16_echo(Float16(1.5))), ("f16_bits", () -> M.f16_bits(Float16(1.5))))
        got = try
            call()
            "returned"
        catch e
            e isa ErrorException && occursin("ABI Safety Trap", e.msg) && occursin("_Float16", e.msg) ?
                "trap" : "threw $(first(sprint(showerror, e), 120))"
        end
        got == "trap" || push!(p, "$what $got")
    end
end

println("PROBE_DONE")
