# Subprocess probe for test_narrow_args_live.jl: passes enums and integers
# narrower than `int` through the wrapper's thunks to callees that read the
# argument 32 bits wide. Each case is repeated with other calls in between,
# since what lies above an unextended byte is whatever the stack held. Prints
# one "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_narrow_arg.jl <wrapper_path>

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
end

# Leave different bytes on the stack between calls.
@noinline churn(n) = sum(ntuple(i -> UInt64(0xfedcba9876543210) ⊻ UInt64(i * n), 24))
const ROUNDS = 200

# Jolt's shape: a member function indexing an array with a one-byte enum.
probe("enum_uint8_indexes_an_array") do p
    t = Ref(M.Table((0x0, 0x0, 0x0, 0x0)))
    M.na_Table_Table(t)
    for r in 1:ROUNDS, (k, want) in ((M.A, 10), (M.B, 20), (M.C, 30))
        churn(r)
        (got = M.na_Table_count(t, k)) == want || push!(p, "count($k) = $got")
    end
    M.na_Table_bump(t, M.C, 0x05)
    (got = M.na_Table_count(t, M.C)) == 35 || push!(p, "count(C) after bump = $got")
end

probe("enum_uint8_zero_extended") do p
    for r in 1:ROUNDS, (k, want) in ((M.A, 0), (M.B, 1), (M.C, 2))
        churn(r)
        (got = M.na_kind_index(k)) == want || push!(p, "kind_index($k) = $got")
    end
    # An open enum takes any value of its type: 0xff must come through as 255.
    (got = M.na_kind_index(M.Kind(0xff))) == 255 || push!(p, "kind_index(0xff) = $got")
end

probe("enum_int8_sign_extended") do p
    for r in 1:ROUNDS, (l, want) in ((M.Low, -2), (M.Mid, 0), (M.High, 3))
        churn(r)
        (got = M.na_level_value(l)) == want || push!(p, "level_value($l) = $got")
    end
end

probe("enum_uint16_and_int") do p
    for r in 1:ROUNDS
        churn(r)
        (got = M.na_wide16_value(M.W9)) == 9 || push!(p, "wide16_value(W9) = $got")
        (got = M.na_wide16_value(M.WMax)) == 0xffff || push!(p, "wide16_value(WMax) = $got")
        (got = M.na_full_value(M.F7)) == 7 || push!(p, "full_value(F7) = $got")
    end
end

probe("plain_narrow_integers") do p
    for r in 1:ROUNDS
        churn(r)
        (got = M.na_u8_value(0xfe)) == 254 || push!(p, "u8_value(0xfe) = $got")
        (got = M.na_s8_value(Int8(-3))) == -3 || push!(p, "s8_value(-3) = $got")
        (got = M.na_u16_value(0xfffe)) == 65534 || push!(p, "u16_value(0xfffe) = $got")
        (got = M.na_s16_value(-300)) == -300 || push!(p, "s16_value(-300) = $got")
        (got = M.na_bool_value(true)) == 100 || push!(p, "bool_value(true) = $got")
        (got = M.na_bool_value(false)) == 200 || push!(p, "bool_value(false) = $got")
    end
end

# Narrow arguments of every kind among a long and a double.
probe("narrow_among_wide") do p
    for r in 1:ROUNDS
        churn(r)
        got = M.na_mix(5, M.C, Int8(-4), 0x0100, true, 9.0, M.Low)
        want = 5 + 1000 * 2 + 100 * (-4) + 256 + 7 + 9 + 10000 * (-2)
        got == want || push!(p, "mix = $got, want $want")
    end
end

println("PROBE_DONE")
