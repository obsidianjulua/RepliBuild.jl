# Subprocess probe for test_type_mapping_live.jl: calls every function of
# test/typemap_cpp_test through its wrapper's thunks. Before the fix a `const
# Color&` argument was converted to its integer and handed to a thunk that
# loads a pointer (`color_is(Blue, Blue)` was false, `color_out` wrote through
# garbage), and the character types and `std::byte` were trap stubs.
# Prints one "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_typemap_cpp.jl <wrapper_path>

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
const ROUNDS = 100

function right_or_refused(f, args...)
    try
        return f(args...)
    catch e
        (e isa ErrorException && occursin("ABI Safety Trap", e.msg)) || rethrow()
        return :refused
    end
end

probe("enum_reference_out") do p
    for r in 1:ROUNDS
        churn(r)
        c = Ref(M.Red)
        M.tmap_color_out(c)
        c[] == M.Blue || push!(p, "color_out left $(c[])")
    end
    sizeof(M.Color) == 1 || push!(p, "sizeof(Color) = $(sizeof(M.Color))")
end

probe("enum_const_reference") do p
    for r in 1:ROUNDS, (a, b, want) in ((M.Blue, M.Blue, true), (M.Red, M.Blue, false),
                                        (M.Green, M.Green, true), (M.Blue, M.Red, false))
        churn(r)
        (got = M.tmap_color_is(Ref(a), b)) === want || push!(p, "color_is($a, $b) = $got")
    end
    # The by-value position still takes an integer, as an enum parameter does.
    M.tmap_color_is(Ref(M.Blue), 250) === true || push!(p, "color_is(Blue, 250) is not true")
end

probe("enum_bare_value_refused") do p
    # A value where a reference is declared never reaches the thunk.
    for (what, call) in (("color_out(Red)", () -> M.tmap_color_out(M.Red)),
                         ("color_is(Blue, Blue)", () -> M.tmap_color_is(M.Blue, M.Blue)),
                         ("color_swap(Red, Green)", () -> M.tmap_color_swap(M.Red, M.Green)),
                         ("color_out(Ref(1))", () -> M.tmap_color_out(Ref(1))))
        got = try
            call()
            "returned"
        catch e
            e isa MethodError ? "MethodError" : "threw $(typeof(e))"
        end
        got == "MethodError" || push!(p, "$what $got")
    end
end

probe("enum_pointer_param") do p
    for r in 1:ROUNDS
        churn(r)
        l = Ref(M.Low)
        M.tmap_level_out(l)                      # a plain ccall, `Ptr{Level}`
        l[] == M.High || push!(p, "level_out left $(l[])")
        xs = [M.Low, M.Mid, M.High, M.High]
        (s = M.tmap_levels_sum(xs, 4)) == 300 || push!(p, "levels_sum = $s")
        a = Ref(M.Red); b = Ref(M.Green)
        M.tmap_color_swap(a, b)                  # a plain ccall, two `Ref{Color}`
        (a[], b[]) == (M.Green, M.Red) || push!(p, "color_swap gave $((a[], b[]))")
    end
    sizeof(M.Level) == 2 || push!(p, "sizeof(Level) = $(sizeof(M.Level))")
end

probe("enum_pointer_return") do p
    q = M.tmap_color_store()
    q isa Ptr{M.Color} || push!(p, "color_store() isa $(typeof(q))")
    (unsafe_load(Ptr{M.Color}(q), 1), unsafe_load(Ptr{M.Color}(q), 2)) == (M.Green, M.Blue) || push!(p, "wrong elements")
    f = M.tmap_color_first()
    f isa Ptr{M.Color} || push!(p, "color_first() isa $(typeof(f))")
    Ptr{Cvoid}(f) == Ptr{Cvoid}(q) || push!(p, "color_first() is not the store")
end

probe("enum_pointer_member") do p
    fieldtype(M.Palette, :colors) == Ptr{M.Color} || push!(p, "Palette.colors isa $(fieldtype(M.Palette, :colors))")
    pl = M.tmap_palette_make()
    (pl.n, pl.floor) == (2, M.Low) || push!(p, "n, floor = $((pl.n, pl.floor))")
    unsafe_load(Ptr{M.Color}(pl.colors), 2) == M.Blue || push!(p, "colors[2] wrong")
end

probe("char_types") do p
    for r in 1:ROUNDS
        churn(r)
        (got = M.tmap_u8_echo(0x41)) === 0x41 || push!(p, "u8_echo = $(repr(got))")
        (got = M.tmap_u8_echo(0xfe)) === 0xfe || push!(p, "u8_echo(0xfe) = $(repr(got))")
        (got = M.tmap_c16_echo(0x263a)) === 0x263a || push!(p, "c16_echo = $(repr(got))")
        (got = M.tmap_c32_echo(0x0001f600)) === 0x0001f600 || push!(p, "c32_echo = $(repr(got))")
        (got = M.tmap_c16_next(0xffff)) === 0x0000 || push!(p, "c16_next(0xffff) = $(repr(got))")
        # The callee reads each narrow argument 32 bits wide.
        want = 0xfe + 1000 * 0xfffe + 100000000 * (0x0010fffe % 7) + 5
        (got = M.tmap_chars_mix(0xfe, 0xfffe, 0x0010fffe, 5)) == want || push!(p, "chars_mix = $got, not $want")
    end
end

probe("std_byte") do p
    sizeof(M.byte) == 1 || push!(p, "sizeof(byte) = $(sizeof(M.byte))")
    for r in 1:ROUNDS
        churn(r)
        b = M.tmap_byte_flip(0x0f)               # an integer passes, as to any enum parameter
        b isa M.byte || push!(p, "byte_flip returns $(typeof(b))")
        UInt8(b) == 0xf0 || push!(p, "byte_flip(0x0f) = $(repr(UInt8(b)))")
        M.tmap_byte_flip(M.byte(0xf0)) == M.byte(0x0f) || push!(p, "byte_flip(byte(0xf0)) wrong")
    end
    buf = fill(M.byte(0), 5)
    M.tmap_bytes_fill(buf, 5)                    # a plain ccall, `Ptr{byte}`
    UInt8.(buf) == 1:5 || push!(p, "bytes_fill gave $(UInt8.(buf))")
    (s = M.tmap_bytes_sum(buf, 5)) == 15 || push!(p, "bytes_sum = $s")
end

probe("empty_library_enum") do p
    sizeof(M.Handle) == 4 || push!(p, "sizeof(Handle) = $(sizeof(M.Handle))")
    for r in 1:ROUNDS
        churn(r)
        (h = M.tmap_handle_next(M.Handle(41))) == M.Handle(42) || push!(p, "handle_next(41) = $h")
        (h = M.tmap_handle_next(0xfffffffe)) == M.Handle(0xffffffff) || push!(p, "handle_next(0xfffffffe) = $h")
    end
end

probe("declared_only_enum_pointer") do p
    # An enum this library only declares has no type to emit: the pointer is
    # typed with an opaque placeholder, and the module still loads.
    r = Ref{Cint}(0)
    GC.@preserve r M.tmap_opaque_set(Ptr{M.Opaque}(Base.unsafe_convert(Ptr{Cint}, r)))
    r[] == 7 || push!(p, "opaque_set left $(r[])")
end

const BIG = (UInt128(0xfedcba9876543210) << 64) | 0x0123456789abcdef

probe("u128_through_a_thunk") do p
    # On every host: the value C computes, or a refusal. Never another value.
    cases = (
        ("u128_echo", () -> M.tmap_u128_echo(BIG), BIG),
        ("u128_after5", () -> M.tmap_u128_after5(1, 2, 3, 4, 5, BIG), BIG + 15),
        ("u128_after7", () -> M.tmap_u128_after7(1, 2, 3, 4, 5, 6, 7, BIG), BIG + 28),
        ("u128_hi", () -> M.tmap_u128_hi(BIG), 0xfedcba9876543210),
    )
    called = 0
    for r in 1:ROUNDS, (name, call, want) in cases
        churn(r)
        got = right_or_refused(call)
        got === :refused && continue
        called += 1
        got == want || push!(p, "$name = $(repr(got))")
    end
    println("NOTE u128 calls made: ", called)
end

println("PROBE_DONE")
