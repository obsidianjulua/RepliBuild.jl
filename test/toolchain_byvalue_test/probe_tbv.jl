# Subprocess probe for test_toolchain_byvalue_live.jl: calls every function the
# fixture exports and compares the result with what the C++ computes. A wrong
# by-value crossing returns plausible garbage or crashes, so this runs in a child
# process. Prints one "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_tbv.jl <wrapper_path>

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
check(p, what, got, want) = got == want || push!(p, "$what: got $got, want $want")
zeroof(::Type{T}) where {T<:Tuple} = ntuple(i -> zeroof(fieldtype(T, i)), fieldcount(T))
zeroof(::Type{T}) where {T} = zero(T)
build(T; kw...) = T((haskey(kw, n) ? kw[n] : zeroof(fieldtype(T, i)) for (i, n) in enumerate(fieldnames(T)))...)

probe("pair_sse_sse") do p
    r = M.minmax_pair(3.0, 1.5)
    check(p, "minmax_pair", (r.first, r.second), (1.5, 3.0))
end
probe("pair_int_sse") do p
    r = M.tagged(7, 2.25)
    check(p, "tagged", (r.first, r.second), (Int32(7), 4.5))
end
probe("pair_one_integer") do p
    r = M.divmod(17, 5)
    check(p, "divmod", (r.first, r.second), (Int32(3), Int32(2)))
end
probe("pair_two_floats") do p
    r = M.fpair(2.5f0, 1.0f0)
    check(p, "fpair", (r.first, r.second), (3.5f0, 1.5f0))
end
probe("pair_memory_return") do p
    r = M.triple(1.0, 2.0, 3.0)
    check(p, "triple", (r.first, r.second.first, r.second.second), (1.0, 2.0, 3.0))
end
probe("pair_inside_pair") do p
    r = M.nested(4, 9, 0.5)
    check(p, "nested", (r.first.first, r.first.second, r.second), (Int32(4), Int32(9), 0.5))
end
probe("pair_pointer_in_arguments") do p
    xs = Int32[5, 8, 13]
    GC.@preserve xs begin
        r = M.find_first(pointer(xs), 3, 8)
        check(p, "found", r.second, true)
        check(p, "address", r.first, pointer(xs) + sizeof(Int32))
        miss = M.find_first(pointer(xs), 3, 99)
        check(p, "missing", (miss.first, miss.second), (Ptr{Cint}(0), false))
    end
end
probe("pair_parameters") do p
    check(p, "pair_sum", M.pair_sum(M.pair_double_double(1.25, 2.5)), 3.75)
    # pair<int, double> has a `_pad_0` between its fields; zero it.
    check(p, "pair_imix", M.pair_imix(build(M.pair_int_double; first = Int32(3), second = 4.9), 10), Int32(34))
end
probe("library_record_holds_pair") do p
    s = M.make_span(2, 7, 1.5)
    check(p, "make_span", (s.range.first, s.range.second, s.weight), (Int32(2), Int32(7), 1.5))
    check(p, "span_weight", M.span_weight(s), 7.5)
end
probe("std_array") do p
    check(p, "arr2", M.arr2(-1.0, 8.0)._M_elems, (-1.0, 8.0))
end
probe("string_view") do p
    msg = "hello world"
    GC.@preserve msg begin
        v = M.tail_view(pointer(msg), 5)
        check(p, "length", Int(v._M_len), 5)
        check(p, "text", unsafe_string(v._M_str, v._M_len), "world")
    end
end
probe("map_lookup_is_a_plain_call") do p
    check(p, "map_lookup(2)", M.map_lookup(2), Int32(23))
    check(p, "map_lookup(9)", M.map_lookup(9), Int32(-1))
end
# Laid out from their bases (2026-11-13). Returned records are read as their C
# bytes, so the check does not depend on the Julia type's field names, then
# passed back in by value.
bytes(x) = (r = Ref(x); GC.@preserve r copy(unsafe_wrap(Array, Ptr{UInt8}(pointer_from_objref(r)), sizeof(x))))
at(T, b, off) = GC.@preserve b unsafe_load(Ptr{T}(pointer(b) + off))
probe("tuple_from_bases") do p
    t = M.tup(7, 2.5)
    b = bytes(t)
    check(p, "sizeof tuple<int, double>", length(b), 16)
    # libstdc++ stores the elements in reverse: the double at 0, the int at 8.
    check(p, "tup", (at(Float64, b, 0), at(Int32, b, 8)), (2.5, Int32(7)))
    check(p, "tup_sum", M.tup_sum(t), 72.5)
end
probe("optional_from_bases") do p
    some, none = M.maybe(true, 3.25), M.maybe(false, 9.0)
    bs, bn = bytes(some), bytes(none)
    check(p, "maybe(true)", (at(Float64, bs, 0), bs[9]), (3.25, 0x01))
    check(p, "maybe(false) engaged", bn[9], 0x00)
    check(p, "maybe_or(some)", M.maybe_or(some, -1.0), 3.25)
    check(p, "maybe_or(none)", M.maybe_or(none, -1.0), -1.0)
    si, ni = M.maybe_int(true, 41), M.maybe_int(false, 5)
    bi = bytes(si)
    check(p, "maybe_int(true)", (at(Int32, bi, 0), bi[5]), (Int32(41), 0x01))
    check(p, "maybe_int_or(some)", M.maybe_int_or(si, -1), Int32(41))
    check(p, "maybe_int_or(none)", M.maybe_int_or(ni, -1), Int32(-1))
end
probe("complex_width") do p
    z = M.cplx(1.5, -2.0)
    bz = bytes(z)
    check(p, "cplx", (length(bz), at(Float64, bz, 0), at(Float64, bz, 8)), (16, 1.5, -2.0))
    check(p, "cplx_norm", M.cplx_norm(z), 148.0)
    w = M.cplxf(0.5f0, 4.0f0)
    bw = bytes(w)
    check(p, "cplxf", (length(bw), at(Float32, bw, 0), at(Float32, bw, 4)), (8, 0.5f0, 4.0f0))
    check(p, "cplxf_norm", M.cplxf_norm(w), 54.0f0)
end
# A record holding a `long double _Complex`: MEMORY (sret) on x86-64, an HFA of
# two quads on AArch64. C++ reads it back in the host's own `long double`, so
# the probe asks no format question (an x87 decoder read a quad's zero low
# fraction bytes as the mantissa and got 0.0 on the arm64 CI cell).
probe("complex_long_double") do p
    z = M.cplxl(1.5, -2.25)
    check(p, "sizeof", sizeof(z), 32)
    r = Ref(z)
    check(p, "cplxl", (M.cplxl_re(r), M.cplxl_im(r)), (1.5, -2.25))
end
probe("control_empty_base") do p
    r = M.make_lp(1.0, 2.0)
    check(p, "make_lp", (r.a, r.b), (1.0, 2.0))
    check(p, "lp_sum", M.lp_sum(M.LP(0.25, 0.5)), 0.75)
end
probe("control_template_in_memory_return") do p
    o = M.make_outer3(1.0, 2.0, 3.0)
    check(p, "make_outer3", (o.x, o.in.a, o.in.b), (1.0, 2.0, 3.0))
end
println("PROBE_DONE")
