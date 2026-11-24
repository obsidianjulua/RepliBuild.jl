include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()

const S = "std::span<const int>"
const CSpan = M.span_const_int_18446744073709551615UL
const MSpan = M.span_int_18446744073709551615UL
const Extent = M._extent_storage_18446744073709551615UL

case(S, "make") do
    v = Cint[1, 2, 3, 4]
    GC.@preserve v works(M.span_sum(CSpan(pointer(v), Extent(length(v)))), 10, "built by hand: pointer(v), and the length inside a second struct; the caller keeps the Vector alive")
end
case(S, "takes a Julia Vector") do
    takes(10, "a Vector{Cint}") do
        M.span_sum(Cint[1, 2, 3, 4])
    end
end
case(S, "pass by value") do
    v = Cint[5, 6]
    GC.@preserve v works(M.span_sum(CSpan(pointer(v), Extent(2))), 11)
end
case(S, "pass by reference") do
    v = Cint[5, 6, 7]
    GC.@preserve v works(M.span_sum_ref(Ref(CSpan(pointer(v), Extent(3)))), 18)
end

const W = "std::span<int>"
case(W, "pass by value") do
    v = Cint[1, 2, 3]
    GC.@preserve v M.span_fill(MSpan(pointer(v), Extent(3)), 9)
    works(v, Cint[9, 9, 9], "the library writes the Julia Vector's memory")
end
case(W, "return by value") do
    r = M.span_global()
    works(unsafe_wrap(Array, r._M_ptr, r._M_extent._M_extent_value), Cint[1, 2, 3, 4], "pointer and length read from the struct's fields")
end

const F = "std::span<const int, 3> (fixed extent)"
case(F, "pass by value") do
    r = M.span_fixed(M.span_const_int_3UL())
    "ERROR the call was made with an empty struct and answered $(repr(r))"
end
done()
