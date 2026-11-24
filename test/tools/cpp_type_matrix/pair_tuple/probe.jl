include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()

const P = "std::pair<int, double>"
case(P, "make") do
    p = M.setproperties(M.pair_int_double(); first = 3, second = 2.5)
    works(M.pair_sum(p), 5.5, "`pair_int_double()` zeroed, then setproperties; the positional constructor also wants the padding field")
end
case(P, "takes a Julia Pair or Tuple") do
    takes(5.5, "a Tuple{Cint,Float64}") do
        M.pair_sum((Cint(3), 2.5))
    end
end
case(P, "pass by value") do
    works(M.pair_sum(M.pair_make(3, 2.5)), 5.5)
end
case(P, "pass by reference") do
    works(M.pair_sum_ref(Ref(M.pair_make(3, 2.5))), 5.5)
end
case(P, "return by value") do
    p = M.pair_make(3, 2.5)
    works((p.first, p.second), (Cint(3), 2.5), "fields `first` and `second`")
end

const S = "std::pair<std::string, int>"
case(S, "return by value") do
    p = fn(:pairs_make)(4)
    opaque((p.second, M.pairs_len(Ref(p))), (Cint(4), 7), "`second` is a field; `first` is the string's bytes, a $(describe(p.first)), not a Julia String")
end
case(S, "intact after the return") do
    p = fn(:pairs_make)(4)
    intact(M.pairs_intact_self(), M.pairs_intact(Ref(p)), 1)
end

const T = "std::tuple<int, double, char>"
const TT = M.tuple_int_double_char
case(T, "return by value") do
    t = M.tup_make(1, 2.5, 3)
    names = filter(n -> !startswith(string(n), "_pad"), fieldnames(TT))
    works(Tuple(getfield(t, n) for n in names), (UInt8(3), 2.5, Cint(1)),
          "the elements are fields in the order the library stores them (last first): $(join(names, ", "))")
end
case(T, "make") do
    t = M.setproperties(TT(); _M_head_impl = 3, _M_head_impl_1 = 2.5, _M_head_impl_2 = 1)
    works(M.tup_sum(t), 6.5, "zeroed, then setproperties on the library's internal field names")
end
case(T, "takes a Julia Tuple") do
    takes(6.5, "a Tuple{Cint,Float64,Cchar}") do
        M.tup_sum((Cint(1), 2.5, Cchar(3)))
    end
end
case(T, "pass by value") do
    works(M.tup_sum(M.tup_make(1, 2.5, 3)), 6.5)
end
case(T, "pass by reference") do
    works(M.tup_sum_ref(Ref(M.tup_make(1, 2.5, 3))), 6.5)
end
done()
