include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()

const A = "std::array<int, 4>"
case(A, "make") do
    works(M.arr_sum(M.array_int_4UL((1, 2, 3, 4))), 10, "the struct, from a tuple")
end
case(A, "takes a Julia Tuple") do
    takes(10, "an NTuple{4,Cint}") do
        M.arr_sum(Cint.((1, 2, 3, 4)))
    end
end
case(A, "pass by value") do
    works(M.arr_sum(M.array_int_4UL((5, 6, 7, 8))), 26)
end
case(A, "pass by reference") do
    works(M.arr_sum_ref(Ref(M.array_int_4UL((1, 2, 3, 4)))), 10, "const&, a Ref of the struct")
end
case(A, "pass by reference") do
    r = Ref(M.array_int_4UL((1, 2, 3, 4)))
    M.arr_fill(r, 7)
    works(r[]._M_elems, Cint.((7, 7, 7, 7)), "the library writes through the Ref")
end
case(A, "return by value") do
    works(M.arr_make(3)._M_elems, Cint.((3, 4, 5, 6)), "read from the field `_M_elems`")
end

const D = "std::array<double, 3>"
case(D, "make") do
    works(M.arrd_sum(M.array_double_3UL((1.0, 2.0, 3.5))), 6.5)
end
case(D, "pass by value") do
    works(M.arrd_sum(M.array_double_3UL((0.5, 0.25, 0.25))), 1.0)
end
case(D, "return by value") do
    works(M.arrd_make(1.5)._M_elems, (1.5, 3.0, 4.5))
end
done()
