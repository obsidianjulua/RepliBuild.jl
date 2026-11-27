include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()
const T = "std::any"

case(T, "make") do
    works(M.any_has(M.any()), 0, "an empty one: `M.any()`, all zero")
end
case(T, "make (holding a value)") do
    nomake(r"any", "the struct is $(sizeof(M.any)) bytes with no field for a value; the library's `any_int` is the way")
end
case(T, "return by value") do
    a = M.any_int(5)
    opaque(M.any_get(Ref(a)), 5, "the C++ object, a $(describe(a)); what it holds is read only through the library")
end
case(T, "pass by reference") do
    works(M.any_get(Ref(M.any_int(41))), 41, "Ref of the returned struct")
end
case(T, "pass by value") do
    a = M.any_int(5)
    works((M.any_has(a), M.any_has(a)), (1, 1), "passed twice")
end
done()
