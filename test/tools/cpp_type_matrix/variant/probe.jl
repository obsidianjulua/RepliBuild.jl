include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()

const V = "std::variant<int, double>"
case(V, "make") do
    nomake(r"variant", "`M.variant_int_double()` is an empty struct")
end
case(V, "return by value") do
    r = M.var_make(0)
    "ERROR the call was made and answered a $(describe(r))"
end
case(V, "pass by value") do
    r = M.var_index(M.variant_int_double())
    "ERROR the call was made with an empty struct and answered $(repr(r))"
end
case(V, "pass by reference") do
    try
        M.var_make(0)
    catch
        return noway("`var_value` is bound and takes a `Ref{variant_int_double}`, an empty struct; nothing makes or returns one")
    end
    "ERROR var_make returned, and this probe does not pass its result on"
end

const S = "std::variant<int, std::string>"
case(S, "return by value") do
    r = fn(:vars_make)(1)
    "ERROR the call was made and answered a $(describe(r))"
end
case(S, "pass by reference") do
    try
        fn(:vars_make)(1)
    catch
        return noway("`vars_index` is bound and takes a Ref to an empty struct; nothing makes or returns one")
    end
    "ERROR vars_make returned, and this probe does not pass its result on"
end
done()
