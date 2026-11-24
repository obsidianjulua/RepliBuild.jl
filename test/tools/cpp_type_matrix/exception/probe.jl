include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()

# What Julia sees when the library throws: the exception's type and its text.
function thrown(f)
    try
        f()
    catch e
        return (string(nameof(typeof(e))), sprint(showerror, e))
    end
    return ("nothing", "the call returned")
end
caught(f, text) = ((ty, msg) = thrown(f);
    occursin(text, msg) ? "WORKS a Julia $ty carrying the C++ message" :
    ty == "nothing" ? "WRONG the call returned" : "WRONG a Julia $ty without the C++ message: $(first(msg, 120))")

case("std::runtime_error", "thrown") do
    caught(() -> M.ex_runtime(-1), "negative input")
end
case("std::out_of_range", "thrown") do
    caught(() -> M.ex_out_of_range(10), "too large")
end
case("a class derived from std::exception", "thrown") do
    caught(() -> M.ex_custom(0), "my error")
end
case("int (not a std::exception)", "thrown") do
    (ty, msg) = thrown(() -> M.ex_int(0))
    ty == "nothing" ? "WRONG the call returned" : "WORKS a Julia $ty: $(first(msg, 100))"
end
case("std::runtime_error", "the next call after a throw") do
    thrown(() -> M.ex_runtime(-1)); thrown(() -> M.ex_int(0))
    works((M.ex_runtime(3), M.ex_out_of_range(4)), (3, 4))
end

const E = "std::error_code"
case(E, "return by value") do
    e = M.ec_make(5)
    works((e._M_value, e._M_cat != C_NULL), (Cint(5), true), "the value is a field; the category is a pointer")
end
case(E, "pass by value") do
    works(M.ec_value(M.ec_make(5)), 5)
end
case(E, "make") do
    # The category is a pointer to an object of the C++ runtime: taken from a
    # code the library made.
    e = M.setproperties(M.ec_make(0); _M_value = 12)
    M.ec_value(e) == 12 || return "WRONG got $(M.ec_value(e)), the library's answer is 12"
    noway("the category is a pointer only the library can supply; the value of a code the library returned can be changed with setproperties (12 reads back)")
end
done()
