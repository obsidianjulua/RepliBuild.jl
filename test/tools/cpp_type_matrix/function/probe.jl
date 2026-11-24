include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()

const F = "std::function<int(int)>"
twice(x::Cint)::Cint = 2x

case(F, "make") do
    nomake(r"function", "`M.function_int_int()` is an empty struct, and nothing turns a Julia function or a @cfunction into a std::function")
end
case(F, "takes a Julia function") do
    takes(6, "a Julia function") do
        M.fn_apply_ref(twice, 3)
    end
end
case(F, "takes a @cfunction") do
    takes(6, "a @cfunction pointer") do
        M.fn_apply_ref(@cfunction(twice, Cint, (Cint,)), 3)
    end
end
case(F, "pass by value") do
    r = M.fn_apply(twice, 3)
    "ERROR the call was made and answered $(repr(r))"
end
case(F, "return by value") do
    r = M.fn_adder(2)
    "ERROR the call was made and answered a $(describe(r))"
end
case(F, "pass by reference") do
    # Nothing can be passed: the parameter is a Ref to an empty struct, and the
    # one function that would return a std::function is refused.
    try
        M.fn_adder(2)
    catch
        return noway("`fn_apply_ref` and `fn_empty` are bound and take a `Ref{function_int_int}`, an empty struct; nothing makes or returns one")
    end
    "ERROR fn_adder returned, and this probe does not pass its result on"
end

const V = "std::function<void(int)>"
case(V, "pass by reference") do
    takes(3, "a Julia function") do
        M.fn_each(i -> nothing, 3)
    end
end
const D = "std::function<double(double, double)>"
case(D, "pass by reference") do
    takes(5.0, "a Julia function") do
        M.fn_fold(+, 2.0, 3.0)
    end
end

# The comparison: a plain function pointer, which C++ also calls a callback.
const P = "int (*)(int) (a plain function pointer)"
case(P, "make") do
    works(M.fn_plain(@cfunction(twice, Cint, (Cint,)), 21), 42, "@cfunction")
end
case(P, "takes a Julia function") do
    takes(42, "a Julia function") do
        M.fn_plain(twice, 21)
    end
end
done()
