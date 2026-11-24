include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()

const F = "std::function<int(int)>"
twice(x::Cint)::Cint = 2x

case(F, "make") do
    nomake(r"function", "`M.function_int_int()` is an empty struct, and nothing turns a Julia function or a @cfunction into a std::function")
end
case(F, "takes a Julia function") do
    takes(6, "a Julia function, by reference") do
        M.fn_apply_ref(twice, 3)
    end
end
case(F, "takes a Julia function") do
    takes(6, "a Julia function, by value") do
        M.fn_apply(twice, 3)
    end
end
case(F, "takes a @cfunction") do
    takes(6, "a @cfunction pointer") do
        M.fn_apply_ref(@cfunction(twice, Cint, (Cint,)), 3)
    end
end
case(F, "return by reference") do
    p = M.fn_twice()
    (p isa Ptr && p != C_NULL) ? "OPAQUE one the library owns comes back as a $(describe(p))" : "WRONG got $(repr(p))"
end
case(F, "pass by value") do
    # C++ passes the address of the caller's object, and so does the wrapper:
    # the function works on the object the pointer names.
    p = M.fn_twice()
    works((M.fn_apply(p, 21), M.fn_apply(p, 4)), (42, 8), "a pointer to one the library owns, passed twice; no copy is made")
end
case(F, "pass by reference") do
    p = M.fn_twice()
    works((M.fn_apply_ref(p, 3), M.fn_empty(p)), (6, 0), "the pointer as the library returned it")
end
case(F, "pass by reference") do
    p = Ptr{M.function_int_int}(M.fn_twice())
    works((M.fn_apply_ref(p, 3), M.fn_empty(p)), (6, 0), "the pointer converted by hand: Ptr{function_int_int}(p)")
end
case(F, "return by value") do
    r = M.fn_adder(2)
    "ERROR the call was made and answered a $(describe(r))"
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
