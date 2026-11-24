include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()

const A = "std::atomic<int>"
case(A, "make") do
    nomake(r"atomic", "`M.atomic_int()` is an empty struct")
end
case(A, "return by reference") do
    p = M.at_counter()
    a = M.at_bump(p); b = M.at_bump(p)
    opaque(b - a, 1, "a $(describe(p)): the library counts through it, and Julia cannot read the value")
end
case(A, "pass by reference") do
    p = M.at_counter()
    works(M.at_bump(p) + 1, M.at_bump(p), "the pointer the library returned")
end

const X = "std::mutex"
case(X, "make") do
    nomake(r"mutex", "`M.mutex()` is an empty struct")
end
case(X, "return by reference") do
    p = M.mx_global()
    opaque(M.mx_locked_add(p, 2, 3), 5, "a $(describe(p))")
end
case(X, "pass by reference") do
    works((M.mx_locked_add(M.mx_global(), 2, 3), M.mx_locked_add(M.mx_global(), 4, 5)), (5, 9), "locked and released twice")
end
done()
