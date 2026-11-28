include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()

const VI = "std::vector<int>"
case(VI, "make") do
    v = M.create_std_vector_int()
    push!(v, 2); push!(v, 5); push!(v, 9)
    works((length(v), v[1], v[3], collect(v)), (3, Cint(2), Cint(9), Cint[2, 5, 9]), "create_std_vector_int(), push!, indexing")
end
case(VI, "takes a Julia Vector") do
    takes(6, "a Vector{Cint}") do
        M.vec_sum_ref(Cint[1, 2, 3])
    end
end
case(VI, "pass by reference") do
    v = M.create_std_vector_int(); push!(v, 2); push!(v, 5)
    works(M.vec_sum_ref(v), 7, "const std::vector<int>&, a CppVector")
end
case(VI, "pass by reference") do
    v = M.create_std_vector_int(); push!(v, 2)
    M.vec_push(v, 40)
    works((length(v), v[2]), (2, Cint(40)), "std::vector<int>& the library grows")
end
case(VI, "pass by value") do
    v = M.create_std_vector_int(); push!(v, 2); push!(v, 5)
    works((M.vec_sum_val(v), M.vec_sum_val(v), length(v)), (7, 7, 2), "a CppVector, passed twice")
end
case(VI, "return by value") do
    r = M.vec_make(4)
    opaque((M.vec_sum_val(r), M.vec_sum_ref(Ref(r))), (10, 10), "comes back as the C++ object, a $(describe(r)): it passes back by value (a copy) and by reference, and its elements cannot be read from Julia")
end

const VD = "std::vector<double>"
case(VD, "make") do
    v = M.create_std_vector_double(); push!(v, 1.5); push!(v, 2.0)
    works((length(v), v[2], M.vecd_sum(v)), (2, 2.0, 3.5))
end
case(VD, "return by value") do
    r = M.vecd_make(4)
    opaque(M.vecd_sum(Ref(r)), 2.0, "the C++ object")
end

const VP = "std::vector<P2> (a library struct)"
case(VP, "make") do
    v = M.create_std_vector_P2()
    push!(v, M.P2(1, 2)); push!(v, M.P2(10, 20))
    works((length(v), v[2], M.vecp_sum(v)), (2, M.P2(10, 20), 33))
end
case(VP, "return by value") do
    r = M.vecp_make(3)
    opaque(M.vecp_sum(Ref(r)), 9, "the C++ object")
end

const VS = "std::vector<std::string>"
case(VS, "make") do
    nomake(r"vector.*string", "the entry is in [types] templates and gets no constructor")
end
case(VS, "return by value") do
    r = fn(:vecs_make)(3)
    opaque(M.vecs_total(Ref(r)), 6, "the C++ object; the strings cannot be read from Julia")
end

const VV = "std::vector<std::vector<int>>"
case(VV, "make") do
    nomake(r"vector.*vector", "the entry is in [types] templates and gets no constructor")
end
case(VV, "return by value") do
    r = M.vecv_make(3)
    opaque(M.vecv_total(Ref(r)), 9, "the C++ object")
end

const VB = "std::vector<bool>"
case(VB, "make") do
    nomake(r"vector.*bool", "not in [types] templates")
end
case(VB, "return by value") do
    r = M.vecb_make(5)
    opaque(M.vecb_count(Ref(r)), 5, "the C++ object, a $(describe(r))")
end
done()
