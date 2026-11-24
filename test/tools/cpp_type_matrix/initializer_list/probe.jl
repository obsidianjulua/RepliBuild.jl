include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()

const I = "std::initializer_list<int>"
case(I, "make") do
    v = Cint[1, 2, 3, 4]
    GC.@preserve v works(M.il_sum(M.initializer_list_int(pointer(v), length(v))), 10, "built by hand from pointer(v) and its length; the caller keeps the Vector alive")
end
case(I, "takes a Julia Vector") do
    takes(10, "a Vector{Cint}") do
        M.il_sum(Cint[1, 2, 3, 4])
    end
end
case(I, "pass by value") do
    v = Cint[5, 6]
    GC.@preserve v works(M.il_sum(M.initializer_list_int(pointer(v), 2)), 11)
end
case(I, "pass by reference") do
    v = Cint[5, 6, 7]
    GC.@preserve v works(M.il_count(Ref(M.initializer_list_int(pointer(v), 3))), 3)
end

const D = "std::initializer_list<double>"
case(D, "pass by value") do
    v = [0.5, 1.5]
    GC.@preserve v works(M.il_sumd(M.initializer_list_double(pointer(v), 2)), 2.0)
end
done()
