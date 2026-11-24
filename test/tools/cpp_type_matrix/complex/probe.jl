include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()

const C = "std::complex<double>"
case(C, "make") do
    works(M.cx_abs(M.complex_double(3.0 + 4.0im)), 5.0, "the struct, from a ComplexF64")
end
case(C, "takes a Julia Complex") do
    takes(5.0, "a ComplexF64") do
        M.cx_abs(3.0 + 4.0im)
    end
end
case(C, "pass by value") do
    works(M.cx_abs(M.cx_make(3.0, 4.0)), 5.0)
end
case(C, "pass by reference") do
    works(M.cx_abs_ref(Ref(M.complex_double(3.0 + 4.0im))), 5.0)
end
case(C, "return by value") do
    works(M.cx_make(3.0, 4.0)._M_value, 3.0 + 4.0im, "a ComplexF64 in the field `_M_value`")
end

case(C, "a double parameter takes an Integer") do
    takes(5.0, "an Int where C++ takes a double") do
        M.cx_abs(M.cx_make(3, 4))
    end
end

const F = "std::complex<float>"
case(F, "make") do
    works(M.cxf_re(M.complex_float(1.5f0 + 2.0f0im)), 1.5f0)
end
case(F, "return by value") do
    works(M.cxf_make(1.5f0, 2.0f0)._M_value, 1.5f0 + 2.0f0im)
end
done()
