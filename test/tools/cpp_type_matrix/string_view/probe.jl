include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()
const T = "std::string_view"
const SV = M.basic_string_view_char_std_char_traits_char

# A view over a Julia string, built field by field: the length, then the pointer.
view_of(s::String) = SV(ncodeunits(s), pointer(s))
text_of(v) = unsafe_string(v._M_str, v._M_len)

case(T, "make") do
    s = "hello"
    GC.@preserve s works(M.sv_len(view_of(s)), 5, "built by hand from pointer(s) and its length; the caller keeps the String alive")
end
case(T, "takes a Julia String") do
    takes(5, "a Julia String") do
        M.sv_len("hello")
    end
end
case(T, "pass by value") do
    s = "hello"
    GC.@preserve s works((M.sv_len(view_of(s)), Char(M.sv_at(view_of(s), 1))), (5, 'e'))
end
case(T, "pass by reference") do
    s = "hello"
    GC.@preserve s works(M.sv_len_ref(Ref(view_of(s))), 5)
end
case(T, "return by value") do
    works(text_of(M.sv_literal()), "literal", "a struct of length and pointer, read with unsafe_string")
end
case(T, "return by value") do
    s = "hello"
    GC.@preserve s works(text_of(M.sv_tail(view_of(s), 2)), "lo", "a view into the caller's String")
end
done()
