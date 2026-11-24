include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()
const T = "std::string"

case(T, "make") do
    works(M.str_len_ref("héllo"), 6, "a Julia String is taken where a std::string is")
end
case(T, "pass by value") do
    works((M.str_len_val("hello"), M.str_len_val("")), (5, 0))
end
case(T, "pass by reference") do
    works(M.str_len_ref("hello"), 5, "const std::string&")
end
case(T, "return by value") do
    works(fn(:str_make)(3), "aaa", "a Julia String")
end
case(T, "return by value") do
    # Longer than any short-string buffer, so the characters are on the heap.
    works(fn(:str_make)(100), "a"^100, "100 characters")
end
case(T, "return by reference") do
    p = fn(:str_global)()
    opaque(M.str_len_ref(p), 11, "const std::string& comes back as a $(describe(p)): it passes back to the library, and nothing the manual names reads its text")
end
case(T, "return by reference") do
    p = fn(:str_global_ptr)()
    opaque(M.str_len_ref(p), 11, "std::string* comes back as a $(describe(p))")
end

# A std::string the function writes into: the manual says to pass a string
# object, and `create_std_basic_string_char()` is the one a user can make.
const W = "std::string& (written by the function)"
case(W, "make") do
    s = M.create_std_basic_string_char()
    M.str_append(s, "abc")
    works(String(s), "abc", "create_std_basic_string_char(), a CppString; filled by the library, read back with String(s)")
end
case(W, "an empty CppString reads back as \"\"") do
    works(String(M.create_std_basic_string_char()), "")
end
case(W, "pass by reference") do
    s = M.create_std_basic_string_char()
    M.str_append(s, "abc"); M.str_append(s, "de")
    works((String(s), length(s)), ("abcde", 5), "through a CppString")
end
case(W, "takes a Julia String") do
    takes(nothing, "a Julia String") do
        M.str_append("abc", "de")
    end
end
case(W, "pass by value") do
    s = M.create_std_basic_string_char(); M.str_append(s, "abcd")
    works((M.str_len_val(s), M.str_len_val(s)), (4, 4), "a CppString, passed twice")
end
done()
