include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()

const W = "std::wstring"
case(W, "make") do
    nomake(r"wstring|wchar|basic_string", "no Julia value is converted either: wide strings are not decoded in either direction")
end
case(W, "takes a Julia String") do
    takes(3, "a Julia String") do
        M.ws_len("abc")
    end
end
case(W, "return by value") do
    r = fn(:ws_make)(3)
    opaque(M.ws_len(Ref(r)), 3, "comes back as its bytes, a $(describe(r)); the characters are not decoded")
end
case(W, "return by value") do
    # Enough characters to be on the heap.
    r = fn(:ws_make)(40)
    opaque(M.ws_len(Ref(r)), 40, "40 characters")
end
case(W, "pass by reference") do
    r = fn(:ws_make)(3)
    works(M.ws_len(Ref(r)), 3, "Ref of the bytes the library returned")
end
case(W, "intact after the return") do
    # A short string keeps its characters inside itself; the library checks
    # that the copy Julia holds still does.
    r = fn(:ws_make)(3)
    intact(M.ws_intact_self(), M.ws_intact(Ref(r)), 1)
end

const U = "std::u16string"
case(U, "return by value") do
    r = fn(:u16_make)(4)
    opaque(M.u16_len(Ref(r)), 4, "its bytes, a $(describe(r))")
end
case(U, "intact after the return") do
    r = fn(:u16_make)(4)
    intact(M.u16_intact_self(), M.u16_intact(Ref(r)), 1)
end

const U8 = "std::u8string"
case(U8, "return by value") do
    r = fn(:u8_make)(2)
    r isa String ? works(r, "xx") : noway("comes back as a $(describe(r)), and no function of the library takes one to read it")
end
done()
