include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()

const OSS = "std::ostringstream"
const OS = "std::ostream"
const OStream = M.basic_ostream_char_std_char_traits_char

case(OSS, "make") do
    nomake(r"stream", "the library's `oss_new` is the way")
end
case(OSS, "return by reference") do
    o = fn(:oss_new)()
    try
        opaque(M.oss_text(o), "", "a pointer, a $(describe(o)); its text is read through the library, as a Julia String")
    finally
        M.oss_delete(o)
    end
end
case(OSS, "pass by reference") do
    o = fn(:oss_new)()
    try
        works(M.oss_text(o), "", "const std::ostringstream&, the pointer")
    finally
        M.oss_delete(o)
    end
end

case(OS, "pass by reference") do
    # A std::ostringstream where a std::ostream& is taken, as C++ allows.
    o = fn(:oss_new)()
    try
        M.os_write(o, 5)
        works(M.oss_text(o), "v=5", "a std::ostringstream* passed as it came")
    finally
        M.oss_delete(o)
    end
end
case(OS, "pass by reference") do
    # The same, with the pointer converted by hand to the base class.
    o = fn(:oss_new)()
    try
        M.os_write(Ptr{OStream}(o), 5)
        works(M.oss_text(o), "v=5", "the pointer converted by hand: Ptr{$(nameof(OStream))}(o)")
    finally
        M.oss_delete(o)
    end
end
case(OS, "return by reference") do
    p = M.os_cout()
    p isa Ptr && p != C_NULL ? "OPAQUE std::cout comes back as a $(describe(p))" : "WRONG got $(repr(p))"
end

const IS = "std::istream"
case(IS, "pass by reference") do
    nomake(r"stream", "`is_read` is bound and takes a `Ref{$(nameof(M.basic_istream_char_std_char_traits_char))}`; no function of the wrapper makes an input stream")
end
done()
