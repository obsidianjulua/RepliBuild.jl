include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()
const T = "std::filesystem::path"

case(T, "make") do
    nomake(r"path", "`M.path()` is zeroed bytes, not a constructed path, and no field takes text")
end
case(T, "takes a Julia String") do
    takes(7, "a Julia String") do
        M.path_len("a/b.txt")
    end
end
case(T, "return by value") do
    p = fn(:path_make)("a/b.txt")
    opaque(M.path_len(Ref(p)), 7, "a struct of bytes, a $(describe(p)); its text is read only through the library")
end
case(T, "pass by reference") do
    p = fn(:path_make)("a/b.txt")
    works(M.path_len(Ref(p)), 7, "Ref of the value the library returned")
end
case(T, "pass by value") do
    p = fn(:path_make)("a/b.txt")
    works(M.path_ext(p), ".txt", "and the std::string it returns is a Julia String")
end
case(T, "pass by value") do
    # A long path: its characters are on the heap.
    long = "dir/" * "x"^60 * ".tar"
    p = fn(:path_make)(long)
    works((M.path_len(Ref(p)), M.path_ext(p)), (ncodeunits(long), ".tar"), "68 characters")
end
case(T, "intact after the return") do
    p = fn(:path_make)("a/b.txt")
    intact(M.path_intact_self(), M.path_intact(Ref(p)), 1)
end
case(T, "return by value") do
    a = fn(:path_make)("dir")
    j = M.path_join(Ref(a), "file.txt")
    opaque(M.path_len(Ref(j)), 12, "operator/ result")
end
done()
