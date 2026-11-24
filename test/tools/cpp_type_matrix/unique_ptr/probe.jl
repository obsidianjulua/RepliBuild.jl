include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()

const U = "std::unique_ptr<UNode>"
const URef = M.unique_ptr_UNode_std_default_delete_UNode

case(U, "make") do
    nomake(r"unique_ptr", "the library's `up_make` is the way")
end
case(U, "return by value") do
    r = M.up_make(5)
    # The bytes are the pointer: read the object it owns.
    p = Ptr{M.UNode}(reinterpret(UInt, collect(r))[1])
    opaque(unsafe_load(p).v, 5, "comes back as its bytes, a $(describe(r)), which are the pointer it owns")
end
case(U, "pass by reference") do
    r = M.up_make(5)
    works(M.up_read(Ref(r)), 5, "Ref(r), as the manual says for a returned container")
end
case(U, "pass by reference") do
    # What does work when Ref(r) is not taken: a pointer of the declared type.
    r = Ref(M.up_make(5))
    GC.@preserve r begin
        p = Ptr{URef}(Base.unsafe_convert(Ptr{Cvoid}, r))
        works((M.up_read(p), unsafe_load(M.up_raw(p)).v), (5, 5), "a hand-made Ptr{$(nameof(URef))} to the bytes")
    end
end
case(U, "pass by value") do
    r = M.up_make(5)
    works(M.up_take(r), 5, "the bytes")
end
const C = "std::unique_ptr<UCounted> (counts live objects)"
case(C, "ended by Julia") do
    before = M.upc_live()
    M.upc_make(1); M.upc_make(2)
    GC.gc(); GC.gc()
    after = M.upc_live()
    after == before ? "WORKS both objects were deleted" :
        noway("nothing ends a unique_ptr returned by value: $(after - before) of 2 objects are still alive after GC, and the wrapper has no function that would delete them")
end
case(C, "ended by a by-value pass") do
    # C++ moves a unique_ptr into a by-value parameter, and the object is
    # deleted when the call returns.
    before = M.upc_live()
    r = M.upc_make(1)
    got = M.upc_take(r)
    left = M.upc_live() - before
    got == 1 || return "WRONG got $got, the library's answer is 1"
    left == 0 ? "WORKS the object is deleted when the call returns, as in C++" :
        noway("the object is still alive after the call: passing the bytes lends the pointer, it does not move it, and nothing ends the object later")
end

const A = "std::unique_ptr<int[]>"
case(A, "return by value") do
    r = M.upa_make(4)
    p = Ptr{Cint}(reinterpret(UInt, collect(r))[1])
    opaque(unsafe_wrap(Array, p, 4), Cint[0, 1, 2, 3], "its bytes, the pointer to the array")
end
done()
