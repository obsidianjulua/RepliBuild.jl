include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()
const RB = M.RepliBuild                 # what `using RepliBuild` gives a user

const U = "std::unique_ptr<UNode>"
const URef = M.unique_ptr_UNode_std_default_delete_UNode

case(U, "make") do
    nomake(r"unique_ptr", "the library's `up_make` is the way")
end
case(U, "return by value") do
    r = M.up_make(5)
    # Its bytes (`r[]`) are the pointer: read the object it owns.
    p = Ptr{M.UNode}(reinterpret(UInt, collect(r[]))[1])
    opaque(unsafe_load(p).v, 5, "comes back as the C++ object, a $(describe(r)), held where C++ built it; its bytes are the pointer it owns")
end
case(U, "pass by reference") do
    r = M.up_make(5)
    works((M.up_read(r), M.up_read(Ref(r))), (5, 5), "the object, and Ref(r) as the manual said while a container was its bytes")
end
case(U, "pass by reference") do
    # A pointer of the declared type still passes.
    r = Ref(M.up_make(5))
    GC.@preserve r begin
        p = Ptr{URef}(Base.unsafe_convert(Ptr{Cvoid}, r))
        works((M.up_read(p), unsafe_load(M.up_raw(p)).v), (5, 5), "a hand-made Ptr{$(nameof(URef))} to the object")
    end
end
case(U, "pass by value") do
    r = M.up_make(5)
    got = M.up_take(r)
    again = try M.up_take(r); "passed again" catch e; e isa ArgumentError ? "refused" : "threw a $(typeof(e))" end
    works((got, RB.isalive(r), again), (5, false, "refused"), "moved, as C++ moves what it cannot copy: the object is spent, and passing it again is an ArgumentError")
end
const C = "std::unique_ptr<UCounted> (counts live objects)"
case(C, "ended by Julia") do
    before = M.upc_live()
    (() -> (M.upc_make(1); M.upc_make(2); nothing))()
    GC.gc(); GC.gc()
    after = M.upc_live()
    after == before ? "WORKS both objects were deleted when Julia collected the two unique_ptrs" :
        noway("nothing ends a unique_ptr returned by value: $(after - before) of 2 objects are still alive after GC, and the wrapper has no function that would delete them")
end
case(C, "ended by release!") do
    before = M.upc_live()
    r = M.upc_make(1)
    made = M.upc_live() - before
    RB.release!(r); RB.release!(r)
    works((made, M.upc_live() - before), (1, 0), "RepliBuild.release!(r) runs the destructor at once, and once")
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
    p = Ptr{Cint}(reinterpret(UInt, collect(r[]))[1])
    opaque(unsafe_wrap(Array, p, 4), Cint[0, 1, 2, 3], "the C++ object; its bytes are the pointer to the array")
end
done()
