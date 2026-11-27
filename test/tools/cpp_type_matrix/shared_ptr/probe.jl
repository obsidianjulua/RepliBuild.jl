include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()

const RB = M.RepliBuild                 # what `using RepliBuild` gives a user
const S = "std::shared_ptr<SNode>"
const SRef = M.shared_ptr_SNode
asptr(r::Base.RefValue) = Ptr{SRef}(Base.unsafe_convert(Ptr{Cvoid}, r))

case(S, "make") do
    nomake(r"shared_ptr", "the library's `sp_make` is the way")
end
case(S, "return by value") do
    r = M.sp_make(5)
    p = Ptr{M.SNode}(reinterpret(UInt, collect(r[]))[1])
    opaque(unsafe_load(p).v, 5, "comes back as the C++ object, a $(describe(r)), held where C++ built it; its bytes are the object pointer, then the control block's")
end
case(S, "pass by reference") do
    r = M.sp_make(5)
    works((M.sp_read(r), M.sp_read(Ref(r))), (5, 5), "the object, and Ref(r) as the manual said while a container was its bytes")
end
case(S, "pass by reference") do
    r = Ref(M.sp_make(5))
    GC.@preserve r works((M.sp_read(asptr(r)), M.sp_count(asptr(r)), unsafe_load(M.sp_raw(asptr(r))).v), (5, 1, 5), "a hand-made Ptr{shared_ptr_SNode} to the object")
end
case(S, "pass by value") do
    r = M.sp_make(5)
    works((M.sp_take(r), M.sp_take(r), RB.isalive(r)), (5, 5, true), "the object, passed twice: C++ copies it, and the caller keeps its own")
end
case(S, "use count after a by-value pass") do
    # C++ copies a shared_ptr into a by-value parameter and ends the copy when
    # the call returns: the count is 1 again.
    r = M.sp_make(5)
    M.sp_take(r); M.sp_take(r)
    works(M.sp_count(r), 1)
end
case(S, "copied and ended by Julia") do
    # `copy` is the shared_ptr's own copy constructor, and ending a copy is its
    # destructor: the library's use count follows both.
    r = M.sp_make(5)
    c = copy(r)
    two = M.sp_count(r)
    RB.release!(c)
    one = M.sp_count(r)
    (() -> (copy(r); copy(r); nothing))()
    GC.gc(); GC.gc()
    works((two, one, M.sp_count(r)), (2, 1, 1), "copy(r) is a second owner; release!, or a collection, ends it")
end

const W = "std::weak_ptr<SNode>"
case(W, "return by value") do
    r = M.sp_make(5)
    w = M.wp_of(r)
    works((M.wp_alive(w), w._M_ptr != C_NULL), (1, true), "the C++ object, a $(describe(w)); the object pointer reads as a field")
end
case(W, "pass by reference") do
    r = M.sp_make(5)
    w = M.wp_of(r)
    works((M.wp_alive(w), M.wp_alive(Ref(w))), (1, 1), "the returned object, and Ref of it")
end
done()
