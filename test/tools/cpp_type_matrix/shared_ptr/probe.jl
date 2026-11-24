include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()

const S = "std::shared_ptr<SNode>"
const SRef = M.shared_ptr_SNode
asptr(r::Base.RefValue) = Ptr{SRef}(Base.unsafe_convert(Ptr{Cvoid}, r))

case(S, "make") do
    nomake(r"shared_ptr", "the library's `sp_make` is the way")
end
case(S, "return by value") do
    r = M.sp_make(5)
    p = Ptr{M.SNode}(reinterpret(UInt, collect(r))[1])
    opaque(unsafe_load(p).v, 5, "comes back as its bytes, a $(describe(r)): the object pointer, then the control block's")
end
case(S, "pass by reference") do
    r = M.sp_make(5)
    works(M.sp_read(Ref(r)), 5, "Ref(r), as the manual says for a returned container")
end
case(S, "pass by reference") do
    r = Ref(M.sp_make(5))
    GC.@preserve r works((M.sp_read(asptr(r)), M.sp_count(asptr(r)), unsafe_load(M.sp_raw(asptr(r))).v), (5, 1, 5), "a hand-made Ptr{shared_ptr_SNode} to the bytes")
end
case(S, "pass by value") do
    r = M.sp_make(5)
    works((M.sp_take(r), M.sp_take(r)), (5, 5), "the bytes, passed twice")
end
case(S, "use count after a by-value pass") do
    # C++ copies a shared_ptr into a by-value parameter and ends the copy when
    # the call returns: the count is 1 again.
    r = Ref(M.sp_make(5))
    GC.@preserve r begin
        M.sp_take(r[]); M.sp_take(r[])
        works(M.sp_count(asptr(r)), 1)
    end
end

const W = "std::weak_ptr<SNode>"
case(W, "return by value") do
    r = Ref(M.sp_make(5))
    GC.@preserve r begin
        w = M.wp_of(asptr(r))
        works((M.wp_alive(Ref(w)), w._M_ptr != C_NULL), (1, true), "a struct with the object pointer as a field")
    end
end
case(W, "pass by reference") do
    r = Ref(M.sp_make(5))
    GC.@preserve r works(M.wp_alive(Ref(M.wp_of(asptr(r)))), 1, "Ref of the returned struct")
end
done()
