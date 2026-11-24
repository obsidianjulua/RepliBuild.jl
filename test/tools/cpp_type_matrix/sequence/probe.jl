include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()

const DQ = "std::deque<int>"
case(DQ, "make") do
    nomake(r"deque", "not in [types] templates, and a deque gets no constructor there either")
end
case(DQ, "return by value") do
    r = M.dq_make(5)
    opaque(M.dq_sum(Ref(r)), 10, "comes back as its bytes, a $(describe(r))")
end

const LI = "std::list<int>"
case(LI, "make") do
    nomake(r"list", "not in [types] templates, and a list gets no constructor there either")
end
case(LI, "intact after the return") do
    # A list's end marker is inside the list object. The library walks size()
    # nodes from begin() and says whether it arrived at end().
    r = fn(:list_make)(4)
    intact(M.list_intact_self(), M.list_intact(Ref(r)), 1)
end
case(LI, "return by value") do
    r = fn(:list_make)(4)
    # Not walked when the end marker is gone: the walk would not stop.
    M.list_intact(Ref(r)) == 1 || return "WRONG the list's last node points at the buffer it was returned in, not at the copy Julia holds: a walk to end() never arrives (not run)"
    opaque(M.list_sum(Ref(r)), 6, "its bytes, a $(describe(r))")
end

const BS = "std::bitset<8>"
case(BS, "make") do
    works(M.bits_count(M.bitset_8UL(0xff)), 8, "the struct's one field, a word")
end
case(BS, "pass by value") do
    works(M.bits_count(M.bitset_8UL(0b1011)), 3)
end
case(BS, "return by value") do
    works(M.bits_make(0b1011)._M_w, 0b1011, "read from the field `_M_w`")
end
done()
