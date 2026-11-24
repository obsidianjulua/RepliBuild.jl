include(joinpath(@__DIR__, "probe_support.jl"))
const M = load_wrapper()

const O = "std::optional<int>"
# The value and the "has a value" flag, read by hand from the struct's bytes.
bytes_of(o) = collect(o._M_payload._data)
int_of(o) = reinterpret(Cint, bytes_of(o)[1:4])[1]
engaged(o) = bytes_of(o)[5] != 0x00

case(O, "make") do
    works(M.opt_or(M.optional_int(), 9), 9, "an empty one: `M.optional_int()`, all zero")
end
case(O, "make (holding a value)") do
    fields = fieldnames(M.optional_int)
    noway("the struct's only field is $(fields[1]), a block of bytes: a value and its flag have to be written into the bytes by hand")
end
case(O, "pass by value") do
    works((M.opt_or(M.opt_make(5, true), 9), M.opt_or(M.opt_make(5, false), 9)), (5, 9))
end
case(O, "pass by reference") do
    works((M.opt_or_ref(Ref(M.opt_make(5, true)), 9), M.opt_or_ref(Ref(M.opt_make(5, false)), 9)), (5, 9))
end
case(O, "return by value") do
    a = M.opt_make(5, true); b = M.opt_make(5, false)
    opaque((int_of(a), engaged(a), engaged(b)), (Cint(5), true, false), "a struct of bytes: the value and the flag are read by hand at their offsets, no field names them")
end

const D = "std::optional<double>"
case(D, "return by value") do
    a = M.optd_make(2.5, true)
    by = collect(a._M_payload._data)
    opaque((reinterpret(Float64, by[1:8])[1], by[9] != 0x00), (2.5, true), "a struct of bytes")
end

const S = "std::optional<std::string>"
case(S, "return by value") do
    a = fn(:opts_make)(true); b = fn(:opts_make)(false)
    opaque((M.opts_len(Ref(a)), M.opts_len(Ref(b))), (4, -1), "a struct of bytes; the string inside is not a Julia String")
end
case(S, "intact after the return") do
    a = fn(:opts_make)(true)
    intact(M.opts_intact_self(), M.opts_intact(Ref(a)), 1)
end
done()
