# verify_thunks.jl <library> <metadata> — call the fixture crate through thunks
# generated from its metadata (`RustThunks`), in a process of its own
# (test_rust_thunks_live.jl runs it: a wrong call here is a crash).
#
# verify.jl makes the calls `ccall` can make. These are the others, and the
# reason the dialect has `jlcs.native_call`: a pair returned in two registers
# (`Option<i32>`, a `&str`, a two-field record, a tuple, a trait object), a
# three-byte integer, a 128-bit one, a `bool`, and a hidden return pointer
# passed with its attribute, which is how it reaches the right register on
# every host. Every argument and result is in the function's own layout, read
# from the metadata.
#
# One `PROBE <name>: PASS|FAIL …` line per case, then `PROBE_DONE`.

using RepliBuild

const RT = RepliBuild.RustThunks
const LIB = ARGS[1]
const MD = RepliBuild._read_json(ARGS[2])
const FN = Dict(f["path"] => f for f in MD["functions"])
const TYPES = MD["types"]
const S = RT.load(LIB, MD)

probe(name, ok, detail = "") = (println("PROBE ", name, ": ", ok ? "PASS" : "FAIL " * string(detail)); flush(stdout))
function case(body, name)
    r = try
        body()
    catch e
        return probe(name, false, sprint(showerror, e))
    end
    probe(name, r === true, r)
end
sym(path) = FN[path]["symbol"]
size_of(type) = TYPES[type]["size"]
offset(type, field) = only(m["offset"] for m in TYPES[type]["fields"] if m["name"] == field)
at(buf, T, off) = reinterpret(T, buf[off+1:off+sizeof(T)])[1]
# Call `path` with IR-level arguments; the result's bytes, sized by its return type.
function call(path, args...; bytes = nothing)
    row = FN[path]
    n = bytes !== nothing ? bytes : row["return"]["type"] === nothing ? 0 : something(size_of(row["return"]["type"]), 0)
    buf = zeros(UInt8, max(n, 0))
    RT.call_raw(S, row["symbol"], buf, args...)
    return buf
end

case("thunks_for_every_callable") do
    callable = [f["symbol"] for f in MD["functions"] if f["refusal"] === nothing]
    missing_thunks = [s for s in callable if !haskey(S.thunks, s)]
    refused = sort!([f["path"] for f in MD["functions"] if f["refusal"] !== nothing])
    (isempty(missing_thunks), length(S.thunks) == length(callable), sort!([p for (p, f) in FN if haskey(S.skipped, f["symbol"])]) == refused) ==
        (true, true, true) ? true : (missing_thunks, length(S.thunks), length(callable), S.skipped)
end

# --- two scalars returned in two registers ------------------------------------
case("option_i32") do
    opt = TYPES["core::option::Option<i32>"]
    some = call("rb_fix::maybe", Int32(41)); none = call("rb_fix::maybe", Int32(-1))
    tag(b) = at(b, UInt32, opt["discriminant"]["offset"])
    val = only(v for v in opt["variants"] if v["name"] == "Some")
    payload = at(some, Int32, offset(val["type"], "__0"))
    (tag(some) == val["discr_value"], payload, tag(none) == only(v for v in opt["variants"] if v["name"] == "None")["discr_value"]) ==
        (true, 42, true) ? true : (tag(some), payload, tag(none))
end
case("two_field_record") do
    r = call("rb_fix::two16_swap", UInt16(1), UInt16(2))
    (at(r, UInt16, offset("rb_fix::Two16", "a")), at(r, UInt16, offset("rb_fix::Two16", "b"))) == (2, 1)
end
case("record_of_two_doubles") do
    f = call("rb_fix::f2_make", 1.5, 2.5)
    dot = call("rb_fix::f2_dot", at(f, Float64, 0), at(f, Float64, 8), 3.0, 4.0)
    (at(f, Float64, offset("rb_fix::F2", "x")), at(f, Float64, offset("rb_fix::F2", "y")), at(dot, Float64, 0)) == (1.5, 2.5, 14.5)
end
case("tuple") do
    t = call("rb_fix::pair", Int32(4), 1.5)
    (at(t, Int32, offset("(i32, f64)", "__0")), at(t, Float64, offset("(i32, f64)", "__1"))) == (8, 3.0)
end
case("str_returned") do
    s = call("rb_fix::hello")
    unsafe_string(at(s, Ptr{UInt8}, offset("&str", "data_ptr")), at(s, UInt, offset("&str", "length"))) == "Hello, World!"
end
case("method_returning_a_record") do
    acc = call("<rb_fix::Acc>::new")
    for x in (2.0, 5.0)
        RT.call_raw(S, sym("<rb_fix::Acc>::push"), UInt8[], pointer(acc), x)
    end
    mean = GC.@preserve acc call("<rb_fix::Acc>::mean", pointer(acc))
    (at(acc, UInt32, offset("rb_fix::Acc", "count")), at(mean, Float64, 0)) == (2, 3.5)
end
# A boxed trait object is two pointers, made by one function and passed on to another.
case("trait_object") do
    areas = map((0, 1)) do k
        b = call("rb_fix::make_area", UInt32(k))
        at(call("rb_fix::dyn_area", at(b, Ptr{Cvoid}, 0), at(b, Ptr{Cvoid}, 8)), Float64, 0)
    end
    areas == (3.0, 6.0)
end

# --- scalars `ccall` has no type for ------------------------------------------
case("three_byte_integer") do
    call("rb_fix::b3_rot", (0x01, 0x02, 0x03)) == [0x02, 0x03, 0x01]
end
case("u128") do
    x = UInt128(2)^100 + 7
    at(call("rb_fix::wide", x), UInt128, 0) == 3x
end
case("bool") do
    (call("rb_fix::flag", true)[1], call("rb_fix::flag", false)[1]) == (0x00, 0x01)
end
case("fieldless_enum") do
    [call("rb_fix::level_next", UInt8(l))[1] for l in 0:2] == [1, 2, 0]
end
case("char_and_plain_scalars") do
    (at(call("rb_fix::char_code", UInt32('é')), UInt32, 0), at(call("rb_fix::add", Int32(2), Int32(3)), Int32, 0),
     at(call("c_add", Int32(20), Int32(22)), Int32, 0), at(call("rb_fix::checked_div", Int32(9), Int32(3)), Int32, 0)) ==
        (0xe9, 5, 42, 3)
end
# A zero-sized argument is not passed, and a zero-sized result is not returned.
case("zero_sized") do
    isempty(call("rb_fix::z", Int32(1)))
end

# --- a hidden return pointer, on every host -------------------------------------
case("record_through_hidden_pointer") do
    rec = call("rb_fix::rec_make", UInt8(7), 2.5, UInt16(300), UInt32(70000))
    fields = (at(rec, UInt8, offset("rb_fix::Rec", "a")), at(rec, Float64, offset("rb_fix::Rec", "b")),
              at(rec, UInt16, offset("rb_fix::Rec", "c")), at(rec, UInt32, offset("rb_fix::Rec", "d")))
    byval = GC.@preserve rec at(call("rb_fix::rec_sum", pointer(rec)), Float64, 0)
    byref = GC.@preserve rec at(call("rb_fix::rec_sum_ref", pointer(rec)), Float64, 0)
    (fields, byval, byref) == ((7, 2.5, 300, 70000), 70309.5, 70309.5)
end
case("enum_with_data") do
    shape = TYPES["rb_fix::Shape"]
    buf = call("rb_fix::rect", 3.0, 4.0)
    rect = only(v for v in shape["variants"] if v["name"] == "Rect")
    tag = at(buf, UInt64, shape["discriminant"]["offset"])
    area = GC.@preserve buf at(call("rb_fix::area", pointer(buf)), Float64, 0)
    unit = call("rb_fix::unit_circle")
    circle = only(v for v in shape["variants"] if v["name"] == "Circle")
    (tag == rect["discr_value"], area, at(unit, UInt64, 0) == circle["discr_value"]) == (true, 12.0, true)
end
case("string_out_and_in") do
    name = "julia"
    buf = GC.@preserve name call("rb_fix::make_greeting", pointer(name), UInt(ncodeunits(name)))
    leaves = RepliBuild.RustABI.shape(TYPES, "alloc::string::String").leaves
    words = [at(buf, UInt64, l.offset) for l in leaves]
    ptr = only(w for (w, l) in zip(words, leaves) if l.class === :ptr)
    text = unsafe_string(Ptr{UInt8}(ptr), 11)
    len = GC.@preserve buf at(call("rb_fix::string_len_owned", pointer(buf)), UInt, 0)   # takes it: Rust frees it
    (length(buf), 11 in words, text, len) == (24, true, "hello julia", 11)
end

# --- pointers: a slice, a box, a niche -----------------------------------------
case("slices") do
    xs = [1.0, 2.0, 3.5]
    s = "héllo wörld"
    GC.@preserve xs s (at(call("rb_fix::slice_sum", pointer(xs), UInt(3)), Float64, 0),
                       at(call("rb_fix::str_len", pointer(s), UInt(6)), UInt, 0),
                       at(call("rb_fix::largest_f64", pointer(xs), UInt(3)), Float64, 0)) == (6.5, 5, 3.5)
end
case("box_and_niche") do
    b = at(call("rb_fix::boxed", UInt32(9)), Ptr{UInt8}, 0)
    some = at(call("rb_fix::opt_ref_d", b), UInt32, 0)
    none = at(call("rb_fix::opt_ref_d", Ptr{UInt8}(C_NULL)), UInt32, 0)
    freed = at(call("rb_fix::unbox_d", b), UInt32, 0)
    empty = call("rb_fix::slot_make", UInt32(0))
    full = call("rb_fix::slot_make", UInt32(7))
    d = GC.@preserve full at(call("rb_fix::slot_d", pointer(full)), UInt32, 0)
    (some, none, freed, at(empty, UInt, 0), at(full, UInt, 0) != 0, d) == (9, 0, 9, 0, true, 7)
end

# --- the bookkeeping refuses a wrong call before making it ----------------------
case("wrong_calls_are_errors") do
    msg(f) = try
        f()
        ""
    catch e
        sprint(showerror, e)
    end
    width = msg(() -> RT.call_raw(S, sym("rb_fix::add"), zeros(UInt8, 4), Int64(2), Int32(3)))
    count = msg(() -> RT.call_raw(S, sym("rb_fix::add"), zeros(UInt8, 4), Int32(2)))
    refused = msg(() -> RT.call_raw(S, sym("rb_fix::located"), zeros(UInt8, 4), Int32(2)))
    unknown = msg(() -> RT.call_raw(S, "no_such_symbol", zeros(UInt8, 4)))
    (occursin("4 bytes", width), occursin("2 IR arguments", count), occursin("track_caller", refused),
     occursin("no thunk", unknown)) == (true, true, true, true) ? true : (width, count, refused, unknown)
end

println("PROBE_DONE")
