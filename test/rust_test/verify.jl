# verify.jl <library> <metadata> — call the fixture crate's functions from Julia
# with what `compilation_metadata.json` says about them, in a process of its
# own (test_rust_build_live.jl runs it: a wrong call here is a crash).
#
# Nothing in the crate is `extern "C"` except the two functions that say so.
# Each call below is made with plain `ccall`, which is right exactly where the
# function's IR signature is scalars and pointers; before a call, the probe
# checks that the metadata states the signature the `ccall` assumes, so a
# toolchain that lowers a function differently is a FAIL with the signature in
# it, never a wrong call. Functions that return two scalars (`Option<i32>`, a
# `&str`, a two-field record) need a thunk that states the compiler's own
# signature; they are checked in the metadata only.
#
# One `PROBE <name>: PASS|FAIL …` line per case, then `PROBE_DONE`.

using RepliBuild, Libdl

const LIB = ARGS[1]
const MD = RepliBuild._read_json(ARGS[2])
const H = Libdl.dlopen(LIB)
const FN = Dict(f["path"] => f for f in MD["functions"])
const TYPES = MD["types"]

probe(name, ok, detail = "") = (println("PROBE ", name, ": ", ok ? "PASS" : "FAIL " * string(detail)); flush(stdout))
sym(path) = Libdl.dlsym(H, FN[path]["symbol"])
sig(path) = (FN[path]["ir"]["return"]["type"], [p["type"] for p in FN[path]["ir"]["parameters"]])
# Run `body` only when the metadata states the signature the call below is written for.
function with_sig(body, name, path, ret, params)
    haskey(FN, path) || return probe(name, false, "no function $path in the metadata")
    FN[path]["refusal"] === nothing || return probe(name, false, "refused: $(FN[path]["refusal"])")
    sig(path) == (ret, params) || return probe(name, false, "signature is $(sig(path))")
    r = try
        body()
    catch e
        return probe(name, false, sprint(showerror, e))
    end
    probe(name, r === true, r)
end
offset(type, field) = only(m["offset"] for m in TYPES[type]["fields"] if m["name"] == field)
# A hidden return pointer is the first argument on x86-64. On AArch64 it has a
# register of its own, which a `ccall` cannot name: those calls need the thunk.
const SRET_BY_CCALL = Sys.ARCH === :x86_64

with_sig("add", "rb_fix::add", "i32", ["i32", "i32"]) do
    ccall(sym("rb_fix::add"), Cint, (Cint, Cint), 2, 3) == 5
end
with_sig("c_add", "c_add", "i32", ["i32", "i32"]) do
    ccall(sym("c_add"), Cint, (Cint, Cint), 20, 22) == 42
end
with_sig("flag", "rb_fix::flag", "i1", ["i1"]) do
    (ccall(sym("rb_fix::flag"), UInt8, (UInt8,), 1) & 1, ccall(sym("rb_fix::flag"), UInt8, (UInt8,), 0) & 1) == (0, 1)
end
with_sig("char_code", "rb_fix::char_code", "i32", ["i32"]) do
    ccall(sym("rb_fix::char_code"), UInt32, (UInt32,), UInt32('é')) == 0xe9
end
with_sig("level_next", "rb_fix::level_next", "i8", ["i8"]) do
    [ccall(sym("rb_fix::level_next"), UInt8, (UInt8,), l) for l in 0:2] == [1, 2, 0]
end
with_sig("checked_div", "rb_fix::checked_div", "i32", ["i32", "i32"]) do
    ccall(sym("rb_fix::checked_div"), Cint, (Cint, Cint), 9, 3) == 3
end
with_sig("count_static", "rb_fix::count", "i64", String[]) do
    a = ccall(sym("rb_fix::count"), UInt64, ())
    ccall(sym("rb_fix::count"), UInt64, ()) == a + 1
end

# A slice and a `&str` are a pointer and a length: two arguments, no NUL.
with_sig("slice_sum", "rb_fix::slice_sum", "double", ["ptr", "i64"]) do
    xs = [1.0, 2.0, 3.5]
    ccall(sym("rb_fix::slice_sum"), Cdouble, (Ptr{Cdouble}, Csize_t), xs, length(xs)) == 6.5
end
with_sig("str_len_substring", "rb_fix::str_len", "i64", ["ptr", "i64"]) do
    s = SubString("héllo wörld", 1, 6)                  # "héllo", inside a longer string
    ccall(sym("rb_fix::str_len"), Csize_t, (Ptr{UInt8}, Csize_t), pointer(s), ncodeunits(s)) == 5
end
with_sig("largest_f64", "rb_fix::largest_f64", "double", ["ptr", "i64"]) do
    xs = [3.0, 9.5, -1.0]
    ccall(sym("rb_fix::largest_f64"), Cdouble, (Ptr{Cdouble}, Csize_t), xs, length(xs)) == 9.5
end

# Methods take `self` first. A zeroed `Acc` is `Acc::new()`; its fields are where the metadata says.
with_sig("acc_push", "<rb_fix::Acc>::push", "void", ["ptr", "double"]) do
    with_sig("acc_mean", "<rb_fix::Acc>::mean", "double", ["ptr"]) do
        acc = zeros(UInt8, TYPES["rb_fix::Acc"]["size"])
        for x in (2.0, 4.0)
            ccall(sym("<rb_fix::Acc>::push"), Cvoid, (Ptr{UInt8}, Cdouble), acc, x)
        end
        total = reinterpret(Float64, acc[offset("rb_fix::Acc", "total")+1:offset("rb_fix::Acc", "total")+8])[1]
        count = reinterpret(UInt32, acc[offset("rb_fix::Acc", "count")+1:offset("rb_fix::Acc", "count")+4])[1]
        (total, count, ccall(sym("<rb_fix::Acc>::mean"), Cdouble, (Ptr{UInt8},), acc)) == (6.0, 2, 3.0)
    end
    true
end

# A `Box<Rec>` is one pointer that owns a `Rec`; `Option<&Rec>` is a pointer or null;
# `unbox_d` takes the box by value, which frees it.
with_sig("box_roundtrip", "rb_fix::boxed", "ptr", ["i32"]) do
    with_sig("opt_ref_some_none", "rb_fix::opt_ref_d", "i32", ["ptr"]) do
        b = ccall(sym("rb_fix::boxed"), Ptr{UInt8}, (UInt32,), 9)
        d = unsafe_load(Ptr{UInt32}(b + offset("rb_fix::Rec", "d")))
        some = ccall(sym("rb_fix::opt_ref_d"), UInt32, (Ptr{UInt8},), b)
        none = ccall(sym("rb_fix::opt_ref_d"), UInt32, (Ptr{UInt8},), C_NULL)
        freed = ccall(sym("rb_fix::unbox_d"), UInt32, (Ptr{UInt8},), b)
        (d, some, none, freed) == (9, 9, 0, 9)
    end
    true
end
# The niche: `Slot::Empty` is the null pointer, as the variant's discriminant value says.
with_sig("slot_niche", "rb_fix::slot_make", "ptr", ["i32"]) do
    slot = TYPES["rb_fix::Slot"]
    empty = only(v for v in slot["variants"] if v["name"] == "Empty")
    full = only(v for v in slot["variants"] if v["name"] == "Full")
    e = ccall(sym("rb_fix::slot_make"), Ptr{Cvoid}, (UInt32,), 0)
    f = Ref(ccall(sym("rb_fix::slot_make"), Ptr{Cvoid}, (UInt32,), 7))
    d = ccall(sym("rb_fix::slot_d"), UInt32, (Ptr{Ptr{Cvoid}},), f)
    (UInt(e) == empty["discr_value"], full["default"], f[] != C_NULL, d) == (true, true, true, 7)
end

if SRET_BY_CCALL
    # A record comes back through a hidden pointer, with its fields where the
    # compiler put them (it reordered them), and goes in by pointer to a copy.
    with_sig("rec_make_layout", "rb_fix::rec_make", "void", ["ptr", "i8", "double", "i16", "i32"]) do
        rec = zeros(UInt8, TYPES["rb_fix::Rec"]["size"])
        ccall(sym("rb_fix::rec_make"), Cvoid, (Ptr{UInt8}, UInt8, Cdouble, UInt16, UInt32), rec, 7, 2.5, 300, 70000)
        at(T, f) = reinterpret(T, rec[offset("rb_fix::Rec", f)+1:offset("rb_fix::Rec", f)+sizeof(T)])[1]
        byval = ccall(sym("rb_fix::rec_sum"), Cdouble, (Ptr{UInt8},), rec)
        byref = ccall(sym("rb_fix::rec_sum_ref"), Cdouble, (Ptr{UInt8},), rec)
        (at(UInt8, "a"), at(Float64, "b"), at(UInt16, "c"), at(UInt32, "d"), byval, byref) ==
            (7, 2.5, 300, 70000, 70309.5, 70309.5)
    end
    # An enum with data: built by Rust, its discriminant where and what the metadata says.
    with_sig("shape_discriminant", "rb_fix::rect", "void", ["ptr", "double", "double"]) do
        shape = TYPES["rb_fix::Shape"]
        buf = zeros(UInt8, shape["size"])
        ccall(sym("rb_fix::rect"), Cvoid, (Ptr{UInt8}, Cdouble, Cdouble), buf, 3.0, 4.0)
        d = shape["discriminant"]
        tag = reinterpret(UInt64, buf[d["offset"]+1:d["offset"]+8])[1]
        rect = only(v for v in shape["variants"] if v["name"] == "Rect")
        w = reinterpret(Float64, buf[offset(rect["type"], "w")+1:offset(rect["type"], "w")+8])[1]
        (d["size"], tag == rect["discr_value"], w, ccall(sym("rb_fix::area"), Cdouble, (Ptr{UInt8},), buf)) ==
            (8, true, 3.0, 12.0)
    end
    # A `String` is returned into the caller's buffer and freed by the function that takes it by value.
    with_sig("string_out_and_in", "rb_fix::make_greeting", "void", ["ptr", "ptr", "i64"]) do
        st = TYPES["alloc::string::String"]
        buf = zeros(UInt8, st["size"])
        name = "julia"
        ccall(sym("rb_fix::make_greeting"), Cvoid, (Ptr{UInt8}, Ptr{UInt8}, Csize_t), buf, name, ncodeunits(name))
        leaves = RepliBuild.RustABI.shape(TYPES, "alloc::string::String").leaves
        words = [reinterpret(UInt64, buf[l.offset+1:l.offset+8])[1] for l in leaves]
        ptr = only(w for (w, l) in zip(words, leaves) if l.class === :ptr)
        text = unsafe_string(Ptr{UInt8}(ptr), 11)
        len = ccall(sym("rb_fix::string_len_owned"), Csize_t, (Ptr{UInt8},), buf)     # takes it: Rust frees it
        (st["size"], length(leaves), 11 in words, text, len) == (24, 3, true, "hello julia", 11)
    end
end

println("PROBE_DONE")
