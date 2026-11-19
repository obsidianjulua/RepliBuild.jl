# verify_module.jl <wrapper> — use the fixture crate through its GENERATED Julia
# module, in a process of its own (test_rust_wrap_live.jl runs it).
#
# verify.jl and verify_thunks.jl made each call by hand from the metadata.
# This makes none by hand: it loads the module `RepliBuild.wrap` wrote and
# calls it the way a user would, with Julia values in and Julia values out.
# The second half is ownership: values Rust must drop, held by Julia objects,
# moved into Rust, copied out, and dropped exactly once.
#
# One `PROBE <name>: PASS|FAIL …` line per case, then `PROBE_DONE`.

using RepliBuild

const M = include(abspath(ARGS[1]))
const RT = RepliBuild.RustRuntime

probe(name, ok, detail = "") = (println("PROBE ", name, ": ", ok ? "PASS" : "FAIL " * string(detail)); flush(stdout))
function case(body, name)
    r = try
        body()
    catch e
        return probe(name, false, sprint(showerror, e))
    end
    probe(name, r === true, r)
end
raised(f) = try
    f()
    nothing
catch e
    e
end

case("scalars") do
    (M.add(2, 3), M.c_add(20, 22), M.flag(true), M.flag(false), M.char_code('é'), M.wide(UInt128(2)^100 + 7), M.count() + 1 == M.count()) ==
        (Int32(5), Int32(42), false, true, UInt32(0xe9), 3 * (UInt128(2)^100 + 7), true)
end
case("integer_arguments_are_converted_or_refused") do
    r = raised(() -> M.add(2^40, 1))
    (M.add(Int8(2), 0x03), r isa InexactError) == (Int32(5), true)
end
case("strings_and_slices") do
    (M.str_len("héllo"), M.str_len(SubString("héllo wörld", 1, 6)), M.slice_sum([1.0, 2.0, 3.5]), M.largest_f64([3.0, 9.5, -1.0]),
     M.hello(), M.slice_sum(Float64[])) == (UInt(5), UInt(5), 6.5, 9.5, "Hello, World!", 0.0)
end
case("tuples_come_back_as_tuples") do
    M.pair(4, 1.5) === (Int32(8), 3.0)
end
# `[T; N]` is N values in a row: a tuple out, a tuple or a vector in, of exactly N.
case("fixed_size_arrays") do
    b = M.bytes16(10)
    wrong = raised(() -> M.bytes16_sum([1, 2, 3]))
    (b, M.bytes16_sum(b), M.bytes16_sum(collect(1:16)), M.grid(1.0f0), M.opt_bytes(true), M.opt_bytes(false), wrong isa ArgumentError) ==
        (ntuple(i -> UInt8(9 + i), 16), UInt32(280), UInt32(136), ((1.0f0, 2.0f0), (3.0f0, 4.0f0), (5.0f0, 6.0f0)),
         (0x01, 0x02, 0x03, 0x04, 0x05, 0x06), nothing, true)
end
case("a_slice_of_records") do
    rs = [M.rec_make(1, 2.0, 3, 4), M.rec_make(5, 6.5, 7, 8)]
    (M.recs_total(rs), M.recs_total(M.Rec[]), M.recs_total(M.recs(3))) == (20.5, 0.0, 6.0)
end
case("option_is_a_value_or_nothing") do
    (M.maybe(41), M.maybe(-1)) === (Int32(42), nothing)
end
case("records_by_value_and_by_reference") do
    rec = M.rec_make(7, 2.5, 300, 70000)
    fields = (rec.a, rec.b, rec.c, rec.d)
    built = M.Rec(7, 2.5, 300, 70000)
    (fields, rec isa M.Rec, built == rec, M.rec_sum(rec), M.rec_sum_ref(rec), M.rec_sum_ref(Ref(rec)), propertynames(rec)) ==
        ((0x07, 2.5, 0x012c, 0x00011170), true, true, 70309.5, 70309.5, 70309.5, (:a, :b, :c, :d))
end
case("two_scalar_records") do
    f = M.f2_make(1.5, 2.5)
    t = M.two16_swap(M.Two16(1, 2))
    b = M.b3_rot(M.B3(1, 2, 3))
    (f.x, f.y, M.f2_dot(f, M.F2(3.0, 4.0)), t.a, t.b, (b.a, b.b, b.c)) == (1.5, 2.5, 14.5, 0x0002, 0x0001, (0x02, 0x03, 0x01))
end
case("fieldless_enums") do
    # Optimized, `flip(Side)` and `flag(bool)` are the same code. The build keeps them two
    # functions, each with its own types; a compiler that cannot keeps only `flag`'s, and
    # `flip` is then typed as `flag` is (its docstring says so).
    flipped = isdefined(M, :Side) ? M.flip(M.Side_Left) == M.Side_Right : M.flip(true) == false
    # An enum the compiler stores signed (`core::cmp::Ordering`, where `Less` is -1) keeps its values.
    signed = (M.compare(1, 2), M.compare(2, 2), M.compare(3, 2), Int(M.Ordering_Less), M.ordering_sign(M.Ordering_Less),
              M.ordering_sign(M.Ordering_Greater)) == (M.Ordering_Less, M.Ordering_Equal, M.Ordering_Greater, -1, Int32(-1), Int32(1))
    (M.level_next(M.Level_Low), M.level_next(M.Level_High), flipped, M.Level_Mid isa M.Level, signed) ==
        (M.Level_Mid, M.Level_Low, true, true, true)
end
case("enums_with_data") do
    rect = M.rect(3.0, 4.0)
    circle = M.unit_circle()
    V = RepliBuild.RustRuntime.variant
    wrong = raised(() -> rect.Circle)
    (V(rect), rect.Rect.w, rect.Rect.h, M.area(rect), V(circle), circle.Circle.__0, M.area(circle), wrong isa ErrorException) ==
        (:Rect, 3.0, 4.0, 12.0, :Circle, 1.0, 3.0, true)
end
case("one_bit_values") do
    o = M.opts(2.0)
    (o.scale, o.on, M.opts(-1.0).on) == (2.0, true, false)
end
case("methods_take_self_first") do
    acc = Ref(M.Acc_new())
    M.Acc_push(acc, 2.0); M.Acc_push(acc, 5.0)
    (acc[].count, acc[].total, M.Acc_mean(acc), M.Acc_mean(acc[]), M.Shape_Area_area(M.rect(2.0, 5.0))) == (UInt32(2), 7.0, 3.5, 3.5, 10.0)
end
case("optional_reference") do
    rec = M.rec_make(1, 2.0, 3, 9)
    (M.opt_ref_d(Ref(rec)), M.opt_ref_d(nothing)) == (UInt32(9), UInt32(0))
end
# A Rust panic is a Julia exception with the panic's own words, and the library goes on working.
case("a_panic_is_an_exception") do
    e1 = raised(() -> M.checked_div(1, 0))
    e2 = raised(() -> M.checked_div(5, 0))
    after = M.checked_div(9, 3)
    (e1 isa RepliBuild.RustRuntime.RustPanic, occursin("division by zero in rb_fix", e1.message), occursin("lib.rs", e1.message),
     e2 isa RepliBuild.RustRuntime.RustPanic, after) == (true, true, true, true, Int32(3))
end
# What is not wrapped says so, by name and with its reason, instead of being absent.
case("unsupported_functions_say_why") do
    # `Secret` is public in a private module that nothing re-exports: no Rust
    # outside the crate can name it, so nothing can be written that drops one.
    e = raised(() -> M.secret())
    (e isa ErrorException, occursin("no drop function", e.msg), occursin("no path names `rb_fix::hidden::Secret`", M.UNSUPPORTED[:secret]),
     haskey(M.UNSUPPORTED, :located), !haskey(M.FUNCTIONS, :secret), haskey(M.FUNCTIONS, :secret_len)) ==
        (true, true, true, true, true, true)
end
case("how_each_function_calls") do
    f = M.FUNCTIONS
    # A function that may panic goes through the catching frame; one the compiler proved cannot, does not.
    (f[:checked_div], f[:maybe] in (:thunk, :guarded), f[:rec_make] in (:thunk, :guarded), length(f) + length(M.UNSUPPORTED)) ==
        (:guarded, true, true, length(RepliBuild._read_json(joinpath(dirname(ARGS[1]), "compilation_metadata.json"))["functions"]))
end

# --- what Rust must drop ---------------------------------------------------------
# A `String` is a Julia `String` on the way out and takes one on the way in.
case("strings_are_copied_out_and_built_in") do
    (M.make_greeting("julia"), M.string_len_owned("héllo"), M.string_ref_len("abc"), M.make_greeting("")) ==
        ("hello julia", UInt(6), UInt(3), "hello ")
end
# A `&mut String` needs a String that stays: the object that holds one.
case("a_string_rust_changes_in_place") do
    s = M.RustString("hey")
    M.shout(s)
    (String(s), M.string_ref_len(s), RT.isowned(s), M.string_len_owned(s), RT.isowned(s)) == ("HEY!", UInt(4), true, UInt(4), false)
end
case("a_string_must_be_utf8") do
    bad = String(UInt8[0xff, 0xfe])
    (raised(() -> M.string_len_owned(bad)) isa ArgumentError, raised(() -> M.str_len(bad)) isa ArgumentError) == (true, true)
end
case("vectors_of_plain_data") do
    empty = M.squares(0)
    r = M.recs(3)
    (M.squares(4), empty isa Vector{Float64}, isempty(empty), M.vec_sum([1.0, 2.5]), M.vec_sum(Float64[]), M.vec_sum(1:3),
     eltype(r), length(r), r[3].d, r[2].b) == ([0.0, 1.0, 4.0, 9.0], true, true, 3.5, 0.0, 6.0, M.Rec, 3, UInt32(2), 1.0)
end
case("a_vector_rust_changes_in_place") do
    v = M.Vec_f64([1.0])
    M.vec_push(v, 2.0); M.vec_push(v, 3.0)
    (length(v), collect(v), M.vec_sum(v), RT.isowned(v)) == (3, [1.0, 2.0, 3.0], 6.0, false)
end
case("vectors_of_strings") do
    (M.words("a bc  def"), M.words(""), M.join_words(["x", "yz"]), M.join_words(String[])) == (["a", "bc", "def"], String[], "x yz", "")
end
# `Result` is the value, or the error thrown.
case("result_is_a_value_or_an_exception") do
    e = raised(() -> M.parse_int("zz"))
    u = raised(() -> M.check(-1))
    (M.parse_int(" 42 "), e isa RT.RustError, e isa RT.RustError && occursin("not a number", e.value), M.check(1),
     u isa RT.RustError && u.value === nothing, occursin("not a number", sprint(showerror, e))) == (Int32(42), true, true, nothing, true, true)
end
case("option_and_tuple_of_owned_values") do
    (M.find_word("a b c", 1), M.find_word("a", 5), M.name_and_len("abc"), M.opt_box(0), M.opt_box(5)[].d) ==
        ("b", nothing, ("ABC", UInt(3)), nothing, UInt32(5))
end
# A `Box` is an object that holds the pointer; it is passed where a reference to its target is taken.
case("a_box_is_held_and_moved") do
    b = M.boxed(9)
    held = (b[].d, M.box_d(b), M.rec_sum_ref(b), RT.isowned(b))
    moved = (M.unbox_d(b), RT.isowned(b))
    again = raised(() -> M.unbox_d(b))
    used = raised(() -> M.box_d(b))
    (held, moved, again isa ErrorException && occursin("already moved", again.msg), used isa ErrorException) ==
        ((UInt32(9), UInt32(9), 15.0, true), (UInt32(9), false), true, true)
end
case("an_enum_that_owns") do
    s = M.slot_make(7)
    e = M.slot_make(0)
    (M.slot_d(s), RT.variant(s), RT.variant(e), M.slot_take(s), RT.isowned(s), M.slot_take(e)) ==
        (UInt32(7), :Full, :Empty, UInt32(7), false, UInt32(0))
end
case("a_record_that_owns") do
    n = M.named("ada", 3)
    t = M.token(5, "tok")                 # defined in a private module, named by its public path
    (n.name, n.id, propertynames(n), M.named_len(n), M.named_into_name(n), RT.isowned(n), M.token_id(t), t.text) ==
        ("ada", UInt32(3), (:name, :id), UInt(3), "ada", false, UInt32(5), "tok")
end
# `Guard` holds one integer and has drop code: only the compiler knows it must be dropped.
case("dropped_exactly_once") do
    before = M.drops()
    g = M.guard(1)
    RT.drop!(g); RT.drop!(g)                       # the second finds nothing to drop
    one = M.drops() - before
    g2 = M.guard(2)
    id = M.guard_release(g2)                       # Rust drops it inside the call
    RT.drop!(g2)
    two = M.drops() - before
    (() -> (M.guard(3); nothing))()                # never used again: Julia's collector drops it
    GC.gc(); GC.gc()
    three = M.drops() - before
    (one, id, two, three, M.guard_id(M.guard(8))) == (1, UInt32(2), 2, 3, UInt32(8))
end
# `View` holds a raw pointer and has no drop code: the compiler says it is plain data.
case("a_raw_pointer_is_not_ownership") do
    xs = [1.0, 2.0]
    v = GC.@preserve xs M.view(xs)
    (v isa M.View, isbits(v), M.view_len(v), M.view_len(v)) == (true, true, UInt(2), UInt(2))
end
case("what_the_module_says_it_owns") do
    o = M.OWNED
    (o[:Named], o[:Token], o[:RustString], haskey(o, :Guard), haskey(o, :Box_Rec), haskey(o, :View), haskey(o, :Rec)) ==
        ("rb_fix::Named", "rb_fix::inner::Token", "alloc::string::String", true, true, false, false)
end

println("PROBE_DONE")
