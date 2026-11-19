# verify_module.jl <wrapper> — use the fixture crate through its GENERATED Julia
# module, in a process of its own (test_rust_wrap_live.jl runs it).
#
# verify.jl and verify_thunks.jl made each call by hand from the metadata.
# This makes none by hand: it loads the module `RepliBuild.wrap` wrote and
# calls it the way a user would, with Julia values in and Julia values out.
#
# One `PROBE <name>: PASS|FAIL …` line per case, then `PROBE_DONE`.

using RepliBuild

const M = include(abspath(ARGS[1]))

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
    # Optimized, the compiler merges `flip(Side)` into `flag(bool)`: one function, and the library
    # keeps only `flag`'s types. `flip` is then typed as `flag` is, and its docstring says so.
    flipped = isdefined(M, :Side) ? M.flip(M.Side_Left) == M.Side_Right : M.flip(true) == false
    (M.level_next(M.Level_Low), M.level_next(M.Level_High), flipped, M.Level_Mid isa M.Level) ==
        (M.Level_Mid, M.Level_Low, true, true)
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
    V = RepliBuild.RustRuntime.variant
    o = M.opts(2.0)
    (V(M.check(1)), V(M.check(-1)), o.scale, o.on, M.opts(-1.0).on) == (:Ok, :Err, 2.0, true, false)
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
    e = raised(() -> M.make_greeting("x"))
    (e isa ErrorException, occursin("own memory", e.msg), haskey(M.UNSUPPORTED, :make_greeting), haskey(M.UNSUPPORTED, :located),
     !haskey(M.FUNCTIONS, :make_greeting), isdefined(M, :boxed)) == (true, true, true, true, true, true)
end
case("how_each_function_calls") do
    f = M.FUNCTIONS
    # A function that may panic goes through the catching frame; one the compiler proved cannot, does not.
    (f[:checked_div], f[:maybe] in (:thunk, :guarded), f[:rec_make] in (:thunk, :guarded), length(f) + length(M.UNSUPPORTED)) ==
        (:guarded, true, true, length(RepliBuild._read_json(joinpath(dirname(ARGS[1]), "compilation_metadata.json"))["functions"]))
end

println("PROBE_DONE")
