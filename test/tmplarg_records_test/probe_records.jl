# Subprocess probe for test_template_arg_records_live.jl. A wrong return
# convention here writes through an argument register or reads a buffer that is
# too small, so this runs in a child. Prints one "PROBE <case>: PASS|FAIL …"
# line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_records.jl <wrapper_path>

using RepliBuild

wrapper, = ARGS
const M = include(wrapper)

function probe(f, name)
    verdict = try
        p = String[]
        f(p)
        isempty(p) ? "PASS" : "FAIL " * join(p, "; ")
    catch e
        "FAIL threw " * first(sprint(showerror, e), 300)
    end
    println("PROBE $name: ", verdict)
    flush(stdout)
end
check(p, what, got, want) = got == want || push!(p, "$what: got $got, want $want")

# A pointer of the type a by-reference parameter is declared with, to the bytes
# a by-value return gave. (`Ref` of the bytes is not taken at a smart pointer's
# reference: a separate defect the type matrix lists.)
typed(::Type{T}, r::Base.RefValue) where {T} = Ptr{T}(Base.unsafe_convert(Ptr{Cvoid}, r))
# The Julia struct the wrapper emitted for a template instance, by its prefix.
emitted(prefix) = getfield(M, only(n for n in names(M; all = true)
                                   if startswith(string(n), prefix) && getfield(M, n) isa DataType))

# std::unique_ptr<int[]>: eight bytes, built in the caller's buffer. Returned
# in a register instead, the callee stored through the first argument register.
probe("unique_ptr_array_return") do p
    r = Ref(M.upa_make(5))
    check(p, "the value's size", sizeof(r[]), 8)
    GC.@preserve r begin
        a = typed(emitted("unique_ptr_int"), r)
        check(p, "upa_at", [M.upa_at(a, i) for i in 0:4], [0, 3, 6, 9, 12])
    end
end

probe("shared_ptr_array_return") do p
    r = Ref(M.spa_make(4))
    check(p, "the value's size", sizeof(r[]), 16)
    GC.@preserve r begin
        a = typed(emitted("shared_ptr_int"), r)
        check(p, "spa_at", [M.spa_at(a, i) for i in 0:3], [100, 101, 102, 103])
        check(p, "use count", M.spa_count(a), 1)
    end
end

probe("vector_of_pointers_return") do p
    v = M.vp_make()
    check(p, "vp_sum", M.vp_sum(Ref(v)), 18)
end

# The wrapper's value for a std::vector<bool> is as large as the library says
# the object is, whatever that is in this C++ library.
probe("vector_bool_size") do p
    check(p, "sizeof the returned value", sizeof(M.vb_make(1)), M.vb_sizeof())
    check(p, "a vector<int> is still its own size", sizeof(M.vp_make()), M.vi_sizeof())
end

# Enough bits to need more than one word, and a second vector after it: a
# buffer too small for the object is overrun by the first.
probe("vector_bool_return") do p
    v = M.vb_make(200)
    w = M.vb_make(7)
    GC.gc()
    check(p, "length", M.vb_len(Ref(v)), 200)
    check(p, "set bits", M.vb_count(Ref(v)), 100)
    check(p, "second vector", (M.vb_len(Ref(w)), M.vb_count(Ref(w))), (7, 4))
end

probe("vector_bool_by_value") do p
    v = M.vb_make(9)
    check(p, "passed twice", (M.vb_count_val(v), M.vb_count_val(v)), (5, 5))
end

# C++ passes nothing for an empty class, so `a` and `b` are the first two
# arguments. (Right before the fix too, from the record's layout; held here so
# that the fact the build now records for it changes nothing.)
probe("empty_record_is_dropped") do p
    Tag = emitted("Tag_int")
    check(p, "tag_pick", M.tag_pick(Tag(), 3, 4), 34)
    check(p, "tag_pick again", M.tag_pick(Tag(), 9, 1), 91)
end

# A class with a destructor: returned through a hidden pointer. Taken for a
# register return, the callee built it at `p`, the first argument.
probe("class_with_destructor_return") do p
    before = (unsafe_load(M.cell(0)), unsafe_load(M.cell(1)))
    h = M.hold(M.cell(0), 7)
    check(p, "the library's cells are untouched", (unsafe_load(M.cell(0)), unsafe_load(M.cell(1))), before)
    check(p, "fields", (h.v, h.serial), (M.cell(0), Cint(7)))
    check(p, "held_ref", M.held_ref(Ref(h)), 5 + 7)
end

# And taken by value through an address: the callee reads the object. Passed
# in registers, it took the object's first eight bytes for that address. (The
# copy constructor adds 1000 to `serial`: 1008 once the temporary is made with
# it. Today it is a copy of the bytes, because the class's functions are
# filed under the demangler's `Holder<int*>` and its layout under DWARF's
# `Holder<int *>`; that is not this test's question.)
probe("class_with_destructor_by_value") do p
    h = M.hold(M.cell(1), 2)
    got = M.held(h)
    got in (8, 1008) || push!(p, "held: got $got, want 6 + 2 (and 1000 more through the copy constructor)")
    check(p, "the caller's value is unchanged", (h.serial, M.held_ref(Ref(h))), (Cint(2), 8))
    check(p, "a second pass", M.held(h), got)
end

# std::function by value: C++ passes the address of the caller's object.
probe("std_function_by_value_takes_a_pointer") do p
    f = M.fn_twice()
    check(p, "fn_apply", (M.fn_apply(f, 21), M.fn_apply(f, 4)), (42, 8))
end

# Nothing but a pointer to one: a Julia function is refused before any C++ runs.
probe("std_function_by_value_refuses_a_julia_function") do p
    for (what, arg) in (("a Julia function", x -> 2x), ("an integer", 7), ("a tuple", (1, 2)))
        err = try M.fn_apply(arg, 21); nothing catch e; e end
        err isa MethodError || push!(p, "$what: " * (err === nothing ? "the call was made" : "threw a $(typeof(err))"))
    end
end

println("PROBE_DONE")
