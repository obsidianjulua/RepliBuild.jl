# Subprocess probe for test_placeholder_references_live.jl: a wrong pointer
# here is read as an object, so this runs in a child. Prints one
# "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_handles.jl <wrapper_path>

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
refused(f) = try f(); "the call was made" catch e; e isa MethodError ? "" : "threw a $(typeof(e))" end

# The bytes a smart pointer was returned as, in a `Ref`: the manual's way for a
# returned container, at a reference to the smart pointer.
probe("smart_pointer_bytes_by_reference") do p
    r = Ref(M.sp_make(5))
    check(p, "sp_read", M.sp_read(r), 5)
    check(p, "sp_count", M.sp_count(r), 1)
    check(p, "up_read", M.up_read(Ref(M.up_make(7))), 7)
end

# The reference is to the caller's object: the library writes it, and the next
# read sees what was written.
probe("reference_is_the_callers_object") do p
    r = Ref(M.sp_make(5))
    M.sp_set(r, 9)
    check(p, "after sp_set", (M.sp_read(r), M.sp_count(r)), (9, 1))
end

# A std::function* as the library returned it, at a const std::function&: the
# return is typed, so it is the pointer the reference takes.
probe("std_function_pointer_as_returned") do p
    f = M.fn_twice()
    check(p, "the return's type", typeof(f), Ptr{M.function_int_int})
    check(p, "fn_apply_ref", (M.fn_apply_ref(f, 21), M.fn_apply_ref(f, 4)), (42, 8))
    check(p, "fn_empty", M.fn_empty(f), 0)
end

# A pointer typed by the placeholder passes as it always did.
probe("typed_pointer_still_passes") do p
    r = Ref(M.sp_make(3))
    GC.@preserve r begin
        t = Ptr{M.shared_ptr_Node}(Base.unsafe_convert(Ptr{Cvoid}, r))
        check(p, "sp_read", M.sp_read(t), 3)
    end
end

# Nothing else: refused before any C++ runs. An untyped pointer is anything,
# and a Ref of other bytes is another type's value.
probe("anything_else_is_refused") do p
    bytes = M.sp_make(5)
    for (what, x) in (("an integer", 5), ("a string", "text"), ("a tuple", (1, 2)), ("a Julia function", x -> x),
                      ("the bytes themselves, not a Ref", bytes), ("a vector", [1, 2, 3]),
                      ("an untyped pointer", Ptr{Cvoid}(UInt(0x1000))),
                      ("a Ref of a unique_ptr's bytes", Ref(M.up_make(1))))
        why = refused(() -> M.sp_read(x))
        isempty(why) || push!(p, "sp_read($what): $why")
    end
end

# A C function pointer is no std::function: where one is taken by reference it
# is a MethodError. (Taken as any pointer, its code was read as the object.)
twice(x::Cint)::Cint = 2x
probe("function_pointer_is_no_std_function") do p
    cf = @cfunction(twice, Cint, (Cint,))
    check(p, "the pointer is a working callback", M.fn_plain(cf, 21), 42)
    for (what, x) in (("a @cfunction pointer", cf), ("a Julia function", twice), ("an untyped pointer", Ptr{Cvoid}(M.fn_twice())))
        why = refused(() -> M.fn_apply_ref(x, 1))
        isempty(why) || push!(p, "fn_apply_ref($what): $why")
    end
end

# `which(const shared_ptr&)` beside `which(shared_ptr*)`: two C++ functions,
# two Julia methods. A typed pointer reaches the reference's, and so does a Ref
# of the bytes.
probe("overload_set_keeps_two_methods") do p
    check(p, "methods of which", length(methods(M.which)), 2)
    r = Ref(M.sp_make(1))
    GC.@preserve r begin
        t = Ptr{M.shared_ptr_Node}(Base.unsafe_convert(Ptr{Cvoid}, r))
        check(p, "which(typed pointer)", M.which(t), 1)
        check(p, "which(Ref of the bytes)", M.which(r), 1)
    end
end

println("PROBE_DONE")
