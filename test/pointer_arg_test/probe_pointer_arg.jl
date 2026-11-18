# Subprocess probe for test_tier2_pointer_args_live.jl: passes right and wrong
# arguments where the wrapper's C++ functions take a pointer. A wrong one must
# be refused in Julia; before 2026-11-17 most of them were dereferenced, so the
# cases that work come first and the ones that used to end the process last.
# Prints one "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_pointer_arg.jl <wrapper_path>

wrapper, = ARGS
const M = include(wrapper)

function probe(f, name)
    verdict = try
        p = String[]
        f(p)
        isempty(p) ? "PASS" : "FAIL " * join(unique(p)[1:min(end, 4)], "; ")
    catch e
        "FAIL threw " * first(sprint(showerror, e), 300)
    end
    println("PROBE $name: ", verdict)
    flush(stdout)
end

# The exception a call raises, or the value it returned.
outcome(f) = try
    (:returned, f())
catch e
    (:threw, e)
end
# `what` must be refused before C++ runs: the conversion's own MethodError
# (`JITManager.pointer_arg`), or an ArgumentError where `arg` says so.
function refused!(p, what, f; arg::Bool = false)
    kind, v = outcome(f)
    kind === :returned && return push!(p, "$what was accepted and returned $(repr(v))")
    ok = arg ? v isa ArgumentError : (v isa MethodError && nameof(v.f) === :pointer_arg)
    ok || push!(p, "$what threw $(typeof(v)): $(first(sprint(showerror, v), 120))")
end

inc(x::Cint)::Cint = x + Cint(1)
const c_inc = @cfunction(inc, Cint, (Cint,))

# Storage for one T, zeroed, typed.
newobj(T) = Ptr{T}(Libc.calloc(1, max(M.struct_size(nameof(T)), 1)))

# ── What a pointer position takes ────────────────────────────────────────────

probe("this_takes_ptr_ref_and_void_pointer") do p
    v = newobj(M.Vec3)
    M.Vec3_Vec3(v, 1.0, 2.0, 2.0)
    (got = M.Vec3_length(v)) == 3.0 || push!(p, "length(Ptr{Vec3}) = $got")
    (got = M.Vec3_length(Ptr{Cvoid}(v))) == 3.0 || push!(p, "length(Ptr{Cvoid}) = $got")
    r = Ref(unsafe_load(v))
    (got = M.Vec3_length(r)) == 3.0 || push!(p, "length(Ref) = $got")
    (got = M.Vec3_dot(r, v)) == 9.0 || push!(p, "dot(Ref, Ptr) = $got")
    (got = M.Vec3_dot(v, r)) == 9.0 || push!(p, "dot(Ptr, Ref) = $got")
    # An element of an array of the class, by reference.
    a = [unsafe_load(v), M.Vec3(0.0, 3.0, 4.0)]
    (got = M.Vec3_length(Ref(a, 2))) == 5.0 || push!(p, "length(Ref(a, 2)) = $got")
    (got = M.Vec3_length(a)) == 3.0 || push!(p, "length(Vector{Vec3}) = $got")
    Libc.free(v)
end

probe("byte_record_is_passed_by_its_bytes") do p
    f = M.flags_make(33, 70000)
    ismutable(f) || push!(p, "Flags is not the mutable byte record this case is about: $(typeof(f))")
    (got = M.Flags_get_level(f)) == 33 || push!(p, "get_level(f) = $got")           # `this`
    (got = M.Flags_get_level(Ref(f))) == 33 || push!(p, "get_level(Ref(f)) = $got")
    (got = M.flags_id_of(f)) == 70000 || push!(p, "flags_id_of(f) = $got")          # a pointer parameter
    (got = M.flags_id_of(Ref(f))) == 70000 || push!(p, "flags_id_of(Ref(f)) = $got")
end

probe("owning_objects_pass_as_their_pointer") do p
    raw = M.square_new(7, 3)
    (got = M.Square_area(raw)) == 9 || push!(p, "area(raw) = $got")
    (got = M.Shape_shape_id(raw)) == 7 || push!(p, "Shape_shape_id(Ptr{Square}) = $got")
    s = M.ManagedSquare(raw)
    (got = M.Square_area(s)) == 9 || push!(p, "area(Managed) = $got")
    # A derived class's owner where the base's `this` and `const Shape*` go.
    (got = M.Shape_shape_id(s)) == 7 || push!(p, "Shape_shape_id(ManagedSquare) = $got")
    (got = M.shape_id_of(s)) == 7 || push!(p, "shape_id_of(ManagedSquare) = $got")
    # A factory struct, as `this` and as a parameter.
    t = M.Token(Cint(5))
    (got = M.Token_value(t)) == 5 || push!(p, "Token_value(factory) = $got")
    (got = M.token_value_of(t)) == 5 || push!(p, "token_value_of(factory) = $got")
    (got = M.value(t)) == 5 || push!(p, "value(factory) = $got")
end

probe("function_pointers") do p
    (got = M.apply_unary(c_inc, 4)) == 5 || push!(p, "apply_unary(Ptr) = $got")
    k = Cint(10)
    closure = @cfunction($(x -> x + k), Cint, (Cint,))      # a Base.CFunction
    (got = M.apply_unary(closure, 4)) == 14 || push!(p, "apply_unary(CFunction) = $got")
end

probe("void_pointer_takes_what_ccall_takes") do p
    M.echo_ptr(C_NULL) == C_NULL || push!(p, "echo_ptr(C_NULL)")
    a = Cint[1, 2, 3]
    GC.@preserve a begin
        M.echo_ptr(a) == Ptr{Cvoid}(pointer(a)) || push!(p, "echo_ptr(Vector)")
        M.echo_ptr(pointer(a)) == Ptr{Cvoid}(pointer(a)) || push!(p, "echo_ptr(Ptr{Cint})")
    end
    r = Ref{Cint}(5)
    GC.@preserve r (M.echo_ptr(r) == Base.unsafe_convert(Ptr{Cvoid}, r) || push!(p, "echo_ptr(Ref)"))
end

probe("data_pointers_take_arrays_of_the_pointee") do p
    a = Cint[1, 2, 3, 4]
    (got = M.sum_ints(a, 4)) == 10 || push!(p, "sum_ints(Vector) = $got")
    (got = M.sum_ints(view(a, 2:4), 3)) == 9 || push!(p, "sum_ints(view) = $got")
    (got = M.sum_ints(Ref(a, 3), 2)) == 7 || push!(p, "sum_ints(Ref(a, 3)) = $got")
    (got = M.sum_ints(Ref{Cint}(6), 1)) == 6 || push!(p, "sum_ints(Ref) = $got")
    GC.@preserve a ((got = M.sum_ints(pointer(a), 2)) == 3 || push!(p, "sum_ints(Ptr) = $got"))
    out = zeros(Cint, 3)
    M.fill_ints(out, 3, 5)
    out == Cint[5, 6, 7] || push!(p, "fill_ints wrote $out")
    # Bytes of either sign, and the bytes of a string, for `unsigned char*`.
    (got = M.count_byte(UInt8[1, 2, 1], 3, 0x01)) == 2 || push!(p, "count_byte(Vector{UInt8}) = $got")
    (got = M.count_byte(Int8[1, 2, 1], 3, 0x01)) == 2 || push!(p, "count_byte(Vector{Int8}) = $got")
    (got = M.count_byte(codeunits("abca"), 4, UInt8('a'))) == 2 || push!(p, "count_byte(codeunits) = $got")
    (got = M.count_byte("a\0a", 3, UInt8('a'))) == 2 || push!(p, "count_byte(String with a NUL) = $got")
end

# B9: the value the library returned for a container is what a by-value
# parameter takes, and a pointer or `Ref` to an existing container still is.
probe("container_by_value_takes_the_returned_value") do p
    v = M.vec_iota(5)
    (got = M.vec_sum_val(v)) == 10 || push!(p, "vec_sum_val(value) = $got")
    r = Ref(v)
    (got = (M.vec_sum_val(r), M.vec_sum_val(r), M.vec_sum(r))) == (10, 10, 10) || push!(p, "through a Ref, twice, then by reference: $got")
    GC.@preserve r begin
        ptr = Base.unsafe_convert(Ptr{Cvoid}, r)
        (got = (M.vec_sum_val(ptr), M.vec_sum(ptr))) == (10, 10) || push!(p, "through a Ptr: $got")
    end
end

probe("std_string_objects_behind_pointers") do p
    s = M.str_new("abc")
    M.str_append(s, "def")
    (got = M.str_size(s)) == 6 || push!(p, "size after append = $got")
    # A `const std::string&` built from a Julia string keeps its NULs: it is
    # passed with its length, not as a C string.
    (got = M.str_size_ref("a\0b")) == 3 || push!(p, "str_size_ref(\"a\\0b\") = $got")
    (got = M.str_size_ref(s)) == 6 || push!(p, "str_size_ref(object) = $got")
    M.str_delete(s)
end

# B16: Julia's `Cstring` semantics where C++ takes a `char*`.
probe("c_strings") do p
    (got = M.cstr_len("hello")) == 5 || push!(p, "cstr_len(String) = $got")
    (got = M.cstr_len("héllo")) == 6 || push!(p, "cstr_len(UTF-8) = $got")
    (got = M.cstr_len(SubString("hello world", 1, 5))) == 5 || push!(p, "cstr_len(SubString) = $got (read past the substring)")
    (got = M.overloaded(SubString("abcdef", 1, 3))) == "bc" || push!(p, "overloaded(SubString) = $(repr(got))")
    (got = M.overloaded("abc")) == "bc" || push!(p, "overloaded(String) = $(repr(got))")
    buf = UInt8['h', 'i', 0]
    (got = M.cstr_len(buf)) == 2 || push!(p, "cstr_len(Vector{UInt8}) = $got")
    GC.@preserve buf ((got = M.cstr_len(pointer(buf))) == 2 || push!(p, "cstr_len(Ptr{UInt8}) = $got"))
    refused!(p, "a string with an embedded NUL for a `const char*`", () -> M.cstr_len("ab\0cd"); arg = true)
    refused!(p, "a string with an embedded NUL in the overload set", () -> M.overloaded("ab\0cd"); arg = true)
end

probe("overloads_still_dispatch") do p
    M.overloaded(Cint(1)) === Cint(2) || push!(p, "int")
    M.overloaded(Clonglong(1)) === Clonglong(3) || push!(p, "long long")
    M.overloaded(1.0) === 1.5 || push!(p, "double")
    M.overloaded(true) === false || push!(p, "bool")
end

# ── What it refuses. Each of these was read as an address. ───────────────────

probe("string_is_not_a_void_pointer_or_an_object") do p
    refused!(p, "a String for `void*`", () -> M.echo_ptr("abc"))
    refused!(p, "a String for `const int*`", () -> M.sum_ints("abcd", 1))
end

probe("array_of_another_type_is_refused") do p
    refused!(p, "a Vector{Float64} for `const int*`", () -> M.sum_ints([1.0, 2.0], 2))
    refused!(p, "a Vector{Int64} for `const int*`", () -> M.sum_ints([1, 2], 2))
    refused!(p, "a Vector{Cint} for `const unsigned char*`", () -> M.count_byte(Cint[1], 1, 0x01))
end

probe("scalar_does_not_select_the_char_pointer_overload") do p
    refused!(p, "overloaded(Int16(1))", () -> M.overloaded(Int16(1)))
    refused!(p, "overloaded(UInt64(1))", () -> M.overloaded(UInt64(1)))
    refused!(p, "overloaded(Cchar(65))", () -> M.overloaded(Cchar(65)))
    refused!(p, "cstr_len(65)", () -> M.cstr_len(65))
end

probe("julia_function_is_not_a_function_pointer") do p
    refused!(p, "a Julia function for `int (*)(int)`", () -> M.apply_unary(inc, 4))
    refused!(p, "a Ref for `int (*)(int)`", () -> M.apply_unary(Ref{Cint}(1), 4))
    refused!(p, "a Vector for `int (*)(int)`", () -> M.apply_unary(Cint[1], 4))
end

probe("struct_value_is_not_this") do p
    v = M.Vec3(1.0, 2.0, 2.0)
    refused!(p, "a Vec3 value as `this`", () -> M.Vec3_length(v))
    refused!(p, "a Vec3 value for `const Vec3*`", () -> M.Vec3_dot(Ref(v), v))
    refused!(p, "an integer as `this`", () -> M.Vec3_length(0))
    refused!(p, "`nothing` as `this`", () -> M.Vec3_length(nothing))
    refused!(p, "a tuple of the class's bytes as `this`", () -> M.Vec3_length((1.0, 2.0, 2.0)))
end

probe("julia_vector_is_not_a_std_vector") do p
    refused!(p, "a Vector{Cint} for `const std::vector<int>&`", () -> M.vec_sum(Cint[1, 2, 3]))
    refused!(p, "a Vector{Cint} for `std::vector<int>` by value", () -> M.vec_sum_val(Cint[1, 2, 3]))
    refused!(p, "the bare bytes for `const std::vector<int>&`", () -> M.vec_sum(M.vec_iota(3)))
    refused!(p, "bytes of another size for `std::vector<int>` by value", () -> M.vec_sum_val(ntuple(_ -> 0x00, 16)))
end

probe("ref_to_a_julia_object_is_not_c_data") do p
    refused!(p, "a Ref{String} for `std::string&`", () -> M.str_append(Ref("abc"), "def"))
    refused!(p, "a Ref{Vector} for `const int*`", () -> M.sum_ints(Ref(Cint[1, 2]), 2))
    refused!(p, "a Ref{Any} for `const Vec3*`", () -> M.Vec3_length(Ref{Any}(1)))
    s = M.ManagedSquare(M.square_new(1, 2))
    refused!(p, "a Ref to an owning object as `this`", () -> M.Square_area(Ref(s)))
end

println("PROBE_DONE")
