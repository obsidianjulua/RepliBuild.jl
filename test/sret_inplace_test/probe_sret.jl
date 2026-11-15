# Subprocess probe for test_sret_in_caller_buffer.jl. A wrong return convention
# leaves a dangling self-pointer or crashes, so this runs in a child. Prints one
# "PROBE <case>: PASS|FAIL …" line each, then PROBE_DONE.
# Run: julia --project=<RepliBuild> probe_sret.jl <wrapper_path>

using RepliBuild

wrapper, = ARGS
const M = include(wrapper)
const FG = RepliBuild.JLCSIRGenerator.FunctionGen
const MD = RepliBuild._read_json(joinpath(dirname(wrapper), "compilation_metadata.json"))
const SD = MD["struct_definitions"]
const RA = get(MD, "record_abi", Dict{String,Any}())

function probe(f, name)
    verdict = try
        p = String[]
        f(p)
        isempty(p) ? "PASS" : "FAIL " * join(p, "; ")
    catch e
        "FAIL threw " * first(sprint(showerror, e), 300)
    end
    println("PROBE $name: ", verdict)
end
check(p, what, got, want) = got == want || push!(p, "$what: got $got, want $want")
fn(name) = MD["functions"][something(findfirst(f -> get(f, "name", "") == name, MD["functions"]),
                                     findfirst(f -> occursin(string(length(name), name), get(f, "mangled", "")), MD["functions"]))]

# make_self(7), called the way the thunk generator says its thunk takes it:
# in the caller's buffer when `returns_in_caller_buffer` holds, else the old
# shape (the thunk returns the bytes). Either way the object then lives in
# `buf`, and its `self` must be `buf`'s address.
probe("built_in_the_callers_buffer") do p
    f = fn("make_self")
    inplace = FG.returns_in_caller_buffer(f, SD, RA)
    check(p, "make_self returns in the caller's buffer", inplace, true)
    name = "_mlir_ciface_" * FG.thunk_symbol(f["mangled"], inplace)
    buf = Ref{M.SelfRef}()
    GC.@preserve buf begin
        addr = Base.unsafe_convert(Ptr{Cvoid}, buf)
        if inplace
            RepliBuild.JITManager.invoke(name, addr, Int32(7))
        else
            buf[] = RepliBuild.JITManager.invoke(name, M.SelfRef, Int32(7))
        end
        s = buf[]
        check(p, "self == the buffer", UInt(s.self), UInt(addr))
        check(p, "v, w", (s.v, s.w), (Int32(7), 3.5))
        # C++'s own view of the object where it lives.
        check(p, "self_ok", M.self_ok(buf), Int32(1))
        check(p, "self_value", M.self_value(buf), Int32(7))
    end
end

# An STL container by value is pass-by-reference too. A short string keeps its
# characters inside the object, so built in the caller's buffer its data
# points into that buffer (C++'s own check, so it holds for libstdc++ and
# libc++ alike); built in a temporary and moved, it points elsewhere.
probe("stl_string_built_in_the_callers_buffer") do p
    f = fn("short_str")
    inplace = FG.returns_in_caller_buffer(f, SD, RA)
    check(p, "short_str returns in the caller's buffer", inplace, true)
    name = "_mlir_ciface_" * FG.thunk_symbol(f["mangled"], inplace)
    buf = Ref{NTuple{32,UInt8}}()
    GC.@preserve buf begin
        addr = Base.unsafe_convert(Ptr{Cvoid}, buf)
        if inplace
            RepliBuild.JITManager.invoke(name, addr, Int32(2))
        else
            buf[] = RepliBuild.JITManager.invoke(name, NTuple{32,UInt8}, Int32(2))
        end
        check(p, "str_data_inside", M.str_data_inside(addr), Int32(1))
        check(p, "str_size", M.str_size(addr), Int32(2))
    end
end

# A std::string return is a Julia String (StdStrings.take_string!): read where
# C++ built it, then ended there by its destructor. Ended on a moved copy, a
# short string's destructor freed a stack address; a thousand calls of each
# size run the destructor on both of its branches.
probe("std_string_is_a_julia_string") do p
    check(p, "short", M.short_str_abi_cxx11(2), "cc")
    check(p, "heap", M.long_str_abi_cxx11(100), "x"^100 * "100")
    for i in 1:1000
        M.short_str_abi_cxx11(i % 20) == string(Char('a' + i % 20))^2 || (push!(p, "short call $i"); break)
        M.long_str_abi_cxx11(40 + i % 50) == "x"^(40 + i % 50) * string(40 + i % 50) || (push!(p, "heap call $i"); break)
    end
    # On a miss, say WHY the generator kept the bytes: the reason it wrote beside
    # the call, and the layout the build recorded (the four gates of
    # `_string_return_plan` are all visible from these two).
    if !isempty(p)
        src = read(wrapper, String)
        m = match(r"# std::string `basic_string<char[^`]*` returned as its raw bytes: ([^\n]*)", src)
        push!(p, m === nothing ? "wrapper gives no raw-bytes reason for the char string" : "wrapper reason: " * m.captures[1])
        k = findfirst(startswith("basic_string<char"), collect(keys(RA)))
        if k === nothing
            push!(p, "record_abi has no basic_string<char…> key; keys: " * join(filter(startswith("basic_string"), collect(keys(RA))), ", "))
        else
            key = collect(keys(RA))[k]
            lay = get(RA[key], "string_layout", nothing)
            push!(p, lay === nothing ? "record_abi[$key] has no string_layout" :
                  "string_layout: unread=" * repr(get(lay, "unread", nothing)) * " char_type=" * repr(get(lay, "char_type", nothing)) *
                  " destructor=" * repr(get(lay, "destructor", nothing)) * " leaves=" *
                  join([string(get(l, "path", "?"), "@", get(l, "offset", get(l, "data_bit_offset", "?"))) for l in get(lay, "leaves", Any[])], ","))
        end
    end
end

# A wide string is not decoded: it keeps its raw bytes, and the wrapper says why.
probe("wstring_keeps_its_bytes") do p
    check(p, "wide_str type", typeof(M.wide_str_abi_cxx11(3)), NTuple{32,UInt8})
    src = read(wrapper, String)
    occursin(r"# std::string `basic_string<wchar_t[^`]*` returned as its raw bytes: its character type is `wchar_t`", src) ||
        push!(p, "no wchar_t reason beside wide_str's call")
end

# The generated wrapper's own call: it allocates the buffer, calls the in-place
# thunk, and returns the value, with the Julia return type it always had.
probe("wrapper_call_site") do p
    r = M.make_self(9)
    check(p, "make_self returns", typeof(r), M.SelfRef)
    check(p, "v, w", (r.v, r.w), (Int32(9), 4.5))
    check(p, "self was set", r.self != C_NULL, true)
end

# A wrapper generated before the change calls the old name with the old shape.
# It must fail at the call, saying why, never reach the in-place thunk.
probe("stale_call_says_rewrap") do p
    old = "_mlir_ciface_" * fn("make_self")["mangled"] * "_thunk"
    msg = try
        RepliBuild.JITManager.invoke(old, M.SelfRef, Int32(7))
        "no error"
    catch e
        sprint(showerror, e)
    end
    occursin("_inplace` is: this wrapper was generated before", msg) ||
        push!(p, "old-name call: " * first(msg, 200))
end

# A std::string PARAMETER takes a Julia String (2026-11-15): the wrapper builds
# the object in its own slot, passes its address, and ends it after the call.
# Short (in the object) and long (heap) strings, by const& and by value.
probe("std_string_parameter_from_julia_string") do p
    check(p, "str_size short", M.str_size("hello"), Int32(5))
    check(p, "str_size long", M.str_size("x"^100), Int32(100))
    check(p, "str_data_inside short", M.str_data_inside("ab"), Int32(1))
    check(p, "str_size_v by value", M.str_size_v("by value"), Int32(8))
    check(p, "str_size_v long by value", M.str_size_v("y"^77), Int32(77))
    check(p, "two strings", M.str_both("abc", "de"), Int32(302))
    check(p, "a SubString passes", M.str_size(SubString("hello world", 1, 5)), Int32(5))
    check(p, "signature", first(methods(M.str_size)).sig.parameters[2], Union{AbstractString, Ptr{Cvoid}, RepliBuild.STLWrappers.CppString})
    for i in 1:500
        M.str_size("s"^(i % 60)) == Int32(i % 60) || (push!(p, "call $i"); break)
    end
end

# The overload pair: a Julia String picks the std::string one, a Cstring the
# const char* one; both are methods of one Julia function.
probe("std_string_overload_beside_const_char") do p
    check(p, "String → std::string", M.kind_of("abc"), Int32(1))
    c = Base.cconvert(Cstring, "abc")
    GC.@preserve c check(p, "Cstring → const char*", M.kind_of(Base.unsafe_convert(Cstring, c)), Int32(2))
    check(p, "two methods", length(methods(M.kind_of)), 2)
end

# A pointer to an existing std::string passes through: the one C++ returned
# into the caller's buffer, untouched and never ended by the parameter path.
probe("std_string_parameter_pointer_passthrough") do p
    f = fn("short_str")
    name = "_mlir_ciface_" * FG.thunk_symbol(f["mangled"], true)
    buf = Ref{NTuple{32,UInt8}}()
    GC.@preserve buf begin
        addr = Base.unsafe_convert(Ptr{Cvoid}, buf)
        RepliBuild.JITManager.invoke(name, addr, Int32(3))
        r0 = RepliBuild.StdStrings.RELEASES[]
        check(p, "str_size through the pointer", M.str_size(addr), Int32(2))
        check(p, "a pointer is not released", RepliBuild.StdStrings.RELEASES[] - r0, 0)
    end
end

# A throw through the call still ends the slot: `release!` runs in a finally.
probe("std_string_parameter_ended_on_throw") do p
    r0 = RepliBuild.StdStrings.RELEASES[]
    check(p, "no throw", M.str_throw("fine"), Int32(4))
    check(p, "released after a return", RepliBuild.StdStrings.RELEASES[] - r0, 1)
    threw = try
        M.str_throw("boom")
        false
    catch e
        occursin("boom", sprint(showerror, e))
    end
    check(p, "the C++ exception reached Julia", threw, true)
    check(p, "released after a throw", RepliBuild.StdStrings.RELEASES[] - r0, 2)
end

# The control: a trivially copyable record takes the wrapper's own call, as
# it always did.
probe("trivial_record_unchanged") do p
    check(p, "make_plain predicate", FG.returns_in_caller_buffer(fn("make_plain"), SD, RA), false)
    r = M.make_plain(4)
    check(p, "make_plain", (r.a, r.b, r.c), (Int32(4), Int32(5), Int32(6)))
end

println("PROBE_DONE")
