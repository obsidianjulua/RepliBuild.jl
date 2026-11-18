#!/usr/bin/env julia
# test/test_tier2_pointer_args.jl — what a Tier 2 call accepts where C++ takes
# a pointer (2026-11-17).
#
# A pointer parameter and every `this` of a C++ wrapper is `::Any`, and
# `JITManager.invoke` boxed whatever arrived, because the argument-packing plan
# cannot know which position is a pointer: a struct value passed as `this`, a
# Julia function where a function pointer goes, a `Vector` for a
# `const std::vector<int>&` were each read as an address. The same mistakes are
# a `MethodError` in a C wrapper, where `ccall` converts to the declared type.
#
# A Tier 2 body now converts every pointer argument itself, with
# `JITManager.pointer_arg(<declared type>, x)`: the one derivation of what a
# pointer position accepts. The generator names the declared type
# (`_pointer_arg_type`: `Ptr{T}`, `Ptr{Cvoid}`, `Cstring`, `Ptr{Opaque}`,
# `Ptr{Code}`, `ByValue{V}`) and spells the conversion (`_tier2_invoke_args`).
# Three things rode on the same derivation:
#
#   * a record held as its bytes (a bitfield record, a named union) passes as
#     `this` by the address of those bytes, where it crashed;
#   * an STL container taken BY VALUE, which the thunk takes by address, takes
#     the value the library returned for one (`ByValue`), and its signature is
#     a pointer's (`_by_address_ctype`), in the loop and in the overload
#     prediction alike;
#   * a `char*` takes a string with Julia's `Cstring` semantics: a `SubString`
#     is copied and NUL-terminated, an embedded NUL is an `ArgumentError`.
#
# No toolchain: `pointer_arg` on Julia values, and the C++ generator on
# synthetic metadata. The live half is `test_tier2_pointer_args_live.jl`.
#
# Negative checks (each made, seen to fail in the testset it names, restored):
#   * `_plain_data(T) = true`: "a Ref must point at C data";
#   * `_array_matches` returning `true`: "an array is data of the pointee's type";
#   * `_is_byte_record(T) = false`: "a record held as its bytes";
#   * `pointer(x)` for the `Cstring` branch: "a char* takes a C string";
#   * no `PointerArg` branch in `_arg_marshal_plan`: "the plan";
#   * `_tier2_invoke_args` ignoring `pointer_types`: every body check;
#   * `_pointer_arg_type` answering `Ptr{UInt8}` for a `char*`, `Ptr{Cvoid}`
#     for an object nothing emits or for a function pointer, `nothing` for a
#     container by value: "_pointer_arg_type" and the body checks;
#   * `_by_address_ctype` returning `ctype`: "a container by value is spelled
#     as a pointer"; in `_integer_width_overloads` alone: "the prediction is in
#     step with the loop" (one `take` method where two are expected);
#   * no `Ptr{Cvoid}` conversion for `Managed<X>`: "an owning type converts to
#     an untyped pointer".

using Test
using Libdl
using RepliBuild

const _PA_W = RepliBuild.Wrapper
const _PA_J = RepliBuild.JITManager

# A record the wrapper holds as its bytes, an owning type, and a layout struct,
# shaped as the C++ generator emits them.
mutable struct _PABits
    _data::NTuple{4, UInt8}
end
mutable struct _PAOwner
    handle::Ptr{Cvoid}
end
Base.unsafe_convert(::Type{Ptr{Cvoid}}, o::_PAOwner) = o.handle
struct _PAVec
    x::Float64
    y::Float64
end
struct _PAEmpty end

# A refusal is one error everywhere: `pointer_arg`'s own MethodError.
_pa_refused(f) = try
    f()
    false
catch e
    e isa MethodError && e.f === _PA_J.pointer_arg
end

# A closure `@cfunction` is a `Base.CFunction`, and Julia can make one on some
# hosts only: on AArch64 it raises "cfunction: closures are not supported on
# this platform". `nothing` there. The maker is its own function so the answer
# is the same whether Julia raises when it compiles the expression or when it
# runs it.
_pa_inc(x::Cint)::Cint = x + Cint(1)
_pa_make_closure(k::Cint) = @cfunction($(x -> x + k), Cint, (Cint,))
function _pa_closure_cfunction(k::Cint)
    try
        return _pa_make_closure(k)
    catch e
        (e isa ErrorException && occursin("closures are not supported", e.msg)) || rethrow()
        return nothing
    end
end

@testset "Tier 2 pointer arguments" begin

@testset "pointer_arg: the one derivation" begin
    pa = _PA_J.pointer_arg
    Opaque, Code = _PA_J.Opaque, _PA_J.Code

    @testset "a pointer, of any pointee" begin
        p = Ptr{_PAVec}(UInt(0x1000))
        for P in (Ptr{_PAVec}, Ptr{Cvoid}, Ptr{Opaque}, Ptr{Code}, Cstring, Ptr{UInt8}, _PA_J.ByValue{NTuple{24, UInt8}})
            @test pa(P, p).ptr == Ptr{Cvoid}(UInt(0x1000))
            @test pa(P, C_NULL).ptr == C_NULL
            @test pa(P, Ptr{Cint}(UInt(0x2000))).ptr == Ptr{Cvoid}(UInt(0x2000))   # a derived object's pointer, a cast
        end
    end

    @testset "a Ref must point at C data" begin
        r = Ref(_PAVec(1.0, 2.0))
        a = pa(Ptr{_PAVec}, r)
        @test a.keep === r
        @test GC.@preserve r unsafe_load(Ptr{_PAVec}(a.ptr)) == _PAVec(1.0, 2.0)
        @test pa(Ptr{Opaque}, r).ptr == a.ptr                               # an object nothing emits, held as bytes
        ri = Ref{Cint}(7)
        @test GC.@preserve ri unsafe_load(Ptr{Cint}(pa(Ptr{Cint}, ri).ptr)) == 7
        v = Cint[10, 20, 30]
        ra = Ref(v, 2)
        @test GC.@preserve v unsafe_load(Ptr{Cint}(pa(Ptr{Cint}, ra).ptr)) == 20
        # A Julia object behind the Ref is not C data.
        @test _pa_refused(() -> pa(Ptr{Opaque}, Ref("abc")))                # `std::string&` given a Ref{String}
        @test _pa_refused(() -> pa(Ptr{_PAVec}, Ref{Any}(1)))
        @test _pa_refused(() -> pa(Ptr{Cint}, Ref(Cint[1, 2])))
        @test _pa_refused(() -> pa(Ptr{_PAVec}, Ref(_PAOwner(C_NULL))))     # a Ref to an owner is its handle's address
        @test _pa_refused(() -> pa(Ptr{_PAEmpty}, Ref(_PAEmpty())))         # nothing to point at
        @test _pa_refused(() -> pa(Cstring, Ref("abc")))
        # `void*` is untyped: any Ref, as `ccall` takes one.
        s = Ref("abc")
        @test pa(Ptr{Cvoid}, s).ptr == Base.unsafe_convert(Ptr{Cvoid}, s)
    end

    @testset "a record held as its bytes" begin
        b = _PABits((0x01, 0x02, 0x03, 0x04))
        a = pa(Ptr{_PABits}, b)
        @test a.ptr == pointer_from_objref(b) && a.keep === b
        @test GC.@preserve b unsafe_load(Ptr{UInt8}(a.ptr), 3) == 0x03
        @test pa(Ptr{_PABits}, Ref(b)).ptr == pointer_from_objref(b)        # the spelling that always worked
        @test pa(Ptr{Cvoid}, b).ptr == pointer_from_objref(b)
        @test _PA_J._is_byte_record(_PABits)
        @test !_PA_J._is_byte_record(_PAOwner) && !_PA_J._is_byte_record(_PAVec) && !_PA_J._is_byte_record(String)
    end

    @testset "an owning object, by what it converts to" begin
        o = _PAOwner(Ptr{Cvoid}(UInt(0x3000)))
        for P in (Ptr{_PAVec}, Ptr{Cvoid}, Ptr{Opaque}, _PA_J.ByValue{NTuple{24, UInt8}})
            a = pa(P, o)
            @test a.ptr == Ptr{Cvoid}(UInt(0x3000)) && a.keep === o         # kept alive across the call
        end
        @test _pa_refused(() -> pa(Cstring, o))                             # an object is not a C string
        @test _pa_refused(() -> pa(Ptr{Code}, o))
    end

    @testset "a value is not a pointer to it" begin
        for P in (Ptr{_PAVec}, Ptr{Cvoid}, Ptr{Opaque})
            @test _pa_refused(() -> pa(P, _PAVec(1.0, 2.0)))                # a struct value as `this`
            @test _pa_refused(() -> pa(P, 0))
            @test _pa_refused(() -> pa(P, nothing))
            @test _pa_refused(() -> pa(P, sin))
            @test _pa_refused(() -> pa(P, (1.0, 2.0)))
        end
    end

    @testset "an array is data of the pointee's type" begin
        v = Cint[1, 2, 3]
        a = pa(Ptr{Cint}, v)
        @test GC.@preserve v a.ptr == Ptr{Cvoid}(pointer(v))
        @test GC.@preserve v pa(Ptr{Cint}, view(v, 2:3)).ptr == Ptr{Cvoid}(pointer(v, 2))
        @test GC.@preserve v pa(Ptr{Cvoid}, v).ptr == Ptr{Cvoid}(pointer(v))
        recs = [_PAVec(1.0, 2.0), _PAVec(3.0, 4.0)]
        @test GC.@preserve recs pa(Ptr{_PAVec}, recs).ptr == Ptr{Cvoid}(pointer(recs))
        bytes = UInt8[1, 2]
        @test GC.@preserve bytes pa(Ptr{UInt8}, bytes).ptr == Ptr{Cvoid}(pointer(bytes))
        @test pa(Ptr{UInt8}, Int8[1, 2]) isa _PA_J.PointerArg              # bytes of either sign
        @test pa(Cstring, UInt8['a', 0]) isa _PA_J.PointerArg
        @test pa(Ptr{UInt8}, codeunits("ab")) isa _PA_J.PointerArg
        # `T**`: an array of pointers, of any pointee, as a pointer parameter takes them
        handles = Ptr{Cvoid}[C_NULL, C_NULL]
        @test GC.@preserve handles pa(Ptr{Ptr{_PAVec}}, handles).ptr == Ptr{Cvoid}(pointer(handles))
        @test _pa_refused(() -> pa(Ptr{Ptr{_PAVec}}, v))
        @test _pa_refused(() -> pa(Ptr{Cint}, [1.0, 2.0]))
        @test _pa_refused(() -> pa(Ptr{Cint}, Int64[1, 2]))
        @test _pa_refused(() -> pa(Ptr{UInt8}, v))
        @test _pa_refused(() -> pa(Cstring, v))
        @test _pa_refused(() -> pa(Ptr{Opaque}, v))                         # a Julia Vector is not a std::vector
        @test _pa_refused(() -> pa(Ptr{Code}, v))
        @test _pa_refused(() -> pa(_PA_J.ByValue{NTuple{24, UInt8}}, v))
        @test _pa_refused(() -> pa(Ptr{_PAOwner}, [_PAOwner(C_NULL)]))      # an array of Julia objects
    end

    @testset "a char* takes a C string" begin
        s = "hello"
        a = pa(Cstring, s)
        @test a.keep === s && a.ptr == Ptr{Cvoid}(pointer(s))               # no copy of a String
        sub = SubString("abcdef", 1, 3)
        b = pa(Cstring, sub)
        @test GC.@preserve b unsafe_string(Ptr{UInt8}(b.ptr)) == "abc"      # copied and NUL-terminated
        @test b.keep isa String
        @test_throws ArgumentError pa(Cstring, "ab\0cd")
        @test GC.@preserve s unsafe_string(Cstring(pa(Cstring, Base.unsafe_convert(Cstring, s)).ptr)) == "hello"
        @test unsafe_string(Ptr{UInt8}(pa(Cstring, :sym).ptr)) == "sym"     # a Symbol is a C string to Julia
        @test _pa_refused(() -> pa(Ptr{Cvoid}, :sym))
        for x in (Int16(1), UInt64(1), Int8(65), 1.0, true)                 # the overload set's scalars
            @test _pa_refused(() -> pa(Cstring, x))
        end
        # A byte pointer that is not a C string takes the bytes, NULs included.
        @test pa(Ptr{UInt8}, "a\0b") isa _PA_J.PointerArg
        # A string is no other pointer.
        @test _pa_refused(() -> pa(Ptr{Cvoid}, s))
        @test _pa_refused(() -> pa(Ptr{Cint}, s))
        @test _pa_refused(() -> pa(Ptr{Opaque}, s))
        @test _pa_refused(() -> pa(Ptr{Code}, s))
        @test _pa_refused(() -> pa(Ptr{Cvoid}, Base.unsafe_convert(Cstring, s)))
    end

    @testset "a function pointer" begin
        f = x -> x + one(x)
        # What every host has: the pointer a plain `@cfunction` returns.
        plain = @cfunction(_pa_inc, Cint, (Cint,))
        @test pa(Ptr{Code}, plain).ptr == plain
        # A closure's `Base.CFunction`, where the host can make one.
        cf = _pa_closure_cfunction(Cint(3))
        if cf === nothing
            @info "closure @cfunction is not supported on this host: the Base.CFunction case did not run" Sys.ARCH
        else
            @test pa(Ptr{Code}, cf).ptr == Base.unsafe_convert(Ptr{Cvoid}, cf) && pa(Ptr{Code}, cf).keep === cf
        end
        @test _pa_refused(() -> pa(Ptr{Code}, f))                           # a Julia function
        @test _pa_refused(() -> pa(Ptr{Code}, Ref{Cint}(1)))
        @test _pa_refused(() -> pa(Ptr{Code}, _PABits((0x0, 0x0, 0x0, 0x0))))
    end

    @testset "a container taken by value" begin
        V = NTuple{24, UInt8}
        v = ntuple(i -> UInt8(i), 24)
        a = pa(_PA_J.ByValue{V}, v)
        @test a.keep isa Base.RefValue{V}
        @test GC.@preserve a unsafe_load(Ptr{V}(a.ptr)) == v                # the address of the value's bytes
        r = Ref(v)
        @test pa(_PA_J.ByValue{V}, r).ptr == Base.unsafe_convert(Ptr{Cvoid}, r)   # an existing container
        @test _pa_refused(() -> pa(_PA_J.ByValue{V}, ntuple(i -> 0x00, 16)))
        @test _pa_refused(() -> pa(_PA_J.ByValue{V}, 0))
    end

    @testset "a refusal says what a pointer parameter takes" begin
        msg = try
            pa(Ptr{Opaque}, Cint[1])
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("pointer_arg", msg)
        @test occursin("A pointer parameter takes a `Ptr`", msg)
    end
end

@testset "the plan: a converted pointer fills its slot and is kept" begin
    r = Ref{Cint}(41)
    arg = _PA_J.pointer_arg(Ptr{Cint}, r)
    keep, inner = _PA_J.marshal_args(arg, Cint(3))
    GC.@preserve keep begin
        @test unsafe_load(Ptr{Ptr{Cvoid}}(inner[1])) == arg.ptr             # slot -> storage -> the pointer
        @test unsafe_load(Ptr{Cint}(unsafe_load(Ptr{Ptr{Cvoid}}(inner[1])))) == 41
        @test unsafe_load(Ptr{Cint}(inner[2])) == 3
    end
    @test any(k -> k === arg, keep)                                         # and what it keeps, with it
    setup, _, preserve = _PA_J._arg_marshal_plan((typeof(arg), Cint))
    @test occursin(".ptr", string(setup[1]))
    @test :s1 in preserve
end

@testset "_pointer_arg_type: what each position is" begin
    bound(t) = t in ("Rec", "Flags", "Kind") || _PA_W._type_name_bound(t, Set{String}())
    pt(ct, c) = _PA_W._pointer_arg_type(ct, c; resolves = bound)
    jm = "RepliBuild.JITManager"
    vec = "vector<int, std::allocator<int> >"
    # a value is not a pointer position
    @test pt("Cint", "int") === nothing
    @test pt("Rec", "Rec") === nothing
    @test pt("Kind", "Kind") === nothing
    @test pt("Cdouble", "double") === nothing
    # typed pointers and references, `this` among them
    @test pt("Ptr{Rec}", "const Rec*") == "Ptr{Rec}"
    @test pt("Ref{Rec}", "const Rec&") == "Ptr{Rec}"
    @test pt("Ref{Rec}", "Rec&&") == "Ptr{Rec}"
    @test pt("Ptr{Ptr{Rec}}", "Rec**") == "Ptr{Ptr{Rec}}"
    @test pt("Ptr{Cint}", "int*") == "Ptr{Cint}"
    @test pt("Ptr{Kind}", "Kind*") == "Ptr{Kind}"
    # a name the wrapper does not bind is never spelled into a body
    @test pt("Ptr{Unbound}", "Unbound*") == "Ptr{$jm.Opaque}"
    @test pt("Ptr{Ptr{Unbound}}", "Unbound**") == "Ptr{$jm.Opaque}"
    # char pointers: a C string only for a pointer to plain `char`
    @test pt("Ptr{UInt8}", "const char*") == "Cstring"
    @test pt("Ptr{UInt8}", "char*") == "Cstring"
    @test pt("Ptr{UInt8}", "char* const") == "Cstring"
    @test pt("Cstring", "const char*") == "Cstring"
    @test pt("Ptr{UInt8}", "const unsigned char*") == "Ptr{UInt8}"
    @test pt("Ptr{UInt8}", "uint8_t*") == "Ptr{UInt8}"
    @test pt("Ptr{Int8}", "int8_t*") == "Ptr{Int8}"
    @test pt("Ptr{Cchar}", "signed char*") == "Ptr{Cchar}"
    @test pt("Ref{UInt8}", "char&") == "Ptr{UInt8}"                        # one char, not a string
    # untyped: `void*`, a buffer for a scalar Julia cannot hold, a function, an object
    @test pt("Ptr{Cvoid}", "void*") == "Ptr{Cvoid}"
    @test pt("Ptr{Cvoid}", "const void*") == "Ptr{Cvoid}"
    @test pt("Ptr{Cvoid}", "long double&") == "Ptr{Cvoid}"
    @test pt("Ptr{Cvoid}", "function_ptr(int; int)*") == "Ptr{$jm.Code}"
    @test pt("Ptr{Cvoid}", "const $vec&") == "Ptr{$jm.Opaque}"
    @test pt("Ptr{Cvoid}", "basic_string<char, std::char_traits<char>, std::allocator<char> >*") == "Ptr{$jm.Opaque}"
    @test pt("Ptr{Cvoid}", "Unknown*") == "Ptr{$jm.Opaque}"                 # a `this` of a class nothing emits
    @test pt("Ptr{Cvoid}", "unknown") == "Ptr{$jm.Opaque}"
    # a container by value: the thunk takes its address
    @test pt("vector_int_std_allocator_int", vec) == "$jm.ByValue{NTuple{24, UInt8}}"
    @test pt("vector_int_std_allocator_int", "const $vec") == "$jm.ByValue{NTuple{24, UInt8}}"
    @test pt("shared_ptr_Rec", "shared_ptr<Rec>") == "$jm.ByValue{NTuple{16, UInt8}}"

    @testset "a container by value is spelled as a pointer" begin
        @test _PA_W._by_address_ctype("shared_ptr_Rec", "shared_ptr<Rec>") == "Ptr{Cvoid}"
        @test _PA_W._by_address_ctype("vector_int_std_allocator_int", vec) == "Ptr{Cvoid}"
        @test _PA_W._by_address_ctype("Ptr{Cvoid}", "const $vec&") == "Ptr{Cvoid}"
        @test _PA_W._by_address_ctype("Rec", "Rec") == "Rec"
        @test _PA_W._by_address_ctype("pair_int_double", "pair<int, double>") == "pair_int_double"
    end

    @testset "the conversion spelling" begin
        @test _PA_W._tier2_invoke_args(["this", "n", "k", "s", "r"], ["Ptr{Rec}", "Cuint", "Kind", "Ptr{UInt8}", "Rec"],
                                       [false, true, false, false, false], Dict("Kind" => "Int16"),
                                       ["Ptr{Rec}", nothing, nothing, "Cstring", nothing]) ==
              ["$jm.pointer_arg(Ptr{Rec}, this)", "Cuint(n)", "Int16(k)", "$jm.pointer_arg(Cstring, s)", "r"]
    end
end

function _pa_func(name, mangled, params; class="", is_method=false, ret=("int", "Cint"))
    Dict{String,Any}(
        "name" => name, "mangled" => mangled, "demangled" => name,
        "class" => class, "is_method" => is_method, "is_vararg" => false, "is_noexcept" => false,
        "exported" => true, "has_object_pointer" => is_method ? true : nothing,
        "parameters" => Any[Dict{String,Any}("name" => n, "c_type" => c, "julia_type" => j) for (n, c, j) in params],
        "return_type" => Dict{String,Any}("c_type" => ret[1], "julia_type" => ret[2]))
end

@testset "the generated bodies" begin
    dir = mktempdir()
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "pointerargs"
    root = "$(escape_string(dir))"
    [wrap]
    language = "cpp"
    [types]
    strictness = "warn"
    allow_unknown_structs = true
    [cache]
    enabled = false
    """)
    cfg = RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml"))
    libref = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
    vec = "vector<int, std::allocator<int> >"
    structs = Dict{String,Any}(
        "Rec" => Dict{String,Any}("name" => "Rec", "kind" => "struct", "byte_size" => "0x10",
            "members" => Any[Dict{String,Any}("name" => m, "c_type" => "long", "julia_type" => "Clong",
                                              "offset" => string("0x", string(8 * (i - 1), base = 16)), "size" => 8)
                             for (i, m) in enumerate(["a", "b"])]))
    meta = Dict{String,Any}(
        "functions" => Any[
            _pa_func("len", "_ZNK3Rec3lenEv", [("this", "const Rec*", "Ptr{Rec}")]; class = "Rec", is_method = true),
            _pa_func("peek", "_Z4peekPK3Rec", [("r", "const Rec*", "Ptr{Rec}")]),
            _pa_func("byref", "_Z5byrefRK3Rec", [("r", "const Rec&", "Ref{Rec}")]),
            _pa_func("bytes", "_Z5bytesPKhm", [("p", "const unsigned char*", "Ptr{UInt8}"), ("n", "size_t", "Csize_t")]),
            _pa_func("name", "_Z4namePKc", [("s", "const char*", "Ptr{UInt8}")]),
            _pa_func("user", "_Z4userPv", [("p", "void*", "Ptr{Cvoid}")]),
            _pa_func("cb", "_Z2cbPFiiEi", [("f", "function_ptr(int; int)*", "Ptr{Cvoid}"), ("x", "int", "Cint")]),
            _pa_func("vsum", "_Z4vsumRKSt6vectorIiSaIiEE", [("v", "const $vec&", "Ref{$vec}")]),
            _pa_func("vval", "_Z4vvalSt6vectorIiSaIiEE", [("v", vec, vec)]),
            _pa_func("spref", "_Z5sprefRKSt10shared_ptrI3RecE", [("p", "const shared_ptr<Rec>&", "Ref{shared_ptr<Rec>}")]),
            _pa_func("spval", "_Z5spvalSt10shared_ptrI3RecE", [("p", "shared_ptr<Rec>", "shared_ptr<Rec>")]),
            _pa_func("opaque", "_Z6opaqueP7Unknown", [("u", "Unknown*", "Ptr{Cvoid}")]),
            _pa_func("out", "_Z3outPP3Rec", [("o", "Rec**", "Ptr{Ptr{Rec}}")]),
            _pa_func("byval", "_Z5byval3Reci", [("r", "Rec", "Rec"), ("n", "int", "Cint")]),
            # A container by value beside a pointer overload: one Julia name,
            # two methods, which only a prediction in step with the loop keeps.
            _pa_func("take", "_Z4takeSt10shared_ptrI3RecE", [("p", "shared_ptr<Rec>", "shared_ptr<Rec>")]),
            _pa_func("take", "_Z4takePK3Rec", [("r", "const Rec*", "Ptr{Rec}")]),
            # A library deleter: `ManagedRec`.
            _pa_func("rec_free", "_Z8rec_freeP3Rec", [("r", "Rec*", "Ptr{Rec}")]; ret = ("void", "Cvoid")),
        ],
        "globals" => Dict{String,Any}(),
        "struct_definitions" => structs,
        "record_abi" => Dict{String,Any}())
    res = _PA_W.generate_introspective_module_cpp(cfg, libref, meta, "PointerArgs",
                                                  _PA_W.create_type_registry(cfg), true)
    code = res[1]
    sig(name) = [String(m.match) for m in eachmatch(Regex("^function " * name * "\\([^\\n]*\\)", "m"), code)]
    body(name) = (m = match(Regex("^function " * name * "\\([^\\n]*\\n(.*?)^end", "ms"), code);
                  m === nothing ? "no function $name" : String(m.captures[1]))
    jm = "RepliBuild.JITManager"
    pa(t, x) = "$jm.pointer_arg($t, $x)"

    @testset "every pointer position is converted, and says what it is" begin
        @test occursin("thunk\", Cint, " * pa("Ptr{Rec}", "this") * ")", body("Rec_len"))
        @test occursin(pa("Ptr{Rec}", "r"), body("peek"))
        @test occursin(pa("Ptr{Rec}", "r"), body("byref"))
        @test occursin(pa("Ptr{UInt8}", "p") * ", Csize_t(n)", body("bytes"))
        @test occursin(pa("Cstring", "s"), body("name"))
        @test occursin(pa("Ptr{Cvoid}", "p"), body("user"))
        @test occursin(pa("Ptr{$jm.Code}", "f") * ", Cint(x)", body("cb"))
        @test occursin(pa("Ptr{$jm.Opaque}", "v"), body("vsum"))
        @test occursin(pa("$jm.ByValue{NTuple{24, UInt8}}", "v"), body("vval"))
        @test occursin(pa("$jm.ByValue{NTuple{16, UInt8}}", "p"), body("spval"))
        @test occursin(pa("Ptr{$jm.Opaque}", "u"), body("opaque"))
        @test occursin(pa("Ptr{Ptr{Rec}}", "o"), body("out"))
        # a record by value and an integer cross as before
        @test occursin("thunk\", Cint, r, Cint(n))", body("byval"))
        # nothing reaches `invoke` under a pointer parameter's bare name
        calls = [String(m.match) for m in eachmatch(r"JITManager\.invoke\([^\n]*", code)]
        @test length(calls) >= 15
        @test !any(c -> occursin(r"[(,] (?:this|p|s|f|v|u|o)(?:,|\)$)", c), calls)
    end

    @testset "a container by value is spelled as a pointer" begin
        @test sig("vval") == ["function vval(v::Any)"]
        # `shared_ptr_Rec` is a type this wrapper binds (the reference
        # parameter's placeholder); the by-value position does not take it.
        @test occursin(r"^(?:mutable )?struct shared_ptr_Rec\b"m, code)
        @test sig("spval") == ["function spval(p::Any)"]
        @test sig("spref") == ["function spref(p::Ref{shared_ptr_Rec})"]    # a reference keeps its type
    end

    @testset "the prediction is in step with the loop" begin
        s = sig("take")
        @test length(s) == 2
        @test "function take(p::Any)" in s
        @test "function take(r::Union{Ptr{Rec}, Ref{Rec}, AbstractVector{Rec}, ManagedRec})" in s
        strict = _PA_W._integer_width_overloads(meta["functions"], Set{String}(["Rec"]), (jt, ct) -> String(jt);
                                                bound = t -> t == "Rec" || _PA_W._type_name_bound(t, Set{String}()))
        @test get(strict, "_Z4takeSt10shared_ptrI3RecE", nothing) == Set([1])
        @test get(strict, "_Z4takePK3Rec", nothing) == Set([1])
        @test !haskey(strict, "_Z5spvalSt10shared_ptrI3RecE")
    end

    @testset "an owning type converts to an untyped pointer" begin
        # `Managed<X>` defined `unsafe_convert` to `Ptr{X}` alone, which a base
        # class's `this` (`Ptr{Base}`) is not; `pointer_arg` asks an owning
        # object for `Ptr{Cvoid}`, as the factory structs and the STL wrappers
        # already answer.
        @test occursin("unsafe_convert(::Type{Ptr{Rec}}, obj::ManagedRec) = obj.handle", code)
        @test occursin("unsafe_convert(::Type{Ptr{Cvoid}}, obj::ManagedRec) = Ptr{Cvoid}(obj.handle)", code)
        @test count("obj::ManagedRec) =", code) == 2
        @test hasmethod(Base.unsafe_convert, Tuple{Type{Ptr{Cvoid}}, RepliBuild.STLWrappers.CppVector{Cint}})
        @test hasmethod(Base.unsafe_convert, Tuple{Type{Ptr{Cvoid}}, RepliBuild.STLWrappers.CppString})
    end

    # Every method defined once, and the text parses.
    @test allunique(_PA_W._method_sig_keys(code))
    ex = Meta.parseall(code)
    @test !any(a -> a isa Expr && a.head in (:incomplete, :error), ex.args)
end

end
