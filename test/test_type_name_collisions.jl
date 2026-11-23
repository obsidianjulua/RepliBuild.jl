# =============================================================================
# A function or global named like a type the wrapper defines (2026-11-18)
#
# C keeps struct tags in a namespace of their own, so
#
#     struct conf { int level; int flags; };
#     extern const struct conf *conf;          /* curl: `Curl_ssl` */
#     struct pool *pool(void);
#
# are two names in C each, and one in Julia. Every record the wrapper defines
# gets a zero-initializer `T()`, and a global's accessor or a function taking
# no arguments is `T()` as well: one method defined twice. Included as a file
# the later replaced the earlier, so `conf()` read the global instead of
# zeroing the struct, and a precompiled package refused the overwrite.
# `_assert_no_duplicate_methods` now refuses such a text, which refused curl.
#
# The name screen moves the function (`conf_`), as it moves any name that
# collides with the wrapper's own text, and the type keeps `T()`. The global's
# pointer accessor keeps `conf_ptr`. Library-free: synthetic metadata through
# both generators, loaded against libjulia. test_type_name_collisions_live.jl
# builds a library of this shape and calls it.
# =============================================================================

using Test
using RepliBuild
using Libdl

const _TN = RepliBuild.Wrapper
const _TN_LIBREF = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))

_tn_param(n, jt, ct) = Dict{String,Any}("name" => n, "julia_type" => jt, "c_type" => ct)
_tn_fn(n, params, jt, ct) = Dict{String,Any}("name" => n, "mangled" => n, "demangled" => n,
    "parameters" => Any[params...], "return_type" => Dict{String,Any}("julia_type" => jt, "c_type" => ct))
_tn_member(n, jt, ct, off, size) = Dict{String,Any}("name" => n, "julia_type" => jt, "c_type" => ct,
                                                    "offset" => off, "size" => size)
_tn_rec(kind, size, members...) = Dict{String,Any}("kind" => kind, "byte_size" => size, "members" => Any[members...])
_tn_two_ints(a, b) = _tn_rec("struct", "0x8", _tn_member(a, "Cint", "int", "0x0", 4), _tn_member(b, "Cint", "int", "0x4", 4))

# `conf`: a global named like its struct (curl's shape). `cell`: the same with
# a union. `pool`: a function taking nothing, returning a pointer to its
# struct; `token`: one returning its struct by value. `plainrec`, `counter`
# and `plain`: nothing collides.
function _tn_metadata(; cpp::Bool = false)
    tag(s) = cpp ? s : "struct " * s
    globals = Dict{String,Any}(
        "conf" => Dict{String,Any}("julia_type" => "Ptr{conf}", "c_type" => "const $(tag("conf")) *"),
        "conf_default" => Dict{String,Any}("julia_type" => "conf", "c_type" => tag("conf")),
        "cell" => Dict{String,Any}("julia_type" => "Ptr{cell}", "c_type" => (cpp ? "cell" : "union cell") * " *"),
        "counter" => Dict{String,Any}("julia_type" => "Cint", "c_type" => "int"))
    if cpp
        for (k, g) in globals
            g["name"] = k; g["scope"] = ""; g["symbol"] = k
        end
    end
    functions = Any[
        _tn_fn("conf_level", (_tn_param("c", "Ptr{conf}", "const $(tag("conf")) *"),), "Cint", "int"),
        _tn_fn("pool", (), "Ptr{pool}", tag("pool") * " *"),
        _tn_fn("token", (), "token", tag("token")),
        _tn_fn("plain", (_tn_param("x", "Cint", "int"),), "Cint", "int")]
    return Dict{String,Any}(
        "functions" => functions,
        "globals" => globals,
        "struct_definitions" => Dict{String,Any}(
            "conf" => _tn_two_ints("level", "flags"),
            "pool" => _tn_two_ints("size", "used"),
            "token" => _tn_two_ints("kind", "len"),
            "plainrec" => _tn_two_ints("a", "b"),
            "cell" => _tn_rec("union", "0x8", _tn_member("i", "Cint", "int", "0x0", 4),
                              _tn_member("d", "Cdouble", "double", "0x0", 8))))
end

function _tn_config(dir::String, lang::String; rename::String = "")
    toml = joinpath(dir, "replibuild.toml")
    write(toml, """
    [project]
    name = "typenamesynth"
    root = "$(escape_string(dir))"
    [link]
    enable_lto = false
    [wrap]
    language = "$lang"
    $(isempty(rename) ? "" : "[wrap.rename]\n" * rename)
    [types]
    strictness = "warn"
    allow_unknown_structs = true
    [cache]
    enabled = false
    """)
    return RepliBuild.ConfigurationManager.load_config(toml)
end

# The wrap's own loop, as `wrap_library` runs it: generations under the screen,
# each inside `with_renames`, then the rename check.
function _tn_wrap(lang::Symbol, meta; rename::String = "")
    dir = mktempdir()
    cfg = _tn_config(dir, lang === :cpp ? "cpp" : "c"; rename = rename)
    gen = lang === :cpp ? _TN.generate_introspective_module_cpp : _TN.generate_introspective_module_c
    sanitize = lang === :cpp ? _TN._sanitize_cpp_type_name : _TN._sanitize_c_type_name
    runs = Ref(0)
    ((code, rt), sc, _) = redirect_stdout(devnull) do
        _TN._screened_generations(meta, sanitize; what = "TypeNameSynth") do md
            runs[] += 1
            ((out, _), rt) = _TN.with_renames(cfg.wrap.rename) do
                r = gen(cfg, _TN_LIBREF, md, "TypeNameSynth", _TN.create_type_registry(cfg), true)
                (r isa Tuple ? String(first(r)) : String(r), nothing)
            end
            (out, rt)
        end
    end
    _TN._check_renames(rt, code, "TypeNameSynth")
    rm(dir; recursive = true, force = true)
    return (code, sc, runs[])
end

function _tn_parses(code)
    ex = Meta.parseall(code)
    bad = Ref(false)
    walk(x) = x isa Expr && ((x.head in (:incomplete, :error)) ? (bad[] = true) : foreach(walk, x.args))
    walk(ex)
    return !bad[]
end

function _tn_load(code)
    host = Module(:TypeNameHost)
    Base.include_string(host, code)
    return Base.invokelatest(getfield, host, :TypeNameSynth)
end

_tn_has(M, s) = Base.invokelatest(isdefined, M, s)
_tn_call(f, args...) = Base.invokelatest(f, args...)
_tn_zero(M, T, fields) = (v = _tn_call(getfield(M, T)); v isa getfield(M, T) && all(f -> getfield(v, f) == 0, fields))

@testset "the screen moves a function a type's constructor already is" begin
    # A toy text: `conf` is a struct whose zero-initializer is `conf()`, and a
    # function the library spells `conf` takes nothing.
    sc = _TN._NameScreen()
    text = _TN.with_name_screen(sc) do
        _TN._screen_decide!(sc, :type, "conf", nothing)          # the record, as `_screen_records` keeps it
        f = _TN._screened_function_name("conf", "conf")
        g = _TN._screened_function_name("other", "other")
        """
        module Toy
        struct conf
            level::Cint
        end
        conf() = conf(Cint(0))
        function $f()::Ptr{conf}
            return C_NULL
        end
        $g(x::Integer) = x
        conf(x::Float64) = conf(Cint(x))       # a constructor the library did not give
        end
        """
    end
    @test sc.kept[:function] == Set(["conf", "other"])
    next = _TN._screen_verdict(sc, text)
    @test next !== nothing && next.move == Dict((:function, "conf") => :type_constructor)
    # The generation after it moves the function and stands.
    text2 = _TN.with_name_screen(next) do
        _TN._screen_decide!(next, :type, "conf", nothing)
        "module Toy\nstruct conf\n level::Cint\nend\nconf() = conf(Cint(0))\n" *
        "function " * _TN._screened_function_name("conf", "conf") * "()::Ptr{conf}\n C_NULL\nend\nend\n"
    end
    @test _TN._screen_verdict(next, text2) === nothing
    @test occursin("conf_()", text2)
    @test _TN._screen_report(next) ==
          ["function `conf` is bound as `conf_`: the type `conf` defines a method of the same signature itself " *
           "(a constructor such as its zero-initializer `conf()`), and keeps it"]
    # A method of another arity is a constructor beside the type, as before.
    # (One of as many arguments as the type has fields is not: the next
    # testset.)
    sc3 = _TN._NameScreen()
    text3 = _TN.with_name_screen(sc3) do
        _TN._screen_decide!(sc3, :type, "conf", nothing)
        "module Toy\nstruct conf\n level::Cint\nend\nconf() = conf(Cint(0))\n" *
        _TN._screened_function_name("conf", "conf") * "(x::Integer, y::Integer) = x + y\nend\n"
    end
    @test _TN._screen_verdict(sc3, text3) === nothing && isempty(sc3.moves)
end

@testset "the screen moves a function that takes its type's field-constructor calls" begin
    # `struct pair { int a; int b; };` beside `int pair(int a, int b);`: Julia
    # gives the struct `pair(::Cint, ::Cint)` and `pair(a, b)`, and the
    # function's `pair(::Integer, ::Integer)` is more specific than the second,
    # so `pair(1, 2)` called the function and `pair(Cint(1), Cint(2))` built
    # the struct. No text defines a method twice.
    toy(f) = """
        module Toy
        struct pair
            a::Cint
            b::Cint
        end
        pair() = pair(Cint(0), Cint(0))
        function $f(a::Integer, b::Integer)::Cint
            return Cint(a * 10 + b)
        end
        end
        """
    sc = _TN._NameScreen()
    text = _TN.with_name_screen(sc) do
        _TN._screen_decide!(sc, :type, "pair", nothing)
        toy(_TN._screened_function_name("pair", "pair"))
    end
    @test isempty(_TN._method_definitions(text) |> x -> filter(s -> count(==(s), x) > 1, x))
    next = _TN._screen_verdict(sc, text)
    @test next !== nothing && next.move == Dict((:function, "pair") => :field_constructor)
    text2 = _TN.with_name_screen(next) do
        _TN._screen_decide!(next, :type, "pair", nothing)
        toy(_TN._screened_function_name("pair", "pair"))
    end
    @test _TN._screen_verdict(next, text2) === nothing
    @test occursin("function pair_(a::Integer, b::Integer)", text2)
    @test _TN._screen_report(next) ==
          ["function `pair` is bound as `pair_`: `pair(…)` with one argument per field builds the type `pair`, " *
           "a function of that name with as many parameters would take those calls, and the type keeps them"]
    # Loaded: `pair(1, 2)` builds the struct, whatever the integers.
    M = Base.invokelatest(getfield, (m = Module(:PairHost); Base.include_string(m, text2); m), :Toy)
    @test _tn_call(M.pair, 1, 2) isa M.pair && _tn_call(M.pair, Cint(1), Cint(2)) isa M.pair
    @test _tn_call(M.pair_, 1, 2) == 12

    # A varargs method that can take as many arguments clashes too.
    sc4 = _TN._NameScreen()
    text4 = _TN.with_name_screen(sc4) do
        _TN._screen_decide!(sc4, :type, "pair", nothing)
        "module Toy\nstruct pair\n a::Cint\n b::Cint\nend\n" *
        _TN._screened_function_name("pair", "pair") * "(a::Integer, rest...) = a\nend\n"
    end
    @test _TN._screen_verdict(sc4, text4).move == Dict((:function, "pair") => :field_constructor)
end

@testset "which structs Julia gives field constructors" begin
    text = """
    module Toy
    struct pair
        a::Cint
        b::Cint
    end
    struct padded
        c::UInt8
        _pad_1::NTuple{3, UInt8}
        v::Cint
    end
    struct flags
        _data::NTuple{4, UInt8}
    end
    mutable struct num
        data::NTuple{4, UInt8}
    end
    struct packed_s
        _data::NTuple{5, UInt8}
    end
    mutable struct Owner
        handle::Ptr{Cvoid}
        Owner(::Base.Val{:_rb_adopt}, h::Ptr{Cvoid}) = new(h)
    end
    struct empty_rec end
    struct one
        x
    end
    end
    """
    # A byte carrier (a bitfield record, a union, a byte blob), a type with an
    # inner constructor and a type with no field have none a function can
    # take the calls of.
    @test _TN._field_constructor_arities(text) == Dict("pair" => 2, "padded" => 3, "one" => 1)
    @test _TN._signature_arity("pair(Union{Integer,Base.Enum},Union{Integer,Base.Enum})") == (2, false)
    @test _TN._signature_arity("f()") == (0, false)
    @test _TN._signature_arity("f(Ptr{NTuple{2,Cint}},Vararg{Any})") == (1, true)
    @test _TN._signature_arity("g(Any) where {T}") == (1, false)
end

@testset "GeneratorC: $(what)" for (what, rename) in (("screened", ""), ("[wrap.rename] chooses", "pool = \"make_pool\"\n"))
    (code, sc, runs) = _tn_wrap(:c, _tn_metadata(); rename = rename)
    @test runs == 2                                   # the probe, and the generation that moves
    @test _tn_parses(code)
    @test _TN._assert_wrapper_loadable(code, "TypeNameSynth") === nothing
    moves = Dict((m.kind, m.from) => (m.to, m.why) for m in sc.moves)
    @test moves[(:function, "conf")] == ("conf_", :type_constructor)
    @test moves[(:function, "cell")] == ("cell_", :type_constructor)
    @test moves[(:function, "token")] == ("token_", :type_constructor)
    if isempty(rename)
        @test moves[(:function, "pool")] == ("pool_", :type_constructor)
    else
        @test !haskey(moves, (:function, "pool"))     # the chosen name wins, and is not screened
    end
    @test length(sc.moves) == (isempty(rename) ? 4 : 3)
    report = _TN._screen_report(sc)
    @test "function `conf` is bound as `conf_`: the type `conf` defines a method of the same signature itself " *
          "(a constructor such as its zero-initializer `conf()`), and keeps it" in report

    M = _tn_load(code)
    # `T()` is the zeroed record, for every record.
    @test _tn_zero(M, :conf, (:level, :flags)) && _tn_zero(M, :pool, (:size, :used))
    @test _tn_zero(M, :token, (:kind, :len)) && _tn_zero(M, :plainrec, (:a, :b))
    @test _tn_call(M.cell) isa M.cell && all(iszero, _tn_call(M.cell).data)
    # The global's value and the functions are there under the reported names;
    # the global's pointer accessor keeps its name.
    for n in (:conf_, :conf_ptr, :conf_default, :conf_default_ptr, :cell_, :cell_ptr, :token_,
              :conf_level, :counter, :counter_ptr, :plain, (isempty(rename) ? :pool_ : :make_pool))
        @test (n, _tn_has(M, n)) == (n, true)
    end
    for n in (:conf__ptr, :cell__ptr, :counter_, :plain_, (isempty(rename) ? :make_pool : :pool_))
        @test (n, _tn_has(M, n)) == (n, false)
    end
    # Each moved function has exactly the method it had, and `T` exactly its own.
    @test length(methods(M.conf_)) == 1 && hasmethod(M.conf_, Tuple{})
    @test hasmethod(M.token_, Tuple{}) && length(methods(M.cell_)) == 1
    @test occursin("function conf_()::Ptr{conf}\n    ptr = cglobal((:conf, LIBRARY_PATH), Ptr{conf})", code)
    @test occursin("function conf_ptr()::Ptr{Ptr{conf}}\n    return cglobal((:conf, LIBRARY_PATH), Ptr{conf})", code)
    @test occursin("ccall((:token, LIBRARY_PATH), token, ()", code)
    @test occursin(isempty(rename) ? "ccall((:pool, LIBRARY_PATH), Ptr{pool}, ()" :
                                     "function make_pool()", code)
end

@testset "GeneratorC: no collision, no second generation" begin
    meta = _tn_metadata()
    delete!(meta["globals"], "conf"); delete!(meta["globals"], "cell")
    filter!(f -> !(f["name"] in ("pool", "token")), meta["functions"])
    (code, sc, runs) = _tn_wrap(:c, meta)
    @test runs == 1 && isempty(sc.moves)
    @test _TN._assert_wrapper_loadable(code, "TypeNameSynth") === nothing
    M = _tn_load(code)
    @test _tn_zero(M, :conf, (:level, :flags)) && _tn_has(M, :conf_default_ptr) && !_tn_has(M, :conf_)
end

@testset "GeneratorCpp: a global and functions named like a class" begin
    # C++ lets a variable or a function hide a class name too (`struct stat`).
    (code, sc, runs) = _tn_wrap(:cpp, _tn_metadata(cpp = true))
    @test runs == 2
    @test _tn_parses(code)
    @test _TN._assert_wrapper_loadable(code, "TypeNameSynth") === nothing
    moves = Dict((m.kind, m.from) => (m.to, m.why) for m in sc.moves)
    @test moves[(:function, "conf")] == ("conf_", :type_constructor)
    # A pointer to a union (a `mutable struct`) is a value GeneratorCpp does
    # not hold, so `cell` gets its pointer accessor alone and nothing collides.
    @test moves[(:function, "pool")] == ("pool_", :type_constructor)
    @test moves[(:function, "token")] == ("token_", :type_constructor)
    @test length(moves) == 3
    @test length(_TN._screen_report(sc)) == 3
    M = _tn_load(code)
    @test _tn_zero(M, :conf, (:level, :flags)) && _tn_zero(M, :pool, (:size, :used))
    @test _tn_zero(M, :token, (:kind, :len))
    @test hasmethod(M.pool_, Tuple{}) && hasmethod(M.token_, Tuple{}) && _tn_has(M, :plain)
    @test _tn_call(M.cell) isa M.cell
    for n in (:conf_, :conf_ptr, :cell_ptr, :conf_default, :conf_default_ptr, :counter, :counter_ptr)
        @test (n, _tn_has(M, n)) == (n, true)
    end
    @test !_tn_has(M, :conf__ptr) && !_tn_has(M, :counter_) && !_tn_has(M, :cell_)
    @test length(methods(M.conf_)) == 1 && hasmethod(M.conf_, Tuple{})
end

@testset "a name Julia also has: put back by the verdict, then moved for the clash" begin
    # `length` is a function of Base, so the probe moves the record and the
    # global both; the text does not use `length`, so the next generation puts
    # both back, and there the two `length()` collide. A third moves the global.
    meta = Dict{String,Any}(
        "functions" => Any[_tn_fn("plain", (_tn_param("x", "Cint", "int"),), "Cint", "int")],
        "globals" => Dict{String,Any}("length" => Dict{String,Any}("julia_type" => "Ptr{length}",
                                                                     "c_type" => "struct length *")),
        "struct_definitions" => Dict{String,Any}("length" => _tn_two_ints("lo", "hi")))
    (code, sc, runs) = _tn_wrap(:c, meta)
    @test runs == 3
    @test _TN._assert_wrapper_loadable(code, "TypeNameSynth") === nothing
    @test [(m.kind, m.from, m.to, m.why) for m in sc.moves] == [(:function, "length", "length_", :type_constructor)]
    M = _tn_load(code)
    @test _tn_zero(M, :length, (:lo, :hi)) && M.length !== Base.length
    @test hasmethod(M.length_, Tuple{}) && _tn_has(M, :length_ptr) && !_tn_has(M, :length__ptr)
end

# `pair` is a struct of two `int`s beside `int pair(int, int)`: the function
# takes the calls of the struct's field constructor and moves. `flags` (a
# bitfield record), `num` (a union) and `packed_s` (a byte blob, since `v`
# sits at offset 1) are built from their bytes, so the functions beside them
# keep their names; `triple(int)` beside a three-field struct is of another
# arity and keeps its name too.
function _tn_shadow_metadata(; cpp::Bool = false)
    tag(s) = cpp ? s : "struct " * s
    bit(n, off, w) = Dict{String,Any}("name" => n, "julia_type" => "Cuint", "c_type" => "unsigned int",
                                      "offset" => nothing, "size" => 4, "bit_size" => w, "data_bit_offset" => off)
    ints(names...) = [_tn_param(n, "Cint", "int") for n in names]
    return Dict{String,Any}(
        "functions" => Any[
            _tn_fn("pair", ints("a", "b"), "Cint", "int"),
            _tn_fn("flags", ints("lo", "hi"), "Cint", "int"),
            _tn_fn("num", ints("i"), "Cint", "int"),
            _tn_fn("packed_s", ints("c", "v"), "Cint", "int"),
            _tn_fn("triple", ints("x"), "Cint", "int"),
            _tn_fn("make_pair", ints("a", "b"), "pair", tag("pair")),
            _tn_fn("make_flags", (), "flags", tag("flags")),
            _tn_fn("make_packed", (), "packed_s", tag("packed_s")),
            _tn_fn("triple_sum", (_tn_param("t", "Ptr{triple}", tag("triple") * " *"),), "Cint", "int")],
        "globals" => Dict{String,Any}(),
        "struct_definitions" => Dict{String,Any}(
            "pair" => _tn_two_ints("a", "b"),
            "triple" => _tn_rec("struct", "0xc", _tn_member("x", "Cint", "int", "0x0", 4),
                                _tn_member("y", "Cint", "int", "0x4", 4), _tn_member("z", "Cint", "int", "0x8", 4)),
            "flags" => _tn_rec("struct", "0x4", bit("lo", 0, 3), bit("hi", 3, 5)),
            "num" => _tn_rec("union", "0x4", _tn_member("i", "Cint", "int", "0x0", 4),
                             _tn_member("f", "Cfloat", "float", "0x0", 4)),
            "packed_s" => _tn_rec("struct", "0x5", _tn_member("c", "UInt8", "char", "0x0", 1),
                                  _tn_member("v", "Cint", "int", "0x1", 4))))
end

@testset "$gen: a function of a struct's field-constructor arity moves" for (gen, lang) in
        (("GeneratorC", :c), ("GeneratorCpp", :cpp))
    (code, sc, runs) = _tn_wrap(lang, _tn_shadow_metadata(cpp = lang === :cpp))
    @test runs == 2
    @test _tn_parses(code)
    @test _TN._assert_wrapper_loadable(code, "TypeNameSynth") === nothing
    # (GeneratorCpp names a function twice, in its overload prediction and in
    # its loop; the report says it once.)
    @test unique([(m.kind, m.from, m.to, m.why) for m in sc.moves]) == [(:function, "pair", "pair_", :field_constructor)]
    @test length(_TN._screen_report(sc)) == 1
    M = _tn_load(code)
    # `pair(1, 2)` builds the struct, whatever the integers, and the function
    # is `pair_`.
    p = _tn_call(M.pair, 1, 2)
    @test p isa M.pair && (p.a, p.b) == (1, 2)
    @test _tn_call(M.pair, Cint(1), Cint(2)) isa M.pair
    @test hasmethod(M.pair_, Tuple{Int,Int}) && !hasmethod(M.pair, Tuple{Int8,Int8,Int8})
    # Nothing else moved: the byte carriers and the other arity keep the
    # library's names, as functions of the module.
    for n in (:flags, :num, :triple, :make_pair, :make_flags)
        @test (n, _tn_has(M, n)) == (n, true)
    end
    @test hasmethod(M.flags, Tuple{Int,Int}) && hasmethod(M.num, Tuple{Int}) && hasmethod(M.triple, Tuple{Int})
    for n in (:flags_, :num_, :triple_, :packed_s_)
        @test (n, _tn_has(M, n)) == (n, false)
    end
    lang === :c && @test hasmethod(M.packed_s, Tuple{Int,Int})
end
