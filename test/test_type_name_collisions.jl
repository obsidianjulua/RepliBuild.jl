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
    # A method of another signature is a constructor beside the type, as before.
    sc3 = _TN._NameScreen()
    text3 = _TN.with_name_screen(sc3) do
        _TN._screen_decide!(sc3, :type, "conf", nothing)
        "module Toy\nstruct conf\n level::Cint\nend\nconf() = conf(Cint(0))\n" *
        _TN._screened_function_name("conf", "conf") * "(x::Integer) = x\nend\n"
    end
    @test _TN._screen_verdict(sc3, text3) === nothing && isempty(sc3.moves)
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
