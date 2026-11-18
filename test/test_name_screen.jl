# =============================================================================
# The name screen (library-free)
#
# A generated module's namespace belongs to the library, and the wrapper's own
# text is written against Base and Core. A library identifier that is also a
# name of that text changed what the text meant (audit 2026-11-17, B2/B3/B5):
#
#   int String(int)        a METHOD OF Base.String, in every module: after
#                          `using`, `String(1) == 101`
#   int Cint(int)          `Cint(::SomeJuliaEnum)` a MethodError process-wide
#   int nothing(int)       `return nothing` in the wrapper returns a function
#   int eval(int)          "cannot define function eval; it already has a value"
#   int Libdl(int)         the wrapper's own `Libdl.dlopen` hits the C function
#   int ccall(int)         "invalid function name"
#   int LIBRARY_PATH(int)  the wrap refused: one name cost the whole library
#   struct Bool/Cvoid      segfault at load; struct Integer: every `x::Integer`
#                          signature now names the struct
#   module String          `::Union{String,Nothing}` names the module
#
# src/Wrapper/NameScreen.jl is the one screen for every top-level name a
# generator binds. This file drives it without a toolchain: what Julia says
# about a name, what a text binds and references, the verdict over a toy
# generator, the record rename in the metadata, and the real C generator
# through the screen with the module loaded against libjulia. The C++ side and
# real libraries are devtests' test_name_screen_live.jl.
# =============================================================================

using Test
using RepliBuild
using Libdl

const _NS = RepliBuild.Wrapper

@enum NsColor ns_red = 3 ns_green = 4          # a Julia enum of the test's own

# A wrapper-shaped module around the functions a generator would bind. `own`
# is text of the wrapper's own, the part that does not come from the library.
function _ns_toy(functions; own::AbstractString = "", modname::AbstractString = "Toy", runs = Ref(0))
    return function (metadata)
        runs[] += 1
        io = IOBuffer()
        println(io, "module $modname\nusing Libdl\nimport RepliBuild\nimport Base: unsafe_convert")
        println(io, "const LIBRARY_PATH = \"lib\"\n", own)
        for f in functions
            n = _NS._julia_function_name(f, f)
            println(io, "function $n(x::Integer)\n    return ccall((:$f, LIBRARY_PATH), Cint, (Cint,), x)\nend")
        end
        println(io, "end")
        return (String(take!(io)), nothing)
    end
end

_ns_bound(text) = _NS._name_uses(text).bound
_ns_fn_names(text) = sort!([k for (k, v) in _ns_bound(text) if :function in v])

@testset "The name screen" begin

@testset "what Julia says about a name" begin
    refusal = _NS._function_name_refusal
    # A type of Base or Core: a method under that name is a method of the type.
    for n in ("String", "Ptr", "Ref", "Cint", "Integer", "Bool", "Type", "Vector", "Any", "Union", "Task")
        @test (n, refusal(n)) == (n, :base_type)
        @test something(_NS._fresh_binding(n)) isa Type
    end
    # Bindings every module owns.
    @test refusal("eval") === :every_module && refusal("include") === :every_module
    # Not a function name at all: the parser's reserved words, and lowering's.
    for n in ("ccall", "end", "function", "true", "begin")
        @test (n, refusal(n)) == (n, :syntax)
    end
    # Everything else may be bound; what the wrapper's text says decides.
    for n in ("error", "length", "print", "show", "open", "stat", "nothing", "Base", "Core", "Main",
              "isa", "typeof", "in", "where", "plain", "Libdl", "RepliBuild", "LIBRARY_PATH", "unsafe_convert")
        @test (n, refusal(n)) == (n, nothing)
    end
    # ... and the screen knows which of them a fresh module resolves already.
    for n in ("error", "length", "open", "nothing", "Base", "Core", "Main", "isa", "String", "eval")
        @test (n, _NS._fresh_binding(n) !== nothing) == (n, true)
    end
    for n in ("plain", "Libdl", "RepliBuild", "LIBRARY_PATH", "unsafe_convert", "ccall", "not a name", "")
        @test (n, _NS._fresh_binding(n) !== nothing) == (n, false)
    end

    # The refusals are Julia's own answer, not a list: defining the function in
    # a module of its own fails exactly for `:syntax` and `:every_module`. (A
    # `:base_type` name is not defined here: that is the piracy itself.)
    checked = 0
    for n in ("eval", "include", "ccall", "cglobal", "end", "function", "let", "error", "length",
              "nothing", "Base", "isa", "in", "where", "plain", "ans", "var")
        defines = try
            Core.eval(Module(:NameScreenHost), Meta.parse("module M\nfunction $n(x) 1 end\nend"))
            true
        catch
            false
        end
        @test (n, defines) == (n, !(refusal(n) in (:syntax, :every_module)))
        checked += 1
    end
    @test checked == 17
    # The reserved-word list `_julia_function_name` keeps is inside the answer.
    @test all(w -> refusal(w) === :syntax, _NS._JULIA_RESERVED_WORDS)
end

@testset "function names: refused ones move, with or without a wrap" begin
    name = _NS._julia_function_name
    @test _NS._SCREEN[] === nothing
    @test name("String") == "String_" && name("Ptr") == "Ptr_" && name("Ref") == "Ref_"
    @test name("Cint") == "Cint_" && name("eval") == "eval_" && name("include") == "include_"
    @test name("ccall") == "ccall_" && name("cglobal") == "cglobal_"
    @test name("end") == "end_" && name("function") == "function_"          # as before
    # A method keeps its class prefix: only a bare name can be Base's.
    @test name("String_get") == "String_get" && name("Integer::value") == "Integer_value"
    # Names that work keep their spelling. Without a screen (a generator called
    # on its own) nothing but a refusal moves a name.
    for n in ("error", "length", "print", "show", "open", "stat", "all", "nothing", "Base", "Main",
              "plain", "setproperty", "with", "libc_free", "struct_size", "LIBRARY_PATH", "unsafe_convert")
        @test (n, name(n)) == (n, n)
    end
    @test name("__init__") == "_init"                                        # the old derivation
    # The alternative is itself free.
    @test _NS._function_name_refusal("String_") === nothing && _NS._fresh_binding("String_") === nothing
end

@testset "what a text binds and references" begin
    uses = _NS._name_uses("""
    module Demo
    using Libdl; import RepliBuild; import Base: unsafe_convert
    const LIBRARY_PATH = let baked = "x"
        sibling = joinpath(@__DIR__, basename(baked))
        isfile(sibling) ? sibling : baked
    end
    struct Rec{T} <: Number
        v::Cint
        pad::NTuple{4, UInt8}
        Rec{T}(v) where {T} = new{T}(v, zeros4())
    end
    primitive type Kind <: RepliBuild.OpenEnums.OpenEnum{Cuint} 32 end
    const Plain = Kind(6)
    \"\"\"
        f(x)

    A docstring that mentions String(1) and error.
    \"\"\"
    function f(x::Integer, length::Cint; init = nothing)::Union{String,Nothing}
        y = Base.string(x)
        for i in 1:length
            y = y * string(i)
        end
        z = [first(p) for p in pairs((a = 1, b = 2)) if isodd(last(p))]
        g = v -> convert(Cint, v)
        try
            open(y)
        catch err
            @warn "no" exception=err
        end
        GC.@preserve y ccall((:sym, LIBRARY_PATH), Cint, (Ptr{Cvoid},), C_NULL)
        return init
    end
    h(r) = r isa Rec ? r.v : zero(Cint)
    Base.show(io::IO, r::Rec) = print(io, "Rec ", r.pad)
    unsafe_convert(::Type{Ptr{Cvoid}}, r::Rec) = pointer_from_objref(r)
    function __init__()
        Libdl.dlopen(LIBRARY_PATH)
        global counter = 0
    end
    end
    """)
    @test uses !== nothing
    kinds(n) = get(uses.bound, n, Symbol[])
    @test kinds("Demo") == [:module]
    @test kinds("Libdl") == [:import] && kinds("RepliBuild") == [:import]
    @test kinds("unsafe_convert") == [:import, :function]
    @test kinds("LIBRARY_PATH") == [:const] && kinds("Plain") == [:const]
    @test kinds("Rec") == [:struct] && kinds("Kind") == [:type]
    @test kinds("f") == [:function] && kinds("h") == [:function] && kinds("__init__") == [:function]
    # A qualified method is not a binding of this module.
    @test !haskey(uses.bound, "show")
    # References, by Julia's scope rules.
    for n in ("joinpath", "basename", "isfile", "Number", "Cint", "NTuple", "UInt8", "zeros4", "Cuint",
              "Integer", "Union", "String", "Nothing", "nothing", "Base", "string", "first", "pairs",
              "isodd", "last", "convert", "open", "GC", "ccall", "Ptr", "Cvoid", "C_NULL", "isa",
              "zero", "IO", "print", "Type", "pointer_from_objref", "Libdl", "RepliBuild",
              "LIBRARY_PATH", "Rec", "Kind")
        @test (n, n in uses.refs) == (n, true)
    end
    # Not references: a `let` and a function's locals, parameters (one of them
    # named `length`), type parameters, loop and comprehension variables, a
    # `catch` variable, a keyword and a named-tuple field, a field name, a
    # quoted symbol, a qualified name's right side, the docstring's words.
    for n in ("baked", "sibling", "T", "v", "pad", "new", "x", "length", "init", "y", "i", "z", "p",
              "a", "b", "g", "err", "exception", "sym", "r", "io", "show", "dlopen", "error", "counter")
        @test (n, n in uses.refs) == (n, false)
    end

    @test _NS._name_uses("module M\nfunction f(\nend") === nothing        # unparseable: not this guard's
end

@testset "the verdict: only what the wrapper's own text uses moves" begin
    gen(fs; kw...) = _NS._screened_generations(_ns_toy(fs; kw...), Dict{String,Any}(), identity)
    moved(sc) = sort!(unique!(["$(m.from)=>$(m.to):$(m.why)" for m in sc.moves]))

    # No library name Julia knows: one generation, nothing moved.
    runs = Ref(0)
    ((text, _), sc, _) = gen(["plain", "frobnicate"]; runs = runs)
    @test runs[] == 1 && isempty(sc.moves)
    @test _ns_fn_names(text) == ["frobnicate", "plain"]

    # Names Julia knows that the text does not use: they keep their spelling,
    # which takes the second generation.
    runs = Ref(0)
    ((text, _), sc, _) = redirect_stdout(devnull) do
        gen(["error", "length", "print", "show", "stat", "Main", "plain"]; runs = runs)
    end
    @test runs[] == 2 && isempty(sc.moves)
    @test _ns_fn_names(text) == ["Main", "error", "length", "plain", "print", "show", "stat"]

    # Names the wrapper's own text uses: the library's binding moves, in one
    # generation when every candidate had to.
    own = "helper(p) = (isfile(p) ? open(p) : nothing; Base.string(p); Core.typeof(p))"
    runs = Ref(0)
    ((text, _), sc, _) = gen(["open", "isfile", "nothing", "Base", "Core"]; own = own, runs = runs)
    @test runs[] == 1
    @test _ns_fn_names(text) == ["Base_", "Core_", "helper", "isfile_", "nothing_", "open_"]
    @test moved(sc) == ["Base=>Base_:own_use", "Core=>Core_:own_use", "isfile=>isfile_:own_use",
                        "nothing=>nothing_:own_use", "open=>open_:own_use"]

    # A mix: `error` and `string` (qualified in the wrapper's text) stay,
    # `open` moves; a name the wrapper itself binds or imports moves; so does
    # a function named like the module. `setproperty` is the wrapper's own
    # function and takes a method, as it always did.
    own *= "\nfunction setproperty end\nsetproperty(x::String, s::Symbol, v) = x"
    runs = Ref(0)
    ((text, _), sc, _) = redirect_stdout(devnull) do
        gen(["error", "string", "open", "LIBRARY_PATH", "Libdl", "RepliBuild", "unsafe_convert",
             "Toy", "setproperty", "String", "eval", "plain"]; own = own, runs = runs)
    end
    @test runs[] == 2
    @test _ns_fn_names(text) == ["LIBRARY_PATH_", "Libdl_", "RepliBuild_", "String_", "Toy_", "error",
                                 "eval_", "helper", "open_", "plain", "setproperty", "string",
                                 "unsafe_convert_"]
    @test moved(sc) == ["LIBRARY_PATH=>LIBRARY_PATH_:own_binding", "Libdl=>Libdl_:own_binding",
                        "RepliBuild=>RepliBuild_:own_binding", "String=>String_:base_type",
                        "Toy=>Toy_:own_binding", "eval=>eval_:every_module", "open=>open_:own_use",
                        "unsafe_convert=>unsafe_convert_:own_binding"]
    # The text that stood binds no name twice and leaves the wrapper's names its own.
    _NS._assert_no_duplicate_toplevel_names(text, "Toy")
    b = _ns_bound(text)
    @test b["LIBRARY_PATH"] == [:const] && b["Libdl"] == [:import] && b["Toy"] == [:module]
    # The report names each, sorted, with the reason.
    report = _NS._screen_report(sc)
    @test length(report) == 8 && issorted(report)
    @test "function `String` is bound as `String_`: `String` is a type of Base or Core, and a function " *
          "of that name would be a method of it in every module" in report
    @test "function `open` is bound as `open_`: the wrapper's own text uses `open`" in report
    @test "function `LIBRARY_PATH` is bound as `LIBRARY_PATH_`: the wrapper itself binds `LIBRARY_PATH`" in report

    # The screen is installed for the generation only.
    @test _NS._SCREEN[] === nothing
    @test_throws ErrorException _NS.with_name_screen(() -> _NS.with_name_screen(() -> 1, _NS._NameScreen()),
                                                     _NS._NameScreen())
    @test _NS._SCREEN[] === nothing

    # What a generation logs is said once, by the one that stood.
    logged = Ref(0)
    noisy = function (md)
        @warn "from the generator"
        _ns_toy(["error"])(md)
    end
    @test_logs (:warn, "from the generator") redirect_stdout(devnull) do
        _NS._screened_generations(noisy, Dict{String,Any}(), identity)
    end
    # ... and one that threw still says what it said.
    @test_logs (:warn, "before the throw") begin
        @test_throws ErrorException _NS._screened_generations(md -> (@warn("before the throw"); error("boom")),
                                                              Dict{String,Any}(), identity)
    end
end

@testset "[wrap.rename] wins over the screen" begin
    fs = ["String", "open", "error", "plain"]
    own = "helper(p) = open(p)"
    function gen(table)
        local rt
        ((text, _), sc, _) = redirect_stdout(devnull) do
            _NS._screened_generations(Dict{String,Any}(), identity) do md
                ((t, r), rt) = _NS.with_renames(() -> _ns_toy(fs; own = own)(md), table)
                (t, r)
            end
        end
        return (text, sc, rt)
    end
    # A key is the name before the screen (the C name here), or the symbol.
    (text, sc, rt) = gen(Dict("String" => "make_string", "open" => "open_file"))
    @test _ns_fn_names(text) == ["error", "helper", "make_string", "open_file", "plain"]
    @test isempty(sc.moves)                       # the screen moved nothing a key named
    _NS._check_renames(rt, text, "Toy")
    # Without the key the screen's own answer stands, and a chosen name is
    # checked against it like any derived name.
    (text, sc, rt) = gen(Dict("plain" => "open_"))
    @test occursin("`open_`, chosen by \"plain\" (`plain`), is also the derived name of `open`",
                   try _NS._check_renames(rt, text, "Toy"); "" catch e; sprint(showerror, e) end)
    # A chosen name Julia can never bind is refused, naming the key.
    (text, sc, rt) = gen(Dict("plain" => "Ptr"))
    err = try _NS._check_renames(rt, text, "Toy"); "" catch e; sprint(showerror, e) end
    @test occursin("`Ptr`, chosen by \"plain\"", err) && occursin("type of Base or Core", err)
    # A key written as the name the screen gives matches nothing, and the
    # error says how such a function is keyed.
    (text, sc, rt) = gen(Dict("String_" => "make_string"))
    @test "String_" in _ns_fn_names(text)
    err = try _NS._check_renames(rt, text, "Toy"); "" catch e; sprint(showerror, e) end
    @test occursin("String_", err) && occursin("keyed by the\nname it had before", err)
end

# One record per name, a function returning it, one taking it by reference,
# a method of it, and C's own types beside it, as the metadata spells them.
function _ns_record_metadata(names; lang_bool = "bool")
    structs = Dict{String,Any}()
    functions = Any[]
    fn(name, ret, params...; class = "") = Dict{String,Any}(
        "name" => name, "mangled" => "_Z" * name * class, "demangled" => name, "class" => class,
        "is_method" => !isempty(class),
        "return_type" => Dict{String,Any}("c_type" => ret[1], "julia_type" => ret[2]),
        "parameters" => Any[Dict{String,Any}("name" => "a$i", "c_type" => p[1], "julia_type" => p[2])
                            for (i, p) in enumerate(params)])
    for n in names
        structs[n] = Dict{String,Any}("kind" => "struct", "byte_size" => "0x10", "scope" => "", "scopes" => Any[""],
            "members" => Any[
                Dict{String,Any}("name" => "v", "c_type" => "int", "julia_type" => "Cint", "offset" => "0x0", "size" => 4),
                Dict{String,Any}("name" => "w", "c_type" => "double", "julia_type" => "Cdouble", "offset" => "0x8", "size" => 8)])
        structs["Holds" * n] = Dict{String,Any}("kind" => "struct", "byte_size" => "0x18", "scope" => "", "scopes" => Any[""],
            "base_classes" => Any[Dict{String,Any}("type" => n, "offset" => 0)],
            "members" => Any[
                Dict{String,Any}("name" => "inner", "c_type" => n, "julia_type" => n, "offset" => "0x0", "size" => 16),
                Dict{String,Any}("name" => "next", "c_type" => n * "*", "julia_type" => "Ptr{$n}", "offset" => "0x10", "size" => 8)])
        push!(functions, fn("make_" * n, (n, n), ("int", "Cint")))
        push!(functions, fn("take_" * n, ("int", "Cint"), ("const $n&", "Ref{$n}"), ("$n[2]", "NTuple{2, $n}")))
        push!(functions, fn(n, ("void", "Cvoid"), ("$n*", "Ptr{$n}"); class = n))           # the constructor
        push!(functions, fn("~" * n, ("void", "Cvoid"), ("$n*", "Ptr{$n}"); class = n))
        push!(functions, fn("get", ("int", "Cint"), ("const $n*", "Ptr{$n}"); class = n))
    end
    structs["__enum__Mode"] = Dict{String,Any}("kind" => "enum", "underlying_type" => "int", "julia_type" => "Cint",
        "enumerators" => Any[Dict{String,Any}("name" => "Slow", "value" => 0)])
    # C's own types, which the metadata spells with the same Julia names.
    push!(functions, fn("truth", (lang_bool, "Bool"), ("int", "Cint")))
    push!(functions, fn("nop", ("void", "Cvoid"), ("unsigned char", "UInt8"), ("const char*", "Cstring")))
    return Dict{String,Any}("functions" => functions, "struct_definitions" => structs,
                            "record_abi" => Dict{String,Any}(n => Dict{String,Any}("pass" => "value") for n in names),
                            "globals" => Dict{String,Any}("origin" => Dict{String,Any}("c_type" => first(names), "julia_type" => first(names))),
                            "type_registry" => Dict{String,Any}(first(names) * "*" => Dict{String,Any}("inferred" => true, "julia_type" => "Ptr{$(first(names))}")))
end

@testset "records: moved in the metadata, by the C spelling" begin
    names = ["Bool", "Cvoid", "UInt8", "Cstring", "Integer", "Int", "Expr", "Widget"]
    meta = _ns_record_metadata(names)
    pristine = deepcopy(meta)

    # A probe moves every record whose name a fresh module resolves; `Widget` is
    # not one, and neither is a name the C++ sanitizer already moves (`Expr`).
    sc = _NS._NameScreen()
    probed = _NS._screen_records(sc, meta, _NS._sanitize_cpp_type_name)
    @test meta == pristine && probed !== meta                # the library's metadata is not touched
    defs = probed["struct_definitions"]
    for n in ("Bool", "Cvoid", "UInt8", "Cstring", "Integer", "Int")
        @test haskey(defs, "c_" * n) && !haskey(defs, n)
        @test haskey(probed["record_abi"], "c_" * n) && !haskey(probed["record_abi"], n)
    end
    @test haskey(defs, "Widget") && haskey(defs, "Expr") && haskey(defs, "__enum__Mode")
    @test sort!([m.from for m in sc.moves]) == ["Bool", "Cstring", "Cvoid", "Int", "Integer", "UInt8"]
    @test sc.kept[:type] == Set(["Widget", "HoldsBool", "HoldsCvoid", "HoldsUInt8", "HoldsCstring",
                                 "HoldsInteger", "HoldsInt", "HoldsExpr", "HoldsWidget", "Mode"])
    # The C sanitizer moves nothing itself, so there `Expr` is a candidate too.
    sc_c = _NS._NameScreen()
    _NS._screen_records(sc_c, meta, _NS._sanitize_c_type_name)
    @test "Expr" in [m.from for m in sc_c.moves]

    # Only the names a verdict found move in a later generation.
    sc2 = _NS._NameScreen(probe = false, move = Dict((:type, "Bool") => :own_use, (:type, "Integer") => :own_use),
                          judged = Set((:type, n) for n in names))
    moved = _NS._screen_records(sc2, meta, _NS._sanitize_cpp_type_name)
    defs = moved["struct_definitions"]
    @test haskey(defs, "c_Bool") && haskey(defs, "c_Integer") && haskey(defs, "Cvoid") && haskey(defs, "Int")
    by_name = Dict(String(f["name"]) * "/" * String(f["class"]) => f for f in moved["functions"])
    # The record, wherever a C spelling names it; the Julia type follows.
    @test by_name["make_Bool/"]["return_type"] == Dict("c_type" => "c_Bool", "julia_type" => "c_Bool")
    @test by_name["take_Bool/"]["parameters"][1]["c_type"] == "const c_Bool&"
    @test by_name["take_Bool/"]["parameters"][1]["julia_type"] == "Ref{c_Bool}"
    @test by_name["take_Bool/"]["parameters"][2]["julia_type"] == "NTuple{2, c_Bool}"
    # C's `bool` is spelled `Bool` in Julia too, and is not the record.
    @test by_name["truth/"]["return_type"] == Dict("c_type" => "bool", "julia_type" => "Bool")
    @test by_name["nop/"]["parameters"][1]["julia_type"] == "UInt8"
    # The class, and the constructor and destructor named after it.
    @test haskey(by_name, "c_Bool/c_Bool") && haskey(by_name, "~c_Bool/c_Bool") && haskey(by_name, "get/c_Bool")
    @test !haskey(by_name, "Bool/Bool") && haskey(by_name, "Cvoid/Cvoid")
    # A function merely NAMED like the record's maker keeps its name and symbol.
    @test by_name["make_Bool/"]["mangled"] == "_Zmake_Bool" && by_name["get/c_Bool"]["mangled"] == "_ZgetBool"
    # Members, bases and the other tables.
    h = defs["HoldsBool"]
    @test h["members"][1]["c_type"] == "c_Bool" && h["members"][1]["julia_type"] == "c_Bool"
    @test h["members"][2]["julia_type"] == "Ptr{c_Bool}" && h["base_classes"][1]["type"] == "c_Bool"
    @test defs["HoldsCvoid"]["members"][1]["julia_type"] == "Cvoid"                 # not moved this time
    @test defs["c_Bool"]["members"][1] == pristine["struct_definitions"]["Bool"]["members"][1]
    @test moved["globals"]["origin"] == Dict("c_type" => "c_Bool", "julia_type" => "c_Bool")
    @test moved["type_registry"]["c_Bool*"]["julia_type"] == "Ptr{c_Bool}"

    # A record named like the head of a Julia spelling: in `Ptr{Ptr}` and
    # `NTuple{2, Ref}` only the record moves.
    meta_p = _ns_record_metadata(["Ptr", "Ref"])
    moved_p = _NS._screen_records(_NS._NameScreen(), meta_p, _NS._sanitize_c_type_name)
    by_p = Dict(String(f["name"]) * "/" * String(f["class"]) => f for f in moved_p["functions"])
    @test by_p["take_Ptr/"]["parameters"][1] == Dict("name" => "a1", "c_type" => "const c_Ptr&", "julia_type" => "Ref{c_Ptr}")
    @test by_p["take_Ref/"]["parameters"][1]["julia_type"] == "Ref{c_Ref}"
    @test by_p["take_Ref/"]["parameters"][2]["julia_type"] == "NTuple{2, c_Ref}"
    @test by_p["get/c_Ptr"]["parameters"][1]["julia_type"] == "Ptr{c_Ptr}"
    @test moved_p["struct_definitions"]["HoldsPtr"]["members"][2] ==
          Dict("name" => "next", "c_type" => "c_Ptr*", "julia_type" => "Ptr{c_Ptr}", "offset" => "0x10", "size" => 8)
    @test moved_p["type_registry"]["c_Ptr*"]["julia_type"] == "Ptr{c_Ptr}"

    # An enum's key carries a prefix, and a taken alternative is not reused.
    meta2 = Dict{String,Any}("functions" => Any[], "struct_definitions" => Dict{String,Any}(
        "__enum__Task" => Dict{String,Any}("kind" => "enum", "julia_type" => "Cint", "enumerators" => Any[]),
        "c_Task" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x4", "members" => Any[])))
    sc3 = _NS._NameScreen()
    m2 = _NS._screen_records(sc3, meta2, _NS._sanitize_c_type_name)
    @test sort!(collect(keys(m2["struct_definitions"]))) == ["__enum__c_Task_", "c_Task"]

    # A type manifest is keyed and digested by the library's own names.
    manifest = Dict{String,Any}("types" => Dict{String,Any}(
        "c_Bool" => Dict{String,Any}("julia_name" => "c_Bool", "kind" => "struct"),
        "HoldsBool" => Dict{String,Any}("julia_name" => "HoldsBool", "kind" => "struct"),
        "Opaque" => Dict{String,Any}("julia_name" => "Opaque", "kind" => "placeholder")))
    _NS._library_named_manifest!(manifest, meta, sc2.moves)
    @test sort!(collect(keys(manifest["types"]))) == ["Bool", "HoldsBool", "Opaque"]
    @test manifest["types"]["Bool"]["julia_name"] == "c_Bool"
    @test manifest["types"]["HoldsBool"]["entry_sha"] ==
          _NS._sha256_hex(_NS._canonical_text(meta["struct_definitions"]["HoldsBool"]))
end

@testset "a record named like Julia's spelling of a C type is the record, for the thunks too" begin
    # The thunk generator's scalar map answers Julia's spellings of the C
    # types as well (`Bool`, `Cint`, `UInt8`), and reads a library's metadata
    # from disk under the library's own names. A `struct Bool` returned by
    # value was a thunk returning one byte into the wrapper's sixteen.
    T = RepliBuild.JLCSIRGenerator.TypeUtils
    F = RepliBuild.JLCSIRGenerator.FunctionGen
    S = RepliBuild.JLCSIRGenerator.StructGen
    mem(n, c, j, off, size) = Dict{String,Any}("name" => n, "c_type" => c, "julia_type" => j, "offset" => off, "size" => size)
    two(a, b) = Dict{String,Any}("kind" => "struct", "byte_size" => "0x10", "members" => Any[a, b])
    recs = Dict{String,Any}(
        "Bool" => two(mem("v", "int", "Cint", "0x0", 4), mem("w", "double", "Cdouble", "0x8", 8)),
        "UInt8" => two(mem("v", "int", "Cint", "0x0", 4), mem("w", "double", "Cdouble", "0x8", 8)),
        # Eight bytes of two floats, named like Julia's `int`.
        "Cint" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x8",
                                   "members" => Any[mem("a", "float", "Cfloat", "0x0", 4), mem("b", "float", "Cfloat", "0x4", 4)]),
        "Holder" => two(mem("x", "Cint", "Cint", "0x0", 8), mem("y", "long long", "Clonglong", "0x8", 8)),
        "__enum__Cuint" => Dict{String,Any}("kind" => "enum", "julia_type" => "Cuchar", "byte_size" => 1,
                                            "underlying_type" => "unsigned char", "enumerators" => Any[]))
    none = Dict{String,Any}()
    # The scalar map alone still answers the Julia spelling, for the callers
    # that pass one (an enum's underlying type).
    @test T.map_cpp_type("Bool") == "i1" && T.map_cpp_type("Cint") == "i32" && T.map_cpp_type("UInt8") == "i8"
    # Read against the library's records, the name is the record.
    for n in ("Bool", "UInt8", "Cint")
        @test T.map_cpp_type(n, recs) == "!llvm.struct<\"$n\">"
        @test F.by_value_crossing(n, recs, none) == (:layout, n)
        @test S._member_mlir_type(n, recs) == "!llvm.struct<\"$n\">"
        @test F.by_value_crossing(n, none, none) == (:direct, nothing)        # no such record: the scalar
    end
    @test F.by_value_crossing("Cuint", recs, none) == (:enum, "__enum__Cuint")
    # C's own spellings are C's, with or without the records.
    for (c, m) in (("bool", "i1"), ("int", "i32"), ("unsigned char", "i8"), ("double", "f64"))
        @test T.map_cpp_type(c, recs) == m && F.by_value_crossing(c, recs, none) == (:direct, nothing)
    end
    @test T.map_cpp_type("const Bool&", recs) == "!llvm.ptr" && T.map_cpp_type("Widget", recs) == T.map_cpp_type("Widget")
    # A function returning the record and taking C's bool.
    fn = Dict{String,Any}("name" => "make", "mangled" => "_Z4makeb",
        "return_type" => Dict{String,Any}("c_type" => "Bool", "julia_type" => "Bool"),
        "parameters" => Any[Dict{String,Any}("name" => "b", "c_type" => "bool", "julia_type" => "Bool")])
    x = F.thunk_crossings(fn, recs, none; abi = :sysv)
    @test x.ret == (:layout, "Bool") && x.params == [(:direct, nothing)] && x.trap === nothing
    # A member of that type is laid out and classified as the record: two
    # floats are an SSE eightbyte, where Julia's `Cint` would be INTEGER.
    abi = F.sysv_record_abi("Holder", recs)
    @test abi.kind === :regs && abi.classes == [:sse, :int]
    @test S._mlir_layout(S._member_mlir_type("Cint", recs), recs) == (8, 4)
end

@testset "enumerators: prefixed like any collision" begin
    chunks = _NS._EnumChunk[_NS._PendingEnum("Kind", "C enum: Kind", "Cuint",
                                             Pair{String,Any}["String" => 1, "nothing" => 2, "error" => 3,
                                                              "eval" => 4, "Plain" => 5])]
    taken = Dict{String,Symbol}()
    # With no wrap in progress only what no module can bind moves.
    (_, finals) = _NS._render_enum_chunks(chunks, taken)
    @test finals["Kind"] == ["String", "nothing", "error", "Kind_eval", "Plain"]
    # A probe moves every name a fresh module resolves, and says so in the text.
    sc = _NS._NameScreen()
    (out, finals) = _NS.with_name_screen(() -> _NS._render_enum_chunks(chunks, taken), sc)
    @test finals["Kind"] == ["Kind_String", "Kind_nothing", "Kind_error", "Kind_eval", "Plain"]
    @test occursin("the enumerator is `Kind_String`", join(out))
    @test sc.kept[:enumerator] == Set(["Plain"])
    @test Set((m.from, m.to) for m in sc.moves) ==
          Set([("String", "Kind_String"), ("nothing", "Kind_nothing"), ("error", "Kind_error"), ("eval", "Kind_eval")])
    # A later generation moves the ones the verdict named.
    sc2 = _NS._NameScreen(probe = false, move = Dict((:enumerator, "String") => :own_use),
                          judged = Set([(:enumerator, "String"), (:enumerator, "nothing"), (:enumerator, "error")]))
    (out, finals) = _NS.with_name_screen(() -> _NS._render_enum_chunks(chunks, taken), sc2)
    @test finals["Kind"] == ["Kind_String", "nothing", "error", "Kind_eval", "Plain"]
    @test occursin("# the wrapper's own text uses `String`; the enumerator is `Kind_String`.", join(out))
end

@testset "the module's own name" begin
    text(name) = _ns_toy(["plain"]; modname = name,
                         own = "label()::Union{String,Nothing} = nothing")(Dict{String,Any}())[1]
    @test _NS._assert_module_name_free(text("Toy"), "Toy") === nothing
    for name in ("Task", "Ptr", "Dict")                                       # Base names this text does not use
        @test _NS._assert_module_name_free(text(name), name) === nothing
    end
    for name in ("String", "Union", "Nothing", "Cint")                        # ... and ones it does
        err = try _NS._assert_module_name_free(text(name), name, "/p/replibuild.toml"); "" catch e; sprint(showerror, e) end
        @test occursin("module would be named `$name`", err)
        @test occursin("`module_name` under `[wrap]` in /p/replibuild.toml", err)
    end
    for name in ("Libdl", "RepliBuild")                                       # names the wrapper imports
        err = try _NS._assert_module_name_free(text(name), name); "" catch e; sprint(showerror, e) end
        @test occursin("binds `$name` itself", err) && occursin("[wrap]", err)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# The real C generator, through the screen, loaded.
# ─────────────────────────────────────────────────────────────────────────────

function _ns_c_metadata()
    int_fn(n) = Dict{String,Any}("name" => n, "mangled" => n, "demangled" => n,
        "parameters" => Any[Dict{String,Any}("name" => "x", "julia_type" => "Cint", "c_type" => "int")],
        "return_type" => Dict{String,Any}("julia_type" => "Cint", "c_type" => "int"))
    rec(n) = Dict{String,Any}("kind" => "struct", "byte_size" => "0x10",
        "members" => Any[
            Dict{String,Any}("name" => "v", "julia_type" => "Cint", "c_type" => "int", "offset" => "0x0", "size" => 4),
            Dict{String,Any}("name" => "w", "julia_type" => "Cdouble", "c_type" => "double", "offset" => "0x8", "size" => 8)])
    functions = Any[int_fn(n) for n in ("String", "Ptr", "Ref", "Cint", "eval", "include", "Base", "Libdl",
                                        "RepliBuild", "ccall", "LIBRARY_PATH", "METADATA", "nothing",
                                        "unsafe_convert", "unsafe_string", "isfile", "open", "string", "isempty",
                                        "basename", "error", "length", "print", "show", "plain")]
    push!(functions, Dict{String,Any}("name" => "label", "mangled" => "label", "demangled" => "label",
        "parameters" => Any[], "return_type" => Dict{String,Any}("julia_type" => "Cstring", "c_type" => "const char*")))
    push!(functions, Dict{String,Any}("name" => "truth", "mangled" => "truth", "demangled" => "truth",
        "parameters" => Any[Dict{String,Any}("name" => "x", "julia_type" => "Cint", "c_type" => "int")],
        "return_type" => Dict{String,Any}("julia_type" => "Bool", "c_type" => "_Bool")))
    for n in ("Integer", "Bool", "Task")
        push!(functions, Dict{String,Any}("name" => "take_" * n, "mangled" => "take_" * n, "demangled" => "take_" * n,
            "parameters" => Any[Dict{String,Any}("name" => "p", "julia_type" => "Ptr{$n}", "c_type" => "const $n*")],
            "return_type" => Dict{String,Any}("julia_type" => "Cint", "c_type" => "int")))
    end
    return Dict{String,Any}(
        "functions" => functions,
        "struct_definitions" => Dict{String,Any}(
            "Integer" => rec("Integer"), "Bool" => rec("Bool"), "Task" => rec("Task"),
            "__enum__Kind" => Dict{String,Any}("kind" => "enum", "underlying_type" => "unsigned int", "julia_type" => "Cuint",
                "enumerators" => Any[Dict{String,Any}("name" => "Symbol", "value" => 1),
                                     Dict{String,Any}("name" => "stat", "value" => 2),
                                     Dict{String,Any}("name" => "KindPlain", "value" => 3)])),
        "globals" => Dict{String,Any}("Dict" => Dict{String,Any}("julia_type" => "Cint"),
                                      "counter" => Dict{String,Any}("julia_type" => "Cint")))
end

@testset "the C generator through the screen: the module loads, and Base is Base's" begin
    dir = mktempdir()
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "namescreensynth"
    root = "$(escape_string(dir))"
    [link]
    enable_lto = false
    [wrap]
    language = "c"
    [types]
    strictness = "warn"
    allow_unknown_structs = true
    [cache]
    enabled = false
    """)
    cfg = RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml"))
    libref = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
    meta = _ns_c_metadata()
    string_methods = length(methods(String))
    cint_methods = length(methods(Cint))

    runs = Ref(0)
    ((code, _), sc, wrap_meta) = redirect_stdout(devnull) do
        _NS._screened_generations(meta, _NS._sanitize_c_type_name; what = "NameScreenSynth") do md
            runs[] += 1
            (_NS.generate_introspective_module_c(cfg, libref, md, "NameScreenSynth",
                                                 _NS.create_type_registry(cfg), true), nothing)
        end
    end
    @test runs[] == 2                              # `error`, `Task` and others go back
    _NS._assert_wrapper_loadable(code, "NameScreenSynth")
    @test _NS._assert_module_name_free(code, "NameScreenSynth") === nothing

    moves = Dict((m.kind, m.from) => (m.to, m.why) for m in sc.moves)
    # Functions: Julia's refusals, the wrapper's own bindings, its own uses.
    @test moves[(:function, "String")] == ("String_", :base_type)
    @test moves[(:function, "Ptr")] == ("Ptr_", :base_type) && moves[(:function, "Cint")] == ("Cint_", :base_type)
    @test moves[(:function, "Ref")] == ("Ref_", :base_type)
    @test moves[(:function, "eval")] == ("eval_", :every_module) && moves[(:function, "include")] == ("include_", :every_module)
    @test moves[(:function, "ccall")] == ("ccall_", :syntax)
    @test moves[(:function, "Base")] == ("Base_", :own_use) && moves[(:function, "nothing")] == ("nothing_", :own_use)
    # `label` returns a `char*`, and the text that copies it calls `unsafe_string`.
    @test moves[(:function, "unsafe_string")] == ("unsafe_string_", :own_use)
    @test moves[(:function, "Libdl")] == ("Libdl_", :own_binding)
    @test moves[(:function, "RepliBuild")] == ("RepliBuild_", :own_binding)
    @test moves[(:function, "LIBRARY_PATH")] == ("LIBRARY_PATH_", :own_binding)
    @test moves[(:function, "METADATA")] == ("METADATA_", :own_binding)
    @test moves[(:function, "unsafe_convert")] == ("unsafe_convert_", :own_binding)
    # A global's accessor is a function too.
    @test moves[(:function, "Dict")] == ("Dict_", :base_type)
    # Records the text names, and an enumerator.
    @test moves[(:type, "Integer")] == ("c_Integer", :own_use) && moves[(:type, "Bool")] == ("c_Bool", :own_use)
    @test moves[(:enumerator, "Symbol")] == ("Kind_Symbol", :own_use)
    # Names the wrapper's text does not use keep the library's spelling. That
    # includes the functions of Base its header calls (`isfile`, `open`,
    # `string`, `isempty`, `basename`): it calls them as `Base.isfile`, so a
    # library function of that name is the module's own and changes nothing.
    for key in ((:function, "error"), (:function, "length"), (:function, "print"), (:function, "show"),
                (:function, "isfile"), (:function, "open"), (:function, "string"), (:function, "isempty"),
                (:function, "basename"), (:type, "Task"), (:enumerator, "stat"), (:function, "plain"))
        @test (key, haskey(moves, key)) == (key, false)
    end
    bound = _ns_bound(code)
    for n in ("error", "length", "print", "show", "plain", "isfile", "open", "string", "isempty", "basename",
              "String_", "Ptr_", "Ref_", "Cint_", "eval_",
              "include_", "Base_", "Libdl_", "RepliBuild_", "ccall_", "LIBRARY_PATH_", "METADATA_",
              "nothing_", "unsafe_convert_", "unsafe_string_", "Dict_", "Dict__ptr", "counter", "take_Bool")
        @test (n, :function in get(bound, n, Symbol[])) == (n, true)
    end
    @test bound["c_Integer"][1] === :struct && bound["c_Bool"][1] === :struct && bound["Task"][1] === :struct
    @test !haskey(bound, "Integer") && !haskey(bound, "Bool") && !haskey(bound, "String") && !haskey(bound, "nothing")
    @test bound["Kind_Symbol"] == [:const] && bound["stat"] == [:const] && !haskey(bound, "Symbol")
    # The record is `c_Bool` where C names the record, and C's `_Bool` is Bool.
    @test occursin(r"function take_Bool\(p::[^\n]*\)", code)
    @test occursin("Ptr{c_Bool}", code) && occursin("Ptr{c_Integer}", code) && occursin("Ptr{Task}", code)
    @test occursin(r"ccall\(\(:truth, LIBRARY_PATH\), Bool,", code)
    # The export line withholds what it withheld.
    @test occursin(r"# ── Withheld from `export`", code) && !occursin(r"\nexport[^\n]*\berror\b", code)

    @testset "loaded" begin
        genfile = joinpath(dir, "NameScreenSynth.jl")
        write(genfile, code)
        M = include(genfile)
        @test M isa Module
        # Base's types took no method, and are what they were.
        @test length(methods(String)) == string_methods && length(methods(Cint)) == cint_methods
        @test_throws MethodError String(1)
        @test_throws MethodError Ptr(1)
        @test Cint(ns_red) == 3                      # `Cint(::Enum)`: a MethodError once `Cint` had a method here
        @test Bool(1) === true && Integer === Core.Integer
        # Inside the module the wrapper's names mean what its text means.
        @test M.nothing === nothing && M.Base === Base && M.String === String && M.Integer === Integer
        @test M.Bool === Bool && M.Libdl === Libdl && M.RepliBuild === RepliBuild
        @test M.unsafe_string === unsafe_string
        @test M.LIBRARY_PATH == libref && M.METADATA isa Dict
        # The header ran with the library's `isfile`, `open`, `string` and
        # `isempty` in the module: its own calls are Base's.
        @test M._check_build_identity() === nothing
        for n in (:isfile, :open, :string, :isempty, :basename)
            @test parentmodule(getfield(M, n)) === M && getfield(M, n) !== getfield(Base, n)
        end
        # The library's functions are there, under the reported names, and the
        # ones Base also names are the module's own.
        for (n, _) in sc.moves |> x -> [(m.to, m) for m in x if m.kind === :function]
            @test getfield(M, Symbol(n)) isa Function
        end
        @test M.error isa Function && M.error !== Base.error && parentmodule(M.error) === M
        @test M.length !== Base.length && M.show !== Base.show
        @test fieldnames(M.c_Bool) == (:v, :_pad_0, :w) && sizeof(M.c_Integer) == 16
        @test M.Task !== Base.Task && sizeof(M.Task) == 16
        @test M.Kind_Symbol == M.Kind(1) && M.stat == M.Kind(2) && M.KindPlain == M.Kind(3)
        # An integer parameter still takes an integer: `x::Integer` is Julia's.
        @test hasmethod(M.plain, Tuple{Int}) && hasmethod(M.String_, Tuple{Int})
        @test hasmethod(M.take_Bool, Tuple{Base.RefValue{M.c_Bool}})
    end
end

end # The name screen
