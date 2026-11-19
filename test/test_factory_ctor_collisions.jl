# =============================================================================
# Factories that share a constructor signature (library-free, CI-safe)
#
# A factory class (a class with factory functions and a finalizer) gets one
# constructor per factory, taking the factory's parameters. Two factories
# whose parameters have one Julia dispatch signature gave it one method
# twice: `make_animal(int legs)` and `make_animal_doubled(int half)` both
# made `Animal(::Cint)`. Included as a file, the second definition replaced
# the first silently (`Animal(Cint(4))` made 8 legs); in a precompiled
# package Julia refuses the overwrite and the package does not load
# (2026-11-18). The constructor dedup keyed on the parameter TEXT.
#
# Now a signature one factory has stays its plain constructor, and a
# signature several share is none of theirs: each takes `Val(:<factory>)`
# first. The generator runs on metadata shaped as the build writes it, the
# module text is evaluated, and the write-time guard against a method defined
# twice (`_assert_no_duplicate_methods`) is checked on the text and on the
# shape the generator wrote before. test_factory_ctor_collisions_live.jl
# (devtests) builds test/factory_ctor_test/ and calls through it.
# =============================================================================

using Test
using RepliBuild
using Libdl

const _FC_W = RepliBuild.Wrapper
const _FC_LIB = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))

_fc_p(n, jt, ct) = Dict{String,Any}("name" => n, "julia_type" => jt, "c_type" => ct)
const _FC_VOID = Dict{String,Any}("c_type" => "void", "julia_type" => "Cvoid", "size" => 0)
_fc_ret(cls) = Dict{String,Any}("c_type" => "$cls*", "julia_type" => "Ptr{$cls}", "size" => 8)

function _fc_row(mangled, demangled; params = Any[], ret = _FC_VOID, hop = false)
    r = only(RepliBuild.Compiler.parse_function_signatures(
        [Dict{String,Any}("mangled" => mangled, "demangled" => demangled)]))
    r["parameters"] = Any[params...]
    r["return_type"] = ret
    r["is_noexcept"] = true
    r["is_vararg"] = false
    r["has_object_pointer"] = hop
    return r
end

# A factory `name(<ctype> pname)` of `cls`; `code` is the Itanium code of the type.
const _FC_CODES = Dict("int" => "i", "int32_t" => "i", "double" => "d", "long" => "l", "long long" => "x")
_fc_make(cls, name, pname, jt, ct) =
    _fc_row("_Z$(length(name))$(name)$(_FC_CODES[ct])", "$name($ct)";
            params = (_fc_p(pname, jt, ct),), ret = _fc_ret(cls))
_fc_free(cls) = _fc_row("_Z$(length("free_" * lowercase(cls)))free_$(lowercase(cls))P$(length(cls))$(cls)",
                        "free_$(lowercase(cls))($cls*)"; params = (_fc_p("p", "Ptr{$cls}", "$cls*"),))

_fc_mem(n, jt, ct, off, size) = Dict{String,Any}("name" => n, "julia_type" => jt, "c_type" => ct,
                                                 "offset" => "0x" * string(off, base = 16), "size" => size)
_fc_rec(size, members...) = Dict{String,Any}("kind" => "struct", "byte_size" => "0x" * string(size, base = 16),
                                             "members" => Any[members...])
_fc_class() = _fc_rec(4, _fc_mem("v_", "Cint", "int", 0, 4))

const _FC_STRUCTS = Dict{String,Any}(
    "Animal" => _fc_class(), "Bird" => _fc_class(), "Cat" => _fc_class(),
    "Zebra" => _fc_class(), "Lone" => _fc_class(),
    # A record's field names Zebra: its type is emitted early, its
    # constructors as outer ones after every record.
    "Pen" => _fc_rec(8, _fc_mem("z", "Ptr{Zebra}", "Zebra*", 0, 8)),
)

const _FC_FUNCTIONS = Any[
    # Two factories, one signature.
    _fc_make("Animal", "make_animal", "legs", "Cint", "int"),
    _fc_make("Animal", "make_animal_doubled", "half", "Cint", "int"),
    # One function listed twice (two DWARF entries) is one factory.
    _fc_make("Animal", "make_animal", "legs", "Cint", "int"),
    _fc_free("Animal"),
    # Three factories: two collide through two spellings of one type
    # (`int` is `Cint`, `int32_t` is `Int32`), one does not.
    _fc_make("Bird", "make_bird", "n", "Cint", "int"),
    _fc_make("Bird", "make_bird_twice", "m", "Int32", "int32_t"),
    _fc_make("Bird", "make_bird_light", "w", "Cdouble", "double"),
    _fc_free("Bird"),
    # `long` is `long long`'s width on LP64 and `int`'s on LLP64: on every
    # host exactly one pair of these collides, and the third stays plain.
    _fc_make("Cat", "make_cat", "a", "Cint", "int"),
    _fc_make("Cat", "make_cat_long", "b", "Clong", "long"),
    _fc_make("Cat", "make_cat_wide", "c", "Clonglong", "long long"),
    _fc_free("Cat"),
    # The early form, colliding.
    _fc_make("Zebra", "make_zebra", "v", "Cint", "int"),
    _fc_make("Zebra", "make_zebra_striped", "s", "Cint", "int"),
    _fc_free("Zebra"),
    # No collision: the text it always had.
    _fc_make("Lone", "make_lone", "v", "Cint", "int"),
    _fc_free("Lone"),
    # Two overloads of a method whose pointer parameters the proxy takes as
    # `::Any`, under different names: one proxy, which the raw wrapper's own
    # dispatch serves (keyed by text, it was defined twice).
    _fc_row("_ZN4Lone3putEP6Animal", "Lone::put(Animal*)"; hop = true,
            params = (_fc_p("this", "Ptr{Lone}", "Lone*"), _fc_p("a", "Ptr{Animal}", "Animal*"))),
    _fc_row("_ZN4Lone3putEP4Bird", "Lone::put(Bird*)"; hop = true,
            params = (_fc_p("this", "Ptr{Lone}", "Lone*"), _fc_p("b", "Ptr{Bird}", "Bird*"))),
]

function _fc_generate()
    dir = mktempdir()
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "factory_ctors"
    root = "$(escape_string(dir))"
    [link]
    enable_lto = false
    [wrap]
    language = "cpp"
    [types]
    strictness = "warn"
    allow_unknown_structs = true
    [cache]
    enabled = false
    """)
    cfg = RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml"))
    meta = Dict{String,Any}("functions" => deepcopy(_FC_FUNCTIONS), "globals" => Dict{String,Any}(),
                            "struct_definitions" => deepcopy(_FC_STRUCTS))
    log = joinpath(dir, "stdout.txt")
    res = open(log, "w") do io
        redirect_stdout(io) do
            _FC_W.generate_introspective_module_cpp(cfg, _FC_LIB, meta, "FactoryCtors",
                                                    _FC_W.create_type_registry(cfg), true)
        end
    end
    return (res isa Tuple ? String(first(res)) : String(res)), read(log, String)
end

const _FC_CODE, _FC_OUT = _fc_generate()

_fc_at(code, needle) = (r = findfirst(needle, code); r === nothing ? 0 : first(r))

# Which of the Cat pairs share a type on this host, by the types themselves.
const _FC_LONG_IS_LLONG = Clong === Clonglong
const _FC_INT_IS_LONG = Cint === Clong

@testset "factories that share a constructor signature" begin
    code = _FC_CODE

    @testset "the module parses" begin
        bad = Any[]
        walk(x) = x isa Expr && (x.head in (:incomplete, :error) ? push!(bad, x) : foreach(walk, x.args))
        walk(Meta.parseall(code))
        @test isempty(bad)
    end

    @testset "a shared signature is no factory's positional constructor" begin
        @test occursin("function Animal(::Base.Val{:make_animal}, legs::Cint)\n", code)
        @test occursin("function Animal(::Base.Val{:make_animal_doubled}, half::Cint)\n", code)
        @test !occursin("function Animal(legs::Cint)", code)
        @test !occursin("function Animal(half::Cint)", code)
        # Two spellings of one type.
        @test occursin("function Bird(::Base.Val{:make_bird}, n::Cint)\n", code)
        @test occursin("function Bird(::Base.Val{:make_bird_twice}, m::Int32)\n", code)
        @test occursin("function Bird(w::Cdouble)\n", code)
        # The early form: outer constructors, tagged the same way.
        @test occursin("\nfunction Zebra(::Base.Val{:make_zebra}, v::Cint)\n", code)
        @test occursin("\nfunction Zebra(::Base.Val{:make_zebra_striped}, s::Cint)\n", code)
        # One factory listed twice is one constructor.
        @test count("function Animal(::Base.Val{:make_animal}, ", code) == 1
    end

    @testset "two spellings of one width collide where they are one type" begin
        @test _FC_LONG_IS_LLONG != _FC_INT_IS_LONG      # LP64 or LLP64: exactly one pair
        tagged_long = occursin("function Cat(::Base.Val{:make_cat_long}, b::Clong)\n", code)
        @test tagged_long                                # `long` collides on both models
        @test occursin("function Cat(::Base.Val{:make_cat_wide}, c::Clonglong)\n", code) == _FC_LONG_IS_LLONG
        @test occursin("function Cat(c::Clonglong)\n", code) == !_FC_LONG_IS_LLONG
        @test occursin("function Cat(::Base.Val{:make_cat}, a::Cint)\n", code) == _FC_INT_IS_LONG
        @test occursin("function Cat(a::Cint)\n", code) == !_FC_INT_IS_LONG
    end

    @testset "a class with no collision keeps its text" begin
        @test occursin("\nmutable struct Lone\n    handle::Ptr{Cvoid}\nfunction Lone(v::Cint)\n", code)
        @test !occursin("Lone(::Base.Val", code)
        lone_doc = code[_fc_at(code, "Idiomatic Julia wrapper for C++ class `Lone`"):_fc_at(code, "\nmutable struct Lone\n")]
        @test !occursin("called by name", lone_doc)
    end

    @testset "a method proxy is keyed by its dispatch signature" begin
        @test count("\nput(obj::Lone, ", code) == 1
        @test occursin("\nput(obj::Lone, a::Any) = GC.@preserve obj Lone_put(obj.handle, a)\n", code)
    end

    @testset "the docstring and the wrap report name each factory" begin
        doc = code[_fc_at(code, "Idiomatic Julia wrapper for C++ class `Animal`"):_fc_at(code, "\nmutable struct Animal\n")]
        @test occursin("- `Animal(Val(:make_animal), legs)` calls `make_animal`", doc)
        @test occursin("- `Animal(Val(:make_animal_doubled), half)` calls `make_animal_doubled`", doc)
        zdoc = code[_fc_at(code, "Idiomatic Julia wrapper for C++ class `Zebra`"):_fc_at(code, "\nmutable struct Zebra\n")]
        @test occursin("- `Zebra(Val(:make_zebra_striped), s)` calls `make_zebra_striped`", zdoc)
        @test occursin("wrap: class Animal has factories sharing a constructor signature, each called by name " *
                       "as Animal(Val(:<factory>), …): make_animal, make_animal_doubled", _FC_OUT)
        @test occursin("wrap: class Bird has factories sharing a constructor signature, each called by name " *
                       "as Bird(Val(:<factory>), …): make_bird, make_bird_twice", _FC_OUT)
        @test !occursin("class Lone has", _FC_OUT)
    end

    @testset "the early form keeps its order" begin
        pen = _fc_at(code, "\nstruct Pen\n")
        @test pen > 0
        @test 0 < _fc_at(code, "\nmutable struct Zebra\n    handle::Ptr{Cvoid}\n") < pen
        @test _fc_at(code, "\nfunction Zebra(::Base.Val{:make_zebra}, ") > pen
        G = _FC_W._assert_types_defined_before_use
        @test (G(code, "FactoryCtors"); true)
    end

    @testset "no method is defined twice" begin
        defs = _FC_W._method_definitions(code)
        sigs = first.(defs)
        @test length(sigs) > 20                               # the walk saw the module
        @test allunique(sigs)
        @test "Animal(Base.Val{:make_animal},Int32)" in sigs
        @test "Zebra(Base.Val{:_rb_adopt},Ptr{Cvoid})" in sigs  # inner constructors are read
        @test (_FC_W._assert_no_duplicate_methods(code, "FactoryCtors"); true)
        @test (_FC_W._assert_wrapper_loadable(code, "FactoryCtors"); true)
    end

    @testset "the write-time guard: one method defined twice" begin
        refused(text) = try
            _FC_W._assert_no_duplicate_methods(text, "Probe"); ""
        catch e
            sprint(showerror, e)
        end
        # The shape the generator wrote before the fix.
        old = """
        module Probe
        mutable struct Animal
            handle::Ptr{Cvoid}
        function Animal(legs::Cint)
            new(C_NULL)
        end
        function Animal(half::Cint)
            new(C_NULL)
        end
        end
        end
        """
        msg = refused(old)
        @test occursin("1 method signature(s)", msg)
        @test occursin("`Animal(Int32)` at lines 4, 7", msg)
        # The outer form, two spellings of one type, and a proxy pair.
        @test occursin("`Zebra(Int32)`", refused("module P\nfunction Zebra(v::Cint) end\nfunction Zebra(s::Int32) end\nend\n"))
        @test occursin("`f(Animal,Any)`", refused("module P\nf(obj::Animal, a::Any) = 1\nf(obj::Animal, b::Any) = 2\nend\n"))
        # An optional argument defines a shorter method too.
        @test occursin("`g(Int64)`", refused("module P\ng(x::Int64, y::Int64 = 1) = 1\ng(z::Int64) = 2\nend\n"))
        # Not refused: overloads, keywords, `where` clauses that differ,
        # different qualified names, a docstring'd method, branches.
        ok = """
        module P
        f(x::Cint) = 1
        f(x::Cdouble) = 2
        f(x::Cint, y) = 3
        h(x::T) where {T <: Integer} = 1
        h(x::T) where {T <: AbstractFloat} = 2
        k(x; a = 1) = 1
        Base.show(io::IO, x::Int8) = 1
        show(io::IO, x::Int8) = 2
        \"\"\"doc\"\"\"
        m(x::Int8) = 1
        if Sys.iswindows()
            n() = 1
        else
            n() = 2
        end
        struct S
            x::Int
            S(::Base.Val{:a}, x::Int) = new(x)
            S(::Base.Val{:b}, x::Int) = new(x)
        end
        end
        """
        @test refused(ok) == ""
    end

    @testset "the module evaluates, with one method per factory signature" begin
        text = replace(code, "RepliBuild.JITManager.initialize_global_jit(LIBRARY_PATH)" => "nothing")
        m = Module(:FactoryCtorsProbe)
        Base.include_string(m, text)
        mod = Base.invokelatest(getfield, m, :FactoryCtors)
        T(s) = Base.invokelatest(getfield, mod, s)
        nmeth(s) = length(Base.invokelatest(methods, T(s)))
        @test nmeth(:Animal) == 2
        @test nmeth(:Bird) == 3
        @test nmeth(:Cat) == 3
        @test nmeth(:Zebra) == 3                       # the adopting constructor and two factories
        @test nmeth(:Lone) == 1
        has(s, tt) = Base.invokelatest(hasmethod, T(s), tt)
        @test has(:Animal, Tuple{Val{:make_animal}, Cint})
        @test has(:Animal, Tuple{Val{:make_animal_doubled}, Cint})
        @test !has(:Animal, Tuple{Cint})
        @test has(:Bird, Tuple{Cdouble}) && !has(:Bird, Tuple{Cint})
        @test has(:Zebra, Tuple{Val{:make_zebra_striped}, Cint}) && !has(:Zebra, Tuple{Cint})
        @test has(:Lone, Tuple{Cint})
        # The bare shared call is a MethodError, not a pick.
        err = try
            Base.invokelatest(T(:Animal), Cint(4)); nothing
        catch e
            e
        end
        @test err isa MethodError
    end
end
