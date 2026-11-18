# =============================================================================
# A record's field may name a factory class (library-free, CI-safe)
#
# A class with a factory function and a finalizer is emitted as a `mutable
# struct X; handle::Ptr{Cvoid}; end` whose inner constructors call the
# factory (the idiomatic section). That section came after every record, and
# a record's field naming the class, `struct Cage { Animal *pet; }` as
# `pet::Ptr{Animal}`, is resolved when the record is defined: the module
# failed to load with `UndefVarError: Animal` (audit, 2026-11-18).
#
# The class's type now goes ahead of every record when a field names it, and
# its constructors follow every record as outer constructors, so a factory
# that takes the record holding the pointer (the cycle) still loads. A class
# no field names keeps its text.
#
# The generator runs on metadata shaped as the build writes it, and the
# module text is evaluated. test_factory_field_order_live.jl (devtests)
# builds test/factory_field_test/ and calls through it.
# =============================================================================

using Test
using RepliBuild
using Libdl

const _FF_W = RepliBuild.Wrapper
const _FF_LIB = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))

_ff_p(n, jt, ct) = Dict{String,Any}("name" => n, "julia_type" => jt, "c_type" => ct)
const _FF_VOID = Dict{String,Any}("c_type" => "void", "julia_type" => "Cvoid", "size" => 0)
_ff_ret(cls) = Dict{String,Any}("c_type" => "$cls*", "julia_type" => "Ptr{$cls}", "size" => 8)
const _FF_INT = Dict{String,Any}("c_type" => "int", "julia_type" => "Cint", "size" => 4)

function _ff_row(mangled, demangled; params = Any[], ret = _FF_VOID, hop = false)
    r = only(RepliBuild.Compiler.parse_function_signatures(
        [Dict{String,Any}("mangled" => mangled, "demangled" => demangled)]))
    r["parameters"] = Any[params...]
    r["return_type"] = ret
    r["is_noexcept"] = true
    r["is_vararg"] = false
    r["has_object_pointer"] = hop
    return r
end

_ff_mem(n, jt, ct, off, size) = Dict{String,Any}("name" => n, "julia_type" => jt, "c_type" => ct,
                                                 "offset" => "0x" * string(off, base = 16), "size" => size)
_ff_rec(size, members...) = Dict{String,Any}("kind" => "struct", "byte_size" => "0x" * string(size, base = 16),
                                             "members" => Any[members...])

# A factory class: one `int` member, made by `make_<x>` and freed by `free_<x>`.
_ff_class(name) = _ff_rec(4, _ff_mem("v_", "Cint", "int", 0, 4))
_ff_make(cls, params...; suffix = "") =
    _ff_row("_Z$(length("make_" * lowercase(cls) * suffix))make_$(lowercase(cls))$(suffix)i",
            "make_$(lowercase(cls))$(suffix)(int)"; params = params, ret = _ff_ret(cls))
_ff_free(cls) = _ff_row("_Z$(length("free_" * lowercase(cls)))free_$(lowercase(cls))P$(length(cls))$(cls)",
                        "free_$(lowercase(cls))($cls*)"; params = (_ff_p("p", "Ptr{$cls}", "$cls*"),))

const _FF_STRUCTS = Dict{String,Any}(
    "Animal" => _ff_class("Animal"),
    "Bird"   => _ff_class("Bird"),
    "Lone"   => _ff_class("Lone"),
    # Zebra points back at a record that points at it: a pointer cycle, which
    # the record sort breaks alphabetically, so Burrow comes first.
    "Zebra"  => _ff_rec(8, _ff_mem("home", "Ptr{Burrow}", "Burrow*", 0, 8)),
    "Burrow" => _ff_rec(8, _ff_mem("z", "Ptr{Zebra}", "Zebra*", 0, 8)),
    # A class with a deleter and no factory: a `ManagedDog`, its DWARF struct emitted.
    "Dog"    => _ff_class("Dog"),
    "Kennel" => _ff_rec(8, _ff_mem("d", "Ptr{Dog}", "Dog*", 0, 8)),
    # The shapes that name a factory class in a field.
    "Cage"   => _ff_rec(16, _ff_mem("pet", "Ptr{Animal}", "Animal*", 0, 8),
                            _ff_mem("n", "Cint", "int", 8, 4)),
    "Den"    => _ff_rec(8, _ff_mem("pack", "Ptr{Ptr{Animal}}", "Animal**", 0, 8)),
    "Ark"    => _ff_rec(24, _ff_mem("pets", "NTuple{3, Ptr{Animal}}", "Animal*[3]", 0, 24)),
    # Two factory classes in one record.
    "Aviary" => _ff_rec(16, _ff_mem("b", "Ptr{Bird}", "Bird*", 0, 8),
                            _ff_mem("a", "Ptr{Animal}", "Animal*", 8, 8)),
    "__enum__Kind" => Dict{String,Any}("kind" => "enum", "underlying_type" => "int", "julia_type" => "Cint",
                                       "byte_size" => "0x4",
                                       "enumerators" => Any[Dict{String,Any}("name" => "Sparrow", "value" => 0),
                                                            Dict{String,Any}("name" => "Crow", "value" => 1)]),
)

const _FF_FUNCTIONS = Any[
    _ff_make("Animal", _ff_p("legs", "Cint", "int")),
    # The cycle: a factory of the class `Cage` holds a pointer to takes a `Cage`.
    _ff_row("_Z15make_animal_cag4Cage", "make_animal_cag(Cage)";
            params = (_ff_p("c", "Cage", "Cage"),), ret = _ff_ret("Animal")),
    _ff_free("Animal"),
    # A factory naming a record defined after `Bird`'s, another factory class and an enum.
    _ff_row("_Z9make_birdP6AviaryP6Animal4Kind", "make_bird(Aviary*, Animal*, Kind)";
            params = (_ff_p("home", "Ptr{Aviary}", "Aviary*"), _ff_p("prey", "Ptr{Animal}", "Animal*"),
                      _ff_p("kind", "Kind", "Kind")), ret = _ff_ret("Bird")),
    _ff_free("Bird"),
    _ff_make("Lone", _ff_p("v", "Cint", "int")),
    _ff_free("Lone"),
    _ff_make("Zebra", _ff_p("v", "Cint", "int")),
    _ff_free("Zebra"),
    _ff_row("_Z11destroy_dogP3Dog", "destroy_dog(Dog*)"; params = (_ff_p("d", "Ptr{Dog}", "Dog*"),)),
    _ff_row("_Z9cage_legsPK4Cage", "cage_legs(Cage const*)";
            params = (_ff_p("c", "Ptr{Cage}", "Cage const*"),), ret = _FF_INT),
]

function _ff_generate()
    dir = mktempdir()
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "factory_fields"
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
    meta = Dict{String,Any}("functions" => deepcopy(_FF_FUNCTIONS), "globals" => Dict{String,Any}(),
                            "struct_definitions" => deepcopy(_FF_STRUCTS))
    res = redirect_stdout(devnull) do
        _FF_W.generate_introspective_module_cpp(cfg, _FF_LIB, meta, "FactoryFields",
                                                _FF_W.create_type_registry(cfg), true)
    end
    return res isa Tuple ? String(first(res)) : String(res)
end

# Where `needle` first appears, or 0.
_ff_at(code, needle) = (r = findfirst(needle, code); r === nothing ? 0 : first(r))

const _FF_CODE = _ff_generate()

@testset "a record's field may name a factory class" begin
    code = _FF_CODE

    @testset "the module parses" begin
        bad = Any[]
        walk(x) = x isa Expr && (x.head in (:incomplete, :error) ? push!(bad, x) : foreach(walk, x.args))
        walk(Meta.parseall(code))
        @test isempty(bad)
    end

    @testset "every field keeps the class's type" begin
        @test occursin("    pet::Ptr{Animal}\n", code)
        @test occursin("    pack::Ptr{Ptr{Animal}}\n", code)
        @test occursin("    pets::NTuple{3, Ptr{Animal}}\n", code)
        @test occursin("    b::Ptr{Bird}\n", code) && occursin("    a::Ptr{Animal}\n", code)
        # The record the cycle put first: `Ptr{Cvoid}` before this change.
        @test occursin("    z::Ptr{Zebra}\n", code)
        @test occursin("    d::Ptr{Dog}\n", code)
    end

    @testset "a named class's type precedes every record, its constructors follow them" begin
        records = [_ff_at(code, "\nstruct $r\n") for r in ("Cage", "Den", "Ark", "Aviary", "Burrow", "Kennel")]
        @test all(>(0), records)
        for cls in ("Animal", "Bird", "Zebra")
            ty = _ff_at(code, "\nmutable struct $cls\n    handle::Ptr{Cvoid}\n")
            @test (cls, ty > 0 && ty < minimum(records)) == (cls, true)
            # The type has only the inner constructor that stores the handle.
            body = code[ty:ty + first(findfirst("\nend\n", code[ty:end]))]
            @test (cls, occursin("$cls(::Base.Val{:_rb_adopt}, handle::Ptr{Cvoid}) = new(handle)", body)) == (cls, true)
            @test (cls, occursin("function $cls(", body)) == (cls, false)
            # Its factory constructors come after every record, and attach the finalizer.
            ctor = _ff_at(code, "\nfunction $cls(")
            @test (cls, ctor > maximum(records)) == (cls, true)
            @test (cls, occursin("obj = $cls(Base.Val(:_rb_adopt), Ptr{Cvoid}(handle))", code)) == (cls, true)
            @test (cls, occursin("ccall((:_Z$(length("free_" * lowercase(cls)))free_$(lowercase(cls))P$(length(cls))$(cls), LIBRARY_PATH), Cvoid, (Ptr{Cvoid},), o.handle)", code)) == (cls, true)
        end
        # The cycle's constructor takes the record and is after it.
        @test _ff_at(code, "\nfunction Animal(c::Cage)") > _ff_at(code, "\nstruct Cage\n")
        @test occursin("\nfunction Bird(home::Ptr{Aviary}, prey::Ptr{Animal}, kind::Kind)", code)
        # The manifest reads both shapes as a factory struct.
        kinds = _FF_W._emitted_type_kinds(code)
        @test [kinds[c] for c in ("Animal", "Bird", "Zebra", "Lone")] == fill("idiomatic", 4)
        @test kinds["ManagedDog"] == "managed"
    end

    @testset "a class no field names keeps its text" begin
        @test occursin("\nmutable struct Lone\n    handle::Ptr{Cvoid}\nfunction Lone(v::Cint)\n", code)
        @test !occursin("Lone(::Base.Val", code)
        @test _ff_at(code, "\nmutable struct Lone\n") > _ff_at(code, "# Idiomatic Class Wrappers")
    end

    @testset "the write-time guard: a type named above its definition" begin
        G = _FF_W._assert_types_defined_before_use
        refused(text) = try
            G(text, "Probe"); ""
        catch e
            sprint(showerror, e)
        end
        @test refused(code) == ""
        # The shape the generator wrote before the fix: refused, naming the field.
        old = """
        module Probe
        struct Cage
            pet::Ptr{Animal}
            pets::NTuple{3, Ptr{Animal}}
        end
        mutable struct Animal
            handle::Ptr{Cvoid}
            function Animal(c::Cage)
                new(C_NULL)
            end
        end
        end
        """
        msg = refused(old)
        @test occursin("2 type name(s)", msg)
        @test occursin("`Animal` in a field of `Cage`, defined further down", msg)
        # An inner constructor naming a later record, and a method naming a later type.
        @test occursin("`Cage` in a method signature",
                       refused("module P\nmutable struct A\n    h::Ptr{Cvoid}\n    A(c::Cage) = new(C_NULL)\nend\nstruct Cage\n    x::Cint\nend\nend\n"))
        @test occursin("`Later` in a method signature",
                       refused("module P\nf(x::Ptr{Later}; k::Later = Later()) = 1\nstruct Later end\nend\n"))
        @test occursin("`Later` in a type parameter's bound",
                       refused("module P\nstruct S{T <: Later}\n    x::T\nend\nabstract type Later end\nend\n"))
        # What is not refused: a struct naming itself, type parameters and
        # `where` variables, Base and Core names, qualified names, names the
        # module never binds, bodies, defaults and return types.
        ok = """
        module P
        const Imported = Base.Int
        struct Node{T}
            next::Ptr{Node{T}}
            v::T
            n::Imported
            q::Base.Val{:x}
            u::Unbound
        end
        f(x::Ptr{T}, y::Type{Node{T}}) where {T} = Later()
        g(x::Cint = Later())::Later = x
        h(x) = Later
        struct Later end
        end
        """
        @test refused(ok) == ""
    end

    @testset "the module evaluates" begin
        # Loading, not calling, is what failed: the JIT engine `__init__` would
        # start needs the toolchain, so this module starts none.
        text = replace(code, "RepliBuild.JITManager.initialize_global_jit(LIBRARY_PATH)" => "nothing")
        m = Module(:FactoryFieldsProbe)
        Base.include_string(m, text)
        mod = Base.invokelatest(getfield, m, :FactoryFields)
        T(s) = Base.invokelatest(getfield, mod, s)
        @test fieldtype(T(:Cage), :pet) === Ptr{T(:Animal)}
        @test fieldtype(T(:Den), :pack) === Ptr{Ptr{T(:Animal)}}
        @test fieldtype(T(:Ark), :pets) === NTuple{3, Ptr{T(:Animal)}}
        @test fieldtype(T(:Aviary), :b) === Ptr{T(:Bird)}
        @test fieldtype(T(:Burrow), :z) === Ptr{T(:Zebra)}
        @test fieldtype(T(:Kennel), :d) === Ptr{T(:Dog)}
        @test sizeof(T(:Ark)) == 24 && sizeof(T(:Cage)) == 16
        @test fieldtype(T(:ManagedDog), :handle) === Ptr{T(:Dog)}
        # The factory constructors exist, the cycle's included.
        @test Base.invokelatest(hasmethod, T(:Animal), Tuple{Cint})
        @test Base.invokelatest(hasmethod, T(:Animal), Tuple{T(:Cage)})
        @test Base.invokelatest(hasmethod, T(:Bird), Tuple{Ptr{T(:Aviary)}, Ptr{T(:Animal)}, T(:Kind)})
        # A pointer to one converts as before.
        a = Base.invokelatest(T(:Animal), Base.Val(:_rb_adopt), Ptr{Cvoid}(UInt(0x10)))
        @test Base.invokelatest(Base.unsafe_convert, Ptr{T(:Animal)}, a) == Ptr{T(:Animal)}(UInt(0x10))
    end
end
