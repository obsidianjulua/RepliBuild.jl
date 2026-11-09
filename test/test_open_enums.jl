#!/usr/bin/env julia
# test/test_open_enums.jl — a C enum takes any value of its underlying type
# (2026-11-08).
#
# Both generators emitted a C enum as `@enum`, which accepts declared values
# only. A C enum is an integer type with named constants, and APIs pass values
# between or beyond them: libvterm's `VTERM_KEY_FUNCTION(n)` is
# `VTERM_KEY_FUNCTION_0 + n`, so `VTermKey(257)` (F1) was an ArgumentError and
# LlamaChat's terminal could not send function keys; imgui's key chords
# (`ImGuiMod_Ctrl | ImGuiKey_S`) the same. Every non-bitmask enum is now a
# `RepliBuild.OpenEnums.OpenEnum` (`_open_enum_chunk`): a primitive type of the
# underlying width, subtyping `Base.Enum`, constructible from any value.
#
# What is pinned: construction of undeclared values at every width, the
# conversions and `ccall` passing `Base.Enum` already gave, printing (a declared
# value as `@enum` printed it, an undeclared one as `T(v)`, never
# `<invalid #v>`), duplicates, `instances`, combining operators; then a wrapper
# from GeneratorC that passes an undeclared value to libc's `abs` and gets one
# back, and GeneratorCpp's emission. No toolchain.

using Test
using RepliBuild
using Libdl

const OE_W = RepliBuild.Wrapper
const OE = RepliBuild.OpenEnums

# Render an enum section the way both generators do, and load it into a module
# that imports RepliBuild, as every wrapper's header does.
function _oe_load(enums::Vector{Tuple{String,String,Vector{Pair{String,Any}}}}; taken = Dict{String,Symbol}())
    chunks = Any[OE_W._PendingEnum(n, "C enum: $n", u, m) for (n, u, m) in enums]
    out, _ = OE_W._render_enum_chunks(chunks, taken)
    src = join(out)
    m = Module(:OpenEnumProbe)
    Core.eval(m, :(import RepliBuild))
    Base.include_string(m, src)
    return m, src
end
_g(m, s) = Base.invokelatest(getfield, m, s)

@testset "Open-range C enums" begin

@testset "an undeclared value is a value of the enum" begin
    m, src = _oe_load([("VTermKey", "Cuint", Pair{String,Any}[
        "VTERM_KEY_NONE" => 0, "VTERM_KEY_ENTER" => 1,
        "VTERM_KEY_FUNCTION_0" => 256, "VTERM_KEY_FUNCTION_MAX" => 511])])
    @test occursin("primitive type VTermKey <: RepliBuild.OpenEnums.OpenEnum{Cuint} 32 end", src)
    @test !occursin("@enum", src)
    K = _g(m, :VTermKey)
    f1 = K(Int(_g(m, :VTERM_KEY_FUNCTION_0)) + 1)                 # VTERM_KEY_FUNCTION(1)
    @test f1 isa K && f1 isa OE.OpenEnum{Cuint} && f1 isa Base.Enum
    @test Int(f1) == 257 && Cuint(f1) === Cuint(257) && Integer(f1) === Cuint(257)
    @test K(f1) === f1
    @test convert(K, 257) === f1                    # a struct field or ccall slot takes an Int
    @test sizeof(K) == 4 && isprimitivetype(K)
    @test_throws InexactError K(-1)                 # the underlying type still bounds it
    @test_throws InexactError K(2^33)
end

@testset "every width" begin
    for (u, bits, T) in (("UInt8", 8, UInt8), ("Cshort", 16, Cshort), ("Cint", 32, Cint),
                         ("Int64", 64, Int64), ("Clong", 8 * sizeof(Clong), Clong))
        m, src = _oe_load([("E_$u", u, Pair{String,Any}["E_$(u)_A" => 0, "E_$(u)_B" => 5])])
        @test occursin("primitive type E_$u <: RepliBuild.OpenEnums.OpenEnum{$u} $bits end", src)
        E = _g(m, Symbol("E_$u"))
        @test sizeof(E) == sizeof(T)
        @test Integer(E(7)) === T(7)
    end
end

@testset "printing: declared as @enum printed, undeclared as T(v)" begin
    m, _ = _oe_load([("Mode", "Cint", Pair{String,Any}["MODE_A" => 0, "MODE_B" => 3, "MODE_ALIAS" => 3])])
    M = _g(m, :Mode)
    b = _g(m, :MODE_B)
    @test string(b) == "MODE_B"
    @test Symbol(b) === :MODE_B
    @test sprint(show, b; context = :compact => true) == "MODE_B"
    @test sprint(show, MIME"text/plain"(), b) == "MODE_B::Mode = 3"
    # a duplicate value is a second name for it; the first declared prints
    @test _g(m, :MODE_ALIAS) === b && string(_g(m, :MODE_ALIAS)) == "MODE_B"
    u = M(9)
    @test string(u) == "Mode(9)"
    @test Symbol(u) === Symbol("Mode(9)")
    @test endswith(sprint(show, u), "Mode(9)")
    @test !occursin("invalid", sprint(show, u)) && !occursin("invalid", sprint(show, MIME"text/plain"(), u))
    @test sprint(show, MIME"text/plain"(), u) == "Mode(9)::Mode = 9"
    @test instances(M) == (_g(m, :MODE_A), b)       # each value once, declared order
    @test typemin(M) === _g(m, :MODE_A) && typemax(M) === b
    @test occursin("MODE_B = 3", sprint(show, MIME"text/plain"(), M))
    @test OE.enumerators(M) == (:MODE_A => 0, :MODE_B => 3, :MODE_ALIAS => 3)
end

@testset "values combine and compare as integers" begin
    m, _ = _oe_load([("Key", "Cint", Pair{String,Any}["KEY_S" => 564, "MOD_CTRL" => 4096])])
    K = _g(m, :Key)
    chord = _g(m, :MOD_CTRL) | _g(m, :KEY_S)
    @test chord isa K && Int(chord) == 4096 | 564
    @test chord & _g(m, :MOD_CTRL) === _g(m, :MOD_CTRL)
    @test xor(chord, _g(m, :KEY_S)) === _g(m, :MOD_CTRL)
    @test ~K(0) === K(-1)
    @test isless(_g(m, :KEY_S), _g(m, :MOD_CTRL)) && sort([K(9), K(1)]) == [K(1), K(9)]
    @test hash(K(3)) == hash(K(3)) && K(3) == K(3) && K(3) != K(4)
    io = IOBuffer(); write(io, K(70000)); seekstart(io)
    @test read(io, K) === K(70000)
end

@testset "the collision rule still names the bound spelling" begin
    m, src = _oe_load([("kind", "UInt8", Pair{String,Any}["null" => 0, "object" => 7])];
                      taken = Dict{String,Symbol}("object" => :struct))
    @test occursin("const kind_object = kind(7)", src)
    @test occursin(":kind_object => 7", src)
    @test string(_g(m, :kind_object)) == "kind_object"
end

@testset "GeneratorC: an undeclared value crosses a ccall both ways" begin
    dir = mktempdir()
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "oe"
    root = "$(escape_string(dir))"
    [wrap]
    language = "c"
    [types]
    strictness = "warn"
    allow_unknown_structs = true
    [cache]
    enabled = false
    """)
    cfg = RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml"))
    # `abs` is libc's: `int abs(int)`, declared here as taking and returning the
    # enum. A Linux dlopen handle resolves a name through the library's
    # dependencies, so libjulia's path reaches it; Windows looks a name up in
    # the named DLL's own exports only, so the C runtime is named there.
    libref = Sys.iswindows() ? Libdl.dlpath("msvcrt") :
             abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
    @test Libdl.dlsym(Libdl.dlopen(libref), :abs; throw_error = false) !== nothing
    ent(n, v) = Dict{String,Any}("name" => n, "value" => v)
    meta = Dict{String,Any}(
        "functions" => Any[Dict{String,Any}(
            "name" => "abs", "mangled" => "abs", "demangled" => "abs", "class" => "",
            "is_method" => false, "is_vararg" => false, "is_noexcept" => false, "exported" => true,
            "parameters" => Any[Dict{String,Any}("name" => "k", "c_type" => "vkey", "julia_type" => "vkey")],
            "return_type" => Dict{String,Any}("c_type" => "vkey", "julia_type" => "vkey", "size" => 4))],
        "globals" => Dict{String,Any}(),
        "struct_definitions" => Dict{String,Any}("__enum__vkey" => Dict{String,Any}(
            "kind" => "enum", "underlying_type" => "int", "julia_type" => "Cint",
            "enumerators" => Any[ent("VK_NONE", 0), ent("VK_F0", 256), ent("VK_FMAX", 511)])))
    code = OE_W.generate_introspective_module_c(cfg, libref, meta, "OeC", OE_W.create_type_registry(cfg), false)
    @test occursin("primitive type vkey <: RepliBuild.OpenEnums.OpenEnum{Cint} 32 end", code)
    @test occursin(r"^function abs\(k::vkey\)::vkey"m, code)
    holder = Module(:OeCHolder)
    Base.include_string(holder, code)
    M = _g(holder, :OeC)
    vkey = _g(M, :vkey)
    r = Base.invokelatest(_g(M, :abs), vkey(-257))
    @test r === vkey(257)                           # an undeclared value went in and came back
    @test string(r) == "vkey(257)"
    @test Base.invokelatest(_g(M, :abs), _g(M, :VK_F0)) === _g(M, :VK_F0)
end

@testset "GeneratorCpp emits the same type" begin
    dir = mktempdir()
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "oecpp"
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
    meta = Dict{String,Any}("functions" => Any[], "globals" => Dict{String,Any}(), "record_abi" => Dict{String,Any}(),
        "struct_definitions" => Dict{String,Any}("__enum__ImGuiKey" => Dict{String,Any}(
            "kind" => "enum", "underlying_type" => "int", "julia_type" => "Cint",
            "enumerators" => Any[Dict{String,Any}("name" => "ImGuiKey_None", "value" => 0),
                                 Dict{String,Any}("name" => "ImGuiKey_S", "value" => 564),
                                 Dict{String,Any}("name" => "ImGuiMod_Ctrl", "value" => 4096)])))
    res = OE_W.generate_introspective_module_cpp(cfg, libref, meta, "OeCpp", OE_W.create_type_registry(cfg), false)
    code = res isa Tuple ? res[1] : res
    @test occursin("primitive type ImGuiKey <: RepliBuild.OpenEnums.OpenEnum{Cint} 32 end", code)
    @test occursin("const ImGuiMod_Ctrl = ImGuiKey(4096)", code)
    @test !occursin(r"^@enum"m, code)
end

end
