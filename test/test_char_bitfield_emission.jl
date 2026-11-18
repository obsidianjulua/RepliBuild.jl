# =============================================================================
# `char` by value, signed bitfields and enum parameters, emitted (2026-11-18)
#
# The 2026-11-17 audit's B14 and B15, and one C-generator mismatch:
#   * a signed bitfield was read as unsigned (`int c:7 = -3` came back 125)
#     and its setter threw for -3: it now reads sign-extended, typed as its
#     declared C type, and its setter takes that range and throws outside it
#     (`_bitfield_signed_type`, `_bitfield_sign_extend`, `_bitfield_signed_store`);
#   * a `char` parameter by value was typed and passed as `UInt8` while a
#     `char` return is `Cchar`; it is now `Cchar` (`_char_param_ctype`) and
#     converted from any valid `char` or byte (`_integer_arg_conversion`), and
#     the thunk extends it by the host's `char`, not by the metadata's `UInt8`
#     (`FunctionGen._narrow_int_extension`);
#   * a C enum parameter refused an integer, which C converts implicitly.
#
# No toolchain: both generators on synthetic metadata. The live twin is
# test_char_bitfield_live.jl. Every expectation that depends on the host's
# `char` is stated through `Cchar`, never as Int8 or UInt8.
#
# Negative checks: `_bitfield_signed_type` returning `nothing` fails the
# bitfield testsets; `_char_param_ctype` returning `julia_type` fails the
# `char` signatures and ccall tuples; the open-enum branch in GeneratorC's
# parameter loop removed fails "C enum parameter"; the `char` lines of
# `_narrow_int_extension` removed fail "thunk extension" on x86-64.
# =============================================================================

using Test
using Libdl
using RepliBuild

const _CB_W = RepliBuild.Wrapper
const _CB_FG = RepliBuild.JLCSIRGenerator.FunctionGen

"Text of one top-level generated function."
function _cb_fn(code::AbstractString, header::AbstractString)
    i = findfirst(header, code)
    i === nothing && return ""
    rest = SubString(code, first(i))
    j = findfirst(r"\nend\b", rest)
    return j === nothing ? String(rest) : String(rest[1:last(j)])
end

function _cb_config(lang)
    dir = mktempdir()
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "charbitsynth$lang"
    root = "$(escape_string(dir))"
    [link]
    enable_lto = false
    [wrap]
    language = "$lang"
    [types]
    strictness = "warn"
    allow_unknown_structs = true
    [cache]
    enabled = false
    """)
    return RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml"))
end

_cb_libref() = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))

_cb_bf(name, c, j, bits, off) = Dict{String,Any}("name" => name, "c_type" => c, "julia_type" => j,
                                                 "bit_size" => bits, "data_bit_offset" => off,
                                                 "offset" => nothing, "size" => 4)
_cb_p(name, c, j) = Dict{String,Any}("name" => name, "c_type" => c, "julia_type" => j)
function _cb_func(name, params; ret = ("int", "Cint"), mangled = name, noexcept = true)
    Dict{String,Any}("name" => name, "mangled" => mangled, "demangled" => name,
                     "class" => "", "is_method" => false, "is_vararg" => false,
                     "is_noexcept" => noexcept, "nounwind" => true, "exported" => true,
                     "has_object_pointer" => nothing,
                     "parameters" => Any[_cb_p(p...) for p in params],
                     "return_type" => Dict{String,Any}("c_type" => ret[1], "julia_type" => ret[2]))
end

# `int s:7` at bit 3, `unsigned u:5`, `long long w:40` across six bytes,
# `signed char t:4`, `enum Mode m:3` (underlying int), `enum Pos p:2`
# (underlying unsigned int), `char c:5` (the host's `char`), `_Bool b:1`.
const _CB_BITS = Dict{String,Any}(
    "kind" => "struct", "byte_size" => "0x10",
    "members" => Any[
        _cb_bf("a", "unsigned int", "Cuint", 3, 0),
        _cb_bf("s", "int", "Cint", 7, 3),
        _cb_bf("u", "unsigned int", "Cuint", 5, 10),
        _cb_bf("w", "long long", "Clonglong", 40, 15),
        _cb_bf("t", "signed char", "Int8", 4, 56),
        _cb_bf("m", "Mode", "Mode", 3, 64),
        _cb_bf("p", "Pos", "Pos", 2, 67),
        _cb_bf("c", "char", "UInt8", 5, 72),
        _cb_bf("b", "_Bool", "Bool", 1, 77),
    ])
const _CB_ENUMS = Dict{String,Any}(
    "__enum__Mode" => Dict{String,Any}("kind" => "enum", "underlying_type" => "int", "julia_type" => "Cint",
        "enumerators" => Any[Dict{String,Any}("name" => "MODE_LOW", "value" => -1),
                             Dict{String,Any}("name" => "MODE_HIGH", "value" => 1)]),
    "__enum__Pos" => Dict{String,Any}("kind" => "enum", "underlying_type" => "unsigned int", "julia_type" => "Cuint",
        "enumerators" => Any[Dict{String,Any}("name" => "POS_A", "value" => 0),
                             Dict{String,Any}("name" => "POS_B", "value" => 1)]),
    # A bitmask: an integer alias, whose parameters stay as they are.
    "__enum__Mask" => Dict{String,Any}("kind" => "enum", "underlying_type" => "unsigned int", "julia_type" => "Cuint",
        "enumerators" => Any[Dict{String,Any}("name" => "MASK_A", "value" => 1),
                             Dict{String,Any}("name" => "MASK_B", "value" => 2),
                             Dict{String,Any}("name" => "MASK_C", "value" => 4),
                             Dict{String,Any}("name" => "MASK_D", "value" => 8)]),
)

@testset "char, signed bitfields and enum parameters: emitted" begin

@testset "helpers" begin
    @test _CB_W._char_param_ctype("UInt8", "char") == "Cchar"
    @test _CB_W._char_param_ctype("UInt8", "const char") == "Cchar"
    @test _CB_W._char_param_ctype("Cchar", "signed char") == "Int8"
    @test _CB_W._char_param_ctype("UInt8", "unsigned char") == "UInt8"
    # Not by value: pointers to char keep their type.
    @test _CB_W._char_param_ctype("Ptr{UInt8}", "char*") == "Ptr{UInt8}"
    @test _CB_W._char_param_ctype("Ptr{UInt8}", "const char*") == "Ptr{UInt8}"

    conv = eval(Meta.parse("c -> " * _CB_W._integer_arg_conversion("Cchar", "c")))
    byte(b) = reinterpret(Cchar, UInt8(b))
    checked = 0
    for (arg, b) in ((65, 0x41), (0x41, 0x41), (byte(0x41), 0x41), (250, 0xfa), (-6, 0xfa),
                     (0xfa, 0xfa), (-128, 0x80), (255, 0xff), (Int16(0), 0x00), (true, 0x01))
        @test (arg, conv(arg)) === (arg, byte(b))
        checked += 1
    end
    @test checked == 10
    @test_throws InexactError conv(256)
    @test_throws InexactError conv(-129)
    @test _CB_W._integer_arg_conversion("Cint", "x") == "Cint(x)"

    st(m) = _CB_W._bitfield_signed_type(m, _CB_ENUMS)
    mem = Dict(m["name"] => m for m in _CB_BITS["members"])
    @test st(mem["s"]) == "Cint"
    @test st(mem["w"]) == "Clonglong"
    @test st(mem["t"]) == "Int8"
    @test st(mem["m"]) == "Cint"                   # the enum's underlying int
    @test st(mem["p"]) === nothing                 # unsigned underlying
    @test st(mem["c"]) == (Cchar <: Signed ? "Cchar" : nothing)
    @test st(mem["a"]) === nothing
    @test st(mem["b"]) === nothing                 # bool: unsigned, unchanged
    @test st(_cb_bf("x", "int", "Cint", 40, 0)) === nothing   # wider than its type (C++)
end

@testset "thunk extension follows the host's char" begin
    ext(c, j) = _CB_FG._narrow_int_extension("i8", Dict{String,Any}("c_type" => c, "julia_type" => j))
    @test ext("char", "UInt8") == (Cchar <: Signed ? "sext" : "zext")
    @test ext("const char", "UInt8") == (Cchar <: Signed ? "sext" : "zext")
    @test ext("char", "Cchar") == (Cchar <: Signed ? "sext" : "zext")
    @test ext("signed char", "Cchar") == "sext"
    @test ext("unsigned char", "UInt8") == "zext"
    # The Tier 2 body converts a `char` as the ccall body does.
    @test _CB_W._tier2_invoke_args(["c"], ["Cchar"], [true], Dict{String,String}()) ==
          [_CB_W._integer_arg_conversion("Cchar", "c")]
end

@testset "C generator" begin
    cfg = _cb_config("c")
    meta = Dict{String,Any}(
        "functions" => Any[
            _cb_func("put_char", [("c", "char", "UInt8")]),
            _cb_func("put_schar", [("c", "signed char", "Cchar")]),
            _cb_func("put_uchar", [("c", "unsigned char", "UInt8")]),
            _cb_func("put_cstr", [("s", "const char*", "Ptr{UInt8}")]),
            _cb_func("put_mode", [("m", "Mode", "Mode")]),
            _cb_func("put_mask", [("k", "Mask", "Mask")]),
            _cb_func("get_char", []; ret = ("char", "Cchar")),
        ],
        "struct_definitions" => merge(Dict{String,Any}("Bits" => _CB_BITS), _CB_ENUMS),
        "globals" => Dict{String,Any}())
    code = _CB_W.generate_introspective_module_c(cfg, _cb_libref(), meta, "CharBitC",
                                                 _CB_W.create_type_registry(cfg), true; abi = :sysv)

    @testset "char by value" begin
        b = _cb_fn(code, "function put_char(")
        @test occursin("function put_char(c::Union{Integer, Base.Enum})", b)
        @test occursin("c_c = " * _CB_W._integer_arg_conversion("Cchar", "c"), b)
        @test occursin("(Cchar,), c_c)", b)
        # signed char is signed on every host and takes any Integer now
        b = _cb_fn(code, "function put_schar(")
        @test occursin("(Int8,), c_c)", b)
        @test occursin("c_c = Int8(c)", b)
        @test occursin("(UInt8,), c_c)", _cb_fn(code, "function put_uchar("))
        # A pointer to char is not a char.
        @test occursin("Ptr{UInt8}", _cb_fn(code, "function put_cstr("))
        @test !occursin("Cchar", _cb_fn(code, "function put_cstr("))
    end

    @testset "C enum parameter takes an integer" begin
        b = _cb_fn(code, "function put_mode(")
        @test occursin("function put_mode(m::Union{Mode, Integer})", b)
        @test occursin("m_c = Mode(m)", b)
        @test occursin("(Mode,), m_c)", b)
        # A bitmask alias is unchanged.
        b = _cb_fn(code, "function put_mask(")
        @test !isempty(b)
        @test !occursin("Union{Mask, Integer}", b)
    end

    M = Module(:CharBitC_eval)
    Base.include_string(M, code)
    Mod = getfield(M, :CharBitC)

    @testset "C signed bitfields" begin
        # s = -3 (bits 3..9), w = -123456789012 (bits 15..54), t = -8 (56..59),
        # m = -1 (64..66), p = 1 (67..68), a = 5, u = 17.
        img = zeros(UInt8, 16)
        put!(v, off, n) = for i in 0:n-1
            bit = off + i
            ((v >> i) & 1) == 1 && (img[bit >> 3 + 1] |= UInt8(1) << (bit & 7))
        end
        put!(5, 0, 3); put!(-3, 3, 7); put!(17, 10, 5); put!(-123456789012, 15, 40)
        put!(-8, 56, 4); put!(-1, 64, 3); put!(1, 67, 2); put!(0x13, 72, 5)
        s = Mod.Bits(NTuple{16,UInt8}(img))
        @test Mod.get_a(s) === UInt32(5)
        @test Mod.get_u(s) === UInt32(17)
        @test Mod.get_s(s) === Cint(-3)
        @test Mod.get_w(s) === Clonglong(-123456789012)
        @test Mod.get_t(s) === Int8(-8)
        @test Mod.get_m(s) === Cint(-1)
        @test Mod.get_p(s) === UInt32(1)
        # `char c:5` holding 0x13 (top bit set): the host's `char` decides.
        @test Mod.get_c(s) == (Cchar <: Signed ? -13 : 0x13)
        checked = 0
        for v in (-64, -3, -1, 0, 1, 63)
            s2 = Mod.set_s(s, v)
            @test (v, Mod.get_s(s2)) == (v, v)
            @test (Mod.get_a(s2), Mod.get_u(s2), Mod.get_w(s2)) == (5, 17, -123456789012)
            checked += 1
        end
        @test checked == 6
        @test_throws InexactError Mod.set_s(s, 64)
        @test_throws InexactError Mod.set_s(s, -65)
        for v in (-(Int64(1) << 39), (Int64(1) << 39) - 1, -2)
            s2 = Mod.set_w(s, v)
            @test (v, Mod.get_w(s2)) == (v, v)
            @test (Mod.get_s(s2), Mod.get_t(s2)) == (-3, -8)
        end
        @test_throws InexactError Mod.set_w(s, Int64(1) << 39)
        @test Mod.get_t(Mod.set_t(s, 7)) == 7
        @test_throws InexactError Mod.set_t(s, -9)
        @test Mod.get_m(Mod.set_m(s, -4)) == -4
        # Unsigned fields keep their masking store.
        @test Mod.get_u(Mod.set_u(s, 33)) == 1
        @test Mod.get_p(Mod.set_p(s, 3)) == 3
    end
end

@testset "C++ generator" begin
    cfg = _cb_config("cpp")
    flags = Dict{String,Any}("kind" => "struct", "byte_size" => "0x4",
        "members" => Any[_cb_bf("on", "unsigned int", "Cuint", 1, 0),
                         _cb_bf("level", "int", "Cint", 4, 1),
                         _cb_bf("rest", "unsigned int", "Cuint", 11, 5)])
    meta = Dict{String,Any}(
        "functions" => Any[
            _cb_func("ns::char_value", [("c", "char", "UInt8")]; mangled = "_ZN2ns10char_valueEc", noexcept = false),
            _cb_func("ns::char_nx", [("c", "char", "UInt8")]; mangled = "_ZN2ns7char_nxEc"),
            _cb_func("ns::ovl", [("x", "int", "Cint")]; mangled = "_ZN2ns3ovlEi"),
            _cb_func("ns::ovl", [("c", "char", "UInt8")]; mangled = "_ZN2ns3ovlEc"),
            _cb_func("ns::ovl", [("c", "signed char", "Cchar")]; mangled = "_ZN2ns3ovlEa"),
            _cb_func("ns::ovl", [("c", "unsigned char", "UInt8")]; mangled = "_ZN2ns3ovlEh"),
        ],
        "struct_definitions" => Dict{String,Any}("Flags" => flags),
        "globals" => Dict{String,Any}(),
        "record_abi" => Dict{String,Any}())
    res = _CB_W.generate_introspective_module_cpp(cfg, _cb_libref(), meta, "CharBitCpp",
                                                  _CB_W.create_type_registry(cfg), true)
    code = res isa Tuple ? res[1] : res

    @testset "char by value" begin
        @test occursin("function ns_char_value(c::Integer)", code)
        @test occursin(_CB_W._integer_arg_conversion("Cchar", "c"), _cb_fn(code, "function ns_char_value("))
        b = _cb_fn(code, "function ns_char_nx(")
        @test occursin("function ns_char_nx(c::Integer)", b)
        @test occursin("(Cchar,), c_c)", b)
    end

    @testset "char in an overload set" begin
        # The prediction, for each host's `char`: the `int` and `char`
        # overloads share `ns_ovl(::Integer)`, so both are strict; the
        # `signed char` and `unsigned char` ones are already their own types.
        funcs = meta["functions"]
        for cchar in ("Int8", "UInt8")
            conc(t) = t == "Cchar" ? cchar : _CB_W._concrete_ctype(t)
            strict = _CB_W._integer_width_overloads(funcs[3:6], Set{String}(),
                (j, c) -> _CB_W._char_param_ctype(j, c); concrete = conc)
            @test (cchar, sort!(collect(keys(strict)))) == (cchar, ["_ZN2ns3ovlEc", "_ZN2ns3ovlEi"])
        end
        sigs = [m.match for m in eachmatch(r"^function ns_ovl\([^)]*\)"m, code)]
        @test "function ns_ovl(x::Cint)" in sigs
        # One method per dispatch type: on the host where `char` is the same
        # type as `signed char` or `unsigned char`, those two are one method.
        keys_ = vcat([_CB_W._method_sig_keys(m) for m in sigs]...)
        @test allunique(keys_)
        @test length(keys_) == length(unique([Cint, Cchar, Int8, UInt8]))
    end

    @testset "C++ signed bitfield" begin
        @test occursin("function get_level(s::Flags)::Cint", code)
        @test occursin("function get_on(s::Flags)::UInt32", code)
        # Only the record and its accessors: the module's __init__ needs a JIT.
        txt = join([match(r"mutable struct Flags\n.*?\nend\nFlags\(\) = [^\n]*"s, code).match,
                    (_cb_fn(code, "function $f(") for f in ("get_on", "get_level", "get_rest",
                                                             "set_level!", "set_rest!"))...], "\n")
        F = Module(:CharBitCppFlags)
        Base.include_string(F, txt)
        f = F.Flags(NTuple{4,UInt8}(reinterpret(NTuple{4,UInt8}, htol(UInt32(1 | (11 << 1) | (1000 << 5))))))
        @test F.get_on(f) === UInt32(1)
        @test F.get_level(f) === Cint(-5)
        @test F.get_rest(f) === UInt32(1000)
        for v in (-8, -1, 0, 7)
            F.set_level!(f, v)
            @test (v, F.get_level(f), F.get_rest(f), F.get_on(f)) == (v, v, 1000, 1)
        end
        @test_throws InexactError F.set_level!(f, 8)
        @test_throws InexactError F.set_level!(f, -9)
    end
end

end
