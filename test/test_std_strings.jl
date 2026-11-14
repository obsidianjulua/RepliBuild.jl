# test/test_std_strings.jl — a std::string returned by value is a Julia String
# (2026-11-13). Toolchain-free: libstdc++- and libc++-shaped metadata and bytes.
#
# The build records each by-value basic_string's layout (`Compiler._string_layout`:
# leaves with dotted paths and absolute offsets, the character type, the declared
# destructor) on its `record_abi` entry. `StdStrings.string_reader` turns that
# into a `Reader` for the two layouts that exist, and refuses anything else with
# a reason; GeneratorCpp's `_string_return_plan` decides per function, and
# `StdStrings.take_string!` reads the object where C++ built it and ends it.
# The live calls are devtests' test_sret_in_caller_buffer.jl.

using Test
using RepliBuild

const _SS = RepliBuild.StdStrings
const _SW = RepliBuild.Wrapper
const _SC = RepliBuild.Compiler

_leaf(path, ct, off; size = 8) = Dict{String,Any}("path" => path, "c_type" => ct, "offset" => off, "size" => size)
_bits(path, ct, bit, n) = Dict{String,Any}("path" => path, "c_type" => ct, "bit_size" => n, "data_bit_offset" => bit)

# What the build records for libstdc++'s cxx11 string (read off clang 23's DWARF).
const _LIBSTDCXX = Dict{String,Any}("char_type" => "char",
    "destructor" => "_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED4Ev",
    "leaves" => Any[_leaf("_M_dataplus._M_p", "char*", 0), _leaf("_M_string_length", "size_t", 8),
                    _leaf("_anon1._M_local_buf", "char[16]", 16; size = 16),
                    _leaf("_anon1._M_allocated_capacity", "size_t", 16)])
# libc++'s standard layout since LLVM 19: the long/short flag is bit 0, a short
# string's length the seven bits above it and its characters from byte 1; a long
# one's length at 8 and its data pointer at 16.
const _LIBCXX = Dict{String,Any}("char_type" => "char", "destructor" => "_ZNSt3__112basic_stringIcEED4Ev",
    "leaves" => Any[_bits("__rep_._anon1.__l.__is_long_", "size_t", 0, 1),
                    _bits("__rep_._anon1.__l.__cap_", "size_t", 1, 63),
                    _leaf("__rep_._anon1.__l.__size_", "size_t", 8),
                    _leaf("__rep_._anon1.__l.__data_", "char*", 16),
                    _bits("__rep_._anon1.__s.__is_long_", "unsigned char", 0, 1),
                    _bits("__rep_._anon1.__s.__size_", "unsigned char", 1, 7),
                    _leaf("__rep_._anon1.__s.__data_", "char[23]", 1; size = 23)])

@testset "std::string returned by value is a Julia String" begin

@testset "the layout reader: two layouts, refusals that say why" begin
    r = _SS.string_reader(_LIBSTDCXX)
    @test r isa _SS.Reader
    @test (r.kind, r.ptr, r.len) == (:pointer_length, 0, 8)
    c = _SS.string_reader(_LIBCXX)
    @test c isa _SS.Reader
    @test (c.kind, c.ptr, c.len, c.flag_bit, c.short_data, c.short_len_bit, c.short_len_bits) ==
          (:short_long, 16, 8, 0, 1, 1, 7)
    # Refused, each with its reason, never guessed.
    @test occursin("character type is `wchar_t`", _SS.string_reader(merge(_LIBSTDCXX, Dict("char_type" => "wchar_t"))))
    @test occursin("could not be read", _SS.string_reader(Dict{String,Any}("unread" => "virtual base `X`",
                                                                          "char_type" => "char", "leaves" => Any[])))
    old_libcxx = merge(_LIBCXX, Dict{String,Any}("leaves" => filter(l -> !endswith(l["path"], "__is_long_"), _LIBCXX["leaves"])))
    @test occursin("without the `__is_long_` bit", _SS.string_reader(old_libcxx))
    @test occursin("neither libstdc++'s",
                   _SS.string_reader(Dict{String,Any}("char_type" => "char", "leaves" => Any[_leaf("p", "char*", 0)])))
end

@testset "reading the characters where the object is" begin
    # libstdc++, short: `_M_p` points at the object's own local buffer.
    obj = zeros(UInt8, 32)
    GC.@preserve obj begin
        base = pointer(obj)
        unsafe_store!(Ptr{Ptr{UInt8}}(base), base + 16)
        unsafe_store!(Ptr{Csize_t}(base + 8), 5)
        copyto!(obj, 17, codeunits("hello"), 1, 5)
        @test _SS.read_string(base, _SS.string_reader(_LIBSTDCXX)) == "hello"
    end
    # libc++, short: flag 0, length 3 in bits 1..7, characters from byte 1.
    sobj = zeros(UInt8, 24)
    sobj[1] = UInt8(3 << 1)
    copyto!(sobj, 2, codeunits("abc"), 1, 3)
    GC.@preserve sobj @test _SS.read_string(pointer(sobj), _SS.string_reader(_LIBCXX)) == "abc"
    # libc++, long: flag 1, length at 8, data pointer at 16.
    heap = Vector{UInt8}(codeunits("a long string on the heap"))
    lobj = zeros(UInt8, 24)
    GC.@preserve heap lobj begin
        lobj[1] = 0x01
        unsafe_store!(Ptr{Csize_t}(pointer(lobj) + 8), length(heap))
        unsafe_store!(Ptr{Ptr{UInt8}}(pointer(lobj) + 16), pointer(heap))
        @test _SS.read_string(pointer(lobj), _SS.string_reader(_LIBCXX)) == "a long string on the heap"
    end
end

@testset "per function: decoded, or the stand-in with a reason naming the type" begin
    key = "basic_string<char, std::char_traits<char>, std::allocator<char> >"
    ra = Dict{String,Any}(key => Dict{String,Any}("pass" => "reference", "byte_size" => "0x20",
                                                  "string_layout" => _LIBSTDCXX))
    plan = _SW._string_return_plan(key, ra)
    @test plan isa Tuple
    @test plan[2] == "_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev"   # D4 → D1
    @test _SW._string_return_plan("const std::$key", ra) isa Tuple                  # spelled qualified
    # Metadata from before the layout was recorded: a rebuild, said so.
    old = Dict{String,Any}(key => Dict{String,Any}("pass" => "reference", "byte_size" => "0x20"))
    why = _SW._string_return_plan(key, old)
    @test why isa String && occursin("std::string `$key` returned as its raw bytes", why) &&
          occursin("rebuild", why)
    # A wide string names its type and why.
    wkey = "basic_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t> >"
    wra = Dict{String,Any}(wkey => Dict{String,Any}("pass" => "reference",
                                                    "string_layout" => merge(_LIBSTDCXX, Dict("char_type" => "wchar_t"))))
    @test occursin("`$wkey` returned as its raw bytes: its character type is `wchar_t`", _SW._string_return_plan(wkey, wra))
    # No destructor in the DWARF: the stand-in.
    nod = Dict{String,Any}(key => Dict{String,Any}("string_layout" => merge(_LIBSTDCXX, Dict("destructor" => nothing))))
    @test occursin("declares no destructor", _SW._string_return_plan(key, nod))
    # Not a string at all.
    @test _SW._string_return_plan("Ticket", ra) === nothing
end

@testset "the destructor: the library's handle, then the C++ runtime by host" begin
    @test isempty(_SS.cxx_runtime_libraries(:sysv))
    @test isempty(_SS.cxx_runtime_libraries(:aapcs64))
    @test _SS.cxx_runtime_libraries(:win64) == ["libc++.dll", "libstdc++-6.dll"]
    # A symbol nothing defines: C_NULL, never an error.
    lib = Base.Libc.Libdl.dlpath(Base.Libc.Libdl.dlopen(Base.libllvm_path()))
    @test _SS.destructor(lib, "_ZN11no_such_type4noneD1Ev") == C_NULL
    # And with none, the stand-in comes back with a warning naming the function, once.
    slot = Ref(ntuple(i -> UInt8(i), 32))
    @test_logs (:warn, r"f_without_dtor: std::string returned as its raw bytes") begin
        @test _SS.take_string!(slot, _SS.string_reader(_LIBSTDCXX), lib, "_ZN11no_such_type4noneD1Ev",
                               "f_without_dtor") == slot[]
    end
    @test_logs _SS.take_string!(slot, _SS.string_reader(_LIBSTDCXX), lib, "_ZN11no_such_type4noneD1Ev",
                                "f_without_dtor")
end

@testset "the build records the layout, through bases and unions" begin
    # libstdc++'s string as `parse_dwarf_dump`'s toolchain table holds it.
    m(name, ct, off, size) = Dict{String,Any}("name" => name, "c_type" => ct, "offset" => off, "size" => size)
    key = "basic_string<char, std::char_traits<char>, std::allocator<char> >"
    defs = Dict{String,Dict{String,Any}}(
        key => Dict{String,Any}("kind" => "class", "byte_size" => "0x20",
                                "template_params" => Any[Dict{String,Any}("kind" => "type", "name" => "_CharT", "type" => "char")],
                                "members" => Any[m("_M_dataplus", "_Alloc_hider", "0x0", 8),
                                                 m("_M_string_length", "size_t", "0x8", 8),
                                                 m("_anon1", "_anon_u", "0x10", 16)]),
        "_Alloc_hider" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x8",
                                           "members" => Any[m("_M_p", "char*", "0x0", 8)],
                                           "base_classes" => Any[Dict{String,Any}("type" => "allocator<char>",
                                                                                  "offset" => 0, "virtual" => false)]),
        "_anon_u" => Dict{String,Any}("kind" => "union", "byte_size" => "0x10",
                                      "members" => Any[m("_M_local_buf", "char[16]", "0x0", 16),
                                                       m("_M_allocated_capacity", "size_t", "0x0", 8)]))
    # `allocator<char>` has no definition but one byte: an empty base.
    sizes = t -> t == "allocator<char>" ? 1 : nothing
    L = _SC._string_layout(key, defs, Dict{String,Any}(), "_ZN…D4Ev"; byte_size_of = sizes)
    @test !haskey(L, "unread")
    @test L["char_type"] == "char" && L["destructor"] == "_ZN…D4Ev"
    @test [(l["path"], l["offset"]) for l in L["leaves"]] ==
          [("_M_dataplus._M_p", 0), ("_M_string_length", 8), ("_anon1._M_local_buf", 16),
           ("_anon1._M_allocated_capacity", 16)]
    @test _SS.string_reader(L) isa _SS.Reader
    # A base with no definition that is not one empty byte: unread, said why.
    L2 = _SC._string_layout(key, defs, Dict{String,Any}(), nothing)
    @test occursin("base `allocator<char>` has no definition", L2["unread"])
    @test isempty(L2["leaves"])
end

end
