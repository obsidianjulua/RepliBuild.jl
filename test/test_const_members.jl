# =============================================================================
# Records with const-qualified members, and a tiled record whose body is a
# byte region (2026-11-21, library-free)
#
# Found by the Hub's llama.cpp on the 4.6.0 release sweep: the library's whole
# thunk module failed to lower ("MarshalRetOp: input must be an LLVM struct
# type"), so nothing of llama.cpp ran through Tier 2. The thunk was
# `LLM_TN::operator()`, returning `LLM_TN_IMPL` by value: five const members
# that tile 24 bytes. Two causes, both older than the function that showed
# them (an inline method the forced unit defines since 4.5.1):
#
#   A  `map_cpp_type` is a table of exact spellings. `const int`, `const Kind`
#      and `volatile int` match nothing and get its `!llvm.ptr` fallback.
#      `StructGen._member_mlir_type` cut the qualifiers for its template
#      checks and then handed the UNCUT spelling to `map_cpp_type`, so every
#      qualified scalar member was eight bytes of pointer. The body failed its
#      own size check and the record fell to a byte region.
#   B  FunctionGen asks `is_struct_packed` (the metadata's members tile the
#      record), takes the emitted body as the "packed" type, and marshals it
#      field by field. It never looked at the body: a byte region has no
#      fields, `jlcs.marshal_ret` / `jlcs.marshal_arg` on an array does not
#      verify, and one such thunk takes the module.
#
#   C  The same table, asked about a SIGNATURE. A definition's DWARF keeps a
#      by-value parameter's qualifiers (`double scaled(const double x, const
#      int n)`), so the thunk declared its callee `(ptr, ptr) -> f64`: the
#      double was passed in an integer register, and `scaled(2.5, 4)` was
#      0.0 with no error. Found while testing A, not by llama.cpp.
#
# A and C are fixed where the spelling is handed to the table (the qualifiers
# are cut first; a pointer's spelling maps the same either way). B is fixed
# on both crossing paths:
# a byte region crosses as its bytes (`_byte_blob_type`), whatever the
# metadata says about tiling, so a member StructGen cannot type costs the
# record its field body and nothing else. The live calls are in
# test_const_members_live.jl.
# =============================================================================

using Test
using RepliBuild

const _CMFG = RepliBuild.JLCSIRGenerator.FunctionGen
const _CMSG = RepliBuild.JLCSIRGenerator.StructGen

_cmm(name, ct, off, size) = Dict{String,Any}("name" => name, "c_type" => ct, "julia_type" => "Any",
                                             "offset" => off, "size" => size)
_cmrec(size, members...) =
    Dict{String,Any}("kind" => "struct", "byte_size" => "0x" * string(size, base = 16), "members" => Any[members...])
_cmfn(name, ret, params...) =
    Dict{String,Any}("name" => name, "mangled" => name,
                     "parameters" => Any[Dict{String,Any}("name" => "p$i", "c_type" => t) for (i, t) in enumerate(params)],
                     "return_type" => Dict{String,Any}("c_type" => ret))

const _CM_STRUCTS = Dict{String,Any}(
    "Inner"  => _cmrec(8, _cmm("a", "int", "0x0", 4), _cmm("b", "int", "0x4", 4)),
    # llama.cpp's LLM_TN_IMPL, its two enums as the integers they are.
    "Name"   => _cmrec(24, _cmm("arch", "const int", "0x0", 4), _cmm("tensor", "const unsigned int", "0x4", 4),
                           _cmm("suffix", "const const char*", "0x8", 8),
                           _cmm("bid", "const int", "0x10", 4), _cmm("xid", "const int", "0x14", 4)),
    "Pair"   => _cmrec(8, _cmm("a", "const int", "0x0", 4), _cmm("b", "const unsigned int", "0x4", 4)),
    # Padding after the first member: not tiled, so never on the marshal path.
    "Mixed"  => _cmrec(16, _cmm("tag", "const char", "0x0", 1), _cmm("count", "volatile int", "0x4", 4),
                           _cmm("scale", "const double", "0x8", 8)),
    "Holder" => _cmrec(12, _cmm("in", "const Inner", "0x0", 8), _cmm("n", "const int", "0x8", 4)),
    # The metadata tiles, and one member has a spelling the table has no
    # entry for: the body is a byte region whatever A does.
    "Odd"    => _cmrec(8, _cmm("a", "_Atomic int", "0x0", 4), _cmm("b", "int", "0x4", 4)),
)

# The type signature of every marshal op in a module: `(operand) -> result`.
_cm_marshal_sigs(ir) = [String(m.captures[1]) for m in eachmatch(r"jlcs\.marshal_(?:ret|arg) [^\n]* : (\([^\n]*)", ir)]

@testset "Const-qualified members, and a tiled record with a byte-region body" begin
    S = _CM_STRUCTS

    @testset "A: a qualified member is typed as the type it qualifies" begin
        for (spelling, want) in ("const int" => "i32", "volatile int" => "i32", "const volatile int" => "i32",
                                 "int const" => "i32", "const unsigned char" => "i8", "const bool" => "i1",
                                 "const long long" => "i64", "const float" => "f32", "const double" => "f64",
                                 # Pointers were right before and stay so.
                                 "const char*" => "!llvm.ptr", "const const char*" => "!llvm.ptr",
                                 "char *const" => "!llvm.ptr",
                                 # A record of the library held const is that record.
                                 "const Inner" => "!llvm.struct<\"Inner\">")
            @test (spelling, _CMSG._member_mlir_type(spelling, S)) == (spelling, want)
        end
        # Unqualified spellings are what they were.
        @test _CMSG._member_mlir_type("int", S) == "i32"
        @test _CMSG._member_mlir_type("Inner", S) == "!llvm.struct<\"Inner\">"
    end

    @testset "A: a record of const members has a field body of its DWARF size" begin
        @test _CMSG.get_llvm_equivalent_type_string("Name", S["Name"], S) ==
              "!llvm.struct<packed (i32, i32, !llvm.ptr, i32, i32) >"
        @test _CMSG.get_llvm_equivalent_type_string("Pair", S["Pair"], S) == "!llvm.struct<packed (i32, i32) >"
        @test _CMSG.get_struct_definition_string("Mixed", S["Mixed"], S) ==
              "!llvm.struct<\"Mixed\", (i8, !llvm.array<3 x i8>, i32, f64)>"
        for (k, size) in ("Name" => 24, "Pair" => 8, "Holder" => 12)
            lay = _CMSG._mlir_layout(_CMSG.get_llvm_equivalent_type_string(k, S[k], S), S)
            @test (k, lay === nothing ? nothing : lay[1]) == (k, size)
        end
        # None of them is a byte region any more.
        for k in ("Name", "Pair", "Mixed", "Holder")
            @test (k, startswith(_CMSG.get_struct_definition_string(k, S[k], S), "!llvm.array<")) == (k, false)
        end
    end

    @testset "C: a qualified scalar in a signature is the scalar" begin
        funcs = Any[_cmfn("scaled", "double", "const double", "const int"),
                    _cmfn("doubled", "const double", "const double"),
                    _cmfn("halved", "float", "const float"),
                    _cmfn("narrow", "int", "const unsigned char", "const signed char", "const short", "const bool"),
                    _cmfn("wide", "const long long", "const long long", "volatile int"),
                    # Pointers and references map as they did.
                    _cmfn("text", "const char*", "const char*", "char *const", "const int&")]
        ir = _CMFG.generate_function_thunks(funcs, S; abi = :sysv)
        decls = Dict(m.captures[1] => m.match for m in eachmatch(r"func\.func private @(\w+)\(.*", ir))
        @test get(decls, "scaled", "") == "func.func private @scaled(f64, i32) -> f64"
        @test get(decls, "doubled", "") == "func.func private @doubled(f64) -> f64"
        @test get(decls, "halved", "") == "func.func private @halved(f32) -> f32"
        # Narrower than int: the callee is declared taking the extended i32.
        @test get(decls, "narrow", "") == "func.func private @narrow(i32, i32, i32, i32) -> i32"
        @test get(decls, "wide", "") == "func.func private @wide(i64, i32) -> i64"
        @test get(decls, "text", "") == "func.func private @text(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr"
        # Each narrow argument is extended by its own signedness.
        body = String(match(r"func\.func @narrow_thunk.*?\n\}"s, ir).match)
        @test [m.captures[1] for m in eachmatch(r"llvm\.(zext|sext) %val_\d", body)] == ["zext", "sext", "sext", "zext"]
        # A record qualified in a signature was right before and stays so.
        # (`long long`, not `long`: `long` is 32 bits on Win64, and this pins a width.)
        rec = _CMFG.generate_function_thunks(Any[_cmfn("total", "long long", "const Name"), _cmfn("make", "const Pair")], S; abi = :sysv)
        @test occursin("func.func private @total(!llvm.struct<packed (i32, i32, !llvm.ptr, i32, i32) >) -> i64", rec)
        @test occursin("func.func private @make() -> !llvm.struct<packed (i32, i32) >", rec)
    end

    @testset "B: the case the guard is for is real" begin
        # Tiled by the metadata, a byte region as emitted. Without this the
        # checks below would pass on a generator that types `_Atomic int`.
        @test _CMSG.is_struct_packed(S["Odd"])
        @test _CMSG.get_llvm_equivalent_type_string("Odd", S["Odd"], S) == "!llvm.array<8 x i8>"
    end

    @testset "B: a byte-region body crosses as its bytes, never through a marshal op" begin
        funcs = Any[_cmfn("make_odd", "Odd"), _cmfn("odd_sum", "int", "Odd"),
                    _cmfn("make_name", "Name"), _cmfn("name_total", "long long", "Name"),
                    _cmfn("pair_of", "Pair", "int"), _cmfn("pair_sum", "long long", "Pair"),
                    _cmfn("holder_sum", "int", "Holder")]
        ir = _CMFG.generate_function_thunks(funcs, S; abi = :sysv)   # the SysV statements, on every host
        decls = Dict(m.captures[1] => m.match for m in eachmatch(r"func\.func private @(\w+)\(.*", ir))
        @test !occursin("// no thunk for", ir)

        # `Odd` in both directions: the blob, and the class its facts state.
        @test get(decls, "make_odd", "") == "func.func private @make_odd() -> !llvm.struct<packed (i64)>"
        @test get(decls, "odd_sum", "") == "func.func private @odd_sum(!llvm.struct<packed (i64)>) -> i32"
        @test occursin("callee = @make_odd, sysv_ret = \"regs:int\"", ir)
        @test occursin("callee = @odd_sum, sysv_args = [\"regs:int\"]", ir)

        # What a marshal op takes and gives is a struct, in every thunk.
        sigs = _cm_marshal_sigs(ir)
        @test length(sigs) == 5      # name ×2, pair ×2, holder: the loop below ran
        for sig in sigs
            @test !occursin(r"^\(!llvm\.array<", sig)
            @test !occursin(r"-> !llvm\.array<", sig)
        end
        # No declaration takes or returns a bare byte array.
        @test !any(d -> occursin(r"[(>] ?!llvm\.array<\d+ x i8>\)?(?: ->|$)", d), values(decls))

        # The const-member records are marshalled as the structs they are.
        @test occursin("jlcs.marshal_ret %ret_packed { numMembers = 5 : i64 } : " *
                       "(!llvm.struct<packed (i32, i32, !llvm.ptr, i32, i32) >)", ir)
        @test occursin("memberTypes = [i32, i32, !llvm.ptr, i32, i32], " *
                       "juliaOffsets = [0 : i64, 4 : i64, 8 : i64, 16 : i64, 20 : i64]", ir)
        @test occursin("callee = @make_name, sysv_ret = \"mem:8\"", ir)
        @test occursin("callee = @pair_of, sysv_ret = \"regs:int\"", ir)
    end
end
