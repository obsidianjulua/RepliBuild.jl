# =============================================================================
# Tier 2 thunk bodies for byte regions and `#pragma pack` records
# (library-free, CI-safe)
#
#   X2  A union or a bitfield record has no field-by-field body, so StructGen
#       emits it as `!llvm.array<N x i8>`. By value the thunk passed it as that
#       array; LLVM splits an array argument into one register per element, so
#       the callee read only the first byte, and on Win64 a size passed by
#       reference arrived as that byte read as an address. It now crosses as a
#       packed struct of the same bytes. Its class comes from the record's
#       DWARF facts, not the bytes (`sysv_record_abi`, stated on the call as
#       `sysv_args`), so a float union of 16 bytes or less, which SysV used to
#       refuse as unclassifiable, is INTEGER like C's and gets a thunk.
#   X8  A `#pragma pack` record holding a pointer-aligned record: the member
#       was measured by its bytes (alignment 1) while emitted as an alias of an
#       8-aligned struct, so the body came out unpacked and LLVM moved the member
#       from offset 4 to 8. Measured as what is emitted, the body is packed.
#   X9  A tiled record (member sizes summing to its size) holding a bitfield:
#       the body put the bitfield's unit after the members before it, but the
#       marshal, finding no DWARF byte offset on a bitfield, read the whole
#       record at aligned guesses. A `pack(1) {unsigned; unsigned long long : 7}`
#       had its unit read from 8 instead of 4. And a matching sum is not a tiling:
#       Itanium's `{short; int : 4; short}` is 8 bytes with the second short at 4.
#
# (abi_fuzz, 2026-10-18 and 2026-10-21.) The live calls are in
# test_cpp_record_routing.jl.
# =============================================================================

using Test
using RepliBuild

const _TBFG = RepliBuild.JLCSIRGenerator.FunctionGen
const _TBSG = RepliBuild.JLCSIRGenerator.StructGen

_tbm(name, ct, off, size) = Dict{String,Any}("name" => name, "c_type" => ct, "julia_type" => "Any",
                                             "offset" => off, "size" => size)
_tbbit(name, ct, bits, at) = Dict{String,Any}("name" => name, "c_type" => ct, "julia_type" => "Any",
                                              "offset" => nothing, "size" => 4,
                                              "bit_size" => bits, "data_bit_offset" => at)
_tbbitw(name, ct, bits, at, size) = Dict{String,Any}("name" => name, "c_type" => ct, "julia_type" => "Any",
                                                    "offset" => nothing, "size" => size,
                                                    "bit_size" => bits, "data_bit_offset" => at)
_tbrec(size, members...; kind = "struct") =
    Dict{String,Any}("kind" => kind, "byte_size" => "0x" * string(size, base = 16), "members" => Any[members...])
_tbfn(name, ret, params...) =
    Dict{String,Any}("name" => name, "mangled" => name,
                     "parameters" => Any[Dict{String,Any}("name" => "p$i", "c_type" => t) for (i, t) in enumerate(params)],
                     "return_type" => Dict{String,Any}("c_type" => ret))

const _TB_STRUCTS = Dict{String,Any}(
    # X8: `PR` is 8-aligned (a pointer two levels down); `PK28` packs it at 4.
    "PIn"   => _tbrec(16, _tbm("a", "int", "0x0", 4), _tbm("p", "void*", "0x8", 8)),
    "PR"    => _tbrec(24, _tbm("m1", "int", "0x0", 4), _tbm("in", "PIn", "0x8", 16)),
    "PK28"  => _tbrec(28, _tbm("c", "char", "0x0", 1), _tbm("r", "PR", "0x4", 24)),
    "NAT32" => _tbrec(32, _tbm("c", "char", "0x0", 1), _tbm("r", "PR", "0x8", 24)),
    # X2: byte regions.
    "U4"   => _tbrec(4, _tbm("i", "int", "0x0", 4), _tbm("b", "unsigned char[4]", "0x0", 4); kind = "union"),
    "U12"  => _tbrec(12, _tbm("i", "int[3]", "0x0", 12), _tbm("c", "char", "0x0", 1); kind = "union"),
    "BF8"  => _tbrec(8, _tbbit("a", "unsigned int", 3, 0), _tbbit("b", "unsigned int", 7, 3),
                        _tbm("n", "int", "0x4", 4)),
    "UF4"  => _tbrec(4, _tbm("f", "float", "0x0", 4), _tbm("i", "int", "0x0", 4); kind = "union"),
    "UF24" => _tbrec(24, _tbm("d", "double[3]", "0x0", 24), _tbm("i", "int", "0x0", 4); kind = "union"),
    # X9: tiled records holding a bitfield, as MinGW lays them out (abi_fuzz units 12, 31, 39).
    "PB12" => _tbrec(12, _tbm("m1", "unsigned int", "0x0", 4), _tbbitw("m2", "unsigned long long", 7, 32, 8)),
    "PB28" => _tbrec(28, _tbm("m1", "void*", "0x0", 8), _tbm("m2", "int", "0x8", 4),
                         _tbbitw("m3", "long long", 7, 96, 8), _tbbitw("m4", "unsigned long long", 60, 160, 8)),
    "U8P"  => _tbrec(8, _tbm("p", "void*", "0x0", 8), _tbm("s", "short", "0x0", 2); kind = "union"),
    "PB13" => _tbrec(13, _tbm("m1", "unsigned char[4]", "0x0", 4), _tbbitw("m2", "bool", 1, 32, 1),
                         _tbm("m3", "U8P", "0x5", 8)),
    # Sums to its size without tiling: Itanium puts the second short at 4.
    "IT8"  => _tbrec(8, _tbm("s", "short", "0x0", 2), _tbbitw("a", "int", 4, 16, 4), _tbm("t", "short", "0x4", 2)),
    # A bitfield whose bits lie outside the unit the tiling gives it.
    "OUT8" => _tbrec(8, _tbm("m1", "int", "0x0", 4), _tbbitw("m2", "int", 4, 0, 4)),
)

@testset "Tier 2 thunk bodies: byte regions, #pragma pack, tiled bitfield records (X2, X8, X9)" begin
    S = _TB_STRUCTS

    @testset "X8: a member is measured as what is emitted" begin
        @test _TBSG.get_struct_definition_string("PR", S["PR"], S) ==
              "!llvm.struct<\"PR\", (i32, !llvm.array<4 x i8>, !Struct_PIn)>"
        # At offset 4 the 8-aligned member forces a packed body...
        @test _TBSG.get_struct_definition_string("PK28", S["PK28"], S) ==
              "!llvm.struct<\"PK28\", packed (i8, !llvm.array<3 x i8>, !Struct_PR)>"
        # ...and at its natural offset it does not.
        @test _TBSG.get_struct_definition_string("NAT32", S["NAT32"], S) ==
              "!llvm.struct<\"NAT32\", (i8, !llvm.array<7 x i8>, !Struct_PR)>"
    end

    @testset "X2: which records are byte regions" begin
        @test _TBFG._byte_region_size("U4", S) == 4
        @test _TBFG._byte_region_size("U12", S) == 12
        @test _TBFG._byte_region_size("BF8", S) == 8
        @test _TBFG._byte_region_size("PR", S) === nothing      # a real body
        # The bytes say nothing about the class; the record's facts do. A
        # float union of 16 bytes or less, once refused on SysV, is INTEGER,
        # as C merges its members. Neither host refuses any of these.
        want = Dict("U4" => "regs:int", "U12" => "regs:int,int", "BF8" => "regs:int",
                    "UF4" => "regs:int", "UF24" => "mem:8")
        for key in sort!(collect(keys(want)))
            @test (key, _TBFG.sysv_call_fact(_TBFG.sysv_record_abi(key, S))) == (key, want[key])
            @test (key, :sysv, _TBFG.sysv_trap_reason(key, S; ret = false, abi = :sysv)) == (key, :sysv, nothing)
            @test (key, :win64, _TBFG.sysv_trap_reason(key, S; ret = true, abi = :win64)) == (key, :win64, nothing)
        end
        @test_throws ArgumentError _TBFG.sysv_trap_reason("U4", S; ret = false, abi = :aapcs64)
    end

    @testset "X2: a byte region crosses as a packed struct of its bytes" begin
        funcs = Any[_tbfn("u4_sum", "int", "U4"), _tbfn("u12_sum", "int", "U12"),
                    _tbfn("bf8_sum", "int", "BF8"), _tbfn("make_u12", "U12"),
                    _tbfn("uf4_sum", "int", "UF4"), _tbfn("pk28_sum", "int", "PK28")]
        ir = _TBFG.generate_function_thunks(funcs, S)
        decls = Dict(m.captures[1] => m.match for m in eachmatch(r"func\.func private @(\w+)\(.*", ir))
        @test get(decls, "u4_sum", "") == "func.func private @u4_sum(!llvm.struct<packed (i8, i8, i8, i8)>) -> i32"
        @test get(decls, "u12_sum", "") == "func.func private @u12_sum(!llvm.struct<packed (i64, i8, i8, i8, i8)>) -> i32"
        @test get(decls, "bf8_sum", "") == "func.func private @bf8_sum(!llvm.struct<packed (i64)>) -> i32"
        @test get(decls, "make_u12", "") == "func.func private @make_u12() -> !llvm.struct<packed (i64, i8, i8, i8, i8)>"
        @test startswith(get(decls, "pk28_sum", ""),
                         "func.func private @pk28_sum(!llvm.struct<\"PK28\", packed (i8, !llvm.array<3 x i8>, ")
        # No declaration takes or returns a bare byte array any more.
        @test !any(d -> occursin("!llvm.array<", replace(d, r"!llvm\.struct<\"[^\"]*\",[^>]*>" => "")), values(decls))
        # The float union gets a thunk on both hosts, and the call states its
        # class, which the bytes alone would have got wrong on SysV.
        @test haskey(decls, "uf4_sum")
        @test !occursin("// no thunk for", ir)
        @test occursin(r"callee = @uf4_sum, sysv_args = \[\"regs:int\"\]", ir)
        @test occursin(r"callee = @make_u12, sysv_ret = \"regs:int,int\"", ir)
        @test occursin(r"callee = @pk28_sum, sysv_args = \[\"mem:8\"\]", ir)
    end

    @testset "X9: a tiled record's bitfield is read where the body puts it" begin
        @test all(k -> _TBSG.is_struct_packed(S[k]), ("PB12", "PB28", "PB13"))
        @test _TBFG._dwarf_member_offsets(S["PB12"]) == [0, 4]            # the aligned guess said 8
        @test _TBFG._dwarf_member_offsets(S["PB28"]) == [0, 8, 12, 20]    # it said 16, 24
        @test _TBFG._dwarf_member_offsets(S["PB13"]) == [0, 4, 5]         # the union after a bool : 1
        ir = _TBFG.generate_function_thunks(Any[_tbfn("pb12_sum", "int", "PB12"), _tbfn("pb28_sum", "int", "PB28"),
                                                _tbfn("pb13_sum", "int", "PB13")], S)
        offs = [m.captures[1] for m in eachmatch(r"jlcs\.marshal_arg [^{]*\{[^}]*juliaOffsets = \[([^\]]*)\]", ir)]
        @test offs == ["0 : i64, 4 : i64", "0 : i64, 8 : i64, 12 : i64, 20 : i64", "0 : i64, 4 : i64, 5 : i64"]
    end

    @testset "X9: a sum that matches is not a tiling" begin
        @test !_TBSG.is_struct_packed(S["IT8"])
        @test !_TBSG.is_struct_packed(S["OUT8"])
        # Not tiled, it crosses as its bytes instead of a marshalled body.
        @test _TBSG.get_struct_definition_string("IT8", S["IT8"], S) == "!llvm.array<8 x i8>"
        ir = _TBFG.generate_function_thunks(Any[_tbfn("it8_sum", "int", "IT8")], S)
        @test occursin("func.func private @it8_sum(!llvm.struct<packed (i64)>) -> i32", ir)
        @test !occursin("jlcs.marshal_arg", ir)
    end
end
