# =============================================================================
# Tier 2 thunk bodies for byte regions and `#pragma pack` records
# (library-free, CI-safe)
#
#   X2  A union or a bitfield record has no field-by-field body, so StructGen
#       emits it as `!llvm.array<N x i8>`. By value the thunk passed it as that
#       array; LLVM splits an array argument into one register per element, so
#       the callee read only the first byte, and on Win64 a size passed by
#       reference arrived as that byte read as an address. It now crosses as a
#       packed struct of the same bytes, which the dialect's classifiers place
#       as C does. On SysV a region of 16 bytes or less that may hold a float is
#       refused: C passes it in SSE registers its bytes cannot state.
#   X8  A `#pragma pack` record holding a pointer-aligned record: the member
#       was measured by its bytes (alignment 1) while emitted as an alias of an
#       8-aligned struct, so the body came out unpacked and LLVM moved the member
#       from offset 4 to 8. Measured as what is emitted, the body is packed.
#
# (abi_fuzz, 2026-10-18.) The live calls are in test_cpp_record_routing.jl.
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
)

@testset "Tier 2 thunk bodies: byte regions and #pragma pack (X2, X8)" begin
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
        # SysV: at 16 bytes or less a float makes the classes unknowable from bytes.
        for (key, sysv) in (("U4", false), ("U12", false), ("BF8", false), ("UF4", true), ("UF24", false))
            @test (key, :sysv, _TBFG._byte_region_unclassifiable(key, S; abi = :sysv)) == (key, :sysv, sysv)
            @test (key, :win64, _TBFG._byte_region_unclassifiable(key, S; abi = :win64)) == (key, :win64, false)
        end
        @test_throws ArgumentError _TBFG._byte_region_unclassifiable("U4", S; abi = :aapcs64)
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
        # The float union: refused on SysV, a call on Win64 (this host's answer).
        if Sys.iswindows()
            @test haskey(decls, "uf4_sum")
        else
            @test !haskey(decls, "uf4_sum")
            @test occursin(r"// no thunk for uf4_sum: .*SSE", ir)
        end
    end
end
