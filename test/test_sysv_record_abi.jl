# =============================================================================
# SysV record classes from DWARF facts (library-free, CI-safe)
#
# `FunctionGen.sysv_record_abi` classifies a by-value record for x86-64 SysV
# from its metadata (members at their offsets, bitfields by bit range, nested
# records, unions, the scalar kinds, DW_AT_alignment), `sysv_call_fact` spells
# the verdict the thunk generator states on a `jlcs.ffe_call`/`jlcs.try_call`
# (`sysv_args`, `sysv_ret`), and `sysv_trap_reason` names what a SysV host
# refuses. The dialect used to classify from StructGen's emitted body, which
# cannot carry these facts; test_sysv_abi_oracle.jl (devtests) diffs the
# result against LLVM's own classifier and test_sysv_live.jl calls the records.
# This file pins the rules on synthetic metadata, one record per rule, with no
# toolchain, and that the generator states them on the call.
# =============================================================================

using Test
using RepliBuild

const _SRFG = RepliBuild.JLCSIRGenerator.FunctionGen
const _SRSG = RepliBuild.JLCSIRGenerator.StructGen

_srm(name, ct, off, size; jt = "Any", kw...) =
    Dict{String,Any}("name" => name, "c_type" => ct, "julia_type" => jt, "offset" => off, "size" => size,
                     (String(k) => v for (k, v) in kw)...)
_srbit(name, ct, bits, at; size = 4) =
    Dict{String,Any}("name" => name, "c_type" => ct, "julia_type" => "Cint", "offset" => nothing,
                     "size" => size, "bit_size" => bits, "data_bit_offset" => at)
_srrec(size, members...; kind = "struct", kw...) =
    Dict{String,Any}("kind" => kind, "byte_size" => "0x" * string(size, base = 16),
                     "members" => Any[members...], (String(k) => v for (k, v) in kw)...)
_srfn(name, ret, params...) =
    Dict{String,Any}("name" => name, "mangled" => name,
                     "parameters" => Any[Dict{String,Any}("name" => "p$i", "c_type" => t) for (i, t) in enumerate(params)],
                     "return_type" => Dict{String,Any}("c_type" => ret))

const _SR = Dict{String,Any}(
    # an eightbyte holding a float and padding is SSE
    "c_f_d"    => _srrec(16, _srm("x", "float", "0x0", 4), _srm("y", "double", "0x8", 8)),
    "c_d_f"    => _srrec(16, _srm("d", "double", "0x0", 8), _srm("f", "float", "0x8", 4)),
    "c_p_f"    => _srrec(16, _srm("p", "void*", "0x0", 8), _srm("f", "float", "0x8", 4)),
    "c_ubf_f"  => _srrec(8, _srm("f", "float", "0x4", 4)),                 # { int : 32; float f; }
    "c_al16_f" => _srrec(16, _srm("f", "float", "0x0", 4; alignment = 16)),
    "c_cf"     => _srrec(8, _srm("z", "complex float", "0x0", 8; jt = "ComplexF32")),
    "c_cd"     => _srrec(16, _srm("z", "complex double", "0x0", 16; jt = "ComplexF64")),
    "c_wrap_df" => _srrec(16, _srm("in", "c_d_f", "0x0", 16; jt = "c_d_f")),
    # a low eightbyte of padding is passed nowhere
    "c_ubf_d"  => _srrec(16, _srm("d", "double", "0x8", 8)),               # { long long : 64; double d; }
    # controls
    "c_f_i"    => _srrec(8, _srm("x", "float", "0x0", 4), _srm("i", "int", "0x4", 4)),
    "c_c_d"    => _srrec(16, _srm("c", "char", "0x0", 1), _srm("d", "double", "0x8", 8)),
    "c_fa3"    => _srrec(12, _srm("v", "float[3]", "0x0", 12; jt = "NTuple{3, Cfloat}")),
    "c_d3"     => _srrec(24, _srm("a", "double", "0x0", 8), _srm("b", "double", "0x8", 8), _srm("c", "double", "0x10", 8)),
    # misaligned fields make a record MEMORY, through a nested record or a union
    "c_s_f"    => _srrec(8, _srm("s", "short", "0x0", 2), _srm("f", "float", "0x4", 4)),
    "c_pk_sf"  => _srrec(9, _srm("c", "char", "0x0", 1), _srm("in", "c_s_f", "0x1", 8; jt = "c_s_f")),
    "c_pk_ci"  => _srrec(5, _srm("c", "char", "0x0", 1), _srm("i", "int", "0x1", 4)),
    "c_pk_ii"  => _srrec(8, _srm("a", "int", "0x0", 4), _srm("b", "int", "0x4", 4)),     # packed, but every field aligned
    "c_u_pk"   => _srrec(8, _srm("p", "c_pk_ci", "0x0", 5; jt = "c_pk_ci"), _srm("i", "int", "0x0", 4); kind = "union"),
    "c_al8_c"  => _srrec(8, _srm("x", "char", "0x0", 1); alignment = 8),
    "c_pk_alnest" => _srrec(9, _srm("c", "char", "0x0", 1), _srm("s", "c_al8_c", "0x1", 8; jt = "c_al8_c"); alignment = 1),
    # bitfields classify by their bits, with no alignment check
    "c_pp1_bf" => _srrec(5, _srm("c", "char", "0x0", 1), _srbit("b", "int", 31, 8); alignment = 1),
    "c_bf2_f"  => _srrec(8, _srbit("a", "int", 3, 0), _srbit("b", "int", 5, 3), _srm("f", "float", "0x4", 4)),
    "c_bf_f"   => _srrec(8, _srbit("a", "int", 3, 0), _srm("f", "float", "0x4", 4)),
    # unions merge every member
    "c_u_fi"   => _srrec(4, _srm("f", "float", "0x0", 4), _srm("i", "int", "0x0", 4); kind = "union"),
    "c_u_ff"   => _srrec(4, _srm("f", "float", "0x0", 4), _srm("g", "float", "0x0", 4); kind = "union"),
    "c_u_ld_i" => _srrec(16, _srm("x", "long double", "0x0", 16), _srm("i", "int", "0x0", 4); kind = "union"),
    # x87, __int128, over-alignment
    "c_ld"     => _srrec(16, _srm("x", "long double", "0x0", 16)),
    "c_i128"   => _srrec(16, _srm("x", "__int128", "0x0", 16; jt = "Int128")),
    "c_i128_l" => _srrec(32, _srm("x", "__int128", "0x0", 16; jt = "Int128"), _srm("y", "long long", "0x10", 8)),
    "c_al16_big" => _srrec(32, _srm("a", "long long", "0x0", 8), _srm("b", "long long", "0x8", 8),
                           _srm("c", "long long", "0x10", 8); alignment = 16),
    "c_pk_c_ld" => _srrec(17, _srm("c", "char", "0x0", 1), _srm("x", "long double", "0x1", 16)),
    # padding only, and empty
    "c_pad_only" => _srrec(4),                                              # { int : 32; }
    "c_empty"  => _srrec(0),
    # enums, typedefs and what cannot be read
    "__enum__Kind" => Dict{String,Any}("kind" => "enum", "byte_size" => "0x4", "julia_type" => "Cint",
                                       "underlying_type" => "int", "enumerators" => Any[]),
    "c_e_f"    => _srrec(8, _srm("k", "Kind", "0x0", 4; jt = "Kind"), _srm("f", "float", "0x4", 4)),
    "c_real2"  => _srrec(8, _srm("x", "real", "0x0", 4; jt = "Cfloat"), _srm("y", "real", "0x4", 4; jt = "Cfloat")),
    "c_mystery" => _srrec(8, _srm("m", "mystery_t", "0x0", 8)),
    "c_big_mystery" => _srrec(24, _srm("m", "mystery_t", "0x0", 8), _srm("a", "long long", "0x8", 8),
                              _srm("b", "long long", "0x10", 8)),
    "c_flex"   => _srrec(8, _srm("n", "int", "0x0", 4), _srm("tail", "int[]", "0x4", 0)),
)

@testset "SysV record classes from DWARF facts" begin
    abi(k) = _SRFG.sysv_record_abi(k, _SR)
    fact(k) = _SRFG.sysv_call_fact(abi(k))

    @testset "padding never classifies" begin
        for k in ("c_f_d", "c_d_f", "c_cd", "c_wrap_df")
            @test (k, fact(k)) == (k, "regs:sse,sse")
        end
        @test fact("c_p_f") == "regs:int,sse"
        @test fact("c_ubf_f") == "regs:sse"
        @test fact("c_al16_f") == "regs:sse"          # the high eightbyte is padding: dropped
        @test fact("c_cf") == "regs:sse"
        @test fact("c_ubf_d") == "regs:none,sse"      # the low eightbyte is padding: passed nowhere
        @test fact("c_f_i") == "regs:int"
        @test fact("c_c_d") == "regs:int,sse"
        @test fact("c_fa3") == "regs:sse,sse"
        @test fact("c_d3") == "mem:8"
    end

    @testset "a misaligned field anywhere is MEMORY" begin
        @test fact("c_pk_ci") == "mem:8"
        @test fact("c_pk_sf") == "mem:8"              # the float inside the nested record sits at 5
        @test fact("c_u_pk") == "mem:8"               # a union member's member
        @test fact("c_pk_alnest") == "mem:8"          # the nested record's own position, not its one char
        @test fact("c_al8_c") == "regs:int"
        @test fact("c_pk_ii") == "regs:int"           # packed, yet every field is where it would be anyway
        @test abi("c_pk_ci").align == 1               # no DW_AT_alignment: lowered until the layout fits
        @test abi("c_pk_ii").align == 4
    end

    @testset "bitfields classify by bit range" begin
        @test fact("c_pp1_bf") == "regs:int"
        @test fact("c_bf2_f") == "regs:int"           # merged with the float: INTEGER
        @test fact("c_bf_f") == "regs:int"
    end

    @testset "unions merge every member" begin
        @test fact("c_u_fi") == "regs:int"
        @test fact("c_u_ff") == "regs:sse"
        @test fact("c_u_ld_i") == "mem:16"            # X87UP under INTEGER: memory
    end

    @testset "long double, __int128, 16-aligned records" begin
        @test abi("c_ld").kind === :x87
        @test fact("c_ld") == "mem:16"                # as an argument
        @test fact("c_i128") == "regs:int,int"
        @test fact("c_i128_l") == "mem:16"
        @test fact("c_al16_big") == "mem:16"
        @test abi("c_pk_c_ld").align == 1             # 16 does not fit 17 bytes: clang's packed alignment
        @test fact("c_pk_c_ld") == "mem:8"
    end

    @testset "padding only, empty, enums, typedefs" begin
        @test abi("c_pad_only").kind === :ignore
        @test fact("c_pad_only") == "ignore"
        @test abi("c_empty").kind === :ignore
        @test fact("c_e_f") == "regs:int"
        @test fact("c_real2") == "regs:sse"           # a typedef'd float, read through julia_type
    end

    @testset "what cannot be read" begin
        u = abi("c_mystery")
        @test u.kind === :unknown
        @test occursin("mystery_t", u.reason)
        @test_throws ArgumentError _SRFG.sysv_call_fact(u)
        @test fact("c_big_mystery") == "mem:8"        # over 16 bytes the member does not matter
        @test fact("c_flex") == "mem:8"               # a flexible array member
    end

    @testset "what a SysV host refuses, and Win64 does not" begin
        trap(k; ret) = _SRFG.sysv_trap_reason(k, _SR; ret = ret, abi = :sysv)
        @test trap("c_ld"; ret = false) === nothing
        @test occursin("st0", trap("c_ld"; ret = true))
        @test occursin("mystery_t", trap("c_mystery"; ret = false))
        @test trap("c_u_fi"; ret = true) === nothing
        @test trap("c_u_ld_i"; ret = true) === nothing
        @test trap("c_big_mystery"; ret = true) === nothing
        for k in ("c_ld", "c_mystery")
            @test (k, _SRFG.sysv_trap_reason(k, _SR; ret = true, abi = :win64)) == (k, nothing)
        end
        @test_throws ArgumentError _SRFG.sysv_trap_reason("c_ld", _SR; ret = true, abi = :aapcs64)
    end

    @testset "the generator states the class on the call" begin
        funcs = Any[_srfn("take_fd", "double", "c_f_d", "int"), _srfn("make_fd", "c_f_d"),
                    _srfn("take_ubfd", "double", "c_ubf_d"), _srfn("take_pad", "double", "c_pad_only", "int"),
                    _srfn("make_pad", "c_pad_only"), _srfn("take_ld", "double", "c_ld"),
                    _srfn("make_ld", "c_ld"), _srfn("take_mystery", "double", "c_mystery"),
                    _srfn("scalars", "double", "int", "double"), _srfn("take_big", "double", "c_al16_big")]
        ir = _SRFG.generate_function_thunks(funcs, _SR)
        @test occursin("callee = @take_fd, sysv_args = [\"regs:sse,sse\", \"\"]", ir)
        @test occursin("callee = @make_fd, sysv_ret = \"regs:sse,sse\"", ir)
        @test occursin("callee = @take_ubfd, sysv_args = [\"regs:none,sse\"]", ir)
        @test occursin("callee = @take_pad, sysv_args = [\"ignore\", \"\"]", ir)
        @test occursin("callee = @make_pad, sysv_ret = \"ignore\"", ir)
        @test occursin("callee = @take_ld, sysv_args = [\"mem:16\"]", ir)
        @test occursin("callee = @take_big, sysv_args = [\"mem:16\"]", ir)
        # A call with no record by value states nothing, so its IR is unchanged.
        @test occursin("{ callee = @scalars }", ir)
        # A record whose class cannot be read states nothing either: on SysV it
        # never reaches a call (refused below), on Win64 the statement is not
        # read. Throwing instead took every thunk of a Win64 library with it.
        @test _SRFG.sysv_statement("c_mystery", _SR) == ""
        @test _SRFG.sysv_statement("c_f_d", _SR) == "regs:sse,sse"
        # The refusals, with the convention's own reason, apart from the layout ones.
        c = _SRFG.thunk_crossings(_srfn("make_ld", "c_ld"), _SR, Dict{String,Any}())
        if Sys.iswindows()
            @test c.trap === nothing
            @test occursin("{ callee = @take_mystery }", ir)
            @test occursin("callee = @make_ld, sysv_ret = \"mem:16\"", ir)
        else
            @test occursin("st0", c.abi_trap)
            @test c.layout_trap === nothing
            @test occursin(r"// no thunk for make_ld: returns `c_ld` by value, which SysV returns on the x87 stack", ir)
            @test occursin(r"// no thunk for take_mystery: takes `c_mystery` by value, whose SysV class cannot be read", ir)
        end
    end

    @testset "StructGen emits long double, _Complex and __int128 at their size" begin
        @test _SRSG.get_struct_definition_string("c_ld", _SR["c_ld"], _SR) ==
              "!jlcs.c_struct<\"c_ld\", [f80], [[0 : i64]], packed = true>"
        @test _SRSG.get_llvm_equivalent_type_string("c_ld", _SR["c_ld"], _SR) == "!llvm.struct<packed (f80) >"
        @test _SRSG.get_struct_definition_string("c_cd", _SR["c_cd"], _SR) ==
              "!jlcs.c_struct<\"c_cd\", [!llvm.struct<(f64, f64)>], [[0 : i64]], packed = true>"
        @test _SRSG.get_struct_definition_string("c_i128_l", _SR["c_i128_l"], _SR) ==
              "!llvm.struct<\"c_i128_l\", (i128, i64, !llvm.array<8 x i8>)>"
        # Measured as emitted: 16 bytes each, 16-aligned.
        @test _SRSG._mlir_layout("!llvm.struct<packed (f80) >", _SR) == (16, 1)
        @test _SRSG._mlir_layout("!llvm.struct<(i128, i64, !llvm.array<8 x i8>)>", _SR) == (32, 16)
        @test _SRSG._mlir_layout("!llvm.struct<(f32, f32)>", _SR) == (8, 4)
    end
end
