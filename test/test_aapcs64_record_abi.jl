# =============================================================================
# AAPCS64 record classes from DWARF facts (library-free, CI-safe)
#
# `FunctionGen.aapcs64_record_abi` classifies a by-value record for AArch64
# AAPCS64 (Linux) from its metadata, `aapcs64_call_fact` spells the verdict the
# thunk generator states on a `jlcs.ffe_call`/`jlcs.try_call` (`aapcs64_args`,
# `aapcs64_ret`) when it generates for an AAPCS64 host, and `sysv_trap_reason`
# with `abi = :aapcs64` names what such a host refuses. GeneratorC's Tier-3
# predicates take the same `abi`. This file pins the rules on synthetic
# metadata, one record per rule, with no toolchain; every expected answer here
# was read off `clang --target=aarch64-linux-gnu -S -emit-llvm` (clang 23), and
# test_aapcs64_abi_spec.jl (devtests) diffs the dialect's call shapes against
# that lowering over records built with real DWARF.
#
# The rules, in clang's words (AArch64ABIInfo::classifyArgumentType /
# classifyReturnType): an HFA of up to four leaves of one floating type goes in
# floating registers at any size; otherwise 16 bytes or less goes in general
# registers, `i128` when the record's unadjusted (argument) or adjusted
# (return) alignment is 16, a return of 8 bytes or less as `iN` of its size;
# over 16 bytes is a pointer to a copy, or an sret slot. Field alignment plays
# no part; the record's own `aligned` counts for a return only.
# =============================================================================

using Test
using RepliBuild

const _ARFG = RepliBuild.JLCSIRGenerator.FunctionGen
const _ARW = RepliBuild.Wrapper

_arm(name, ct, off, size; jt = "Any", kw...) =
    Dict{String,Any}("name" => name, "c_type" => ct, "julia_type" => jt, "offset" => off, "size" => size,
                     (String(k) => v for (k, v) in kw)...)
_arbit(name, ct, bits, at; size = 4) =
    Dict{String,Any}("name" => name, "c_type" => ct, "julia_type" => "Cint", "offset" => nothing,
                     "size" => size, "bit_size" => bits, "data_bit_offset" => at)
_arrec(size, members...; kind = "struct", kw...) =
    Dict{String,Any}("kind" => kind, "byte_size" => "0x" * string(size, base = 16),
                     "members" => Any[members...], (String(k) => v for (k, v) in kw)...)
_arfn(name, ret, params...) =
    Dict{String,Any}("name" => name, "mangled" => name,
                     "parameters" => Any[Dict{String,Any}("name" => "p$i", "c_type" => t) for (i, t) in enumerate(params)],
                     "return_type" => Dict{String,Any}("c_type" => ret))

const _AR = Dict{String,Any}(
    # homogeneous floating-point aggregates: one floating register per leaf, at any size
    "HFA2"   => _arrec(8, _arm("a", "float", "0x0", 4), _arm("b", "float", "0x4", 4)),
    "HFA4"   => _arrec(16, _arm("a", "float", "0x0", 4), _arm("b", "float[3]", "0x4", 12; jt = "NTuple{3, Cfloat}")),
    "HFAD4"  => _arrec(32, _arm("a", "double", "0x0", 8), _arm("b", "double", "0x8", 8),
                       _arm("c", "double", "0x10", 8), _arm("d", "double", "0x18", 8)),
    "D3"     => _arrec(24, _arm("a", "double", "0x0", 8), _arm("b", "double", "0x8", 8), _arm("c", "double", "0x10", 8)),
    "NHFA"   => _arrec(12, _arm("in", "HFA2", "0x0", 8; jt = "HFA2"), _arm("c", "float", "0x8", 4)),
    "AIN"    => _arrec(8, _arm("in", "F1[2]", "0x0", 8; jt = "NTuple{2, F1}")),
    "F1"     => _arrec(4, _arm("f", "float", "0x0", 4)),
    "CF"     => _arrec(8, _arm("z", "complex float", "0x0", 8; jt = "ComplexF32")),
    "CD"     => _arrec(16, _arm("z", "complex double", "0x0", 16; jt = "ComplexF64")),
    "UFF"    => _arrec(4, _arm("f", "float", "0x0", 4), _arm("g", "float", "0x0", 4); kind = "union"),
    "HALF3"  => _arrec(6, _arm("h", "_Float16[3]", "0x0", 6; jt = "NTuple{3, Float16}")),
    "LD"     => _arrec(16, _arm("x", "long double", "0x0", 16)),
    "LD2"    => _arrec(32, _arm("a", "long double", "0x0", 16), _arm("b", "long double", "0x10", 16)),
    "EB"     => _arrec(4, _arm("f", "float", "0x0", 4);
                       base_classes = Any[Dict{String,Any}("type" => "E0", "offset" => 0, "empty" => true)]),
    # what breaks an HFA: five leaves, padding, a non-float, a bitfield, a union with an int
    "HFA5"   => _arrec(20, _arm("v", "float[5]", "0x0", 20; jt = "NTuple{5, Cfloat}")),
    "HFAPAD" => _arrec(12, _arm("a", "float", "0x0", 4), _arm("b", "float", "0x4", 4), _arm("pad", "char", "0x8", 1)),
    "MIX"    => _arrec(16, _arm("d", "double", "0x0", 8), _arm("f", "float", "0x8", 4)),
    "FI"     => _arrec(8, _arm("f", "float", "0x0", 4), _arm("i", "int", "0x4", 4)),
    "BFF"    => _arrec(8, _arbit("a", "int", 3, 0), _arm("f", "float", "0x4", 4)),
    "UFI"    => _arrec(4, _arm("f", "float", "0x0", 4), _arm("i", "int", "0x0", 4); kind = "union"),
    # general registers by size; field alignment never matters
    "S8"     => _arrec(8, _arm("c", "char", "0x0", 1), _arm("i", "int", "0x4", 4)),
    "P5"     => _arrec(5, _arm("c", "char", "0x0", 1), _arm("i", "int", "0x1", 4)),
    "PCD"    => _arrec(9, _arm("c", "char", "0x0", 1), _arm("d", "double", "0x1", 8)),
    "S12"    => _arrec(12, _arm("a", "int", "0x0", 4), _arm("b", "int", "0x4", 4), _arm("c", "int", "0x8", 4)),
    "S9"     => _arrec(16, _arm("a", "long long", "0x0", 8), _arm("b", "char", "0x8", 1)),
    "C16"    => _arrec(16, _arm("a", "char[16]", "0x0", 16; jt = "NTuple{16, Cchar}")),
    # the even register pair: a member's alignment, a nested record's own, a 16-byte scalar
    "MA16"   => _arrec(16, _arm("a", "long long", "0x0", 8; alignment = 16)),
    "NA16"   => _arrec(16, _arm("a", "A16", "0x0", 16; jt = "A16")),
    "I128"   => _arrec(16, _arm("x", "__int128", "0x0", 16; jt = "Int128")),
    # the record's own `aligned`: a return reads it, an argument does not
    "A16"    => _arrec(16, _arm("a", "long long", "0x0", 8); alignment = 16),
    # over 16 bytes: a pointer to a copy, an sret slot, with the record's alignment
    "S17"    => _arrec(17, _arm("a", "char[17]", "0x0", 17; jt = "NTuple{17, Cchar}")),
    "CI128"  => _arrec(32, _arm("c", "char", "0x0", 1), _arm("x", "__int128", "0x10", 16; jt = "Int128")),
    "BIG16"  => _arrec(32, _arm("a", "long long", "0x0", 8), _arm("b", "long long", "0x8", 8),
                       _arm("c", "long long", "0x10", 8); alignment = 16),
    # nothing, and nothing with a size
    "E0"     => _arrec(0),
    "PADONLY" => _arrec(4),                                              # { int : 32; }
    # refused: a SIMD vector member (a homogeneous vector aggregate the thunk does not model)
    "V4"     => _arrec(16, _arm("v", "float[4]", "0x0", 16; jt = "NTuple{4, Cfloat}", vector_size = 16, alignment = 16)),
    # refused: a member whose kind cannot be read, unless the record is too big to be an HFA
    "MYST"   => _arrec(8, _arm("m", "mystery_t", "0x0", 8)),
    "BIGMYST" => _arrec(72, _arm("m", "mystery_t", "0x0", 8), _arm("a", "char[64]", "0x8", 64; jt = "NTuple{64, Cchar}")),
)

@testset "AAPCS64 record classes from DWARF facts" begin
    abi(k; ret) = _ARFG.aapcs64_record_abi(k, _AR; ret)
    fact(k; ret) = _ARFG.aapcs64_call_fact(abi(k; ret))
    both(k) = (fact(k; ret = false), fact(k; ret = true))

    @testset "the host's convention is one of three" begin
        @test RepliBuild.host_record_abi() in (:sysv, :win64, :aapcs64)
        @test RepliBuild.host_record_abi() ===
              (Sys.iswindows() ? :win64 : Sys.ARCH === :aarch64 ? :aapcs64 : :sysv)
        @test _ARW._c_record_abi() === RepliBuild.host_record_abi()
        @test RepliBuild.C_LONG_DOUBLE_MLIR == (Sys.ARCH === :aarch64 ? "f128" : "f80")
        @test RepliBuild.JLCSIRGenerator.TypeUtils.map_cpp_type("long double") == RepliBuild.C_LONG_DOUBLE_MLIR
    end

    @testset "an HFA goes in floating registers at any size" begin
        want = Dict("HFA2" => "hfa:f32:2", "HFA4" => "hfa:f32:4", "HFAD4" => "hfa:f64:4", "D3" => "hfa:f64:3",
                    "NHFA" => "hfa:f32:3", "AIN" => "hfa:f32:2", "CF" => "hfa:f32:2", "CD" => "hfa:f64:2",
                    "UFF" => "hfa:f32:1", "HALF3" => "hfa:f16:3", "LD" => "hfa:f128:1", "LD2" => "hfa:f128:2",
                    "EB" => "hfa:f32:1")
        checked = 0
        for k in sort!(collect(keys(want)))
            @test (k, both(k)) == (k, (want[k], want[k]))
            checked += 1
        end
        @test checked == length(want)
    end

    @testset "what breaks an HFA goes by size" begin
        @test both("HFA5") == ("mem:8", "mem:8")             # five leaves, 20 bytes
        @test both("HFAPAD") == ("gpr:2", "gpr:2")           # 2 × 4 ≠ 12: padding
        @test both("MIX") == ("gpr:2", "gpr:2")              # two widths
        @test both("FI") == ("gpr:1", "int:64")              # an int among the floats
        @test both("BFF") == ("gpr:1", "int:64")             # a bitfield
        @test both("UFI") == ("gpr:1", "int:32")             # a union with an int
    end

    @testset "general registers by size, field alignment unread" begin
        @test both("S8") == ("gpr:1", "int:64")
        @test both("P5") == ("gpr:1", "int:40")              # a packed 5-byte record: iN of its size on return
        @test both("PCD") == ("gpr:2", "gpr:2")              # packed { char; double }: no misalignment rule
        @test both("S12") == ("gpr:2", "gpr:2")
        @test both("S9") == ("gpr:2", "gpr:2")
        @test both("C16") == ("gpr:2", "gpr:2")
    end

    @testset "the even register pair reads the right alignment" begin
        @test both("MA16") == ("i128", "i128")               # a member's alignas(16)
        @test both("NA16") == ("i128", "i128")               # a nested record's own aligned(16)
        @test both("I128") == ("i128", "i128")               # a 16-byte scalar
        @test both("A16") == ("gpr:2", "i128")               # the record's own aligned(16): a return reads it, an argument does not
        @test _ARFG._sysv_record_align("A16", _AR) == 16
        @test _ARFG._sysv_record_align("A16", _AR; own = false) == 8
        @test _ARFG._sysv_record_align("NA16", _AR; own = false) == 16
    end

    @testset "over 16 bytes: a pointer or an sret slot, with the record's alignment" begin
        @test both("S17") == ("mem:8", "mem:8")
        @test both("CI128") == ("mem:16", "mem:16")
        @test both("BIG16") == ("mem:16", "mem:16")
        @test both("BIGMYST") == ("mem:8", "mem:8")          # 72 bytes: no HFA is that large, so the member does not matter
    end

    @testset "nothing, and nothing with a size" begin
        @test both("E0") == ("ignore", "ignore")
        @test abi("E0"; ret = false).kind === :ignore
        @test both("PADONLY") == ("gpr:1", "ignore")         # C++: one i64 in, nothing out
    end

    @testset "what an AAPCS64 host refuses" begin
        trap(k; ret) = _ARFG.sysv_trap_reason(k, _AR; ret, abi = :aapcs64)
        v = abi("V4"; ret = false)
        @test v.kind === :unknown
        @test occursin("SIMD vector", v.reason)
        @test_throws ArgumentError _ARFG.aapcs64_call_fact(v)
        @test _ARFG.aapcs64_statement("V4", _AR; ret = false) == ""
        @test occursin("AAPCS64 class cannot be read", trap("V4"; ret = true))
        @test occursin("mystery_t", trap("MYST"; ret = false))
        for k in ("HFA2", "S17", "A16", "LD", "BIGMYST", "E0")
            @test (k, trap(k; ret = false), trap(k; ret = true)) == (k, nothing, nothing)
        end
        # SysV's own refusals are not AAPCS64's: a `long double` return is an HFA here.
        @test occursin("st0", _ARFG.sysv_trap_reason("LD", _AR; ret = true, abi = :sysv))
        @test _ARFG.sysv_trap_reason("LD", _AR; ret = true, abi = :win64) === nothing
        @test_throws ArgumentError _ARFG.sysv_trap_reason("LD", _AR; ret = true, abi = :aapcs32)
    end

    @testset "the generator states the class on the call, under the convention asked" begin
        funcs = Any[_arfn("take_hfa", "double", "HFA2", "int"), _arfn("make_hfa", "HFA2"),
                    _arfn("take_a16", "double", "A16"), _arfn("make_a16", "A16"),
                    _arfn("take_p5", "double", "P5"), _arfn("make_p5", "P5"),
                    _arfn("take_big", "double", "S17"), _arfn("make_big", "S17"),
                    _arfn("take_e0", "double", "E0", "int"), _arfn("take_v4", "double", "V4"),
                    _arfn("scalars", "double", "int", "double")]
        ir = _ARFG.generate_function_thunks(funcs, _AR; abi = :aapcs64)
        @test occursin("callee = @take_hfa, aapcs64_args = [\"hfa:f32:2\", \"\"]", ir)
        @test occursin("callee = @make_hfa, aapcs64_ret = \"hfa:f32:2\"", ir)
        @test occursin("callee = @take_a16, aapcs64_args = [\"gpr:2\"]", ir)
        @test occursin("callee = @make_a16, aapcs64_ret = \"i128\"", ir)
        @test occursin("callee = @take_p5, aapcs64_args = [\"gpr:1\"]", ir)
        @test occursin("callee = @make_p5, aapcs64_ret = \"int:40\"", ir)
        @test occursin("callee = @take_big, aapcs64_args = [\"mem:8\"]", ir)
        @test occursin("callee = @make_big, aapcs64_ret = \"mem:8\"", ir)
        @test occursin("callee = @take_e0, aapcs64_args = [\"ignore\", \"\"]", ir)
        @test occursin("{ callee = @scalars }", ir)
        @test occursin(r"// no thunk for take_v4: takes `V4` by value, whose AAPCS64 class cannot be read", ir)
        @test !occursin("sysv_", ir)
        # The same functions for an x86-64 host: SysV statements, nothing AAPCS64.
        for other in (:sysv, :win64)
            irx = _ARFG.generate_function_thunks(funcs, _AR; abi = other)
            @test (other, occursin("callee = @take_hfa, sysv_args = [\"regs:sse\", \"\"]", irx)) == (other, true)
            @test (other, occursin("aapcs64", irx)) == (other, false)
        end
        @test_throws ArgumentError _ARFG.generate_function_thunks(funcs, _AR; abi = :aapcs32)
        # The default is the host's.
        @test _ARFG.generate_function_thunks(funcs, _AR) ==
              _ARFG.generate_function_thunks(funcs, _AR; abi = RepliBuild.host_record_abi())
    end

    @testset "a C++ empty record: one i64 in, nothing out" begin
        facts = Dict{String,Any}("Unit" => Dict{String,Any}("pass" => "value", "empty" => true, "byte_size" => "0x1"))
        cross(abi; ret) = _ARFG.by_value_crossing("Unit", Dict{String,Any}(), facts; abi, ret)
        @test cross(:aapcs64; ret = false) == (:regpad, "i64")
        @test cross(:aapcs64; ret = true) == (:ignore, nothing)
        @test cross(:sysv; ret = false) == (:ignore, nothing)
        @test cross(:sysv; ret = true) == (:ignore, nothing)
        @test cross(:win64; ret = false) == (:regpad, "i8")
        @test cross(:win64; ret = true) == (:regpad, "i8")
        # A GNU size-0 C struct is nothing everywhere.
        facts0 = Dict{String,Any}("Z" => Dict{String,Any}("pass" => nothing, "empty" => true, "byte_size" => "0x0"))
        for a in (:sysv, :win64, :aapcs64)
            @test (a, _ARFG.by_value_crossing("Z", Dict{String,Any}(), facts0; abi = a)) == (a, (:ignore, nothing))
        end
        c = _ARFG.thunk_crossings(_arfn("unit_in_out", "Unit", "Unit", "int"), Dict{String,Any}(), facts; abi = :aapcs64)
        @test c.params == [(:regpad, "i64"), (:direct, nothing)]
        @test c.ret == (:ignore, nothing)
        @test c.trap === nothing
    end

    @testset "a pass-by-reference return names its hidden pointer for x8" begin
        # DW_CC_pass_by_reference: the thunk owns an sret slot and passes it
        # first. AAPCS64 wants that pointer in x8, which only `llvm.sret`
        # reaches, so the statement names the slot and its bytes; x86-64 puts
        # it in the first argument register and states nothing.
        structs = Dict{String,Any}("Ticket" => _arrec(12, _arm("n", "int", "0x0", 4), _arm("a", "int", "0x4", 4),
                                                      _arm("b", "int", "0x8", 4)))
        ra = Dict{String,Any}("Ticket" => Dict{String,Any}("pass" => "reference", "byte_size" => "0xc"))
        f = _arfn("make_ticket", "Ticket", "int")
        @test _ARFG.aapcs64_sret_statement("Ticket", structs) == "sret:12"
        ir = _ARFG.generate_function_thunks(Any[f], structs; record_abi = ra, abi = :aapcs64)
        @test occursin("%sret_slot", ir)
        @test occursin("callee = @make_ticket, aapcs64_args = [\"sret:12\", \"\"]", ir)
        for other in (:sysv, :win64)
            irx = _ARFG.generate_function_thunks(Any[f], structs; record_abi = ra, abi = other)
            @test (other, occursin("%sret_slot", irx)) == (other, true)
            @test (other, occursin("sret:", irx)) == (other, false)
            @test (other, occursin("{ callee = @make_ticket }", irx)) == (other, true)
        end
    end

    @testset "GeneratorC's Tier-3 predicates under AAPCS64" begin
        @test _ARW._c_class_window(:sysv) == 16
        @test _ARW._c_class_window(:aapcs64) == 64
        @test _ARW._c_class_window(:win64) == 0
        # Registers at 16 bytes or less; above, an HFA or a hidden pointer the stand-in's fields settle.
        @test _ARW._c_return_in_memory(8, nothing; abi = :aapcs64) == false
        @test _ARW._c_return_in_memory(16, true; abi = :aapcs64) == false
        @test _ARW._c_return_in_memory(24, nothing; abi = :aapcs64) === nothing
        @test !_ARW._c_explicit_sret(24, nothing; abi = :aapcs64)
        @test !_ARW._c_explicit_sret(8, true; abi = :aapcs64)
        @test !_ARW._c_blob_param_misclassified(8, nothing; abi = :aapcs64)
        @test !_ARW._c_blob_param_misclassified(5, true; abi = :aapcs64)
        # A float-bearing blob may be an HFA at any size up to four long doubles.
        table = [(8, true, true), (16, true, true), (24, true, true), (64, true, true), (65, true, false),
                 (8, false, false), (24, false, false)]
        checked = 0
        for (n, risk, want) in table
            @test (n, risk, _ARW._c_register_class_misstated(n, risk; abi = :aapcs64)) == (n, risk, want)
            checked += 1
        end
        @test checked == length(table)
        # The facts settle it where they can: a float-bearing blob the facts
        # call no HFA travels by size, as the blob does.
        @test !_ARW._c_register_class_misstated(24, true; abi = :aapcs64, hfa = false)
        @test _ARW._c_register_class_misstated(24, true; abi = :aapcs64, hfa = true)
        @test _ARW._c_register_class_misstated(24, true; abi = :aapcs64, hfa = nothing)
        @test _ARW._c_register_class_misstated(8, true; abi = :sysv, hfa = false)      # SysV has no HFA to rule out
        # No slot and no copy to align.
        for (n, mem, al, exact) in ((32, true, 16, true), (24, true, 32, false), (8, false, 16, true))
            @test (n, _ARW._c_byval_param_shape(n, mem, al, exact; abi = :aapcs64)) == (n, :own)
        end
        # The even register pair Julia's copy can miss: 16 bytes or less, unadjusted 16, stand-in below 16.
        pair(n, al, sa; abi = :aapcs64) = _ARW._c_register_pair_misplaced(n, al, sa; abi)
        @test pair(16, 16, 8)
        @test pair(16, 16, 1)
        @test pair(16, 16, nothing)
        @test !pair(16, 16, 16)
        @test !pair(16, 8, 8)
        @test !pair(32, 16, 8)                 # a pointer to a copy
        @test !pair(16, 16, 8; abi = :sysv)
        @test !pair(16, 16, 8; abi = :win64)
        @test occursin("even-numbered register pair", _ARW._c_register_pair_trap_phrase("MA16", 16))
        # The unadjusted alignment GeneratorC reads for it.
        @test _ARW._c_record_align(_AR["A16"], _AR) == 16
        @test _ARW._c_record_align(_AR["A16"], _AR; own = false) == 8
        @test _ARW._c_record_align(_AR["MA16"], _AR; own = false) == 16
        @test _ARW._c_record_align(_AR["NA16"], _AR; own = false) == 16
        @test_throws ArgumentError _ARW._c_return_in_memory(8, nothing; abi = :aapcs32)
    end
end
