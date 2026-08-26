#!/usr/bin/env julia
# test/test_win64_abi.jl — pins the Win64 (Microsoft x64) struct decision table
# that classifyWin64Struct in src/mlir/impl/JLCSPasses.cpp implements.
#
# This is a SPECIFICATION test, not a behavioural one, and the distinction
# matters. test_struct_abi.jl verifies the SysV path by calling a real
# clang++-compiled callee, because a self-JIT'd callee shares the JIT's own
# convention and cannot catch a mismatch. That trick is unavailable here: a
# Win64 callee cannot be loaded or run on Linux. What CAN be done from Linux is
# ask clang to lower the same signatures for the Windows target and compare its
# answer to the table we encoded — clang targets x86_64-w64-windows-gnu for IR
# generation without needing mingw headers, a linker, or a Windows host.
#
# So: this catches an encoded rule that disagrees with clang. It does NOT prove
# the lowering runs correctly on Windows. Only a Windows host does that, and
# until one exists the Win64 path stays unproven in the way that matters.
#
# The rules being pinned (all diverge from SysV, three of them silently):
#   * size is the only criterion — 1/2/4/8 bytes in a register, everything else
#     indirect, INCLUDING the 9..16-byte band SysV splits across two registers
#   * aggregates never reach XMM: {float,float} is i64 here, XMM0 under SysV
#   * coercion is iN of the struct's own size, not always i64
#   * indirect arguments take NO byval attribute (clang: ByVal=false)

using Test

const CLANG  = Sys.which("clang")
const TRIPLE = "x86_64-w64-windows-gnu"

# Probe: a clang without the Windows target registered cannot act as oracle.
const CAN_TARGET_WIN = CLANG === nothing ? false : try
    mktempdir() do d
        p = joinpath(d, "probe.c")
        write(p, "int f(void){return 0;}\n")
        success(`$CLANG --target=$TRIPLE -S -emit-llvm -o /dev/null $p`)
    end
catch
    false
end

# The unavailable-oracle path is a SKIP, never an `exit`. This file is included
# by devtests.jl, and `exit(0)` inside an include ends the whole suite with a
# success status — every testset after this one silently never runs, and the run
# still looks green. Standalone, the two are indistinguishable; in a suite they
# are opposites.
const ORACLE_OK = CAN_TARGET_WIN

# ── Fixture ───────────────────────────────────────────────────────────────────
# One arg-taking and one value-returning declaration per shape. Bodies force the
# declarations to be emitted with their lowered signatures.

const SHAPES = [
    # name      C definition                                    size
    ("S1",      "struct { char a; }",                            1),
    ("S2",      "struct { short a; }",                           2),
    ("S3",      "struct { char a,b,c; }",                        3),
    ("S4",      "struct { int a; }",                             4),
    ("S4f",     "struct { float a; }",                           4),
    ("S5",      "struct { char a,b,c,d,e; }",                    5),
    ("S7",      "struct { char a[7]; }",                         7),
    ("S8d",     "struct { double a; }",                          8),
    ("S8ff",    "struct { float a,b; }",                         8),
    ("S8ii",    "struct { int a,b; }",                           8),
    ("S8p",     "struct { void *p; }",                           8),
    ("S12",     "struct { int a,b,c; }",                        12),
    ("S16",     "struct { long long a,b; }",                    16),
    ("S16dd",   "struct { double a,b; }",                       16),
    ("S72",     "struct { char a[72]; }",                       72),
    ("P5",      "struct __attribute__((packed)) { char a; int b; }", 5),
]

# What classifyWin64Struct must decide: `nothing` → MEMORY class (indirect),
# an integer width in bits → register class coerced to that iN.
win64_expected(size::Int) = size in (1, 2, 4, 8) ? size * 8 : nothing

function clang_win64_lowering()
    src = IOBuffer()
    for (name, cdef, _) in SHAPES
        println(src, "typedef $cdef $name;")
    end
    for (name, _, _) in SHAPES
        println(src, "void arg_$name($name v); $name ret_$name(void);")
        println(src, "void use_$name($name v){ arg_$name(v); }")
        println(src, "$name mk_$name(void){ return ret_$name(); }")
    end

    mktempdir() do d
        cfile = joinpath(d, "abi.c")
        write(cfile, String(take!(src)))
        ir = read(`$CLANG --target=$TRIPLE -O0 -S -emit-llvm -o - $cfile`, String)

        arg_sig  = Dict{String,String}()
        ret_sig  = Dict{String,String}()
        arg_line = Dict{String,String}()
        for line in eachline(IOBuffer(ir))
            startswith(line, "declare") || continue
            m = match(r"@(arg|ret)_(\w+)\(([^)]*)\)", line)
            m === nothing && continue
            kind, name, params = m.captures
            if kind == "arg"
                arg_sig[name] = strip(params)
                arg_line[name] = line
            else
                r = match(r"declare\s+(?:dso_local\s+)?(\S+)\s+@ret_", line)
                ret_sig[name] = r === nothing ? "?" : r.captures[1]
                # A void return with an sret parameter is the indirect form.
                if ret_sig[name] == "void" && occursin("sret(", params)
                    ret_sig[name] = "sret"
                end
            end
        end
        return arg_sig, ret_sig, arg_line
    end
end

const ARG_SIG, RET_SIG, ARG_LINE =
    ORACLE_OK ? clang_win64_lowering() :
                (Dict{String,String}(), Dict{String,String}(), Dict{String,String}())

@testset "Win64 struct ABI decision table" begin
    if !ORACLE_OK
        @info string("skipping Win64 ABI table test — ",
                     CLANG === nothing ? "clang not found" :
                                         "clang cannot target $TRIPLE")
        @test_skip false
    else
    @test !isempty(ARG_SIG)

    @testset "$name ($size B)" for (name, _, size) in SHAPES
        expect = win64_expected(size)
        argp = get(ARG_SIG, name, missing)
        retp = get(RET_SIG, name, missing)
        @test argp !== missing
        @test retp !== missing
        argp === missing && continue

        if expect === nothing
            # MEMORY class: pointer argument, sret return.
            @test occursin("ptr", argp)
            @test retp == "sret"
            # And crucially NO byval — the alloca+store in buildCallShape IS
            # the caller-allocated temporary; byval would request a second copy.
            @test !occursin("byval", ARG_LINE[name])
        else
            # Register class: coerced to iN of the struct's own size, both ways.
            @test occursin(Regex("\\bi$expect\\b"), argp)
            @test retp == "i$expect"
            # Never a float/double register, however float-shaped the struct is.
            @test !occursin("float", argp)
            @test !occursin("double", argp)
        end
    end

    # The divergences from SysV, called out explicitly so a regression names
    # itself rather than showing up as one row of a table.
    @testset "divergences from SysV" begin
        @test RET_SIG["S8ff"] == "i64"      # SysV: XMM0 (all-float eightbyte)
        @test RET_SIG["S8d"]  == "i64"      # SysV: XMM0
        @test RET_SIG["S4f"]  == "i32"      # SysV: XMM0
        @test RET_SIG["S16"]  == "sret"     # SysV: register class, RAX:RDX
        @test RET_SIG["S16dd"] == "sret"    # SysV: register class, XMM0:XMM1
        @test RET_SIG["P5"]   == "sret"     # size 5 — not a power of two
    end
    end  # ORACLE_OK
end
