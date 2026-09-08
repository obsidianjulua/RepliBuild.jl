#!/usr/bin/env julia
# test/test_jit_unwind.jl — COFF SEH / JIT teardown must not poison unwind.
#
# RuntimeDyld used to feed .pdata to libunwind's __register_frame. Destroying
# that engine then __deregister_frame'd SEH bytes as DWARF, so the *next*
# throw through a live JIT frame died with "libunwind: pc not in table".
# This file is the reproduction: create+destroy a dummy engine, then throw
# through a try_call thunk.

using Test
using Libdl

const MLIR_AVAILABLE = try
    using RepliBuild
    isfile(RepliBuild.MLIRNative.libJLCS)
catch
    false
end

const CLANGXX = Sys.which("clang++")
if !MLIR_AVAILABLE || CLANGXX === nothing
    # Do not exit(0): this file is included from devtests.jl, and an exit
    # would skip every test after it while reporting success.
    @info "libJLCS or clang++ not found — skipping JIT unwind tests"
else

using RepliBuild.MLIRNative

const FIXTURE_DIR = joinpath(@__DIR__, "jit_unwind")
const FIXTURE_SRC = joinpath(FIXTURE_DIR, "thrower.cpp")
const FIXTURE_LIB = joinpath(FIXTURE_DIR, "libthrower." * Libdl.dlext)

mkpath(FIXTURE_DIR)
write(FIXTURE_SRC, """
#include <stdexcept>
#include <string>

extern "C" {

int thrower_ok(int x) {
    return x * 2;
}

int thrower_boom(int x) {
    throw std::runtime_error("thrower_boom x=" + std::to_string(x));
}

}
""")

cmd = `$CLANGXX -shared -O1 -std=c++17 -o $FIXTURE_LIB $FIXTURE_SRC`
if Sys.iswindows()
    cmd = `$CLANGXX -shared -O1 -std=c++17 -Wl,--export-all-symbols -o $FIXTURE_LIB $FIXTURE_SRC`
end
if !isfile(FIXTURE_LIB) || mtime(FIXTURE_SRC) > mtime(FIXTURE_LIB)
    run(cmd)
end

function make_engine(ir)
    ctx = create_context()
    mod = parse_module(ctx, ir)
    @assert lower_to_llvm(mod)
    jit = create_jit(mod, opt_level=0,
                     shared_libs=[FIXTURE_LIB, RepliBuild.MLIRNative.libJLCS])
    return ctx, jit
end

@testset "JIT unwind survives prior engine teardown" begin
    # 1. A dummy engine that is created and destroyed first — the poison step.
    let
        ctx, jit = make_engine("""
        module {
          func.func @add(%a: i32, %b: i32) -> i32 attributes {llvm.emit_c_interface} {
            %r = arith.addi %a, %b : i32
            return %r : i32
          }
        }
        """)
        @test jit != C_NULL
        destroy_jit(jit)
        destroy_context(ctx)
    end

    # 2. A live engine whose try_call must catch a C++ throw. If .pdata was
    #    never registered, or if step 1 poisoned libunwind, this aborts the
    #    process with "libunwind: pc not in table" instead of returning.
    ir = """
    module {
      func.func private @thrower_ok(i32) -> i32
      func.func private @thrower_boom(i32) -> i32

      func.func @call_ok(%x: i32) -> i32 attributes {llvm.emit_c_interface} {
        %r = jlcs.try_call %x { callee = @thrower_ok } : (i32) -> i32
        return %r : i32
      }
      func.func @call_boom(%x: i32) -> i32 attributes {llvm.emit_c_interface} {
        %r = jlcs.try_call %x { callee = @thrower_boom } : (i32) -> i32
        return %r : i32
      }
    }
    """
    ctx, jit = make_engine(ir)
    @test jit != C_NULL

    ok = Ref(Int32(0))
    arg = Ref(Int32(21))
    ptrs = Ptr{Cvoid}[Base.unsafe_convert(Ptr{Cvoid}, arg),
                      Base.unsafe_convert(Ptr{Cvoid}, ok)]
    GC.@preserve arg ok begin
        @test jit_invoke(jit, "call_ok", ptrs)
    end
    @test ok[] == 42
    @test !has_pending_exception()

    boom = Ref(Int32(0))
    barg = Ref(Int32(7))
    bptrs = Ptr{Cvoid}[Base.unsafe_convert(Ptr{Cvoid}, barg),
                       Base.unsafe_convert(Ptr{Cvoid}, boom)]
    GC.@preserve barg boom begin
        @test jit_invoke(jit, "call_boom", bptrs)
    end
    @test boom[] == 0  # sentinel
    @test has_pending_exception()
    msg = get_pending_exception()
    @test occursin("thrower_boom", msg)
    @test occursin("7", msg)
    clear_pending_exception()

    destroy_jit(jit)
    destroy_context(ctx)
end

end # MLIR_AVAILABLE && clang++
