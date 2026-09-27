# Test script for C++ exception handling through JLCS try_call
# Run: julia --project=. test/callback_test/test_exceptions.jl

using Test

# Rebuild wrapper if needed
wrapper_path = joinpath(@__DIR__, "julia", "CallbackTest.jl")
if !isfile(wrapper_path)
    error("Run build first: julia --project=. -e 'using RepliBuild; RepliBuild.discover(\"test/callback_test\"); RepliBuild.build(\"test/callback_test/replibuild.toml\"); RepliBuild.wrap(\"test/callback_test/replibuild.toml\")'")
end

include(wrapper_path)
using .CallbackTest

JITManager = Base.get_extension(CallbackTest, :JITManager)
# Access JITManager through RepliBuild
using RepliBuild
JM = RepliBuild.JITManager
using RepliBuild.MLIRNative

@testset "C++ Exception Handling" begin
    # Prior JIT teardown used to poison process-wide unwind: RuntimeDyld
    # fed COFF .pdata to libunwind's __register_frame, then
    # __deregister_frame on destroy. The next throw through a live JIT
    # frame then died with "libunwind: pc not in table". Standalone this
    # file did not reproduce because nothing had been destroyed yet.
    @testset "prior JIT teardown does not poison unwind" begin
        ctx = create_context()
        try
            ir = """
            module {
              func.func @add(%a: i32, %b: i32) -> i32
                  attributes {llvm.emit_c_interface} {
                %r = arith.addi %a, %b : i32
                return %r : i32
              }
            }
            """
            mod = parse_module(ctx, ir)
            @test lower_to_llvm(mod)
            jit = create_jit(mod)
            @test jit != C_NULL
            destroy_jit(jit)
        finally
            destroy_context(ctx)
        end
    end

    @testset "noexcept functions stay on ccall (fast path)" begin
        # safe_multiply is noexcept — should use ccall, not JIT
        @test CallbackTest.safe_multiply(3, 4) == 12
        @test CallbackTest.safe_multiply(0, 100) == 0
        @test CallbackTest.safe_multiply(-5, 3) == -15
    end

    @testset "Non-throwing path works through JIT" begin
        # throws_if_negative with positive value — goes through JIT but doesn't throw
        @test CallbackTest.throws_if_negative(5) == 10
        @test CallbackTest.throws_if_negative(0) == 0
        @test CallbackTest.throws_if_negative(100) == 200
    end

    @testset "std::runtime_error is caught as CxxException" begin
        # always_throws should be caught
        ex = try
            CallbackTest.always_throws(99)
            nothing
        catch e
            e
        end
        @test ex isa JM.CxxException
        @test occursin("always_throws", ex.message)
        @test occursin("99", ex.message)
    end

    @testset "Conditional throw works" begin
        # throws_if_negative: positive works, negative throws
        @test CallbackTest.throws_if_negative(10) == 20

        ex = try
            CallbackTest.throws_if_negative(-7)
            nothing
        catch e
            e
        end
        @test ex isa JM.CxxException
        @test occursin("negative", ex.message)
        @test occursin("-7", ex.message)
    end

    @testset "Void function throw" begin
        ex = try
            CallbackTest.void_thrower()
            nothing
        catch e
            e
        end
        @test ex isa JM.CxxException
        @test occursin("void function threw", ex.message)
    end

    @testset "Non-std::exception (throw int) caught" begin
        ex = try
            CallbackTest.throws_int(0)
            nothing
        catch e
            e
        end
        @test ex isa JM.CxxException
        @test occursin("unknown", ex.message)
    end

    @testset "Exception midway through computation" begin
        ex = try
            CallbackTest.throws_midway(10)
            nothing
        catch e
            e
        end
        @test ex isa JM.CxxException
        @test occursin("iteration", ex.message)
    end

    @testset "bare-name noexcept collision" begin
        # Loud::value shares its bare name with Quiet::value() noexcept, and
        # checked is noexcept(false). The regex scan named all three. Only Quiet
        # is nounwind in the IR, so only Quiet may take the ccall path.
        tier = CallbackTest.DISPATCH_TIER
        @test tier[:collide_Quiet_value] == :tier3
        @test tier[:collide_Loud_value] != :tier3
        @test tier[:collide_checked] != :tier3
        # Non-throwing calls work on both tiers.
        q = Ref{UInt8}(0)
        GC.@preserve q begin
            p = Base.unsafe_convert(Ptr{UInt8}, q)
            @test CallbackTest.collide_Quiet_value(p, 1) == 2
            @test CallbackTest.collide_Loud_value(p, 1) == 3
        end
        @test CallbackTest.collide_checked(2) == 6

        # The throws themselves run in a SUBPROCESS. A regression here is not
        # a failed test but an abort (exit 134), and in-process it would take
        # every later devtests file down with it.
        code = raw"""
            include(ARGS[1]); import .CallbackTest as C
            q = Ref{UInt8}(0)
            for (label, f) in (("loud",    () -> GC.@preserve(q, C.collide_Loud_value(Base.unsafe_convert(Ptr{UInt8}, q), -1))),
                               ("checked", () -> C.collide_checked(13)))
                try f(); println(label, "=returned") catch e; println(label, "=", typeof(e)) end
            end
            """
        out = IOBuffer()
        proc = run(pipeline(ignorestatus(`$(Base.julia_cmd()) --project=$(dirname(dirname(@__DIR__))) -e $code $wrapper_path`);
                            stdout=out, stderr=devnull))
        res = String(take!(out))
        # `success`, not `exitcode == 0`: an abort is SIGABRT, and a signalled
        # process reports exitcode 0 with termsignal 6 — the weaker check passes
        # on exactly the failure this testset exists to catch.
        @test success(proc)
        @test occursin("loud=RepliBuild.JITManager.CxxException", res)
        @test occursin("checked=RepliBuild.JITManager.CxxException", res)
    end

    @testset "Original extern C callbacks still work" begin
        my_add_fn(a::Cint, b::Cint)::Cint = a + b
        c_add = @cfunction($my_add_fn, Cint, (Cint, Cint))
        @test CallbackTest.execute_binary_op(Base.unsafe_convert(Ptr{Cvoid}, c_add), 10, 20) == 30
    end
end
