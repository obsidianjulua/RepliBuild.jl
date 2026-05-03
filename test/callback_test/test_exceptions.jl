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

@testset "C++ Exception Handling" begin
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

    @testset "Original extern C callbacks still work" begin
        my_add_fn(a::Cint, b::Cint)::Cint = a + b
        c_add = @cfunction($my_add_fn, Cint, (Cint, Cint))
        @test CallbackTest.execute_binary_op(Base.unsafe_convert(Ptr{Cvoid}, c_add), 10, 20) == 30
    end
end
