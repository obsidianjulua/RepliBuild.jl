# Verification script for cross-boundary callbacks
# Expects: wrapper already generated at julia/CallbackTest.jl

using Test

wrapper_path = joinpath(@__DIR__, "julia", "CallbackTest.jl")
if !isfile(wrapper_path)
    error("Wrapper not found at $wrapper_path. Did you run build + wrap?")
end

include(wrapper_path)

function my_add(a::Cint, b::Cint)::Cint
    return a + b
end

global progress_updates = 0
global last_progress = 0.0f0

function my_progress(p::Cfloat)::Cvoid
    global progress_updates
    global last_progress
    progress_updates += 1
    last_progress = p
    return nothing
end

@testset "Callback Verification" begin
    println("  Testing BinaryOp callback...")
    c_add = @cfunction(my_add, Cint, (Cint, Cint))
    res = CallbackTest.execute_binary_op(Base.unsafe_convert(Ptr{Cvoid}, c_add), Int32(10), Int32(20))
    @test res == 30

    println("  Testing Progress callback...")
    global progress_updates = 0
    global last_progress = 0.0f0
    c_progress = @cfunction(my_progress, Cvoid, (Cfloat,))
    CallbackTest.simulate_work(Int32(5), Base.unsafe_convert(Ptr{Cvoid}, c_progress))
    @test progress_updates == 5
    @test last_progress ≈ 1.0f0
end
