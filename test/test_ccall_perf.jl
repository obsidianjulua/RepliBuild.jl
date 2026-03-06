using Pkg
Pkg.activate(".")
using BenchmarkTools

const LIB_PATH = "libm.so.6"

function call_literal(x::Float64)
    ccall((:sin, "libm.so.6"), Float64, (Float64,), x)
end

function call_const(x::Float64)
    ccall((:sin, LIB_PATH), Float64, (Float64,), x)
end

function loop_literal(n)
    acc = 0.0
    for i in 1:n
        acc += call_literal(1.0)
    end
    acc
end

function loop_const(n)
    acc = 0.0
    for i in 1:n
        acc += call_const(1.0)
    end
    acc
end

println("literal:")
@btime loop_literal(1000)

println("const:")
@btime loop_const(1000)
