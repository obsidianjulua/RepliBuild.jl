using Pkg
Pkg.activate(".")
using BenchmarkTools

const IR = """
define double @add_to(double %0, double %1) {
top:
  %2 = fadd double %0, %1
  ret double %2
}
"""

const LTO_IR = IR
const LIBRARY_PATH = "dummy"

function add_to_llvmcall(acc::Float64, val::Float64)
    if !isempty(LTO_IR)
        return Base.llvmcall((LTO_IR, "add_to"), Float64, Tuple{Float64, Float64}, acc, val)
    else
        return ccall((:add_to, LIBRARY_PATH), Float64, (Float64, Float64,), acc, val)
    end
end

function loop_llvmcall(n)
    acc = 0.0
    for i in 1:n
        acc = add_to_llvmcall(acc, 1.0)
    end
    acc
end

println("llvmcall without @inline:")
@btime loop_llvmcall(1000)
