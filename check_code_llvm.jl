using Pkg
Pkg.activate(".")

# We need the structs to be able to include DiscBench.jl
wrapper_path = "test/discourse_benchmark/julia/DiscBench.jl"
src = read(wrapper_path, String)

structs_stub = """
struct Point2D
    x::Cdouble
    y::Cdouble
end
struct PackedRecord
    tag::UInt8
    value::Cint
    flag::UInt8
end
"""
patched = replace(src, r"(^module DiscBench\s*$)"m => SubstitutionString("\\1\n" * structs_stub); count=1)
write("test/discourse_benchmark/julia/DiscBench_patched.jl", patched)

include("test/discourse_benchmark/julia/DiscBench_patched.jl")

using InteractiveUtils
println("Code LLVM for DiscBench.add_to:")
@code_llvm DiscBench.add_to(1.0, 2.0)

println("\n\nCode LLVM for loop:")
data = rand(1000)
function loop(data)
    acc = 0.0
    for i in 1:length(data)
        acc = DiscBench.add_to(acc, data[i])
    end
    acc
end
@code_llvm loop(data)
