# 1. Include the generated wrapper file
# Path is relative to this script file
include("julia/HelloWorldTest.jl")

# 2. Use the module
using .HelloWorldTest

println("--- Executing C++ Functions via Julia Wrapper ---")

# 3. Call the void function (hello_world)
print("Calling hello_world(): ")
hello_world() # Prints to stdout from C++

# 4. Call the function with arguments and return value (add)
a, b = 10, 32
result = add(a, b)
println("Calling add($a, $b): $result")