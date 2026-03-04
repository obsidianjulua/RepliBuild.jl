# 1. Include the generated wrapper file
# Path is relative to this script file
include("julia/HelloWorldTest.jl")

# 2. Use the module
using .HelloWorldTest

HelloWorldTest.hello_world()
#