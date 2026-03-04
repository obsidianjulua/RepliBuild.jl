using Pkg
Pkg.activate(".")
using RepliBuild

println("Step 1: Discover")
# Force overwrite if exists to ensure clean state
RepliBuild.discover("test/hello_world_test", force=true)

println("\nStep 2: Build")
RepliBuild.build("test/hello_world_test/replibuild.toml")

println("\nStep 3: Wrap")
RepliBuild.wrap("test/hello_world_test/replibuild.toml")

println("\nStep 4: Verify Wrapper Content")
wrapper_path = "test/hello_world_test/julia/HelloWorldTest.jl"
if isfile(wrapper_path)
    println("Wrapper generated at: $wrapper_path")
    println("--- Wrapper Content Preview ---")
    println(read(wrapper_path, String))
    println("--- End Preview ---")
else
    println("Error: Wrapper not found at $wrapper_path")
end
