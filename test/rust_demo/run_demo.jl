using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using RepliBuild

println("Building RustDemo...")
toml = joinpath(@__DIR__, "replibuild.toml")

# Build the Rust library
lib = RepliBuild.build(toml)
println("Library built: ", lib)

# Generate the wrapper
wrap = RepliBuild.wrap(toml)
println("Wrapper generated: ", wrap)

println("\n--- Generated Wrapper Content ---")
println(read(wrap, String))
println("---------------------------------\n")

# Load and test the wrapper
include(wrap)
using .RustDemo

println("Testing wrapper...")
p = RustDemo.create_particle(1.0f0, 2.0f0, 5.0)
println("Particle created: x=$(p.x), y=$(p.y), mass=$(p.mass)")

p_ref = Ref(p)
new_state = GC.@preserve p_ref begin
    RustDemo.update_particle(Base.unsafe_convert(Ptr{RustDemo.Particle}, p_ref), RustDemo.Inactive)
end
println("Particle updated: x=$(p_ref[].x), y=$(p_ref[].y), mass=$(p_ref[].mass)")
println("New state: ", new_state)

ptr = RustDemo.get_status_string(new_state)
str = unsafe_string(ptr)
println("Status string: ", str)
RustDemo.free_string(ptr)

flags_res = RustDemo.check_flags(RustDemo.HighBit)
println("Check Flags HighBit: ", flags_res)

println("RustDemo Tests passed successfully!")
