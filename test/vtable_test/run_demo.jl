using Pkg
Pkg.activate(".") # RepliBuild project

# Include the generated wrapper
include("julia/VtableTest.jl")
using .VtableTest

println("Successfully loaded VtableTest")

# 1. Create shapes via C API (ccall)
println("\n[1] Creating shapes...")
rect = create_rectangle(10.0, 20.0)
println("  Created Rectangle: $rect")

circ = create_circle(5.0)
println("  Created Circle: $circ")

# 2. Call methods via C API wrappers (MLIR Dispatch!)
# get_area uses RepliBuild.JITManager.invoke because Shape is a class
println("\n[2] Calling get_area via MLIR JIT Dispatch...")

# Rectangle area: 10 * 20 = 200
area_rect = get_area(rect)
println("  Rectangle Area: $area_rect")
if area_rect ≈ 200.0
    println("  ✓ Correct")
else
    println("  ✗ Incorrect (expected 200.0)")
end

# Circle area: pi * 5^2 ≈ 78.5398
area_circ = get_area(circ)
println("  Circle Area:    $area_circ")
if area_circ ≈ 78.53981633974483
    println("  ✓ Correct")
else
    println("  ✗ Incorrect")
end

# 3. Call methods via C API wrappers (MLIR Dispatch!)
println("\n[3] Calling get_perimeter via MLIR JIT Dispatch...")
perim_rect = get_perimeter(rect)
println("  Rectangle Perimeter: $perim_rect")
if perim_rect ≈ 60.0
    println("  ✓ Correct")
else
    println("  ✗ Incorrect")
end

# 4. Clean up
println("\n[4] Cleaning up...")
delete_shape(rect)
delete_shape(circ)
println("  Deleted shapes")

println("\nTest Complete!")
