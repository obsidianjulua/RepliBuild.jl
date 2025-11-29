#!/usr/bin/env julia
# Test the generated bindings - validate quality and usability

println("="^70)
println("BINDING QUALITY TEST")
println("="^70)

# Load the generated wrapper
include("julia_bindings/DwarfTestBindings.jl")
using .DwarfTestBindings

println("\n✓ Wrapper loaded successfully")

# Test 1: Simple functions
println("\n" * "="^70)
println("TEST 1: Simple C Functions")
println("="^70)

result = c_add(5, 3)
println("  c_add(5, 3) = $result")
@assert result == 8

result = mul64(1000, 2000)
println("  mul64(1000, 2000) = $result")
@assert result == 2000000

println("  ✓ Simple functions work")

# Test 2: Enums
println("\n" * "="^70)
println("TEST 2: Enums")
println("="^70)

println("  Color enum values:")
println("    RED = $(Int(RED))")
println("    GREEN = $(Int(GREEN))")
println("    BLUE = $(Int(BLUE))")

@assert Int(RED) == 0
@assert Int(GREEN) == 1
@assert Int(BLUE) == 2

println("  Status enum values:")
println("    IDLE = $(Int(IDLE))")
println("    RUNNING = $(Int(RUNNING))")
println("    ERROR = $(Int(ERROR))")

@assert Int(IDLE) == 0
@assert Int(RUNNING) == 10
@assert Int(ERROR) == 255

println("  ✓ Enums work correctly")

# Test 3: Structs
println("\n" * "="^70)
println("TEST 3: Structs")
println("="^70)

p = Point2D(3.0, 4.0)
println("  Created Point2D: x=$(p.x), y=$(p.y)")
@assert p.x == 3.0
@assert p.y == 4.0

v = Vector3D(1.0, 2.0, 3.0)
println("  Created Vector3D: x=$(v.x), y=$(v.y), z=$(v.z)")
@assert v.x == 1.0
@assert v.y == 2.0
@assert v.z == 3.0

println("  ✓ Structs constructed correctly")

# Test 4: Namespace functions
println("\n" * "="^70)
println("TEST 4: Namespace Functions")
println("="^70)

pi_val = math_pi()
println("  math::pi() = $pi_val")
@assert pi_val ≈ 3.14159 atol=0.001

deg = 180.0
rad = math_deg_to_rad(deg)
println("  math::deg_to_rad(180.0) = $rad")
@assert rad ≈ 3.14159 atol=0.001

println("  ✓ Namespace functions work")

# Test 5: Type conversion
println("\n" * "="^70)
println("TEST 5: Integer Type Conversion")
println("="^70)

# Test that Integer types work (not just Cint)
result = c_add(Int64(10), Int32(20))
println("  c_add(Int64(10), Int32(20)) = $result")
@assert result == 30

println("  ✓ Type conversion works")

# Summary
println("\n" * "="^70)
println("✅ ALL TESTS PASSED")
println("="^70)

println("\nBinding Quality Assessment:")
println("  ✓ Simple functions callable")
println("  ✓ Enums extracted and usable")
println("  ✓ Structs constructed")
println("  ✓ Namespace functions accessible (math_*)")
println("  ✓ Type conversion automatic (Integer → Cint)")
println("  ✓ No manual wrapper edits needed")

println("\nKnown Limitations:")
println("  • Template struct names invalid Julia syntax (Pair<int>)")
println("    → Recommendation: Sanitize < > to _ in struct names")
println("  • Methods missing 'this' pointer parameter")
println("    → Recommendation: Detect DW_AT_object_pointer")
println("  • Inheritance not exposed in Julia types")
println("    → Acceptable: metadata available for docs")
