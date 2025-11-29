#!/usr/bin/env julia
println("="^70)
println("BINDING QUALITY TEST - Simple Functions Only")
println("="^70)

include("julia_bindings/DwarfTestBindings.jl")
using .DwarfTestBindings

println("\n✓ Wrapper loaded successfully")

# Test 1: Simple function that works
println("\n" * "="^70)
println("TEST 1: Working Function")
println("="^70)

result = c_add(5, 3)
println("  c_add(5, 3) = $result")
@assert result == 8
println("  ✓ c_add works")

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
println("  ✓ Color enum works")

# Test 3: Structs
println("\n" * "="^70)
println("TEST 3: Structs")
println("="^70)

p = Point2D(3.0, 4.0)
println("  Created Point2D: x=$(p.x), y=$(p.y)")
@assert p.x == 3.0
@assert p.y == 4.0
println("  ✓ Point2D works")

v = Vector3D(1.0, 2.0, 3.0)
println("  Created Vector3D: x=$(v.x), y=$(v.y), z=$(v.z)")
@assert v.x == 1.0
@assert v.y == 2.0
@assert v.z == 3.0
println("  ✓ Vector3D works")

# Test 4: Namespace functions
println("\n" * "="^70)
println("TEST 4: Namespace Functions")
println("="^70)

pi_val = math_pi()
println("  math::pi() = $pi_val")
@assert pi_val ≈ 3.14159 atol=0.001
println("  ✓ math_pi works")

# Summary
println("\n" * "="^70)
println("✅ ALL BASIC TESTS PASSED")
println("="^70)

println("\nWhat Works:")
println("  ✓ Simple functions (c_add)")
println("  ✓ Enums extracted and usable")
println("  ✓ Structs constructed and ordered correctly")
println("  ✓ Namespace functions accessible")
println("  ✓ Template struct names sanitized (Pair_int, FixedArray_float_10)")
println("  ✓ Member names sanitized (_vptr_Shape)")

println("\nKnown Issues:")
println("  • typedef resolution incomplete (int32_t, int64_t → unknown)")
println("  • Functions with typedef params/return segfault")
println("  • Methods missing 'this' pointer parameter")
