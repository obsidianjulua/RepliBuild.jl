# Verification script for vtable_test

using Test
using Libdl

wrapper_path = joinpath(@__DIR__, "julia", "VtableTest.jl")
if !isfile(wrapper_path)
    error("Wrapper not found at $wrapper_path")
end

include(wrapper_path)
using .VtableTest

@testset "VtableTest Verification" begin
    println("Verifying VtableTest (Virtual Dispatch)...")

    # 1. Create objects (Using Safe Wrappers)
    println("  Creating Rectangle (Safe)...")
    rect = create_rectangle_safe(10.0, 20.0) # Returns ManagedShape
    
    println("  Creating Circle (Safe)...")
    circle = create_circle_safe(5.0) # Returns ManagedShape

    # 2. Call virtual methods (Polymorphism)
    # The unsafe_convert defined in wrapper allows passing ManagedShape directly to ccall
    println("  Testing virtual dispatch (get_area)...")
    
    # Area of 10x20 rect = 200.0
    area_rect = get_area(rect)
    @test area_rect ≈ 200.0
    
    # Area of radius 5 circle = pi * 25
    area_circle = get_area(circle)
    @test area_circle ≈ (π * 25.0)

    # 3. Call another virtual method
    println("  Testing virtual dispatch (get_perimeter)...")
    perim_rect = get_perimeter(rect)
    @test perim_rect ≈ 60.0 # 2*(10+20)

    # 4. Cleanup
    # No explicit delete needed! GC will handle it.
    println("  Implicit cleanup via GC (no segfault expected)")
    finalize(rect)   # Manually trigger finalizer to verify it doesn't crash
    finalize(circle)
    
    println("✓ VtableTest Passed")
end
