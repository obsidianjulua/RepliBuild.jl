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

    # 1. Create objects (Using Idiomatic Constructors)
    println("  Creating Rectangle...")
    rect = Rectangle(10.0, 20.0)
    
    println("  Creating Circle...")
    circle = Circle(5.0)

    # 2. Call virtual methods (Polymorphism via Method Proxies)
    println("  Testing virtual dispatch (get_area)...")
    
    # Area of 10x20 rect = 200.0
    area_rect = area(rect)
    @test area_rect ≈ 200.0
    
    # Area of radius 5 circle = pi * 25
    area_circle = area(circle)
    @test area_circle ≈ (π * 25.0)

    # 3. Call another virtual method
    println("  Testing virtual dispatch (get_perimeter)...")
    perim_rect = perimeter(rect)
    @test perim_rect ≈ 60.0 # 2*(10+20)

    # 4. Cleanup
    # No explicit delete needed! GC will handle it via finalizers.
    println("  Implicit cleanup via GC (no segfault expected)")
    finalize(rect)
    finalize(circle)
    
    println("✓ VtableTest Passed")
end
