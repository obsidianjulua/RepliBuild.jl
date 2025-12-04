#!/usr/bin/env julia
# test_vtable_pipeline.jl - End-to-end test: C++ → DWARF → JLCS IR → MLIR

using Test

# Load our modules
include("../src/DWARFParser.jl")
include("../src/JLCSIRGenerator.jl")

using .DWARFParser
using .JLCSIRGenerator

@testset "Vtable Pipeline" begin
    binary = joinpath(@__DIR__, "..", "examples", "vtable_test", "test_vtable")

    @test isfile(binary)

    # Step 1: Parse DWARF
    println("\n=== Step 1: Parse DWARF ===")
    vtinfo = parse_vtables(binary)

    @test length(vtinfo.classes) > 0
    @test haskey(vtinfo.classes, "Base")
    @test haskey(vtinfo.classes, "Derived")
    @test haskey(vtinfo.vtable_addresses, "Base")

    # Step 2: Generate JLCS IR
    println("\n=== Step 2: Generate JLCS IR ===")
    ir = generate_jlcs_ir(vtinfo)

    @test contains(ir, "module {")
    @test contains(ir, "jlcs.type_info")
    @test contains(ir, "Base")

    println(ir)

    # Step 3: Save to file
    output = joinpath(@__DIR__, "..", "examples", "vtable_test", "test.mlir")
    save_mlir_module(binary, output)

    @test isfile(output)

    println("\n=== Pipeline Complete ===")
    println("Binary:     $binary")
    println("MLIR IR:    $output")
    println("Classes:    $(length(vtinfo.classes))")
    println("Vtables:    $(length(vtinfo.vtable_addresses))")
    println("Methods:    $(length(vtinfo.method_addresses))")
end
