#!/usr/bin/env julia
# test_qt_delegation.jl - Test BuildSystemDelegate Qt integration

using Pkg
Pkg.activate(".")

# Load RepliBuild
using RepliBuild

println("=" ^ 70)
println("Testing BuildSystemDelegate - Qt Integration")
println("=" ^ 70)

# Test 1: Environment detection
println("\n📋 Test 1: Environment Detection")
println("   Julia environment: ", RepliBuild.BuildSystemDelegate.is_julia_environment())
println("   Should use JLL: ", RepliBuild.BuildSystemDelegate.should_use_jll())

# Test 2: Build system detection
println("\n📋 Test 2: Build System Detection")
test_project_dir = pwd()
detected = RepliBuild.BuildSystemDelegate.detect_build_system(test_project_dir)
println("   Detected build system: $detected")

# Test 3: Check if Qt5Base_jll is available
println("\n📋 Test 3: Qt5Base_jll Availability")
try
    using Qt5Base_jll
    println("   ✓ Qt5Base_jll is installed")

    # Try to find qmake
    println("\n📋 Test 4: Finding qmake in Qt5Base_jll")
    println("   Qt5Base artifact dir:")
    for name in names(Qt5Base_jll; all=true)
        if occursin("path", string(name)) || occursin("dir", string(name))
            try
                val = getfield(Qt5Base_jll, name)
                if isa(val, String) && isdir(val)
                    println("     $name = $val")
                end
            catch
            end
        end
    end
catch e
    println("   ⚠ Qt5Base_jll not installed: $e")
    println("   Install with: Pkg.add(\"Qt5Base_jll\")")
end

# Test 5: System qmake fallback
println("\n📋 Test 5: System qmake Fallback")
system_qmake = RepliBuild.BuildSystemDelegate.find_system_qmake()
if !isempty(system_qmake)
    println("   ✓ System qmake found: $system_qmake")
else
    println("   ⚠ No system qmake found")
end

println("\n" * "=" ^ 70)
println("Test complete!")
println("=" ^ 70)
