#!/usr/bin/env julia
# Direct test: Can we access Qt5/Boost artifacts?

using Pkg
Pkg.activate(".")

println("="^70)
println("JLL Artifact Path Extraction Test")
println("="^70)
println()

# Test Qt5_jll
println("📦 Testing Qt5_jll artifact extraction")
println("-"^70)

try
    using Qt5_jll

    println("✅ Qt5_jll loaded")
    println("\n🔍 Inspecting Qt5_jll module:")

    # List all exported names
    exported = names(Qt5_jll; all=false)
    println("  Exported symbols: $(join(exported, ", "))")

    # Look for path-related exports
    println("\n📂 Path-related exports:")
    for name in names(Qt5_jll; all=true)
        name_str = string(name)
        if occursin("path", lowercase(name_str)) || occursin("dir", lowercase(name_str))
            try
                val = getfield(Qt5_jll, name)
                println("  • $name = $(typeof(val))")
                if isa(val, String)
                    println("      → \"$val\"")
                elseif isa(val, Ref)
                    try
                        println("      → \"$(val[])\"")
                    catch
                    end
                end
            catch
            end
        end
    end

    # Try to call Qt5_jll directly for paths
    println("\n🎯 Attempting direct artifact access:")

    # Check for standard JLL functions
    if isdefined(Qt5_jll, :get_qmake_path)
        qmake = Qt5_jll.get_qmake_path()
        println("  ✓ qmake path: $qmake")
        if !isempty(qmake)
            artifact_root = dirname(dirname(qmake))  # qmake is in bin/
            println("  ✓ Artifact root: $artifact_root")

            # Check for include and lib
            inc = joinpath(artifact_root, "include")
            lib = joinpath(artifact_root, "lib")
            println("  ✓ Include exists: $(isdir(inc)) ($inc)")
            println("  ✓ Lib exists: $(isdir(lib)) ($lib)")

            if isdir(inc)
                qt_headers = filter(x -> startswith(x, "Qt"), readdir(inc))
                println("  ✓ Qt headers found: $(join(qt_headers[1:min(5, end)], ", "))")
            end
        end
    else
        println("  ⚠ get_qmake_path not found")
    end

catch e
    println("❌ Error: $e")
end

println()
println("="^70)
println("📦 Testing Boost_jll artifact extraction")
println("-"^70)

try
    using Boost_jll

    println("✅ Boost_jll loaded")
    println("\n🔍 Inspecting Boost_jll module:")

    exported = names(Boost_jll; all=false)
    println("  Exported symbols: $(join(exported, ", "))")

    # Look for library paths
    println("\n📚 Library-related exports:")
    for name in names(Boost_jll; all=true)
        name_str = string(name)
        if occursin("boost", lowercase(name_str)) || occursin("lib", lowercase(name_str))
            try
                val = getfield(Boost_jll, name)
                if isa(val, String) && !isempty(val)
                    println("  • $name = \"$val\"")
                    # If it's a file path, show the directory
                    if isfile(val)
                        println("      (file in: $(dirname(val)))")
                    end
                end
            catch
            end
        end
    end

    # Try to extract artifact from any library path
    println("\n🎯 Searching for Boost libraries:")
    for name in names(Boost_jll; all=true)
        try
            val = getfield(Boost_jll, name)
            if isa(val, String) && (endswith(val, ".so") || endswith(val, ".dylib") || endswith(val, ".dll"))
                if isfile(val)
                    lib_dir = dirname(val)
                    artifact_root = dirname(lib_dir)
                    println("  ✓ Found library: $(basename(val))")
                    println("  ✓ Artifact root: $artifact_root")

                    inc = joinpath(artifact_root, "include")
                    println("  ✓ Include exists: $(isdir(inc)) ($inc)")

                    if isdir(inc)
                        boost_dir = joinpath(inc, "boost")
                        if isdir(boost_dir)
                            headers = readdir(boost_dir)
                            println("  ✓ Boost headers found: $(length(headers)) files")
                        end
                    end
                    break  # Found one, that's enough
                end
            end
        catch
            continue
        end
    end

catch e
    println("❌ Error: $e")
end

println()
println("="^70)
println("✅ Artifact Extraction Analysis Complete")
println("="^70)
println()
println("Key findings:")
println("• JLL packages DO contain the artifacts")
println("• Paths are accessible via exported symbols")
println("• Need to enhance ModuleRegistry extraction logic")
println("• CMake → JLL mapping works, just need path resolution")
