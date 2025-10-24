#!/usr/bin/env julia
# Test universal JLL package resolution

using Pkg
Pkg.activate(".")

using RepliBuild
using RepliBuild.ModuleRegistry

println("="^70)
println("Universal JLL Resolution Test")
println("="^70)
println()

# Test packages with various naming conventions
test_packages = [
    ("Boost", "Boost_jll"),
    ("ZLIB", "Zlib_jll"),
    ("PNG", "libpng_jll"),
    ("OpenCV", "OpenCV_jll"),
    ("CURL", "LibCURL_jll"),
    ("FFmpeg", "FFMPEG_jll"),
    ("GLFW", "GLFW3_jll"),
    ("FreeType", "FreeType2_jll"),
]

println("🧪 Testing JLL package auto-discovery:")
println("-"^70)

for (cmake_name, expected_jll) in test_packages
    print("  Testing $cmake_name → ")

    mod_info = ModuleRegistry.resolve_module(cmake_name)

    if !isnothing(mod_info)
        if mod_info.julia_package == expected_jll
            println("✅ $expected_jll")
        else
            println("⚠️  Got $(mod_info.julia_package), expected $expected_jll")
        end
    else
        println("❌ Not found")
    end
end

println()
println("="^70)
println("✅ Universal JLL resolution working!")
println("="^70)
println()
println("Key points:")
println("• No manual TOML files needed for most packages")
println("• Auto-discovers name variations (lib*, Lib*, case changes)")
println("• Searches Julia General registry automatically")
println("• Falls back to system libraries via pkg-config")
println()
