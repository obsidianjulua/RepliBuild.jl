#!/usr/bin/env julia
# Check which dependencies are actually used

deps = [
    "Artifacts", "Clang", "Configurations", "CxxWrap", "DBInterface",
    "DaemonMode", "DataFrames", "Dates", "Distributed", "Documenter",
    "FreeType", "GLFW_jll", "JSON", "LLVM_full_assert_jll", "Libdl",
    "PackageCompiler", "Pkg", "ProgressMeter", "SQLite", "Sockets",
    "TOML", "UUIDs", "libpng_jll"
]

println("Checking dependency usage in src/ and test/...")
println("=" ^ 60)

used = String[]
unused = String[]
test_only = String[]

for pkg in deps
    # Check src/
    src_count = 0
    for file in readdir("src", join=true)
        if endswith(file, ".jl")
            content = read(file, String)
            if occursin(Regex("\\busing $pkg\\b|\\bimport $pkg\\b"), content)
                src_count += 1
            end
        end
    end

    # Check test/
    test_count = 0
    for (root, dirs, files) in walkdir("test")
        for file in files
            if endswith(file, ".jl")
                filepath = joinpath(root, file)
                content = read(filepath, String)
                if occursin(Regex("\\busing $pkg\\b|\\bimport $pkg\\b"), content)
                    test_count += 1
                end
            end
        end
    end

    if src_count > 0
        push!(used, pkg)
        println("✅ $pkg - src: $src_count files, test: $test_count files")
    elseif test_count > 0
        push!(test_only, pkg)
        println("🧪 $pkg - TEST ONLY ($test_count files)")
    else
        push!(unused, pkg)
        println("❌ $pkg - NOT USED")
    end
end

println("\n" * "=" ^ 60)
println("Summary:")
println("  Used in src/: $(length(used))")
println("  Test only: $(length(test_only))")
println("  Unused: $(length(unused))")

if !isempty(unused)
    println("\n⚠️  Unused dependencies (can be removed):")
    for pkg in unused
        println("  - $pkg")
    end
end

if !isempty(test_only)
    println("\n📋 Test-only dependencies (move to [extras]):")
    for pkg in test_only
        println("  - $pkg")
    end
end

println("\n✅ Required dependencies:")
for pkg in used
    println("  - $pkg")
end
