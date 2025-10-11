#!/usr/bin/env julia
# Build custom sysimage for RepliBuild
# This creates a compiled sysimage with RepliBuild precompiled for faster startup

using Pkg

# Ensure PackageCompiler is installed
if !haskey(Pkg.project().dependencies, "PackageCompiler")
    println("📦 Installing PackageCompiler.jl...")
    Pkg.add("PackageCompiler")
end

using PackageCompiler

println("🚀 Building RepliBuild sysimage...")
println("="^60)

# Configuration
build_dir = @__DIR__
project_dir = dirname(build_dir)  # Parent directory (RepliBuild root)
sysimage_path = joinpath(build_dir, "RepliBuildSysimage.so")
precompile_file = joinpath(build_dir, "precompile_jmake.jl")

# Build sysimage
println("📁 Project directory: $project_dir")
println("📝 Precompile file: $precompile_file")
println("💾 Output sysimage: $sysimage_path")
println("="^60)

try
    # Create sysimage with RepliBuild and all dependencies precompiled
    create_sysimage(
        [:RepliBuild, :TOML, :JSON, :Dates, :Libdl];
        sysimage_path=sysimage_path,
        precompile_execution_file=precompile_file,
        project=project_dir
    )

    println("\n" * "="^60)
    println("✅ Sysimage built successfully!")
    println("💾 Location: $sysimage_path")
    println("\nTo use the sysimage:")
    println("  julia -J $sysimage_path")
    println("\nOr create an alias:")
    println("  alias replibuild='julia -J $sysimage_path'")
    println("="^60)

    # Display size reduction info
    sysimage_size_mb = filesize(sysimage_path) / (1024 * 1024)
    println("\n📊 Sysimage size: $(round(sysimage_size_mb, digits=2)) MB")

catch e
    println("\n❌ Error building sysimage:")
    println(e)
    rethrow(e)
end
