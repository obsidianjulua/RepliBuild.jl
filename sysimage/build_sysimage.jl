#!/usr/bin/env julia
# Build custom sysimage for RepliBuild
# This creates a compiled sysimage with RepliBuild and all core modules precompiled
# for instant daemon startup and faster build operations

using Pkg

# Ensure PackageCompiler is installed
if !haskey(Pkg.project().dependencies, "PackageCompiler")
    println("📦 Installing PackageCompiler.jl...")
    Pkg.add("PackageCompiler")
end

using PackageCompiler

println("="^70)
println("🚀 Building RepliBuild Custom Sysimage")
println("="^70)

# Configuration
build_dir = @__DIR__
project_dir = dirname(build_dir)  # Parent directory (RepliBuild root)
sysimage_path = joinpath(build_dir, "RepliBuildSysimage.so")
precompile_file = joinpath(build_dir, "precompile_jmake.jl")

# Display configuration
println("\n📋 Configuration:")
println("   Project:      $project_dir")
println("   Precompile:   $precompile_file")
println("   Output:       $sysimage_path")
println()

# Check that precompile file exists
if !isfile(precompile_file)
    error("Precompile file not found: $precompile_file")
end

println("📦 Modules to precompile:")
modules_to_include = [
    :RepliBuild,
    :TOML,           # Configuration parsing
    :JSON,           # Metadata export
    :Dates,          # Timestamps
    :Libdl,          # Dynamic library loading
    :Distributed,    # Parallel compilation
    :SQLite,         # Error learning database
]

for mod in modules_to_include
    println("   • $mod")
end

println("\n" * "="^70)
println("⏳ Building sysimage (this may take several minutes)...")
println("="^70)

try
    # Create sysimage with RepliBuild and all dependencies precompiled
    create_sysimage(
        modules_to_include;
        sysimage_path=sysimage_path,
        precompile_execution_file=precompile_file,
        project=project_dir
    )

    println("\n" * "="^70)
    println("✅ SYSIMAGE BUILD SUCCESSFUL!")
    println("="^70)

    # Display file info
    sysimage_size_mb = filesize(sysimage_path) / (1024 * 1024)
    println("\n📊 Sysimage Info:")
    println("   Location:  $sysimage_path")
    println("   Size:      $(round(sysimage_size_mb, digits=2)) MB")

    println("\n🚀 Usage:")
    println("\n1. Direct usage:")
    println("   julia -J $sysimage_path")

    println("\n2. Create a shell alias:")
    println("   alias replibuild='julia -J $sysimage_path --project=$project_dir'")

    println("\n3. Start daemons with sysimage:")
    println("   julia -J $sysimage_path --project=$project_dir daemons/servers/compilation_daemon.jl")

    println("\n💡 Benefits:")
    println("   • Instant startup (no compilation lag)")
    println("   • Faster daemon initialization")
    println("   • All core modules precompiled")
    println("   • LLVM toolchain pre-initialized")

    println("\n" * "="^70)

catch e
    println("\n" * "="^70)
    println("❌ SYSIMAGE BUILD FAILED")
    println("="^70)
    println("\nError: $e")

    if isa(e, ErrorException)
        println("\n💡 Troubleshooting:")
        println("   1. Ensure RepliBuild is properly installed")
        println("   2. Check that all dependencies are available")
        println("   3. Try: julia --project=$project_dir -e 'using RepliBuild'")
        println("   4. Review the precompile script: $precompile_file")
    end

    rethrow(e)
end
