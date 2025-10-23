#!/usr/bin/env julia
# Generate module templates for common libraries using pkg-config

using RepliBuild

# Common libraries available via pkg-config
common_libs = [
    # Core system libraries
    "zlib",
    "sqlite3",

    # Image libraries
    "libpng",
    "libpng16",
    "libjpeg",
    "libtiff-4",

    # Networking
    "libcurl",
    "libssl",
    "libcrypto",

    # XML/Text processing
    "libxml-2.0",
    "libxslt",

    # Graphics
    "freetype2",
    "fontconfig",
    "cairo",

    # Other common libs
    "libffi",
    "liblzma",
    "libbz2",
]

println("="^60)
println("Generating Common Library Modules")
println("="^60)
println()

let generated = 0, skipped = 0, failed = 0
    for lib in common_libs
        try
            if success(`pkg-config --exists $lib`)
                print("  📦 $lib...")

                # Get version
                version = try
                    readchomp(`pkg-config --modversion $lib`)
                catch
                    "unknown"
                end

                RepliBuild.generate_from_pkg_config(lib)
                println(" ✅ v$version")
                generated += 1
            else
                println("  ⚠️  $lib: not found via pkg-config")
                skipped += 1
            end
        catch e
            println("  ❌ $lib: error - $e")
            failed += 1
        end
    end

    println()
    println("="^60)
    println("Summary:")
    println("  Generated: $generated modules")
    println("  Skipped:   $skipped (not available)")
    println("  Failed:    $failed")
    println()
    println("Modules saved to:")
    println("  $(RepliBuild.get_replibuild_dir())/modules/")
    println("="^60)

    # List all available modules
    println()
    println("All available modules:")
    all_modules = RepliBuild.list_modules()
    for mod in all_modules
        if mod isa RepliBuild.ModuleRegistry.ModuleInfo
            println("  - $(mod.name) v$(mod.version)")
        else
            # Handle case where it might be a string path
            println("  - $mod")
        end
    end
end
