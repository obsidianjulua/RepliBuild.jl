# RepliBuild Quick Start

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/obsidianjulua/RepliBuild.jl")
```

## First Run

On first use, RepliBuild automatically sets up its directory structure:

```julia
julia> using RepliBuild
🔧 Initializing RepliBuild...
   📁 Creating ~/.julia/replibuild/
   📁 Creating modules/
   📁 Creating cache/
   📁 Creating registries/
   📁 Creating logs/
   📝 Creating default config
   ✅ RepliBuild initialized
```

## Directory Structure

Everything is user-local (never system-wide):

```
~/.julia/replibuild/          # All RepliBuild data
├── config.toml               # Your preferences
├── modules/                  # Your custom modules
├── cache/                    # Build cache (auto-managed)
├── registries/               # Module registries
└── logs/                     # Build logs
```

## Quick Commands

### Check Your Setup

```julia
using RepliBuild

# Show directory structure
RepliBuild.print_paths_info()

# Output:
# ======================================================================
# RepliBuild Directory Structure
# ======================================================================
#
# 📁 Base directory: /home/user/.julia/replibuild
#    Exists: true
#
# 📂 Subdirectories:
#    Modules: /home/user/.julia/replibuild/modules
#       4 module files
#    Cache: /home/user/.julia/replibuild/cache
#       ~0.0 GB
#    ...
```

### Create a Module

```julia
# Generate module template
RepliBuild.create_module_template("MyLibrary")
# Output: ~/.julia/replibuild/modules/MyLibrary.toml

# From pkg-config
RepliBuild.generate_from_pkg_config("sdl2")

# From CMake
RepliBuild.generate_from_cmake("OpenCV")
```

### Use Modules in Projects

Create `replibuild.toml` in your project:

```toml
[project]
name = "MyApp"

[dependencies]
Qt5 = { components = ["Core", "Widgets"], version = ">=5.15" }
Boost = { components = ["system", "filesystem"] }
```

Then build:

```julia
using RepliBuild

# Initialize project
config = RepliBuild.init("myapp")

# Modules are auto-resolved
RepliBuild.build()
```

## Configuration

Edit `~/.julia/replibuild/config.toml`:

```toml
[cache]
max_size_gb = 50
cleanup_after_days = 90

[build]
parallel_jobs = 8
default_optimization = "O2"

[llvm]
prefer_source = "jll"  # or "system", "intree"
```

Or use environment variables:

```bash
# Override RepliBuild directory
export JULIA_REPLIBUILD_DIR="$HOME/.julia/replibuild"

# Override cache (e.g., to use fast SSD)
export REPLIBUILD_CACHE_DIR="/mnt/ssd/replibuild_cache"

# Additional module paths
export REPLIBUILD_MODULE_PATH="$HOME/modules:/shared/modules"
```

## Built-in Modules

RepliBuild ships with modules for common libraries:

```julia
# List available modules
RepliBuild.list_modules()

# Resolve a module
qt5 = RepliBuild.resolve_module("Qt5")
println(qt5.include_dirs)
println(qt5.libraries)
```

Available modules:
- `Qt5` - Qt5 framework
- `Boost` - Boost C++ libraries
- `Eigen` - Linear algebra (header-only)
- `Zlib` - Compression library

## Next Steps

- Read [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md) for full path details
- Read [MODULE_SYSTEM.md](MODULE_SYSTEM.md) for module system design
- Check [modules/README.md](../modules/README.md) for module examples
- Create your own modules for your projects!

## Troubleshooting

**Modules not found?**
```julia
# Check search paths
RepliBuild.get_module_search_paths()
```

**Cache too large?**
```julia
# Check size
RepliBuild.cache_info()

# Clean old builds
RepliBuild.cleanup_cache(older_than_days=90)
```

**Want to reset everything?**
```julia
# Full reset
RepliBuild.reset_cache(confirm=true)

# Or manually
rm -rf ~/.julia/replibuild/cache
```

## Philosophy

RepliBuild follows Julia's user-local conventions:

✅ Everything in `~/.julia/replibuild/` (user-local)
✅ Project-specific data in `.replibuild/` (version-controlled if needed)
✅ No system-wide installations
✅ Leverages Julia's Pkg and Artifacts ecosystem
✅ Reproducible builds with JLL packages

