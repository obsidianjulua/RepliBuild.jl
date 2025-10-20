# RepliBuild Directory Structure

## Julia-Local Philosophy

RepliBuild follows Julia's local conventions.

## Directory Hierarchy

```
~/.julia/
├── replibuild/                          # RepliBuild user directory
│   ├── config.toml                      # Global RepliBuild settings
│   ├── modules/                         # User module registry
│   │   ├── Qt5.toml
│   │   ├── MyCustomLib.toml
│   │   └── ...
│   ├── cache/                           # Build artifacts cache
│   │   ├── toolchains/                  # Cached toolchain discoveries
│   │   │   ├── llvm_20_1_2.toml
│   │   │   └── gcc_13_2.toml
│   │   ├── modules/                     # Resolved module info
│   │   │   ├── qt5_5.15.2.toml
│   │   │   └── boost_1.76.0.toml
│   │   └── builds/                      # Custom build outputs
│   │       └── LLVM_17.0.6/
│   │           ├── bin/
│   │           ├── lib/
│   │           └── include/
│   ├── registries/                      # Module registries (like Pkg registries)
│   │   ├── General/                     # Default RepliBuild module registry
│   │   │   ├── Registry.toml
│   │   │   ├── Q/Qt5/
│   │   │   ├── B/Boost/
│   │   │   └── ...
│   │   └── MyCompany/                   # Private company registry
│   │       └── ...
│   └── logs/                            # Build logs and error learning
│       ├── error_patterns.toml
│       └── build_history.json
│
├── artifacts/                           # Julia's standard artifacts (JLLs live here)
│   └── ...                              # Managed by Pkg/Artifacts.jl
│
└── environments/                        # Standard Julia environments
    └── v1.10/                           # Your Julia version
        ├── Project.toml                 # Global environment
        └── Manifest.toml

# Project-local structure
/path/to/myproject/
├── Project.toml                         # Julia project file
├── replibuild.toml                      # RepliBuild project config
├── .replibuild/                         # Project-local RepliBuild data
│   ├── cache/                           # Project build cache
│   │   ├── build_cache.toml             # Runtime cache (tool paths, etc.)
│   │   ├── dependency_graph.json
│   │   └── compiled/                    # Compiled IR/binaries
│   ├── modules/                         # Project-specific module overrides
│   │   └── CustomQt5.toml               # Override global Qt5 module
│   └── logs/
│       └── last_build.log
├── src/                                 # C++ source
├── julia/                               # Generated Julia bindings
└── build/                               # Build outputs
    ├── ir/
    ├── linked/
    └── lib/
```

## Path Resolution Order

### Module Discovery
When `resolve_module("Qt5")` is called:

1. **Project-local**: `.replibuild/modules/Qt5.toml` (highest priority)
2. **User modules**: `~/.julia/replibuild/modules/Qt5.toml`
3. **Registered**: `~/.julia/replibuild/registries/*/Q/Qt5/Qt5.toml`
4. **Built-in**: `<RepliBuild.jl>/modules/Qt5.toml`

### Cache Resolution
When looking for cached toolchain info:

1. **Project cache**: `.replibuild/cache/build_cache.toml`
2. **User cache**: `~/.julia/replibuild/cache/toolchains/llvm_20.toml`
3. **Shared cache**: Can be configured in global `config.toml`

## Global Configuration

`~/.julia/replibuild/config.toml`:

```toml
version = "0.1.0"
last_updated = "2025-10-20"

[cache]
enabled = true
max_size_gb = 50  # Maximum cache size
cleanup_after_days = 90  # Clean builds older than this

[modules]
# Additional module search paths
search_paths = [
    "/company/shared/replibuild/modules",  # Company-wide modules
    "~/my_modules"
]

# Module registry URLs
registries = [
    "https://github.com/RepliBuild/Modules",  # Default registry
    "https://internal.company.com/modules"    # Private registry
]

[build]
# Default build settings
parallel_jobs = 8  # Number of parallel compilation jobs
default_optimization = "O2"
cache_ir = true  # Cache LLVM IR
cache_objects = true  # Cache object files

[llvm]
# LLVM preferences
prefer_source = "jll"  # "jll", "intree", "system"
isolated = true  # Isolate LLVM environment

[logging]
level = "info"  # "debug", "info", "warn", "error"
keep_logs = 100  # Keep last N build logs

[error_learning]
enabled = true
share_anonymous_errors = false  # Opt-in to share error patterns
```

## Environment Variables

RepliBuild respects these environment variables:

```bash
# Override RepliBuild directory
export JULIA_REPLIBUILD_DIR="$HOME/.julia/replibuild"

# Override cache directory (e.g., to use faster SSD)
export REPLIBUILD_CACHE_DIR="/mnt/fast_ssd/replibuild_cache"

# Module search paths (colon-separated)
export REPLIBUILD_MODULE_PATH="$HOME/modules:/shared/modules"

# Force specific LLVM
export REPLIBUILD_LLVM_ROOT="/opt/llvm-20"
```

## Disk Space Management

### Expected Sizes

| Directory | Typical Size | Notes |
|-----------|-------------|-------|
| `modules/` | < 1 MB | Just TOML files |
| `cache/toolchains/` | < 10 MB | Cached discovery results |
| `cache/modules/` | < 5 MB | Resolved module info |
| `cache/builds/` | 100 MB - 10 GB | Custom-built libraries |
| `logs/` | < 100 MB | Build logs and error patterns |
| **Total** | ~1-20 GB | Depends on custom builds |

### Cleanup Commands

```julia
using RepliBuild

# Clean old cache
RepliBuild.cleanup_cache(older_than_days=90)

# Clean specific build
RepliBuild.cleanup_build("LLVM_17.0.6")

# Full cache reset
RepliBuild.reset_cache(confirm=true)

# Check cache size
RepliBuild.cache_info()
# Output:
# Cache directory: ~/.julia/replibuild/cache
# Total size: 2.4 GB
# Builds: 3
# Toolchains cached: 2
# Modules cached: 15
```

## Migration from Old Structure

If you previously had `~/.replibuild/`, auto-migrate:

```julia
using RepliBuild

# Auto-detects old structure and migrates
RepliBuild.migrate_old_structure()
# Output:
# 🔄 Migrating from ~/.replibuild to ~/.julia/replibuild
#    ✓ Moved modules (5 files)
#    ✓ Moved cache (1.2 GB)
#    ✓ Updated config
# ✅ Migration complete
```

## Registry Structure (Future)

Similar to Julia's General registry:

```
~/.julia/replibuild/registries/General/
├── Registry.toml                        # Registry metadata
│   [name = "RepliBuild General"]
│   [repo = "https://github.com/RepliBuild/Modules.git"]
│
├── Q/
│   └── Qt5/
│       ├── Package.toml                 # Module metadata
│       │   [name = "Qt5"]
│       │   [uuid = "..."]
│       └── Versions.toml                # Available versions
│           ["5.15.2"]
│           git-tree-sha1 = "..."
│           ["6.2.0"]
│           git-tree-sha1 = "..."
│
├── B/
│   └── Boost/
│       ├── Package.toml
│       └── Versions.toml
│
└── ...
```

## Best Practices

### For Individual Users

1. **Use user modules** for personal overrides
   ```bash
   ~/.julia/replibuild/modules/MyQt5.toml
   ```

2. **Let cache auto-manage** - don't manually edit cache files

3. **Use project-local modules** for project-specific needs
   ```bash
   myproject/.replibuild/modules/CustomLib.toml
   ```

### For Organizations

1. **Set up shared registry**
   ```toml
   # ~/.julia/replibuild/config.toml
   [modules]
   registries = [
       "https://github.com/MyCompany/RepliBuildModules"
   ]
   ```

2. **Use shared cache for CI/CD**
   ```bash
   export REPLIBUILD_CACHE_DIR="/shared/ci/replibuild_cache"
   ```

3. **Provide company module templates**
   ```bash
   /company/shared/replibuild/modules/CompanyLib.toml
   ```

### For Package Developers

If you're creating a Julia package that uses RepliBuild:

```julia
# In your Package.jl
module MyPackage

using RepliBuild

# Package build script
function __init__()
    # Ensure RepliBuild is configured
    RepliBuild.ensure_initialized()

    # Build native code
    RepliBuild.build_if_needed(@__DIR__)
end

end
```

## Initialization

On first run, RepliBuild automatically creates:

```julia
julia> using RepliBuild
🔧 Initializing RepliBuild...
   📁 Creating ~/.julia/replibuild/
   📁 Creating ~/.julia/replibuild/modules/
   📁 Creating ~/.julia/replibuild/cache/
   📁 Creating ~/.julia/replibuild/registries/
   📝 Creating default config
   ✅ RepliBuild initialized
```

Manual initialization:

```julia
RepliBuild.initialize(
    cache_dir = "custom/path",  # Override cache location
    module_search_paths = ["/shared/modules"],
    registries = ["https://mycompany.com/modules"]
)
```

## Summary

**Key Points:**

1. ✅ **User-local**: Everything in `~/.julia/replibuild/`
2. ✅ **No root/system installs**: Pure Julia conventions
3. ✅ **Project isolation**: `.replibuild/` in each project
4. ✅ **Registry-based**: Like Pkg, but for build modules
5. ✅ **Configurable**: Global config + environment variables
6. ✅ **Cache management**: Automatic cleanup and size limits

This structure follows Julia's philosophy while providing the build system infrastructure you need.
