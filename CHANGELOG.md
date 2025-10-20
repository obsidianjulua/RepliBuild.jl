# RepliBuild Changelog

## v1.2.0 - Module System & Julia-Local Architecture (2025-10-20)

### Major Features

#### 🎯 Module System - Revolutionary C/C++ Library Integration
- **Module descriptors** (.toml files) that combine JLL packages + build logic
- **Template generator** for creating new modules (`create_module_template()`)
- **Built-in modules** for common libraries: Qt5, Boost, Eigen, Zlib
- **Auto-resolution** from JLL packages with system fallback
- **Component selection** (e.g., Boost::filesystem, Qt5::Widgets)
- **Version constraints** and validation
- **Flag management** (compile flags, link flags, defines per component)

#### 📁 Julia-Local Directory Structure
- All data in `~/.julia/replibuild/` (user-local, never system-wide)
- Project-local `.replibuild/` directories
- Follows Julia's DEPOT_PATH conventions
- Environment variable overrides (`JULIA_REPLIBUILD_DIR`, `REPLIBUILD_CACHE_DIR`)

#### ⚙️ Global Configuration System
- `~/.julia/replibuild/config.toml` for user preferences
- Cache management (size limits, cleanup policies)
- Build defaults (parallel jobs, optimization level)
- LLVM preferences (JLL vs system)
- Module search paths and registries

#### 🔍 Smart Path Resolution
Priority-based module search:
1. Project-local (`.replibuild/modules/`)
2. User modules (`~/.julia/replibuild/modules/`)
3. Registries (`~/.julia/replibuild/registries/`)
4. Built-in (`RepliBuild.jl/modules/`)

#### 🗂️ Build Cache Improvements
- Separate cache for tool paths (`.replibuild_cache/build_cache.toml`)
- LLVM tool discovery cached (no re-discovery every build)
- Module resolution cached
- Clean TOML files (runtime data moved to cache)

### API Changes

#### New Functions
```julia
# Path management
RepliBuild.print_paths_info()
RepliBuild.get_replibuild_dir()
RepliBuild.get_cache_dir()
RepliBuild.initialize_directories()

# Module system
RepliBuild.create_module_template("LibName")
RepliBuild.generate_from_pkg_config("sdl2")
RepliBuild.generate_from_cmake("Qt5")
RepliBuild.resolve_module("Qt5")
RepliBuild.list_modules()

# Configuration
RepliBuild.get_config_value("cache.max_size_gb")
RepliBuild.set_config_value("build.parallel_jobs", 16)
```

#### Updated Functions
- `save_config()` - Now separates user settings from runtime data
- `load_config()` - Auto-loads cached runtime data
- `init_toolchain()` - Uses cached tool paths when available

### Documentation

New documentation:
- `docs/MODULE_SYSTEM.md` - Complete module system design
- `docs/DIRECTORY_STRUCTURE.md` - Julia-local directory structure
- `docs/QUICK_START.md` - Getting started guide
- `modules/README.md` - Module usage and creation

### Breaking Changes

**Directory Structure:**
- Old: `~/.replibuild/`
- New: `~/.julia/replibuild/`
- Migration: Use `RepliBuild.migrate_old_structure()`

**TOML Format:**
- Runtime data (tool paths, discovery results) moved to cache
- `replibuild.toml` is now cleaner (user settings only)
- Old TOMLs still work, data automatically migrated to cache

### Performance Improvements

- **Tool discovery**: Cached after first run (60+ tools, ~70ms → 0ms)
- **Module resolution**: Cached lookups for repeated builds
- **TOML file size**: Reduced from 5KB+ to ~1.7KB

### Module Registry

Built-in modules shipped:
- **Qt5.toml** - Qt5 framework (8 components)
- **Boost.toml** - Boost libraries (11+ components)
- **Eigen.toml** - Linear algebra (header-only)
- **Zlib.toml** - Compression library

### Bug Fixes

- Fixed TOML pollution with discovered tool paths
- Fixed redundant tool discovery on every build
- Fixed module import warnings during precompilation
- Fixed cache directory permissions

### Developer Experience

**Before:**
```toml
# replibuild.toml (bloated)
[llvm.tools]
clang++ = "/usr/bin/clang++"
opt = "/usr/bin/opt"
llc = "/usr/bin/llc"
# ... 70+ more tools
```

**After:**
```toml
# replibuild.toml (clean)
[dependencies]
Qt5 = { components = ["Core", "Widgets"] }

# Tools auto-cached in .replibuild_cache/
```

### Migration Guide

**From v1.x to v1.2:**

1. **Directory migration** (automatic):
   ```julia
   using RepliBuild
   # Auto-migrates on first use
   ```

2. **Manual migration** (if needed):
   ```bash
   mv ~/.replibuild ~/.julia/replibuild
   ```

3. **Update project TOMLs** (optional):
   ```toml
   # Old: Large [llvm.tools] section
   # New: Auto-cached, just remove the section
   ```

4. **Test your builds**:
   ```julia
   using RepliBuild
   RepliBuild.build()
   ```

### Platform Support

- ✅ Linux (tested)
- ✅ macOS (should work)
- ⚠️  Windows (untested, may need path adjustments)

### Dependencies

No new dependencies added.

### Known Issues

- Precompilation warnings about undeclared imports (harmless, Julia issue)
- Windows path handling may need testing
- Custom build scripts need more examples

### Future Plans

- **Module registry** - Centralized registry like Julia's General
- **Build presets** - Common build configurations (Debug, Release, etc.)
- **Dependency tracking** - Smart rebuild on source changes
- **Binary caching** - Share compiled artifacts across projects
- **CI/CD integration** - Docker images, GitHub Actions

---

## v1.0 - Initial Release

- Basic LLVM toolchain management
- C++ to Julia compilation pipeline
- TOML-based configuration
- Project templates
- Error learning system

---

For detailed changes, see commit history and pull requests.
