# RepliBuild v1.1 Release Notes

## 🚀 Major Release: Module System & Julia-Local Architecture

**Release Date:** 2025-10-20
**Version:** 1.1.0
**Focus:** Revolutionary module system for C/C++ library integration

---

## 🎯 What's New

### 1. Module System - The Missing Bridge

RepliBuild now provides what Julia's C/C++ ecosystem has been missing: **build logic that works with JLL packages**.

**Before:**
```julia
# JLL packages = just binaries, no build integration
using Qt5Base_jll  # Now what? How do I use this in a build?
```

**After:**
```toml
# replibuild.toml
[dependencies]
Qt5 = { components = ["Core", "Widgets"], version = ">=5.15" }
```
```julia
# Just works - auto-resolves, validates, applies flags
RepliBuild.build()
```

### 2. Julia-Local Everything

All data now lives in `~/.julia/replibuild/` following Julia's user-local philosophy:

```
~/.julia/replibuild/
├── config.toml          # Your preferences
├── modules/             # Your modules
├── cache/               # Build cache
├── registries/          # Module registries
└── logs/                # Build logs
```

### 3. Smart Caching

- **Tool discovery cached** - No re-scanning 70+ LLVM tools every build
- **Module resolution cached** - Instant lookups for repeated builds
- **Clean TOMLs** - Runtime data moved to `.replibuild_cache/`

### 4. Built-in Modules

Ships with modules for common libraries:
- **Qt5** - 8 components (Core, Widgets, Network, Gui, Sql, Xml, DBus, Concurrent)
- **Boost** - 11+ components (system, filesystem, thread, regex, etc.)
- **Eigen** - Header-only linear algebra
- **Zlib** - Compression library

---

## ✨ Key Features

### Module Descriptors

`.toml` files that combine JLL packages with build logic:

```toml
[module]
name = "Qt5"
cmake_name = "Qt5"

[jll]
package = "Qt5Base_jll"
auto_install = true

[components]
Core = { jll_export = "libQt5Core", required = true }
Widgets = { jll_export = "libQt5Widgets", required = false }

[flags]
compile = ["-fPIC"]
link = []
```

### Module Template Generator

Create modules in seconds:

```julia
# From scratch
RepliBuild.create_module_template("MyLib")

# From pkg-config
RepliBuild.generate_from_pkg_config("sdl2")

# From CMake
RepliBuild.generate_from_cmake("OpenCV")
```

### Smart Resolution

Priority-based module search:
1. Project-local (`.replibuild/modules/`)
2. User modules (`~/.julia/replibuild/modules/`)
3. Registries (`~/.julia/replibuild/registries/`)
4. Built-in (`RepliBuild.jl/modules/`)

---

## 📚 New API

```julia
using RepliBuild

# Path management
RepliBuild.print_paths_info()
RepliBuild.get_replibuild_dir()
RepliBuild.get_cache_dir()

# Module system
RepliBuild.create_module_template("LibName")
RepliBuild.resolve_module("Qt5")
RepliBuild.list_modules()

# Configuration
RepliBuild.get_config_value("cache.max_size_gb")
RepliBuild.set_config_value("build.parallel_jobs", 16)
```

---

## 🔧 Breaking Changes

### Directory Migration

**Old:** `~/.replibuild/`
**New:** `~/.julia/replibuild/`

**Migration:** Automatic on first use, or run:
```julia
RepliBuild.migrate_old_structure()
```

### TOML Format

Runtime data (tool paths, discovery results) moved to cache.

**Old:**
```toml
[llvm.tools]
clang++ = "/usr/bin/clang++"
opt = "/usr/bin/opt"
# ... 70+ more lines
```

**New:**
```toml
[llvm]
prefer_source = "jll"
# Tools auto-cached in .replibuild_cache/
```

---

## 📖 Documentation

### New Docs
- `docs/MODULE_SYSTEM.md` - Complete module system guide
- `docs/DIRECTORY_STRUCTURE.md` - Directory layout
- `docs/QUICK_START.md` - Getting started
- `docs/MODULE_REGISTRY.md` - Future registry plans
- `modules/README.md` - Module usage guide

### Updated
- `CHANGELOG.md` - Complete version history
- `.gitignore` - Proper exclusions for cache/build files

---

## 🚀 Performance

- **Tool discovery:** Cached (~70 tools, 60ms → 0ms after first run)
- **Module resolution:** Instant for repeated builds
- **TOML size:** Reduced from 5KB+ to ~1.7KB

---

## 🛠️ Under the Hood

### New Modules
- `src/RepliBuildPaths.jl` - Path management
- `src/ModuleTemplateGenerator.jl` - Template generator

### Updated Modules
- `src/ConfigurationManager.jl` - Caching system
- `src/ModuleRegistry.jl` - Julia-local paths
- `src/LLVMEnvironment.jl` - Cached tool usage

---

## 🎯 Use Cases

### 1. Qt5 Application

```toml
# replibuild.toml
[dependencies]
Qt5 = { components = ["Core", "Widgets"] }
```

```julia
using RepliBuild
config = RepliBuild.init("myqtapp")
RepliBuild.build()
# Qt5Base_jll auto-resolved, configured, ready to build
```

### 2. Boost Project

```toml
[dependencies]
Boost = {
    components = ["system", "filesystem"],
    version = ">=1.75"
}
```

### 3. Custom Module

```julia
# Create module for your company's library
RepliBuild.create_module_template("CompanyLib")
# Edit ~/.julia/replibuild/modules/CompanyLib.toml
# Use in all projects instantly
```

---

## 🔮 Roadmap

### v1.2 - Registry Infrastructure
- Registry.toml specification
- Registry management commands
- Module validation system

### v1.3 - Auto-Generation
- CMakeLists.txt parser
- pkg-config bulk import
- JLL package introspection

### v1.4 - Community Registry
- Official RepliBuild Modules registry
- 500+ auto-generated modules from Julia General
- Community contributions

---

## 🐛 Known Issues

1. **Precompilation warnings** - Harmless warnings about undeclared imports
2. **Windows support** - Path handling untested on Windows
3. **Module versioning** - Version constraints not enforced yet (coming in v1.2)

---

## 📦 Installation

```julia
using Pkg
Pkg.add(url="https://github.com/yourusername/RepliBuild.jl")
```

First run automatically initializes:
```julia
using RepliBuild
# 🔧 Initializing RepliBuild...
#    📁 Creating ~/.julia/replibuild/
#    ✅ RepliBuild ready
```

---

## 🙏 Credits

This release represents a fundamental shift in how Julia integrates with C/C++ build systems. The module system bridges years of missing build logic in the Julia ecosystem.

Special thanks to:
- Julia community for JLL packages
- CMake and pkg-config for inspiration
- All early testers and contributors

---

## 🚀 Get Started

```julia
using RepliBuild

# Check your setup
RepliBuild.print_paths_info()

# Create a module
RepliBuild.create_module_template("MyLib")

# Build a project
config = RepliBuild.init("myproject")
RepliBuild.build()
```

**Welcome to the future of C/C++ integration in Julia!**

---

For full changelog, see [CHANGELOG.md](CHANGELOG.md)
For documentation, see [docs/](docs/)
For examples, see [examples/](examples/)
