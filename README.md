# RepliBuild.jl

**A Julia-native build system for C/C++ projects that actually integrates with the Julia ecosystem.**

[![Julia](https://img.shields.io/badge/Julia-1.9+-blue.svg)](https://julialang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Early%20Production-yellow.svg)](https://github.com/yourusername/RepliBuild.jl)

---

## What Is This?

RepliBuild is a build system that bridges the gap between Julia's JLL packages (prebuilt binaries) and real-world C/C++ build complexity. It's not a wrapper around CMake or Make - it's a native Julia build system that understands both worlds.

**The Problem:**
- Julia has JLL packages (e.g., `Qt5Base_jll`, `Boost_jll`) but no way to use them in builds
- Existing tools are just CMake wrappers or shell script generators
- No integration between Julia's reproducible artifacts and C/C++ build flags

**The Solution:**
- Module system that combines JLL packages with build logic
- User-local architecture (`~/.julia/replibuild/`)
- Smart caching (tool discovery, module resolution)
- TOML-based configuration

---

## Current Status: Early Production (v1.1)

**What Works:**
- ✅ Module system with JLL integration
- ✅ Built-in modules: Qt5, Boost, Eigen, Zlib
- ✅ Template generator for creating modules
- ✅ Smart caching (tool paths, module resolution)
- ✅ Julia-local directory structure
- ✅ Configuration management

**What's Experimental:**
- ⚠️  C++ to LLVM IR compilation pipeline
- ⚠️  Julia binding generation
- ⚠️  Error learning system

**What's Not Ready:**
- ❌ Module registry (planned for v1.2)
- ❌ CMake/pkg-config parsers (designed but not implemented)
- ❌ Windows support (untested)

**Be Honest:** This is a working prototype that solves real problems, but it's not battle-tested across hundreds of projects yet. The module system is the most mature part.

---

## Quick Start

### Installation

```julia
using Pkg
Pkg.add(url="https://github.com/obsidianjulua/RepliBuild.jl")
```

On first use, RepliBuild initializes its user-local directory:

```julia
julia> using RepliBuild
🔧 Initializing RepliBuild...
   📁 Creating ~/.julia/replibuild/
   📁 Creating modules/
   📁 Creating cache/
   ✅ RepliBuild ready
```

### Check Your Setup

```julia
julia> RepliBuild.print_paths_info()
======================================================================
RepliBuild Directory Structure
======================================================================

📁 Base directory: /home/user/.julia/replibuild
   Exists: true

📂 Subdirectories:
   Modules: /home/user/.julia/replibuild/modules
      0 module files
   Cache: /home/user/.julia/replibuild/cache
      ~0.0 GB
   ...

🔍 Module search paths:
   1. ✓ /home/user/.julia/replibuild/modules
   2. ✓ /path/to/RepliBuild.jl/modules
======================================================================
```

### List Available Modules

```julia
julia> RepliBuild.list_modules()
4-element Vector{String}:
 "Boost"
 "Eigen"
 "Qt5"
 "Zlib"
```

### Create Your First Module

```julia
julia> RepliBuild.create_module_template("SDL2")
🎨 Creating module template for: SDL2
✅ Module template created: /home/user/.julia/replibuild/modules/SDL2.toml

📝 Next steps:
   1. Edit /home/user/.julia/replibuild/modules/SDL2.toml
   2. Verify JLL package name: SDL2_jll
   3. Adjust component names and exports
   4. Add compiler/linker flags if needed
   5. Test with: ModuleRegistry.resolve_module("SDL2")
```

Edit the generated file:

```toml
# ~/.julia/replibuild/modules/SDL2.toml
[module]
name = "SDL2"
cmake_name = "SDL2"
description = "Simple DirectMedia Layer 2"

[jll]
package = "SDL2_jll"
auto_install = true

[components]
SDL2 = { jll_export = "libSDL2", required = true }

[flags]
compile = []
link = []
```

Test it:

```julia
julia> mod = RepliBuild.resolve_module("SDL2")
🔍 Resolving module: SDL2
  🔎 Searching Julia General registry for SDL2...
  ✓ Found JLL package in registry: SDL2_jll
  📦 Adding JLL package: SDL2_jll to project...
  ✓ JLL package installed: SDL2_jll
  ✓ Resolved via JLL: SDL2_jll
```

### Use in a Project

Create `replibuild.toml` in your project:

```toml
[project]
name = "MyApp"

[dependencies]
Qt5 = { components = ["Core", "Widgets"], version = ">=5.15" }
SDL2 = {}
```

Then (this part is experimental):

```julia
using RepliBuild

config = RepliBuild.init("myapp")
RepliBuild.build()  # Experimental - may need manual intervention
```

---

## Architecture

### Directory Structure

Everything is user-local, following Julia conventions:

```
~/.julia/replibuild/              # All RepliBuild data
├── config.toml                   # Your preferences
├── modules/                      # Your custom modules
│   ├── SDL2.toml
│   └── MyCompanyLib.toml
├── cache/                        # Build cache (auto-managed)
│   ├── toolchains/               # Cached tool discoveries
│   ├── modules/                  # Resolved module info
│   └── builds/                   # Custom build outputs
├── registries/                   # Module registries (future)
└── logs/                         # Build logs

# Project-local
myproject/
├── replibuild.toml               # Project config
└── .replibuild/                  # Project cache
    └── cache/
        └── build_cache.toml      # Tool paths, discovery results
```

### Module System

Modules are `.toml` files that describe how to use a library:

```toml
[module]
name = "Qt5"

[resolution]
prefer = "jll"                    # Try JLL first
fallback = ["system"]             # Fall back to system

[jll]
package = "Qt5Base_jll"
auto_install = true

[components]
Core = { jll_export = "libQt5Core", required = true }
Widgets = { jll_export = "libQt5Widgets", required = false }

[component_deps]
Widgets = ["Core"]                # Widgets needs Core

[flags]
compile = ["-fPIC"]
link = []
```

When you use a module, RepliBuild:
1. Searches for it (project → user → registry → builtin)
2. Resolves it (JLL → system → custom)
3. Validates components and versions
4. Applies flags
5. Caches the result

### Caching

RepliBuild caches expensive operations:

- **Tool discovery**: LLVM tools discovered once, cached in `.replibuild_cache/build_cache.toml`
- **Module resolution**: Resolved modules cached for instant lookup
- **Configuration**: Runtime data separated from user settings

Before caching:
```toml
# replibuild.toml was 5KB+ with 70+ tool paths
[llvm.tools]
clang++ = "/usr/bin/clang++"
opt = "/usr/bin/opt"
# ... 68 more lines
```

After caching:
```toml
# replibuild.toml is clean (1.7KB)
[llvm]
prefer_source = "jll"
# Tools auto-cached!
```

---

## Real-World Usage

### Example 1: Qt5 Project (Works)

```toml
# replibuild.toml
[dependencies]
Qt5 = { components = ["Core", "Widgets"] }
```

```julia
julia> using RepliBuild
julia> mod = RepliBuild.resolve_module("Qt5")
# Resolves Qt5Base_jll, extracts paths, applies flags
julia> println(mod.include_dirs)
# ["/home/user/.julia/artifacts/abc123/include"]
```

This part works reliably. What's experimental is the automatic compilation pipeline.

### Example 2: Custom Library

```julia
# Generate from pkg-config (if available on your system)
julia> RepliBuild.generate_from_pkg_config("opencv4")
📦 Discovered from pkg-config:
   Name: opencv4
   Version: 4.5.0
   CFLAGS: -I/usr/include/opencv4
   LIBS: -lopencv_core -lopencv_imgproc
✅ Module template created: /home/user/.julia/replibuild/modules/Opencv4.toml
```

### Example 3: Boost Project

```toml
[dependencies]
Boost = {
    components = ["system", "filesystem"],
    version = ">=1.75"
}
```

The module resolves correctly, gives you the right paths and flags. Compilation is where you might need to customize.

---

## What You Can Actually Do Today

**Reliable:**
1. ✅ Create module descriptors for libraries
2. ✅ Resolve JLL packages automatically
3. ✅ Get include directories and library paths
4. ✅ Cache tool discoveries
5. ✅ Manage user-local configuration

**Experimental:**
6. ⚠️  Full build pipeline (works but needs testing)
7. ⚠️  Julia binding generation (basic)
8. ⚠️  Error learning (needs more data)

**Recommended Workflow:**
1. Use RepliBuild for module management
2. Use the resolved paths in your own build scripts
3. Contribute modules to help build the registry
4. Test the compilation pipeline on simple projects

---

## Contributing

**The most valuable contribution right now: Building the module registry.**

### Priority: Module Registry

We need modules for common libraries. Here's how to help:

1. **Create modules for libraries you use:**

```julia
# For libraries with JLL packages
RepliBuild.create_module_template("FFTW")
# Edit the generated TOML
# Test: RepliBuild.resolve_module("FFTW")

# For system libraries
RepliBuild.generate_from_pkg_config("gtk+-3.0")
# Review and adjust the generated TOML
```

2. **Test existing modules:**

Create a real project using Qt5/Boost modules, report what works and what doesn't.

3. **Document edge cases:**

Found a library that needs special flags? Document it in the module file with comments.

### Roadmap for Contributors

**v1.2 - Registry Infrastructure**
- [ ] Registry.toml specification
- [ ] Module validation system
- [ ] Registry management commands
- [ ] CI/CD for module PRs

**v1.3 - Auto-Generation**
- [ ] CMakeLists.txt parser (extend existing CMakeParser.jl)
- [ ] Bulk JLL introspection
- [ ] pkg-config bulk import

**v1.4 - Module Registry Launch**
- [ ] GitHub repo: RepliBuild/Modules
- [ ] Auto-generate 100+ modules from Julia General
- [ ] Community contribution guidelines

### Development Setup

```bash
git clone https://github.com/yourusername/RepliBuild.jl
cd RepliBuild.jl

# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run tests (when available)
julia --project=. test/runtests.jl

# Test module system
julia --project=. -e '
    using RepliBuild
    RepliBuild.print_paths_info()
    println(RepliBuild.list_modules())
'
```

### Contributing Guidelines

**For Module Contributions:**
1. Create/test module locally
2. Document in module file (comments explaining special cases)
3. Validate with at least one real project
4. Submit PR with usage example

**For Code Contributions:**
1. Focus on module system improvements
2. Add tests for new features
3. Update documentation
4. Follow Julia style guide

**For Documentation:**
1. Real-world examples preferred
2. Be honest about limitations
3. Show actual commands and output

---

## Documentation

- **[Quick Start](docs/QUICK_START.md)** - Get running in 5 minutes
- **[Module System](docs/MODULE_SYSTEM.md)** - Complete module design
- **[Directory Structure](docs/DIRECTORY_STRUCTURE.md)** - Where everything lives
- **[Module Registry](docs/MODULE_REGISTRY.md)** - Future registry plans
- **[API Reference](docs/API.md)** - Complete API (when available)
- **[Changelog](CHANGELOG.md)** - Version history

---

## Comparison to Other Tools

**vs. CMake:**
- RepliBuild: Julia-native, JLL integration, module system
- CMake: Industry standard, mature, complex, no Julia integration

**vs. BinaryBuilder.jl:**
- RepliBuild: Build system for projects using JLLs
- BinaryBuilder: Creates JLL packages from source

**vs. Clang.jl:**
- RepliBuild: Full build system with module management
- Clang.jl: Binding generator (RepliBuild can use Clang.jl)

**vs. CxxWrap.jl:**
- RepliBuild: Build system + binding generation
- CxxWrap.jl: Runtime C++ wrapper (RepliBuild can target CxxWrap)

**Unique Value:** RepliBuild is the only system that bridges JLL packages with build logic. The module system is unique.

---

## Known Issues

1. **Precompilation warnings**: Harmless warnings about undeclared imports (Julia issue, not ours)
2. **Windows untested**: Path handling may need adjustments
3. **Build pipeline experimental**: Module resolution is solid, full compilation needs more testing
4. **No test suite yet**: Coming in v1.2
5. **Documentation gaps**: Some features documented before implementation

---

## FAQ

**Q: Is this production-ready?**
A: The module system is production-ready. The full build pipeline is experimental. Use it for module management, test the compilation features.

**Q: Do I need to abandon CMake?**
A: No! Use RepliBuild's module resolution to get paths, then use those in your CMake scripts if you want.

**Q: What's the relationship with JLL packages?**
A: JLL packages provide binaries. RepliBuild provides the build logic to use them. They complement each other.

**Q: How do I report bugs?**
A: Open an issue on GitHub with:
- What you tried (exact commands)
- What you expected
- What actually happened
- Your OS and Julia version

**Q: Can I use this in production?**
A: For module management: yes. For full automated builds: test thoroughly first. We're honest about maturity.

---

## License

MIT License - See [LICENSE](LICENSE) file

---

## Citation

If you use RepliBuild in research:

```bibtex
@software{replibuild2025,
  title = {RepliBuild.jl: A Julia-native build system for C/C++ integration},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/RepliBuild.jl}
}
```

---

## Acknowledgments

- Julia community for JLL packages and Artifacts.jl
- LLVM project for compiler infrastructure
- CMake and pkg-config for inspiration

---

**Status:** Early production, actively developed, contributions welcome.

**Focus:** Building the module registry is the top priority. Help us catalog the Julia C/C++ ecosystem!
