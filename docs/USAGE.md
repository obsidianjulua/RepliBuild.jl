# RepliBuild Usage Guide

**Complete reference for RepliBuild v2.0**

---

## Quick Links

- [QUICKSTART.md](QUICKSTART.md) - Fast 3-step tutorial
- [examples/](examples/) - Real working examples
- [replibuild.toml.example](replibuild.toml.example) - Complete config reference
- [REAL_OUTPUT.md](REAL_OUTPUT.md) - See actual generated wrappers

---

## The 3-Command Workflow

RepliBuild has a simple, clear API:

### Step 0: `Discovery.discover()` - Create Config (One Time)

```julia
using RepliBuild

# Auto-scan your C++ project and generate replibuild.toml
RepliBuild.Discovery.discover()

# If config already exists, force regeneration:
RepliBuild.Discovery.discover(force=true)
```

**What it does:**
- Scans for C++ files (`*.cpp`, `*.cc`, `*.cxx`)
- Detects include directories
- Infers project structure
- Generates `replibuild.toml` configuration file

**Output:** `replibuild.toml` in your project root

**When to use:**
- First time using RepliBuild in a project
- Want to regenerate config after major project changes
- Need to update config with new source files

---

### Step 1: `build()` - Compile C++ → Library

```julia
using RepliBuild

# Compile your C++ code to a shared library
RepliBuild.build()

# Or specify project path
RepliBuild.build("/path/to/project")
```

**What it does:**
- Reads `replibuild.toml` configuration
- Compiles C++ → LLVM IR → Optimized binary
- Generates shared library (`.so`/`.dylib`/`.dll`)
- Extracts DWARF debug information
- Extracts symbol table
- Saves metadata to `julia/compilation_metadata.json`

**Output:**
- `julia/libproject.so` (or `.dylib`, `.dll`)
- `julia/compilation_metadata.json`

**What it does NOT do:**
- Does NOT create `replibuild.toml` (use `Discovery.discover()`)
- Does NOT generate Julia wrappers (use `wrap()`)

**Requirements:**
- `replibuild.toml` must exist in project root
- LLVM toolchain must be available
- Source files must compile without errors

---

### Step 2: `wrap()` - Generate Julia Wrappers

```julia
# Generate Julia wrapper from compiled library
RepliBuild.wrap()

# Or specify project path
RepliBuild.wrap("/path/to/project")

# With additional C++ headers for better type information
RepliBuild.wrap(headers=["include/mylib.h", "include/types.h"])
```

**What it does:**
- Reads `julia/compilation_metadata.json`
- Extracts struct definitions from DWARF
- Extracts function signatures from symbols
- Generates Julia module with:
  - `mutable struct` definitions (from DWARF)
  - `ccall` wrapper functions
  - Complete documentation
  - Type-safe signatures
- Saves to `julia/` directory

**Output:**
- `julia/Projectname.jl` (Julia module ready to use)

**Requirements:**
- Must run `build()` first
- `julia/compilation_metadata.json` must exist
- Compile flags must include `-g` for struct extraction

**Wrapper Quality:**
- **Without `-g`**: Basic wrappers (functions only, ~40% quality)
- **With `-g`**: Introspective wrappers (functions + structs, ~95% quality) ⭐ Recommended

---

### Step 3: `info()` - Check Project Status

```julia
# Check what's built
RepliBuild.info()

# Or specify project path
RepliBuild.info("/path/to/project")
```

**Shows:**
- Project name
- ✓ Library built? (path shown)
- ✓ Wrapper generated? (path shown)
- ❌ What's missing (with instructions)

**Example output:**
```
Project: mathlib

✓ Library: julia/libmathlib.so
✓ Wrapper: julia/Mathlib.jl

Usage:
  include("julia/Mathlib.jl")
  using .Mathlib
```

---

## Complete Workflow Example

```julia
using RepliBuild

# Step 0: First time setup (creates replibuild.toml)
RepliBuild.Discovery.discover()

# Step 1: Compile C++ code
RepliBuild.build()

# Step 2: Generate Julia wrapper
RepliBuild.wrap()

# Step 3: Check status
RepliBuild.info()

# Step 4: Use the wrapper!
include("julia/Myproject.jl")
using .Myproject

# Call your C++ functions from Julia
result = my_cpp_function(42)
```

---

## Configuration: `replibuild.toml`

### Minimal Configuration

```toml
[project]
name = "mylib"
root = "."

[compile]
source_files = ["src/main.cpp"]
flags = ["-std=c++17", "-fPIC", "-g"]

[binary]
type = "library"
```

### Complete Configuration

See [replibuild.toml.example](replibuild.toml.example) for all available options including:
- Preprocessor defines
- Include directories
- Link libraries
- LLVM toolchain settings
- Cache settings
- Wrapper generation options
- And much more...

---

## Utility Functions

### `clean()` - Remove Build Artifacts

```julia
# Clean build artifacts (keeps source)
RepliBuild.clean()

# Or specify project path
RepliBuild.clean("/path/to/project")
```

**What it removes:**
- `julia/` directory (libraries, wrappers, metadata)
- Build cache
- Intermediate files

**What it keeps:**
- Source code
- `replibuild.toml`
- Your project files

---

## Common Workflows

### Workflow 1: Fresh Project

```julia
using RepliBuild

# First time: Discover → Build → Wrap
RepliBuild.Discovery.discover()  # Creates config
RepliBuild.build()               # Compiles library
RepliBuild.wrap()                # Generates wrapper

# Use it
include("julia/Myproject.jl")
using .Myproject
```

### Workflow 2: Existing Project

```julia
using RepliBuild

# Already have replibuild.toml
RepliBuild.build()
RepliBuild.wrap()
```

### Workflow 3: After Code Changes

```julia
using RepliBuild

# Changed C++ code? Just rebuild
RepliBuild.build()  # Recompiles
RepliBuild.wrap()   # Regenerates wrapper
```

### Workflow 4: Clean Build

```julia
using RepliBuild

# Start fresh
RepliBuild.clean()   # Remove old artifacts
RepliBuild.build()   # Fresh compile
RepliBuild.wrap()    # Fresh wrapper
```

### Workflow 5: Force Config Regeneration

```julia
using RepliBuild

# Added new source files? Regenerate config
RepliBuild.Discovery.discover(force=true)
RepliBuild.build()
RepliBuild.wrap()
```

---

## Important Configuration Options

### Enable DWARF Struct Extraction (Recommended)

```toml
[compile]
flags = [
    "-std=c++17",
    "-fPIC",       # Required for shared libraries
    "-g",          # ⭐ REQUIRED for struct extraction
    "-O2"          # Optimization level
]
```

**Without `-g`**: Functions only, no struct definitions (~40% quality)
**With `-g`**: Functions + structs with members (~95% quality) ⭐

### Wrapper Generation Settings

```toml
[wrap]
enabled = true
style = "introspective"  # Options: "basic", "advanced", "introspective"
module_name = "MyProject"
headers = ["include/myproject.h"]
generate_tests = false
generate_docs = true
```

### Binary Type

```toml
[binary]
type = "library"  # Options: "library", "shared", "executable"
```

**Use "library" or "shared" for Julia FFI** (executable generates binary without wrappers)

---

## Troubleshooting

### Problem: "No replibuild.toml found"

**Solution:**
```julia
RepliBuild.Discovery.discover()
```

### Problem: "Library not found"

**Solution:** Run `build()` before `wrap()`:
```julia
RepliBuild.build()
RepliBuild.wrap()
```

### Problem: "No metadata found" or "No struct definitions"

**Solution:** Add `-g` flag to compile flags in `replibuild.toml`:
```toml
[compile]
flags = ["-std=c++17", "-fPIC", "-g"]  # Add -g
```

Then rebuild:
```julia
RepliBuild.build()
RepliBuild.wrap()
```

### Problem: "Functions crash or return garbage"

**Cause:** Using non-standard-layout types (virtual methods, inheritance, STL containers)

**Solution:** RepliBuild only supports:
- Standard-layout C structs
- Trivially-copyable C++ structs (POD types)
- Functions with C-compatible signatures

See [LIMITATIONS.md](../LIMITATIONS.md) for details.

### Problem: "Compilation failed"

**Check:**
1. Source files exist: `ls src/`
2. Paths are correct in `replibuild.toml`
3. Code compiles manually: `clang++ -c src/myfile.cpp`
4. LLVM toolchain installed: `clang++ --version`

### Problem: "Wrapper loads but functions not found"

**Cause:** Name mangling (C++ functions without `extern "C"`)

**Solution:** Use `extern "C"` for exported functions:
```cpp
extern "C" {
    int my_function(int a, int b) {
        return a + b;
    }
}
```

Or check mangled name:
```bash
nm -g julia/libproject.so | grep my_function
```

---

## Tips & Best Practices

### ✅ Do This

1. **Use `-g` flag** for DWARF extraction (struct definitions)
2. **Use `-fPIC` flag** for shared libraries
3. **Use `extern "C"`** for C++ functions you want to call
4. **Keep structs simple** (POD types only)
5. **Run `info()` frequently** to check status
6. **Use `Discovery.discover()`** instead of writing config manually

### ❌ Don't Do This

1. ❌ Skip `-g` flag (won't get struct definitions)
2. ❌ Skip `-fPIC` flag (library won't build correctly)
3. ❌ Use virtual methods (not supported)
4. ❌ Use inheritance (not supported)
5. ❌ Use STL containers (ABI-unstable, use C arrays/pointers)
6. ❌ Forget to run `build()` before `wrap()`

---

## Advanced Usage (Power Users)

### Direct Module Access

If you need low-level control:

```julia
using RepliBuild

# Discovery module
RepliBuild.Discovery.discover(".", force=true)
RepliBuild.Discovery.find_cpp_files("src/")

# Compiler module
config = RepliBuild.ConfigurationManager.load_config("replibuild.toml")
RepliBuild.Compiler.compile_project(config)

# Wrapper module
RepliBuild.Wrapper.wrap_library(config, library_path)

# ConfigurationManager
RepliBuild.ConfigurationManager.get_module_name(config)
RepliBuild.ConfigurationManager.get_library_name(config)
```

**Note:** 99% of users should just use `build()` + `wrap()`. Only use direct module access if you know what you're doing.

---

## Real Examples

Want to see RepliBuild in action?

- [examples/01_simple_math.md](examples/01_simple_math.md) - 5 min tutorial
- [examples/02_structs_and_classes.md](examples/02_structs_and_classes.md) - 10 min DWARF tutorial
- [examples/StructTest.jl](examples/StructTest.jl) - Real generated wrapper (111 lines)
- [REAL_OUTPUT.md](REAL_OUTPUT.md) - Showcase of actual output

---

## Quick Reference

| Command | What it does | Requirements |
|---------|--------------|--------------|
| `Discovery.discover()` | Create `replibuild.toml` | C++ project directory |
| `build()` | Compile C++ → library | `replibuild.toml` exists |
| `wrap()` | Generate Julia wrapper | `build()` completed first |
| `info()` | Show project status | None |
| `clean()` | Remove build artifacts | None |

**Typical workflow:**
```julia
Discovery.discover() → build() → wrap() → include() + using
```

---

## Need More Help?

- **Quick start**: [QUICKSTART.md](QUICKSTART.md)
- **Examples**: [examples/](examples/)
- **Config reference**: [replibuild.toml.example](replibuild.toml.example)
- **What's supported**: [../LIMITATIONS.md](../LIMITATIONS.md)
- **How it works**: [../ARCHITECTURE.md](../ARCHITECTURE.md)
- **Main docs**: [../README.md](../README.md)

---

**That's everything! Go from C++ to Julia in 3 commands.** 🚀
