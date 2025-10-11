# RepliBuild Build Guide - Real-World Solutions

---

**Still confused?** Run `RepliBuild.create_project_interactive()` and answer the questions!

```julia
julia> RepliBuild.create_project_interactive()
🧙 RepliBuild Project Wizard
============================================================

What type of project?
1. C++ Library → Julia bindings (most common)
2. C++ Executable (no Julia)
3. Library + Executable (multi-stage)
4. Import from CMake
5. Project with external libraries (sqlite, zlib, etc.)

Choice (1-5): 
```
---

## 🎯 Quick Start (Choose Your Path)

### Path 1: I'm New → Use Templates
```julia
using RepliBuild

# Interactive wizard (asks questions)
RepliBuild.create_project_interactive()

# Or choose a template directly
RepliBuild.available_templates()
RepliBuild.use_template("simple_lib", "my_project")
```

### Path 2: I Know What I'm Doing → Use API Directly
```julia
using RepliBuild

# Standard workflow
RepliBuild.init("my_project")
RepliBuild.discover("my_project")  # Auto-detect everything
RepliBuild.compile()  # Build it
```

---

## 🔧 Real Build Problems → Solutions

### Problem 1: "I have a CMake project with config.h"

**Solution:** Import CMake, RepliBuild handles config.h

```julia
using RepliBuild

# Import CMake (no need to run CMake!)
RepliBuild.import_cmake("path/to/CMakeLists.txt")

# Build
RepliBuild.compile()
```

**What happens:**
- Parses CMakeLists.txt directly
- Detects if config.h.in exists
- Generates config.h automatically
- Compiles with correct settings

---

### Problem 2: "My project uses external libraries (sqlite, zlib, curl)"

**Solution:** Auto-detection or manual config

#### Auto-Detection (Easy)
```julia
using RepliBuild, RepliBuild.BuildHelpers

# Scan for external libs
libs = BuildHelpers.detect_external_libraries("src")

# Shows what it found
for (name, info) in libs
    println("$name: $(info["available"] ? "✓ found" : "✗ missing")")
end

# Update config
RepliBuild.discover(".")  # Picks up detected libs
RepliBuild.compile()
```

#### Manual Config (replibuild.toml)
```toml
[compile]
libraries = ["sqlite3", "z", "curl"]  # Just the names
# RepliBuild uses pkg-config to find them automatically
```

---

### Problem 3: "Build .so first, then link executable" (MULTI-STAGE)

**Solution:** Two-stage build with proper linking

#### Template Approach (Easiest)
```julia
using RepliBuild

RepliBuild.use_template("lib_and_exe", "my_app")
# Creates build script that shows you how!
# Edit src/lib/* and src/app/*
# Run: julia build.jl
```

#### Manual Approach (Understanding the API)
```julia
using RepliBuild, RepliBuild.LLVMake

# ═══ STAGE 1: Build Library ═══
println("Building library...")

lib_compiler = LLVMJuliaCompiler("lib_config.toml")
# lib_config.toml points to src/lib/

lib_files = find_cpp_files("src/lib")
lib_ir = compile_to_ir(lib_compiler, lib_files)
final_lib_ir = optimize_and_link_ir(lib_compiler, lib_ir, "mylib")
lib_path = compile_ir_to_shared_lib(lib_compiler, final_lib_ir, "mylib")
# → Creates build/libmylib.so

# ═══ STAGE 2: Build Executable ═══
println("Building executable...")

exe_compiler = LLVMJuliaCompiler("exe_config.toml")
# exe_config.toml:
#   [compile]
#   lib_dirs = ["build"]
#   libraries = ["mylib"]  # Links against libmylib.so

exe_files = find_cpp_files("src/app")
exe_ir = compile_to_ir(exe_compiler, exe_files)
final_exe_ir = optimize_and_link_ir(exe_compiler, exe_ir, "myapp")

# Key: link_libs parameter tells it to link against the .so
exe_path = compile_ir_to_executable(exe_compiler, final_exe_ir, "myapp",
                                     link_libs=["mylib"])
# → Creates build/myapp (with rpath to find libmylib.so)

println("✅ Done! Run: ./build/myapp")
```

**Key Points:**
- Stage 1 creates `.so`
- Stage 2 links against it via `link_libs` parameter
- Automatic rpath so executable finds the `.so`

---

### Problem 4: "Undefined reference errors when linking"

**Solution:** Better error messages + manual fixes

RepliBuild now shows helpful hints:

```
💡 Linking Error - Undefined symbols:
  Check that all required libraries are specified in 'libraries = [...]'
  For multi-stage builds, use link_libs parameter

💡 Missing Library: libpthread
  1. Install: sudo apt install libpthread-stubs0-dev
  2. Or add path to lib_dirs in replibuild.toml
```

**Manual Fix:**
```toml
[compile]
libraries = ["pthread", "dl", "m"]  # Common system libs
lib_dirs = ["/usr/local/lib"]  # If in non-standard location
```

---

### Problem 5: "I just want to build an executable, no Julia"

**Solution:** Use compile_ir_to_executable

```julia
using RepliBuild, RepliBuild.LLVMake

# Load config
compiler = LLVMJuliaCompiler("replibuild.toml")

# Compile
cpp_files = find_cpp_files("src")
ir_files = compile_to_ir(compiler, cpp_files)
final_ir = optimize_and_link_ir(compiler, ir_files, "myapp")

# Create executable (not .so!)
exe = compile_ir_to_executable(compiler, final_ir, "myapp")

println("Run: ./build/myapp")
```

---

## 📚 Config Options Explained

### replibuild.toml Structure

```toml
[project]
name = "MyProject"

[paths]
source = "src"      # Where your .cpp files are
output = "julia"    # Where bindings go (or "build" for executables)
build = "build"     # Build artifacts

[compile]
include_dirs = ["include", "/usr/local/include"]  # Header search paths
lib_dirs = ["lib", "/usr/local/lib"]               # Library search paths
libraries = ["sqlite3", "pthread"]                 # Libraries to link

[compile.defines]
MY_DEFINE = "1"
VERSION = "1.0.0"

[bindings]
style = "simple"  # or "clangjl" for advanced bindings
```

---

## 🧰 BuildHelpers (The Smart Stuff)

### Auto-detect config.h
```julia
using RepliBuild.BuildHelpers

# Find template
template = BuildHelpers.detect_config_h_template(".")

if !isnothing(template)
    # Generate it
    BuildHelpers.generate_config_h(
        template,
        "include/config.h",
        Dict("VERSION" => "1.0.0", "HAVE_PTHREAD" => "1")
    )
end
```

### Check for pkg-config libraries
```julia
using RepliBuild.BuildHelpers

if BuildHelpers.pkg_config_exists("sqlite3")
    cflags, libs = BuildHelpers.get_pkg_config_flags("sqlite3")
    println("Compile flags: $cflags")
    println("Link flags: $libs")
end
```

### Auto-detect build type
```julia
using RepliBuild.BuildHelpers

reqs = BuildHelpers.auto_detect_build_requirements(".")
# Returns:
#   build_type: "executable" or "library"
#   external_libs: detected dependencies
#   config_template: path if found
```

---

## 🎓 Understanding the API

**Normal user flow:**
```julia
# 1. Create project
init("my_project") or use_template("simple_lib", "my_project")

# 2. Add your C++ code to src/

# 3. Discover (auto-configures)
discover("my_project")

# 4. Compile
compile()

# Done!
```

**Advanced user flow (executables, multi-stage):**
```julia
using RepliBuild.LLVMake

# Load compiler
compiler = LLVMJuliaCompiler("replibuild.toml")

# Compile to IR
files = find_cpp_files("src")
ir = compile_to_ir(compiler, files)
final_ir = optimize_and_link_ir(compiler, ir, "name")

# Choose output:
compile_ir_to_shared_lib(compiler, final_ir, "mylib")     # .so for Julia
compile_ir_to_executable(compiler, final_ir, "myapp")     # executable
```

---

## ❓ FAQ

**Q: Do I need to understand LLVM?**
No. Use templates or `discover()` + `compile()`.

**Q: What if I have a complex CMake project?**
`import_cmake()` handles it. No need to run CMake.

**Q: Can I build both library and executable?**
Yes! Use `lib_and_exe` template or manual multi-stage build (see Problem 3).

**Q: How do I debug linking errors?**
RepliBuild now shows helpful messages with suggestions. Check `libraries = []` in config.

**Q: Where's my config.h?**
Use `BuildHelpers.detect_config_h_template()` and `generate_config_h()`.

---

## 🚀 Common Patterns

### Pattern: System Library Project
```julia
# 1. Create from template
use_template("external_libs", "db_app")

# 2. Add code that uses sqlite3
# src/database.cpp with #include <sqlite3.h>

# 3. Discover (auto-detects sqlite3)
discover("db_app")

# 4. Compile (uses pkg-config)
compile()
```

### Pattern: Multi-file Executable
```julia
# 1. Use executable template
use_template("executable", "my_app")

# 2. Add multiple .cpp files to src/

# 3. Build
using RepliBuild.LLVMake
compiler = LLVMJuliaCompiler("my_app/replibuild.toml")
files = find_cpp_files("my_app/src")
ir = compile_to_ir(compiler, files)
final = optimize_and_link_ir(compiler, ir, "my_app")
compile_ir_to_executable(compiler, final, "my_app")
```

### Pattern: Library → Executable
See Problem 3 above. Use the template!

---

**Still confused?** Run `RepliBuild.create_project_interactive()` and answer the questions!
