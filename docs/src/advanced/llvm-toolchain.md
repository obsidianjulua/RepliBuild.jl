# LLVM Toolchain Management

RepliBuild uses LLVM/Clang for C++ compilation and analysis. This guide covers toolchain management.

## Overview

RepliBuild supports multiple LLVM sources:

1. **JLL packages** (recommended) - Managed by Julia
2. **System LLVM** - Installed via package manager
3. **Custom LLVM** - User-provided installation

## Toolchain Detection

### Automatic Detection

```julia
using RepliBuild

# Get current toolchain
toolchain = RepliBuild.get_toolchain()

println("LLVM version: ", toolchain[:version])
println("clang: ", toolchain[:clang])
println("clang++: ", toolchain[Symbol("clang++")])
```

Output:
```
LLVM version: 14.0.6
clang: /usr/bin/clang
clang++: /usr/bin/clang++
```

### Detection Order

1. JLL packages (if `use_jll = true`)
2. Environment variables (`LLVM_DIR`, `CC`, `CXX`)
3. System PATH
4. Common installation directories

### Print Toolchain Info

```julia
RepliBuild.print_toolchain_info()
```

Output:
```
LLVM Toolchain Information:
╔═══════════════════════════════════════╗
║  LLVM/Clang Toolchain                ║
╚═══════════════════════════════════════╝

Source: JLL Package (LLVM_full_assert_jll)
Version: 14.0.6

Tools:
  clang:       /path/to/julia/artifacts/.../bin/clang
  clang++:     /path/to/julia/artifacts/.../bin/clang++
  llvm-config: /path/to/julia/artifacts/.../bin/llvm-config
  llvm-ar:     /path/to/julia/artifacts/.../bin/llvm-ar
  llvm-nm:     /path/to/julia/artifacts/.../bin/llvm-nm

Status: ✅ Functional
```

## Using JLL Packages

### Configuration

In `replibuild.toml`:

```toml
[llvm]
use_jll = true
prefer_system = false  # Prefer JLL over system LLVM

[dependencies]
jll_packages = ["LLVM_full_assert_jll"]
```

### Benefits

- **Reproducible** - Same LLVM version everywhere
- **Cross-platform** - Works on Linux, macOS, Windows
- **No installation** - Automatically downloaded
- **Isolated** - Doesn't conflict with system LLVM

### Installation

```julia
using Pkg

# Install LLVM JLL
Pkg.add("LLVM_full_assert_jll")

# RepliBuild will use it automatically
```

## Using System LLVM

### Configuration

```toml
[llvm]
use_jll = false
prefer_system = true
llvm_dir = "/usr/lib/llvm-14"  # Optional: specific version

[llvm.search_paths]
paths = [
    "/usr/lib/llvm-14",
    "/usr/local/opt/llvm",
    "/opt/llvm"
]
```

### Install System LLVM

**Ubuntu/Debian:**
```bash
# Install specific version
sudo apt-get install llvm-14 clang-14

# Or latest
sudo apt-get install llvm clang
```

**macOS:**
```bash
# Via Homebrew
brew install llvm

# Add to PATH
export PATH="/usr/local/opt/llvm/bin:$PATH"
```

**Fedora/RHEL:**
```bash
sudo dnf install llvm clang
```

### Verify Installation

```bash
clang --version
clang++ --version
llvm-config --version
```

## Custom LLVM Installation

### Build LLVM from Source

```bash
# Download LLVM
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

# Configure
cmake -S llvm -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local/llvm-custom \
    -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra" \
    -DLLVM_ENABLE_RTTI=ON

# Build (takes a while!)
cmake --build build -j$(nproc)

# Install
sudo cmake --install build
```

### Configure RepliBuild

```toml
[llvm]
llvm_dir = "/usr/local/llvm-custom"
use_jll = false

[llvm.environment]
CC = "/usr/local/llvm-custom/bin/clang"
CXX = "/usr/local/llvm-custom/bin/clang++"
LLVM_CONFIG = "/usr/local/llvm-custom/bin/llvm-config"
```

## Environment Variables

### Override Toolchain

```bash
# Use specific clang
export CC=/usr/bin/clang-14
export CXX=/usr/bin/clang++-14

# Set LLVM directory
export LLVM_DIR=/usr/lib/llvm-14

# Run Julia
julia
```

```julia
# RepliBuild uses environment variables
using RepliBuild
RepliBuild.print_toolchain_info()
```

### Temporary Override

```julia
using RepliBuild

# Run with specific LLVM
RepliBuild.with_llvm_env() do
    ENV["CC"] = "/usr/bin/clang-14"
    ENV["CXX"] = "/usr/bin/clang++-14"

    RepliBuild.compile()
end
```

## Toolchain Verification

### Verify Functionality

```julia
using RepliBuild

# Test toolchain
try
    RepliBuild.verify_toolchain()
    println("✅ Toolchain verified successfully")
catch e
    println("❌ Toolchain verification failed:")
    println(e.msg)
end
```

Verification tests:
- Can compile C code
- Can compile C++ code
- Can link shared library
- Tools are correct version

### Manual Verification

```julia
using RepliBuild

toolchain = RepliBuild.get_toolchain()

# Test clang
run(`$(toolchain[:clang]) --version`)

# Test clang++
run(`$(toolchain[Symbol("clang++")]) --version`)

# Test compilation
mktemp() do path, io
    write(io, "int main() { return 0; }")
    close(io)

    run(`$(toolchain[Symbol("clang++")]) $path -o /tmp/test`)
    println("✅ Compilation test passed")
    rm("/tmp/test")
end
```

## LLVM Version Requirements

### Minimum Version

RepliBuild requires LLVM 10.0 or later.

### Recommended Version

- **LLVM 14.0+** for best compatibility
- **LLVM 15.0+** for C++20 support
- **LLVM 16.0+** for C++23 support

### Check Version

```julia
toolchain = RepliBuild.get_toolchain()
version = toolchain[:version]

println("LLVM version: $version")

# Parse version
major, minor, patch = split(version, '.')
major = parse(Int, major)

if major >= 14
    println("✅ Version is recommended")
elseif major >= 10
    println("⚠️  Version is minimum, upgrade recommended")
else
    println("❌ Version too old, please upgrade")
end
```

## Troubleshooting

### LLVM Not Found

**Error:**
```
ERROR: LLVM toolchain not found
```

**Solutions:**

1. Install LLVM JLL:
```julia
using Pkg
Pkg.add("LLVM_full_assert_jll")
```

2. Install system LLVM:
```bash
sudo apt-get install llvm clang
```

3. Set environment variable:
```bash
export LLVM_DIR=/usr/lib/llvm-14
```

### Wrong LLVM Version

**Error:**
```
ERROR: LLVM version 9.0.1 is too old (minimum: 10.0)
```

**Solutions:**

1. Use JLL (always current):
```julia
# In replibuild.toml
[llvm]
use_jll = true
```

2. Install newer version:
```bash
# Ubuntu
sudo apt-get install llvm-14 clang-14

# Update alternatives
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-14 100
```

### Multiple LLVM Installations

**Problem:** System has multiple LLVM versions

**Solution:** Specify exact version

```toml
[llvm]
llvm_dir = "/usr/lib/llvm-14"

[llvm.environment]
CC = "/usr/bin/clang-14"
CXX = "/usr/bin/clang++-14"
```

### JLL vs System Conflicts

**Problem:** JLL and system LLVM conflict

**Solution:** Choose one explicitly

```toml
# Use only JLL
[llvm]
use_jll = true
prefer_system = false

# Or use only system
[llvm]
use_jll = false
prefer_system = true
```

## Platform-Specific Notes

### Linux

**Default location:** `/usr/lib/llvm-<version>`

**Multiple versions:**
```bash
# Install multiple versions
sudo apt-get install llvm-12 llvm-13 llvm-14

# Switch with update-alternatives
sudo update-alternatives --config clang
```

### macOS

**Default location:** `/usr/local/opt/llvm` (Homebrew)

**Xcode CLT:** macOS includes Apple Clang, not LLVM

```bash
# Check which clang
which clang
# /usr/bin/clang (Apple Clang)

# Install LLVM
brew install llvm
# /usr/local/opt/llvm/bin/clang (LLVM Clang)

# Prefer LLVM
export PATH="/usr/local/opt/llvm/bin:$PATH"
```

### Windows

**Installation:**

1. Install Visual Studio with C++ support
2. Download LLVM from llvm.org
3. Or use LLVM JLL (recommended)

**Configuration:**
```toml
[llvm.windows]
llvm_dir = "C:/Program Files/LLVM"

[llvm.environment]
CC = "C:/Program Files/LLVM/bin/clang.exe"
CXX = "C:/Program Files/LLVM/bin/clang++.exe"
```

## Advanced Configuration

### Compiler Flags

```toml
[llvm]
# Default compiler flags
default_cxx_flags = [
    "-std=c++17",
    "-fPIC",
    "-Wall"
]

# Target triple
target_triple = "x86_64-unknown-linux-gnu"

# Optimization
default_optimization = "2"
```

### Resource Directories

```julia
using RepliBuild

toolchain = RepliBuild.get_toolchain()

# Get resource directory
resource_dir = readchomp(`$(toolchain[:clang]) -print-resource-dir`)
println("Resource dir: $resource_dir")

# Get include directories
includes = readchomp(`$(toolchain[:clang]) -E -x c++ - -v` |> devnull)
println("System includes: $includes")
```

### Cross-Compilation

```toml
[llvm]
target_triple = "aarch64-linux-gnu"
sysroot = "/usr/aarch64-linux-gnu"

[compilation]
cxx_flags = [
    "--target=aarch64-linux-gnu",
    "--sysroot=/usr/aarch64-linux-gnu"
]
```

## Performance Tuning

### Parallel Compilation

```toml
[compilation]
parallel = true
num_jobs = 0  # Use all cores
```

### LTO (Link-Time Optimization)

```toml
[compilation]
cxx_flags = ["-flto=thin"]
link_flags = ["-flto=thin"]
```

### PGO (Profile-Guided Optimization)

```bash
# Step 1: Build with profiling
# Add to replibuild.toml
[compilation]
cxx_flags = ["-fprofile-generate"]
link_flags = ["-fprofile-generate"]

# Step 2: Run representative workload
# (generates *.profraw files)

# Step 3: Merge profiles
llvm-profdata merge -output=default.profdata *.profraw

# Step 4: Rebuild with profile
[compilation]
cxx_flags = ["-fprofile-use=default.profdata"]
link_flags = ["-fprofile-use=default.profdata"]
```

## Next Steps

- **[Daemons](daemons.md)**: Daemon system
- **[Error Learning](error-learning.md)**: Error handling
- **[API Reference](../api/advanced.md)**: Advanced API
