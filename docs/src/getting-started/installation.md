# Installation

## From Julia Package Registry

```julia
using Pkg
Pkg.add("RepliBuild")
```

## From GitHub

```julia
using Pkg
Pkg.add(url="https://github.com/obsidianjulua/RepliBuild.jl")
```

## Development Install

```bash
git clone https://github.com/obsidianjulua/RepliBuild.jl
cd RepliBuild.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## System Dependencies

RepliBuild automatically manages LLVM/Clang through JLL packages, but some workflows require system tools:

### For C++ Compilation
- **Automatically managed**: LLVM/Clang via `LLVM_full_assert_jll`
- **Optional**: System C++ compiler for testing

### For Binary Wrapping
- **Linux**: `nm` and `objdump` (from binutils)
- **macOS**: `nm` and `otool` (included with Xcode)
- **Windows**: `dumpbin` (Visual Studio)

### For Build System Integration
Install as needed for your project:
- **CMake**: `cmake` version 3.10+
- **qmake**: Qt5 or Qt6 development tools
- **Meson**: `meson` and `ninja`
- **Autotools**: `autoconf`, `automake`, `libtool`

## Verification

Test your installation:

```julia
using RepliBuild

# Display version and info
RepliBuild.info()

# Initialize a test project
RepliBuild.init("test_project")

# Check LLVM toolchain
RepliBuild.print_toolchain_info()
```

Expected output:
```
╔══════════════════════════════════════════════════════════════╗
║                  RepliBuild Build System v0.1.1              ║
╠══════════════════════════════════════════════════════════════╣
║  A TOML-based build system leveraging LLVM/Clang           ║
║  for automatic Julia bindings generation                    ║
╚══════════════════════════════════════════════════════════════╝
```

## Troubleshooting

### LLVM Not Found
RepliBuild uses JLL packages for LLVM. If you encounter issues:

```julia
using Pkg
Pkg.build("LLVM_full_assert_jll")
```

### Binary Tools Missing
Install platform-specific tools:

**Ubuntu/Debian:**
```bash
sudo apt-get install binutils
```

**Fedora/RHEL:**
```bash
sudo dnf install binutils
```

**macOS:**
```bash
xcode-select --install
```

**Windows:**
Install Visual Studio with C++ tools, or use MinGW-w64.

### Module Path Issues
Reset RepliBuild paths:

```julia
using RepliBuild
RepliBuild.initialize_directories()
RepliBuild.print_paths_info()
```

## Next Steps

- **[Quick Start](quickstart.md)**: Build your first project
- **[Project Structure](project-structure.md)**: Understand directory layout
