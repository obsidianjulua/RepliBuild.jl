# RepliBuild

A Julia-based build system that compiles C++ code to LLVM IR and generates Julia bindings.

## Features

- **LLVM IR Pipeline**: C++ → LLVM IR → Optimized IR → Shared Libraries
- **CMake Import**: Import existing CMake projects automatically
- **Smart Build Helpers**: Auto-detect external libraries via pkg-config
- **Config.h Generation**: Process CMake template files (config.h.in)
- **Multi-stage Builds**: Build libraries first, then executables
- **Julia Bindings**: Automatic generation of Julia wrappers for C++ code
- **Project Templates**: Interactive wizard for quick project setup

## Installation

```julia
# From the Julia REPL
using Pkg
Pkg.add("RepliBuild")
```

## Quick Start

### Import a CMake Project

```julia
using RepliBuild

# Import CMakeLists.txt and generate replibuild.toml files
RepliBuild.cmake_replicate("/path/to/project")

# Compile the project
RepliBuild.compile("replibuild.toml")
```

### Create a New Project

```julia
using RepliBuild

# Interactive wizard
RepliBuild.create_project_interactive()

# Or use a template directly
RepliBuild.use_template("simple_lib", "my_project")
```

### Available Templates

- `simple_lib` - Single C++ library with Julia bindings
- `executable` - Standalone C++ executable
- `lib_and_exe` - Multi-stage: library + executable
- `cmake_import` - Import existing CMake project
- `external_libs` - Project with external dependencies

```julia
# See all templates
RepliBuild.available_templates()
```

### Manual Build

```julia
using RepliBuild

# Compile C++ to shared library
RepliBuild.compile("replibuild.toml")

# Use in Julia
include("julia/myproject.jl")
```

## Configuration

Create a `replibuild.toml` file:

```toml
[project]
name = "myproject"

[paths]
source = "src"
output = "julia"
build = "build"

[compile]
include_dirs = []
lib_dirs = []
link_libraries = []

[target]
opt_level = "O2"
```

## Build Stages

RepliBuild uses a multi-stage pipeline:

1. **discover_tools** - Find LLVM toolchain (clang++, opt, llvm-link)
2. **compile_to_ir** - Compile C++ files to LLVM IR (.ll files)
3. **link_ir** - Link all IR files together
4. **optimize_ir** - Run LLVM optimization passes
5. **create_library** - Generate shared library (.so)
6. **create_executable** - Generate executable (alternative to library)

## Examples

### Multi-stage Build (Libraries + Executable)

```julia
using RepliBuild

# Stage 1: Build libraries
RepliBuild.compile("lib1/replibuild.toml")
RepliBuild.compile("lib2/replibuild.toml")

# Stage 2: Build executable linked against libraries
RepliBuild.compile("app/replibuild.toml")
```

### Generate config.h from Template

```julia
using RepliBuild.BuildHelpers

generate_config_h(
    "config.h.in",
    "config.h",
    Dict(
        "PROJECT_VERSION" => "1.0.0",
        "ENABLE_FEATURE" => "1"
    )
)
```

## Requirements

- Julia 1.6+
- LLVM toolchain (clang++, opt, llvm-link, llvm-ar)
- System linker (ld)

## Project Structure

```
RepliBuild.jl/
├── src/
│   ├── RepliBuild.jl          # Main module
│   ├── Bridge_LLVM.jl         # LLVM compilation pipeline
│   ├── BuildHelpers.jl        # config.h, pkg-config utilities
│   ├── CMakeParser.jl         # CMake import functionality
│   ├── LLVMEnvironment.jl     # LLVM toolchain detection
│   └── ProjectWizard.jl       # Interactive templates
├── docs/
│   └── BUILD_GUIDE.md         # Detailed documentation
└── examples/                   # Example projects
```

## Troubleshooting

### LLVM Tools Not Found

```julia
# RepliBuild auto-discovers LLVM from:
# 1. System package manager (/usr/bin)
# 2. Julia LLVM_jll artifacts
# 3. Custom LLVM installation

# Check what was found:
using RepliBuild.BuildBridge
env = LLVMEnvironment()
env.tools  # Shows discovered tools
```

### Missing External Libraries

```julia
# Auto-detect dependencies
using RepliBuild.BuildHelpers
detect_external_libraries("src")
```

## License

Apache 2.0

## Contributing

Bug reports and pull requests welcome!
