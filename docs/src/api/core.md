# Core API

Core functions for initializing, compiling, and wrapping C++ code.

## Project Initialization

```@docs
RepliBuild.init
```

### Usage

```julia
# Initialize C++ project
RepliBuild.init("myproject")

# Initialize binary wrapping project
RepliBuild.init("mybindings", type=:binary)

# Initialize in specific directory
RepliBuild.init("/path/to/project")
```

## Compilation

```@docs
RepliBuild.compile
RepliBuild.compile_project
```

### Usage

```julia
# Compile with default config
RepliBuild.compile()

# Compile with custom config
RepliBuild.compile("custom_config.toml")

# Compile from directory
cd("myproject")
RepliBuild.compile()
```

## Binary Wrapping

```@docs
RepliBuild.wrap
RepliBuild.wrap_binary
RepliBuild.generate_wrappers
RepliBuild.scan_binaries
```

### Usage

```julia
# Wrap all configured binaries
RepliBuild.wrap()

# Wrap specific binary
RepliBuild.wrap_binary("/usr/lib/libmath.so")

# Scan binary for symbols
RepliBuild.scan_binaries("lib/mylib.so")
```

## Project Templates

```@docs
RepliBuild.create_project_interactive
RepliBuild.available_templates
RepliBuild.use_template
```

### Usage

```julia
# Interactive project creation
RepliBuild.create_project_interactive()

# List templates
templates = RepliBuild.available_templates()

# Use specific template
RepliBuild.use_template("simple_cpp", "myproject")
```

## Discovery and Analysis

```@docs
RepliBuild.discover
RepliBuild.scan
RepliBuild.analyze
```

### Usage

```julia
# Scan directory for C++ files
result = RepliBuild.scan("path/to/code")

# Analyze project structure
analysis = RepliBuild.analyze(".")

# Discover with auto-config generation
RepliBuild.discover(".", force=true)
```

## CMake Integration

```@docs
RepliBuild.import_cmake
```

### Usage

```julia
# Import CMake project
project = RepliBuild.import_cmake("CMakeLists.txt")

# Import specific target
RepliBuild.import_cmake("CMakeLists.txt",
                        target="mylib",
                        output="mylib.toml")

# Inspect imported project
println("Targets: ", keys(project.targets))
```

## Information

```@docs
RepliBuild.info
RepliBuild.help
```

### Usage

```julia
# Show RepliBuild info
RepliBuild.info()

# Show command help
RepliBuild.help()
```

## Binding Generation

### generate_bindings_clangjl

```julia
generate_bindings_clangjl(library_path::String; headers::Vector{String}=String[])
```

Generate Julia bindings using Clang.jl integration.

**Arguments:**
- `library_path::String`: Path to library file
- `headers::Vector{String}`: Header files to parse (keyword)

**Returns:** Nothing

**Examples:**
```julia
# Generate bindings with Clang.jl
RepliBuild.generate_bindings_clangjl("lib/libmylib.so",
                                     headers=["include/api.h"])
```

## Function Reference

### init

```julia
init(project_dir::String="."; type::Symbol=:cpp)
```

Initialize a new RepliBuild project.

**Arguments:**
- `project_dir::String`: Directory to initialize (default: current directory)
- `type::Symbol`: Project type - `:cpp` for C++ source, `:binary` for binary wrapping

**Returns:** Nothing

**Examples:**
```julia
RepliBuild.init("mylib")
RepliBuild.init("wrappers", type=:binary)
```

---

### compile

```julia
compile(config_file::String="replibuild.toml")
```

Compile a C++ project to Julia bindings.

**Arguments:**
- `config_file::String`: Path to replibuild.toml configuration file

**Returns:** Nothing

**Examples:**
```julia
RepliBuild.compile()
RepliBuild.compile("custom.toml")
```

---

### wrap

```julia
wrap(config_file::String="wrapper_config.toml")
```

Generate Julia wrappers for existing binary files.

**Arguments:**
- `config_file::String`: Path to wrapper configuration file

**Returns:** Nothing

**Examples:**
```julia
RepliBuild.wrap()
RepliBuild.wrap("custom_wrapper.toml")
```

---

### wrap_binary

```julia
wrap_binary(binary_path::String; config_file::String="wrapper_config.toml")
```

Wrap a specific binary file to Julia bindings.

**Arguments:**
- `binary_path::String`: Path to the binary file (.so, .dll, .dylib)
- `config_file::String`: Path to wrapper configuration file (keyword)

**Returns:** Nothing

**Examples:**
```julia
RepliBuild.wrap_binary("/usr/lib/libmath.so")
RepliBuild.wrap_binary("build/libmylib.so")
```

---

### scan

```julia
scan(path="."; generate_config=true, output="replibuild.toml")
```

Scan a directory and analyze its structure for RepliBuild compilation.

**Arguments:**
- `path`: Directory to scan (default: current directory)
- `generate_config`: Auto-generate replibuild.toml if true
- `output`: Output configuration file name

**Returns:** Dictionary with scan results

**Examples:**
```julia
result = RepliBuild.scan()
RepliBuild.scan("path/to/code", output="my_config.toml")
```

---

### analyze

```julia
analyze(path=".")
```

Analyze project structure and return detailed analysis.

**Arguments:**
- `path`: Directory to analyze

**Returns:** Dictionary with analysis results

**Examples:**
```julia
result = RepliBuild.analyze("path/to/project")
println("Found $(length(result[:scan_results].cpp_sources)) C++ files")
```

---

### import_cmake

```julia
import_cmake(cmake_file::String="CMakeLists.txt"; target::String="", output::String="replibuild.toml")
```

Import a CMake project and generate replibuild.toml configuration.

**Arguments:**
- `cmake_file::String`: Path to CMakeLists.txt file
- `target::String`: Specific target to import (empty = first target)
- `output::String`: Output path for replibuild.toml

**Returns:** CMakeProject object

**Examples:**
```julia
project = RepliBuild.import_cmake("CMakeLists.txt")
RepliBuild.import_cmake("opencv/CMakeLists.txt", target="opencv_core")
```

---

### info

```julia
info()
```

Display information about the RepliBuild build system.

**Returns:** Nothing

---

### help

```julia
help()
```

Display help information about RepliBuild commands.

**Returns:** Nothing

---

## Error Handling

All core functions may throw:
- `ArgumentError`: Invalid arguments
- `ErrorException`: Compilation/wrapping failures
- `SystemError`: File I/O errors

Handle errors:

```julia
try
    RepliBuild.compile()
catch e
    if isa(e, ErrorException)
        println("Compilation failed: ", e.msg)
        # Check error log
        RepliBuild.export_errors("errors.md")
    else
        rethrow(e)
    end
end
```

## See Also

- **[Build System API](build-system.md)**: Build system functions
- **[Module Registry API](modules.md)**: Module system functions
- **[Advanced API](advanced.md)**: Advanced features
