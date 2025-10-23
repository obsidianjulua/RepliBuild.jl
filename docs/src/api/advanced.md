# Advanced API

Advanced features for power users and tool developers.

## Daemon Management

```@docs
RepliBuild.start_daemons
RepliBuild.stop_daemons
RepliBuild.daemon_status
RepliBuild.ensure_daemons
```

### Usage

```julia
# Start background daemons for faster compilation
RepliBuild.start_daemons()

# Check daemon status
RepliBuild.daemon_status()

# Stop daemons
RepliBuild.stop_daemons()

# Ensure daemons are running
RepliBuild.ensure_daemons()
```

## LLVM Toolchain

```@docs
RepliBuild.get_toolchain
RepliBuild.verify_toolchain
RepliBuild.print_toolchain_info
RepliBuild.with_llvm_env
```

### Usage

```julia
# Get LLVM toolchain configuration
toolchain = RepliBuild.get_toolchain()

# Verify toolchain is functional
RepliBuild.verify_toolchain()

# Print toolchain information
RepliBuild.print_toolchain_info()

# Run code with LLVM environment
RepliBuild.with_llvm_env() do
    # Code here uses RepliBuild's LLVM
    run(`clang --version`)
end
```

## Path Management

```@docs
RepliBuild.get_replibuild_dir
RepliBuild.initialize_directories
RepliBuild.ensure_initialized
RepliBuild.get_cache_dir
RepliBuild.print_paths_info
RepliBuild.migrate_old_structure
```

### Usage

```julia
# Get RepliBuild directory
dir = RepliBuild.get_replibuild_dir()
println("RepliBuild directory: $dir")

# Initialize all directories
RepliBuild.initialize_directories()

# Get cache directory
cache = RepliBuild.get_cache_dir()

# Print all paths
RepliBuild.print_paths_info()
```

## Configuration Management

```@docs
RepliBuild.get_config_value
RepliBuild.set_config_value
```

### Usage

```julia
# Get configuration value
compiler = RepliBuild.get_config_value("default_compiler")
println("Default compiler: $compiler")

# Set configuration value
RepliBuild.set_config_value("default_compiler", "clang++")
RepliBuild.set_config_value("default_optimization", "3")
```

## Function Reference

### Daemon Management

#### start_daemons

```julia
start_daemons(; project_root=pwd())
```

Start all RepliBuild daemon servers.

**Daemons started:**
- Discovery daemon - Watches for file changes
- Setup daemon - Manages configuration
- Compilation daemon - Handles builds
- Orchestrator daemon - Coordinates workflows

**Arguments:**
- `project_root`: Project directory (default: current directory)

**Returns:** DaemonSystem object

**Examples:**
```julia
# Start in current directory
RepliBuild.start_daemons()

# Start in specific directory
RepliBuild.start_daemons(project_root="/path/to/project")

# Now compilations are faster
RepliBuild.compile()  # Uses daemon cache
```

---

#### stop_daemons

```julia
stop_daemons()
```

Stop all running RepliBuild daemons gracefully.

**Returns:** Nothing

**Examples:**
```julia
RepliBuild.stop_daemons()
```

---

#### daemon_status

```julia
daemon_status()
```

Display status of all RepliBuild daemons.

**Returns:** Nothing (prints status)

**Examples:**
```julia
RepliBuild.daemon_status()
```

Output:
```
RepliBuild Daemon Status:
  Discovery:    ✅ Running (PID 12345)
  Setup:        ✅ Running (PID 12346)
  Compilation:  ✅ Running (PID 12347)
  Orchestrator: ✅ Running (PID 12348)
```

---

#### ensure_daemons

```julia
ensure_daemons()
```

Check if all daemons are running and restart any that have crashed.

**Returns:** `true` if all daemons are healthy

**Examples:**
```julia
if !RepliBuild.ensure_daemons()
    println("Some daemons failed to restart")
end
```

---

### LLVM Toolchain

#### get_toolchain

```julia
get_toolchain()
```

Get LLVM toolchain configuration.

**Returns:** Dict with toolchain information:
- `:llvm_dir` - LLVM installation directory
- `:clang` - Path to clang
- `:clang++` - Path to clang++
- `:llvm_config` - Path to llvm-config
- `:version` - LLVM version

**Examples:**
```julia
toolchain = RepliBuild.get_toolchain()

println("LLVM version: ", toolchain[:version])
println("clang: ", toolchain[:clang])
println("clang++: ", toolchain[Symbol("clang++")])
```

---

#### verify_toolchain

```julia
verify_toolchain()
```

Verify LLVM toolchain is functional.

**Checks:**
- clang can compile C code
- clang++ can compile C++ code
- Tools are correct version

**Returns:** `true` if toolchain is functional

**Throws:** `ErrorException` if verification fails

**Examples:**
```julia
try
    RepliBuild.verify_toolchain()
    println("✅ Toolchain OK")
catch e
    println("❌ Toolchain verification failed: ", e.msg)
end
```

---

#### print_toolchain_info

```julia
print_toolchain_info()
```

Print detailed LLVM toolchain information.

**Returns:** Nothing (prints information)

**Examples:**
```julia
RepliBuild.print_toolchain_info()
```

Output:
```
LLVM Toolchain Information:
  LLVM Directory: /usr/lib/llvm-14
  Version: 14.0.6
  Tools:
    clang:       /usr/bin/clang
    clang++:     /usr/bin/clang++
    llvm-config: /usr/bin/llvm-config-14
    llvm-ar:     /usr/bin/llvm-ar-14
  Status: ✅ Functional
```

---

#### with_llvm_env

```julia
with_llvm_env(f::Function)
```

Execute function with LLVM environment configured.

**Arguments:**
- `f::Function`: Function to execute

**Returns:** Return value of `f()`

**Examples:**
```julia
# Run command with RepliBuild's LLVM
RepliBuild.with_llvm_env() do
    run(`clang --version`)
    run(`llvm-config --version`)
end

# Compile with specific LLVM
RepliBuild.with_llvm_env() do
    RepliBuild.compile()
end
```

---

### Path Management

#### get_replibuild_dir

```julia
get_replibuild_dir()
```

Get RepliBuild user directory.

**Returns:** Path string

**Default locations:**
- Linux/macOS: `~/.replibuild`
- Windows: `%LOCALAPPDATA%/RepliBuild`

**Examples:**
```julia
dir = RepliBuild.get_replibuild_dir()
println("RepliBuild directory: $dir")
```

---

#### initialize_directories

```julia
initialize_directories()
```

Initialize all RepliBuild directories.

**Creates:**
- Main RepliBuild directory
- Modules directory
- Cache directory
- Configuration file (if missing)
- Error database

**Returns:** Nothing

**Examples:**
```julia
RepliBuild.initialize_directories()
```

---

#### ensure_initialized

```julia
ensure_initialized()
```

Ensure RepliBuild directories are initialized.

**Returns:** `true` if initialized

**Examples:**
```julia
RepliBuild.ensure_initialized()
```

---

#### get_cache_dir

```julia
get_cache_dir()
```

Get RepliBuild cache directory.

**Returns:** Path to cache directory

**Examples:**
```julia
cache = RepliBuild.get_cache_dir()
println("Cache directory: $cache")

# Clear cache
rm(cache, recursive=true)
RepliBuild.initialize_directories()
```

---

#### print_paths_info

```julia
print_paths_info()
```

Print all RepliBuild paths.

**Returns:** Nothing (prints information)

**Examples:**
```julia
RepliBuild.print_paths_info()
```

Output:
```
RepliBuild Paths:
  Main directory: /home/user/.replibuild
  Modules:        /home/user/.replibuild/modules
  Cache:          /home/user/.replibuild/cache
  Config:         /home/user/.replibuild/config.toml
  Error DB:       /home/user/.replibuild/error_db.sqlite
  Module search paths:
    - /home/user/.replibuild/modules
    - /usr/share/replibuild/modules
```

---

#### migrate_old_structure

```julia
migrate_old_structure()
```

Migrate old RepliBuild directory structure to new format.

**Returns:** Nothing

**Examples:**
```julia
# Migrate from old version
RepliBuild.migrate_old_structure()
```

---

### Configuration Management

#### get_config_value

```julia
get_config_value(key::String; default=nothing)
```

Get value from global configuration.

**Arguments:**
- `key::String`: Configuration key (supports dot notation)
- `default`: Default value if key not found

**Returns:** Configuration value

**Examples:**
```julia
# Get default compiler
compiler = RepliBuild.get_config_value("defaults.compiler")

# With default value
jobs = RepliBuild.get_config_value("defaults.jobs", 0)

# Nested keys
llvm_dir = RepliBuild.get_config_value("llvm.llvm_dir")
```

---

#### set_config_value

```julia
set_config_value(key::String, value)
```

Set value in global configuration.

**Arguments:**
- `key::String`: Configuration key (supports dot notation)
- `value`: Value to set

**Returns:** Nothing

**Examples:**
```julia
# Set default compiler
RepliBuild.set_config_value("defaults.compiler", "clang++")

# Set optimization level
RepliBuild.set_config_value("defaults.optimization", "3")

# Set LLVM preference
RepliBuild.set_config_value("llvm.prefer_system", false)

# Set jobs
RepliBuild.set_config_value("defaults.jobs", 8)
```

---

## Error Learning System

### export_errors

```julia
export_errors(output_path::String="error_log.md")
```

Export error learning database to markdown.

**Arguments:**
- `output_path::String`: Output file path

**Returns:** Nothing

**Examples:**
```julia
# Export errors to markdown
RepliBuild.export_errors("docs/errors.md")

# View error statistics
stats = RepliBuild.get_error_stats()
println("Total errors: ", stats[:total])
println("Unique errors: ", stats[:unique])
```

---

## Low-Level Functions

These functions are for advanced users and tool developers.

### discover

```julia
discover(path::String; force::Bool=false)
```

Run discovery pipeline on a directory.

**Arguments:**
- `path::String`: Directory to analyze
- `force::Bool`: Force re-scan even if cached

**Returns:** Dict with discovery results

**Internal use:**
Used by `scan()` and `analyze()` functions.

---

### compile_project

```julia
compile_project(config::BridgeCompilerConfig)
```

Compile project with configuration object.

**Arguments:**
- `config::BridgeCompilerConfig`: Compiler configuration

**Returns:** Nothing

**Internal use:**
Used by `compile()` function.

---

## See Also

- **[Core API](core.md)**: Core functions
- **[Build System API](build-system.md)**: Build system functions
- **[Module Registry API](modules.md)**: Module functions
- **[Daemon Guide](../advanced/daemons.md)**: Daemon system guide
- **[LLVM Toolchain Guide](../advanced/llvm-toolchain.md)**: Toolchain details
