# RepliBuild Architecture Redesign

**Issue**: We lost the clean separation between user-facing API and internal implementation when LLVMake/JuliaWrapItUp were removed.

**Goal**: Restore clean architecture with centralized configuration.

---

## Original Design (LLVMake + JuliaWrapItUp)

### What Made It Good

```
User's Workflow:
┌─────────────────┐
│   LLVMake.jl    │  ← User-facing, modifiable API
│  (compilation)  │  ← Exported functions for flexibility
└────────┬────────┘
         │
         v
┌─────────────────┐
│ JuliaWrapItUp.jl│  ← User-facing wrapper API
│   (wrapping)    │  ← Fully standalone
└─────────────────┘
```

**Benefits**:
- Users could modify LLVMake.jl directly for custom workflows
- Clean separation: compile vs wrap
- Exported functions for build flexibility
- Two files = complete tool

---

## Current Problem (After Cleanup)

```
Current Mess:
┌──────────────────┐
│  RepliBuild.jl   │  ← High-level orchestration
└────────┬─────────┘
         │
         v
┌──────────────────┐
│ Bridge_LLVM.jl   │  ← Internal compiler (NOT a module!)
│   (bare code)    │  ← Has its own BridgeCompilerConfig
└────────┬─────────┘  ← No clean user API
         │
         v
┌──────────────────┐
│ConfigurationMgr  │  ← Has RepliBuildConfig
│                  │  ← Supposed to be central but isn't!
└──────────────────┘
```

**Problems**:
1. **Two config structs** (BridgeCompilerConfig AND RepliBuildConfig)
2. **No user-modifiable API** (everything is internal)
3. **Config not centralized** (each module parses TOML itself)
4. **Bridge_LLVM is bare code** (not a clean module)

---

## Proposed Architecture

### Principle: Separation of Concerns

```
┌─────────────────────────────────────────────┐
│          USER-FACING API LAYER              │
│  (What users import and can modify)         │
├─────────────────────────────────────────────┤
│                                             │
│  RepliBuild.jl                              │
│    ├─ discover(path)                        │
│    ├─ build(path)                           │
│    ├─ compile(sources, config)   ← NEW     │
│    ├─ wrap(library, config)      ← NEW     │
│    └─ clean(path)                           │
│                                             │
└──────────────┬──────────────────────────────┘
               │
               v
┌─────────────────────────────────────────────┐
│       CENTRALIZED CONFIGURATION             │
│  (Single source of truth)                   │
├─────────────────────────────────────────────┤
│                                             │
│  ConfigurationManager.jl                    │
│    ├─ RepliBuildConfig (immutable)          │
│    ├─ load_config(toml) → Config            │
│    ├─ validate_config(config)               │
│    ├─ save_config(config, toml)             │
│    └─ merge_configs(base, override)         │
│                                             │
└──────────────┬──────────────────────────────┘
               │
               v
┌─────────────────────────────────────────────┐
│         INTERNAL IMPLEMENTATION             │
│  (Modules do the work)                      │
├─────────────────────────────────────────────┤
│                                             │
│  Compiler::                                 │
│    ├─ compile_to_ir(config, sources)        │
│    ├─ link_ir(config, ir_files)             │
│    └─ create_binary(config, linked_ir)      │
│                                             │
│  Wrapper::                                  │
│    ├─ extract_symbols(binary)               │
│    ├─ generate_bindings(symbols, config)    │
│    └─ write_wrapper(bindings, output)       │
│                                             │
│  Discovery::                                │
│    └─ scan_project(path) → sources, deps    │
│                                             │
└─────────────────────────────────────────────┘
```

### Key Changes

1. **Single Config Struct** (RepliBuildConfig - immutable)
2. **Centralized TOML parsing** (only ConfigurationManager touches TOML)
3. **Clean user API** (exported functions that are flexible)
4. **Modular internals** (proper modules, not bare code)

---

## Configuration Architecture

### Single Source of Truth: `replibuild.toml`

```toml
[project]
name = "myproject"
root = "."
uuid = "..."

[paths]
source = "src"
include = "include"
output = "julia"
build = "build"
cache = ".replibuild_cache"

[discovery]
enabled = true
walk_dependencies = true
max_depth = 10
ignore_patterns = ["build/*", ".*"]

[compile]
source_files = []  # Auto-discovered if empty
include_dirs = []  # Auto-discovered if empty
flags = ["-std=c++17", "-fPIC"]
defines = {}
parallel = true

[link]
optimization_level = "2"
enable_lto = false
link_libraries = []

[binary]
type = "shared"  # "shared", "static", "executable"
output_name = ""  # Auto from project.name
strip_symbols = false

[wrap]
enabled = true
style = "clang"  # "clang", "basic", "none"
module_name = ""  # Auto from project.name
use_clang_jl = true

[llvm]
toolchain = "auto"  # "auto", "system", "jll"
version = ""  # Auto-detect

[workflow]
stages = ["discover", "compile", "link", "binary", "wrap"]

[cache]
enabled = true
directory = ".replibuild_cache"
```

### Config Struct (Immutable)

```julia
module ConfigurationManager

using TOML

# Nested immutable structs for each section
struct ProjectConfig
    name::String
    root::String
    uuid::UUID
end

struct PathsConfig
    source::String
    include::String
    output::String
    build::String
    cache::String
end

struct DiscoveryConfig
    enabled::Bool
    walk_dependencies::Bool
    max_depth::Int
    ignore_patterns::Vector{String}
end

struct CompileConfig
    source_files::Vector{String}
    include_dirs::Vector{String}
    flags::Vector{String}
    defines::Dict{String,String}
    parallel::Bool
end

struct LinkConfig
    optimization_level::String
    enable_lto::Bool
    link_libraries::Vector{String}
end

struct BinaryConfig
    type::Symbol  # :shared, :static, :executable
    output_name::String
    strip_symbols::Bool
end

struct WrapConfig
    enabled::Bool
    style::Symbol  # :clang, :basic, :none
    module_name::String
    use_clang_jl::Bool
end

struct LLVMConfig
    toolchain::Symbol  # :auto, :system, :jll
    version::String
end

struct WorkflowConfig
    stages::Vector{Symbol}
end

struct CacheConfig
    enabled::Bool
    directory::String
end

# Main config (immutable!)
struct RepliBuildConfig
    project::ProjectConfig
    paths::PathsConfig
    discovery::DiscoveryConfig
    compile::CompileConfig
    link::LinkConfig
    binary::BinaryConfig
    wrap::WrapConfig
    llvm::LLVMConfig
    workflow::WorkflowConfig
    cache::CacheConfig

    # Metadata
    config_file::String
    loaded_at::DateTime
end

# Constructor with validation
function RepliBuildConfig(toml_path::String)
    data = TOML.parsefile(toml_path)

    # Validate
    validate_config_data(data)

    # Build immutable structs
    config = RepliBuildConfig(
        parse_project_config(data),
        parse_paths_config(data),
        parse_discovery_config(data),
        parse_compile_config(data),
        parse_link_config(data),
        parse_binary_config(data),
        parse_wrap_config(data),
        parse_llvm_config(data),
        parse_workflow_config(data),
        parse_cache_config(data),
        toml_path,
        now()
    )

    return config
end

# Validation
function validate_config_data(data::Dict)
    required = ["project"]
    for key in required
        if !haskey(data, key)
            error("Missing required section: [$key]")
        end
    end

    if !haskey(data["project"], "name")
        error("Missing required field: project.name")
    end

    # More validation...
end

# Helper accessors
get_source_files(c::RepliBuildConfig) = c.compile.source_files
get_include_dirs(c::RepliBuildConfig) = c.compile.include_dirs
get_output_path(c::RepliBuildConfig) = joinpath(c.project.root, c.paths.output)
should_run_discovery(c::RepliBuildConfig) = c.discovery.enabled
# ... more helpers

end # module
```

---

## User API Layer

### Clean, Flexible Functions

```julia
# In RepliBuild.jl (exported to users)

"""
    compile(sources::Vector{String}; config=nothing, flags=[], output="") → String

Compile C++ sources to LLVM IR and link to shared library.

# Arguments
- `sources`: C++ source files to compile
- `config`: RepliBuildConfig or path to TOML (default: "replibuild.toml")
- `flags`: Additional compiler flags
- `output`: Output library name (default: auto from config)

# Returns
- Path to compiled library

# Examples
```julia
# Simple
lib = RepliBuild.compile(["main.cpp", "utils.cpp"])

# With custom flags
lib = RepliBuild.compile(
    ["src/app.cpp"],
    flags=["-std=c++20", "-O3"],
    output="libmyapp.so"
)

# With custom config
config = RepliBuild.load_config("custom.toml")
lib = RepliBuild.compile(sources, config=config)
```
"""
function compile(sources::Vector{String};
                 config=nothing,
                 flags::Vector{String}=String[],
                 output::String="")

    # Load or use provided config
    cfg = if config isa RepliBuildConfig
        config
    elseif config isa String
        ConfigurationManager.load_config(config)
    elseif isnothing(config)
        ConfigurationManager.load_config()
    else
        error("config must be RepliBuildConfig, path string, or nothing")
    end

    # Merge with overrides
    if !isempty(flags)
        cfg = merge_compile_flags(cfg, flags)
    end

    # Call internal compiler
    return Compiler.compile_project(cfg, sources, output)
end

"""
    wrap(library::String; config=nothing, headers=[], style=:auto) → String

Generate Julia bindings for compiled library.

# Arguments
- `library`: Path to .so/.dylib/.dll
- `config`: RepliBuildConfig or TOML path
- `headers`: Header files (for :clang style)
- `style`: :clang (type-aware) or :basic (symbol-only)

# Returns
- Path to generated Julia wrapper

# Examples
```julia
# Automatic with headers
wrapper = RepliBuild.wrap("libmyapp.so", headers=["app.h"])

# Binary-only (basic)
wrapper = RepliBuild.wrap("external.so", style=:basic)

# With custom config
wrapper = RepliBuild.wrap(lib, config="wrap_config.toml")
```
"""
function wrap(library::String;
              config=nothing,
              headers::Vector{String}=String[],
              style::Symbol=:auto)

    cfg = load_or_use_config(config)

    # Auto-detect style
    if style == :auto
        style = isempty(headers) ? :basic : :clang
    end

    # Call internal wrapper
    if style == :clang
        return Wrapper.wrap_with_clang(cfg, library, headers)
    else
        return Wrapper.wrap_basic(cfg, library)
    end
end

"""
    build(path="."; config=nothing, clean=false, parallel=true) → String

Complete build: discover, compile, link, wrap.

# Examples
```julia
# Simple
lib = RepliBuild.build()

# Custom path
lib = RepliBuild.build("../myproject")

# Clean build
lib = RepliBuild.build(clean=true)
```
"""
function build(path::String=".";
               config=nothing,
               clean::Bool=false,
               parallel::Bool=true)

    cd(path) do
        cfg = load_or_use_config(config)

        if clean
            clean_artifacts(cfg)
        end

        # Run workflow stages
        for stage in cfg.workflow.stages
            run_stage(stage, cfg)
        end
    end
end

# Export user-facing API
export compile, wrap, build, discover, clean, info
export load_config, merge_configs  # Config utilities
```

---

## Implementation Plan

### Phase 1: Centralize Configuration ✅

1. **Make RepliBuildConfig immutable**
   - Nested structs for each section
   - Validation on construction
   - Helper accessors

2. **Remove BridgeCompilerConfig**
   - Replace with RepliBuildConfig everywhere
   - Update all Bridge_LLVM functions to accept RepliBuildConfig

3. **Single TOML parser**
   - Only ConfigurationManager parses TOML
   - Other modules receive pre-parsed config struct

### Phase 2: Refactor Bridge_LLVM

1. **Convert to proper module** (optional)
   - Or keep as bare code but clean it up
   - Remove config parsing (use passed config)
   - Remove redundant code

2. **Rename to Compiler**
   - More intuitive
   - Clear responsibility

3. **Clean API**
   ```julia
   module Compiler

   function compile_project(config::RepliBuildConfig, sources, output)
       # Use config.compile.*, config.paths.*, etc.
   end

   function compile_to_ir(config, sources)
       # ...
   end

   function link_ir(config, ir_files)
       # ...
   end

   end
   ```

### Phase 3: Create Wrapper Module

```julia
module Wrapper

function wrap_with_clang(config::RepliBuildConfig, library, headers)
    # Use ClangJLBridge
end

function wrap_basic(config::RepliBuildConfig, library)
    # Simple symbol extraction
end

end
```

### Phase 4: Polish User API

- Add examples to docstrings
- Test flexibility (override any setting)
- Ensure clean imports

---

## Benefits

### For Users

✅ **Clean API**: `compile()`, `wrap()`, `build()`
✅ **Flexible**: Override any setting
✅ **Discoverable**: Good docstrings
✅ **Modifiable**: Can copy/modify functions

### For Developers

✅ **Single config source**: One struct to rule them all
✅ **Immutable**: Thread-safe, clear data flow
✅ **Validated**: Errors caught early
✅ **Testable**: Pure functions with clear inputs

### For Maintenance

✅ **No duplication**: One config parser
✅ **Clear separation**: API vs implementation
✅ **Type-safe**: Compiler helps us
✅ **Documented**: Architecture is clear

---

## Migration Checklist

- [ ] Design immutable RepliBuildConfig structure
- [ ] Implement validation functions
- [ ] Create helper accessor functions
- [ ] Remove BridgeCompilerConfig from Bridge_LLVM
- [ ] Update all functions to use RepliBuildConfig
- [ ] Move TOML parsing to ConfigurationManager only
- [ ] Create clean `compile()` user function
- [ ] Create clean `wrap()` user function
- [ ] Update `build()` to use new architecture
- [ ] Add comprehensive tests
- [ ] Update documentation

---

## Open Questions

1. **Bridge_LLVM**: Convert to module or keep as bare code?
   - **Option A**: Keep bare code (current), document it well
   - **Option B**: Make proper Compiler module (cleaner but more work)

2. **Config mutability**: Completely immutable or allow some overrides?
   - **Option A**: 100% immutable (use helper to create new config)
   - **Option B**: Allow specific overrides (merge_configs function)

3. **TOML schema**: Lock it down or keep flexible?
   - **Option A**: Strict schema, validation
   - **Option B**: Flexible, allow custom sections

---

## Recommendation

**Immediate**:
1. Make RepliBuildConfig immutable with validation
2. Remove BridgeCompilerConfig duplication
3. Centralize TOML parsing

**Short-term**:
4. Add clean user API functions
5. Move Bridge_LLVM to proper Compiler module

**Long-term**:
6. Consider plugin system for custom workflows
7. Add configuration presets (debug, release, etc.)

This restores the spirit of LLVMake/JuliaWrapItUp while modernizing the architecture!
