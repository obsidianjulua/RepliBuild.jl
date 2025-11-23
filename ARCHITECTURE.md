# RepliBuild Architecture - Unified Vision

**Date:** 2025-11-23
**Version:** Post-simplification (Phase 3.1)
**Original Codebase:** ~50k+ LOC → Reduced to ~10k LOC
**Purpose:** Language-agnostic LLVM orchestration with automatic Julia wrapping

---

## Core Philosophy

> **RepliBuild is NOT a C++ build system.**
> **RepliBuild IS an orchestration layer for LLVM-based compilation → Julia integration.**

### What RepliBuild Does:
1. ✅ **Orchestrates** LLVM/Clang (doesn't replace them)
2. ✅ **Automates** the Clang.jl wrapper generation process
3. ✅ **Manages** build state, caches, and project configuration
4. ✅ **Coordinates** compilation → linking → wrapping → testing
5. ✅ **Supports** ANY language that compiles to LLVM IR

### What RepliBuild Does NOT Do:
- ❌ Replace LLVM/Clang toolchain
- ❌ Implement custom C++ parser
- ❌ Re-invent build systems (use CMake/Make/Cargo as sources)

---

## Historical Context: The Simplification

### Original Codebase (~50k+ LOC)
**Removed Systems** (during Phase 1 cleanup):
- `ErrorLearning.jl` - Error database and learning system
- `ModuleRegistry.jl` - External package resolution
- `Daemon/` - Distributed background compilation
- `BuildSystemDelegates/` - CMake/Make/Meson integration layers
- `JuliaWrapItUp.jl` (1519 LOC) - Merged into `Wrapper.jl`
- `LLVMake.jl` (1176 LOC duplicate) - Merged into `Compiler.jl`
- Various UX helpers and status displays

**Why Simplified:**
- Original complexity made LLM assistance difficult
- 50k+ LOC scattered context across too many files
- Core functionality (compile → wrap) was buried
- Need stable foundation before re-adding advanced features

**Current Focus:** Get the core pipeline rock-solid first

---

## Architectural Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 1: User Interface                                            │
│ ───────────────────────────────────────────────────────────────────│
│ • Public API: discover(), build(), wrap(), clean(), info()         │
│ • REPL API: rbuild(), rdiscover(), rwrap(), etc.                   │
│ • CLI: julia -e 'using RepliBuild; build(".")'                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 2: Orchestration (The RepliBuild Core)                       │
│ ───────────────────────────────────────────────────────────────────│
│ • ConfigurationManager: Single source of truth (TOML ↔ structs)    │
│ • Discovery: Project scanning + AST dependency graphs              │
│ • Compiler: Orchestrates LLVM/Clang compilation                    │
│ • Wrapper: Orchestrates Julia binding generation                   │
│ • WorkspaceBuilder: Multi-library parallel builds                  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 3: LLVM/Clang Integration                                    │
│ ───────────────────────────────────────────────────────────────────│
│ • LLVMEnvironment: Toolchain discovery (JLL/system/bundled)        │
│ • BuildBridge: Execute LLVM tools (clang++, llvm-link, opt, etc.)  │
│ • ASTWalker: Use Clang to parse C++ AST for dependencies           │
│ • ClangJLBridge: Use Clang.jl for header-aware wrapping            │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 4: External Tools (Not Part of RepliBuild)                   │
│ ───────────────────────────────────────────────────────────────────│
│ • LLVM Toolchain: clang, clang++, llvm-link, opt, llc              │
│ • Language Frontends: flang (Fortran), rustc (Rust), swiftc, etc.  │
│ • Clang.jl: Header parsing for Julia bindings                      │
│ • CMake/Make: Build system import (future)                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: The Complete Pipeline

### Phase 1: Discovery
```
User: RepliBuild.discover("project/")
         ↓
  ┌──────────────────┐
  │ Discovery.jl     │
  ├──────────────────┤
  │ 1. Scan for src  │
  │ 2. Find headers  │
  │ 3. Build AST     │ ← Uses Clang AST parser
  │ 4. Infer config  │
  │ 5. Generate TOML │
  └──────────────────┘
         ↓
  replibuild.toml (generated)
  .replibuild_cache/dependency_graph.json
```

### Phase 2: Compilation
```
User: RepliBuild.build("project/")
         ↓
  ┌────────────────────────┐
  │ ConfigurationManager   │
  │ load_config("*.toml")  │ ← ONLY module that parses TOML
  └────────────────────────┘
         ↓
  RepliBuildConfig (immutable struct)
         ↓
  ┌────────────────────────┐
  │ Compiler.jl            │
  ├────────────────────────┤
  │ 1. C++ → LLVM IR       │ ← clang++ -S -emit-llvm
  │ 2. Link IR files       │ ← llvm-link
  │ 3. Optimize            │ ← opt -O2
  │ 4. Create binary       │ ← clang++ -shared OR create exe
  │ 5. **Save metadata**   │ ← compilation_metadata.json
  └────────────────────────┘
         ↓
  Outputs:
  - build/*.ll (LLVM IR)
  - julia/lib*.so (shared library)
  - julia/*_test (executable)
  - julia/compilation_metadata.json (NEW!)
```

### Phase 3: Wrapping
```
User: RepliBuild.wrap("julia/lib*.so")
         ↓
  ┌────────────────────────┐
  │ Wrapper.jl             │
  ├────────────────────────┤
  │ Auto-detect tier:      │
  │                        │
  │ If metadata.json → T3  │ ← Introspective (95% accuracy)
  │ Elif headers → T2      │ ← Advanced (85% accuracy)
  │ Else → T1              │ ← Basic (40% accuracy)
  └────────────────────────┘
         ↓
  ┌─────────────────────────────────────┐
  │ Tier 1: wrap_basic()                │
  │ - nm symbol extraction              │
  │ - Conservative Any types            │
  └─────────────────────────────────────┘
         ↓
  ┌─────────────────────────────────────┐
  │ Tier 2: wrap_with_clang()           │
  │ - Clang.jl header parsing           │
  │ - Type-aware bindings               │
  └─────────────────────────────────────┘
         ↓
  ┌─────────────────────────────────────┐
  │ Tier 3: wrap_with_metadata() (NEW!) │
  │ - Read compilation_metadata.json    │
  │ - Exact function signatures         │
  │ - Perfect type mappings             │
  │ - Zero manual configuration         │
  └─────────────────────────────────────┘
         ↓
  julia/MyProject.jl (generated module)
```

---

## The Metadata Innovation (Your Original Vision)

### Problem with Current Clang.jl Approach:
```julia
# Manual configuration hell:
headers = ["lib.h", "utils.h", "internal.h"]  # How do you know all of them?
include_dirs = ["/usr/include", "include/"]    # Platform-specific paths
library_path = "libproject.so"

Clang.generate_bindings(headers, include_dirs, library_path)
# ERROR: unknown type name 'bool'  ← C++ parsing breaks
```

### Solution: Capture During Compilation
```julia
# Stage 1: Compilation
Compiler.compile_project(config)
  ↓
Creates:
  - libproject.so
  - compilation_metadata.json  ← THE KEY!

# Stage 2: Wrapping (zero manual config!)
Wrapper.wrap_library("libproject.so")
  ↓
Auto-detects metadata.json
Uses EXACT types from compilation
No headers needed!
```

### What Gets Captured:
```json
{
  "functions": [
    {
      "name": "add",
      "demangled": "add(int, int)",
      "mangled": "_Z3addii",
      "source_file": "math.cpp",
      "line": 42,
      "return_type": {"c_type": "int", "julia_type": "Cint"},
      "parameters": [
        {"name": "a", "c_type": "int", "julia_type": "Cint"},
        {"name": "b", "c_type": "int", "julia_type": "Cint"}
      ]
    }
  ],
  "types": {
    "int": {"size": 4, "julia_type": "Cint"},
    "double": {"size": 8, "julia_type": "Cdouble"}
  },
  "compiler_info": {
    "llvm_version": "15.0.0",
    "target_triple": "x86_64-unknown-linux-gnu",
    "source_files": ["math.cpp"]
  }
}
```

**Result:** Wrapper has PERFECT type information, no manual configuration!

---

## Module Architecture (Current Simplified)

### Core Modules (src/)

```
RepliBuild.jl (main)
├── ConfigurationManager.jl   [TOML ↔ RepliBuildConfig structs]
│   ├── load_config()          ← ONLY place TOML is parsed
│   ├── save_config()          ← ONLY place TOML is written
│   └── Accessor functions     ← get_source_files(), is_parallel_enabled()
│
├── Discovery.jl              [Project scanning + config generation]
│   ├── discover()             ← Scan project, build AST
│   ├── generate_config()      ← Create RepliBuildConfig
│   └── Uses: ASTWalker
│
├── Compiler.jl               [C++ → LLVM IR → Binary]
│   ├── compile_to_ir()        ← C++ → .ll files
│   ├── link_optimize_ir()     ← llvm-link + opt
│   ├── create_library()       ← .ll → .so
│   ├── create_executable()    ← .ll → binary
│   └── **save_metadata()**    ← NEW: Save compilation_metadata.json
│
├── Wrapper.jl                [Binary → Julia bindings]
│   ├── wrap_library()         ← Auto-detect tier
│   ├── wrap_basic()           ← Tier 1: nm symbols
│   ├── wrap_with_clang()      ← Tier 2: Clang.jl headers
│   └── **wrap_with_metadata()**  ← NEW: Tier 3: metadata
│
├── WorkspaceBuilder.jl       [Multi-library orchestration]
│   ├── discover_workspace()   ← Find all sub-projects
│   ├── compute_build_order()  ← Topological sort
│   └── build_workspace()      ← Parallel builds
│
├── ASTWalker.jl              [Clang AST parsing for dependencies]
├── ClangJLBridge.jl          [Clang.jl integration]
├── LLVMEnvironment.jl        [LLVM toolchain discovery]
├── BuildBridge.jl            [Execute LLVM tools]
└── REPL_API.jl               [User-friendly commands]
```

### Dependency Order (Critical!):
```
1. RepliBuildPaths, LLVMEnvironment, ConfigurationManager, BuildBridge
2. ASTWalker, Discovery, CMakeParser, ClangJLBridge
3. Compiler, Wrapper
4. WorkspaceBuilder
5. REPL_API
```

---

## Configuration System

### Single Source of Truth: RepliBuildConfig

```julia
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
    config_file::String
    loaded_at::DateTime
end
```

**Immutable by design:**
- Thread-safe for parallel builds
- No module can modify config
- Predictable data flow

**Access Pattern:**
```julia
# ❌ WRONG (modules don't parse TOML)
data = TOML.parsefile("replibuild.toml")
files = data["compile"]["source_files"]

# ✅ RIGHT (use accessor functions)
config = ConfigurationManager.load_config("replibuild.toml")
files = ConfigurationManager.get_source_files(config)
```

---

## Multi-Language Support (Future)

### The LLVM IR Common Ground

```
C++     →  clang++ -S -emit-llvm        →  .ll (LLVM IR)
C       →  clang -S -emit-llvm          →  .ll (LLVM IR)
Fortran →  flang -S -emit-llvm          →  .ll (LLVM IR)
Rust    →  rustc --emit=llvm-ir         →  .ll (LLVM IR)
Swift   →  swiftc -emit-ir              →  .ll (LLVM IR)
Zig     →  zig build-lib -femit-llvm-ir →  .ll (LLVM IR)

ALL → Same IR format → RepliBuild links + wraps!
```

### Language Detection (Future Enhancement):
```julia
function detect_language(source_file::String)
    ext = splitext(source_file)[2]
    return if ext in [".cpp", ".cc", ".cxx", ".hpp"]
        LanguageConfig("clang++", "-S -emit-llvm", ["-std=c++17"])
    elseif ext == ".c"
        LanguageConfig("clang", "-S -emit-llvm", ["-std=c11"])
    elseif ext in [".f90", ".f95", ".f03"]
        LanguageConfig("flang", "-S -emit-llvm", [])
    elseif ext == ".rs"
        LanguageConfig("rustc", "--emit=llvm-ir", [])
    else
        error("Unsupported: $ext")
    end
end
```

---

## Current Status & Roadmap

### ✅ Working Now (Phase 3.1)
1. Discovery: Scan C++ projects, build AST graphs
2. Compilation: C++ → IR → Shared lib OR executable
3. Wrapping: Tier 1 (basic symbols) + Tier 2 (partial Clang.jl)
4. Configuration: Immutable structs, TOML serialization
5. Workspace: Multi-library parallel builds
6. Incremental builds: Cache hit detection

### 🚧 In Progress
1. **Metadata generation** - Save compilation_metadata.json
2. **Tier 3 wrapper** - Use metadata for perfect types
3. **Multi-target builds** - shared + executable in one build
4. **Auto-wrapping flow** - Zero manual configuration

### 🎯 Near-Term Roadmap
1. Complete metadata pipeline (this session!)
2. Multi-target binary outputs
3. Component-based builds (separate libraries per component)
4. Cross-compilation support (target triple, CPU, features)

### 🌟 Long-Term Vision
1. Multi-language support (Fortran, Rust, Zig, Swift)
2. Build system import (CMake → replibuild.toml)
3. Distributed compilation (daemon mode)
4. Error learning system
5. External module registry

---

## Design Principles

### 1. Orchestration, Not Replacement
RepliBuild coordinates LLVM/Clang, doesn't reimplement them.

### 2. Metadata Over Manual Configuration
Capture info during compilation, use it during wrapping.

### 3. Language Agnostic
Works with ANY language that compiles to LLVM IR.

### 4. Zero Configuration Goal
User runs: `RepliBuild.build(".")` → Gets everything.

### 5. Incremental Everything
Cache compilation, cache AST, cache wrappers.

### 6. Immutable Configuration
Thread-safe, predictable, functional approach.

---

## Key Innovations

### 1. AST-Driven Discovery
Not just scanning files - building full dependency graph via Clang AST.

### 2. Compilation Metadata
Capturing type information during compilation for wrapper generation.

### 3. 3-Tier Wrapper System
Graceful degradation: metadata (best) → headers (good) → symbols (ok).

### 4. Unified Multi-Language Pipeline
Same infrastructure for C, C++, Fortran, Rust, etc.

### 5. Workspace-Aware Builds
Multi-library projects with dependency ordering.

---

## For LLM Context: How to Work with This Codebase

### When Modifying Code:

1. **Respect the layers:** Don't bypass orchestration
2. **Use ConfigurationManager:** Never parse TOML in modules
3. **Immutable config:** Use `with_*` helpers to create new configs
4. **Metadata-first:** Capture during compilation, use during wrapping
5. **Language-agnostic:** Don't hardcode C++ assumptions

### Module Loading Order Matters:
Always load in this order (see RepliBuild.jl):
```julia
include("ConfigurationManager.jl")  # First!
include("Compiler.jl")
include("Wrapper.jl")
include("WorkspaceBuilder.jl")  # Last!
```

### Testing Pattern:
```julia
# 1. Create test project
# 2. Discover
# 3. Build (with metadata!)
# 4. Wrap (auto-detect tier)
# 5. Verify all outputs
```

---

## Summary

**RepliBuild is:**
- Orchestration layer for LLVM-based compilation
- Automatic Julia wrapper generator
- Build state and cache manager
- Language-agnostic IR pipeline

**RepliBuild is NOT:**
- A C++ build system (use CMake for that)
- A replacement for LLVM/Clang
- A new programming language

**The Vision:**
```julia
# One command does everything:
RepliBuild.build("myproject")

# Creates:
# - libmyproject.so (for Julia)
# - myproject_test (for testing)
# - MyProject.jl (perfect bindings)
# - compilation_metadata.json (for future wrapping)

# Zero manual configuration!
```

**We're 80% there. Let's finish the metadata pipeline!**
