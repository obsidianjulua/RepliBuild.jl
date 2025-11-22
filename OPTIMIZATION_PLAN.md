# RepliBuild.jl Optimization & Cleanup Plan

**Date**: 2025-11-22
**Goal**: Simplify and optimize core build process without excessive complexity

---

## Current State Analysis

### Module Inventory

1. **Core Pipeline** (✅ Keep - Well designed)
   - `RepliBuild.jl` - Main entry point, clean API
   - `Discovery.jl` - Project scanning, ~600 LOC
   - `WorkspaceBuilder.jl` - Multi-library builds, ~387 LOC
   - `Bridge_LLVM.jl` - Main compiler, ~1119 LOC

2. **Infrastructure** (✅ Keep - Essential)
   - `LLVMEnvironment.jl` - Toolchain management
   - `BuildBridge.jl` - Command execution (~328 LOC, simple & focused)
   - `ConfigurationManager.jl` - TOML handling
   - `RepliBuildPaths.jl` - Path management
   - `ASTWalker.jl` - Dependency analysis

3. **Redundant/Heavy Modules** (⚠️ Need evaluation)
   - `LLVMake.jl` - **1176 LOC, HEAVY DUPLICATION**
   - `JuliaWrapItUp.jl` - **1520 LOC, OVERLY COMPLEX**
   - `ClangJLBridge.jl` - ~297 LOC (moderate, could stay)

4. **Import/Support** (✅ Keep)
   - `CMakeParser.jl` - CMake import functionality
   - `REPL_API.jl` - User-friendly commands

---

## Problems Identified

### 1. **Massive Code Duplication**

**LLVMake.jl vs Bridge_LLVM.jl**
- Both compile C++ → LLVM IR
- Both link and optimize IR
- Both generate Julia bindings
- Both handle errors
- **Result**: ~2300 LOC doing essentially the same thing twice

**Specific overlaps**:
- `LLVMake.compile_to_ir()` vs `Bridge_LLVM.compile_to_ir()`
- `LLVMake.optimize_and_link_ir()` vs `Bridge_LLVM.link_optimize_ir()`
- `LLVMake.compile_ir_to_shared_lib()` vs `Bridge_LLVM.create_library()`
- `LLVMake.generate_julia_bindings()` vs `Bridge_LLVM.generate_julia_bindings()`

### 2. **Over-Engineered Wrapper System**

**JuliaWrapItUp.jl** is 1520 LOC for:
- Binary introspection (nm, objdump)
- Symbol parsing
- C++ demangling
- Test generation
- Documentation generation

**But**: The actual binding generation in `Bridge_LLVM.jl` (lines 646-783) is ~140 LOC and works fine.

### 3. **Configuration Complexity**

Multiple config styles:
- `replibuild.toml` (main)
- `wrapper_config.toml` (JuliaWrapItUp)
- Metadata JSON files
- LLVMake custom config schema

### 4. **Missing References**

Both LLVMake.jl and JuliaWrapItUp.jl reference modules that don't exist:
- `BuildBridge.ErrorLearning` (deleted)
- `ModuleRegistry` (moved/deleted)

---

## Recommended Actions

### Phase 1: Eliminate Redundancy (HIGH PRIORITY)

#### 1.1 Remove LLVMake.jl ❌ DELETE

**Reason**: 100% redundant with Bridge_LLVM.jl

**Current users**: None in main RepliBuild.jl

**Migration**:
```julia
# LLVMake features already in Bridge_LLVM:
# ✓ Compile to IR
# ✓ Link & optimize
# ✓ Create shared libs/executables
# ✓ Generate bindings
# ✓ Parse AST
# ✓ Type mappings
```

**Action**: Delete `src/LLVMake.jl` completely.

#### 1.2 Simplify JuliaWrapItUp.jl ⚠️ MAJOR REFACTOR

**Keep only**: Symbol extraction via `nm` for binary-only wrapping

**Remove** (~1200 LOC):
- Stage 1/2 integration (unused)
- objdump extraction (redundant with nm)
- libclang extraction (stub, never implemented)
- Test generation (belongs elsewhere)
- Documentation generation (belongs elsewhere)
- Complex configuration system
- Parameter parsing heuristics

**Result**: Reduce to ~300 LOC focused module

**New interface**:
```julia
module BinaryWrapper

"""Extract symbols from compiled binary using nm"""
function extract_symbols(lib_path::String; demangle=true)::Vector{Symbol}
    # ~50 LOC
end

"""Generate basic Julia wrapper from symbols"""
function generate_wrapper(lib_path::String, symbols::Vector{Symbol})::String
    # ~100 LOC - basic ccall wrappers
end

"""Wrap external binary (main entry point)"""
function wrap_binary(lib_path::String, output_file::String)
    symbols = extract_symbols(lib_path)
    wrapper = generate_wrapper(lib_path, symbols)
    write(output_file, wrapper)
end

end
```

### Phase 2: Improve Configuration (MEDIUM PRIORITY)

#### 2.1 Redesign ConfigurationManager

**Current issues**:
- Mutable config struct leads to confusion
- No validation
- Scattered defaults
- Unclear when to use cache vs TOML

**Proposed design**:
```julia
# Immutable config with validation
struct RepliBuildConfig
    project::ProjectConfig
    discovery::DiscoveryConfig
    compile::CompileConfig
    link::LinkConfig
    binary::BinaryConfig
    llvm::LLVMConfig

    # Constructor with validation
    function RepliBuildConfig(toml_path::String)
        data = TOML.parsefile(toml_path)

        # Validate required fields
        validate_config(data)

        # Build immutable structs
        new(
            ProjectConfig(data["project"]),
            DiscoveryConfig(data["discovery"]),
            # ... etc
        )
    end
end

# Separate validation
function validate_config(data::Dict)
    @assert haskey(data, "project") "Missing [project] section"
    @assert haskey(data["project"], "name") "Missing project.name"
    # ... more validations
end

# Helpers for common queries
get_include_dirs(config::RepliBuildConfig) = config.compile.include_dirs
get_source_files(config::RepliBuildConfig) = config.compile.source_files
needs_discovery(config::RepliBuildConfig) = !config.discovery.completed
```

**Benefits**:
- Immutable = thread-safe for parallel builds
- Validation catches errors early
- Clear data flow
- Easier to test

#### 2.2 Single Configuration File

**Remove**:
- `wrapper_config.toml`
- `replibuild_auto.toml`
- Metadata JSON files

**Keep**:
- `replibuild.toml` (one source of truth)

**Add sections**:
```toml
[wrap]
enabled = true
style = "clang"  # or "binary", "none"
use_clang_jl = true
module_name = "MyProject"
```

### Phase 3: Julia Ecosystem Integration

#### 3.1 Leverage Existing Packages

**For C++ wrapping**:
```julia
# Already using:
Clang.jl  # ✅ Keep - best-in-class C++ parsing

# Consider adding:
CxxWrap.jl  # For complex C++ (templates, classes)
  - Pro: Type-safe, handles C++ idioms
  - Con: Requires C++ shim code
  - Use case: Complex C++ libraries

# Consider removing:
# - Custom binding generation (use Clang.jl exclusively)
```

**For build system**:
```julia
# Current: Custom everything
# Consider:
BinaryBuilder.jl  # For cross-compilation, dependencies
  - Pro: Industry standard, handles artifacts
  - Con: Heavyweight for simple builds
  - Use case: Optional for complex projects

# Pkg.jl Artifacts system
  - Pro: Native Julia packaging
  - Con: Requires Artifacts.toml
  - Use case: Distributing compiled libraries
```

**For LLVM**:
```julia
# Current: LLVM_full_assert_jll ✅ Good
# Alternative: LLVM.jl (more features but heavier)
```

#### 3.2 Configuration Format

**Current**: TOML (good choice)

**Alternatives considered**:
- JSON: Less human-friendly ❌
- YAML: Whitespace issues ❌
- Julia files: Security concerns ❌

**Verdict**: Keep TOML ✅

### Phase 4: Simplify Core Build Logic

#### 4.1 Bridge_LLVM.jl Cleanup

**Current**: 1119 LOC with some cruft

**Optimize**:
```julia
# Remove unused features:
- Line 511-527: ErrorLearning references (deleted module)
- Line 1030-1048: Clang.jl integration (move to ClangJLBridge)
- Line 646-783: Basic binding gen (keep only if not using Clang.jl)

# Simplify stages:
"stages" => ["compile", "link", "binary"]  # Remove: discover_tools, parse_ast, etc.

# Make incremental build default:
cache_enabled = true  # Always
```

**Result**: ~800-900 LOC focused compiler

#### 4.2 Discovery.jl Cleanup

**Current**: 600 LOC, mostly good

**Minor improvements**:
```julia
# Remove:
- Binary detection (not core to C++ discovery)
- Error learning integration (deleted)

# Focus on:
- Find C++ sources ✅
- Build dependency graph ✅
- Detect include dirs ✅
- Generate config ✅
```

**Result**: ~450 LOC focused discovery

### Phase 5: Documentation & Testing

#### 5.1 Update CLAUDE.md

After changes, update with:
- New simplified module structure
- Removed modules (LLVMake, old JuliaWrapItUp)
- New ConfigurationManager patterns
- Julia ecosystem integrations

#### 5.2 Reinstate Tests

**Priority**:
1. ConfigurationManager tests (validation, loading)
2. Discovery tests (file finding, dependency graphs)
3. Bridge_LLVM tests (compile, link, lib creation)
4. WorkspaceBuilder tests (multi-library builds)

**Test structure**:
```julia
# test/runtests.jl
using Test
using RepliBuild

@testset "RepliBuild.jl" begin
    include("test_config.jl")
    include("test_discovery.jl")
    include("test_compilation.jl")
    include("test_workspace.jl")
end
```

---

## Implementation Priority

### Immediate (Do First) 🔥
1. ✅ Delete `src/LLVMake.jl` - pure waste
2. ✅ Remove ErrorLearning refs from Bridge_LLVM.jl
3. ✅ Simplify JuliaWrapItUp.jl to ~300 LOC
4. ✅ Update CLAUDE.md

### Short-term (This Week) 📅
5. Redesign ConfigurationManager (immutable, validated)
6. Clean up Bridge_LLVM.jl (remove unused stages)
7. Add basic tests for core modules

### Medium-term (This Month) 📆
8. Improve error messages and user experience
9. Add examples directory with sample projects
10. Document Julia ecosystem integration patterns

### Long-term (Future) 🔮
11. Consider CxxWrap.jl for complex C++ projects
12. Investigate BinaryBuilder.jl integration
13. Add cross-compilation support

---

## Julia Ecosystem Research

### Build Tools
- **BinaryBuilder.jl**: Cross-platform binary building, dependency management
  - *Use for*: Complex external dependencies
  - *Skip for*: Simple single-library projects

- **Pkg.jl Artifacts**: Native Julia artifact system
  - *Use for*: Distributing pre-compiled binaries
  - *Integration*: Generate Artifacts.toml from build

### C++ Wrapping
- **Clang.jl** ✅ Already using
  - Best choice for automatic wrapping
  - Type-aware bindings

- **CxxWrap.jl**: Julia ↔ C++ bridge
  - Requires C++ wrapper code
  - Better for complex C++ (templates, inheritance)
  - Use case: Qt, Boost, complex OOP

- **MethodAnalysis.jl**: Julia method introspection
  - Could help with generated binding quality

### LLVM
- **LLVM.jl**: Full LLVM bindings
  - Overkill for our use case
  - Stick with JLL packages ✅

### Configuration
- **Preferences.jl**: User/project preferences
  - Could use for user-specific settings
  - Keep TOML for project config

### Testing
- **Test.jl** ✅ Standard library
- **TestEnv.jl**: Isolated test environments
  - Useful for testing with sample projects

---

## Migration Path

### Step 1: Backup & Branch
```bash
git checkout -b optimize-core
git add -A
git commit -m "Checkpoint before optimization"
```

### Step 2: Delete Redundant Code
```bash
rm src/LLVMake.jl
# Update RepliBuild.jl to remove include("LLVMake.jl")
```

### Step 3: Refactor JuliaWrapItUp
```julia
# Rename to BinaryWrapper.jl
# Keep only:
# - extract_symbols (nm-based)
# - generate_wrapper (basic ccall)
# - wrap_binary (main entry point)
```

### Step 4: Update Bridge_LLVM
```julia
# Remove:
# - ErrorLearning calls
# - Duplicate binding generation paths
# - Unused stages

# Simplify:
# - Use ClangJLBridge.jl exclusively for smart wrapping
# - Use BinaryWrapper.jl for binary-only wrapping
# - Single clear path through compilation
```

### Step 5: Redesign ConfigurationManager
```julia
# New immutable design
# Validation functions
# Helper accessors
# Clear cache vs TOML separation
```

### Step 6: Test & Document
```julia
# Add tests
# Update CLAUDE.md
# Add examples/
```

---

## Success Metrics

**Code Reduction**:
- Current: ~6000 LOC
- Target: ~3500 LOC (40% reduction)

**Module Count**:
- Current: 14 modules
- Target: 11 modules (remove 3)

**Complexity**:
- Eliminate: All duplicate compilation code
- Simplify: Wrapper generation (1500 → 300 LOC)
- Clarify: Configuration management

**Quality**:
- Add: Test suite (0 → 100+ tests)
- Improve: Error messages
- Document: Architecture clearly

---

## Questions for User

1. **CxxWrap.jl**: Do you need to wrap complex C++ (templates, classes, inheritance)?
   - If YES → Consider CxxWrap integration
   - If NO → Clang.jl is sufficient

2. **BinaryBuilder.jl**: Do you need cross-compilation or external deps?
   - If YES → Consider BinaryBuilder
   - If NO → Current approach is fine

3. **Artifacts**: Do you want to distribute compiled libs via Julia's Pkg system?
   - If YES → Add Artifacts.toml generation
   - If NO → Skip

4. **Testing**: What should be the priority test cases?
   - Simple single-file project
   - Multi-library workspace
   - CMake import
   - Binary wrapping

---

## Conclusion

**Main recommendation**: Start with **delete LLVMake.jl** and **simplify JuliaWrapItUp.jl**. This alone removes ~2500 LOC of redundant/overly complex code and makes the system much clearer.

**Core principle**: Do one thing well. RepliBuild should be the best way to build C++ for Julia, not try to be a universal build system.

**Keep it simple**:
- One compilation path (Bridge_LLVM)
- One config file (replibuild.toml)
- One wrapper per need (Clang.jl for smart, BinaryWrapper for dumb)
- Clear, validated configuration
- Good error messages
- Comprehensive tests
