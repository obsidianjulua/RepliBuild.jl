# RepliBuild API Unification - Complete

## Problem Solved

**Before:** Confusing mess of 50+ exported functions, unclear workflow, even the creator struggled to use it
**After:** Clean 3-command API that anyone can understand

## The New API

### Core Functions (ONLY 3!)

```julia
using RepliBuild

# 1. Compile C++ → library
RepliBuild.build()

# 2. Generate Julia wrapper
RepliBuild.wrap()

# 3. Check status
RepliBuild.info()
```

### Utility Function

```julia
# Clean build artifacts
RepliBuild.clean()
```

### Advanced (Power Users)

```julia
# Direct module access for advanced users
RepliBuild.Compiler.compile_to_ir(config, files)
RepliBuild.Wrapper.wrap_library(config, lib_path)
RepliBuild.Discovery.discover(".", force=true)
RepliBuild.ConfigurationManager.load_config("replibuild.toml")
```

## What Changed

### Exports (src/RepliBuild.jl)

**Before:**
```julia
export discover, build, import_cmake, clean, info
export ASTWalker, Discovery, CMakeParser, WorkspaceBuilder, LLVMEnvironment
export REPL_API, rbuild, rdiscover, rclean, rinfo, rwrap,
       rbuild_fast, rcompile, rparallel, rthreads, rcache_status
```

**After:**
```julia
# Core 3-function user API
export build, wrap, info

# Utility
export clean

# Advanced modules (for power users)
export Compiler, Wrapper, Discovery, ConfigurationManager
```

### Function Behavior

**`build()`:**
- **Before:** Only compiled, didn't wrap, unclear what it did
- **After:** Compiles C++ → library + extracts metadata, tells you to run `wrap()` next

**`wrap()` (NEW):**
- Generates Julia wrappers from compiled library
- Uses metadata from `build()`
- Clear error if you didn't run `build()` first

**`info()`:**
- **Before:** Showed complex workspace structure, confusing output
- **After:** Simple checklist: ✓ Library built? ✓ Wrapper generated?

## Files Modified

1. **src/RepliBuild.jl** - Simplified exports, rewrote core functions
2. **src/REPL_API.jl** - Disabled noisy startup message
3. **USAGE.md** - Created new simple usage guide
4. **README.md** - Updated with 3-command API
5. **LLMREADME.md** - Updated with 3-command API
6. **test_cpp_project/build_simple.jl** - Created simple example script

## Tested

```bash
cd test_cpp_project/
julia --project=.. -e 'using RepliBuild; RepliBuild.build()'  # ✓ Works
julia --project=.. -e 'using RepliBuild; RepliBuild.wrap()'   # ✓ Works
julia --project=.. -e 'using RepliBuild; RepliBuild.info()'   # ✓ Works
```

## Migration Guide

**Old way (confusing):**
```julia
RepliBuild.discover()
RepliBuild.build()
# ... now what? where's my wrapper?
RepliBuild.rwrap("julia/libproject.so", tier=:introspective)
```

**New way (obvious):**
```julia
RepliBuild.build()  # Compile C++
RepliBuild.wrap()   # Generate Julia wrapper
# Done!
```

## Result

**API crisis solved.** Users now have a simple, predictable workflow:
1. Create `replibuild.toml` config
2. Run `build()` to compile
3. Run `wrap()` to generate bindings
4. Use their C++ library from Julia

No more confusion about `rbuild` vs `build`, no more wondering which of 50 functions to call, no more complex workflows. Just 3 commands.
