# Changelog

All notable changes to RepliBuild.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2025-11-24

### 🎯 MAJOR: API Unification

**Breaking Changes:**
- Simplified API from 50+ exports down to 3 core functions
- Removed confusing `rbuild()`, `rdiscover()`, `rwrap()` REPL shortcuts
- Removed workspace builder functions from public API
-Removed multi-step build pipeline from user-facing API

**New Simple API:**
```julia
RepliBuild.build()  # Compile C++ → library + metadata
RepliBuild.wrap()   # Generate Julia wrapper
RepliBuild.info()   # Check status
```

### Added
- `wrap()` function - generates Julia wrappers from compiled libraries
- `info()` function - shows project status (library built? wrapper generated?)
- Comprehensive test suite in `test/runtests.jl`
- End-to-end tests proving C++ → Julia binding works
- `USAGE.md` - dead-simple usage guide
- `API_UNIFIED.md` - complete API change documentation

### Changed
- `build()` now clearly communicates it only compiles (doesn't wrap)
- `build()` tells user to run `wrap()` next
- `clean()` simplified to just remove artifacts
- REPL_API module kept for backwards compatibility but deprecated
- All internal functions hidden from exports

### Fixed
- API confusion that made even the creator struggle to use it
- Unclear workflow (what do I call after build?)
- Documentation now accurate with working examples

### Documentation
- Updated README.md with 3-command workflow
- Updated LLMREADME.md with simple examples
- Created test_cpp_project/build_simple.jl example
- All docs now show the correct, simple API

## [1.1.0] - 2024-11-24

### Added
- DWARF debug information extraction system
- Three-way metadata validation (DWARF + LLVM IR + Symbols)
- Automatic C++ FFI generation for standard-layout types
- Support for POD types and trivially-copyable structs
- Compiler.jl module (replaces Bridge_LLVM.jl)
- Wrapper.jl module with 3-tier wrapping system
- Validated on Eigen (20K+ type DIEs extracted)

### Documentation
- Created ARCHITECTURE.md
- Created LIMITATIONS.md with explicit correctness boundaries
- Acknowledged DragonFFI as prior art
- Positioned as "Novel for Julia, rare for any language"

### Fixed
- Removed overclaiming ("100% accuracy", "revolutionary")
- Added explicit rejection rules for unsupported types
- Qualified all claims with technical constraints
- Documented ABI assumptions (x86_64 Linux System V)

## [1.0.0] - Initial Release

### Added
- Basic C++ compilation pipeline
- CMake project import
- LLVM toolchain detection
- Configuration management system
- Initial FFI generation capabilities

---

## Migration Guide

### From 1.x to 2.0

**Old way:**
```julia
RepliBuild.discover()
RepliBuild.build()
# ... now what?
RepliBuild.rwrap("julia/libproject.so", tier=:introspective)
```

**New way:**
```julia
RepliBuild.build()  # Compile
RepliBuild.wrap()   # Generate wrapper
# Done!
```

**Advanced users:**
If you were using internal functions directly, they're still available through module access:
```julia
RepliBuild.Compiler.compile_to_ir(config, files)
RepliBuild.Wrapper.wrap_library(config, lib_path)
RepliBuild.Discovery.discover(".", force=true)
```
