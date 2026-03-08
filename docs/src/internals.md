# RepliBuild Internals

This section documents the internal modules that power RepliBuild. These are generally not needed for standard usage but are valuable for contributors or advanced integration.

## Compiler

The `Compiler` module handles the translation of C++ source code into LLVM IR and shared libraries.

### Bitcode Assembly

`Compiler.assemble_bitcode(ll_path, bc_path)` converts a sanitized LLVM IR text file (`.ll`) to binary bitcode (`.bc`). It prefers `Clang_unified_jll.clang -emit-llvm` so the resulting bitcode exactly matches the LLVM version bundled with Julia, maximising `Base.llvmcall` compatibility. If the JLL is unavailable it falls back to the system `llvm-as`.

This function is called by both the main LTO pipeline (`link_optimize_ir`) and the AOT thunks path (`_build_aot_thunks`).

```@autodocs
Modules = [RepliBuild.Compiler]
Order = [:function, :type]
Private = false
```

## Configuration Manager

The `ConfigurationManager` is the single source of truth for all build settings, handling TOML parsing and validation.

```@autodocs
Modules = [RepliBuild.ConfigurationManager]
Order = [:function, :type]
Private = false
```

## Discovery

The `Discovery` module scans the filesystem to identify C++ source files, headers, and dependencies.

```@autodocs
Modules = [RepliBuild.Discovery]
Order = [:function, :type]
Private = false
```
