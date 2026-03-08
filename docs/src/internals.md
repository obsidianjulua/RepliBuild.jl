# RepliBuild Internals

This section documents the internal modules that power RepliBuild. These are generally not needed for standard usage but are valuable for contributors or advanced integration.

## Wrapper

The `Wrapper` package generates Julia FFI modules from DWARF metadata and binary symbol tables. It is structured as a two-track system: a C generator and a C++ generator, selected automatically via `config.wrap.language`.

### Module Layout

| Module | Role |
|--------|------|
| `Wrapper.Generator` | Top-level `wrap_library()` entry point; dispatches to C or C++ generator |
| `Wrapper.TypeRegistry` | `TypeRegistry` and `TypeStrictness` — shared type-resolution context |
| `Wrapper.Symbols` | `ParamInfo` / `SymbolInfo` structs for structured symbol data |
| `Wrapper.FunctionPointers` | DWARF `function_ptr(...)` signature → Julia `@cfunction` type string |
| `Wrapper.Utils` | Keyword escaping, identifier sanitization shared between generators |
| `Wrapper.C.GeneratorC` | Full C wrapper generator (structs, enums, functions, LTO, thunks) |
| `Wrapper.C.TypesC` | C type heuristics and base type map |
| `Wrapper.Cpp.GeneratorCpp` | Full C++ wrapper generator (same feature set + virtual dispatch) |
| `Wrapper.Cpp.TypesCpp` | C++ type map including STL, templates, references |
| `Wrapper.Cpp.IdentifiersCpp` | Namespace stripping, operator sanitization |

### Language Selection

```toml
[wrap]
language = "c"   # selects C generator + clang toolchain
language = "cpp" # selects C++ generator + clang++ toolchain (default)
```

`discover()` sets this automatically based on the scanned source files.

## Compiler

The `Compiler` module handles the translation of C/C++ source code into LLVM IR and shared libraries.

### Language-Aware Compilation

`.c` files are compiled with `clang`; `.cpp` files with `clang++`. For C projects, `create_library()` and `create_executable()` also use `clang` as the linker driver.

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

The `Discovery` module scans the filesystem to identify C/C++ source files, headers, and dependencies. It now auto-detects project language (`:c` vs `:cpp`) from the scanned source extensions and sets `wrap.language` accordingly in the generated `replibuild.toml`.

```@autodocs
Modules = [RepliBuild.Discovery]
Order = [:function, :type]
Private = false
```
