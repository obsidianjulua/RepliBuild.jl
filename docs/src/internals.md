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

`wrap.language` is an extensible dispatch key — `"c"` and `"cpp"` are the first two targets, with additional language generators planned:

```toml
[wrap]
language = "c"   # selects C generator + clang toolchain
language = "cpp" # selects C++ generator + clang++ toolchain (default)
```

`discover()` sets this automatically based on the scanned source files. Adding a new language means adding a generator under `src/Wrapper/<Lang>/` and registering it in `Wrapper/Generator.jl`.

## Compiler

The `Compiler` module handles the translation of C/C++ source code into LLVM IR and shared libraries. It oversees the entire build pipeline from high-level dependency management down to low-level IR optimization.

### Build Pipeline

From a high level down to the lowest level, the build process involves:

1. **Auto-Discovery & Dependency Resolution**: Scans the project directory, resolving file paths and external Git/local dependencies to merge into the build graph.
2. **Pre-processing (Shims & Templates)**: Dynamically generates C/C++ shim files for configured macros and explicitly instantiates templates based on the `replibuild.toml` settings. This allows normally invisible constructs to manifest in the final binary and DWARF metadata.
3. **Compilation to LLVM IR**: Translates source code into `.ll` text format via `clang`/`clang++`.
4. **IR Transformation & Sanitization**: To ensure compatibility with Julia's internal LLVM JIT and `llvmcall` dispatch, the compiler applies strict transformations to the LLVM IR. Crucially, it identifies and removes `varargs` (`...`) function bodies—stripping `va_start`/`va_end` intrinsics that cannot be safely JIT-resolved by MLIR/Julia. (Varargs functions are intercepted and routed entirely through direct `ccall` wrapper generation instead). It also strips out mismatched LLVM 19+ attributes and debug metadata.
5. **Bitcode Assembly**: The sanitized IR is converted into `.bc` binary format for fast, zero-cost LTO loading in Julia.
6. **Linking**: Finally, object files are linked into the target `.so`/`.dylib`/`.dll` shared library.

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
