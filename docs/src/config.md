# Configuration Reference

The `replibuild.toml` file is the central nervous system of your build process. It allows you to customize everything from the project name to the strictness of type mappings in the generated wrapper.

This reference documents all available sections and keys.

## `[project]`

Basic project metadata.

| Key | Type | Description | Default |
|:--- |:---- |:----------- |:------- |
| `name` | String | The name of the project. Used for library naming. | Folder name |
| `root` | String | The root directory of the project. | `.` |
| `uuid` | String | Unique identifier for the project. | Auto-generated |

## `[paths]`

Locations for source code and build artifacts.

| Key | Type | Description | Default |
|:--- |:---- |:----------- |:------- |
| `source` | String | Directory containing C++ source files. | `"src"` |
| `include` | String | Directory containing C++ header files. | `"include"` |
| `output` | String | Directory where generated Julia wrappers are saved. | `"julia"` |
| `build` | String | Directory for intermediate build artifacts (IR, objects). | `"build"` |
| `cache` | String | Directory for caching build metadata. | `".replibuild_cache"` |

## `[discovery]`

Settings for the automatic source file discovery process.

| Key | Type | Description | Default |
|:--- |:---- |:----------- |:------- |
| `enabled` | Bool | Whether to run discovery automatically. | `true` |
| `walk_dependencies` | Bool | Whether to follow `#include` directives to find dependencies. | `true` |
| `max_depth` | Int | Maximum recursion depth for directory scanning. | `10` |
| `ignore_patterns` | Vector{String} | Files/folders to ignore during scan. | `["build", ".git", ".cache"]` |
| `parse_ast` | Bool | Whether to use Clang AST for deeper dependency analysis. | `true` |

## `[compile]`

Compiler settings for turning C++ into LLVM IR.

| Key | Type | Description | Default |
|:--- |:---- |:----------- |:------- |
| `flags` | Vector{String} | Flags passed to `clang++` (e.g., `["-O3", "-fPIC"]`). | `["-std=c++17", "-fPIC"]` |
| `defines` | Dict{String, String} | Preprocessor definitions (macros). | `{}` |
| `parallel` | Bool | Enable multi-threaded compilation. | `true` |
| `source_files` | Vector{String} | Explicit list of source files (overrides discovery). | `[]` |
| `include_dirs` | Vector{String} | Explicit list of include directories. | `[]` |
| `aot_thunks` | Bool | Pre-compile MLIR virtual-dispatch thunks into a static `_thunks.so` at build time. Eliminates JIT startup cost for virtual methods; requires the JLCS dialect to be built. | `false` |

## `[link]`

Settings for linking and optimizing the LLVM IR.

| Key | Type | Description | Default |
|:--- |:---- |:----------- |:------- |
| `optimization_level` | String | Optimization level (`"0"`, `"1"`, `"2"`, `"3"`, `"s"`, `"z"`). | `"2"` |
| `enable_lto` | Bool | Enable Link-Time Optimization. When `true`, emits `<name>_lto.bc` (LLVM bitcode) alongside the shared library. The generated Julia wrapper loads this bitcode at parse time and routes eligible functions through `Base.llvmcall` so Julia's JIT can inline C++ code directly into hot loops. Falls back to `ccall` automatically if the `.bc` file is absent. Bitcode is assembled via `Clang_unified_jll` to guarantee LLVM version compatibility with Julia's internal LLVM. | `false` |
| `link_libraries` | Vector{String} | External libraries to link against (e.g., `["stdc++fs"]`). | `[]` |

## `[binary]`

Settings for the final binary artifact.

| Key | Type | Description | Default |
|:--- |:---- |:----------- |:------- |
| `type` | String | Output type: `"shared"`, `"static"`, or `"executable"`. | `"shared"` |
| `output_name` | String | Custom name for the output file. | Auto-generated (e.g., `libProject.so`) |
| `strip_symbols` | Bool | Strip debug symbols from the final binary (reduces size). | `false` |

## `[wrap]`

Settings for the Julia wrapper generator.

| Key | Type | Description | Default |
|:--- |:---- |:----------- |:------- |
| `enabled` | Bool | Whether to generate the Julia wrapper. | `true` |
| `style` | String | Wrapping style: `"clang"` (full), `"basic"` (simple ccall), `"none"`. | `"clang"` |
| `module_name` | String | Name of the generated Julia module. | Project Name (CamelCase) |
| `use_clang_jl` | Bool | Use Clang.jl for AST parsing during wrapping. | `true` |
| `language` | String | Extensible dispatch key for the generator: `"c"` or `"cpp"`. | `"cpp"` |
| `shim_headers` | Vector{String} | Headers `#include`d in the auto-generated macro shim stub. | `[]` |

### `[wrap.varargs]`

Overrides for C varargs functions (`...`). Since Julia cannot easily call C varargs functions natively via `ccall` without knowing the exact types, you can define type-specific overloads here. RepliBuild will generate concrete function bindings for each signature, bypassing the need for manual shims.

```toml
[wrap.varargs]
printf = [
    ["const char*", "int"],
    ["const char*", "double", "int"]
]
```

### `[wrap.macros]`

Auto-generates typed C/C++ shims for preprocessor macros so they appear in DWARF metadata and can be safely wrapped as regular functions. This replaces the need to write your own wrapper functions for macros.

```toml
[wrap.macros.MY_MATH_MACRO]
ret = "int"
args = ["int", "float"]
```

## `[types]`

Control how C++ types map to Julia types. This is critical for FFI safety.

| Key | Type | Description | Default |
|:--- |:---- |:----------- |:------- |
| `strictness` | String | Type checking mode: `"strict"`, `"warn"`, `"permissive"`. | `"warn"` |
| `allow_unknown_structs` | Bool | Generate opaque pointers for unknown structs instead of failing. | `true` |
| `allow_unknown_enums` | Bool | Map unknown enums to `Int32`. | `false` |
| `allow_function_pointers` | Bool | Generate `Ptr{Cvoid}` for function pointers. | `true` |
| `custom` | Dict | Custom type mappings (e.g., `{"MyType" = "Int32"}`). | `{}` |
| `templates` | Vector{String} | C++ template instantiations to force-emit into DWARF (e.g., `["std::vector<int>"]`). RepliBuild auto-generates a stub `.cpp` that instantiates these types so they appear in metadata. | `[]` |
| `template_headers` | Vector{String} | Headers `#include`d in the auto-generated template stub (e.g., `["<vector>", "\"mylib.h\""]`). | `[]` |

### Strictness Modes

- **`strict`**: Fails the build if any type cannot be perfectly mapped.
- **`warn`**: Emits a warning for imperfect mappings but attempts to proceed (e.g., mapping `void*` for complex pointers).
- **`permissive`**: Silently falls back to `Ptr{Cvoid}` or `Any` for unknown types.

## `[llvm]`

LLVM toolchain selection.

| Key | Type | Description | Default |
|:--- |:---- |:----------- |:------- |
| `toolchain` | String | How to find LLVM: `"auto"`, `"system"`, `"jll"`. | `"auto"` |
| `version` | String | Specific LLVM version to use (if available). | `""` |

## `[workflow]`

Customize the build pipeline stages.

| Key | Type | Description | Default |
|:--- |:---- |:----------- |:------- |
| `stages` | Vector{String} | Order of operations. | `["discover", "compile", "link", "binary", "wrap"]` |

## `[cache]`

Build caching settings.

| Key | Type | Description | Default |
|:--- |:---- |:----------- |:------- |
| `enabled` | Bool | Enable build caching to skip unchanged files. | `true` |
| `directory` | String | Directory for cache files. | `".replibuild_cache"` |

## `[dependencies]`

RepliBuild can automatically fetch, filter, and compile external C/C++ libraries from git repositories, local paths, or system packages — no BinaryBuilder or JLL packages required.

Each dependency is declared as a named sub-table:

```toml
[dependencies.cjson]
type    = "git"
url     = "https://github.com/DaveGamble/cJSON"
tag     = "v1.7.18"
exclude = ["test", "fuzzing", "CMakeLists.txt"]

[dependencies.mylocal]
type = "local"
path = "../vendor/mylib"
exclude = ["docs"]

[dependencies.zlib]
type = "system"
pkg_config = "zlib"
```

### Dependency item fields

| Key | Type | Description | Default |
|:--- |:---- |:----------- |:------- |
| `type` | String | Source kind: `"git"`, `"local"`, or `"system"`. | — (required) |
| `url` | String | Git clone URL. Required when `type = "git"`. | `""` |
| `tag` | String | Git tag or branch to check out. | `""` (uses default branch) |
| `path` | String | Filesystem path. Required when `type = "local"`. | `""` |
| `pkg_config` | String | `pkg-config` package name. Used when `type = "system"` to resolve include/link flags. | `""` |
| `exclude` | Vector{String} | Files or subdirectories to skip during source injection (glob-matched against relative paths). Useful for silencing test files, build scripts, and unrelated C files in single-directory amalgamations. | `[]` |

### How it works

1. **`git` dependencies** are cloned into `.replibuild_cache/deps/<name>/` on first use and updated to the requested `tag` on subsequent builds. The clone is shallow (`--depth 1`) when a tag is specified.
2. **`local` dependencies** are scanned in-place. No copying is performed.
3. **`system` dependencies** run `pkg-config --cflags` to inject include paths into the compile flags.

In all cases, resolved source files are merged into the compilation graph before the `[compile]` step runs. The `exclude` list is applied after scanning, so you can trim large repos down to just the files you need.

### Example: wrapping cJSON from git

```toml
[project]
name = "my_cjson_wrapper"

[dependencies.cjson]
type    = "git"
url     = "https://github.com/DaveGamble/cJSON"
tag     = "v1.7.18"
exclude = ["test", "fuzzing"]

[types]
allow_unknown_structs = true
```

After `RepliBuild.build()` and `RepliBuild.wrap()`, the generated module exposes the full cJSON C API as type-safe Julia functions without any manual binding work.
