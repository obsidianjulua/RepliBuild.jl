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

## `[link]`

Settings for linking and optimizing the LLVM IR.

| Key | Type | Description | Default |
|:--- |:---- |:----------- |:------- |
| `optimization_level` | String | Optimization level (`"0"`, `"1"`, `"2"`, `"3"`, `"s"`, `"z"`). | `"2"` |
| `enable_lto` | Bool | Enable Link Time Optimization (LTO). | `false` |
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

## `[types]`

Control how C++ types map to Julia types. This is critical for FFI safety.

| Key | Type | Description | Default |
|:--- |:---- |:----------- |:------- |
| `strictness` | String | Type checking mode: `"strict"`, `"warn"`, `"permissive"`. | `"warn"` |
| `allow_unknown_structs` | Bool | Generate opaque pointers for unknown structs instead of failing. | `true` |
| `allow_unknown_enums` | Bool | Map unknown enums to `Int32`. | `false` |
| `allow_function_pointers` | Bool | Generate `Ptr{Cvoid}` for function pointers. | `true` |
| `custom` | Dict | Custom type mappings (e.g., `{"MyType" = "Int32"}`). | `{}` |

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
