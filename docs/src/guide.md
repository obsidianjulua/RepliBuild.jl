# User Guide

This guide covers the common workflows for using `RepliBuild.jl`.

## Basic Workflow

The standard workflow involves three steps: discovery, building, and wrapping.

### 1. Discovery

The `discover` function scans your directory for C++ files and generates a `replibuild.toml` configuration file.

```julia
RepliBuild.discover()
```

You can also specify a directory:

```julia
RepliBuild.discover("path/to/project")
```

### 2. Building

Once configured, the `build` function compiles your C++ code into a shared library.

```julia
RepliBuild.build()
```

This step performs:
- Compilation of C++ to LLVM IR (`.c` files use `clang`, `.cpp` files use `clang++`).
- Linking and optimization.
- Generation of the shared library.
- Extraction of metadata for wrapping.

### 3. Wrapping

Finally, generate the Julia wrapper module.

```julia
RepliBuild.wrap()
```

This will create a Julia file in the `julia/` directory that you can `include` and `use`.

## Automated Workflow

You can chain these steps together using the flags in `discover`:

```julia
# Discover, Build, and Wrap in one go
RepliBuild.discover(build=true, wrap=true)
```

## Package Registry

RepliBuild includes a global package registry (`~/.replibuild/registry/`) that caches build artifacts so repeated loads are instant.

```julia
# Register a project (discover() does this automatically)
RepliBuild.register("replibuild.toml")

# Load a registered package — builds on first call, cached thereafter
Lua = RepliBuild.use("lua_wrapper")

# List all registered packages with hash, source, and build status
RepliBuild.list_registry()

# Remove a package from the registry and clean cached builds
RepliBuild.unregister("lua_wrapper")

# Scaffold a distributable Julia package from a registered project
RepliBuild.scaffold_package("LuaWrapper")
```

The `use()` function handles the full lifecycle: resolve dependencies, build (or load from cache), wrap, and return a loaded Julia module. The `REPLIBUILD_HOME` environment variable can override the default registry location.

## Configuration

The `replibuild.toml` file controls the build process. You can edit this file to customize:
- Compiler flags
- Include directories
- Output names
- Optimization levels

See the **[Configuration Reference](config.md)** for a complete list of available options and sections.

## C vs C++ Projects

RepliBuild uses `wrap.language` as an extensible dispatch key to select the generator, compiler toolchain, and build defaults for a project. `"c"` and `"cpp"` are the first two targets:

```toml
[wrap]
language = "c"   # pure-C project: uses clang, LTO on by default
language = "cpp" # C++ project (default): uses clang++
```

`discover()` sets this automatically from the scanned source file extensions. For C projects the `enable_lto` default is `true`, so you get zero-cost `llvmcall` dispatch out of the box without any extra configuration.

Additional language targets are planned — the `language` field is the hook that will route each new language to its own generator and toolchain.

## Git & External Dependencies

RepliBuild can automatically pull external C/C++ libraries from git, local paths, or your system into the compilation pipeline. Declare them in `replibuild.toml` under `[dependencies]`:

```toml
[dependencies.cjson]
type    = "git"
url     = "https://github.com/DaveGamble/cJSON"
tag     = "v1.7.18"
exclude = ["test", "fuzzing"]
```

Run the normal pipeline — the dependency is cloned, filtered, and compiled transparently:

```julia
RepliBuild.build("replibuild.toml")
RepliBuild.wrap("replibuild.toml")
```

The first build clones into `.replibuild_cache/deps/cjson/`. Subsequent builds are cached and only re-clone when the `tag` changes.

For a local library:

```toml
[dependencies.mylib]
type    = "local"
path    = "../vendor/mylib"
exclude = ["docs", "examples"]
```

For a system library (uses `pkg-config`):

```toml
[dependencies.zlib]
type       = "system"
pkg_config = "zlib"
```

See the **[Configuration Reference](config.md#dependencies)** for all fields.

## Idiomatic Julia Class Wrappers

When your C++ library exposes a class through factory/destructor pairs, RepliBuild automatically generates an idiomatic `mutable struct` wrapper on top of the raw FFI bindings.

**Detection**: the wrapper generator scans for:
- Factory functions whose name matches `create_X`, `new_X`, `make_X`, `alloc_X`, `init_X`, or whose return type is `X*`.
- Destructor/deleter functions whose name matches `delete_X`, `destroy_X`, `free_X`, `dealloc_X`, or `X_destroy`.
- Instance methods associated with the same class via the DWARF `class` field.

**Generated output** for a `Circle` class:

```julia
# Raw bindings (always generated)
function create_circle(radius::Cdouble)::Ptr{Cvoid} ... end
function get_area(this::Ptr{Cvoid})::Cdouble ... end
function delete_shape(this::Ptr{Cvoid}) ... end

# Idiomatic wrapper (generated automatically on top)
mutable struct Circle
    handle::Ptr{Cvoid}

    function Circle(radius::Cdouble)
        handle = create_circle(radius)
        obj = new(handle)
        finalizer(obj) do o
            delete_shape(o.handle)
        end
        return obj
    end
end

# Method proxies via multiple dispatch
get_area(c::Circle) = get_area(c.handle)
```

User code needs no manual memory management:

```julia
c = Circle(5.0)      # allocates C++ object, registers GC finalizer
get_area(c)          # dispatch on Circle type, no .handle needed
# c goes out of scope → GC calls delete_shape automatically
```

## Replacing Manual Shims

When wrapping C/C++ libraries, developers often have to write manual C wrappers ("shims") for things that aren't native functions: templates, varargs, and preprocessor macros. RepliBuild handles all of these automatically via `replibuild.toml` without you having to write a single line of C/C++ code.

### Template Instantiation

C++ templates are only emitted into DWARF if the compiler actually instantiates them. To force instantiation for types you want to wrap, add them to `[types]` in your config:

```toml
[types]
templates        = ["std::vector<int>", "std::vector<double>", "std::pair<int,float>"]
template_headers = ["<vector>", "<utility>"]
```

RepliBuild auto-generates a stub `.cpp` file that explicitly instantiates each requested type, ensuring it appears in the DWARF metadata and is available in the generated Julia module.

### Varargs Interception

Julia's `ccall` cannot call C `...` (varargs) functions natively without knowing the exact types at the call site. Instead of writing a manual C wrapper for each type combination, you can configure overloads in `[wrap.varargs]`:

```toml
[wrap.varargs]
printf = [
    ["const char*", "int"],
    ["const char*", "double", "int"]
]
```

RepliBuild generates concrete Julia bindings for `printf` for each of these signatures, completely bypassing the varargs limitation.

### Macro Expansion

C/C++ preprocessor macros don't exist in compiled binaries or DWARF metadata. To expose them to Julia, you can configure `[wrap.macros]`:

```toml
[wrap]
shim_headers = ["<stdio.h>"]

[wrap.macros.MY_MATH_MACRO]
ret = "int"
args = ["int", "float"]
```

RepliBuild automatically generates a C/C++ source file that wraps `MY_MATH_MACRO` inside a typed function and compiles it alongside your project. The resulting wrapper gives you a native Julia function.

## Zero-Cost LTO Dispatch

When `enable_lto = true` (or for C projects, where it is the default), the linker emits both the shared library **and** LLVM bitcode (`<name>_lto.bc`) in the `julia/` output directory.

```toml
[link]
enable_lto = true
optimization_level = "3"
```

The generated Julia wrapper loads the bitcode at module parse time:

```julia
const LTO_IR_PATH = joinpath(@__DIR__, "mylib_lto.bc")
const LTO_IR = isfile(LTO_IR_PATH) ? read(LTO_IR_PATH) : UInt8[]
```

For every eligible function (primitive/pointer types, no `Cstring`, no virtual dispatch, no struct-by-value return), the wrapper emits a dual-dispatch body:

```julia
function vector_dot(a::Ptr{Cvoid}, b::Ptr{Cvoid}, n::Cint)::Cdouble
    if !isempty(LTO_IR)
        return Base.llvmcall((LTO_IR, "_Z10vector_dotPdPdi"),
                             Cdouble, Tuple{Ptr{Cvoid}, Ptr{Cvoid}, Cint},
                             a, b, n)
    else
        return ccall((:_Z10vector_dotPdPdi, LIBRARY_PATH),
                     Cdouble, (Ptr{Cvoid}, Ptr{Cvoid}, Cint),
                     a, b, n)
    end
end
```

When the bitcode is present, Julia's LLVM JIT merges the C++ IR directly into the calling Julia function's IR, enabling full cross-language inlining and vectorization. The `ccall` fallback fires automatically if the `.bc` file is missing (e.g., an LTO-disabled build was deployed).

## AOT Thunks for Virtual Dispatch

Virtual method dispatch normally requires the MLIR JIT to compile thunks at runtime. Setting `aot_thunks = true` pre-compiles those thunks at build time:

```toml
[compile]
aot_thunks = true
```

During `RepliBuild.build()`, the JLCS MLIR dialect generates and compiles all virtual dispatch thunks into a companion `<name>_thunks.so` placed alongside the main library. The generated wrapper emits purely static `ccall` bindings that load from `THUNKS_LIBRARY_PATH` — no `JITManager` startup, no lock, no on-demand compilation:

```julia
function Circle_area(this::Ptr{Cvoid})::Cdouble
    return ccall((:thunk_Circle_area, THUNKS_LIBRARY_PATH), Cdouble, (Ptr{Cvoid},), this)
end
```

This is the recommended setting for production deployments where predictable latency matters. Requires `src/mlir/build/libJLCS.so` to be built first (`cd src/mlir && ./build.sh`).

## Running Tests

The CI suite (stress test + MLIR unit tests + registry tests):

```bash
julia --project=. test/runtests.jl
```

The full developer integration suite (Lua, SQLite, cJSON, Duktape, vtable, JIT edge cases):

```bash
julia --project=. test/devtests.jl
```

External sources are downloaded on first run via `setup.jl` scripts.
