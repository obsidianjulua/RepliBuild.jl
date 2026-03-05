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
- Compilation of C++ to LLVM IR.
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

## Configuration

The `replibuild.toml` file controls the build process. You can edit this file to customize:
- Compiler flags
- Include directories
- Output names
- Optimization levels

See the **[Configuration Reference](config.md)** for a complete list of available options and sections.

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

## Template Instantiation

C++ templates are only emitted into DWARF if the compiler actually instantiates them. To force instantiation for types you want to wrap, add them to `[types]` in your config:

```toml
[types]
templates        = ["std::vector<int>", "std::vector<double>", "std::pair<int,float>"]
template_headers = ["<vector>", "<utility>"]
```

RepliBuild auto-generates a stub `.cpp` file that explicitly instantiates each requested type, ensuring it appears in the DWARF metadata and is available in the generated Julia module.

## Zero-Cost LTO Dispatch

When `enable_lto = true`, the linker emits both the shared library **and** an LLVM text IR file (`<name>_lto.ll`) in the `julia/` output directory.

```toml
[link]
enable_lto = true
optimization_level = "3"
```

The generated Julia wrapper loads this IR at module parse time:

```julia
const LTO_IR = isfile(LTO_IR_PATH) ? read(LTO_IR_PATH, String) : ""
```

For every eligible function (primitive/pointer types, no `Cstring`, no virtual dispatch, no struct-by-value return), the wrapper emits a dual-dispatch body:

```julia
function vector_dot(a::Ptr{Cvoid}, b::Ptr{Cvoid}, n::Cint)::Cdouble
    if !isempty(LTO_IR)
        return Base.llvmcall(("_Z10vector_dotPdPdi", LTO_IR),
                             Cdouble, Tuple{Ptr{Cvoid}, Ptr{Cvoid}, Cint},
                             a, b, n)
    else
        return ccall((:_Z10vector_dotPdPdi, LIBRARY_PATH),
                     Cdouble, (Ptr{Cvoid}, Ptr{Cvoid}, Cint),
                     a, b, n)
    end
end
```

When the IR is present, Julia's LLVM JIT merges the C++ IR directly into the calling Julia function's IR, enabling full cross-language inlining and vectorization. The `ccall` fallback fires automatically if the `.ll` file is missing (e.g., an LTO-disabled build was deployed).

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
