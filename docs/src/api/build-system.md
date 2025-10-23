# Build System API

Functions for build system integration and delegation.

## Build Functions

```@docs
RepliBuild.build
RepliBuild.detect_build_system
RepliBuild.delegate_build
```

### Usage

```julia
# Universal build
RepliBuild.build()

# Build specific project
RepliBuild.build("/path/to/project")

# Detect build system
system = RepliBuild.detect_build_system(".")
println("Detected: $system")

# Delegate to specific build system
artifacts = RepliBuild.delegate_build(".", toml_path="replibuild.toml")
```

## Tool Discovery

```@docs
RepliBuild.discover_tools
```

### Usage

```julia
# Discover LLVM/Clang tools
RepliBuild.discover_tools()

# With custom config
RepliBuild.discover_tools("custom.toml")
```

## Function Reference

### build

```julia
build(project_dir::String="."; config_file::String="replibuild.toml")
```

Universal build function that intelligently delegates to the appropriate build system.

**Workflow:**
1. Reads `replibuild.toml` to determine build system
2. Uses Julia artifacts (JLL packages) when available
3. Falls back to system tools
4. Returns build artifacts

**Arguments:**
- `project_dir::String`: Project directory to build
- `config_file::String`: Configuration file name

**Returns:** Dict with keys:
- `:libraries` - Array of built library paths
- `:executables` - Array of built executable paths
- `:build_dir` - Build directory path

**Configuration:**

Add to `replibuild.toml`:
```toml
[build]
system = "qmake"  # or "cmake", "meson", "autotools", "make"
qt_version = "Qt5"
build_dir = "build"
```

**Examples:**
```julia
# Auto-detect build system
RepliBuild.build()

# Build Qt project
RepliBuild.build("qt_project")

# Custom config
RepliBuild.build(".", config_file="release.toml")
```

---

### detect_build_system

```julia
detect_build_system(project_dir::String=".")
```

Detect which build system a project uses.

**Detection order:**
1. CMake - checks for `CMakeLists.txt`
2. qmake - checks for `*.pro` files
3. Meson - checks for `meson.build`
4. Autotools - checks for `configure.ac`/`configure.in`
5. Make - checks for `Makefile`/`makefile`

**Arguments:**
- `project_dir::String`: Project directory to analyze

**Returns:** Symbol - `:cmake`, `:qmake`, `:meson`, `:autotools`, `:make`, or `:unknown`

**Examples:**
```julia
system = RepliBuild.detect_build_system(".")
if system == :cmake
    println("CMake project detected")
elseif system == :qmake
    println("Qt/qmake project detected")
end
```

---

### delegate_build

```julia
delegate_build(project_dir::String; toml_path::String="replibuild.toml")
```

Delegate build to detected or configured build system.

**Arguments:**
- `project_dir::String`: Project directory
- `toml_path::String`: Path to configuration file

**Returns:** Dict with build artifacts

**Examples:**
```julia
# Delegate to auto-detected system
artifacts = RepliBuild.delegate_build(".")

println("Built libraries: ", artifacts[:libraries])
println("Built executables: ", artifacts[:executables])
```

---

### discover_tools

```julia
discover_tools(config_file::String="replibuild.toml")
```

Discover LLVM/Clang tools available on the system.

Searches for:
- `clang`
- `clang++`
- `llvm-config`
- `llvm-ar`
- `llvm-nm`

**Arguments:**
- `config_file::String`: Configuration file path

**Returns:** Nothing (prints discovered tools)

**Examples:**
```julia
RepliBuild.discover_tools()
```

Output:
```
🔍 RepliBuild - Discovering LLVM tools
✅ Found clang: /usr/bin/clang
✅ Found clang++: /usr/bin/clang++
✅ Found llvm-config: /usr/bin/llvm-config-14
```

---

## Build System Specific

### CMake

```julia
# Configuration
[build]
system = "cmake"
build_dir = "build"
cmake_options = ["-DCMAKE_BUILD_TYPE=Release"]
```

Functions:
- Runs `cmake -B build [options] .`
- Runs `cmake --build build`
- Locates built libraries in `build/`

### qmake (Qt)

```julia
# Configuration
[build]
system = "qmake"
qt_version = "Qt5"
pro_file = "myapp.pro"
```

Functions:
- Runs `qmake myapp.pro -o build/`
- Runs `make -C build`
- Locates built libraries

### Meson

```julia
# Configuration
[build]
system = "meson"
build_dir = "builddir"
meson_options = ["-Dbuildtype=release"]
```

Functions:
- Runs `meson setup builddir [options]`
- Runs `ninja -C builddir`

### Autotools

```julia
# Configuration
[build]
system = "autotools"
configure_options = ["--prefix=/usr/local"]
bootstrap = true
```

Functions:
- Optionally runs `autoreconf -i`
- Runs `./configure [options]`
- Runs `make`

### Make

```julia
# Configuration
[build]
system = "make"
make_targets = ["all"]
make_options = ["-j4"]
```

Functions:
- Runs `make [targets] [options]`

---

## JLL Integration

Using Julia JLL packages for build tools:

```julia
# Automatic JLL usage
[build]
use_jll = true

[build.jll_packages]
cmake = "CMAKE_jll"
ninja = "Ninja_jll"
qt5 = "Qt5Base_jll"
```

RepliBuild automatically loads and uses JLL-provided tools.

---

## Environment Variables

Set build environment:

```julia
[build.environment]
CC = "clang"
CXX = "clang++"
CFLAGS = "-O3 -march=native"
CXXFLAGS = "-O3 -march=native"
LDFLAGS = "-Wl,-rpath=/usr/local/lib"
PKG_CONFIG_PATH = "/usr/local/lib/pkgconfig"
```

---

## Platform-Specific

Override build configuration per platform:

```julia
[build.linux]
system = "cmake"
cmake_options = ["-DLINUX=ON"]
environment = { CC = "gcc" }

[build.macos]
system = "cmake"
cmake_options = ["-DCMAKE_OSX_DEPLOYMENT_TARGET=10.15"]

[build.windows]
system = "cmake"
cmake_options = ["-G", "Visual Studio 16 2019"]
```

---

## Error Handling

Build functions may throw:
- `ErrorException`: Build failed
- `SystemError`: Build tool not found
- `ArgumentError`: Invalid configuration

Handle build errors:

```julia
try
    artifacts = RepliBuild.build()
    println("Build successful!")
catch e
    if isa(e, ErrorException)
        println("Build failed: ", e.msg)
        # Check build log
    else
        rethrow(e)
    end
end
```

---

## See Also

- **[Core API](core.md)**: Core functions
- **[Build Systems Guide](../guide/build-systems.md)**: Build system integration guide
- **[Configuration](../guide/configuration.md)**: Build configuration reference
