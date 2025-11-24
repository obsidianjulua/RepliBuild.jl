# RepliBuild Quick Start

## Complete Workflow (3 Steps)

### Step 0: Create Config (One Time)

**Option A: Auto-discover** (easiest)
```julia
using RepliBuild
RepliBuild.Discovery.discover()  # Scans your C++ project, generates replibuild.toml

# If replibuild.toml already exists, force regeneration:
RepliBuild.Discovery.discover(force=true)
```

**Option B: Write config manually:**
```toml
# replibuild.toml
[project]
name = "mylib"
root = "."

[compile]
source_files = ["src/mycode.cpp"]
include_dirs = ["include"]
flags = ["-std=c++17", "-fPIC", "-g"]

[binary]
type = "library"
```

### Step 1: Build

```julia
using RepliBuild
RepliBuild.build()  # Compiles C++ → library + extracts metadata
```

Output:
```
✓ Library: julia/libmylib.so
✓ Metadata saved
```

### Step 2: Wrap

```julia
RepliBuild.wrap()  # Generates Julia wrapper
```

Output:
```
✓ Wrapper: julia/Mylib.jl
  Functions wrapped: 5
```

### Step 3: Use It!

```julia
include("julia/Mylib.jl")
using .Mylib

# Call your C++ functions from Julia!
result = my_cpp_function(42)
```

## Complete Example

```bash
# In your C++ project directory
julia
```

```julia
using RepliBuild

# First time: discover project
RepliBuild.Discovery.discover()
# → Creates replibuild.toml

# Build library
RepliBuild.build()
# → Creates julia/libmyproject.so

# Generate wrapper
RepliBuild.wrap()
# → Creates julia/Myproject.jl

# Use it!
include("julia/Myproject.jl")
using .Myproject
```

## The 3 Commands You Need

### 1. `Discovery.discover()` - Create Config (once)
```julia
RepliBuild.Discovery.discover()
```
**What it does:**
- Scans for C++ files (*.cpp, *.cc, *.cxx)
- Detects include directories
- Generates replibuild.toml

### 2. `build()` - Compile C++
```julia
RepliBuild.build()
```
**What it does:**
- Compiles C++ → LLVM IR
- Links & optimizes
- Generates library (.so)
- Extracts metadata (DWARF + symbols)

### 3. `wrap()` - Generate Julia
```julia
RepliBuild.wrap()
```
**What it does:**
- Reads metadata
- Generates Julia module
- Creates ccall wrappers
- Extracts struct definitions (if -g flag)

## Checking Status

```julia
RepliBuild.info()
```

Shows:
```
Project: mylib

✓ Library: libmylib.so
✓ Wrapper: Mylib.jl
```

## Full Workflow Diagram

```
Your C++ Project
  ↓
RepliBuild.Discovery.discover()  ← One time setup
  ↓
replibuild.toml created
  ↓
RepliBuild.build()               ← Compile
  ↓
julia/libmyproject.so            ← Library
julia/compilation_metadata.json  ← Metadata
  ↓
RepliBuild.wrap()                ← Generate wrapper
  ↓
julia/Myproject.jl               ← Julia module
  ↓
include("julia/Myproject.jl")    ← Use it!
using .Myproject
```

## Common Patterns

### Pattern 1: Fresh Project
```julia
using RepliBuild

# Discover → Build → Wrap
RepliBuild.Discovery.discover()
RepliBuild.build()
RepliBuild.wrap()
```

### Pattern 2: Existing Config
```julia
using RepliBuild

# Already have replibuild.toml
RepliBuild.build()
RepliBuild.wrap()
```

### Pattern 3: Rebuild After Changes
```julia
using RepliBuild

# Changed C++ code? Just rebuild
RepliBuild.build()  # Recompiles
RepliBuild.wrap()   # Regenerates wrapper
```

### Pattern 4: Clean Build
```julia
using RepliBuild

RepliBuild.clean()   # Remove old artifacts
RepliBuild.build()   # Fresh compile
RepliBuild.wrap()    # Fresh wrapper
```

## Tips

### ✅ Do This
- Add `-g` flag for struct extraction
- Use `extern "C"` for C++ functions
- Keep structs simple (POD types)
- Run `discover()` once, then `build()` + `wrap()` as needed

### ❌ Don't Do This
- Forget to create replibuild.toml first
- Use virtual methods (not supported)
- Use STL containers (ABI-unstable)
- Skip the `-g` flag (won't extract structs)

## Troubleshooting

**"No replibuild.toml found"**
→ Run `RepliBuild.Discovery.discover()` first

**"Library not found"**
→ Run `RepliBuild.build()` before `wrap()`

**"No metadata"**
→ Add `-g` flag to compile.flags

**"Functions crash"**
→ Check you're using standard-layout types only

## Next Steps

- Read [examples/01_simple_math.md](examples/01_simple_math.md)
- Try it with your own C++ code
- See [USAGE.md](../USAGE.md) for complete guide
- Check [LIMITATIONS.md](../LIMITATIONS.md) for what's supported

---

**That's it! Go from C++ to Julia in 3 commands.** 🚀
