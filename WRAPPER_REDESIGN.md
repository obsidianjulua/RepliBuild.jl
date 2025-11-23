# Wrapper Module Redesign - Enterprise-Grade Binding Generation

## Vision

The Wrapper module is RepliBuild's **crown jewel** - what sets it apart from all other C++ → Julia tools. Users come to RepliBuild because:

1. They want to **avoid writing bindings by hand**
2. They need **type-safe, production-ready wrappers** automatically
3. They want **one command** to go from C++ lib → usable Julia package
4. They need it to **just work** without deep FFI knowledge

## Current State Analysis

**What we have** (Wrapper.jl - 257 LOC):
- ✅ Two modes: Clang.jl (type-aware) vs basic (symbol-only)
- ✅ Symbol extraction via `nm`
- ✅ Basic Julia identifier sanitization
- ✅ Integration with ClangJLBridge

**Critical gaps**:
- ❌ Type inference is basically absent (defaults to `Any`)
- ❌ No C++ → Julia type registry
- ❌ No safety checks or library load validation
- ❌ No test generation
- ❌ No documentation generation
- ❌ Single symbol extraction method (only nm)
- ❌ No parameter name/type inference
- ❌ No signature parsing from symbols
- ❌ No binary metadata extraction
- ❌ Generated code is barebone skeletons

## Architecture: Three-Tier Wrapping Strategy

### Tier 1: Basic (Symbol-Only)
**When**: No headers available, binary-only wrapping
**Quality**: ~40% - Usable but requires manual type fixes
**Features**:
- Symbol extraction via nm
- Julia identifier generation
- Placeholder ccall signatures (conservative types)
- Library load management
- Safety macros
- Basic docs stubs

**Output example**:
```julia
module MyLib

const _LIB = Ref{Ptr{Nothing}}(C_NULL)
function __init__()
    _LIB[] = dlopen("libmylib.so")
end

macro check_loaded()
    quote
        _LIB[] == C_NULL && error("Library not loaded")
    end
end

"""
    compute_score(arg1, arg2)

Wrapper for C++ function `compute_score`.
Original signature: Unknown (inferred from binary)

# Type safety: ⚠️  BASIC
Types are conservative placeholders. Consider regenerating with headers for type safety.
"""
function compute_score(arg1::Any, arg2::Any)::Any
    @check_loaded()
    ccall((:compute_score, _LIB[]), Any, (Any, Any), arg1, arg2)
end

export compute_score

end
```

### Tier 2: Advanced (Header-Assisted)
**When**: Headers available, use Clang.jl parser
**Quality**: ~85% - Production-ready with minor tweaks
**Features**:
- Full Clang.jl AST parsing
- Accurate C/C++ type → Julia type mapping
- Parameter names from headers
- Function signatures with docs
- Enum/struct/typedef generation
- Safety checks
- Generated tests
- Full API documentation

**Output example**:
```julia
"""
    compute_score(data::Ptr{Float64}, size::Csize_t)::Cdouble

Compute statistical score from data array.

# Arguments
- `data::Ptr{Float64}`: Pointer to data array
- `size::Csize_t`: Number of elements in array

# Returns
- `Cdouble`: Computed score value

# Safety
This function validates library loading before execution.

# C++ Signature
```cpp
double compute_score(const double* data, size_t size);
```

# Type safety: ✅ FULL
Types inferred from headers via Clang.jl.
"""
function compute_score(data::Ptr{Float64}, size::Csize_t)::Cdouble
    @check_loaded()
    ccall((:compute_score, _LIB[]), Cdouble, (Ptr{Float64}, Csize_t), data, size)
end
```

### Tier 3: Introspective (Metadata-Aware)
**When**: Compiled with RepliBuild from source
**Quality**: ~95% - Full IDE support, test coverage, examples
**Features**:
- All Tier 2 features +
- Inherits type mappings from compilation metadata
- Cross-library dependency resolution
- Automatic test generation with sample data
- Usage examples in documentation
- Integration with Julia type system
- Validation layer for complex types

## Core Components

### 1. Type Inference Engine

**Type Registry** - Comprehensive C/C++ → Julia mapping:

```julia
struct TypeRegistry
    # Core mappings
    base_types::Dict{String,String}      # int → Cint, etc
    stl_types::Dict{String,String}       # std::string → String
    custom_types::Dict{String,String}    # User types from headers

    # Advanced mappings
    template_rules::Vector{TemplateRule} # std::vector<T> → Vector{T}
    pointer_rules::Dict{String,String}   # T* → Ptr{T}
    const_rules::Dict{String,String}     # const T → T (immutable)
    reference_rules::Dict{String,String} # T& → Ref{T}

    # Metadata from compilation
    compilation_metadata::Union{Nothing,Dict}
end
```

**Default mappings** (231 built-in types):
- C primitives: void, char, int, long, float, double, etc
- C sized types: int8_t, uint32_t, size_t, ptrdiff_t, etc
- C++ primitives: bool, wchar_t, char16_t, char32_t
- C++ STL basics: std::string, std::string_view
- Pointers: T* → Ptr{T}, const T* → Ptr{T}
- References: T& → Ref{T}, const T& → Ref{T}
- Arrays: T[N] → NTuple{N,T}

**Type inference algorithm**:
1. Check compilation metadata (if available)
2. Check type registry exact match
3. Apply template rules (std::vector<int> → Vector{Int32})
4. Apply pointer/reference rules
5. Parse complex types (function pointers, nested templates)
6. Conservative fallback: Ptr{Cvoid} for pointers, Any for unknown

### 2. Symbol Extraction Engine

**Multi-method extraction**:

```julia
struct SymbolInfo
    name::String              # Mangled/demangled name
    julia_name::String        # Valid Julia identifier
    type::Symbol              # :function, :data, :weak
    signature::String         # Raw C++ signature (if available)
    return_type::String       # Parsed return type
    parameters::Vector{ParamInfo}
    visibility::Symbol        # :public, :private, :protected
    source_file::String       # From debug symbols
    line_number::Int          # From debug symbols
    metadata::Dict{String,Any}
end

struct ParamInfo
    name::String
    type::String
    default::Union{String,Nothing}
    is_const::Bool
    is_reference::Bool
    is_pointer::Bool
end
```

**Extraction methods**:
1. **nm**: Fast, basic symbols (name + type classification)
2. **objdump**: Detailed with debug info (line numbers, file names)
3. **Clang.jl**: Full AST parsing from headers (complete signatures)
4. **dwarf info**: Debug symbols for source correlation

### 3. Wrapper Generation Templates

**Module structure**:

```julia
# Header with metadata
# - Generated timestamp
# - RepliBuild version
# - Source library info
# - Quality tier indicator

module {ModuleName}

using Libdl

# === LIBRARY MANAGEMENT ===
const _LIB_PATH = raw"{path}"
const _LIB = Ref{Ptr{Nothing}}(C_NULL)
const _LOAD_ERRORS = String[]

function __init__()
    # Load library with error handling
end

# === UTILITIES ===
is_loaded() = _LIB[] != C_NULL
get_load_errors() = copy(_LOAD_ERRORS)
get_lib_path() = _LIB_PATH

# === SAFETY MACROS ===
macro check_loaded()
    # Validation macro
end

# === TYPE DEFINITIONS ===
# Enums, structs, typedefs from headers

# === FUNCTION WRAPPERS ===
# Generated ccall wrappers

# === DATA ACCESSORS ===
# Global variable accessors

# === METADATA ===
function library_info()
    # Return library metadata dict
end

# === EXPORTS ===
export ...

end # module
```

### 4. Test Generation

Auto-generate tests for each wrapped function:

```julia
@testset "{ModuleName} Library Tests" begin
    @testset "Library Loading" begin
        @test {ModuleName}.is_loaded()
        @test isempty({ModuleName}.get_load_errors())
    end

    @testset "Function: compute_score" begin
        # Type validation
        @test hasmethod(compute_score, (Ptr{Float64}, Csize_t))

        # Basic execution (if safe)
        # data = [1.0, 2.0, 3.0]
        # result = compute_score(pointer(data), length(data))
        # @test result isa Cdouble
    end

    # ... more function tests
end
```

### 5. Documentation Generation

Generate comprehensive docs:

```julia
\"\"\"
# {ModuleName}

Auto-generated Julia bindings for `lib{libname}.so`

## Generation Info
- **Generated**: {timestamp}
- **RepliBuild Version**: {version}
- **Quality Tier**: {tier} ({percentage}% type coverage)
- **Library Path**: `{path}`
- **Functions Wrapped**: {count}

## Quick Start

```julia
using {ModuleName}

# Check library loaded
@assert {ModuleName}.is_loaded()

# Call functions
result = compute_score(data_ptr, size)
```

## API Reference

### Functions

{generated function docs...}

### Data

{generated data accessor docs...}

## Library Information

```julia
julia> {ModuleName}.library_info()
Dict{Symbol, Any} with 7 entries:
  :name         => "mylib"
  :loaded       => true
  :functions    => 42
  :architecture => "x86_64"
  ...
```

## Safety Notes

This library includes runtime safety checks. All functions verify library
loading before execution. Disable with `{ModuleName}.SAFETY_CHECKS[] = false`.

\"\"\"
module {ModuleName}
```

## Integration Points

### With ConfigurationManager

```julia
# Wrapper settings in RepliBuildConfig
struct WrapConfig
    enabled::Bool
    tier::Symbol              # :basic, :advanced, :introspective
    generate_tests::Bool
    generate_docs::Bool
    safety_checks::Bool
    type_inference::Bool
    symbol_methods::Vector{Symbol}  # [:nm, :objdump, :clang]
    header_discovery::Bool
    type_hints::Dict{String,String}
end
```

### With Compiler Module

- Inherit type mappings from compilation metadata
- Use same include directories for header parsing
- Link compiled library paths automatically
- Preserve compiler flags for Clang.jl parsing

### With ClangJLBridge

- Use ClangJLBridge for Tier 2/3 header parsing
- Enhance with our type registry
- Add our safety/test/doc layers on top
- Preserve Clang.jl's struct/enum generation

## User API

### Simple (Auto-detect tier)

```julia
using RepliBuild

# Build and wrap in one go
RepliBuild.build()  # Compiles to library
RepliBuild.wrap()   # Auto-detects headers, generates Tier 2/3

# Just wrap existing library
RepliBuild.wrap("libmylib.so")  # Tier 1 (basic)
RepliBuild.wrap("libmylib.so", headers=["mylib.h"])  # Tier 2 (advanced)
```

### Advanced (Explicit control)

```julia
using RepliBuild

# Load config
config = RepliBuild.load_config("replibuild.toml")

# Customize wrapper settings
config = with_wrapper_tier(config, :introspective)
config = with_wrapper_tests(config, true)
config = with_wrapper_docs(config, true)

# Generate wrappers
RepliBuild.wrap(config, "libmylib.so")
```

### REPL API

```julia
rwrap("libmylib.so")                              # Tier 1: basic
rwrap("libmylib.so", headers=["mylib.h"])         # Tier 2: advanced
rwrap("libmylib.so", tier=:introspective)         # Tier 3: full metadata
rwrap("libmylib.so", tests=true, docs=true)       # With extras
```

## Implementation Plan

### Phase 1: Type System (2-3 days)
- [ ] Create TypeRegistry with comprehensive mappings
- [ ] Implement type inference algorithm
- [ ] Add template/pointer/reference rules
- [ ] Test with common C/C++ types

### Phase 2: Symbol Extraction (2-3 days)
- [ ] Enhance nm extraction with signature parsing
- [ ] Add objdump extraction method
- [ ] Integrate with existing ClangJLBridge
- [ ] Multi-method symbol merging

### Phase 3: Wrapper Generation (3-4 days)
- [ ] Design module template system
- [ ] Implement Tier 1 (basic) generator
- [ ] Enhance Tier 2 (advanced) with type inference
- [ ] Add Tier 3 (introspective) with metadata
- [ ] Safety macro generation
- [ ] Library management code

### Phase 4: Tests & Docs (2 days)
- [ ] Test generation for wrapped functions
- [ ] Documentation generation
- [ ] Usage examples in docs
- [ ] API reference generation

### Phase 5: Integration (1-2 days)
- [ ] Integrate with ConfigurationManager
- [ ] Connect to Compiler metadata
- [ ] Enhance ClangJLBridge integration
- [ ] Update user API (build/wrap/rwrap)

### Phase 6: Polish (1-2 days)
- [ ] Error handling and validation
- [ ] Progress indicators
- [ ] Quality metrics reporting
- [ ] Examples and tutorials

**Total: ~12-16 days for world-class wrapper system**

## Success Metrics

1. **Type Coverage**: >80% of common C/C++ types correctly mapped
2. **Automation**: <5 minutes from C++ lib → working Julia package
3. **Quality**: Generated code passes `julia-format`, has docs, tests
4. **Usability**: Non-FFI-experts can use with zero manual tweaking
5. **Robustness**: Handles edge cases (templates, function pointers, etc)

## Future Enhancements

- C++ template specialization handling
- Operator overloading support
- C++ class → Julia struct with methods
- Callback function generation (Julia → C++)
- ABI compatibility checking
- Version management for wrapped libraries
- Automatic dependency resolution
- Package.jl generation for distribution
