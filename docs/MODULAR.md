# Modular Architecture Vision (Future)

**STATUS**: 🚫 NOT READY FOR IMPLEMENTATION

This document outlines the future modular architecture for RepliBuild. **DO NOT IMPLEMENT** until core FFI logic for C/C++ is perfected.

---

## Why Modularize?

RepliBuild is building something unprecedented: a **unified build system + compiler + binding generator** that handles the entire FFI pipeline. As we expand to multiple source languages (C, C++, Rust) and multiple target languages (Julia, Python, Ruby), modularization becomes essential.

### Current Monolithic Structure
```
src/
├── Compiler.jl     (~1500 LOC) - Everything mixed together
└── Wrapper.jl      (~1300 LOC) - Everything mixed together
```

**Problems:**
- C++ compilation logic mixed with generic LLVM pipeline
- Julia-specific binding generation can't be reused for Python
- Type extraction (DWARF/AST/IR) scattered across files
- Adding Rust support requires forking the entire compiler
- No way to reuse type information across projects

### Future Modular Structure
```
src/
├── compiler/
│   ├── CompilerCore.jl              # Language-agnostic LLVM pipeline
│   │   • IR linking, optimization
│   │   • Binary generation (.so/.dll)
│   │   • Works with ANY LLVM IR source
│   │
│   ├── languages/                   # Language-specific frontends
│   │   ├── cpp/
│   │   │   ├── CppCompiler.jl      # C++ → LLVM IR
│   │   │   ├── CppParser.jl        # C++ AST parsing
│   │   │   └── CppTypes.jl         # C++ type system
│   │   ├── c/
│   │   │   └── CCompiler.jl        # Pure C → LLVM IR
│   │   └── rust/
│   │       └── RustCompiler.jl     # Rust → LLVM IR (rustc bridge)
│   │
│   └── metadata/                    # Language-agnostic metadata extraction
│       ├── DWARFExtractor.jl       # Debug info extraction
│       ├── ASTExtractor.jl         # Clang AST traversal
│       ├── IRAnalyzer.jl           # LLVM IR type analysis
│       └── TypeRegistry.jl         # Persistent JSON storage
│
└── bindings/
    ├── common/
    │   ├── BindingCore.jl          # Shared binding interface
    │   └── TypeMapper.jl           # Base type mapping logic
    │
    ├── julia/
    │   ├── JuliaWrapper.jl         # ccall() wrapper generation
    │   ├── JuliaTypes.jl           # C/C++ → Julia type mapping
    │   └── JuliaCodegen.jl         # Julia code emission
    │
    ├── python/
    │   ├── PythonWrapper.jl        # ctypes/cffi generation
    │   ├── PythonTypes.jl          # C/C++ → Python type mapping
    │   └── PythonCodegen.jl        # Python code emission
    │
    └── rust/
        ├── RustWrapper.jl          # Rust FFI generation
        ├── RustTypes.jl            # C/C++ → Rust type mapping
        └── RustCodegen.jl          # Rust code emission
```

---

## Key Components

### 1. CompilerCore (Language-Agnostic)

**Purpose**: Generic LLVM operations that work with IR from ANY language

```julia
# compiler/CompilerCore.jl

"""
Link multiple LLVM IR files and optimize.
Works with IR from C++, C, Rust, or any LLVM frontend.
"""
function link_optimize_ir(ir_files, optimization_level)
    # llvm-link + opt passes
end

"""
Create shared library from LLVM IR.
"""
function create_library(ir_file, output_name)
    # clang++ -shared for .so/.dylib/.dll
end
```

**Why separate?**
- Rust can use the SAME pipeline: `RustCompiler.compile_to_ir()` → `CompilerCore.create_library()`
- C can use it: `CCompiler.compile_to_ir()` → same pipeline
- Zero code duplication

### 2. Language-Specific Frontends

**Purpose**: Convert source code → LLVM IR

```julia
# compiler/languages/cpp/CppCompiler.jl
function compile_cpp_to_ir(source_files, flags)
    # clang++ specific invocation
    # C++ specific flags and semantics
end

# compiler/languages/rust/RustCompiler.jl
function compile_rust_to_ir(crate_path, target)
    # rustc --emit=llvm-ir
    # Rust-specific cargo integration
end
```

**Why separate?**
- Each language has unique compilation needs
- Easy to add new languages without touching core
- Maintainable: C++ expert fixes C++, Rust expert fixes Rust

### 3. TypeRegistry (Persistent Storage)

**Purpose**: Store type information in JSON for reuse across projects

```julia
# compiler/metadata/TypeRegistry.jl

"""
Save type to persistent JSON storage at ~/.julia/replibuild/type_registry/
"""
function save_type(name, metadata, source, confidence)
    # Save to: types/<hash>.json
    # Update index for fast lookup
end

"""
Get type mapping for target language.
"""
function get_type_mapping(cpp_type, target_lang)
    # Query: mappings/cpp_to_<target_lang>.json
end
```

**Storage Structure:**
```
~/.julia/replibuild/type_registry/
├── types/
│   ├── a1b2c3.json  # std::vector<int>
│   └── index.json
└── mappings/
    ├── cpp_to_julia.json
    ├── cpp_to_python.json
    └── cpp_to_rust.json
```

**Why persistent?**
- Type info extracted once, reused across all projects
- Mappings improve over time (community contributions)
- Cross-language consistency

### 4. Binding Generation (Multi-Language)

**Purpose**: Generate FFI bindings for target language

```julia
# bindings/julia/JuliaWrapper.jl
function generate_julia_wrapper(metadata, types)
    # Generate ccall() wrappers
    # Use JuliaTypes.map_type() for conversions
end

# bindings/python/PythonWrapper.jl
function generate_python_wrapper(metadata, types)
    # Generate ctypes/cffi bindings
    # Use PythonTypes.map_type() for conversions
end
```

**Why separate per language?**
- Julia uses `ccall()`, Python uses `ctypes`, Rust uses `extern "C"`
- Type conversions differ: `int` → `Int32` (Julia) vs `c_int` (Python)
- Each language has unique idioms and best practices

---

## Three-Source Type Extraction

**Current**: Only DWARF debug info
**Future**: Merge AST + IR + DWARF with confidence scoring

```julia
# compiler/metadata/TypeExtractor.jl (NEW)

"""
Extract types from THREE sources and merge with confidence:
1. AST (Clang LibTooling) - confidence: 1.0 (source truth)
2. DWARF (debug info) - confidence: 1.0 (binary truth)
3. IR (LLVM types) - confidence: 0.8 (validation/fallback)
"""
function extract_all_types(binary, sources, ir_files)
    ast_types = ASTExtractor.extract(sources)      # From C++ source
    dwarf_types = DWARFExtractor.extract(binary)   # From compiled binary
    ir_types = IRAnalyzer.extract(ir_files)        # From LLVM IR

    # Merge with conflict resolution
    merged = merge_type_sources(ast_types, dwarf_types, ir_types)

    # Save to persistent registry
    for type in merged
        TypeRegistry.save_type(type.name, type.metadata, type.source, type.confidence)
    end
end
```

**Priority Order:**
1. **AST** - Highest accuracy (source code is truth)
2. **DWARF** - High accuracy (binary with debug info)
3. **IR** - Validation/fallback (type-erased but still useful)

---

## Prerequisites Before Modularization

**DO NOT modularize until ALL of these are ✅:**

### Core FFI Accuracy
- [ ] **Enum extraction**: Enums extracted from DWARF with correct values
- [ ] **Enum mapping**: C++ enums map to `@enum` in Julia correctly
- [ ] **Array dimensions**: Multi-dimensional arrays flatten correctly (`int[4][4]` → `NTuple{16, Cint}`)
- [ ] **Function pointers**: Detected and mapped to `Ptr{Cvoid}` or typed
- [ ] **Parameter extraction**: All function parameters in generated bindings
- [ ] **Struct members**: All struct fields with correct types and offsets
- [ ] **Return types**: Accurate for all functions
- [ ] **Const correctness**: `const` qualifiers preserved
- [ ] **Reference handling**: References handled properly

### Test Coverage
- [ ] **Comprehensive test suite** covering ALL edge cases
- [ ] **test_advanced_types.cpp** produces perfect bindings
- [ ] **All 27+ tests passing** with 100% accuracy
- [ ] **Binding verification agent** validates generated code automatically
- [ ] **Real-world library test** (e.g., compile and wrap a small C++ library end-to-end)

### Documentation
- [ ] **All functions documented** with clear docstrings
- [ ] **Architecture diagrams** showing data flow
- [ ] **Type mapping tables** (C++ → Julia for all types)
- [ ] **Examples** showing common use cases

---

## Lessons Learned (November 2025)

### What Went Wrong
We attempted modularization on November 25, 2025 before the core FFI logic was solid. This caused:
- **Increased complexity** without solving fundamental issues
- **Lost focus** on actual problems (enum/array/function pointer accuracy)
- **Debugging nightmare** trying to trace through new module structure
- **Wasted time** that should have been spent fixing type mappings

### What We Should Have Done
1. **Fix the core FFI first** - parameters, enums, arrays, function pointers
2. **Build comprehensive test suite** - catch all edge cases
3. **Achieve 100% accuracy** - every test passing, every type correct
4. **THEN modularize** with clear architectural plan

### Key Insights
- **Premature optimization/refactoring is dangerous**
- **Architecture needs clear blueprints** (comments, docs, diagrams)
- **User's vision matters** - I needed clearer guidance upfront
- **Test coverage prevents regression** - can't refactor safely without it
- **Comments are documentation** - not just for humans, for AI too!

---

## When to Modularize

**Trigger conditions (ALL must be true):**
1. ✅ Core FFI for C/C++ is **bulletproof** (100% test pass rate)
2. ✅ User explicitly says "now we modularize"
3. ✅ Clear architectural blueprint exists (this doc + user input)
4. ✅ Comprehensive test suite ensures no regressions
5. ✅ Second language (Python/Rust) bindings are needed

**Don't modularize because:**
- ❌ "The code is getting big" - size alone isn't a reason
- ❌ "It would be cleaner" - cleanliness comes after correctness
- ❌ "We might need it later" - YAGNI (You Aren't Gonna Need It)

---

## Implementation Plan (Future)

When the time comes, follow this order:

### Phase 1: Extract CompilerCore
1. Create `src/compiler/CompilerCore.jl`
2. Move LLVM IR operations (link, optimize, binary generation)
3. Make it work with existing C++ compilation
4. All tests still pass

### Phase 2: Extract C++ Frontend
1. Create `src/compiler/languages/cpp/CppCompiler.jl`
2. Move C++ → IR compilation logic
3. Hook into CompilerCore
4. All tests still pass

### Phase 3: Extract Metadata System
1. Create `src/compiler/metadata/` directory
2. Extract DWARF/AST/IR parsers
3. Implement TypeRegistry with JSON storage
4. All tests still pass

### Phase 4: Extract Julia Bindings
1. Create `src/bindings/julia/` directory
2. Move wrapper generation logic
3. All tests still pass

### Phase 5: Add Second Language
1. Pick target: Python or Rust
2. Implement `src/bindings/<lang>/` following same pattern
3. Prove modularity works!

**Critical**: After EACH phase, **all tests must pass**. No exceptions.

---

## References

- **Current State**: Monolithic `src/Compiler.jl` + `src/Wrapper.jl`
- **User Vision**: Multi-language compiler + binding generator
- **Inspiration**: SWIG, pybind11, bindgen (but unified and better)
- **Backup**: User has repo zip from November 25, 2025 before modularization

---

**Last Updated**: November 25, 2025
**Status**: Planning document only - DO NOT IMPLEMENT
**Next Review**: After C/C++ FFI is perfected
