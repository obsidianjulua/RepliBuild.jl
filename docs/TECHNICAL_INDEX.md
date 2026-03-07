# RepliBuild.jl - Technical Documentation Index

## 📋 Generated Documents

Two comprehensive technical documents have been created:

### 1. **DEEP_TECHNICAL_ANALYSIS.md** (31 KB, 1028 lines)
**Detailed reference with actual code and IR examples**

Complete technical specification including:
- **Section 1:** MLIR JLCS Dialect Definition
  - Type system: `!jlcs.c_struct<>`, `!jlcs.array_view<>`
  - Operations: `type_info`, `get_field`, `set_field`, `vcall`, array ops
  - CMake build configuration
  
- **Section 2:** MLIR IR Generation from DWARF
  - DWARFParser data structures (ClassInfo, VtableInfo, VirtualMethod)
  - Type mapping (C++ → MLIR)
  - Complete IR generation code
  - Example module output
  
- **Section 3:** Tier Selection Logic
  - Three-tier dispatch system (ccall, JIT, LTO)
  - Actual `is_ccall_safe()` conditions
  - Dispatch decision code
  
- **Section 4:** LLVM IR Generation with LTO
  - LTO IR loading mechanism
  - `Base.llvmcall` wrapper generation
  - LTO eligibility criteria
  - Example wrappers with Ref{T} handling
  - Real LLVM IR example from jit_edges.ll
  
- **Section 5:** JIT Manager Lock-Free Read Path
  - Context structure
  - `_lookup_cached()` implementation
  - Arity-specialized invoke methods (1-4 args)
  - @generated method handling
  - MLIR calling convention
  
- **Section 6:** Example MLIR Module Output
  - Complete generated module
  - Test generation code
  
- **Summary Table:** Component overview

### 2. **TECHNICAL_SUMMARY.txt** (12 KB)
**Quick reference guide for architecture decisions**

Organized by topic:
- Operation definitions
- Data structures
- IR generation flow
- Tier selection conditions
- llvmcall format
- JIT dispatch mechanism
- Example MLIR
- File locations
- Quick reference table

---

## 🎯 Key Findings

### JLCS Dialect Operations (The Real Definitions)

```mlir
!jlcs.c_struct<"Name", [field_types...], [byte_offsets...], packed=bool>
!jlcs.array_view<ElementType, Rank>

jlcs.type_info     - Declare struct & C++ inheritance
jlcs.get_field     - Read field at byte offset
jlcs.set_field     - Write field
jlcs.vcall         - Virtual method dispatch via vtable
jlcs.load_array_element   - Strided array access
jlcs.store_array_element  - Strided array mutation
jlcs.ffe_call      - External C function
```

### Tier Selection Logic (The Exact Conditions)

**Tier 1 (ccall)** if ALL true:
- ✓ No STL containers
- ✓ Return: primitive | pointer | void | small aligned struct
- ✓ Params: NO unions, NO packed structs, NO non-POD classes

**Tier 2 (JIT/LTO)** if is_ccall_safe = false:
- LTO if: NOT virtual, NOT struct return, NO Cstring
- JIT if: config.compile.aot_thunks = false

### DWARF Extraction

**Data extracted per class:**
- `vtable_ptr_offset` - Usually 0
- `members` - Name, type, byte offset
- `virtual_methods` - Mangled name, slot, signature
- `base_classes` - Direct parents
- `size` - Total struct size

### IR Generation

**Module structure:**
```
module {
  llvm.func @symbol(types) -> type     // Declarations
  jlcs.type_info "Class", struct_type, super
  func.func @thunk(...) -> type { ... }
}
```

### JIT Dispatch Mechanism

**Lock-free hot path:**
```julia
_lookup_cached(name) → Dict read (no lock) → return cached ptr
                  → miss → lock → JIT lookup → cache → return
```

**Calling convention:**
```
All args: Ref{T} → inner_ptrs = [ptr_to_arg1, ptr_to_arg2, ...]
Returns: isprimitivetype → direct, else → sret
```

### LTO Embedding

```julia
const LTO_IR = read("project_lto.ll")     # Load at module parse time
Base.llvmcall((LTO_IR, symbol), RetType, Tuple{ArgTypes}, args...)
```

---

## 📁 Source Code References

### Core MLIR Dialect
- `src/mlir/JLCSDialect.td` - Dialect definition
- `src/mlir/JLCSOps.td` - Operation definitions
- `src/mlir/Types.td` - Type system
- `src/mlir/CMakeLists.txt` - Build config

### DWARF & IR Generation
- `src/DWARFParser.jl` - Extract from debug info
- `src/JLCSIRGenerator.jl` - Generate MLIR module
  - `src/ir_gen/TypeUtils.jl` - Type mapping
  - `src/ir_gen/StructGen.jl` - Struct generation
  - `src/ir_gen/FunctionGen.jl` - Function thunks

### Dispatch & Wrapping
- `src/Wrapper.jl` (lines 1566-3835)
  - **1566-1665:** `is_ccall_safe()` - Decision logic
  - **3316-3835:** Tier routing
  - **3918-3932:** llvmcall generation

### Execution
- `src/JITManager.jl`
  - **42-68:** Lock-free cached lookup
  - **85-99:** @generated ABI handling
  - **111-152:** Arity-specialized invoke

### Tests
- `test/test_mlir.jl` - MLIR generation tests
- `test/test_mlir_safety.jl` - Type safety
- `test/jit_edge_test/` - Example LLVM IR

---

## 🔍 How to Use These Documents

### For Architecture Review
→ Read **TECHNICAL_SUMMARY.txt** first (5 min)
→ Then **DEEP_TECHNICAL_ANALYSIS.md** Section 3 & 4

### For Implementation
→ **DEEP_TECHNICAL_ANALYSIS.md** Section 2 (IR generation)
→ **DEEP_TECHNICAL_ANALYSIS.md** Section 5 (JIT dispatch)

### For Debugging Dispatch Issues
→ **DEEP_TECHNICAL_ANALYSIS.md** Section 3 (Tier selection)
→ **Wrapper.jl** lines 1566-1665 (is_ccall_safe conditions)

### For LTO/llvmcall Issues
→ **DEEP_TECHNICAL_ANALYSIS.md** Section 4
→ **Wrapper.jl** lines 1850-1861, 3918-3932

### For JIT Performance
→ **DEEP_TECHNICAL_ANALYSIS.md** Section 5
→ **JITManager.jl** lines 42-68 (_lookup_cached)

---

## 📊 Document Statistics

| Document | Size | Lines | Focus |
|----------|------|-------|-------|
| DEEP_TECHNICAL_ANALYSIS.md | 31 KB | 1028 | Complete specifications, code, examples |
| TECHNICAL_SUMMARY.txt | 12 KB | ~250 | Quick reference, conditions, locations |
| TECHNICAL_INDEX.md | This file | - | Navigation guide |

---

## ✅ What's Covered

### ✓ Tier Selection
- [x] Exact conditions for ccall vs JIT/LTO
- [x] is_ccall_safe() complete implementation
- [x] Decision points in code

### ✓ MLIR Dialect
- [x] All JLCS operations (type_info, vcall, get_field, etc.)
- [x] Type definitions (c_struct, array_view)
- [x] Example IR syntax

### ✓ DWARF Extraction
- [x] ClassInfo, VtableInfo structures
- [x] What gets extracted (members, virtuals, bases)
- [x] Example parsing flow

### ✓ IR Generation
- [x] Type mapping algorithm
- [x] Complete module structure
- [x] Example generated IR

### ✓ LTO/llvmcall
- [x] IR embedding mechanism
- [x] Eligibility criteria
- [x] Ref{T} conversion handling
- [x] Example wrappers
- [x] Real LLVM IR examples

### ✓ JIT Dispatch
- [x] Lock-free read path
- [x] Arity-specialized invoke methods
- [x] Calling convention
- [x] Scalar vs struct returns

---

## 🎓 Reading Path

**5-minute overview:** TECHNICAL_SUMMARY.txt (Sections 1-3)

**30-minute architecture:** 
- TECHNICAL_SUMMARY.txt (all)
- DEEP_TECHNICAL_ANALYSIS.md (Sections 1, 3)

**Deep dive (2 hours):**
- DEEP_TECHNICAL_ANALYSIS.md (all sections)
- Cross-reference with source files

**Implementation reference:**
- Use TECHNICAL_SUMMARY.txt for lookups
- Use DEEP_TECHNICAL_ANALYSIS.md for full context

---

## 📝 Notes

- All code examples are **actual code** from the repository (not summaries)
- Line numbers match the source files exactly
- LLVM IR example is from real test output (jit_edges.ll)
- MLIR examples are actual TableGen + generated IR

---

Generated: 2024-03-06
Source: /home/john/Desktop/Projects/RepliBuild.jl
