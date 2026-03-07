# RepliBuild.jl - Deep Technical Architecture Document

## 1. MLIR JLCS Dialect Definition

### 1.1 Dialect Overview
**Location:** `src/mlir/JLCSDialect.td`, `src/mlir/JLCSOps.td`, `src/mlir/Types.td`

The JLCS (Julia C-Struct) dialect models C-ABI-compatible structs and FFE (Foreign Function Execution) for interop.

### 1.2 Type System - CStruct Type Definition

**File:** `src/mlir/Types.td` (lines 16-38)

```mlir
def CStructType : JLCS_Type<"CStruct", "c_struct"> {
  let summary = "A C-ABI-compatible struct with explicit field types and offsets.";
  
  let parameters = (ins
    "StringAttr":$juliaTypeName,           // e.g., "MyModule.Outer"
    ArrayRefParameter<"Type", "field types">:$fieldTypes,  // [T1, T2, ...]
    "ArrayAttr":$fieldOffsets,             // Explicit byte offsets: [O1:i64, O2:i64]
    "bool":$isPacked                       // Packing flag
  );
  
  let assemblyFormat = "`<` $juliaTypeName `,` `[` $fieldTypes `]` `,` `[` $fieldOffsets `]` `,` `packed` `=` $isPacked `>`";
}
```

**Example MLIR syntax:**
```mlir
!jlcs.c_struct<"MyStruct", [i32, i64, f64], [0 : i64, 4 : i64, 12 : i64], packed = false>
```

### 1.3 Array View Type

**File:** `src/mlir/Types.td` (lines 44-77)

Universal strided array descriptor for cross-language arrays (Julia, NumPy, C++, Rust).

```mlir
def ArrayViewType : JLCS_Type<"ArrayView", "array_view"> {
  let parameters = (ins
    "Type":$elementType,      // Element type (f64, i32, !jlcs.c_struct<...>, etc.)
    "unsigned":$rank          // Number of dimensions (compile-time constant)
  );
  
  let assemblyFormat = "`<` $elementType `,` $rank `>`";
}
```

**Runtime memory layout (C struct):**
```c
struct ArrayView {
  T* data_ptr;          // offset 0:  pointer to element data
  int64_t* dims_ptr;    // offset 8:  pointer to dimension sizes
  int64_t* strides_ptr; // offset 16: pointer to stride values (in elements)
  int64_t rank;         // offset 24: number of dimensions
};
```

**Example MLIR syntax:**
```mlir
!jlcs.array_view<f64, 3>    // 3D array of float64
```

### 1.4 Operation Definitions

**File:** `src/mlir/JLCSOps.td`

#### 1.4.1 Type Info Op - `jlcs.type_info`

**Lines 17-40**

Declares a CStruct type and its C++ base class mapping (placed in module's top region).

```mlir
def TypeInfoOp : JLCS_Op<"type_info", [Pure, IsolatedFromAbove]> {
  let arguments = (ins
    StrAttr   :$typeName,           // Julia type name
    TypeAttr  :$structType,         // Must be CStructType
    DefaultValuedStrAttr<StrAttr, "\"\"">:$superType  // Supertype for inheritance
  );
  
  let assemblyFormat = "$typeName `,` $structType `,` $superType attr-dict";
}
```

**Example IR:**
```mlir
jlcs.type_info "Base", !jlcs.c_struct<"Base", [i32, i32], [0 : i64, 4 : i64], packed = false>, ""
```

#### 1.4.2 Field Access - `jlcs.get_field`

**Lines 43-61**

Generic operation to read a field from a C-compatible struct using byte offset.

```mlir
def GetFieldOp : JLCS_Op<"get_field"> {
  let arguments = (ins
    AnyType:$structValue,      // Struct reference
    I64Attr:$fieldOffset       // Byte offset
  );
  let results = (outs AnyType:$result);
}
```

**Example IR:**
```mlir
%value = jlcs.get_field %struct_ref { fieldOffset = 4 : i64 } : (!llvm.ptr) -> i32
```

#### 1.4.3 Field Mutation - `jlcs.set_field`

**Lines 64-84**

Generic operation to write a field into a C-compatible struct.

```mlir
def SetFieldOp : JLCS_Op<"set_field", [Pure]> {
  let arguments = (ins
    AnyType:$structValue,      // Struct reference
    AnyType:$newValue,         // New value
    I64Attr:$fieldOffset       // Byte offset
  );
  let results = (outs);
}
```

**Example IR:**
```mlir
jlcs.set_field %struct_ref, %new_value { fieldOffset = 4 : i64 } : (!llvm.ptr, i32) -> ()
```

#### 1.4.4 Virtual Method Call - `jlcs.vcall`

**Lines 91-129**

Call C++ virtual method via vtable dispatch.

```mlir
def VirtualCallOp : JLCS_Op<"vcall"> {
  let arguments = (ins
    SymbolRefAttr:$class_name,    // Class name (e.g., @Base)
    Variadic<AnyType>:$args,      // Arguments (first is always object pointer)
    I64Attr:$vtable_offset,       // Offset of vptr in object
    I64Attr:$slot                 // Vtable slot index
  );
  
  let results = (outs Optional<AnyType>:$result);
}
```

**Operation semantics:**
1. Read vtable pointer from object at `vtable_offset`
2. Load function pointer from `vtable[slot]`
3. Call the function with object pointer + arguments

**Example IR:**
```mlir
%result = jlcs.vcall @Base::foo(%obj) {vtable_offset = 0 : i64, slot = 0 : i64} : (!llvm.ptr) -> i32
```

#### 1.4.5 Array Operations

**Lines 136-204**

Multi-dimensional strided array access with zero-copy semantics.

```mlir
def LoadArrayElementOp : JLCS_Op<"load_array_element"> {
  let arguments = (ins
    AnyType:$view,               // Pointer to ArrayView struct
    Variadic<Index>:$indices     // Multi-dimensional indices
  );
  let results = (outs AnyType:$result);
}

def StoreArrayElementOp : JLCS_Op<"store_array_element"> {
  let arguments = (ins
    AnyType:$value,              // Value to store
    AnyType:$view,               // Pointer to ArrayView struct
    Variadic<Index>:$indices     // Multi-dimensional indices
  );
  let results = (outs);
}
```

**Index computation:** `linear_offset = sum(index_i * stride_i)`

**Example IR:**
```mlir
%elem = jlcs.load_array_element %view[%i, %j, %k] : !jlcs.array_view<f64, 3> -> f64
jlcs.store_array_element %value, %view[%i, %j] : f64, !jlcs.array_view<f64, 2>
```

#### 1.4.6 Foreign Function Execution - `jlcs.ffe_call`

**Lines 207-212**

Call external C function using FFE metadata.

```mlir
def FFECallOp : JLCS_Op<"ffe_call", [MemoryEffects<[]>]> {
  let summary = "Call an external C function using FFE metadata";
  let arguments = (ins Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$results);
}
```

### 1.5 Build Configuration

**File:** `src/mlir/CMakeLists.txt`

Key components:
- **TableGen processing:** Generates `.inc` files from `.td` definitions
- **Linking:** Explicit whole-archive linking for JIT registration
- **Dependencies:** 
  - `MLIRExecutionEngine` - JIT compilation
  - `MLIRTargetLLVMIRExport` - MLIR→LLVM IR translation
  - `MLIRLLVMToLLVMIRTranslation` - LLVM dialect lowering

---

## 2. MLIR IR Generation from DWARF

### 2.1 DWARFParser Data Structures

**File:** `src/DWARFParser.jl` (lines 14-50)

```julia
struct VirtualMethod
    name::String              # e.g., "foo"
    mangled_name::String      # e.g., "_ZN4Base3fooEv"
    slot::Int                 # Vtable slot index
    return_type::String       # C++ type name
    parameters::Vector{String} # Parameter types
end

struct MemberInfo
    name::String
    type_name::String
    offset::Int  # Byte offset in struct
end

struct ClassInfo
    name::String                      # Full class name
    vtable_ptr_offset::Int           # Usually 0
    base_classes::Vector{String}     # Immediate bases
    virtual_methods::Vector{VirtualMethod}
    members::Vector{MemberInfo}      # Data members
    size::Int                        # Total size in bytes
end

struct VtableInfo
    classes::Dict{String, ClassInfo}           # class_name => ClassInfo
    vtable_addresses::Dict{String, UInt64}    # class_name => vtable address
    method_addresses::Dict{String, UInt64}    # mangled_name => function address
end
```

**DWARF Parsing Flow:**
1. Parse `llvm-dwarfdump` output for `DW_TAG_class_type`, `DW_TAG_structure_type`
2. Extract `DW_AT_name`, `DW_AT_byte_size`, `DW_AT_data_member_location`
3. Identify virtual methods via `DW_TAG_subprogram` with virtual flag
4. Build inheritance chain via `DW_TAG_inheritance`

### 2.2 MLIR IR Generation

**File:** `src/JLCSIRGenerator.jl` (lines 23-249)

#### 2.2.1 Type Mapping

```julia
function map_cpp_type_to_mlir(cpp_type::String)
    # Examples:
    # "double" → "f64"
    # "int*" → "!llvm.ptr"
    # "unsigned int" → "i32"
    # "void" → "none"
    # Fallback for unknown: "!llvm.ptr"
end
```

#### 2.2.2 Type Info IR Generation

**Lines 63-123**

```julia
function generate_type_info_ir(class_name::String, info::ClassInfo, vtable_addr::UInt64)
    # Sanitize name: "::" → "_", "<>" → "_", etc.
    mlir_name = replace(class_name, "::" => "_", "<" => "_", ">" => "_", ...)
    
    # Build field lists from members (sorted by offset)
    field_types = String[]
    field_offsets = Int[]
    for m in sort(info.members, by = m -> m.offset)
        push!(field_types, map_cpp_type_to_mlir(m.type_name))
        push!(field_offsets, m.offset)
    end
    
    # Format: [T1, T2, ...], [O1 : i64, O2 : i64, ...]
    field_types_str = join(field_types, ", ")
    field_offsets_attr = "[" * join(["$(o) : i64" for o in field_offsets], ", ") * "]"
    
    # Build CStruct type
    struct_type_str = "!jlcs.c_struct<\"$(class_name)\", [$(field_types_str)], [$(field_offsets_attr)], packed = false>"
    
    # Generate type_info operation
    return """
    jlcs.type_info "$(mlir_name)", $(struct_type_str), "$(super_type)" """
end
```

**Example output for class with two int members:**
```mlir
jlcs.type_info "Base", !jlcs.c_struct<"Base", [i32, i32], [0 : i64, 4 : i64], packed = false>, ""
```

#### 2.2.3 Virtual Method IR Generation

**Lines 130-159**

```julia
function generate_virtual_method_ir(method::VirtualMethod, addr::UInt64)
    call_target = method.mangled_name
    mlir_name = "thunk_$(call_target)"
    
    (ret_type, arg_types_str) = get_llvm_signature(method)
    
    arg_names = ["%arg$i" for i in 0:length(method.parameters)]
    args_sig = "(" * join(["$(arg_names[i]): $(t)" for ...], ", ") * ")"
    
    # Generate function that calls the method
    call_stmt = ret_type == "" ? 
        "llvm.call @$(call_target)$(args_vals) : $(call_sig) -> ()" : 
        "%result = llvm.call @$(call_target)$(args_vals) : $(call_sig) -> $(ret_type)"
    
    return_stmt = ret_type == "" ? "return" : "return %result : $(ret_type)"
    
    return """
    func.func @$(mlir_name)$(args_sig) -> $(ret_type == "" ? "()" : ret_type) {
        $(call_stmt)
        $(return_stmt)
    }"""
end
```

**Example output for method returning i32:**
```mlir
func.func @thunk__ZN4Base3fooEv(%arg0: !llvm.ptr) -> i32 {
  %result = llvm.call @_ZN4Base3fooEv(%arg0) : (!llvm.ptr) -> i32
  return %result : i32
}
```

#### 2.2.4 Complete Module Generation

**Lines 166-247**

```julia
function generate_jlcs_ir(vtinfo::VtableInfo, metadata::Any=Dict())
    io = IOBuffer()
    
    println(io, "module {")
    
    # 1. External Dispatch Declarations
    for (class_name, class_info) in vtinfo.classes
        for method in class_info.virtual_methods
            method_addr = get(vtinfo.method_addresses, method.mangled_name, UInt64(0))
            if method_addr != 0
                (ret_type, arg_types) = get_llvm_signature(method)
                decl_ret = ret_type == "" ? "!llvm.void" : ret_type
                println(io, "  llvm.func @$(method.mangled_name)($(arg_types)) -> $(decl_ret)")
            end
        end
    end
    
    # 2. Type Info & VMethods
    for (class_name, class_info) in vtinfo.classes
        if !isempty(class_info.members)
            println(io, generate_type_info_ir(class_name, class_info, vtable_addr))
            for method in class_info.virtual_methods
                println(io, generate_virtual_method_ir(method, method_addr))
            end
        end
    end
    
    # 3. Function Thunks
    if haskey(metadata, "functions")
        println(io, generate_function_thunks(metadata["functions"], structs_meta))
    end
    
    # 4. STL Container Thunks
    if haskey(metadata, "stl_methods")
        println(io, generate_stl_thunks(metadata["stl_methods"], metadata))
    end
    
    println(io, "}")
    return String(take!(io))
end
```

**Complete example module:**
```mlir
module {
  // External Dispatch Declarations
  llvm.func @_ZN4Base3fooEv(!llvm.ptr) -> i32
  
  // Type Info
  jlcs.type_info "Base", !jlcs.c_struct<"Base", [i32], [0 : i64], packed = false>, ""
  
  // Virtual Method Wrapper
  func.func @thunk__ZN4Base3fooEv(%arg0: !llvm.ptr) -> i32 {
    %result = llvm.call @_ZN4Base3fooEv(%arg0) : (!llvm.ptr) -> i32
    return %result : i32
  }
  
  // Function Thunks
  func.func @thunk__ZN4Base3barEv(%arg0: !llvm.ptr, %arg1: i32) -> i32 {
    %result = llvm.call @_ZN4Base3barEv(%arg0, %arg1) : (!llvm.ptr, i32) -> i32
    return %result : i32
  }
}
```

---

## 3. Tier Selection Logic - Dispatch Decision

### 3.1 Overview

**File:** `src/Wrapper.jl` (lines 3310-3835)

RepliBuild uses a **three-tier dispatch system**:

| Tier | Method | Condition | Speed | ABI Safety |
|------|--------|-----------|-------|-----------|
| **1** | `ccall` | Safe primitives/pointers | Fast | High |
| **2** | `JITManager.invoke()` | Complex ABI (JIT) | Medium | High |
| **2** | `Base.llvmcall()` with LTO IR | Complex ABI (AOT) | Fast | High |

### 3.2 Dispatch Decision Function

**Lines 1566-1665**

```julia
function is_ccall_safe(func_info, dwarf_structs)::Bool
    # Check 1: STL Container Types
    ret_type_str = String(get(func_info["return_type"], "c_type", ""))
    if is_stl_container_type(ret_type_str)
        return false
    end
    for param in func_info["parameters"]
        if is_stl_container_type(get(param, "c_type", ""))
            return false
        end
    end
    
    # Check 2: Return Type Safety
    ret_type = ret_type_str
    is_template_ret = occursin('<', ret_type)
    is_primitive_ret = !is_template_ret && 
        (contains(ret_type, "int") || contains(ret_type, "float") || ...)
    
    if !contains(ret_type, "*") && !contains(ret_type, "void") && !is_primitive_ret
        # Template returns → route to MLIR (unpredictable ABI)
        if is_template_ret
            return false
        end
        
        # Struct return by value > 16 bytes → unsafe
        if haskey(dwarf_structs, ret_type)
            s_info = dwarf_structs[ret_type]
            if parse(Int, get(s_info, "byte_size", "0")) > 16
                return false  # Too large for ccall sret ABI
            end
            
            # Check if it's a non-POD class
            if get(s_info, "kind", "struct") == "class"
                return false
            end
            
            # Check if PACKED (DWARF size ≠ Julia aligned size)
            dwarf_size = parse(Int, get(s_info, "byte_size", "0"))
            julia_size = get_julia_aligned_size(members)
            if dwarf_size > 0 && dwarf_size != julia_size
                return false  # Packed struct → sret ABI mismatch
            end
        end
    end
    
    # Check 3: Parameter Type Safety
    for param in func_info["parameters"]
        c_type = get(param, "c_type", "")
        
        # Pointers are always safe
        if contains(c_type, "*")
            continue
        end
        
        base_type = String(strip(replace(c_type, "const" => "")))
        
        if haskey(dwarf_structs, base_type)
            s_info = dwarf_structs[base_type]
            
            # Unions → unsafe
            if get(s_info, "kind", "struct") == "union"
                return false
            end
            
            # Packed structs → unsafe
            dwarf_size = parse(Int, get(s_info, "byte_size", "0"))
            julia_size = get_julia_aligned_size(get(s_info, "members", []))
            if dwarf_size > 0 && dwarf_size != julia_size
                return false  # Packed → ABI mismatch
            end
        end
    end
    
    return true  # Safe for ccall
end
```

**Return value semantics:**
- `true` → Use `ccall` (Tier 1)
- `false` → Use JIT/LTO dispatch (Tier 2)

### 3.3 Dispatch Decision Point

**Lines 3316-3322**

```julia
# Determine if we should use MLIR or ccall
use_mlir_dispatch = !is_ccall_safe(func, dwarf_structs)

if func_name == "pack_record"
    println("DEBUG: pack_record use_mlir_dispatch = $use_mlir_dispatch")
    println("DEBUG: is_ccall_safe returned ", is_ccall_safe(func, dwarf_structs))
end
```

### 3.4 Tier 2: JIT Dispatch (Runtime)

**Lines 3750-3835**

```julia
if use_mlir_dispatch
    # Set up argument references
    if config.compile.aot_thunks
        # Tier 2a: AOT-compiled thunks via Base.llvmcall
        ptr_setup = """
        refs = ($(join(["Ref($a)" for a in param_names], ", ")),)
        inner_ptrs = Ptr{Cvoid}[...]
        GC.@preserve refs inner_ptrs begin
        """
        
        if is_void_ret
            invoke_call = """
            if !isempty(THUNKS_LTO_IR)
                Base.llvmcall((THUNKS_LTO_IR, "_mlir_ciface_$(mangled)_thunk"), Cvoid, Tuple{Ptr{Ptr{Cvoid}}}, inner_ptrs)
            else
                ccall((:_mlir_ciface_$(mangled)_thunk, THUNKS_LIBRARY_PATH), Cvoid, (Ptr{Ptr{Cvoid}},), inner_ptrs)
            end
            return nothing
            """
        else
            invoke_call = """
            if !isempty(THUNKS_LTO_IR)
                ret = Base.llvmcall((THUNKS_LTO_IR, "_mlir_ciface_$(mangled)_thunk"), $jit_ret_type, Tuple{Ptr{Ptr{Cvoid}}}, inner_ptrs)
            else
                ret = ccall((:_mlir_ciface_$(mangled)_thunk, THUNKS_LIBRARY_PATH), $jit_ret_type, (Ptr{Ptr{Cvoid}},), inner_ptrs)
            end
            return ret
            """
        end
    else
        # Tier 2b: JIT dispatch via JITManager.invoke
        if is_void_ret
            invoke_call = if isempty(invoke_args)
                "RepliBuild.JITManager.invoke(\"_mlir_ciface_$(mangled)_thunk\")"
            else
                "RepliBuild.JITManager.invoke(\"_mlir_ciface_$(mangled)_thunk\", $invoke_args)"
            end
        else
            invoke_call = if isempty(invoke_args)
                "RepliBuild.JITManager.invoke(\"_mlir_ciface_$(mangled)_thunk\", $jit_ret_type)"
            else
                "RepliBuild.JITManager.invoke(\"_mlir_ciface_$(mangled)_thunk\", $jit_ret_type, $invoke_args)"
            end
        end
    end
end
```

---

## 4. LLVM IR Generation with LTO

### 4.1 LTO IR Loading

**File:** `src/Wrapper.jl` (lines 1850-1861)

```julia
const LTO_IR_PATH = joinpath(@__DIR__, "$(lto_name)_lto.ll")
const LTO_IR = isfile(LTO_IR_PATH) ? read(LTO_IR_PATH, String) : ""
const THUNKS_LTO_IR_PATH = joinpath(@__DIR__, "$(lto_name)_thunks_lto.ll")
const THUNKS_LTO_IR = isfile(THUNKS_LTO_IR_PATH) ? read(THUNKS_LTO_IR_PATH, String) : ""
```

LTO IR is embedded as **module-level constants** — the LLVM IR text is read from `.ll` files at module load time.

### 4.2 Base.llvmcall Wrapper Generation

**Lines 3918-3932**

```julia
function _build_llvmcall_expr(ret_type_str, indent="        ")
    # Core llvmcall call
    call_expr = "Base.llvmcall((LTO_IR, \"$mangled\"), $ret_type_str, Tuple{$llvmcall_types}, $llvmcall_args)"
    
    if !has_ref_params
        return "$(indent)return $call_expr"
    end
    
    # Handle Ref{T} → Ptr{T} conversion
    lines = String[]
    for l in llvmcall_conversion_lines  # e.g., "__ptr_x = Base.unsafe_convert(Ptr{T}, x)"
        push!(lines, "$indent$l")
    end
    
    preserve_list = join(llvmcall_ref_args, " ")
    push!(lines, "$(indent)GC.@preserve $preserve_list begin")
    push!(lines, "$(indent)    return $call_expr")
    push!(lines, "$(indent)end")
    
    return join(lines, "\n")
end
```

### 4.3 LTO Eligibility Criteria

**Lines 3934-3941**

```julia
lto_eligible = config.link.enable_lto &&
    !is_virtual &&
    !returns_known_struct &&
    julia_return_type != "Cstring" &&
    !any(t -> t == "Cstring", param_types)
```

**Conditions for LTO use:**
1. `config.link.enable_lto` is true
2. NOT a virtual method
3. Does NOT return a struct by value
4. No `Cstring` parameters or returns (llvmcall won't auto-convert like ccall)

### 4.4 Generated llvmcall Wrapper Example

For function `int add(int a, int b)`:

```julia
function add(a::Cint, b::Cint)::Cint
    if !isempty(LTO_IR)
        return Base.llvmcall((LTO_IR, "_Z3addii"), Cint, Tuple{Cint, Cint}, a, b)
    else
        return ccall((:_Z3addii, LIBRARY_PATH), Cint, (Cint, Cint), a, b)
    end
end
```

For function with Ref{T} parameter: `void fill(Matrix<double>& m)`:

```julia
function fill(m::Ref{Matrix_double})
    __ptr_m = Base.unsafe_convert(Ptr{Matrix_double}, m)
    GC.@preserve m begin
        return Base.llvmcall((LTO_IR, "_Z4fillRM"), Cvoid, Tuple{Ptr{Matrix_double}}, __ptr_m)
    end
end
```

### 4.5 Example LLVM IR

**From:** `test/jit_edge_test/build/jit_edges.ll` (first 50 lines)

```llvm
; ModuleID = '/home/john/Desktop/Projects/RepliBuild.jl/test/jit_edge_test/src/jit_edges.cpp'
source_filename = "/home/john/Desktop/Projects/RepliBuild.jl/test/jit_edge_test/src/jit_edges.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.PairResult = type { i32, i32 }
%struct.PackedTriplet = type <{ i8, i32, i8 }>

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define i32 @scalar_add(i32 noundef %0, i32 noundef %1) #0 !dbg !9 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  store i32 %1, ptr %4, align 4
  %5 = load i32, ptr %3, align 4, !dbg !19
  %6 = load i32, ptr %4, align 4, !dbg !20
  %7 = add nsw i32 %5, %6, !dbg !21
  ret i32 %7, !dbg !22
}

define double @scalar_mul(double noundef %0, double noundef %1) #0 !dbg !23 {
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double %0, ptr %3, align 8
  store double %1, ptr %4, align 8
  %5 = load double, ptr %3, align 8, !dbg !31
  %6 = load double, ptr %4, align 8, !dbg !32
  %7 = fmul double %5, %6, !dbg !33
  ret double %7, !dbg !34
}

define i32 @identity(i32 noundef %0) #0 !dbg !35 {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  %3 = load i32, ptr %2, align 4, !dbg !40
  ret i32 %3, !dbg !41
}

define void @write_sum(ptr noundef %0, ptr noundef %1, ptr noundef %2) #0 !dbg !42 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  ...
}
```

---

## 5. JIT Manager - Lock-Free Read Path

### 5.1 Context Structure

**File:** `src/JITManager.jl` (lines 16-29)

```julia
mutable struct JITContext
    mlir_ctx::Ptr{Cvoid}
    jit_engine::Union{Ptr{Cvoid}, Nothing}
    compiled_symbols::Dict{String, Ptr{Cvoid}}      # Cache
    vtable_info::Union{VtableInfo, Nothing}
    initialized::Bool
    lock::ReentrantLock
end

const GLOBAL_JIT = JITContext()
```

### 5.2 Lock-Free Cached Lookup

**Lines 42-68**

```julia
@inline function _lookup_cached(func_name::String)::Ptr{Cvoid}
    # FAST PATH: Lock-free Dict read
    # (Julia Dict reads are safe under single-writer pattern)
    ptr = get(GLOBAL_JIT.compiled_symbols, func_name, C_NULL)
    if ptr != C_NULL
        return ptr
    end

    # SLOW PATH: Acquire lock, double-check, JIT lookup, cache
    lock(GLOBAL_JIT.lock) do
        ptr = get(GLOBAL_JIT.compiled_symbols, func_name, C_NULL)
        if ptr != C_NULL
            return ptr
        end

        ptr = MLIRNative.lookup(GLOBAL_JIT.jit_engine, func_name)
        if ptr == C_NULL
            ptr = MLIRNative.lookup(GLOBAL_JIT.jit_engine, "_" * func_name)
        end
        if ptr == C_NULL
            throw(ErrorException("JIT Error: Symbol not found: $func_name"))
        end

        GLOBAL_JIT.compiled_symbols[func_name] = ptr
        return ptr
    end
end
```

**Performance model:**
- **Hit** (cached): O(1) read, no lock
- **Miss** (first call): Acquire lock, JIT lookup, cache → subsequent calls hit cache

### 5.3 Arity-Specialized Invoke Methods

**Lines 85-170**

Generated invoke methods for arity 0-4 with zero heap allocation.

#### 5.3.1 Generic Invoke Call

**Lines 85-99**

Uses `@generated` to resolve ccall return type at compile time:

```julia
@generated function _invoke_call(fptr::Ptr{Cvoid}, ::Type{T}, inner_ptrs::Vector{Ptr{Cvoid}}) where T
    if isprimitivetype(T)
        # Scalar return: T ciface(void** args_ptr) → direct return
        return :(ccall(fptr, $T, (Ptr{Ptr{Cvoid}},), inner_ptrs))
    else
        # Struct return: void ciface(T* sret, void** args_ptr) → sret convention
        return quote
            ret_buf = Ref{$T}()
            GC.@preserve ret_buf begin
                ccall(fptr, Cvoid, (Ptr{$T}, Ptr{Ptr{Cvoid}}), ret_buf, inner_ptrs)
            end
            ret_buf[]
        end
    end
end
```

#### 5.3.2 1-Argument Specialization

**Lines 111-119**

```julia
function invoke(func_name::String, ::Type{T}, a1) where T
    GLOBAL_JIT.initialized || error("JIT not initialized.")
    fptr = _lookup_cached(func_name)
    r1 = Ref(a1)
    inner_ptrs = Ptr{Cvoid}[Base.unsafe_convert(Ptr{Cvoid}, r1)]
    GC.@preserve r1 begin
        return _invoke_call(fptr, T, inner_ptrs)
    end
end
```

#### 5.3.3 4-Argument Specialization

**Lines 144-152**

```julia
function invoke(func_name::String, ::Type{T}, a1, a2, a3, a4) where T
    GLOBAL_JIT.initialized || error("JIT not initialized.")
    fptr = _lookup_cached(func_name)
    r1 = Ref(a1); r2 = Ref(a2); r3 = Ref(a3); r4 = Ref(a4)
    inner_ptrs = Ptr{Cvoid}[
        Base.unsafe_convert(Ptr{Cvoid}, r1),
        Base.unsafe_convert(Ptr{Cvoid}, r2),
        Base.unsafe_convert(Ptr{Cvoid}, r3),
        Base.unsafe_convert(Ptr{Cvoid}, r4)
    ]
    GC.@preserve r1 r2 r3 r4 begin
        return _invoke_call(fptr, T, inner_ptrs)
    end
end
```

#### 5.3.4 Variadic Fallback

**Lines 155-170**

```julia
function invoke(func_name::String, ::Type{T}, args::Vararg{Any, N}) where {T, N}
    GLOBAL_JIT.initialized || error("JIT not initialized.")
    fptr = _lookup_cached(func_name)

    ref_args = Vector{Any}(undef, N)
    inner_ptrs = Vector{Ptr{Cvoid}}(undef, N)
    for (i, arg) in enumerate(args)
        r = Ref(arg)
        ref_args[i] = r
        inner_ptrs[i] = Base.unsafe_convert(Ptr{Cvoid}, r)
    end

    GC.@preserve ref_args begin
        return _invoke_call(fptr, T, inner_ptrs)
    end
end
```

### 5.4 Void-Return Invoke

**Lines 176-192**

```julia
function invoke(func_name::String, args::Vararg{Any, N}) where N
    GLOBAL_JIT.initialized || error("JIT not initialized.")
    fptr = _lookup_cached(func_name)

    ref_args = Vector{Any}(undef, N)
    inner_ptrs = Vector{Ptr{Cvoid}}(undef, N)
    for (i, arg) in enumerate(args)
        r = Ref(arg)
        ref_args[i] = r
        inner_ptrs[i] = Base.unsafe_convert(Ptr{Cvoid}, r)
    end

    GC.@preserve ref_args inner_ptrs begin
        ccall(fptr, Cvoid, (Ptr{Ptr{Cvoid}},), inner_ptrs)
    end
    return nothing
end
```

### 5.5 MLIR Calling Convention

The JIT uses a **unified calling convention** for all functions:

```
Scalar return:    T    ciface(void** args_ptr)
Struct return:    void ciface(T* sret, void** args_ptr)
Void return:      void ciface(void** args_ptr)
```

All arguments are passed as **pointers to the values** (via `Ref{T}` conversion):
```
inner_ptrs = [ptr_to_arg1, ptr_to_arg2, ..., ptr_to_argN]
```

---

## 6. Example MLIR IR Output

### 6.1 Complete Generated Module

**From:** `test/test_mlir.jl` + actual generation

```mlir
module {
  // External Dispatch Declarations
  llvm.func @_ZN4Base3fooEv(!llvm.ptr) -> i32
  llvm.func @_ZN4Base3barEv(!llvm.ptr, i32) -> i32

  // Type Info - Base class with vtable and two int members
  jlcs.type_info "Base", !jlcs.c_struct<"Base", [!llvm.ptr, i32, i32], [0 : i64, 8 : i64, 12 : i64], packed = false>, ""

  // Virtual Method Wrapper for foo()
  func.func @thunk__ZN4Base3fooEv(%arg0: !llvm.ptr) -> i32 {
    %result = llvm.call @_ZN4Base3fooEv(%arg0) : (!llvm.ptr) -> i32
    return %result : i32
  }

  // Virtual Method Wrapper for bar(int)
  func.func @thunk__ZN4Base3barEv(%arg0: !llvm.ptr, %arg1: i32) -> i32 {
    %result = llvm.call @_ZN4Base3barEv(%arg0, %arg1) : (!llvm.ptr, i32) -> i32
    return %result : i32
  }

  // Regular Function Thunk
  func.func @thunk__ZN4add_onesEv(%arg0: i32) -> i32 {
    %result = llvm.call @_Z8add_onesi(%arg0) : (i32) -> i32
    return %result : i32
  }

  // Struct Return Thunk
  func.func @thunk__ZN6getpairEv() -> !llvm.struct<"PairResult", (i32, i32)> {
    %result = llvm.call @_Z7getpairv() : () -> !llvm.struct<"PairResult", (i32, i32)>
    return %result : !llvm.struct<"PairResult", (i32, i32)>
  }
}
```

### 6.2 Test Generation Code

**File:** `test/test_mlir.jl` (lines 1-80)

```julia
using Test
using RepliBuild.JLCSIRGenerator
using RepliBuild.DWARFParser

@testset "MLIR Integration Tests" begin
    @testset "JLCSIRGenerator" begin
        # Mock virtual method
        vm = VirtualMethod("foo", "_ZN4Base3fooEv", 0, "int", [])
        ci = ClassInfo("Base", 0, String[], [vm], MemberInfo[], 8)
        
        # Generate type info
        ir_type = JLCSIRGenerator.generate_type_info_ir("Base", ci, UInt64(0x1000))
        @test contains(ir_type, "jlcs.type_info \"Base\"")
        @test contains(ir_type, "!jlcs.c_struct<\"Base\"")
        
        # Generate virtual method
        ir_method = JLCSIRGenerator.generate_virtual_method_ir(vm, UInt64(0x2000))
        @test contains(ir_method, "func.func @thunk__ZN4Base3fooEv(%arg0: !llvm.ptr) -> i32")
        @test contains(ir_method, "llvm.call")
    end
    
    @testset "Integration: Generate & Parse" begin
        # Create full vtable info
        vm = VirtualMethod("bar", "_ZN4Base3barEv", 0, "void", ["int"])
        ci = ClassInfo("Base", 0, String[], [vm], MemberInfo[], 8)
        
        classes = Dict("Base" => ci)
        vtable_addrs = Dict("Base" => UInt64(0x1000))
        method_addrs = Dict("_ZN4Base3barEv" => UInt64(0x2000))
        
        vtinfo = VtableInfo(classes, vtable_addrs, method_addrs)
        generated_ir = generate_jlcs_ir(vtinfo)
        
        # Verify structure
        @test contains(generated_ir, "module {")
        @test contains(generated_ir, "jlcs.type_info")
        @test contains(generated_ir, "func.func")
        @test contains(generated_ir, "}")
    end
end
```

---

## Summary Table

| Component | File | Key Types | Semantics |
|-----------|------|-----------|-----------|
| **JLCS Dialect** | `src/mlir/*.td` | `!jlcs.c_struct<>`, `!jlcs.array_view<>` | MLIR type system for C ABIs |
| **Operations** | `src/mlir/JLCSOps.td` | `type_info`, `get_field`, `vcall`, `load_array_element` | High-level C++ interop primitives |
| **DWARF Parser** | `src/DWARFParser.jl` | `ClassInfo`, `VtableInfo`, `VirtualMethod` | Extract class/vtable info from debug symbols |
| **IR Generator** | `src/JLCSIRGenerator.jl` | `generate_jlcs_ir()`, `generate_type_info_ir()` | Emit MLIR module from DWARF |
| **Tier Selection** | `src/Wrapper.jl` (lines 1566-3835) | `is_ccall_safe()` | Route to ccall (Tier 1) or JIT (Tier 2) |
| **LTO Generation** | `src/Wrapper.jl` (lines 1850-3932) | `Base.llvmcall()` | Embed LLVM IR for zero-cost AOT dispatch |
| **JIT Dispatch** | `src/JITManager.jl` | `_lookup_cached()`, `invoke()` | Lock-free runtime function lookup & call |
| **Test IR** | `test/test_mlir.jl` | Module structure | Example MLIR generation & validation |

