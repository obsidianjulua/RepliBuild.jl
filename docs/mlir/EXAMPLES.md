# JLCS Dialect Examples - Practical Usage Guide

> **Real-world examples of using the JLCS dialect from Julia**

## Table of Contents

1. [Basic Setup](#basic-setup)
2. [Example 1: Struct Field Access](#example-1-struct-field-access)
3. [Example 2: Virtual Method Calls](#example-2-virtual-method-calls)
4. [Example 3: Strided Array Operations](#example-3-strided-array-operations)
5. [Example 4: JIT Compilation](#example-4-jit-compilation)
6. [Example 5: Building Complete IR](#example-5-building-complete-ir)
7. [Example 6: Pass Pipeline](#example-6-pass-pipeline)
8. [Debugging Tips](#debugging-tips)

---

## Basic Setup

### 1. Build the Dialect

```bash
cd examples/Mlir
./build_dialect.sh
```

### 2. Start Julia

```julia
# Load MLIR bindings
include("../../src/MLIRNative.jl")
using .MLIRNative
```

### 3. Verify Installation

```julia
julia> MLIRNative.test_dialect()
======================================================================
 JLCS MLIR Dialect Test
======================================================================
Checking library... ✓
Creating MLIR context... ✓
Creating MLIR module... ✓
...
All tests passed!
======================================================================
```

---

## Example 1: Struct Field Access

### C++ Struct

```cpp
struct Point {
    double x;  // offset 0
    double y;  // offset 8
};
```

### MLIR IR with JLCS

We'll create IR that reads the `y` field (at offset 8):

```mlir
module {
  func.func @get_y(%ptr: !llvm.ptr) -> f64 {
    %y = jlcs.get_field %ptr, 8 : f64
    return %y : f64
  }
}
```

### Julia Code to Build This IR

```julia
using .MLIRNative

# Create context and module
ctx = create_context()
mod = create_module(ctx)

# Get the module operation (needed for adding functions)
mod_op = get_module_operation(mod)

# Create a location (for error reporting)
loc = ccall((:mlirLocationUnknownGet, libMLIR), MlirLocation, (MlirContext,), ctx)

# Create function type: (!llvm.ptr) -> f64
ptr_type = ccall((:mlirLLVMPointerTypeGet, libMLIR), MlirType, (MlirContext, Cuint), ctx, 0)
f64_type = ccall((:mlirF64TypeGet, libMLIR), MlirType, (MlirContext,), ctx)

input_types = [ptr_type]
result_types = [f64_type]

func_type = ccall(
    (:mlirFunctionTypeGet, libMLIR),
    MlirType,
    (MlirContext, Cint, Ptr{MlirType}, Cint, Ptr{MlirType}),
    ctx, 1, input_types, 1, result_types
)

# Create function operation
# (In practice, you'd use the Func dialect's function builder)

# Print the module
print_module(mod)

# Cleanup
destroy_context(ctx)
```

### Expected Challenges

The C API for building complex IR is verbose. For production use, consider:

1. **Write IR as text** and parse it
2. **Use C++ builders** and wrap them
3. **Generate IR programmatically** in C++

### Alternative: Parse Text IR

```julia
using .MLIRNative

ctx = create_context()

# Create module from string
ir_text = """
module {
  func.func @get_y(%ptr: !llvm.ptr) -> f64 {
    %y = jlcs.get_field %ptr, 8 : f64
    return %y : f64
  }
}
"""

# Parse the IR (note: requires parser setup)
# This is a simplified example - full implementation needs MLIR parser registration
```

---

## Example 2: Virtual Method Calls

### C++ Class with Virtual Method

```cpp
class Shape {
public:
    virtual double area() = 0;  // vtable slot 0
};

class Circle : public Shape {
    double radius;
public:
    double area() override { return 3.14159 * radius * radius; }
};
```

### MLIR IR with JLCS

```mlir
module {
  // Call Shape::area() via vtable
  func.func @call_area(%shape_ptr: !llvm.ptr) -> f64 {
    // vtable is at offset 0, area() is at slot 0
    %area = jlcs.vcall @Shape(%shape_ptr), 0, 0 : f64
    return %area : f64
  }
}
```

### Breakdown

- `jlcs.vcall @Shape(%shape_ptr), 0, 0 : f64`
  - `@Shape` - class name (symbol reference)
  - `%shape_ptr` - object pointer (first argument)
  - `0` - vtable offset in bytes (usually 0 for first field)
  - `0` - vtable slot index (0 = first virtual method)
  - `: f64` - return type

### Lowering to LLVM

After running the lowering pass, this becomes:

```mlir
func.func @call_area(%shape_ptr: !llvm.ptr) -> f64 {
  // 1. Load vtable pointer from object
  %c0 = arith.constant 0 : i64
  %vtable_addr = llvm.getelementptr %shape_ptr[%c0] : (!llvm.ptr, i64) -> !llvm.ptr
  %vtable_ptr = llvm.load %vtable_addr : !llvm.ptr -> !llvm.ptr

  // 2. Get function pointer from vtable
  %c0_slot = arith.constant 0 : i64
  %func_ptr_addr = llvm.getelementptr %vtable_ptr[%c0_slot] : (!llvm.ptr, i64) -> !llvm.ptr
  %func_ptr = llvm.load %func_ptr_addr : !llvm.ptr -> !llvm.ptr

  // 3. Call the function pointer
  %result = llvm.call %func_ptr(%shape_ptr) : !llvm.ptr, (!llvm.ptr) -> f64
  return %result : f64
}
```

---

## Example 3: Strided Array Operations

### Julia Array Wrapper

JLCS provides `ArrayView` type for cross-language arrays:

```mlir
// ArrayView layout:
// struct ArrayView {
//   void* data;      // offset 0
//   int64_t* shape;  // offset 8
//   int64_t* strides;// offset 16
//   int64_t rank;    // offset 24
// };
```

### Load Element from 2D Array

```mlir
module {
  func.func @get_element(%view: !llvm.ptr, %i: index, %j: index) -> f64 {
    %elem = jlcs.load_array_element %view[%i, %j] : f64
    return %elem : f64
  }
}
```

### How It Works

The operation computes: `data[i * strides[0] + j * strides[1]]`

After lowering:

```mlir
func.func @get_element(%view: !llvm.ptr, %i: index, %j: index) -> f64 {
  // Load data pointer (offset 0)
  %c0 = arith.constant 0 : i64
  %data_addr = llvm.getelementptr %view[%c0] : (!llvm.ptr, i64) -> !llvm.ptr
  %data_ptr = llvm.load %data_addr : !llvm.ptr -> !llvm.ptr

  // Load strides pointer (offset 16)
  %c16 = arith.constant 16 : i64
  %strides_addr = llvm.getelementptr %view[%c16] : (!llvm.ptr, i64) -> !llvm.ptr
  %strides_ptr = llvm.load %strides_addr : !llvm.ptr -> !llvm.ptr

  // Load stride[0]
  %c0_idx = arith.constant 0 : i64
  %stride0_addr = llvm.getelementptr %strides_ptr[%c0_idx] : (!llvm.ptr, i64) -> !llvm.ptr
  %stride0 = llvm.load %stride0_addr : !llvm.ptr -> i64

  // Load stride[1]
  %c1_idx = arith.constant 1 : i64
  %stride1_addr = llvm.getelementptr %strides_ptr[%c1_idx] : (!llvm.ptr, i64) -> !llvm.ptr
  %stride1 = llvm.load %stride1_addr : !llvm.ptr -> i64

  // Calculate offset: i * stride[0] + j * stride[1]
  %i_offset = arith.muli %i, %stride0 : i64
  %j_offset = arith.muli %j, %stride1 : i64
  %total_offset = arith.addi %i_offset, %j_offset : i64

  // Load element
  %elem_addr = llvm.getelementptr %data_ptr[%total_offset] : (!llvm.ptr, i64) -> !llvm.ptr
  %elem = llvm.load %elem_addr : !llvm.ptr -> f64

  return %elem : f64
}
```

### Store Element to Array

```mlir
func.func @set_element(%view: !llvm.ptr, %i: index, %j: index, %value: f64) {
  jlcs.store_array_element %value, %view[%i, %j] : f64
  return
}
```

---

## Example 4: JIT Compilation

### Compiling and Executing MLIR IR

```julia
using .MLIRNative

# 1. Create context and module
ctx = create_context()

# 2. Build IR (simplified - in practice, use builder API)
ir_text = """
module {
  func.func @add(%a: i64, %b: i64) -> i64 {
    %sum = arith.addi %a, %b : i64
    return %sum : i64
  }
}
"""

# Note: Full JIT example requires:
# - Parsing the IR text into a module
# - Running optimization passes
# - Lowering to LLVM dialect
# - Creating execution engine
# - Invoking the function

# The C API helpers in JLCSCHelpers.cpp provide JIT support:
# - jlcs_create_jit(module, optLevel, enableObjectDump)
# - jlcs_jit_lookup(jit, "function_name")
# - jlcs_jit_invoke(jit, "function_name", args)
```

### Full JIT Wrapper (Extended MLIRNative.jl)

```julia
module MLIRNative

# ... existing code ...

# JIT execution engine type
const MlirExecutionEngine = Ptr{Cvoid}

function create_jit(mod::MlirModule, opt_level::Int = 2)
    jit = ccall(
        (:jlcs_create_jit, libJLCS_path),
        MlirExecutionEngine,
        (MlirModule, Cint, Bool),
        mod, opt_level, false
    )

    if jit == C_NULL
        error("Failed to create JIT execution engine")
    end

    return jit
end

function jit_lookup(jit::MlirExecutionEngine, func_name::String)
    ptr = ccall(
        (:jlcs_jit_lookup, libJLCS_path),
        Ptr{Cvoid},
        (MlirExecutionEngine, Cstring),
        jit, func_name
    )

    if ptr == C_NULL
        error("Function '$func_name' not found in JIT")
    end

    return ptr
end

function destroy_jit(jit::MlirExecutionEngine)
    ccall(
        (:jlcs_jit_destroy, libJLCS_path),
        Cvoid,
        (MlirExecutionEngine,),
        jit
    )
end

end # module
```

### Using the JIT

```julia
# Create module with function
ctx = create_context()
mod = create_module(ctx)

# ... build IR for @add function ...

# Create JIT
jit = create_jit(mod, 2)  # optimization level 2

# Lookup function
add_ptr = jit_lookup(jit, "add")

# Call it! (wrap in Julia function)
function call_add(a::Int64, b::Int64)
    result_ref = Ref{Int64}(0)
    args = [Ref(a), Ref(b), result_ref]

    success = ccall(
        (:jlcs_jit_invoke, libJLCS_path),
        Bool,
        (MlirExecutionEngine, Cstring, Ptr{Ptr{Cvoid}}),
        jit, "add", args
    )

    return result_ref[]
end

# Test
@assert call_add(10, 32) == 42

# Cleanup
destroy_jit(jit)
destroy_context(ctx)
```

---

## Example 5: Building Complete IR

### Goal: Generate IR for C++ Method Call

Let's generate complete MLIR IR for this C++ code:

```cpp
struct Point {
    double x, y;
};

double distance(Point* p1, Point* p2) {
    double dx = p2->x - p1->x;
    double dy = p2->y - p1->y;
    return sqrt(dx*dx + dy*dy);
}
```

### Target MLIR IR

```mlir
module {
  func.func @distance(%p1: !llvm.ptr, %p2: !llvm.ptr) -> f64 {
    // Load p1->x (offset 0)
    %p1_x = jlcs.get_field %p1, 0 : f64

    // Load p1->y (offset 8)
    %p1_y = jlcs.get_field %p1, 8 : f64

    // Load p2->x (offset 0)
    %p2_x = jlcs.get_field %p2, 0 : f64

    // Load p2->y (offset 8)
    %p2_y = jlcs.get_field %p2, 8 : f64

    // dx = p2->x - p1->x
    %dx = arith.subf %p2_x, %p1_x : f64

    // dy = p2->y - p1->y
    %dy = arith.subf %p2_y, %p1_y : f64

    // dx*dx
    %dx2 = arith.mulf %dx, %dx : f64

    // dy*dy
    %dy2 = arith.mulf %dy, %dy : f64

    // dx*dx + dy*dy
    %sum = arith.addf %dx2, %dy2 : f64

    // sqrt(sum)
    %dist = math.sqrt %sum : f64

    return %dist : f64
  }
}
```

### Generation Strategy

For complex IR, it's easier to:

1. **Write the IR as a string template**
2. **Use Julia string interpolation** for parameters
3. **Parse the IR** using MLIR's parser

```julia
function generate_distance_function(struct_name::String, field_offsets::Dict{String, Int})
    x_offset = field_offsets["x"]
    y_offset = field_offsets["y"]

    ir = """
    module {
      func.func @distance(%p1: !llvm.ptr, %p2: !llvm.ptr) -> f64 {
        %p1_x = jlcs.get_field %p1, $x_offset : f64
        %p1_y = jlcs.get_field %p1, $y_offset : f64
        %p2_x = jlcs.get_field %p2, $x_offset : f64
        %p2_y = jlcs.get_field %p2, $y_offset : f64

        %dx = arith.subf %p2_x, %p1_x : f64
        %dy = arith.subf %p2_y, %p1_y : f64

        %dx2 = arith.mulf %dx, %dx : f64
        %dy2 = arith.mulf %dy, %dy : f64

        %sum = arith.addf %dx2, %dy2 : f64
        %dist = math.sqrt %sum : f64

        return %dist : f64
      }
    }
    """

    return ir
end

# Usage
offsets = Dict("x" => 0, "y" => 8)
ir_code = generate_distance_function("Point", offsets)
println(ir_code)
```

---

## Example 6: Pass Pipeline

### Running Optimization and Lowering Passes

MLIR transformations happen through **passes**:

```julia
# After building JLCS IR module...

# 1. Lower JLCS to LLVM dialect
run_pass(mod, "jlcs-lower-to-llvm")

# 2. Optimize LLVM IR
run_pass(mod, "llvm-optimize")

# 3. Convert to executable code
jit = create_jit(mod)
```

### Pass Pipeline in C++

For performance, implement pass pipelines in C++:

```cpp
// src/JLCSPasses.cpp
void runJLCSPipeline(mlir::ModuleOp module) {
    mlir::PassManager pm(module.getContext());

    // Add JLCS lowering pass
    pm.addPass(mlir::jlcs::createLowerJLCSToLLVMPass());

    // Add LLVM optimization passes
    pm.addPass(mlir::createCSEPass());           // Common subexpression elimination
    pm.addPass(mlir::createCanonicalizerPass()); // Canonicalization
    pm.addPass(mlir::createLoopInvariantCodeMotionPass());

    // Run pipeline
    if (failed(pm.run(module))) {
        llvm::errs() << "Pass pipeline failed\n";
    }
}
```

### Calling from Julia

```cpp
// src/JLCSCHelpers.cpp
extern "C" {

void jlcs_run_optimization_pipeline(MlirModule module) {
    ModuleOp mod = unwrap(module);
    runJLCSPipeline(mod);
}

}
```

```julia
# MLIRNative.jl
function optimize!(mod::MlirModule)
    ccall(
        (:jlcs_run_optimization_pipeline, libJLCS_path),
        Cvoid,
        (MlirModule,),
        mod
    )
end
```

---

## Debugging Tips

### 1. Print IR at Each Stage

```julia
ctx = create_context()
mod = create_module(ctx)

# Build IR...
println("=== Original IR ===")
print_module(mod)

# Run lowering pass...
println("\n=== After Lowering ===")
print_module(mod)

# Run optimization...
println("\n=== After Optimization ===")
print_module(mod)
```

### 2. Verify IR

```julia
function verify_module(mod::MlirModule)
    mod_op = get_module_operation(mod)

    is_valid = ccall(
        (:mlirOperationVerify, libMLIR),
        Bool,
        (MlirOperation,),
        mod_op
    )

    if !is_valid
        error("Module verification failed!")
    end

    println("✓ Module is valid")
end
```

### 3. Enable MLIR Diagnostics

```cpp
// In C++ code
context->getDiagEngine().registerHandler([](Diagnostic &diag) {
    llvm::errs() << diag << "\n";
});
```

### 4. Dump Generated Assembly

```julia
# After creating JIT
ccall(
    (:jlcs_jit_dump_to_object, libJLCS_path),
    Cvoid,
    (MlirExecutionEngine, Cstring),
    jit, "output.o"
)

# Disassemble
run(`objdump -d output.o`)
```

### 5. Check TableGen Output

```bash
# Manually run TableGen to see generated code
cd examples/Mlir/build
mlir-tblgen -gen-op-decls ../JLCS.td -I /usr/include

# Check for syntax errors
mlir-tblgen --help
```

### 6. Use MLIR-OPT for Testing

```bash
# Create test IR file
cat > test.mlir << 'EOF'
module {
  func.func @test(%ptr: !llvm.ptr) -> f64 {
    %val = jlcs.get_field %ptr, 8 : f64
    return %val : f64
  }
}
EOF

# Test parsing
mlir-opt test.mlir

# Run lowering pass
mlir-opt test.mlir --jlcs-lower-to-llvm

# Full pipeline
mlir-opt test.mlir \
  --jlcs-lower-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts
```

---

## Complete Working Example

Here's a full end-to-end example you can run:

```julia
#!/usr/bin/env julia

# Load MLIR bindings
include("../../src/MLIRNative.jl")
using .MLIRNative

println("="^70)
println(" JLCS Dialect Example: Struct Field Access")
println("="^70)

# 1. Create context and module
println("\n1. Creating MLIR context...")
ctx = create_context()
println("   ✓ Context created")

println("\n2. Creating MLIR module...")
mod = create_module(ctx)
println("   ✓ Module created")

# 3. Display empty module
println("\n3. Empty module IR:")
print_module(mod)

# 4. In a real application, you would:
#    - Build operations using the MLIR API
#    - Add functions to the module
#    - Run optimization passes
#    - Compile to native code
#    - Execute via JIT

println("\n4. Next steps:")
println("   - Add function definitions")
println("   - Build JLCS operations")
println("   - Run lowering passes")
println("   - Execute via JIT")

# 5. Cleanup
println("\n5. Cleaning up...")
destroy_context(ctx)
println("   ✓ Context destroyed")

println("\n" * "="^70)
println(" Example completed successfully!")
println("="^70)
```

Save as `example.jl` and run:

```bash
julia examples/Mlir/example.jl
```

---

## Next Steps

1. **Study the C API**: Review `src/JLCSCHelpers.cpp` for available functions
2. **Extend MLIRNative.jl**: Add more high-level builders
3. **Create domain-specific IR builders**: Abstract common patterns
4. **Integrate with RepliBuild**: Connect MLIR generation to DWARF parsing

### Advanced Topics (Future Documentation)

- **Type inference and verification**
- **Custom optimization passes**
- **Multi-threaded compilation**
- **GPU code generation**
- **Automatic differentiation**

---

**Happy MLIR hacking!**
