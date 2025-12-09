# TableGen Deep Dive for Julia Developers

> **Mastering MLIR's code generation language from a Julia perspective**

## Table of Contents

1. [What is TableGen?](#what-is-tablegen)
2. [TableGen Syntax Basics](#tablegen-syntax-basics)
3. [Defining Dialects](#defining-dialects)
4. [Defining Types](#defining-types)
5. [Defining Operations](#defining-operations)
6. [Attributes and Parameters](#attributes-and-parameters)
7. [Operation Traits](#operation-traits)
8. [Assembly Format](#assembly-format)
9. [Custom Builders](#custom-builders)
10. [Interfaces](#interfaces)
11. [Complete Examples](#complete-examples)
12. [Best Practices](#best-practices)

---

## What is TableGen?

**TableGen** is a domain-specific language (DSL) used by LLVM/MLIR to generate C++ code. Think of it as:

- **Julia metaprogramming** (`@generated`, macros) but external
- **Schema definition** like JSON Schema or Protocol Buffers
- **Code template system** that ensures consistency

### TableGen vs Julia Comparison

| TableGen | Julia Equivalent | Purpose |
|----------|------------------|---------|
| `def MyOp : Op<...>` | `struct MyOp <: Operation` | Define new type |
| `let arguments = (ins ...)` | Field declarations | Operation inputs |
| `let results = (outs ...)` | Return type annotation | Operation outputs |
| `mlir-tblgen -gen-op-decls` | `@generated` function | Generate code |

### The TableGen Pipeline

```
YourDialect.td  ──→  mlir-tblgen  ──→  Generated.h.inc
                                  ├──→  Generated.cpp.inc
                                  └──→  (Used by your C++ code)
```

---

## TableGen Syntax Basics

### File Structure

Every TableGen file (`.td`) has this structure:

```tablegen
// 1. Include base definitions
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

// 2. Define your dialect
def MyDialect : Dialect { ... }

// 3. Define base classes (optional)
class MyDialect_Op<string mnemonic> : Op<MyDialect, mnemonic> { }

// 4. Define operations
def MyFirstOp : MyDialect_Op<"my_first"> { ... }
def MySecondOp : MyDialect_Op<"my_second"> { ... }

// 5. Define types
def MyType : TypeDef<MyDialect, "MyType"> { ... }
```

### Basic Syntax Elements

#### 1. Classes and Records

```tablegen
// Class: template/abstract definition
class BaseOp<string mnemonic> {
  string opName = mnemonic;
}

// Record: concrete instance (generates code)
def AddOp : BaseOp<"add"> {
  // Inherits opName = "add"
}
```

**Julia analogy**:
```julia
# Class
abstract type BaseOp{Mnemonic} end

# Record
struct AddOp <: BaseOp{"add"} end
```

#### 2. Let Statements

```tablegen
def MyOp : Op<...> {
  let summary = "Brief description";
  let description = [{
    Multi-line description
    with markdown support
  }];
  let arguments = (ins I64:$input);
  let results = (outs I64:$output);
}
```

**Julia analogy**:
```julia
struct MyOp
    summary::String
    description::String
    arguments::NamedTuple
    results::NamedTuple
end
```

#### 3. DAG (Directed Acyclic Graph) Types

DAGs represent structured data:

```tablegen
let arguments = (ins
  I64:$first,
  F64:$second,
  Optional<I32>:$third
);
```

**Structure**: `(operator child1, child2, ...)`

**Julia analogy**:
```julia
arguments = (
    first = Int64,
    second = Float64,
    third = Union{Int32, Nothing}
)
```

---

## Defining Dialects

### Minimal Dialect

```tablegen
include "mlir/IR/OpBase.td"

def MyDialect : Dialect {
  let name = "mydialect";
  let cppNamespace = "::mlir::mydialect";
  let summary = "One-line description";
  let description = [{
    Detailed description of what this dialect does.
    Can span multiple lines.
  }];
}
```

### Generated C++ Header

This generates in `MyDialect.h.inc`:

```cpp
namespace mlir {
namespace mydialect {

class MyDialectDialect : public ::mlir::Dialect {
public:
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("mydialect");
  }

  void initialize();
  // ... more methods
};

} // namespace mydialect
} // namespace mlir
```

### Dialect with Custom Methods

```tablegen
def MyDialect : Dialect {
  let name = "mydialect";
  let cppNamespace = "::mlir::mydialect";

  // Add custom C++ methods to dialect class
  let extraClassDeclaration = [{
    // These methods will be added to MyDialectDialect class
    void registerTypes();
    void registerOps();

    // Custom type parsing
    Type parseType(DialectAsmParser &parser) const override;
    void printType(Type type, DialectAsmPrinter &printer) const override;
  }];
}
```

**When to use**: Add custom parsing, type conversions, or dialect-wide utilities.

---

## Defining Types

### Basic Type Definition

```tablegen
// Define base class for your dialect's types
class MyDialect_Type<string name, string typeMnemonic>
    : TypeDef<MyDialect, name> {
  let mnemonic = typeMnemonic;
}

// Define concrete type
def IntegerType : MyDialect_Type<"Integer", "int"> {
  let summary = "Arbitrary precision integer";

  // Type parameters (like Julia type parameters)
  let parameters = (ins
    "unsigned":$width
  );

  // Assembly format: !mydialect.int<32>
  let assemblyFormat = "`<` $width `>`";
}
```

### MLIR IR Usage

```mlir
%value : !mydialect.int<32>
```

### Julia Analogy

```julia
struct IntegerType{Width}
    width::UInt
end

# Usage
x = IntegerType{32}(32)
```

### Complex Type with Multiple Parameters

```tablegen
def TensorType : MyDialect_Type<"Tensor", "tensor"> {
  let summary = "Multidimensional array type";

  let parameters = (ins
    "Type":$elementType,                    // nested type
    ArrayRefParameter<"int64_t">:$shape,    // array parameter
    "unsigned":$rank                         // scalar parameter
  );

  let assemblyFormat = [{
    `<` $elementType `,` `[` $shape `]` `,` $rank `>`
  }];
}
```

**MLIR IR**:
```mlir
!mydialect.tensor<i64, [4, 8, 16], 3>
```

**Julia equivalent**:
```julia
struct TensorType{ElementType, Shape, Rank}
    element_type::Type
    shape::Vector{Int64}
    rank::UInt
end
```

### Type Storage (Advanced)

For complex types, implement custom storage in C++:

```cpp
// src/MyDialectTypes.cpp
struct TensorTypeStorage : public mlir::TypeStorage {
    using KeyTy = std::tuple<Type, ArrayRef<int64_t>, unsigned>;

    TensorTypeStorage(Type elemType, ArrayRef<int64_t> shape, unsigned rank)
        : elementType(elemType), shape(shape), rank(rank) {}

    bool operator==(const KeyTy &key) const {
        return std::get<0>(key) == elementType &&
               std::get<1>(key) == shape &&
               std::get<2>(key) == rank;
    }

    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_combine(std::get<0>(key), std::get<1>(key), std::get<2>(key));
    }

    Type elementType;
    ArrayRef<int64_t> shape;
    unsigned rank;
};
```

---

## Defining Operations

### Minimal Operation

```tablegen
def AddOp : MyDialect_Op<"add"> {
  let summary = "Integer addition";

  let arguments = (ins
    I64:$lhs,
    I64:$rhs
  );

  let results = (outs
    I64:$result
  );
}
```

**MLIR IR**:
```mlir
%result = mydialect.add %lhs, %rhs : (i64, i64) -> i64
```

**Julia concept**:
```julia
function add(lhs::Int64, rhs::Int64)::Int64
    # Implementation in lowering pass
end
```

### Operation Anatomy

```tablegen
def CompleteOp : MyDialect_Op<"example"> {
  // 1. Documentation
  let summary = "One-line summary";
  let description = [{
    Detailed description with examples:

    ```mlir
    %result = mydialect.example %input : i64 -> f64
    ```
  }];

  // 2. Arguments (inputs)
  let arguments = (ins
    AnyType:$input,           // Value argument
    I64Attr:$constParam       // Compile-time attribute
  );

  // 3. Results (outputs)
  let results = (outs
    AnyType:$output
  );

  // 4. Regions (code blocks)
  let regions = (region
    SizedRegion<1>:$body      // Single required region
  );

  // 5. Successors (control flow)
  let successors = (successor
    VariadicSuccessor<AnySuccessor>:$targets
  );

  // 6. Traits
  let traits = [Pure, Commutative];

  // 7. Assembly format
  let assemblyFormat = "$input `,` $constParam attr-dict `:` type($input) `->` type($output)";
}
```

---

## Attributes and Parameters

### Attribute Types

| TableGen Type | MLIR Type | Julia Equivalent | Example |
|---------------|-----------|------------------|---------|
| `I64Attr` | Integer | `Int64` | `42` |
| `F64Attr` | Float | `Float64` | `3.14` |
| `StrAttr` | String | `String` | `"hello"` |
| `ArrayAttr` | Array | `Vector` | `[1, 2, 3]` |
| `TypeAttr` | Type | `Type` | `i64` |
| `SymbolRefAttr` | Symbol | `Symbol` | `@func_name` |

### Using Attributes

```tablegen
def LoadOp : MyDialect_Op<"load"> {
  let arguments = (ins
    AnyType:$ptr,               // Runtime value
    I64Attr:$offset,            // Compile-time constant
    StrAttr:$name,              // String metadata
    DefaultValuedAttr<I32Attr, "0">:$alignment  // Optional with default
  );

  let results = (outs
    AnyType:$result
  );
}
```

**MLIR IR**:
```mlir
%result = mydialect.load %ptr, 16, "field_name", 8 : !mydialect.ptr -> i64
```

**Julia analogy**:
```julia
@kwdef struct LoadOp
    ptr::Any                    # Runtime
    offset::Int64               # Compile-time
    name::String                # Metadata
    alignment::Int32 = 0        # Default value
end
```

### Optional and Variadic Arguments

```tablegen
def CallOp : MyDialect_Op<"call"> {
  let arguments = (ins
    SymbolRefAttr:$callee,                  // Required
    Variadic<AnyType>:$args,                // 0 or more
    Optional<I32Attr>:$inline_threshold     // 0 or 1
  );
}
```

**Usage**:
```mlir
// No optional arg
mydialect.call @func(%a, %b, %c)

// With optional arg
mydialect.call @func(%a, %b) {inline_threshold = 100}
```

---

## Operation Traits

Traits define operation properties and enable optimizations.

### Common Traits

```tablegen
def MyOp : MyDialect_Op<"my_op", [
  Pure,                        // No side effects (like @pure in Julia)
  Commutative,                 // a+b == b+a
  Idempotent,                  // f(f(x)) == f(x)
  IsolatedFromAbove,           // Self-contained (like Julia function scope)
  SameOperandsAndResultType,   // All types must match
  Terminator                   // Ends a basic block (like return/throw)
]> {
  // ...
}
```

### Trait Meanings

| Trait | Julia Analogy | Effect |
|-------|---------------|--------|
| `Pure` | `@pure` | Can be optimized away, reordered |
| `Commutative` | `a + b == b + a` | Operands can be swapped |
| `IsolatedFromAbove` | Function scope | Can't reference outer values |
| `Terminator` | `return`, `throw` | Ends control flow block |

### Custom Traits

```tablegen
// Define custom trait
def MyCustomTrait : OpTrait, NativeOpTrait<"MyCustomTrait"> {
  let cppNamespace = "::mlir::mydialect";
}

// Use it
def MyOp : MyDialect_Op<"my_op", [MyCustomTrait]> {
  // ...
}
```

Implement in C++:
```cpp
// IR/MyDialectTraits.h
template <typename ConcreteOp>
class MyCustomTrait : public OpTrait::TraitBase<ConcreteOp, MyCustomTrait> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    // Custom verification logic
    return success();
  }
};
```

---

## Assembly Format

The assembly format defines how operations appear in MLIR IR.

### Declarative Format

```tablegen
def AddOp : MyDialect_Op<"add"> {
  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64:$result);

  // Simple format
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}
```

**Generated IR**:
```mlir
%result = mydialect.add %a, %b : i64
```

### Format Directives

| Directive | Purpose | Example |
|-----------|---------|---------|
| `$varname` | Print variable | `$input` |
| `` ` ` `` | Literal | `` `+` `` |
| `attr-dict` | Attributes | `{key = value}` |
| `type($var)` | Type of variable | `i64` |
| `(` `)` | Grouping | `($a, $b)` |
| `[` `]` | Optional | `[`optional`]` |
| `custom<...>` | Custom parser | `custom<MyParser>($var)` |

### Advanced Assembly Formats

#### Optional Operands

```tablegen
def LoadOp : MyDialect_Op<"load"> {
  let arguments = (ins
    AnyType:$ptr,
    Optional<I64>:$offset
  );

  let assemblyFormat = [{
    $ptr (`,` $offset^)? attr-dict `:` type($ptr)
  }];
}
```

**Generated IR**:
```mlir
// Without offset
%val = mydialect.load %ptr : !mydialect.ptr

// With offset
%val = mydialect.load %ptr, %offset : !mydialect.ptr
```

#### Variadic Operands

```tablegen
def CallOp : MyDialect_Op<"call"> {
  let arguments = (ins
    SymbolRefAttr:$callee,
    Variadic<AnyType>:$args
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $callee `(` $args `)` attr-dict `:`
    functional-type($args, $results)
  }];
}
```

**Generated IR**:
```mlir
%r1, %r2 = mydialect.call @func(%a, %b, %c) : (i64, i64, i64) -> (i64, f64)
```

#### Regions

```tablegen
def IfOp : MyDialect_Op<"if"> {
  let arguments = (ins I1:$condition);
  let regions = (region
    SizedRegion<1>:$thenRegion,
    AnyRegion:$elseRegion
  );

  let assemblyFormat = [{
    $condition $thenRegion (`else` $elseRegion^)? attr-dict
  }];
}
```

**Generated IR**:
```mlir
mydialect.if %cond {
  // then block
} else {
  // else block
}
```

### Custom Format Parser

When declarative format isn't enough:

```tablegen
def ComplexOp : MyDialect_Op<"complex"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);

  // Disable automatic format
  let hasCustomAssemblyFormat = 1;
}
```

Implement in C++:
```cpp
// src/MyDialectOps.cpp
ParseResult ComplexOp::parse(OpAsmParser &parser, OperationState &result) {
    // Custom parsing logic
    OpAsmParser::UnresolvedOperand input;
    Type type;

    if (parser.parseOperand(input) ||
        parser.parseColon() ||
        parser.parseType(type))
        return failure();

    // Resolve and add to result
    parser.resolveOperand(input, type, result.operands);
    result.addTypes(type);

    return success();
}

void ComplexOp::print(OpAsmPrinter &printer) {
    // Custom printing logic
    printer << " " << getInput() << " : " << getInput().getType();
}
```

---

## Custom Builders

Builders provide convenient C++ constructors for operations.

### Default Builders

TableGen automatically generates builders based on arguments/results:

```tablegen
def AddOp : MyDialect_Op<"add"> {
  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64:$result);
}
```

**Generated builder**:
```cpp
static void build(OpBuilder &builder, OperationState &state,
                  Value lhs, Value rhs) {
  // Infers result type from operands
}
```

### Skip Default Builders

```tablegen
def CustomOp : MyDialect_Op<"custom"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);

  // Don't generate default builders
  let skipDefaultBuilders = 1;

  // Define custom builders
  let builders = [
    OpBuilder<(ins "Value":$input, "Type":$outputType), [{
      // Builder body in C++
      $_state.addOperands(input);
      $_state.addTypes(outputType);
    }]>,

    OpBuilder<(ins "Value":$input), [{
      // Infer output type from input
      build($_builder, $_state, input, input.getType());
    }]>
  ];
}
```

**Usage in C++**:
```cpp
// Use first builder
auto op1 = builder.create<CustomOp>(loc, input, outputType);

// Use second builder (type inference)
auto op2 = builder.create<CustomOp>(loc, input);
```

### Builder with Complex Logic

```tablegen
def MatMulOp : MyDialect_Op<"matmul"> {
  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs
  );
  let results = (outs AnyType:$result);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
      // Compute result shape from inputs
      auto lhsType = lhs.getType().cast<TensorType>();
      auto rhsType = rhs.getType().cast<TensorType>();

      auto lhsShape = lhsType.getShape();
      auto rhsShape = rhsType.getShape();

      // Result shape: [M, N] = [M, K] @ [K, N]
      SmallVector<int64_t> resultShape = {lhsShape[0], rhsShape[1]};
      auto resultType = TensorType::get(resultShape, lhsType.getElementType());

      $_state.addOperands({lhs, rhs});
      $_state.addTypes(resultType);
    }]>
  ];
}
```

---

## Interfaces

Interfaces add common behavior across multiple operations.

### Using Built-in Interfaces

```tablegen
def GetElementOp : MyDialect_Op<"get_element", [
  MemoryEffects<[MemRead]>,        // Reads memory
  DeclareOpInterfaceMethods<InferTypeOpInterface>  // Implements type inference
]> {
  let arguments = (ins AnyType:$container, I64:$index);
  let results = (outs AnyType:$element);
}
```

Implement interface in C++:
```cpp
// src/MyDialectOps.cpp
LogicalResult GetElementOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> location,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {

  // Infer element type from container
  auto containerType = operands[0].getType().cast<ContainerType>();
  inferredReturnTypes.push_back(containerType.getElementType());

  return success();
}
```

### Defining Custom Interfaces

```tablegen
// Define the interface
def MyInterface : OpInterface<"MyInterface"> {
  let description = "Operations that support my custom behavior";

  let methods = [
    InterfaceMethod<
      /*description=*/"Get the operation's priority",
      /*returnType=*/"unsigned",
      /*methodName=*/"getPriority",
      /*arguments=*/(ins),
      /*methodBody=*/[{
        // Default implementation
        return 0;
      }],
      /*defaultImplementation=*/[{
        return $_op.getPriorityAttr().getInt();
      }]
    >
  ];
}

// Use the interface
def HighPriorityOp : MyDialect_Op<"high_priority", [
  DeclareOpInterfaceMethods<MyInterface>
]> {
  let arguments = (ins I32Attr:$priority);
}
```

Implement in C++:
```cpp
// IR/MyDialectInterfaces.h (generated from TableGen)
class MyInterface : public OpInterface<MyInterface, MyInterfaceInterfaceTraits> {
public:
  using OpInterface<MyInterface, MyInterfaceInterfaceTraits>::OpInterface;

  unsigned getPriority() {
    return getImpl()->getPriority(getOperation());
  }
};

// src/MyDialectOps.cpp
unsigned HighPriorityOp::getPriority() {
  return getPriorityAttr().getInt();
}
```

---

## Complete Examples

### Example 1: Simple Arithmetic Dialect

```tablegen
//===- ArithDialect.td - Simple arithmetic dialect ----------*- tablegen -*-===//

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect Definition
//===----------------------------------------------------------------------===//

def Arith_Dialect : Dialect {
  let name = "arith";
  let cppNamespace = "::mlir::arith";
  let summary = "Basic arithmetic operations";
}

//===----------------------------------------------------------------------===//
// Base Classes
//===----------------------------------------------------------------------===//

class Arith_Op<string mnemonic, list<Trait> traits = []> :
    Op<Arith_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

def AddIOp : Arith_Op<"addi", [Pure, Commutative]> {
  let summary = "Integer addition";
  let description = [{
    Adds two integer operands and returns the result.

    Example:
    ```mlir
    %sum = arith.addi %a, %b : i64
    ```
  }];

  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";

  let hasFolder = 1;  // Enable constant folding
}

def MulIOp : Arith_Op<"muli", [Pure, Commutative]> {
  let summary = "Integer multiplication";

  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def ConstantOp : Arith_Op<"constant", [Pure]> {
  let summary = "Integer constant";

  let arguments = (ins I64Attr:$value);
  let results = (outs I64:$result);

  let assemblyFormat = "$value attr-dict";

  let builders = [
    OpBuilder<(ins "int64_t":$value), [{
      auto attr = $_builder.getI64IntegerAttr(value);
      build($_builder, $_state, $_builder.getI64Type(), attr);
    }]>
  ];
}
```

### Example 2: Control Flow Dialect

```tablegen
//===- ControlFlowDialect.td - Control flow operations -----*- tablegen -*-===//

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def CF_Dialect : Dialect {
  let name = "cf";
  let cppNamespace = "::mlir::cf";
  let summary = "Control flow operations";
}

class CF_Op<string mnemonic, list<Trait> traits = []> :
    Op<CF_Dialect, mnemonic, traits>;

def BranchOp : CF_Op<"br", [Terminator]> {
  let summary = "Unconditional branch";

  let successors = (successor AnySuccessor:$dest);
  let arguments = (ins Variadic<AnyType>:$destOperands);

  let assemblyFormat = "$dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict";
}

def CondBranchOp : CF_Op<"cond_br", [Terminator]> {
  let summary = "Conditional branch";

  let arguments = (ins
    I1:$condition,
    Variadic<AnyType>:$trueOperands,
    Variadic<AnyType>:$falseOperands
  );

  let successors = (successor
    AnySuccessor:$trueDest,
    AnySuccessor:$falseDest
  );

  let assemblyFormat = [{
    $condition `,`
    $trueDest (`(` $trueOperands^ `:` type($trueOperands) `)`)? `,`
    $falseDest (`(` $falseOperands^ `:` type($falseOperands) `)`)?
    attr-dict
  }];
}
```

---

## Best Practices

### 1. Naming Conventions

```tablegen
// Good: Clear, consistent naming
def LoadOp : MyDialect_Op<"load"> { ... }
def StoreOp : MyDialect_Op<"store"> { ... }
def GetFieldOp : MyDialect_Op<"get_field"> { ... }

// Bad: Inconsistent, unclear
def LdOp : MyDialect_Op<"l"> { ... }
def StOp : MyDialect_Op<"save"> { ... }
def FldGet : MyDialect_Op<"field"> { ... }
```

### 2. Documentation

```tablegen
// Good: Comprehensive documentation
def MatMulOp : MyDialect_Op<"matmul"> {
  let summary = "Matrix multiplication operation";
  let description = [{
    Performs matrix multiplication C = A @ B where:
    - A has shape [M, K]
    - B has shape [K, N]
    - C has shape [M, N]

    Example:
    ```mlir
    %c = mydialect.matmul %a, %b : (tensor<4x8xf64>, tensor<8x16xf64>) -> tensor<4x16xf64>
    ```
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
}

// Bad: No documentation
def MatMulOp : MyDialect_Op<"matmul"> {
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
}
```

### 3. Use Appropriate Traits

```tablegen
// Good: Accurate traits enable optimizations
def AddOp : MyDialect_Op<"add", [Pure, Commutative]> {
  // Compiler knows: no side effects, order doesn't matter
}

def LoadOp : MyDialect_Op<"load", [MemoryEffects<[MemRead]>]> {
  // Compiler knows: reads memory, but doesn't write
}

// Bad: Missing traits prevent optimizations
def AddOp : MyDialect_Op<"add"> {
  // Compiler assumes: might have side effects, order matters
}
```

### 4. Assembly Format Clarity

```tablegen
// Good: Readable, unambiguous
def CallOp : MyDialect_Op<"call"> {
  let assemblyFormat = [{
    $callee `(` $args `)` attr-dict `:` functional-type($args, $results)
  }];
}
// Result: mydialect.call @func(%a, %b) : (i64, i64) -> i64

// Bad: Hard to parse
def CallOp : MyDialect_Op<"call"> {
  let assemblyFormat = "$callee $args attr-dict type($results)";
}
// Result: mydialect.call @func%a%b i64 (confusing!)
```

### 5. Builder Ergonomics

```tablegen
// Good: Multiple convenient builders
def LoadOp : MyDialect_Op<"load"> {
  let arguments = (ins AnyType:$ptr, I64Attr:$offset);
  let results = (outs AnyType:$value);

  let builders = [
    // Full control
    OpBuilder<(ins "Value":$ptr, "IntegerAttr":$offset, "Type":$resultType)>,

    // Convenience: infer type from pointer
    OpBuilder<(ins "Value":$ptr, "int64_t":$offset), [{
      auto ptrType = ptr.getType().cast<PointerType>();
      auto elemType = ptrType.getElementType();
      auto offsetAttr = $_builder.getI64IntegerAttr(offset);
      build($_builder, $_state, elemType, ptr, offsetAttr);
    }]>
  ];
}

// Usage in C++
builder.create<LoadOp>(loc, ptr, 16);  // Easy!
```

### 6. Type Safety

```tablegen
// Good: Specific types where possible
def MatMulOp : MyDialect_Op<"matmul"> {
  let arguments = (ins
    2DTensorOf<[F32, F64]>:$lhs,  // Must be 2D float tensor
    2DTensorOf<[F32, F64]>:$rhs
  );
  let results = (outs 2DTensorOf<[F32, F64]>:$result);
}

// Bad: Too permissive
def MatMulOp : MyDialect_Op<"matmul"> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);  // Anything goes!
  let results = (outs AnyType:$result);
}
```

---

## Common Patterns

### Pattern 1: Operation with Optional Result

```tablegen
def CallOp : MyDialect_Op<"call"> {
  let arguments = (ins SymbolRefAttr:$callee, Variadic<AnyType>:$args);
  let results = (outs Optional<AnyType>:$result);  // void or non-void

  let assemblyFormat = [{
    $callee `(` $args `)` attr-dict `:` functional-type($args, results)
  }];
}
```

### Pattern 2: Memory Operations

```tablegen
def LoadOp : MyDialect_Op<"load", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins AnyType:$ptr, I64Attr:$offset);
  let results = (outs AnyType:$value);
}

def StoreOp : MyDialect_Op<"store", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins AnyType:$value, AnyType:$ptr, I64Attr:$offset);
  let results = (outs);
}
```

### Pattern 3: Type-Polymorphic Operations

```tablegen
def CastOp : MyDialect_Op<"cast", [Pure]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";

  let hasVerifier = 1;  // Add custom verification
}
```

---

## Next Steps

1. **Practice**: Modify the JLCS dialect operations
2. **Experiment**: Add your own operations
3. **Read**: Study MLIR's built-in dialects (`Arith`, `Func`, `LLVM`)
4. **Build**: Create a dialect for your domain

### Further Reading

- [MLIR ODS Documentation](https://mlir.llvm.org/docs/DefiningDialects/Operations/)
- [TableGen Language Reference](https://llvm.org/docs/TableGen/ProgRef.html)
- [MLIR Dialect Examples](https://github.com/llvm/llvm-project/tree/main/mlir/test/Dialect)

---

**Questions?** Open an issue or check the MLIR forums!
