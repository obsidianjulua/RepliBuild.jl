# MLIR & JLCS Dialect

`RepliBuild.MLIRNative` provides low-level bindings to the Multi-Level Intermediate Representation (MLIR) C API, specifically tailored for the custom JLCS (Julia-C++ Schema) dialect. This module is intended for advanced users who need to generate or manipulate IR directly or interface with the JIT compiler.

## Quick Start

The core workflow involves creating a context, defining a module, and managing JIT execution.

```julia
using RepliBuild.MLIRNative

# Use the helper macro for safe context management
@with_context begin
    # Create an empty module
    mod = create_module(ctx)
    
    # Parse some IR (or generate it programmatically)
    parsed_mod = parse_module(ctx, "...")
    
    # Print the result
    print_module(parsed_mod)
end
```

## Context & Modules

Manage the lifecycle of MLIR contexts and modules.

```@docs
RepliBuild.MLIRNative.create_context
RepliBuild.MLIRNative.destroy_context
RepliBuild.MLIRNative.@with_context
RepliBuild.MLIRNative.create_module
RepliBuild.MLIRNative.parse_module
RepliBuild.MLIRNative.clone_module
RepliBuild.MLIRNative.print_module
```

## JIT Execution

Compile and execute MLIR modules on the fly.

```@docs
RepliBuild.MLIRNative.create_jit
RepliBuild.MLIRNative.destroy_jit
RepliBuild.MLIRNative.register_symbol
RepliBuild.MLIRNative.lookup
RepliBuild.MLIRNative.jit_invoke
RepliBuild.MLIRNative.invoke_safe
```

## Transformations

Apply passes to your MLIR modules.

```@docs
RepliBuild.MLIRNative.lower_to_llvm
```

## Diagnostics

```@docs
RepliBuild.MLIRNative.test_dialect
```
