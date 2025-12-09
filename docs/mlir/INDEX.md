# JLCS MLIR Dialect Documentation

> **Complete documentation for building MLIR dialects with Julia**

## Documentation Structure

### 📘 Getting Started

**[README.md](README.md)** - Start here!
- Installation and prerequisites
- Quick start guide
- Understanding MLIR from a Julia perspective
- Project structure walkthrough
- Building and testing
- Creating your own dialect

### 📗 Language Reference

**[TABLEGEN_GUIDE.md](TABLEGEN_GUIDE.md)** - Deep dive into TableGen
- TableGen syntax and concepts
- Defining dialects, types, and operations
- Attributes, traits, and assembly formats
- Custom builders and interfaces
- Complete examples and best practices

### 📙 Practical Guide

**[EXAMPLES.md](EXAMPLES.md)** - Real-world usage
- Struct field access patterns
- Virtual method dispatch
- Strided array operations
- JIT compilation and execution
- Building complete IR
- Debugging techniques

## Learning Path

### For Julia Developers New to MLIR

1. **Start**: Read [README.md](README.md) sections 1-6
2. **Build**: Follow the Quick Start to build the dialect
3. **Understand**: Read "Understanding MLIR from a Julia Perspective"
4. **Practice**: Try examples from [EXAMPLES.md](EXAMPLES.md)
5. **Deep Dive**: Study [TABLEGEN_GUIDE.md](TABLEGEN_GUIDE.md)
6. **Create**: Build your own dialect

### For MLIR Users Learning Julia

1. **Context**: Read the introduction in [README.md](README.md)
2. **Bindings**: Study how Julia calls MLIR C API in [EXAMPLES.md](EXAMPLES.md)
3. **Integration**: See how JLCS fits into RepliBuild toolchain
4. **Extend**: Use Julia metaprogramming to generate MLIR IR

### For Contributors

1. **Architecture**: Understand the full pipeline in [README.md](README.md)
2. **Code**: Review [src/mlir/](../../src/mlir/) source code
3. **Conventions**: Follow best practices in [TABLEGEN_GUIDE.md](TABLEGEN_GUIDE.md)
4. **Test**: Add tests following patterns in [EXAMPLES.md](EXAMPLES.md)

## Quick Links

### Tutorials
- [Your First Dialect](README.md#creating-your-own-dialect)
- [TableGen Basics](TABLEGEN_GUIDE.md#tablegen-syntax-basics)
- [Building IR from Julia](EXAMPLES.md#example-5-building-complete-ir)

### Reference
- [Dialect Definition](TABLEGEN_GUIDE.md#defining-dialects)
- [Operation Definition](TABLEGEN_GUIDE.md#defining-operations)
- [Type Definition](TABLEGEN_GUIDE.md#defining-types)
- [Assembly Format](TABLEGEN_GUIDE.md#assembly-format)

### Practical Examples
- [Struct Field Access](EXAMPLES.md#example-1-struct-field-access)
- [Virtual Calls](EXAMPLES.md#example-2-virtual-method-calls)
- [Array Operations](EXAMPLES.md#example-3-strided-array-operations)
- [JIT Compilation](EXAMPLES.md#example-4-jit-compilation)

### Troubleshooting
- [Build Issues](README.md#build-issues)
- [Runtime Issues](README.md#runtime-issues)
- [Debugging Tips](EXAMPLES.md#debugging-tips)

## External Resources

- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
- [TableGen Programmer's Reference](https://llvm.org/docs/TableGen/)
- [MLIR C API Documentation](https://mlir.llvm.org/docs/CAPI/)
- [Writing MLIR Passes](https://mlir.llvm.org/docs/PassManagement/)

## Source Code

- [Production Source](../../src/mlir/) - JLCS dialect implementation
- [Julia Bindings](../../src/MLIRNative.jl) - ccall interface
- [IR Generator](../../src/JLCSIRGenerator.jl) - IR generation utilities

## Related

- [RepliBuild Main README](../../README.md)
- [RepliBuild API Documentation](../../README.md#api-with-params)
- [DWARF Parser](../../src/DWARFParser.jl)

---

**Last Updated**: December 2024
**MLIR Version**: 18+
**Julia Version**: 1.9+
