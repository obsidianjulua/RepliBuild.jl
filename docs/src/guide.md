# User Guide

This guide covers the common workflows for using `RepliBuild.jl`.

## Basic Workflow

The standard workflow involves three steps: discovery, building, and wrapping.

### 1. Discovery

The `discover` function scans your directory for C++ files and generates a `replibuild.toml` configuration file.

```julia
RepliBuild.discover()
```

You can also specify a directory:

```julia
RepliBuild.discover("path/to/project")
```

### 2. Building

Once configured, the `build` function compiles your C++ code into a shared library.

```julia
RepliBuild.build()
```

This step performs:
- Compilation of C++ to LLVM IR.
- Linking and optimization.
- Generation of the shared library.
- Extraction of metadata for wrapping.

### 3. Wrapping

Finally, generate the Julia wrapper module.

```julia
RepliBuild.wrap()
```

This will create a Julia file in the `julia/` directory that you can `include` and `use`.

## Automated Workflow

You can chain these steps together using the flags in `discover`:

```julia
# Discover, Build, and Wrap in one go
RepliBuild.discover(build=true, wrap=true)
```

## Configuration

The `replibuild.toml` file controls the build process. You can edit this file to customize:
- Compiler flags
- Include directories
- Output names
- Optimization levels

See the **[Configuration Reference](config.md)** for a complete list of available options and sections.
