# RepliBuild Internals

This section documents the internal modules that power RepliBuild. These are generally not needed for standard usage but are valuable for contributors or advanced integration.

## Compiler

The `Compiler` module handles the translation of C++ source code into LLVM IR and shared libraries.

```@autodocs
Modules = [RepliBuild.Compiler]
Order = [:function, :type]
Private = false
```

## Configuration Manager

The `ConfigurationManager` is the single source of truth for all build settings, handling TOML parsing and validation.

```@autodocs
Modules = [RepliBuild.ConfigurationManager]
Order = [:function, :type]
Private = false
```

## Discovery

The `Discovery` module scans the filesystem to identify C++ source files, headers, and dependencies.

```@autodocs
Modules = [RepliBuild.Discovery]
Order = [:function, :type]
Private = false
```
