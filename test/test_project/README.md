## Easy Example for Bindings

```julia
julia> using RepliBuild

julia> RepliBuild.discover(build=true, wrap=true)
 RepliBuild Discovery Pipeline
======================================================================
 Target: .
🔒 Safety: Scoped to ~/RepliBuild.jl/test/test_project/ and subdirectories only

 Stage 1: Scanning files...
   Scanning scope: ~/RepliBuild.jl/test/test_project/
    Scan Results:
      C++ Sources:    1
      C++ Headers:    1
      C Sources:      0
      C Headers:      0
      Binaries:       0
      Static Libs:    0
      Shared Libs:    0
      Julia Files:    0
      Total Files:    3

 Stage 2: Detecting binaries...
    No binaries detected

 Stage 3: Building include paths...
   Found 3 include directories

 Stage 4: Walking AST dependencies...
Initializing LLVM Toolchain (auto-discover)
┌ Warning: In-tree LLVM not found: ErrorException("RepliBuild LLVM installation not found at: ~/RepliBuild.jl/LLVM\nExpected location: ~/.julia/julia/RepliBuild/LLVM")
└ @ RepliBuild.LLVMEnvironment ~/RepliBuild.jl/src/LLVMEnvironment.jl:158
[ Info: Searching for system LLVM installation...
[ Info: Using system LLVM toolchain at: /usr
   Root: /usr
   Source: system
   Version: 21.1.6 (LLVM 21.1.6)
   Tools: Auto-discovering...
   Tools: Discovered 48 tools (cached for next run)
   Libraries: 2723 discovered
 LLVM Toolchain initialized
      Dependency graph built:
      Files analyzed: 2
      Include relationships: 128
======================================================================
Dependency Graph Summary
======================================================================

 File Statistics:
   Total files:     2
   Headers:         1
   Sources:         1
   With errors:     1

 Dependency Statistics:
   Total includes:  5
   Avg per file:    2.5

 Structure Statistics:
   Namespaces:      0
   Classes:         14
   Functions:       49

 Compilation Order:
   129 files in dependency order

  Errors:
   ❌ math_types.h:
      • Clang preprocessing failed with exit code 1
======================================================================
Exported dependency graph: ./.replibuild_cache/dependency_graph.json

 Stage 5: Generating replibuild.toml...

 Stage 6: Initializing LLVM toolchain...
Initializing LLVM Toolchain (auto-discover)
┌ Warning: In-tree LLVM not found: ErrorException("RepliBuild LLVM installation not found at: ~/RepliBuild.jl/LLVM\nExpected location: ~/.julia/julia/RepliBuild/LLVM")
└ @ RepliBuild.LLVMEnvironment ~/RepliBuild.jl/src/LLVMEnvironment.jl:158
[ Info: Searching for system LLVM installation...
[ Info: Using system LLVM toolchain at: /usr
   Root: /usr
   Source: system
   Version: 21.1.6 (LLVM 21.1.6)
   Tools: Auto-discovering...
   Tools: Discovered 48 tools (cached for next run)
   Libraries: 2723 discovered
 LLVM Toolchain initialized
   ✓ Configured 3 include directories
   ✓ Configured 1 source files

 Discovery complete!
 Configuration: ./replibuild.toml

 Running build pipeline...
======================================================================
══════════════════════════════════════════════════════════════════════
 RepliBuild - Compile C++
══════════════════════════════════════════════════════════════════════
======================================================================
RepliBuild Compiler
======================================================================
Project: project
Root:    ~/RepliBuild.jl/test/test_project/
======================================================================

Source files: 1
Compiler flags: -std=c++17 -fPIC -O2
Include dirs: 3

Compiling to LLVM IR...
Compiling 1 files...
Linking and optimizing IR...
Linked 1 IR files → project_linked.ll
Optimizing (O2)...
Optimized
Creating shared library...
Created: libproject.so (0.05 MB)
 Extracting compilation metadata...
   Found 49 exported symbols
Parsing DWARF debug info...
Types collected: 17 base, 20 pointer, 15 struct, 0 class
   Advanced types: 0 enum, 2 array, 4 function_pointer
   Struct/class members: 32, Enum enumerators: 0
    Extracted 262 return types from DWARF
    Extracted 11 struct/class definitions with members
    Saved metadata: ~/RepliBuild.jl/test/test_project/julia/compilation_metadata.json

======================================================================
Build successful (2.01 seconds)
Binary: ~/RepliBuild.jl/test/test_project/julia/libproject.so
Metadata: ~/RepliBuild.jl/test/test_project/julia/compilation_metadata.json
======================================================================

✓ Library: ~/RepliBuild.jl/test/test_project/julia/libproject.so
✓ Metadata saved

Next: RepliBuild.wrap("~/RepliBuild.jl/test/test_project/replibuild.toml") to generate Julia bindings
══════════════════════════════════════════════════════════════════════

 Running wrap pipeline...
======================================================================
══════════════════════════════════════════════════════════════════════
 RepliBuild - Generate Julia Wrappers
══════════════════════════════════════════════════════════════════════
 Library: libproject.so

 RepliBuild Wrapper Generator
======================================================================
   Library: libproject.so
   Tier: Introspective (Metadata-rich)

  Generating Tier 3 (Introspective) wrapper...
   Method: Compilation metadata + Clang.jl verification
   Type safety:  Perfect (from compilation)

   Loading compilation metadata...
  ✓ Found 49 functions with type information

   Generated: ~/RepliBuild.jl/test/test_project/julia/Project.jl
   Functions wrapped: 49
   Type accuracy: ~95% (from compilation metadata)


✓ Wrapper: ~/RepliBuild.jl/test/test_project/julia/Project.jl

Usage:
  include("/home/grim/RepliBuild.jl/test/test_project/julia/Project.jl")
  using .Project
══════════════════════════════════════════════════════════════════════

======================================================================
 Full pipeline complete!
 - Config:  ./replibuild.toml
 - Library: ~/RepliBuild.jl/test/test_project/julia/libproject.so
 - Wrapper: ~/RepliBuild.jl/test/test_project/julia/Project.jl
======================================================================
"./replibuild.toml"

```


