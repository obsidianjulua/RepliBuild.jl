# Compiler.jl - C++ to LLVM IR compilation and linking orchestration
# Coordinates IRCompiler, DWARFExtractor, and MetadataExtractor submodules

module Compiler

using Dates

# Import from parent RepliBuild module
import ..ConfigurationManager: RepliBuildConfig, get_source_files, get_include_dirs,
                                get_compile_flags, get_build_path, get_output_path,
                                get_library_name, get_module_name, is_parallel_enabled,
                                is_cache_enabled
import ..BuildBridge
import ..LLVMEnvironment

# Include submodules
include("compiler/DWARFExtractor.jl")
include("compiler/IRCompiler.jl")
include("compiler/MetadataExtractor.jl")

# Import submodule functionality
using .DWARFExtractor
using .IRCompiler
using .MetadataExtractor

# Re-export public API
export compile_to_ir, link_optimize_ir, create_library, create_executable, compile_project,
       extract_compilation_metadata, save_compilation_metadata,
       dwarf_type_to_julia, get_type_size, extract_dwarf_return_types

# =============================================================================
# HIGH-LEVEL COMPILATION ORCHESTRATION
# =============================================================================

"""
Complete compilation workflow: discover sources, compile, link, create binary.
This is the main entry point for building a project.

Orchestrates:
- IRCompiler: C++ → LLVM IR compilation and linking
- DWARFExtractor: Debug info parsing for types and structures
- MetadataExtractor: Symbol extraction and metadata persistence
"""
function compile_project(config::RepliBuildConfig)
    println("="^70)
    println("RepliBuild: $(config.project.name)")
    println("="^70)

    start_time = time()

    # Step 1: Discover and compile source files to IR
    cpp_files = get_source_files(config)
    println("Source files: $(length(cpp_files))")

    ir_files = IRCompiler.compile_to_ir(config, cpp_files)

    # Step 2: Link & optimize IR
    output_name = config.project.name
    linked_ir = IRCompiler.link_optimize_ir(config, ir_files, output_name)

    # Step 3: Create binary (library or executable)
    binary_path = if config.binary.type == :executable
        IRCompiler.create_executable(config, linked_ir, config.project.name)
    else
        IRCompiler.create_library(config, linked_ir)
    end

    elapsed = round(time() - start_time, digits=2)

    # Step 4: Extract and save compilation metadata
    # This extracts DWARF info, symbols, function signatures
    metadata_path = MetadataExtractor.save_compilation_metadata(config, cpp_files, binary_path)

    println()
    println("="^70)
    println("Build successful ($elapsed seconds)")
    println("Binary: $binary_path")
    println("Metadata: $metadata_path")
    println("="^70)

    return binary_path
end

end # module Compiler
