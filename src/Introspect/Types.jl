#!/usr/bin/env julia
# Types.jl - Structured data types for introspection results
# All types include custom Base.show() methods for pretty printing

# ============================================================================
# BINARY INTROSPECTION TYPES
# ============================================================================

"""
    SymbolInfo

Information about a single symbol from a binary.

# Fields
- `name::String` - Mangled symbol name
- `demangled::String` - Human-readable demangled name
- `address::String` - Memory address (hex string)
- `type::Symbol` - Symbol type (:function, :data, :weak, :other)
- `size::Int` - Symbol size in bytes (0 if unknown)
"""
struct SymbolInfo
    name::String
    demangled::String
    address::String
    type::Symbol
    size::Int
end

function Base.show(io::IO, sym::SymbolInfo)
    println(io, "Symbol: $(sym.demangled)")
    println(io, "  Mangled: $(sym.name)")
    println(io, "  Address: $(sym.address)")
    println(io, "  Type: $(sym.type)")
    print(io, "  Size: $(sym.size) bytes")
end

"""
    MemberInfo

Information about a struct/class member field.

# Fields
- `name::String` - Field name
- `c_type::String` - C/C++ type
- `julia_type::String` - Mapped Julia type
- `offset::Int` - Byte offset from struct start
- `size::Int` - Field size in bytes
"""
struct MemberInfo
    name::String
    c_type::String
    julia_type::String
    offset::Int
    size::Int
end

function Base.show(io::IO, member::MemberInfo)
    print(io, "$(member.name): $(member.c_type) @ +$(member.offset) ($(member.size) bytes)")
end

"""
    StructInfo

Complete struct/class layout information from DWARF.

# Fields
- `name::String` - Struct/class name
- `size::Int` - Total size in bytes
- `alignment::Int` - Alignment requirement
- `members::Vector{MemberInfo}` - Field layout
- `base_classes::Vector{String}` - Inherited classes
- `is_polymorphic::Bool` - Has virtual functions
- `vtable_offset::Union{Int,Nothing}` - Vtable pointer offset (if polymorphic)
"""
struct StructInfo
    name::String
    size::Int
    alignment::Int
    members::Vector{MemberInfo}
    base_classes::Vector{String}
    is_polymorphic::Bool
    vtable_offset::Union{Int,Nothing}
end

function Base.show(io::IO, struct_info::StructInfo)
    println(io, "Struct: $(struct_info.name)")
    println(io, "  Size: $(struct_info.size) bytes, Alignment: $(struct_info.alignment)")
    if !isempty(struct_info.base_classes)
        println(io, "  Inherits: $(join(struct_info.base_classes, ", "))")
    end
    if struct_info.is_polymorphic
        println(io, "  Polymorphic: yes (vtable @ +$(struct_info.vtable_offset))")
    end
    if !isempty(struct_info.members)
        println(io, "  Members:")
        for member in struct_info.members
            println(io, "    $(member)")
        end
    end
end

"""
    FunctionInfo

Function signature information from DWARF and symbols.

# Fields
- `name::String` - Function name
- `mangled::String` - Mangled name
- `demangled::String` - Demangled name
- `return_type::String` - Return type
- `parameters::Vector{Tuple{String,String}}` - (name, type) pairs
- `is_method::Bool` - Is this a class method
- `class::Union{String,Nothing}` - Parent class (if method)
"""
struct FunctionInfo
    name::String
    mangled::String
    demangled::String
    return_type::String
    parameters::Vector{Tuple{String,String}}
    is_method::Bool
    class::Union{String,Nothing}
end

function Base.show(io::IO, func::FunctionInfo)
    println(io, "Function: $(func.demangled)")
    println(io, "  Mangled: $(func.mangled)")
    print(io, "  Signature: $(func.return_type) $(func.name)(")
    print(io, join(["$(p[2]) $(p[1])" for p in func.parameters], ", "))
    println(io, ")")
    if func.is_method && func.class !== nothing
        println(io, "  Class: $(func.class)")
    end
end

"""
    DWARFInfo

Complete DWARF debug information extracted from a binary.

# Fields
- `binary_path::String` - Path to binary
- `functions::Dict{String,FunctionInfo}` - Functions by name
- `structs::Dict{String,StructInfo}` - Structs/classes by name
- `enums::Dict{String,Vector{Tuple{String,Int}}}` - Enums with (name, value) pairs
"""
struct DWARFInfo
    binary_path::String
    functions::Dict{String,FunctionInfo}
    structs::Dict{String,StructInfo}
    enums::Dict{String,Vector{Tuple{String,Int}}}
end

function Base.show(io::IO, dwarf::DWARFInfo)
    println(io, "DWARF Info: $(basename(dwarf.binary_path))")
    println(io, "  Functions: $(length(dwarf.functions))")
    println(io, "  Structs: $(length(dwarf.structs))")
    print(io, "  Enums: $(length(dwarf.enums))")
end

"""
    HeaderInfo

Binary header and section information.

# Fields
- `binary_path::String` - Path to binary
- `file_type::String` - ELF, Mach-O, PE, etc.
- `architecture::String` - Target architecture
- `sections::Vector{Tuple{String,Int,Int}}` - (name, offset, size)
- `entry_point::String` - Entry point address
"""
struct HeaderInfo
    binary_path::String
    file_type::String
    architecture::String
    sections::Vector{Tuple{String,Int,Int}}
    entry_point::String
end

function Base.show(io::IO, header::HeaderInfo)
    println(io, "Binary: $(basename(header.binary_path))")
    println(io, "  Type: $(header.file_type)")
    println(io, "  Arch: $(header.architecture)")
    println(io, "  Entry: $(header.entry_point)")
    println(io, "  Sections: $(length(header.sections))")
end

# ============================================================================
# JULIA INTROSPECTION TYPES
# ============================================================================

"""
    CodeLoweredInfo

Results from @code_lowered - Julia AST before type inference.

# Fields
- `function_name::String` - Function name
- `types::Tuple` - Argument types
- `code::Vector{Any}` - Lowered code IR
- `slot_names::Vector{Symbol}` - Variable names
"""
struct CodeLoweredInfo
    function_name::String
    types::Tuple
    code::Vector{Any}
    slot_names::Vector{Symbol}
end

function Base.show(io::IO, info::CodeLoweredInfo)
    println(io, "Code Lowered: $(info.function_name)$(info.types)")
    println(io, "  Slots: $(join(info.slot_names, ", "))")
    println(io, "  Instructions: $(length(info.code))")
end

"""
    CodeTypedInfo

Results from @code_typed - Type-inferred Julia IR.

# Fields
- `function_name::String` - Function name
- `types::Tuple` - Argument types
- `code::Any` - Typed code IR
- `return_type::Type` - Inferred return type
- `optimized::Bool` - Was optimization applied
"""
struct CodeTypedInfo
    function_name::String
    types::Tuple
    code::Any
    return_type::Type
    optimized::Bool
end

function Base.show(io::IO, info::CodeTypedInfo)
    println(io, "Code Typed: $(info.function_name)$(info.types)")
    println(io, "  Return Type: $(info.return_type)")
    print(io, "  Optimized: $(info.optimized)")
end

"""
    LLVMIRInfo

Parsed LLVM IR from @code_llvm or llvm-dis.

# Fields
- `function_name::String` - Function name
- `types::Union{Tuple,Nothing}` - Argument types (if from Julia)
- `ir::String` - Raw LLVM IR
- `optimized::Bool` - Was optimization applied
- `instruction_count::Int` - Number of instructions
"""
struct LLVMIRInfo
    function_name::String
    types::Union{Tuple,Nothing}
    ir::String
    optimized::Bool
    instruction_count::Int
end

function Base.show(io::IO, info::LLVMIRInfo)
    if info.types !== nothing
        println(io, "LLVM IR: $(info.function_name)$(info.types)")
    else
        println(io, "LLVM IR: $(info.function_name)")
    end
    println(io, "  Optimized: $(info.optimized)")
    println(io, "  Instructions: $(info.instruction_count)")
    println(io, "\n$(info.ir)")
end

"""
    AssemblyInfo

Native assembly from @code_native.

# Fields
- `function_name::String` - Function name
- `types::Tuple` - Argument types
- `assembly::String` - Raw assembly code
- `syntax::Symbol` - :att or :intel
- `instruction_count::Int` - Number of instructions
"""
struct AssemblyInfo
    function_name::String
    types::Tuple
    assembly::String
    syntax::Symbol
    instruction_count::Int
end

function Base.show(io::IO, info::AssemblyInfo)
    println(io, "Assembly ($(info.syntax)): $(info.function_name)$(info.types)")
    println(io, "  Instructions: $(info.instruction_count)")
    println(io, "\n$(info.assembly)")
end

"""
    TypeInstability

Information about a type-unstable variable.

# Fields
- `variable::String` - Variable name
- `inferred_type::Type` - Inferred type
- `expected_type::Union{Type,Nothing}` - Expected type (if known)
- `line::Union{Int,Nothing}` - Line number
"""
struct TypeInstability
    variable::String
    inferred_type::Type
    expected_type::Union{Type,Nothing}
    line::Union{Int,Nothing}
end

"""
    TypeStabilityAnalysis

Type stability analysis from @code_warntype.

# Fields
- `function_name::String` - Function name
- `types::Tuple` - Argument types
- `is_stable::Bool` - Is fully type-stable
- `unstable_variables::Vector{TypeInstability}` - Unstable vars
- `warnings::Vector{String}` - Warning messages
"""
struct TypeStabilityAnalysis
    function_name::String
    types::Tuple
    is_stable::Bool
    unstable_variables::Vector{TypeInstability}
    warnings::Vector{String}
end

function Base.show(io::IO, analysis::TypeStabilityAnalysis)
    println(io, "Type Stability: $(analysis.function_name)$(analysis.types)")
    if analysis.is_stable
        println(io, "  ✓ Type Stable")
    else
        println(io, "  ⚠ Type Unstable")
        for var in analysis.unstable_variables
            println(io, "    $(var.variable): $(var.inferred_type)")
        end
    end
    if !isempty(analysis.warnings)
        println(io, "  Warnings:")
        for warning in analysis.warnings
            println(io, "    $(warning)")
        end
    end
end

"""
    VectorizedLoop

Information about a vectorized loop.

# Fields
- `loop_id::Int` - Loop identifier
- `vector_width::Int` - SIMD width
- `instruction_count::Int` - Vectorized instructions
"""
struct VectorizedLoop
    loop_id::Int
    vector_width::Int
    instruction_count::Int
end

"""
    SIMDAnalysis

SIMD vectorization analysis from LLVM IR.

# Fields
- `function_name::String` - Function name
- `types::Tuple` - Argument types
- `vectorized_loops::Vector{VectorizedLoop}` - Vectorized loops
- `vector_instructions::Int` - Total vector instructions
- `scalar_instructions::Int` - Total scalar instructions
"""
struct SIMDAnalysis
    function_name::String
    types::Tuple
    vectorized_loops::Vector{VectorizedLoop}
    vector_instructions::Int
    scalar_instructions::Int
end

function Base.show(io::IO, analysis::SIMDAnalysis)
    println(io, "SIMD Analysis: $(analysis.function_name)$(analysis.types)")
    println(io, "  Vectorized Loops: $(length(analysis.vectorized_loops))")
    println(io, "  Vector Instructions: $(analysis.vector_instructions)")
    print(io, "  Scalar Instructions: $(analysis.scalar_instructions)")
end

"""
    AllocationInfo

Information about a single allocation site.

# Fields
- `type::String` - Allocated type
- `size::Int` - Allocation size (bytes)
- `line::Union{Int,Nothing}` - Line number
"""
struct AllocationInfo
    type::String
    size::Int
    line::Union{Int,Nothing}
end

"""
    AllocationAnalysis

Memory allocation analysis from LLVM IR.

# Fields
- `function_name::String` - Function name
- `types::Tuple` - Argument types
- `allocations::Vector{AllocationInfo}` - Allocation sites
- `total_bytes::Int` - Total allocated bytes
- `escapes::Bool` - Do allocations escape
"""
struct AllocationAnalysis
    function_name::String
    types::Tuple
    allocations::Vector{AllocationInfo}
    total_bytes::Int
    escapes::Bool
end

function Base.show(io::IO, analysis::AllocationAnalysis)
    println(io, "Allocation Analysis: $(analysis.function_name)$(analysis.types)")
    println(io, "  Allocations: $(length(analysis.allocations))")
    println(io, "  Total: $(analysis.total_bytes) bytes")
    print(io, "  Escapes: $(analysis.escapes)")
end

"""
    CompilationPipelineResult

Complete compilation pipeline from lowered to native code.

# Fields
- `function_name::String` - Function name
- `types::Tuple` - Argument types
- `lowered::CodeLoweredInfo` - Lowered IR
- `typed::CodeTypedInfo` - Typed IR
- `llvm_ir::LLVMIRInfo` - LLVM IR
- `native::AssemblyInfo` - Native assembly
"""
struct CompilationPipelineResult
    function_name::String
    types::Tuple
    lowered::CodeLoweredInfo
    typed::CodeTypedInfo
    llvm_ir::LLVMIRInfo
    native::AssemblyInfo
end

function Base.show(io::IO, pipeline::CompilationPipelineResult)
    println(io, "Compilation Pipeline: $(pipeline.function_name)$(pipeline.types)")
    println(io, "  Lowered: $(length(pipeline.lowered.code)) instructions")
    println(io, "  LLVM IR: $(pipeline.llvm_ir.instruction_count) instructions")
    print(io, "  Native: $(pipeline.native.instruction_count) instructions")
end

# ============================================================================
# LLVM TOOLING TYPES
# ============================================================================

"""
    OptimizationMetrics

Metrics from LLVM optimization passes.

# Fields
- `instructions_before::Int` - Instructions before optimization
- `instructions_after::Int` - Instructions after optimization
- `reduction_percentage::Float64` - Percentage reduction
- `passes_applied::Vector{String}` - Passes that ran
"""
struct OptimizationMetrics
    instructions_before::Int
    instructions_after::Int
    reduction_percentage::Float64
    passes_applied::Vector{String}
end

"""
    OptimizationResult

Result of LLVM optimization analysis.

# Fields
- `ir_path::String` - Path to IR file
- `opt_level::String` - Optimization level
- `metrics::OptimizationMetrics` - Optimization metrics
- `optimized_ir::String` - Optimized IR
"""
struct OptimizationResult
    ir_path::String
    opt_level::String
    metrics::OptimizationMetrics
    optimized_ir::String
end

function Base.show(io::IO, result::OptimizationResult)
    println(io, "Optimization Result: $(basename(result.ir_path)) -O$(result.opt_level)")
    println(io, "  Before: $(result.metrics.instructions_before) instructions")
    println(io, "  After: $(result.metrics.instructions_after) instructions")
    print(io, "  Reduction: $(round(result.metrics.reduction_percentage, digits=2))%")
end

# ============================================================================
# BENCHMARKING TYPES
# ============================================================================

"""
    BenchmarkResult

Standalone benchmark result with timing and allocation metrics.

# Fields
- `function_name::String` - Function name
- `samples::Int` - Number of samples
- `median_time::Float64` - Median time (nanoseconds)
- `mean_time::Float64` - Mean time (nanoseconds)
- `std_time::Float64` - Standard deviation (nanoseconds)
- `min_time::Float64` - Minimum time (nanoseconds)
- `max_time::Float64` - Maximum time (nanoseconds)
- `allocations::Int` - Number of allocations
- `memory::Int` - Total memory allocated (bytes)
- `gc_time::Float64` - Time spent in GC (nanoseconds)
- `timestamp::DateTime` - When benchmark was run
"""
struct BenchmarkResult
    function_name::String
    samples::Int
    median_time::Float64
    mean_time::Float64
    std_time::Float64
    min_time::Float64
    max_time::Float64
    allocations::Int
    memory::Int
    gc_time::Float64
    timestamp::DateTime
end

function Base.show(io::IO, result::BenchmarkResult)
    println(io, "Benchmark: $(result.function_name)")
    println(io, "  Samples: $(result.samples)")
    println(io, "  Median: $(format_time(result.median_time))")
    println(io, "  Mean: $(format_time(result.mean_time)) ± $(format_time(result.std_time))")
    println(io, "  Range: [$(format_time(result.min_time)), $(format_time(result.max_time))]")
    println(io, "  Allocations: $(result.allocations) ($(format_bytes(result.memory)))")
    print(io, "  GC Time: $(format_time(result.gc_time))")
end

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

"""Format time in appropriate units"""
function format_time(ns::Float64)
    if ns < 1_000
        return "$(round(ns, digits=2)) ns"
    elseif ns < 1_000_000
        return "$(round(ns/1_000, digits=2)) μs"
    elseif ns < 1_000_000_000
        return "$(round(ns/1_000_000, digits=2)) ms"
    else
        return "$(round(ns/1_000_000_000, digits=2)) s"
    end
end

"""Format bytes in appropriate units"""
function format_bytes(bytes::Int)
    if bytes < 1024
        return "$(bytes) bytes"
    elseif bytes < 1024^2
        return "$(round(bytes/1024, digits=2)) KiB"
    elseif bytes < 1024^3
        return "$(round(bytes/1024^2, digits=2)) MiB"
    else
        return "$(round(bytes/1024^3, digits=2)) GiB"
    end
end
