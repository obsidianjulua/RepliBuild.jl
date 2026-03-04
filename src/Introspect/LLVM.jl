#!/usr/bin/env julia
# LLVM.jl - LLVM tooling wrappers (llvm-dis, opt, llc)
# Provides structured access to LLVM optimization and compilation tools

# ============================================================================
# LLVM IR DISASSEMBLY
# ============================================================================

"""
    llvm_ir(bitcode_path::String)

Disassemble LLVM bitcode to readable IR using llvm-dis.

# Arguments
- `bitcode_path::String` - Path to .bc bitcode file

# Returns
LLVMIRInfo

# Examples
```julia
# Disassemble bitcode
ir = llvm_ir("code.bc")
println(ir.ir)
println("Instructions: \$(ir.instruction_count)")
```
"""
function llvm_ir(bitcode_path::String)
    # Validate file exists
    if !isfile(bitcode_path)
        error("Bitcode file not found: $bitcode_path")
    end

    # Get llvm-dis tool
    llvm_dis = LLVMEnvironment.get_tool("llvm-dis")
    if isempty(llvm_dis)
        error("llvm-dis not found. Install LLVM toolchain.")
    end

    # Disassemble to stdout
    (ir_output, exitcode) = LLVMEnvironment.with_llvm_env() do
        BuildBridge.execute(llvm_dis, [bitcode_path, "-o", "-"])
    end

    if exitcode != 0
        error("llvm-dis failed: $ir_output")
    end

    # Count instructions
    instruction_count = count(l -> occursin(r"^\s+%|^\s+[a-z]+", l), split(ir_output, '\n'))

    # Extract function name from IR
    func_match = match(r"define .* @([A-Za-z0-9_\.]+)\(", ir_output)
    func_name = if func_match !== nothing
        func_match.captures[1]
    else
        basename(bitcode_path)
    end

    return LLVMIRInfo(
        func_name,
        nothing,  # No types for raw IR
        ir_output,
        false,    # Bitcode is typically unoptimized
        instruction_count
    )
end

# ============================================================================
# LLVM OPTIMIZATION
# ============================================================================

"""
    optimize_ir(ir_path::String, opt_level::String; passes=nothing)

Optimize LLVM IR using opt tool.

# Arguments
- `ir_path::String` - Path to .ll IR file
- `opt_level::String` - Optimization level ("0", "1", "2", "3")
- `passes` - Optional specific passes to run (Vector{String})

# Returns
OptimizationResult

# Examples
```julia
# Optimize at O2
result = optimize_ir("code.ll", "2")
println("Reduction: \$(result.metrics.reduction_percentage)%")

# Run specific passes
result = optimize_ir("code.ll", "2", passes=["mem2reg", "inline"])
```
"""
function optimize_ir(ir_path::String, opt_level::String; passes=nothing)
    # Validate file exists
    if !isfile(ir_path)
        error("IR file not found: $ir_path")
    end

    # Get opt tool
    opt_tool = LLVMEnvironment.get_tool("opt")
    if isempty(opt_tool)
        error("opt not found. Install LLVM toolchain.")
    end

    # Read original IR
    original_ir = read(ir_path, String)
    instructions_before = count(l -> occursin(r"^\s+%|^\s+[a-z]+", l), split(original_ir, '\n'))

    # Build opt arguments
    args = if passes !== nothing
        # Specific passes
        ["-passes=$(join(passes, ","))", ir_path, "-S", "-o", "-"]
    else
        # Optimization level
        ["-O$opt_level", ir_path, "-S", "-o", "-"]
    end

    # Run opt
    (optimized_ir, exitcode) = LLVMEnvironment.with_llvm_env() do
        BuildBridge.execute(opt_tool, args)
    end

    if exitcode != 0
        error("opt failed: $optimized_ir")
    end

    # Count instructions after
    instructions_after = count(l -> occursin(r"^\s+%|^\s+[a-z]+", l), split(optimized_ir, '\n'))

    # Calculate reduction
    reduction = if instructions_before > 0
        ((instructions_before - instructions_after) / instructions_before) * 100.0
    else
        0.0
    end

    # Determine which passes were applied
    passes_applied = if passes !== nothing
        passes
    else
        ["O$opt_level"]
    end

    metrics = OptimizationMetrics(
        instructions_before,
        instructions_after,
        reduction,
        passes_applied
    )

    return OptimizationResult(
        ir_path,
        opt_level,
        metrics,
        optimized_ir
    )
end

"""
    compare_optimization(ir_path::String, levels::Vector{String})

Compare multiple optimization levels.

# Arguments
- `ir_path::String` - Path to .ll IR file
- `levels::Vector{String}` - Optimization levels to compare (e.g., ["0", "2", "3"])

# Returns
Dict{String, OptimizationResult}

# Examples
```julia
# Compare O0, O2, O3
results = compare_optimization("code.ll", ["0", "2", "3"])
for (level, result) in results
    println("O\$level: \$(result.metrics.instructions_after) instructions")
end
```
"""
function compare_optimization(ir_path::String, levels::Vector{String})
    results = Dict{String, OptimizationResult}()

    for level in levels
        try
            result = optimize_ir(ir_path, level)
            results[level] = result
        catch e
            @warn "Optimization level O$level failed: $e"
        end
    end

    return results
end

# ============================================================================
# PASS ANALYSIS
# ============================================================================

"""
    run_passes(ir_path::String, passes::Vector{String})

Run specific LLVM passes and return optimized IR.

# Arguments
- `ir_path::String` - Path to .ll IR file
- `passes::Vector{String}` - List of pass names

# Returns
OptimizationResult

# Examples
```julia
# Run mem2reg and inline passes
result = run_passes("code.ll", ["mem2reg", "inline"])

# Run loop optimization passes
result = run_passes("code.ll", ["loop-simplify", "loop-unroll"])
```
"""
function run_passes(ir_path::String, passes::Vector{String})
    return optimize_ir(ir_path, "0", passes=passes)
end

# ============================================================================
# CODE GENERATION
# ============================================================================

"""
    compile_to_asm(ir_path::String; opt_level="2", target=nothing)

Compile LLVM IR to native assembly using llc.

# Arguments
- `ir_path::String` - Path to .ll IR file
- `opt_level::String` - Optimization level (default: "2")
- `target` - Target architecture (default: native)

# Returns
String - Generated assembly code

# Examples
```julia
# Compile to native assembly
asm = compile_to_asm("code.ll")

# Compile with O3
asm = compile_to_asm("code.ll", opt_level="3")

# Compile for specific target
asm = compile_to_asm("code.ll", target="x86_64-unknown-linux-gnu")

# Save to file
open("code.s", "w") do io
    write(io, asm)
end
```
"""
function compile_to_asm(ir_path::String; opt_level::String="2", target=nothing)
    # Validate file exists
    if !isfile(ir_path)
        error("IR file not found: $ir_path")
    end

    # Get llc tool
    llc_tool = LLVMEnvironment.get_tool("llc")
    if isempty(llc_tool)
        error("llc not found. Install LLVM toolchain.")
    end

    # Build llc arguments
    args = ["-O$opt_level", ir_path, "-o", "-"]

    if target !== nothing
        push!(args, "-mtriple=$target")
    end

    # Run llc
    (assembly, exitcode) = LLVMEnvironment.with_llvm_env() do
        BuildBridge.execute(llc_tool, args)
    end

    if exitcode != 0
        error("llc failed: $assembly")
    end

    return assembly
end

# ============================================================================
# IR ANALYSIS UTILITIES
# ============================================================================

"""
    analyze_ir_structure(ir::String)

Analyze LLVM IR structure and extract statistics.

# Arguments
- `ir::String` - LLVM IR text

# Returns
Dict with analysis results

# Examples
```julia
ir_info = llvm_ir("code.bc")
analysis = analyze_ir_structure(ir_info.ir)
println("Functions: \$(analysis[:function_count])")
println("Basic blocks: \$(analysis[:basic_block_count])")
```
"""
function analyze_ir_structure(ir::String)
    lines = split(ir, '\n')

    # Count various IR constructs
    function_count = count(l -> occursin(r"^define", l), lines)
    basic_block_count = count(l -> occursin(r"^[a-zA-Z0-9_]+:", l), lines)
    call_count = count(l -> occursin(r"\s+call\s+", l), lines)
    load_count = count(l -> occursin(r"\s+load\s+", l), lines)
    store_count = count(l -> occursin(r"\s+store\s+", l), lines)
    alloca_count = count(l -> occursin(r"\s+alloca\s+", l), lines)
    phi_count = count(l -> occursin(r"\s+phi\s+", l), lines)
    getelementptr_count = count(l -> occursin(r"\s+getelementptr\s+", l), lines)

    return Dict(
        :function_count => function_count,
        :basic_block_count => basic_block_count,
        :call_count => call_count,
        :load_count => load_count,
        :store_count => store_count,
        :alloca_count => alloca_count,
        :phi_count => phi_count,
        :getelementptr_count => getelementptr_count
    )
end

"""
    extract_function_names(ir::String)

Extract all function names defined in LLVM IR.

# Arguments
- `ir::String` - LLVM IR text

# Returns
Vector{String} - Function names

# Examples
```julia
ir_info = llvm_ir("code.bc")
funcs = extract_function_names(ir_info.ir)
println("Functions: \$(join(funcs, ", "))")
```
"""
function extract_function_names(ir::String)
    function_names = String[]

    for line in split(ir, '\n')
        m = match(r"define .* @([A-Za-z0-9_\.]+)\(", line)
        if m !== nothing
            push!(function_names, m.captures[1])
        end
    end

    return function_names
end

"""
    compare_ir_files(ir1_path::String, ir2_path::String)

Compare two LLVM IR files structurally.

# Arguments
- `ir1_path::String` - First IR file
- `ir2_path::String` - Second IR file

# Returns
Dict with comparison results

# Examples
```julia
# Compare unoptimized vs optimized
comparison = compare_ir_files("code_O0.ll", "code_O3.ll")
println("Instruction reduction: \$(comparison[:instruction_reduction])%")
```
"""
function compare_ir_files(ir1_path::String, ir2_path::String)
    if !isfile(ir1_path) || !isfile(ir2_path)
        error("One or both IR files not found")
    end

    ir1 = read(ir1_path, String)
    ir2 = read(ir2_path, String)

    analysis1 = analyze_ir_structure(ir1)
    analysis2 = analyze_ir_structure(ir2)

    # Calculate differences
    instruction_count1 = count(l -> occursin(r"^\s+%|^\s+[a-z]+", l), split(ir1, '\n'))
    instruction_count2 = count(l -> occursin(r"^\s+%|^\s+[a-z]+", l), split(ir2, '\n'))

    reduction = if instruction_count1 > 0
        ((instruction_count1 - instruction_count2) / instruction_count1) * 100.0
    else
        0.0
    end

    return Dict(
        :file1 => ir1_path,
        :file2 => ir2_path,
        :analysis1 => analysis1,
        :analysis2 => analysis2,
        :instruction_count1 => instruction_count1,
        :instruction_count2 => instruction_count2,
        :instruction_reduction => reduction
    )
end
