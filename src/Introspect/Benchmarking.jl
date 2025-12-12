#!/usr/bin/env julia
# Benchmarking.jl - Standalone benchmarking tools
# Provides performance measurement without auto-generation

using Statistics
using Dates

# ============================================================================
# STANDALONE BENCHMARKING
# ============================================================================

"""
    benchmark(func, args...; samples=1000, warmup=10)

Benchmark a function with given arguments.

Performs warmup runs, then measures timing, allocations, and GC time
across multiple samples.

# Arguments
- `func` - Function to benchmark
- `args...` - Arguments to pass to function
- `samples::Int` - Number of samples to collect (default: 1000)
- `warmup::Int` - Number of warmup runs (default: 10)

# Returns
BenchmarkResult

# Examples
```julia
# Benchmark a function
result = benchmark(sort, rand(10000))
println("Median: \$(format_time(result.median_time))")

# Benchmark with more samples
result = benchmark(my_func, data, samples=5000, warmup=50)

# Export results
export_json(result, "benchmark.json")
```
"""
function benchmark(func, args...; samples::Int=1000, warmup::Int=10)
    # Validate inputs
    if samples < 1
        error("samples must be >= 1")
    end

    if warmup < 0
        error("warmup must be >= 0")
    end

    # Warmup runs (compile and optimize)
    for _ in 1:warmup
        try
            func(args...)
        catch
            # Ignore errors during warmup
        end
    end

    # Collect samples
    times = Vector{Float64}(undef, samples)
    allocations = Vector{Int}(undef, samples)
    memory = Vector{Int}(undef, samples)
    gc_times = Vector{Float64}(undef, samples)

    for i in 1:samples
        # Clear compilation effects
        GC.gc(false)

        # Measure
        stats = @timed func(args...)

        times[i] = stats.time * 1e9  # Convert to nanoseconds
        allocations[i] = stats.gcstats.allocd > 0 ? 1 : 0  # Count allocations
        memory[i] = stats.gcstats.allocd
        gc_times[i] = stats.gcstats.total_time * 1e9  # Convert to nanoseconds
    end

    # Calculate statistics
    median_time = median(times)
    mean_time = mean(times)
    std_time = std(times)
    min_time = minimum(times)
    max_time = maximum(times)

    total_allocations = sum(allocations)
    total_memory = sum(memory)
    total_gc_time = sum(gc_times)

    # Extract function name
    func_name = string(func)

    return BenchmarkResult(
        func_name,
        samples,
        median_time,
        mean_time,
        std_time,
        min_time,
        max_time,
        total_allocations,
        total_memory,
        total_gc_time,
        now()
    )
end

# ============================================================================
# BENCHMARK SUITE
# ============================================================================

"""
    benchmark_suite(funcs::Dict{String,Function}; samples=1000, warmup=10)

Benchmark multiple functions and return results as a dictionary.

# Arguments
- `funcs::Dict{String,Function}` - Dictionary mapping names to functions
- `samples::Int` - Number of samples per function (default: 1000)
- `warmup::Int` - Number of warmup runs per function (default: 10)

# Returns
Dict{String, BenchmarkResult}

# Examples
```julia
# Define benchmark suite
funcs = Dict(
    "sort_builtin" => () -> sort(rand(1000)),
    "sort_custom" => () -> my_sort(rand(1000))
)

# Run suite
results = benchmark_suite(funcs)

# Compare results
for (name, result) in results
    println("\$name: \$(format_time(result.median_time))")
end

# Export as dataset
export_dataset(results, "benchmark_suite/")
```
"""
function benchmark_suite(funcs::Dict{String,Function}; samples::Int=1000, warmup::Int=10)
    results = Dict{String, BenchmarkResult}()

    println("Running benchmark suite...")
    println("=" ^ 70)

    for (name, func) in funcs
        print("  $name... ")
        try
            result = benchmark(func, samples=samples, warmup=warmup)
            results[name] = result
            println("✓ $(format_time(result.median_time))")
        catch e
            println("✗ Error: $e")
        end
    end

    println("=" ^ 70)
    println("Completed $(length(results))/$(length(funcs)) benchmarks")

    return results
end

# ============================================================================
# ALLOCATION TRACKING
# ============================================================================

"""
    track_allocations(func, args...)

Detailed allocation tracking for a function call.

Uses @timed to track allocations and returns detailed information.

# Arguments
- `func` - Function to track
- `args...` - Arguments to pass to function

# Returns
Dict with allocation details

# Examples
```julia
# Track allocations
alloc_info = track_allocations(my_func, data)
println("Allocated: \$(alloc_info[:total_bytes]) bytes")
println("GC time: \$(alloc_info[:gc_time]) seconds")
```
"""
function track_allocations(func, args...)
    # Force GC before measurement
    GC.gc(true)

    # Measure with full GC stats
    stats = @timed func(args...)

    return Dict(
        :function => string(func),
        :time => stats.time,
        :allocations => stats.gcstats.allocd > 0 ? 1 : 0,
        :total_bytes => stats.gcstats.allocd,
        :gc_time => stats.gcstats.total_time,
        :gc_pause_time => stats.gcstats.pause,
        :result => stats.value
    )
end

# ============================================================================
# PROFILING INTEGRATION
# ============================================================================

"""
    profile(func, args...; seconds=10)

Profile a function using Julia's Profile stdlib.

Runs function repeatedly for specified duration and collects profiling data.

# Arguments
- `func` - Function to profile
- `args...` - Arguments to pass to function
- `seconds::Int` - Duration to profile (default: 10 seconds)

# Returns
Dict with profiling information

# Examples
```julia
# Profile for 10 seconds
prof_data = profile(my_func, data, seconds=10)

# Print profile
@eval Main using Profile
Profile.print()

# Clear profile data
Profile.clear()
```
"""
function profile(func, args...; seconds::Int=10)
    # Import Profile stdlib
    @eval Main using Profile
    Prof = Main.Profile

    # Clear previous profile data
    Prof.clear()

    println("Profiling for $seconds seconds...")

    # Start profiling
    Prof.@profile begin
        start_time = time()
        iterations = 0

        while time() - start_time < seconds
            func(args...)
            iterations += 1
        end
    end

    println("Completed $iterations iterations")

    # Get profile data
    data = Prof.fetch()

    return Dict(
        :function => string(func),
        :iterations => iterations,
        :duration => seconds,
        :samples => length(data),
        :unique_backtraces => length(unique(data))
    )
end

# ============================================================================
# COMPARISON UTILITIES
# ============================================================================

"""
    compare_benchmarks(results::Vector{BenchmarkResult})

Compare multiple benchmark results and show relative performance.

# Arguments
- `results::Vector{BenchmarkResult}` - Benchmark results to compare

# Returns
DataFrame with comparison

# Examples
```julia
# Run benchmarks
r1 = benchmark(func1, args)
r2 = benchmark(func2, args)
r3 = benchmark(func3, args)

# Compare
comparison = compare_benchmarks([r1, r2, r3])
println(comparison)
```
"""
function compare_benchmarks(results::Vector{BenchmarkResult})
    if isempty(results)
        error("No results to compare")
    end

    # Find fastest
    fastest_time = minimum(r -> r.median_time, results)

    # Build comparison data
    names = [r.function_name for r in results]
    medians = [r.median_time for r in results]
    relatives = [r.median_time / fastest_time for r in results]
    allocs = [r.allocations for r in results]
    memory = [r.memory for r in results]

    using DataFrames
    return DataFrame(
        Function = names,
        Median = [format_time(m) for m in medians],
        Relative = [round(r, digits=2) for r in relatives],
        Allocations = allocs,
        Memory = [format_bytes(m) for m in memory]
    )
end

"""
    fastest(results...)

Find the fastest benchmark result.

# Arguments
- `results...` - Benchmark results to compare

# Returns
BenchmarkResult - Fastest result

# Examples
```julia
r1 = benchmark(func1, args)
r2 = benchmark(func2, args)
winner = fastest(r1, r2)
println("Winner: \$(winner.function_name)")
```
"""
function fastest(results::BenchmarkResult...)
    if isempty(results)
        error("No results provided")
    end

    return argmin(r -> r.median_time, collect(results))
end

"""
    slowest(results...)

Find the slowest benchmark result.

# Arguments
- `results...` - Benchmark results to compare

# Returns
BenchmarkResult - Slowest result
"""
function slowest(results::BenchmarkResult...)
    if isempty(results)
        error("No results provided")
    end

    return argmax(r -> r.median_time, collect(results))
end

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

"""
    is_significant(r1::BenchmarkResult, r2::BenchmarkResult; threshold=0.05)

Test if difference between two benchmarks is statistically significant.

Uses a simple threshold-based test. For more rigorous analysis, use
proper statistical testing tools.

# Arguments
- `r1::BenchmarkResult` - First benchmark result
- `r2::BenchmarkResult` - Second benchmark result
- `threshold::Float64` - Significance threshold (default: 5%)

# Returns
Bool - true if difference is significant

# Examples
```julia
r1 = benchmark(func1, args)
r2 = benchmark(func2, args)
if is_significant(r1, r2)
    println("Performance difference is significant!")
end
```
"""
function is_significant(r1::BenchmarkResult, r2::BenchmarkResult; threshold::Float64=0.05)
    # Simple threshold test: is difference > threshold * min(r1, r2)
    min_time = min(r1.median_time, r2.median_time)
    difference = abs(r1.median_time - r2.median_time)

    return (difference / min_time) > threshold
end

"""
    speedup(baseline::BenchmarkResult, optimized::BenchmarkResult)

Calculate speedup from baseline to optimized.

# Arguments
- `baseline::BenchmarkResult` - Baseline benchmark
- `optimized::BenchmarkResult` - Optimized benchmark

# Returns
Float64 - Speedup factor (> 1 means faster, < 1 means slower)

# Examples
```julia
baseline = benchmark(slow_func, args)
optimized = benchmark(fast_func, args)
factor = speedup(baseline, optimized)
println("Speedup: \$(round(factor, digits=2))x")
```
"""
function speedup(baseline::BenchmarkResult, optimized::BenchmarkResult)
    return baseline.median_time / optimized.median_time
end
