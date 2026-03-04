using Test
using RepliBuild

# Define test cases
# name => directory_name
TEST_CASES = [
    ("Basics (POD, Packing)", "basics_test"),
    ("VTable (Virtual Dispatch)", "vtable_test"),
    # ("StdLib (Containers)", "stdlib_test"), # Uncomment when verification script is ready
    # ("Stress (Complex)", "stress_test")     # Uncomment when ready
]

# Root test directory
TEST_ROOT = @__DIR__

function run_stage(name, func)
    print("  $name... ")
    try
        func()
        println("✓")
        return true
    catch e
        println("✗")
        showerror(stdout, e, catch_backtrace())
        println()
        return false
    end
end

println("="^80)
println("RepliBuild Comprehensive Test Suite")
println("="^80)

global_success = true

for (test_name, dir_name) in TEST_CASES
    println("\nRunning Test Case: $test_name")
    println("-"^40)
    
    test_dir = joinpath(TEST_ROOT, dir_name)
    if !isdir(test_dir)
        println("Skipping $test_name (directory not found: $dir_name)")
        continue
    end

    # Clean
    run_stage("Cleaning", () -> begin
        for d in ["build", "julia", ".replibuild_cache", "replibuild.toml"]
            rm(joinpath(test_dir, d), recursive=true, force=true)
        end
    end)

    # Discover
    local toml_path
    success = run_stage("Discovery", () -> begin
        toml_path = RepliBuild.discover(test_dir, force=true)
    end)
    if !success; global global_success = false; continue; end

    # Build
    success = run_stage("Build", () -> begin
        RepliBuild.build(toml_path)
    end)
    if !success; global global_success = false; continue; end

    # Wrap
    success = run_stage("Wrap", () -> begin
        RepliBuild.wrap(toml_path)
    end)
    if !success; global global_success = false; continue; end

    # Verify Execution
    verify_script = joinpath(test_dir, "verify.jl")
    if isfile(verify_script)
        success = run_stage("Verification (Run Wrapper)", () -> begin
            # Run in a new process to avoid module name conflicts/pollution
            cmd = `julia --project=$(Base.active_project()) $verify_script`
            run(cmd)
        end)
        if !success; global global_success = false; end
    else
        println("  Verification... Skipped (no verify.jl)")
    end
end

println("\n" * "="^80)
if global_success
    println("ALL TESTS PASSED")
    exit(0)
else
    println("SOME TESTS FAILED")
    exit(1)
end
