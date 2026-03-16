#!/usr/bin/env julia
# Debug: capture the MLIR IR that crashes during AOT thunk emission

using RepliBuild
using JSON

const C_TEST_DIR = @__DIR__
toml = joinpath(C_TEST_DIR, "replibuild.toml")

# Clean
try RepliBuild.clean(toml) catch end
isfile(toml) && rm(toml)
isdir(joinpath(C_TEST_DIR, "build")) && rm(joinpath(C_TEST_DIR, "build"); recursive=true)
isdir(joinpath(C_TEST_DIR, "julia")) && rm(joinpath(C_TEST_DIR, "julia"); recursive=true)
isdir(joinpath(C_TEST_DIR, ".replibuild_cache")) && rm(joinpath(C_TEST_DIR, ".replibuild_cache"); recursive=true)

# Discover + patch
RepliBuild.discover(C_TEST_DIR; force=true)
cfg = read(toml, String)
cfg = replace(cfg, "enable_lto = false" => "enable_lto = true")
# DON'T enable aot_thunks — build manually to avoid the crash
write(toml, cfg)

# Build (no AOT) to get metadata + library
lib_path = RepliBuild.build(toml)
println("Built: $lib_path")

# Now manually run the JLCS IR generation
output_dir = joinpath(C_TEST_DIR, "julia")
metadata_path = joinpath(output_dir, "compilation_metadata.json")
vtable_info = RepliBuild.DWARFParser.parse_vtables(lib_path)
metadata = JSON.parsefile(metadata_path)
ir_source = RepliBuild.JLCSIRGenerator.generate_jlcs_ir(vtable_info, metadata)

println("\n=== Generated MLIR IR ===")
println(ir_source)
println("=== END ===\n")

# Parse
ctx = RepliBuild.MLIRNative.create_context()
mod = RepliBuild.MLIRNative.parse_module(ctx, ir_source)
println("Parse: ", mod != C_NULL ? "OK" : "FAILED")

if mod != C_NULL
    # Clone and lower
    mod_jit = RepliBuild.MLIRNative.clone_module(mod)
    ok = RepliBuild.MLIRNative.lower_to_llvm(mod_jit)
    println("Lower: ", ok ? "OK" : "FAILED")

    if ok
        tmp = joinpath(output_dir, "_debug_thunks.ll")
        println("Attempting emit_llvmir...")
        emit_ok = RepliBuild.MLIRNative.emit_llvmir(mod_jit, tmp)
        println("Emit LLVM IR: ", emit_ok ? "OK" : "FAILED")
        if emit_ok && isfile(tmp)
            lines = readlines(tmp)
            println("\n=== LLVM IR ($(length(lines)) lines, first 80) ===")
            for line in lines[1:min(80, length(lines))]
                println(line)
            end
        end
    end
end

RepliBuild.MLIRNative.destroy_context(ctx)
