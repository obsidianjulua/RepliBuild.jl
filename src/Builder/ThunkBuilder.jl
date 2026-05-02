# ThunkBuilder.jl — Compiles sret and MLIR AOT thunks into companion shared libraries.
# Bridge between Builder (compilation) and IRGen (MLIR), loaded after both + Wrapper.

module ThunkBuilder

using JSON
import ..ConfigurationManager
import ..BuildBridge
import ..Compiler
import ..DWARFParser
import ..JLCSIRGenerator
import ..MLIRNative

export build_aot_thunks

function build_aot_thunks(config, library_path)
    output_dir = ConfigurationManager.get_output_path(config)
    metadata_path = joinpath(output_dir, "compilation_metadata.json")

    if !isfile(metadata_path)
        @warn "Cannot AOT compile thunks: metadata not found."
        return
    end

    println("  aot: Generating MLIR thunks...")
    start_time = time()

    vtable_info = DWARFParser.parse_vtables(library_path)
    metadata = JSON.parsefile(metadata_path)
    ir_source = JLCSIRGenerator.generate_jlcs_ir(vtable_info, metadata)

    ctx = MLIRNative.create_context()
    try
        mod = MLIRNative.parse_module(ctx, ir_source)
        if mod == C_NULL
            error("Failed to parse generated MLIR for AOT.")
        end

        if !MLIRNative.lower_to_llvm(mod)
            error("Failed to lower MLIR to LLVM for AOT.")
        end

        thunks_obj = joinpath(output_dir, "thunks.o")
        if !MLIRNative.emit_object(mod, thunks_obj)
            error("Failed to emit object file for AOT thunks.")
        end

        # Link into a companion shared library
        lib_name = basename(library_path)
        thunks_name = replace(lib_name, ".so" => "_thunks.so", ".dylib" => "_thunks.dylib", ".dll" => "_thunks.dll")
        thunks_so = joinpath(output_dir, thunks_name)

        # Link thunks against the main library so C function symbols resolve
        lib_dir = dirname(abspath(library_path))
        linker = config.wrap.language == :c ? "clang" : "clang++"
        link_args = ["-shared", "-fPIC", "-o", thunks_so, thunks_obj,
                     "-L", lib_dir, "-l:$lib_name", "-Wl,-rpath,$lib_dir"]
        (output, exitcode) = BuildBridge.execute(linker, link_args)
        if exitcode != 0
            error("Failed to link thunks.o: $output")
        end

        # Emit LTO text IR for AOT thunks if LTO is enabled
        if config.link.enable_lto
            thunks_lto_name = replace(lib_name, ".so" => "_thunks_lto.ll", ".dylib" => "_thunks_lto.ll", ".dll" => "_thunks_lto.ll")
            thunks_lto_path = joinpath(output_dir, thunks_lto_name)
            if MLIRNative.emit_llvmir(mod, thunks_lto_path)
                lto_ir_text = read(thunks_lto_path, String)
                lto_ir_text = Compiler.sanitize_ir_for_julia(lto_ir_text)
                write(thunks_lto_path, lto_ir_text)

                # Assemble to bitcode via Julia's libLLVM for version-matched bc
                thunks_bc_path = replace(thunks_lto_path, ".ll" => ".bc")
                Compiler.assemble_bitcode(thunks_lto_path, thunks_bc_path)
            else
                @warn "Failed to emit LLVM IR for AOT thunks LTO."
            end
        end

        elapsed = round(time() - start_time, digits=2)
        size_kb = round(filesize(thunks_so) / 1024, digits=1)
        println("  aot: $thunks_name ($size_kb KB) in $(elapsed)s")

        # Cleanup
        rm(thunks_obj, force=true)
    catch e
        @warn "AOT MLIR compilation failed." exception=e
    finally
        MLIRNative.destroy_context(ctx)
    end
end

end # module ThunkBuilder
