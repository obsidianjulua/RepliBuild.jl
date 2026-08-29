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

    # AOT MUST READ THE THUNK MANIFEST, for the same reason the JIT path does
    # (JITManager, "Load thunk manifest").
    #
    # `needed_symbols` is not only dead-thunk elimination — it is what ROUTES a
    # virtual method to the right emitter. A virtual the wrapper binds has to go
    # through the function-thunk pass, because only FunctionGen emits the
    # `_mlir_ciface_<mangled>_thunk` convention `invoke`/`invoke_aot_ptr` looks
    # up; the legacy vmethod-IR pass emits `thunk_<mangled>`, which no wrapper
    # has looked up since 2026-07-17 (see JLCSIRGenerator's comment above
    # `gen_pre`). Passing nothing here sent every virtual down the legacy pass,
    # so AOT built a symbol the wrapper never asks for and omitted the one it
    # does — clipper2's 7 unresolved slots of 156.
    manifest_path = joinpath(output_dir, "thunk_manifest.json")
    needed_symbols = if isfile(manifest_path)
        try
            manifest = JSON.parsefile(manifest_path)
            Set{String}(get(manifest, "function_thunks", String[]))
        catch
            nothing
        end
    else
        nothing
    end

    ir_source = JLCSIRGenerator.generate_jlcs_ir(vtable_info, metadata;
                                                 needed_symbols = needed_symbols)

    ctx = MLIRNative.create_context()
    try
        # debug_base puts the generated MLIR beside the wrapper it describes, so
        # gdb can open it when you break in a thunk — the same reason the JIT path
        # passes it (JITManager.jl). Omitting it here sent every AOT package's
        # `.mlir` to tempdir, which is both un-colocated and cleared on reboot;
        # AOT thunks are the ones that actually ship next to the library, so this
        # path needs it more than the JIT one does.
        mod = MLIRNative.parse_module(ctx, ir_source;
                                      debug_base = MLIRNative.debug_dir_for(library_path))
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
        # `$ORIGIN` FIRST, build directory second.
        #
        # A generated wrapper is portable source: `LIBRARY_PATH` and
        # `THUNKS_LIBRARY_PATH` both resolve sibling-first so the whole set can be
        # vendored into a consumer. The thunks library's own `NEEDED libllamacpp.so`
        # did not — it carried only an absolute RUNPATH into the build tree — so a
        # vendored copy loaded ITS sibling `.so` through the wrapper and the
        # BUILD TREE's `.so` through the loader.
        #
        # Two copies of the same library in one process, each with its own static
        # state, and the C++ runtime tears both down at exit:
        # `double free or corruption (!prev)`, then a hang, because Julia's crash
        # handler calls malloc to symbolize an abort that came from inside malloc.
        # Observed on LlamaChat vendoring llamacpp; the package's own tests never
        # saw it, since there the two paths name the same file.
        #
        # RUNPATH entries are searched in order, so `$ORIGIN` wins when the
        # sibling exists — matching the wrapper — and the absolute path stays as
        # the fallback for a thunks library used where it was built.
        link_args = ["-shared", "-fPIC", "-o", thunks_so, thunks_obj,
                     "-L", lib_dir, "-l:$lib_name",
                     "-Wl,-rpath,\$ORIGIN", "-Wl,-rpath,$lib_dir"]
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
