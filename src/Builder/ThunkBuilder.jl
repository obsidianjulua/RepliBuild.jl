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
import ..Wrapper

export build_c_thunks, build_aot_thunks

"""
    build_c_thunks(config, library_path)

Generate and compile C sret thunk wrappers for functions with packed struct
or union return types.  Uses the same Clang that compiled the library, so
the resulting IR is version-compatible with Julia's internal LLVM — no
sanitizing or MLIR involvement needed.
"""
function build_c_thunks(config, library_path)
    output_dir = ConfigurationManager.get_output_path(config)
    metadata_path = joinpath(output_dir, "compilation_metadata.json")

    if !isfile(metadata_path)
        @warn "Cannot build C thunks: metadata not found."
        return
    end

    println("  c-thunks: Scanning for unsafe returns...")
    start_time = time()

    metadata = JSON.parsefile(metadata_path)
    functions = get(metadata, "functions", [])
    dwarf_structs = get(metadata, "struct_definitions", Dict())

    # Find functions that need sret thunks
    unsafe_funcs = []
    for func in functions
        if !Wrapper.is_c_lto_safe(func, dwarf_structs)
            push!(unsafe_funcs, func)
        end
    end

    if isempty(unsafe_funcs)
        println("  c-thunks: All functions are ccall-safe, no thunks needed.")
        return
    end

    # Collect original headers for #include
    include_dirs = ConfigurationManager.get_include_dirs(config)

    # Find header files to include (from include dirs)
    header_includes = String[]
    for inc_dir in include_dirs
        if isdir(inc_dir)
            for f in readdir(inc_dir)
                if endswith(f, ".h") || endswith(f, ".H")
                    push!(header_includes, "#include \"$f\"")
                end
            end
        end
    end

    # Generate C thunk source
    thunk_lines = String[]
    push!(thunk_lines, "/* Auto-generated sret thunks for packed/union returns */")
    push!(thunk_lines, "/* Compiled by the same Clang as the library — version-matched */")
    push!(thunk_lines, "")
    for inc in header_includes
        push!(thunk_lines, inc)
    end
    push!(thunk_lines, "")

    for func in unsafe_funcs
        mangled = func["mangled"]
        c_ret = get(func["return_type"], "c_type", "void")
        params = get(func, "parameters", [])

        # Build C parameter list
        c_params = String[]
        c_args = String[]
        push!(c_params, "$c_ret *__sret")  # leading sret pointer
        for (i, p) in enumerate(params)
            p_type = get(p, "c_type", "int")
            p_name = get(p, "name", "arg$i")
            push!(c_params, "$p_type $p_name")
            push!(c_args, p_name)
        end

        thunk_name = "_c_sret_$mangled"
        push!(thunk_lines, "void $thunk_name($(join(c_params, ", "))) {")
        push!(thunk_lines, "    *__sret = $mangled($(join(c_args, ", ")));")
        push!(thunk_lines, "}")
        push!(thunk_lines, "")
    end

    thunk_c_path = joinpath(output_dir, "_c_thunks.c")
    write(thunk_c_path, join(thunk_lines, "\n"))

    println("  c-thunks: Generated $(length(unsafe_funcs)) sret wrappers")

    # Compile to object
    thunks_obj = joinpath(output_dir, "_c_thunks.o")
    compile_args = ["-fPIC", "-c", "-O2", "-o", thunks_obj, thunk_c_path]
    for dir in include_dirs
        push!(compile_args, "-I$dir")
    end
    for (k, v) in config.compile.defines
        push!(compile_args, "-D$k=$v")
    end

    (output, exitcode) = BuildBridge.execute("clang", compile_args)
    if exitcode != 0
        @warn "Failed to compile C thunks" output
        return
    end

    # Link into companion shared library
    lib_name = basename(library_path)
    thunks_name = replace(lib_name, ".so" => "_thunks.so", ".dylib" => "_thunks.dylib", ".dll" => "_thunks.dll")
    thunks_so = joinpath(output_dir, thunks_name)
    lib_dir = dirname(abspath(library_path))

    link_args = ["-shared", "-fPIC", "-o", thunks_so, thunks_obj,
                 "-L", lib_dir, "-l:$lib_name", "-Wl,-rpath,$lib_dir"]
    (output, exitcode) = BuildBridge.execute("clang", link_args)
    if exitcode != 0
        @warn "Failed to link C thunks" output
        return
    end

    # If LTO enabled, also emit LLVM IR → bitcode (version-matched)
    if config.link.enable_lto
        thunks_lto_ll = joinpath(output_dir, replace(lib_name, ".so" => "_thunks_lto.ll", ".dylib" => "_thunks_lto.ll"))
        ir_args = ["-S", "-emit-llvm", "-fPIC", "-O2", "-o", thunks_lto_ll, thunk_c_path]
        for dir in include_dirs
            push!(ir_args, "-I$dir")
        end
        for (k, v) in config.compile.defines
            push!(ir_args, "-D$k=$v")
        end

        (output, exitcode) = BuildBridge.execute("clang", ir_args)
        if exitcode != 0
            @warn "Failed to emit C thunks LLVM IR" output
        else
            # Sanitize and assemble to bitcode
            lto_ir_text = read(thunks_lto_ll, String)
            lto_ir_text = Compiler.sanitize_ir_for_julia(lto_ir_text)
            write(thunks_lto_ll, lto_ir_text)

            thunks_bc_path = replace(thunks_lto_ll, ".ll" => ".bc")
            Compiler.assemble_bitcode(thunks_lto_ll, thunks_bc_path)
        end
    end

    elapsed = round(time() - start_time, digits=2)
    size_kb = round(filesize(thunks_so) / 1024, digits=1)
    println("  c-thunks: $thunks_name ($size_kb KB) in $(elapsed)s")

    # Cleanup
    rm(thunks_obj, force=true)
end

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
