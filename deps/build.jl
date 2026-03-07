#!/usr/bin/env julia
# deps/build.jl — Auto-compile the JLCS MLIR dialect when RepliBuild is installed
#
# This script is executed by Julia's package manager after `] add RepliBuild`.
# It detects CMake, LLVM, and MLIR, then compiles src/mlir/ → libJLCS.so.
# If the toolchain is missing, it warns gracefully — Tier 1 (ccall) still works.

const REPLIBUILD_ROOT = dirname(@__DIR__)
const MLIR_SRC_DIR = joinpath(REPLIBUILD_ROOT, "src", "mlir")
const BUILD_DIR = joinpath(MLIR_SRC_DIR, "build")
const LIB_NAME = Sys.isapple() ? "libJLCS.dylib" : (Sys.iswindows() ? "libJLCS.dll" : "libJLCS.so")
const LIB_PATH = joinpath(BUILD_DIR, LIB_NAME)

# ─────────────────────────────────────────────────────────────────────────────
# Hash-based skip: don't rebuild if sources haven't changed
# ─────────────────────────────────────────────────────────────────────────────

function _source_hash()::UInt64
    h = UInt64(0)
    if isdir(MLIR_SRC_DIR)
        for f in readdir(MLIR_SRC_DIR; join=true)
            if isfile(f) && any(endswith(f, ext) for ext in [".td", ".cpp", ".h", ".hpp", ".txt"])
                h = hash(read(f), h)
            end
        end
        # Also hash files in impl/ subdirectory
        impl_dir = joinpath(MLIR_SRC_DIR, "impl")
        if isdir(impl_dir)
            for f in readdir(impl_dir; join=true)
                if isfile(f)
                    h = hash(read(f), h)
                end
            end
        end
    end
    return h
end

function _read_stored_hash()::UInt64
    hash_file = joinpath(BUILD_DIR, ".build_hash")
    isfile(hash_file) || return UInt64(0)
    try
        return parse(UInt64, strip(read(hash_file, String)))
    catch
        return UInt64(0)
    end
end

function _write_stored_hash(h::UInt64)
    mkpath(BUILD_DIR)
    write(joinpath(BUILD_DIR, ".build_hash"), string(h))
end

# ─────────────────────────────────────────────────────────────────────────────
# Tool detection
# ─────────────────────────────────────────────────────────────────────────────

function _find(name::String)::Union{String, Nothing}
    path = Sys.which(name)
    return (path !== nothing && isfile(path)) ? path : nothing
end

function _get_version(tool::String)::String
    try
        output = read(`$tool --version`, String)
        m = match(r"(\d+\.\d+\.\d+)", output)
        return m !== nothing ? m.captures[1] : "unknown"
    catch
        return "unknown"
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Main build logic
# ─────────────────────────────────────────────────────────────────────────────

function main()
    println("[RepliBuild] deps/build.jl — JLCS MLIR dialect compilation")

    # Check if library already exists and sources haven't changed
    current_hash = _source_hash()
    if isfile(LIB_PATH) && current_hash == _read_stored_hash()
        println("[RepliBuild] ✓ libJLCS is up-to-date, skipping build.")
        return
    end

    # Check for cmake
    cmake = _find("cmake")
    if cmake === nothing
        @warn """
        [RepliBuild] cmake not found. Cannot compile JLCS MLIR dialect.
        Tier 1 (ccall) builds will still work. Tier 2 (MLIR JIT/AOT) requires:
          Ubuntu/Debian: sudo apt install cmake
          Arch Linux:    sudo pacman -S cmake
          macOS:         brew install cmake
        """
        return
    end

    # Check for LLVM
    llvm_config = something(_find("llvm-config-21"), _find("llvm-config"), nothing)
    if llvm_config === nothing
        @warn """
        [RepliBuild] llvm-config not found. Cannot compile JLCS MLIR dialect.
        Tier 1 builds still work if clang++ is available separately.
        For Tier 2 (MLIR JIT):
          Ubuntu/Debian: wget https://apt.llvm.org/llvm.sh && sudo bash llvm.sh 21
          Arch Linux:    yay -S llvm-minimal-git
          macOS:         brew install llvm@21
        """
        return
    end

    # Check for mlir-tblgen
    mlir_tblgen = something(_find("mlir-tblgen-21"), _find("mlir-tblgen"), nothing)
    if mlir_tblgen === nothing
        @warn """
        [RepliBuild] mlir-tblgen not found. Cannot compile JLCS MLIR dialect.
        Tier 1 builds still work. For Tier 2 (MLIR JIT):
          Ubuntu/Debian: sudo apt install mlir-21-tools
          Arch Linux:    yay -S mlir-minimal-git
          macOS:         brew install llvm@21  (includes MLIR)
        """
        return
    end

    llvm_version = _get_version(llvm_config)
    println("[RepliBuild] Found LLVM $llvm_version at $llvm_config")
    println("[RepliBuild] Found mlir-tblgen at $mlir_tblgen")

    # Get LLVM/MLIR CMake paths
    llvm_cmake_dir = strip(read(`$llvm_config --cmakedir`, String))
    llvm_prefix = strip(read(`$llvm_config --prefix`, String))
    mlir_cmake_dir = joinpath(llvm_prefix, "lib", "cmake", "mlir")

    if !isdir(mlir_cmake_dir)
        # Try alternative paths
        for alt in [joinpath(llvm_prefix, "lib64", "cmake", "mlir"),
                    joinpath(llvm_prefix, "share", "cmake", "mlir")]
            if isdir(alt)
                mlir_cmake_dir = alt
                break
            end
        end
    end

    println("[RepliBuild] LLVM CMake: $llvm_cmake_dir")
    println("[RepliBuild] MLIR CMake: $mlir_cmake_dir")

    # Build
    mkpath(BUILD_DIR)

    build_type = get(ENV, "REPLIBUILD_BUILD_TYPE", "Release")

    println("[RepliBuild] Configuring CMake ($build_type)...")
    cmake_args = [
        MLIR_SRC_DIR,
        "-DCMAKE_BUILD_TYPE=$build_type",
        "-DLLVM_DIR=$llvm_cmake_dir",
        "-DMLIR_DIR=$mlir_cmake_dir",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    ]

    try
        cd(BUILD_DIR) do
            run(`$cmake $cmake_args`)

            nproc = Sys.CPU_THREADS
            println("[RepliBuild] Building with $nproc threads...")
            run(`$cmake --build . -j$nproc`)
        end
    catch e
        @warn """
        [RepliBuild] MLIR dialect compilation failed.
        Tier 1 (ccall) builds will still work. The MLIR JIT tier requires a working
        LLVM/MLIR development installation. Error: $(sprint(showerror, e))
        """
        return
    end

    if isfile(LIB_PATH)
        _write_stored_hash(current_hash)
        size_kb = round(filesize(LIB_PATH) / 1024, digits=1)
        println("[RepliBuild] ✓ Built $LIB_NAME ($size_kb KB)")
    else
        @warn "[RepliBuild] Build completed but $LIB_NAME not found at expected path: $LIB_PATH"
    end
end

main()
