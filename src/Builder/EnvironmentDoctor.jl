#!/usr/bin/env julia
# EnvironmentDoctor.jl - Bulletproof environment diagnostics for RepliBuild
# Validates toolchain requirements and provides actionable fix instructions

module EnvironmentDoctor

import ..SRC_DIR

export check_environment, ToolchainStatus, ToolStatus

# =============================================================================
# STATUS TYPES
# =============================================================================

"""Status of a single tool in the toolchain."""
struct ToolStatus
    name::String
    required::Bool
    found::Bool
    path::String
    version::String
    meets_requirement::Bool
    message::String
end

"""Aggregate status of the entire toolchain."""
struct ToolchainStatus
    tools::Vector{ToolStatus}
    ready::Bool
    tier1_ready::Bool  # ccall tier (needs clang/llvm)
    tier2_ready::Bool  # MLIR JIT tier (needs mlir-tblgen, cmake, libJLCS.so)
end

# =============================================================================
# ANSI COLOR HELPERS
# =============================================================================

const BOLD    = "\e[1m"
const RED     = "\e[31m"
const GREEN   = "\e[32m"
const YELLOW  = "\e[33m"
const BLUE    = "\e[34m"
const CYAN    = "\e[36m"
const RESET   = "\e[0m"
const DIM     = "\e[2m"

_ok()   = "$(GREEN)✓$(RESET)"
_fail() = "$(RED)✗$(RESET)"
_warn() = "$(YELLOW)!$(RESET)"

# =============================================================================
# TOOL DETECTION
# =============================================================================

function _find_tool(name::String)::Tuple{Bool, String}
    path = Sys.which(name)
    if path !== nothing && isfile(path)
        return (true, path)
    end
    return (false, "")
end

function _get_version(tool_path::String, args::Vector{String}=["--version"])::String
    try
        output = read(`$tool_path $args`, String)
        # Extract version number pattern
        m = match(r"(\d+\.\d+\.\d+)", output)
        if m !== nothing
            return m.captures[1]
        end
        m = match(r"(\d+\.\d+)", output)
        if m !== nothing
            return m.captures[1]
        end
        return strip(split(output, '\n')[1])
    catch
        return "unknown"
    end
end

function _parse_major_version(version::String)::Int
    m = match(r"^(\d+)", version)
    m === nothing && return 0
    return parse(Int, m.captures[1])
end

const MIN_LLVM_VERSION = 21

function _check_llvm_config()::ToolStatus
    # Try versioned names first, then generic
    for name in ["llvm-config-21", "llvm-config"]
        found, path = _find_tool(name)
        if found
            version = _get_version(path, ["--version"])
            major = _parse_major_version(version)
            meets = major >= MIN_LLVM_VERSION
            msg = meets ? "LLVM $version" : "Found LLVM $version — need $MIN_LLVM_VERSION+"
            return ToolStatus("llvm-config", true, true, path, version, meets, msg)
        end
    end
    return ToolStatus("llvm-config", true, false, "", "", false, 
                      "LLVM $MIN_LLVM_VERSION+ not found")
end

function _check_clang()::ToolStatus
    for name in ["clang++-21", "clang++"]
        found, path = _find_tool(name)
        if found
            version = _get_version(path, ["--version"])
            major = _parse_major_version(version)
            meets = major >= MIN_LLVM_VERSION
            msg = meets ? "Clang $version" : "Found Clang $version — need $MIN_LLVM_VERSION+"
            return ToolStatus("clang++", true, true, path, version, meets, msg)
        end
    end
    return ToolStatus("clang++", true, false, "", "", false,
                      "Clang $MIN_LLVM_VERSION+ not found")
end

function _check_mlir_tblgen()::ToolStatus
    for name in ["mlir-tblgen-21", "mlir-tblgen"]
        found, path = _find_tool(name)
        if found
            version = _get_version(path, ["--version"])
            return ToolStatus("mlir-tblgen", false, true, path, version, true,
                              "MLIR TableGen available")
        end
    end
    return ToolStatus("mlir-tblgen", false, false, "", "", false,
                      "mlir-tblgen not found (needed for JIT tier)")
end

function _check_cmake()::ToolStatus
    found, path = _find_tool("cmake")
    if found
        version = _get_version(path, ["--version"])
        return ToolStatus("cmake", false, true, path, version, true, "CMake $version")
    end
    return ToolStatus("cmake", false, false, "", "", false,
                      "cmake not found (needed to build MLIR dialect)")
end

function _check_libJLCS()::ToolStatus
    # Check for the compiled MLIR dialect library
    mlir_dir = joinpath(SRC_DIR, "mlir", "build")
    so_name = Sys.isapple() ? "libJLCS.dylib" : "libJLCS.so"
    lib_path = joinpath(mlir_dir, so_name)
    
    if isfile(lib_path)
        return ToolStatus("libJLCS", false, true, lib_path, "", true,
                          "MLIR dialect compiled")
    end
    return ToolStatus("libJLCS", false, false, "", "", false,
                      "libJLCS not built (run RepliBuild deps/build.jl or src/mlir/build.sh)")
end

# =============================================================================
# INSTALL INSTRUCTIONS
# =============================================================================

function _install_instructions()::String
    return """
    $(BOLD)$(CYAN)To install the required toolchain:$(RESET)

      $(BOLD)Ubuntu/Debian:$(RESET)  wget https://apt.llvm.org/llvm.sh && sudo bash llvm.sh 21
      $(BOLD)Arch Linux:$(RESET)     yay -S llvm-minimal-git mlir-minimal-git
      $(BOLD)macOS:$(RESET)          brew install llvm@21
      $(BOLD)Fedora/RHEL:$(RESET)    sudo dnf install llvm21-devel mlir21-devel clang21-devel

    $(DIM)After installation, ensure llvm-config and clang++ are in your PATH.$(RESET)"""
end

# =============================================================================
# PUBLIC API
# =============================================================================

"""
    check_environment(; verbose=true, throw_on_error=false) -> ToolchainStatus

Run comprehensive environment diagnostics for RepliBuild.

Checks for LLVM 21+, Clang, MLIR tools, CMake, and the compiled JLCS dialect.
Prints a colorful, readable diagnostic report with fix instructions if anything is missing.

# Arguments
- `verbose`: Print diagnostic report to stdout (default: true)
- `throw_on_error`: Throw an error if required tools are missing (default: false)

# Returns
`ToolchainStatus` with per-tool status and overall readiness flags.

# Example
```julia
status = RepliBuild.check_environment()
status.ready          # true if all required tools found
status.tier1_ready    # true if ccall tier works (clang + llvm)
status.tier2_ready    # true if MLIR JIT tier works
```
"""
function check_environment(; verbose::Bool=true, throw_on_error::Bool=false)::ToolchainStatus
    tools = ToolStatus[]

    # Check all tools
    push!(tools, _check_llvm_config())
    push!(tools, _check_clang())
    push!(tools, _check_mlir_tblgen())
    push!(tools, _check_cmake())
    push!(tools, _check_libJLCS())

    # Compute readiness
    llvm_ok = tools[1].found && tools[1].meets_requirement
    clang_ok = tools[2].found && tools[2].meets_requirement
    mlir_ok = tools[3].found && tools[3].meets_requirement
    cmake_ok = tools[4].found && tools[4].meets_requirement
    libjlcs_ok = tools[5].found

    tier1_ready = llvm_ok && clang_ok
    tier2_ready = tier1_ready && mlir_ok && cmake_ok && libjlcs_ok
    ready = tier1_ready  # Tier 1 is the minimum

    status = ToolchainStatus(tools, ready, tier1_ready, tier2_ready)

    if verbose
        _print_report(status)
    end

    if throw_on_error && !ready
        # Build a concise error message
        missing_tools = [t.name for t in tools if t.required && !t.meets_requirement]
        found_llvm = tools[1].found ? "Found LLVM $(tools[1].version) at $(tools[1].path)." : ""
        error_msg = """
[RepliBuild] ❌ Toolchain Error: LLVM $(MIN_LLVM_VERSION)+ and MLIR are required.
$(found_llvm)

To install the required toolchain:
  Ubuntu/Debian: wget https://apt.llvm.org/llvm.sh && sudo bash llvm.sh $(MIN_LLVM_VERSION)
  Arch Linux:    yay -S llvm-minimal-git mlir-minimal-git
  macOS:         brew install llvm@$(MIN_LLVM_VERSION)"""
        throw(ErrorException(error_msg))
    end

    return status
end

function _print_report(status::ToolchainStatus)
    println()
    println("$(BOLD)$(BLUE)[RepliBuild] Environment Diagnostics$(RESET)")
    println("$(DIM)─────────────────────────────────────$(RESET)")

    for tool in status.tools
        if tool.found && tool.meets_requirement
            icon = _ok()
            detail = "$(tool.message)$(DIM) ($(tool.path))$(RESET)"
        elseif tool.found && !tool.meets_requirement
            icon = _fail()
            detail = "$(RED)$(tool.message)$(RESET)$(DIM) ($(tool.path))$(RESET)"
        elseif !tool.found && tool.required
            icon = _fail()
            detail = "$(RED)$(tool.message)$(RESET)"
        else
            icon = _warn()
            detail = "$(YELLOW)$(tool.message)$(RESET)"
        end
        
        label = rpad(tool.name, 14)
        println("  $icon  $(BOLD)$label$(RESET) $detail")
    end

    println()

    if status.ready
        if status.tier2_ready
            println("  $(_ok())  $(GREEN)$(BOLD)All systems go — Tier 1 (ccall) and Tier 2 (MLIR JIT) ready$(RESET)")
        else
            println("  $(_ok())  $(GREEN)$(BOLD)Tier 1 (ccall) ready$(RESET) — standard builds will work")
            println("  $(_warn())  $(YELLOW)Tier 2 (MLIR JIT) unavailable — C++ virtual dispatch requires MLIR$(RESET)")
        end
    else
        println("  $(_fail())  $(RED)$(BOLD)Toolchain incomplete — builds will fail$(RESET)")
        println()
        println(_install_instructions())
    end

    println()
end

end # module EnvironmentDoctor
