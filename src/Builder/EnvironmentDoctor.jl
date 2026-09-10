#!/usr/bin/env julia
# EnvironmentDoctor.jl - Toolchain diagnostics for RepliBuild
#
# Modelled on nvim's `:checkhealth`: sections, one line per probe, and ADVICE
# only under the things that failed.
#
# THE POINT OF THE 2026-09 REWRITE: this file used to probe five tools —
# llvm-config, clang++, mlir-tblgen, cmake, libJLCS — and then print
# "All systems go — Tier 3 (ccall) and Tier 2 (MLIR thunks) ready". It never
# looked at the GNU binutils DWARF dumper (the tool `Compiler._no_dwarf_dumper_error`
# exists for) and never looked at the C bucket's sysroot. Without a GNU dumper
# there is no DWARF extraction, so NO tier produces a wrapper — and the one
# command whose job is to say the toolchain is sound reported green. A doctor
# that certifies a capability it did not test is worse than no doctor, because
# it sends you looking somewhere else.
#
# Rule for anything added here: probe the thing the build ACTUALLY invokes, by
# invoking it. Not "is a binary with this name on PATH" — `objdump` on PATH in
# an MSYS2 CLANG64 shell is llvm-objdump, which answers `--version` happily and
# then parses zero DIEs.

module EnvironmentDoctor

import ..SRC_DIR
import ..Compiler
import ..BuildBridge
using Libdl

export check_environment, ToolchainStatus, ToolStatus

# =============================================================================
# STATUS TYPES
# =============================================================================

"""Status of a single probe.

`section` and `advice` drive the checkhealth layout. The first seven fields are
load-bearing elsewhere — `PackageRegistry` caches `ToolchainStatus` to JSON by
field name — so they keep their names and order.
"""
struct ToolStatus
    name::String
    required::Bool
    found::Bool
    path::String
    version::String
    meets_requirement::Bool
    message::String
    section::String
    advice::Vector{String}
end

ToolStatus(name, required, found, path, version, meets, message;
           section::String="", advice::Vector{String}=String[]) =
    ToolStatus(name, required, found, path, version, meets, message, section, advice)

"""Aggregate toolchain status.

`tier1_ready` is a MISNOMER kept for compatibility: it means Tier **3** (ccall),
the ccall path. `PackageRegistry` persists it under that key in its env-check
cache, so renaming the field silently invalidates every cached check. Read it as
"the ccall path works". `tier3_ready` is provided as the honest spelling.
"""
struct ToolchainStatus
    tools::Vector{ToolStatus}
    ready::Bool
    tier1_ready::Bool   # ccall tier — this is Tier 3, see the docstring
    tier2_ready::Bool   # MLIR thunks
end

# Iterating the report is the REPL affordance: `for t in status`, `status[3]`,
# `filter(t -> t.status_symbol == :fail, collect(status))`.
Base.length(s::ToolchainStatus)      = length(s.tools)
Base.iterate(s::ToolchainStatus)     = iterate(s.tools)
Base.iterate(s::ToolchainStatus, st) = iterate(s.tools, st)
Base.getindex(s::ToolchainStatus, i) = s.tools[i]
Base.eltype(::Type{ToolchainStatus}) = ToolStatus

"""`:ok`, `:warn` or `:fail` for one probe — the checkhealth verdict."""
function verdict(t::ToolStatus)::Symbol
    (t.found && t.meets_requirement) && return :ok
    t.required && return :fail
    return :warn
end

tier3_ready(s::ToolchainStatus) = s.tier1_ready

function Base.show(io::IO, s::ToolchainStatus)
    n_fail = count(t -> verdict(t) === :fail, s.tools)
    n_warn = count(t -> verdict(t) === :warn, s.tools)
    print(io, "ToolchainStatus($(length(s.tools)) probes, $n_fail failed, $n_warn warned; ",
              "tier3=", s.tier1_ready, " tier2=", s.tier2_ready, ")")
end

# =============================================================================
# ANSI (stdlib only — no UI dependency; `printstyled` would also do)
# =============================================================================

const BOLD   = "\e[1m"
const RED    = "\e[31m"
const GREEN  = "\e[32m"
const YELLOW = "\e[33m"
const BLUE   = "\e[34m"
const CYAN   = "\e[36m"
const RESET  = "\e[0m"
const DIM    = "\e[2m"

# =============================================================================
# PROBE HELPERS
# =============================================================================

function _find_tool(name::String)::Tuple{Bool, String}
    path = Sys.which(name)
    (path !== nothing && isfile(path)) ? (true, String(path)) : (false, "")
end

function _get_version(tool_path, args::Vector{String}=["--version"])::String
    try
        output = read(`$tool_path $args`, String)
        m = match(r"(\d+\.\d+\.\d+)", output)
        m !== nothing && return m.captures[1]
        m = match(r"(\d+\.\d+)", output)
        m !== nothing && return m.captures[1]
        return String(strip(first(split(output, '\n'))))
    catch
        return "unknown"
    end
end

_parse_major_version(v::String)::Int =
    (m = match(r"^(\d+)", v)) === nothing ? 0 : parse(Int, m.captures[1])

const MIN_LLVM_VERSION = 21

# One source for the versioned probe names. These used to be string literals
# ("llvm-config-21") that drifted from MIN_LLVM_VERSION the moment it moved.
_versioned(base::String) = [base * "-" * string(MIN_LLVM_VERSION), base]

const SEC_HOST  = "Host"
const SEC_C     = "C bucket — Tier 3 (ccall)"
const SEC_DWARF = "DWARF extraction — required by every tier"
const SEC_CPP   = "C++ — Tier 2 (MLIR thunks)"
const SEC_MSYS  = "MSYS2"

# =============================================================================
# PROBES — Host
# =============================================================================

function _check_platform()::ToolStatus
    detail = string(Sys.MACHINE, "  ·  Julia ", VERSION, "  ·  libLLVM ", Base.libllvm_version)
    ToolStatus("platform", false, true, "", string(VERSION), true, detail; section=SEC_HOST)
end

# =============================================================================
# PROBES — C bucket (Tier 3)
# =============================================================================

# The C bucket does not use system LLVM at all: it compiles through
# Clang_unified_jll, version-locked to Julia's resident libLLVM, and links
# in-process. Reporting on system clang here would describe the wrong compiler.
function _check_c_bucket_clang()::ToolStatus
    try
        Clang_mod = Base.require(Base.PkgId(
            Base.UUID("40e3b903-d033-50b4-a0cc-940c62c95e31"), "Clang"))
        if !isdefined(Clang_mod, :Clang_unified_jll)
            return ToolStatus("clang (JLL)", true, false, "", "", false,
                "Clang.jl has no Clang_unified_jll — the C bucket has no compiler";
                section=SEC_C,
                advice=["Reinstall Clang.jl: `] add Clang`",
                        "The C bucket cannot fall back to system clang without losing the",
                        "version lock to Julia's libLLVM that keeps DWARF intact."])
        end
        # Run the JLL's Cmd, NOT `first(cmd.exec)`. The wrapper carries the
        # LD_LIBRARY_PATH that points at the JLL's own libLLVM.so.18.1jl;
        # invoking the bare path finds no libLLVM, prints a loader error to
        # stderr, and reports the version as "unknown".
        cmd = Clang_mod.Clang_unified_jll.clang()
        ver = try
            out = read(pipeline(`$cmd --version`, stderr=devnull), String)
            m = match(r"(\d+\.\d+\.\d+)", out)
            m === nothing ? "unknown" : m.captures[1]
        catch
            "unknown"
        end
        return ToolStatus("clang (JLL)", true, true, first(cmd.exec), ver, true,
            "clang $ver $(DIM)(version-locked to Julia's libLLVM $(Base.libllvm_version))$(RESET)";
            section=SEC_C)
    catch e
        return ToolStatus("clang (JLL)", true, false, "", "", false,
            "Clang_unified_jll unavailable: $(first(sprint(showerror, e), 60))";
            section=SEC_C,
            advice=["`] add Clang` — the C bucket compiles through this JLL, not system clang."])
    end
end

# A compiler with no headers compiles nothing. On Windows RepliBuild has to be
# TOLD where the sysroot is (Clang_unified_jll ships none, and MSYS2 leaves
# empty ucrt64/mingw64 trees around to confuse the search). On Linux the system
# headers are found through clang's own defaults — so the honest probe is not
# "does a path exist" but "does a TU that includes stdio.h actually compile".
function _check_sysroot()::ToolStatus
    if Sys.iswindows()
        root = try Compiler._c_bucket_sysroot() catch; "" end
        if isempty(root)
            return ToolStatus("sysroot", true, false, "", "", false,
                "no mingw sysroot found — the C bucket cannot compile or link";
                section=SEC_C,
                advice=["In an MSYS2 CLANG64 shell:",
                        "  pacman -S --needed mingw-w64-clang-x86_64-toolchain",
                        "Or point at one:  REPLIBUILD_C_SYSROOT=C:/msys64/clang64"])
        end
        return ToolStatus("sysroot", true, true, root, "", true,
            "$(root) $(DIM)(include/math.h present)$(RESET)"; section=SEC_C)
    end

    mktempdir() do dir
        src = joinpath(dir, "probe.c")
        write(src, "#include <stdio.h>\nint probe(void){return 0;}\n")
        obj = joinpath(dir, "probe.o")
        ok = try
            (_, ec) = Compiler._clang_for_c_bucket("clang", ["-c", src, "-o", obj])
            ec == 0
        catch
            false
        end
        ok && return ToolStatus("sysroot", true, true, "", "", true,
            "system headers reachable $(DIM)(compiled a TU including stdio.h)$(RESET)";
            section=SEC_C)
        return ToolStatus("sysroot", true, false, "", "", false,
            "cannot compile a TU that includes <stdio.h>";
            section=SEC_C,
            advice=["Install the C library headers:",
                    "  Arch:          pacman -S glibc",
                    "  Debian/Ubuntu: apt install libc6-dev",
                    "  Fedora:        dnf install glibc-devel"])
    end
end

# =============================================================================
# PROBES — DWARF extraction (every tier depends on this)
# =============================================================================

function _dumper_advice()::Vector{String}
    if Sys.iswindows()
        return ["`objdump` in a CLANG64 shell IS llvm-objdump — a different dump",
                "dialect that parses to zero functions, so it is rejected on purpose.",
                "Install GNU binutils into the CLANG64 prefix:",
                "  pacman -S mingw-w64-clang-x86_64-binutils",
                "Or point at one:  REPLIBUILD_OBJDUMP=/path/to/objdump.exe"]
    end
    return ["RepliBuild parses GNU binutils' dump format specifically; no other",
            "dumper is substitutable (llvm-readelf/llvm-objdump print a different dialect).",
            "  Arch:          pacman -S binutils",
            "  Debian/Ubuntu: apt install binutils",
            "  Fedora:        dnf install binutils"]
end

function _check_dwarf_dumper()::ToolStatus
    d = try Compiler._dwarf_dumper() catch; nothing end
    if d === nothing
        return ToolStatus("DWARF dumper", true, false, "", "", false,
            "no GNU binutils readelf/objdump — DWARF extraction cannot run";
            section=SEC_DWARF, advice=_dumper_advice())
    end
    # Version via BuildBridge.execute, the same call `_dwarf_dumper` used to
    # accept this tool. Reading it with a bare `read(`$tool --version`)` can
    # disagree — BuildBridge resolves through the LLVM environment — and a probe
    # that resolves a tool differently from the code under test is reporting on
    # a different tool.
    ver = try
        (out, ec) = BuildBridge.execute(d.tool, ["--version"])
        m = ec == 0 ? match(r"(\d+\.\d+(?:\.\d+)?)", out) : nothing
        m === nothing ? "unknown" : m.captures[1]
    catch
        "unknown"
    end
    path = something(Sys.which(d.tool), d.tool)
    ToolStatus("DWARF dumper", true, true, String(path), ver, true,
        "GNU $(basename(String(d.tool))) $ver $(DIM)($(d.dwarf)info)$(RESET)";
        section=SEC_DWARF)
end

# Symbol extraction. ELF asks `nm -D`; PE has no dynamic symbol table and `nm -D`
# is a hard error there, so the export directory via `objdump -p` is the
# authority instead. Two containers, two tools, one question.
function _check_symbol_reader()::ToolStatus
    if Sys.iswindows()
        tool = try Compiler._gnu_objdump() catch; "" end
        if isempty(tool)
            return ToolStatus("symbol reader", true, false, "", "", false,
                "no GNU objdump — PE export directory unreadable";
                section=SEC_DWARF, advice=_dumper_advice())
        end
        return ToolStatus("symbol reader", true, true, tool, _get_version(tool), true,
            "GNU objdump $(DIM)(objdump -p, PE export directory)$(RESET)"; section=SEC_DWARF)
    end

    found, path = _find_tool("nm")
    if !found
        return ToolStatus("symbol reader", true, false, "", "", false,
            "nm not found — symbol extraction cannot run";
            section=SEC_DWARF, advice=_dumper_advice())
    end
    out = try read(`$path --version`, String) catch; "" end
    if !Compiler._is_gnu_binutils(out)
        return ToolStatus("symbol reader", true, true, path, "", false,
            "`nm` on PATH is not GNU binutils";
            section=SEC_DWARF, advice=_dumper_advice())
    end
    ToolStatus("symbol reader", true, true, path, _get_version(path), true,
        "GNU nm $(_get_version(path)) $(DIM)(nm -D --defined-only)$(RESET)"; section=SEC_DWARF)
end

# =============================================================================
# PROBES — C++ / Tier 2
# =============================================================================

const _LLVM_ADVICE = Sys.iswindows() ?
    ["In an MSYS2 CLANG64 shell:",
     "  pacman -S --needed mingw-w64-clang-x86_64-toolchain mingw-w64-clang-x86_64-mlir"] :
    ["  Arch:          yay -S llvm mlir",
     "  Debian/Ubuntu: wget https://apt.llvm.org/llvm.sh && sudo bash llvm.sh $(MIN_LLVM_VERSION)",
     "  Fedora:        dnf install llvm-devel mlir-devel clang-devel",
     "Tier 2 only — the C path needs none of this."]

function _check_llvm_config()::ToolStatus
    for name in _versioned("llvm-config")
        found, path = _find_tool(name)
        found || continue
        version = _get_version(path)
        meets = _parse_major_version(version) >= MIN_LLVM_VERSION
        msg = meets ? "LLVM $version" : "LLVM $version — need $MIN_LLVM_VERSION+"
        return ToolStatus("llvm-config", false, true, path, version, meets, msg;
                          section=SEC_CPP, advice=(meets ? String[] : _LLVM_ADVICE))
    end
    ToolStatus("llvm-config", false, false, "", "", false,
        "LLVM $MIN_LLVM_VERSION+ not found"; section=SEC_CPP, advice=_LLVM_ADVICE)
end

function _check_clang()::ToolStatus
    for name in _versioned("clang++")
        found, path = _find_tool(name)
        found || continue
        version = _get_version(path)
        meets = _parse_major_version(version) >= MIN_LLVM_VERSION
        msg = meets ? "Clang $version" : "Clang $version — need $MIN_LLVM_VERSION+"
        return ToolStatus("clang++", false, true, path, version, meets, msg;
                          section=SEC_CPP, advice=(meets ? String[] : _LLVM_ADVICE))
    end
    ToolStatus("clang++", false, false, "", "", false,
        "Clang $MIN_LLVM_VERSION+ not found"; section=SEC_CPP, advice=_LLVM_ADVICE)
end

function _check_mlir_tblgen()::ToolStatus
    for name in _versioned("mlir-tblgen")
        found, path = _find_tool(name)
        found || continue
        return ToolStatus("mlir-tblgen", false, true, path, _get_version(path), true,
            "MLIR TableGen $(_get_version(path))"; section=SEC_CPP)
    end
    ToolStatus("mlir-tblgen", false, false, "", "", false,
        "mlir-tblgen not found — cannot build the JLCS dialect";
        section=SEC_CPP, advice=_LLVM_ADVICE)
end

function _check_cmake()::ToolStatus
    found, path = _find_tool("cmake")
    found || return ToolStatus("cmake", false, false, "", "", false,
        "cmake not found — cannot build the JLCS dialect"; section=SEC_CPP,
        advice=["  Arch: pacman -S cmake   Debian: apt install cmake   Fedora: dnf install cmake"])
    ToolStatus("cmake", false, true, path, _get_version(path), true,
        "CMake $(_get_version(path))"; section=SEC_CPP)
end

function _check_libJLCS()::ToolStatus
    lib_path = joinpath(SRC_DIR, "mlir", "build", "libJLCS." * Libdl.dlext)
    isfile(lib_path) || return ToolStatus("libJLCS", false, false, "", "", false,
        "dialect not built"; section=SEC_CPP,
        advice=["cd $(joinpath(SRC_DIR, "mlir")) && ./build.sh",
                "Rebuild after any system MLIR upgrade — the artifact pins the minor SONAME."])
    ToolStatus("libJLCS", false, true, lib_path, "", true,
        "dialect compiled"; section=SEC_CPP)
end

# =============================================================================
# PROBES — MSYS2 (Windows only)
# =============================================================================

# Which MSYS2 environment the process is in decides the C++ ABI of every wrapper
# DLL. CLANG64 is the supported one because it matches Julia's own mingw build.
function _check_msys2()::ToolStatus
    prefix = get(ENV, "MSYSTEM", "")
    if isempty(prefix)
        return ToolStatus("MSYSTEM", false, false, "", "", false,
            "not running inside an MSYS2 shell"; section=SEC_MSYS,
            advice=["RepliBuild's Windows target is x86_64-w64-windows-gnu under MSYS2 CLANG64.",
                    "Launch the CLANG64 shell (not MSYS, not MINGW64) and re-run."])
    end
    if uppercase(prefix) != "CLANG64"
        return ToolStatus("MSYSTEM", false, true, "", prefix, false,
            "$prefix — CLANG64 is the supported environment"; section=SEC_MSYS,
            advice=["CLANG64 matches Julia's own mingw build, so a wrapper DLL shares the",
                    "process C++ ABI. $prefix uses a different stdlib/EH model — C++",
                    "exceptions crossing a thunk are what break first."])
    end
    ToolStatus("MSYSTEM", false, true, get(ENV, "MINGW_PREFIX", ""), prefix, true,
        "CLANG64 $(DIM)($(get(ENV, "MINGW_PREFIX", "?")))$(RESET)"; section=SEC_MSYS)
end

# =============================================================================
# PUBLIC API
# =============================================================================

"""
    check_environment(; verbose=true, throw_on_error=false) -> ToolchainStatus

Toolchain checkhealth: probes every tool the build actually invokes, grouped by
the capability it enables, with fix advice under whatever failed.

Probes the C bucket (Clang_unified_jll + sysroot), DWARF extraction (GNU dumper
+ symbol reader), the C++/Tier-2 chain (llvm-config, clang++, mlir-tblgen,
cmake, libJLCS), and on Windows the MSYS2 environment.

The result is iterable, so the REPL can pick it apart:

```julia
status = RepliBuild.check_environment(verbose=false)
status.tier1_ready                                  # ccall path (Tier 3 — see ToolchainStatus)
status.tier2_ready                                  # MLIR thunks
[t.name for t in status if !t.meets_requirement]    # what is broken
first(t for t in status if t.name == "sysroot")     # one probe
```

# Arguments
- `verbose`: print the report (default `true`)
- `throw_on_error`: raise if a required probe failed (default `false`)
"""
function check_environment(; verbose::Bool=true, throw_on_error::Bool=false)::ToolchainStatus
    tools = ToolStatus[
        _check_platform(),
        _check_c_bucket_clang(),
        _check_sysroot(),
        _check_dwarf_dumper(),
        _check_symbol_reader(),
        _check_llvm_config(),
        _check_clang(),
        _check_mlir_tblgen(),
        _check_cmake(),
        _check_libJLCS(),
    ]
    Sys.iswindows() && push!(tools, _check_msys2())

    by(name) = (i = findfirst(t -> t.name == name, tools); i === nothing ? nothing : tools[i])
    ok(name) = (t = by(name); t !== nothing && t.found && t.meets_requirement)

    # DWARF extraction gates EVERY tier — a wrapper is generated from the dump.
    # This is the dependency the old report left out, which is how it could
    # print "All systems go" on a box that cannot wrap anything.
    dwarf_ok = ok("DWARF dumper") && ok("symbol reader")
    tier3 = dwarf_ok && ok("clang (JLL)") && ok("sysroot")
    tier2 = tier3 && ok("llvm-config") && ok("clang++") &&
            ok("mlir-tblgen") && ok("cmake") && ok("libJLCS")

    status = ToolchainStatus(tools, tier3, tier3, tier2)
    verbose && _print_report(status)

    if throw_on_error && !status.ready
        broken = [t for t in tools if t.required && !(t.found && t.meets_requirement)]
        lines = ["[RepliBuild] Toolchain incomplete — $(length(broken)) required probe(s) failed:"]
        for t in broken
            push!(lines, "  ✗ $(t.name): $(_plain(t.message))")
            append!(lines, ("      " * a for a in t.advice))
        end
        throw(ErrorException(join(lines, "\n")))
    end
    return status
end

_plain(s::AbstractString) = replace(s, r"\e\[[0-9;]*m" => "")

function _icon(v::Symbol)
    v === :ok   && return "$(GREEN)OK$(RESET)  "
    v === :warn && return "$(YELLOW)WARN$(RESET)"
    return "$(RED)FAIL$(RESET)"
end

function _print_report(status::ToolchainStatus)
    println()
    println("$(BOLD)$(BLUE)[RepliBuild] checkhealth$(RESET)")

    last_section = ""
    for t in status.tools
        if t.section != last_section
            println()
            println("$(BOLD)$(t.section) $(DIM)~$(RESET)")
            last_section = t.section
        end
        println("  $(_icon(verdict(t))) $(BOLD)$(rpad(t.name, 14))$(RESET) $(t.message)")
        if verdict(t) !== :ok
            for line in t.advice
                println("       $(DIM)│$(RESET) $(CYAN)$line$(RESET)")
            end
            isempty(t.advice) || println()
        elseif !isempty(t.path) && t.section != SEC_HOST
            println("       $(DIM)$(t.path)$(RESET)")
        end
    end

    println()
    println("$(BOLD)Summary $(DIM)~$(RESET)")
    if status.tier1_ready
        println("  $(_icon(:ok)) $(GREEN)Tier 3 (ccall)$(RESET) — C and C++ wrappers will build")
    else
        println("  $(_icon(:fail)) $(RED)Tier 3 (ccall) unavailable — no wrapper can be generated$(RESET)")
    end
    if status.tier2_ready
        println("  $(_icon(:ok)) $(GREEN)Tier 2 (MLIR thunks)$(RESET) — C++ virtual dispatch and exception-safe calls")
    else
        println("  $(_icon(:warn)) $(YELLOW)Tier 2 (MLIR thunks) unavailable$(RESET) — C++ falls back to ccall where it can")
    end
    println()
end

end # module EnvironmentDoctor
