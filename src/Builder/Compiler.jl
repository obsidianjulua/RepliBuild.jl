#!/usr/bin/env julia
# Compiler.jl - C++ to LLVM IR compilation and linking
# Full control over LLVM/Clang for Julia interoperability
# Extracts metadata during compilation for automatic wrapper generation

module Compiler

using Dates
using JSON
using Libdl
using LLVM
import SHA

# Import from parent RepliBuild module
import ..ConfigurationManager
import ..ConfigurationManager: RepliBuildConfig, get_source_files, get_include_dirs,
                                get_compile_flags, get_build_path, get_output_path,
                                get_library_name, get_module_name, is_parallel_enabled,
                                is_cache_enabled, get_cache_path, with_include_dirs
import ..BuildBridge
import ..Tools
import ..LibDW
import ..LLVMEnvironment
import .._read_json
import ..stl_container_size, .._scope_suffixes, .._outside_template_args, ..INTERNAL_TYPE_BLOCKLIST
import ..host_record_abi
import .._glob_matcher

export compile_to_ir, link_optimize_ir, create_library, create_executable, compile_project,
       ingest_library,
       extract_compilation_metadata, save_compilation_metadata,
       compute_project_hash, is_project_cache_valid, save_project_hash,
       assemble_bitcode, sanitize_ir_for_julia,
       # Compiler utilities
       needs_recompile, compile_single_to_ir,
       generate_template_instantiations, generate_macro_shims,
       verify_wrap_surface,
       extract_stl_method_symbols, extract_symbols_from_binary,
       extract_mangled_name, extract_dwarf_return_types,
       extract_function_name, extract_class_name,
       dwarf_type_to_julia, get_type_size,
       cpp_to_julia_type, build_type_registry,
       infer_return_type, parse_parameters, parse_function_signatures

# =============================================================================
# INCREMENTAL BUILD SUPPORT
# =============================================================================

"""
Fingerprint of everything that affects a file's IR codegen EXCEPT its own source
content (tracked per-file by mtime): compile flags, defines, include-dir paths,
the in-process compiler identity, and the target triple. Threaded into
`needs_recompile` so a change to any of these invalidates the per-file IR cache.

Without it the cache key is mtime-only: editing `[compile].flags` (e.g. adding
`-fvisibility=hidden`) leaves every source mtime untouched, so the stale IR —
built with the *old* flags — is silently reused. That is a correctness bug, not
just a staleness one: the resulting `.so` looks fine but was compiled wrong.

Excludes individual source content on purpose, so editing one `.cpp` still
recompiles only that file; a flag/define/include change busts the whole set
(correct — they affect every translation unit).

`toolchain` is a seam for tests; a build always takes `_toolchain_identity`.
"""
function compute_compile_fingerprint(config::RepliBuildConfig;
                                     toolchain::AbstractString=_toolchain_identity(config))::String
    ctx = SHA.SHA256_CTX()
    _key_feed!(ctx, "replibuild-compile-fingerprint-v3")
    for f in get_compile_flags(config)
        _key_feed!(ctx, f)
    end
    for (k, v) in sort(collect(config.compile.defines))
        _key_feed!(ctx, string(k))
        _key_feed!(ctx, string(v))
    end
    # Header *contents*, not just the -I path. The per-file cache otherwise
    # serves IR compiled against the previous header: `needs_recompile` only
    # compares the source file's mtime, and a nested `include/<lib>/<lib>.h`
    # edit does not touch that file. The project hash already notices the
    # edit and refuses the "project unchanged" fast path; without this, the
    # file cache then links the old IR anyway.
    for d in get_include_dirs(config)
        _key_feed!(ctx, d)   # -I path affects header resolution, independent of mtime
        _feed_include_headers!(ctx, d)
    end
    # The compiler and target triple; see `_toolchain_identity`.
    _key_feed!(ctx, toolchain)
    return bytes2hex(SHA.digest!(ctx))
end

"""
    _feed_include_headers!(ctx, dir) -> ctx

Feed every header under the include directory `dir` into a cache key: its path
relative to `dir`, then its contents, in sorted order. `.git`, `build` and
`.replibuild_cache` are not walked. A `dir` that does not exist feeds nothing.
`compute_compile_fingerprint` and the graph build's per-group fingerprint
(`GraphBuild`) both take a directory's headers from here, so the two agree on
which files count.
"""
function _feed_include_headers!(ctx, d::AbstractString)
    isdir(d) || return ctx
    for (root, dirs, files) in walkdir(d)
        filter!(x -> !(x in (".git", "build", ".replibuild_cache")), dirs)
        sort!(dirs)
        for f in sort(files)
            any(endswith(f, ext) for ext in (".h", ".hpp", ".hxx", ".hh")) || continue
            path = joinpath(root, f)
            isfile(path) || continue
            _key_feed!(ctx, relpath(path, d))
            _key_feed!(ctx, read(path))
        end
    end
    return ctx
end

"""
    _key_feed!(ctx, x)

Feed one field into a persisted cache key: its length, then its bytes, so two
consecutive fields cannot run together ("ab" + "c" is not "a" + "bc").

The keys are SHA-256, not `Base.hash`. They are written to disk and read back by
whatever Julia runs the next build, and `Base.hash` is not stable across Julia
releases: 1.12 and 1.13 hash the same bytes to different values. A key made with
it changed whenever Julia did, which is the only reason a Julia switch used to
rebuild a project at all. That is the wrong reason for a C project, whose
compiler did change but was never in the project hash, and a needless full
recompile for a C++ one, whose compiler did not.
"""
_key_feed!(ctx, x::AbstractString) = _key_feed!(ctx, codeunits(String(x)))
function _key_feed!(ctx, bytes::AbstractVector{UInt8})
    SHA.update!(ctx, reinterpret(UInt8, [UInt64(length(bytes))]))
    SHA.update!(ctx, bytes)
    return ctx
end

const _SYSTEM_CLANG_ID = Ref{String}("")

"""The system `clang++ --version` first line, memoized; `""` if it cannot be run."""
function _system_clang_identity()::String
    isempty(_SYSTEM_CLANG_ID[]) || return _SYSTEM_CLANG_ID[]
    (out, rc) = try
        BuildBridge.execute("clang++", ["--version"])
    catch
        ("", 1)
    end
    _SYSTEM_CLANG_ID[] = rc == 0 ? String(strip(first(split(out, '\n')))) : "unavailable"
    return _SYSTEM_CLANG_ID[]
end

"""
    _toolchain_identity(config) -> String

The compiler a project's IR comes from, plus the target triple: one string that
every persisted build key carries (`compute_compile_fingerprint`,
`compute_project_hash`, the registry's `hash_config`), so switching toolchains
rebuilds exactly what the switch affects.

- **C**: `libllvm <Base.libllvm_version>`. The JLL clang is pinned to Julia's
  own libLLVM, link/opt/promotion run in process on it, and Tier 1 slices are
  IR for it. Switching Julia (1.12 is LLVM 18, 1.13 is LLVM 20) changes all of
  that, and newer IR does not parse in an older LLVM (`icmp samesign` is LLVM 20).
- **C++**: `system <clang++ --version>`. Every TU compiles with the system
  clang, which the package manager upgrades on its own (MSYS2 moved 22 to 23),
  and Julia's libLLVM plays no part, so a Julia switch leaves a C++ project
  built.

`_toolchain_identity(language)` is the same answer for a bucket (`:c` or
`:cpp`) rather than a config: the graph build chooses the bucket per library.
"""
_toolchain_identity(config::RepliBuildConfig)::String = _toolchain_identity(config.wrap.language)

function _toolchain_identity(language::Symbol)::String
    compiler = language == :c ?
        "libllvm " * string(Base.libllvm_version) :
        "system " * _system_clang_identity()
    # Which linker made the library is part of what was built: the bundled
    # one, or whatever the driver finds on the machine.
    linker = _bundled_linker(language) === nothing ? "" : " | ld.lld"
    return compiler * " | " * _detect_target_triple() * linker
end

"""
    _bundled_linker(language) -> Union{String,Nothing}

The linker the C bucket's final link runs instead of the system's: `ld.lld`
from LLD_unified_jll, as `Tools` resolved it, or `nothing` for "whatever the
clang driver finds" (`ld` on PATH).

Bundled only where the whole of it is: the C bucket (its clang is the JLL's,
at the same LLVM as this LLD, and the driver starts the linker in the clang's
own environment, which is what lets LLD find its libLLVM), on Linux, when the
JLL clang is the compiler and `Tools` took the JLL's copy or an explicit
`REPLIBUILD_TOOL_LD_LLD`. A C++ build links with the system toolchain it
compiles with. `REPLIBUILD_TOOLS=system` means the system's default linker,
not an `ld.lld` that happens to be on PATH.

The driver still supplies the C runtime's start files and libraries from the
system: a linker is not a sysroot.
"""
function _bundled_linker(language::Symbol; jll_clang::Function = _jll_clang)::Union{String,Nothing}
    language == :c || return nothing
    Sys.islinux() || return nothing
    jll_clang() === nothing && return nothing
    r = Tools.resolve("ld.lld")
    r.source === :override && return r.path
    # `system` is the driver's own linker whether or not an ld.lld is
    # installed: under that order `Tools` falls back to the JLL's copy where
    # PATH has none, and that fallback is not what was asked for here.
    return (Tools.order() === :jll && r.source === :jll) ? r.path : nothing
end

"The driver flag that selects `_bundled_linker`, or nothing."
function _bundled_linker_flags(config::RepliBuildConfig)::Vector{String}
    lld = _bundled_linker(config.wrap.language)
    return lld === nothing ? String[] : ["--ld-path=" * lld]
end

"""
    _ir_output_path(build_dir, source_file) -> String

The `.ll` a source file compiles to. **One derivation, two call sites** — the
cache pre-check in `compile_to_ir` and the emission in `compile_single_to_ir`
must agree on this path or the cache checks one file and the compiler writes
another.

Keyed on the FULL source path, not on `splitext(basename(f))`. The old
basename-only key collided in two ways, both silent:

  * same leaf name in different directories — `src/llama.cpp` vs
    `src/models/llama.cpp`, `ggml-cpu/quants.c` vs `ggml-cpu/arch/x86/quants.c`
  * same stem, different extension in the SAME directory — `ggml.c` vs
    `ggml.cpp` (`splitext` dropped what distinguished them)

A collision cost real damage at three layers: the second compile **overwrote**
the first's IR (that TU silently vanished from the library), `ir_files` still
got one entry per source so the survivor was handed to the linker twice
("multiple definition" for every symbol in it), and `needs_recompile` compared
one source's mtime against the other's IR, so a warm cache could serve TU A's
IR as TU B's. Compilation is `Threads.@threads`, so the two writers could also
tear the same output file. Found on llama.cpp, where 195 sources produced 190
`.ll` files and the link died on duplicate symbols.

The extension is kept and the parent directory contributes an 8-hex digest, so
the name stays readable (`ggml.c-3f2a1b9c.ll`) while being a pure function of
the source path — no dependence on which other files happen to be in the build.

The digest is SHA-256 (`_ir_dir_key`), like every persisted key (see
`_key_feed!`). It was `Base.hash`, which 1.12 and 1.13 compute differently, so
switching Julia renamed every IR file: each TU was compiled again, and the old
set stayed beside the new one in `build/` (2.4 GB for llama.cpp).
"""
function _ir_output_path(build_dir::AbstractString, source_file::AbstractString)::String
    src = abspath(source_file)
    return joinpath(build_dir, string(basename(src), "-", _ir_dir_key(dirname(src)), ".ll"))
end

"""
    _ir_dir_key(dir) -> String

The 8-hex digest of a source directory in its IR file name: the first 8 hex
digits of the SHA-256 of `dir` through `_canon_path`, so one directory spelled
with another drive-letter case or separator on Windows gives one name.
"""
function _ir_dir_key(dir::AbstractString)::String
    ctx = SHA.SHA256_CTX()
    _key_feed!(ctx, "replibuild-ir-dir-v1")
    _key_feed!(ctx, _canon_path(dir))
    return first(bytes2hex(SHA.digest!(ctx)), 8)
end

_compile_key_path(ir_file::String)::String = ir_file * ".key"

"""Record the compile fingerprint that produced `ir_file` (sidecar next to it)."""
function write_compile_key(ir_file::String, fingerprint::String)
    isempty(fingerprint) && return
    try
        write(_compile_key_path(ir_file), fingerprint)
    catch
        # A failed key write just means the next build recompiles this file —
        # safe (never serves stale IR), so don't fail the build over it.
    end
end

"""
Check if a source file needs recompilation.

Cache hit requires BOTH the IR being newer than the source (mtime) AND the
compile fingerprint matching the one that produced it (`<ir>.key` sidecar). A
missing key (cache from before fingerprinting) forces a recompile so the key is
established. Pass `compile_fingerprint=""` to skip the fingerprint check (legacy
mtime-only behavior).
"""
function needs_recompile(source_file::String, ir_file::String, cache_enabled::Bool,
                         compile_fingerprint::String="")::Bool
    if !cache_enabled
        return true  # Always recompile if cache disabled
    end

    if !isfile(ir_file)
        return true  # IR doesn't exist
    end

    # Compile-configuration gate: flags/defines/includes/compiler/target. A change
    # here does not move the source mtime, so without this the stale IR is reused.
    if !isempty(compile_fingerprint)
        key_file = _compile_key_path(ir_file)
        if !isfile(key_file) || strip(read(key_file, String)) != compile_fingerprint
            return true
        end
    end

    # Compare modification times
    source_mtime = mtime(source_file)
    ir_mtime = mtime(ir_file)

    return source_mtime > ir_mtime
end

# =============================================================================
# AGGRESSIVE PROJECT-LEVEL CACHE (hash-based)
# =============================================================================

"""
Compute a content hash for the entire project: replibuild.toml + source files +
headers + git HEAD + the toolchain (`_toolchain_identity`). Returns a hex string
that changes when anything relevant changes, and only then: it is SHA-256, so it
is the same under every Julia release (see `_key_feed!`).

`toolchain` is a seam for tests; a build always takes `_toolchain_identity`.
"""
function compute_project_hash(config::RepliBuildConfig;
                              toolchain::AbstractString=_toolchain_identity(config))::String
    ctx = SHA.SHA256_CTX()
    # The tag names the shape of what a build leaves behind. v3: a Tier 1
    # build's promoted module carries the `__rb_` exported aliases
    # (`_alias_exports!`), which a wrap needs and a v2 build does not have;
    # bumping it rebuilds every project once rather than leaving a stale
    # module whose slices the Slicer must refuse.
    _key_feed!(ctx, "replibuild-project-hash-v3")

    # Hash the replibuild.toml content
    toml_path = config.config_file
    if isfile(toml_path)
        _key_feed!(ctx, read(toml_path))
    end

    # Hash all source files by content
    for src in get_source_files(config)
        if isfile(src)
            _key_feed!(ctx, read(src))
        end
    end

    # Hash include directory contents (header files), RECURSIVELY.
    #
    # `readdir` alone covered only the top level of each include dir, so editing
    # a header one directory down left the project hash unchanged and the build
    # reported "project unchanged" against modified sources. That is not an edge
    # case: `include/<lib>/<lib>.h` is the normal C layout — SUNDIALS generates
    # exactly `include/sundials/sundials_config.h` — so the headers most likely
    # to matter were the ones least likely to be seen.
    #
    # Cost is bounded in practice: a resolved dependency puts its whole clone on
    # the include path, and the largest in the Hub (curl) is ~4400 files with
    # 254 headers. `.git` is skipped because it never feeds the compiler, and
    # build/cache dirs because hashing our own outputs would make the hash
    # depend on the build it is supposed to gate.
    for inc_dir in get_include_dirs(config)
        isdir(inc_dir) || continue
        for (root, dirs, files) in walkdir(inc_dir)
            filter!(d -> !(d in (".git", "build", ".replibuild_cache")), dirs)
            sort!(dirs)                       # deterministic descent
            for f in sort(files)
                any(endswith(f, ext) for ext in (".h", ".hpp", ".hxx", ".hh")) || continue
                path = joinpath(root, f)
                isfile(path) || continue
                # Path as well as content: renaming a header, or swapping the
                # contents of two, changes what the compiler sees.
                _key_feed!(ctx, relpath(path, inc_dir))
                _key_feed!(ctx, read(path))
            end
        end
    end

    # Hash git HEAD of the project root (captures upstream repo state)
    project_root = config.project.root
    git_head = _get_git_head(project_root)
    if !isempty(git_head)
        _key_feed!(ctx, git_head)
    end

    # The toolchain. Without it, switching Julia left a C project "unchanged":
    # the build was skipped, and the next wrap re-sliced or re-served Tier 1
    # IR printed by the other libLLVM.
    _key_feed!(ctx, toolchain)

    return bytes2hex(SHA.digest!(ctx))
end

function _get_git_head(dir::String)::String
    try
        head_file = joinpath(dir, ".git", "HEAD")
        if isfile(head_file)
            head_content = strip(read(head_file, String))
            if startswith(head_content, "ref: ")
                ref_path = joinpath(dir, ".git", head_content[6:end])
                if isfile(ref_path)
                    return strip(read(ref_path, String))
                end
            end
            return head_content
        end
    catch
    end
    return ""
end

"""
Detect the host target triple from clang or Julia's Sys.MACHINE.
"""
function _detect_target_triple()::String
    # Try clang -dumpmachine first (most accurate for the actual compiler in use)
    try
        (output, exitcode) = BuildBridge.execute("clang", ["-dumpmachine"])
        if exitcode == 0
            triple = strip(output)
            if !isempty(triple) && contains(triple, '-')
                return triple
            end
        end
    catch
    end

    # Fallback: Julia's Sys.MACHINE (always available, matches the Julia build target)
    return Sys.MACHINE
end

_project_hash_path(config::RepliBuildConfig) = joinpath(get_cache_path(config), "project_hash")

"""
    is_project_cache_valid(config; hash = compute_project_hash(config)) -> Bool

Check if the project-level cache is valid. Returns `true` if all artifacts exist
and the content hash matches, meaning the build can be skipped entirely.

`hash` is the current project hash. `compile_project` passes the one it took
before the build read anything, and saves that same value when the build
succeeds (see there).
"""
function is_project_cache_valid(config::RepliBuildConfig;
                                hash::Union{Nothing,AbstractString}=nothing)::Bool
    if !is_cache_enabled(config)
        return false
    end

    hash_file = _project_hash_path(config)

    # Check hash file exists
    if !isfile(hash_file)
        return false
    end

    # Check all required artifacts exist
    output_dir = get_output_path(config)
    lib_name = get_library_name(config)
    library_path = joinpath(output_dir, lib_name)
    metadata_path = joinpath(output_dir, "compilation_metadata.json")

    if !isfile(library_path) || !isfile(metadata_path)
        return false
    end

    # Compare stored hash with current
    stored_hash = strip(read(hash_file, String))
    current_hash = hash === nothing ? compute_project_hash(config) : hash

    return stored_hash == current_hash
end

"""
    save_project_hash(config; hash = compute_project_hash(config))

Save the project hash after a successful build. `compile_project` passes the
hash it took before the build began, never one taken now: the build compiled
the files as they were then.
"""
function save_project_hash(config::RepliBuildConfig;
                           hash::AbstractString=compute_project_hash(config))
    mkpath(get_cache_path(config))
    write(_project_hash_path(config), hash)
end

# =============================================================================
# COMPILATION: C++ → LLVM IR
# =============================================================================

"""
Compile a single C++ file to LLVM IR.
Returns (ir_file_path, success, exitcode).

`compile_fingerprint` is the compile-config hash gating the per-file cache; the
batch path (`compile_to_ir`) computes it once and threads it in. When called
standalone (empty default) it is computed here so the fingerprint check still
applies — `_detect_target_triple` shells out, so the batch avoids paying that
per file by passing it down.
"""
# _c_bucket_sysroot() -> String
#
# The `--sysroot` the C bucket's compiler needs on Windows, or "" when none is
# needed or none can be found.
#
# Clang_unified_jll's clang is built for `x86_64-w64-windows-gnu` — the right
# triple, matching Julia's own mingw-w64 build — but it is a bare compiler
# artifact: it ships no libc headers and no mingw CRT. Its InstalledDir is a
# Julia artifact directory holding tools and nothing else, so the header search
# it derives from that prefix is empty and `#include <math.h>` fails on the
# first C file. A system clang finds those through its own InstalledDir; the JLL
# has no equivalent to find.
#
# Pointing it at the MSYS2 CLANG64 prefix supplies exactly the missing half and
# disturbs nothing else: JLL clang 18 still emits the IR, Julia's resident
# libLLVM still consumes it, so the version lock this bucket exists to enforce
# (see _clang_for_c_bucket) is untouched. The triple does not move either — it
# is already the mingw one.
#
# Discovery, in order:
#   1. REPLIBUILD_C_SYSROOT — an explicit override always wins.
#   2. The prefix of whatever `clang` is on PATH. Under the supported Windows
#      setup that is MSYS2 CLANG64, and following PATH means a non-default MSYS2
#      install needs no configuration at all.
#   3. MSYS2_ROOT (default C:/msys64) x {clang64, ucrt64, mingw64}.
#
# A candidate is accepted only if it actually holds include/math.h. Testing for
# the directory would not do: an MSYS2 install carries empty `ucrt64/` and
# `mingw64/` trees for environments that were never installed, and handing clang
# one of those trades "no sysroot" for "wrong sysroot" — the same failure with a
# worse error message.
const _C_SYSROOT = Ref{Union{Nothing,String}}(nothing)

function _c_bucket_sysroot()::String
    Sys.iswindows() || return ""
    cached = _C_SYSROOT[]
    cached === nothing || return cached

    _usable(p) = !isempty(p) && isfile(joinpath(p, "include", "math.h"))

    found = ""

    override = get(ENV, "REPLIBUILD_C_SYSROOT", "")
    if !isempty(override)
        if _usable(override)
            found = override
        else
            @warn "REPLIBUILD_C_SYSROOT is set but holds no include/math.h — ignoring it" REPLIBUILD_C_SYSROOT=override
        end
    end

    if isempty(found)
        path_clang = Sys.which("clang")
        if path_clang !== nothing
            prefix = dirname(dirname(abspath(path_clang)))
            _usable(prefix) && (found = prefix)
        end
    end

    if isempty(found)
        msys_root = get(ENV, "MSYS2_ROOT", "C:/msys64")
        for env in ("clang64", "ucrt64", "mingw64")
            cand = joinpath(msys_root, env)
            if _usable(cand)
                found = cand
                break
            end
        end
    end

    if isempty(found)
        @warn """
              No mingw sysroot found for the C bucket. Clang_unified_jll's clang ships no
              libc headers of its own, so compiling a .c file will fail on its first system
              #include. Install the MSYS2 CLANG64 environment, or point REPLIBUILD_C_SYSROOT
              at a prefix containing include/math.h."""
    end

    # Cached either way, including the empty result: the search is the same every
    # time within a session, and this keeps the warning above to one line.
    _C_SYSROOT[] = found
    return found
end

# _c_bucket_rtlib() -> String
#
# The `-rtlib` the C bucket's JLL clang must link with, or "" to leave its
# default alone.
#
# Giving the JLL clang a sysroot (above) fixes compiling and only half of
# linking. clang's mingw driver defaults to `-rtlib=libgcc`, so the link ends in
#
#     lld: error: unable to find library -lgcc_s
#     lld: error: unable to find library -lgcc
#
# against MSYS2 CLANG64, which is a compiler-rt/libc++/libunwind environment and
# ships no libgcc at all. MSYS2's own clang is configured with
# CLANG_DEFAULT_RTLIB=compiler-rt; the JLL is not, and cannot be told at build
# time, so it is told here.
#
# This is asked of the SYSROOT, not the platform: UCRT64 and MINGW64 are GCC
# environments where libgcc is present and correct, and forcing compiler-rt on
# them would break a link that works. So the probe looks for a libgcc to link
# against and only overrides when there is none.
const _C_RTLIB = Ref{Union{Nothing,String}}(nothing)

function _c_bucket_rtlib()::String
    Sys.iswindows() || return ""
    cached = _C_RTLIB[]
    cached === nothing || return cached

    rtlib = ""
    sysroot = _c_bucket_sysroot()
    if !isempty(sysroot)
        libdir = joinpath(sysroot, "lib")
        # libgcc.a sits directly in lib/ on some layouts and under
        # lib/gcc/<triple>/<version>/ on others, so this walks rather than
        # guessing a path.
        has_libgcc = false
        if isdir(libdir)
            for (_, _, files) in walkdir(libdir)
                if "libgcc.a" in files || "libgcc_s.a" in files
                    has_libgcc = true
                    break
                end
            end
        end
        has_libgcc || (rtlib = "compiler-rt")
    end

    _C_RTLIB[] = rtlib
    return rtlib
end

# _jll_clang() -> Union{Cmd,Nothing}
#
# Clang_unified_jll's clang, or `nothing` when the Clang package or that JLL
# cannot be loaded. `nothing` is the one case in which the C bucket uses the
# system clang (see _clang_for_c_bucket).
function _jll_clang()
    try
        Clang_mod = Base.require(Base.PkgId(
            Base.UUID("40e3b903-d033-50b4-a0cc-940c62c95e31"), "Clang"))
        isdefined(Clang_mod, :Clang_unified_jll) || return nothing
        return Clang_mod.Clang_unified_jll.clang()
    catch
        return nothing
    end
end

# _clang_for_c_bucket(compiler, cmd_args; jll = _jll_clang) -> (output, exitcode)
#
# Run `compiler` over `cmd_args`, routing the C BUCKET (`compiler == "clang"`)
# through Clang_unified_jll so the tool is version-locked to Julia's resident
# libLLVM. Anything else, and a JLL that cannot be loaded (`jll()` returns
# `nothing`), falls back to BuildBridge.execute, i.e. the system PATH.
#
# A JLL clang that loads but does not START is reported, not replaced:
# `(message, BuildBridge.SPAWN_FAILED)`. Falling back there ran the system clang
# over the C bucket's IR without a word, which is the version mismatch below with
# nothing to explain it. A binary the OS refuses to run would do that (Smart App
# Control refused Clang_unified_jll's pkgimage and libclang.dll on 2026-10-18), and
# so would a command line past Windows' 32,767 characters. `jll` exists for the
# tests, which hand it a clang that is absent or cannot start.
#
# The C bucket's invariant is that every tool touching its IR is version-matched:
# JLL clang emits it, Julia's libLLVM links and optimizes it. `create_library`
# used to shell straight to system `clang` for the final `.ll -> .so`, which
# quietly stepped off that path. JLL clang 18 legally emits `icmp` constant
# expressions; LLVM 19 removed them; a system clang 22 then rejects IR that this
# pipeline itself produced. Any C library containing a weak-symbol NULL check
# (`if (weak_fn != NULL)`) emits that form — zstd's tracing hooks were the first
# in the Hub to hit it (packages/zstd/GENERATOR-icmp-constexpr.md).
#
# LLVMEnvironment.resolve_tool(name, :c) is deliberately NOT used here. It looks
# for a parallel SYSTEM install (/usr/lib/llvm18/bin), and Arch's `llvm18`
# package ships LLVM tools with no clang in it at all — so it finds the
# directory, misses the binary, and silently returns the bare name, landing back
# on the system compiler. It is the right hook for llvm-link/opt, which that
# package does ship.
function _clang_for_c_bucket(compiler::String, cmd_args; jll::Function = _jll_clang)
    compiler == "clang" || return BuildBridge.execute(compiler, cmd_args)
    clang_cmd = jll()
    clang_cmd === nothing && return BuildBridge.execute(compiler, cmd_args)

    cmd = nothing
    try
        # On Windows the JLL compiler brings no libc of its own — see
        # _c_bucket_sysroot. This covers compiling AND linking: without a sysroot
        # the link step cannot find the mingw CRT and startup objects either. A
        # caller that already chose a sysroot keeps it.
        sysroot = any(a -> startswith(string(a), "--sysroot"), cmd_args) ? "" :
                  _c_bucket_sysroot()

        # -rtlib is a LINK option; clang warns it is unused on a compile-only
        # run, and that warning would print once per C file. These flags are
        # what make an invocation compile-only.
        is_link = !any(a -> string(a) in ("-S", "-c", "-emit-llvm", "-fsyntax-only"),
                       cmd_args)
        rtlib = (is_link && !any(a -> startswith(string(a), "-rtlib"), cmd_args)) ?
                _c_bucket_rtlib() : ""

        extra = String[]
        isempty(sysroot) || push!(extra, "--sysroot=$sysroot")
        isempty(rtlib)   || push!(extra, "-rtlib=$rtlib")
        cmd = ignorestatus(`$(clang_cmd) $extra $cmd_args`)

        # `Pipe()` plus a read after `run` deadlocks once the child writes more
        # than the OS pipe buffer (~64 KiB): the child blocks on write, the
        # parent blocks on the child's exit, and nobody drains. `IOBuffer`
        # through `pipeline` is drained while the process runs — the same shape
        # `BuildBridge._run_command_impl` already uses.
        out_buf = IOBuffer()
        err_buf = IOBuffer()
        process = run(pipeline(cmd, stdout = out_buf, stderr = err_buf))

        return (String(take!(out_buf)) * "\n" * String(take!(err_buf)),
                process.exitcode)
    catch e
        if cmd !== nothing && BuildBridge._is_spawn_failure(e)
            return ("The C bucket's clang (Clang_unified_jll) did not start, and the " *
                    "system clang is not used in its place: its IR need not match Julia's " *
                    "libLLVM. " * BuildBridge._spawn_failure_message(cmd, e),
                    BuildBridge.SPAWN_FAILED)
        end
        return BuildBridge.execute(compiler, cmd_args)
    end
end

function compile_single_to_ir(config::RepliBuildConfig, cpp_file::String,
                              compile_fingerprint::String="")
    # Generate IR file path
    build_dir = get_build_path(config)
    mkpath(build_dir)

    ir_file = _ir_output_path(build_dir, cpp_file)

    if isempty(compile_fingerprint) && is_cache_enabled(config)
        compile_fingerprint = compute_compile_fingerprint(config)
    end
    file_fp = _file_compile_fingerprint(compile_fingerprint, cpp_file)

    # Check if recompilation needed
    if !needs_recompile(cpp_file, ir_file, is_cache_enabled(config), file_fp)
        return (ir_file, true, 0)  # Cache hit
    end

    # The IR here is about to be replaced: it and its key go first, and only a
    # compile that exits 0 counts. Success used to be `isfile(ir_file)`. A
    # compiler that never ran (SPAWN_FAILED: a command line past Windows'
    # 32,767 characters, a JLL clang the OS refuses) deletes nothing, so the
    # last build's IR passed for this one's, was stamped with the new
    # fingerprint, and was linked; the build reported success, and every later
    # build with those flags took it from the cache. `GraphBuild._compile_one`
    # does the same.
    rm(ir_file; force=true)
    rm(_compile_key_path(ir_file); force=true)

    # Build compiler command
    # Ensure -g is always present for DWARF metadata extraction
    base_flags = get_compile_flags(config)
    if !("-g" in base_flags)
        base_flags = vcat(base_flags, ["-g"])
    end

    # Strip C++-only flags when compiling C files
    if endswith(cpp_file, ".c")
        base_flags = filter(f -> !startswith(f, "-std=c++") && !startswith(f, "-std=gnu++"), base_flags)
    end

    cmd_args = vcat(
        ["-S", "-emit-llvm"],  # Emit LLVM IR
        base_flags,
        _template_tu_compile_flags(cpp_file),
        ["-I$dir" for dir in get_include_dirs(config)],
        ["-D$k=$v" for (k, v) in config.compile.defines],
        ["-o", ir_file, cpp_file]
    )

    # The C BUCKET is a property of the package, not of the file extension. A C
    # package compiles with Julia's internal Clang so the IR matches Julia's LLVM,
    # which then links and optimizes it in-process (shared with create_library:
    # both ends must agree on the compiler — see the helper).
    #
    # A C++ package links through the SYSTEM llvm-link/opt, so its `.c` files must
    # come from the system clang too: one LLVM per module. Routing them to the JLL
    # clang 18 handed LLVM 23's llvm-link IR whose debug info it rejects
    # ("Function-local variables are not allowed in a DICompileUnit's global
    # variables list"), and it then drops ALL of that file's debug info with only
    # a warning. llama.cpp lost every ggml.c signature to "inferred" that way:
    # ggml_version wrapped as returning Cint. `_assert_debug_info_kept` makes that
    # loud should any other mix reach the linker.
    compiler = endswith(cpp_file, ".c") ? "clang" : "clang++"
    (output, exitcode) = config.wrap.language == :c ?
        _clang_for_c_bucket(compiler, cmd_args) :
        BuildBridge.execute(compiler, cmd_args)

    if !isempty(output) && exitcode != 0
        println("  $(basename(cpp_file)): $output")
    end

    success = exitcode == 0 && isfile(ir_file)
    if success
        # Stamp the fingerprint that produced this IR so a later flag/define/
        # include/compiler change invalidates it (not just a source-mtime change).
        write_compile_key(ir_file, file_fp)
    end
    return (ir_file, success, exitcode)
end

"""
Compile multiple C++ files to LLVM IR (with parallel support).
Returns vector of IR file paths.
"""
function compile_to_ir(config::RepliBuildConfig, cpp_files::Vector{String})
    # Compile to LLVM IR

    if isempty(cpp_files)
        @warn "No source files to compile"
        return String[]
    end

    # Check which files need compilation
    build_dir = get_build_path(config)
    mkpath(build_dir)

    # Compile-config fingerprint computed once for the whole batch (it shells out
    # to detect the target triple), then threaded into both the cache pre-check
    # below and each compile_single_to_ir call.
    compile_fingerprint = is_cache_enabled(config) ? compute_compile_fingerprint(config) : ""

    files_to_compile = String[]
    ir_files = String[]
    cached_files = 0

    # One IR path per source, and every path distinct. Two sources sharing an
    # output used to be silent all the way to the linker (see _ir_output_path);
    # now a repeated source is folded and anything else is a hard error rather
    # than an overwrite. `owner` also catches the same file arriving twice under
    # different spellings (./x.c vs x.c), which would double-link it.
    owner = Dict{String,String}()
    for cpp_file in cpp_files
        ir_file = _ir_output_path(build_dir, cpp_file)

        prev = get(owner, ir_file, "")
        if !isempty(prev)
            # `_canon_path`, as the IR name's directory key is: on Windows
            # `C:\p\a.c` and `c:/p/a.c` are one file and one IR name.
            if _canon_path(abspath(prev)) != _canon_path(abspath(cpp_file))
                error("""
                      Two different sources map to the same IR output:
                        $(prev)
                        $(cpp_file)
                        → $(ir_file)
                      This would overwrite one TU and double-link the other.""")
            end
            continue   # same file listed twice — fold it
        end
        owner[ir_file] = cpp_file

        if needs_recompile(cpp_file, ir_file, is_cache_enabled(config),
                           _file_compile_fingerprint(compile_fingerprint, cpp_file))
            push!(files_to_compile, cpp_file)
        else
            cached_files += 1
        end
        push!(ir_files, ir_file)
    end

    if isempty(files_to_compile)
        println("  compile: $(length(cpp_files)) cached")
        return ir_files
    end

    if cached_files > 0
        print("  compile: $(length(files_to_compile))/$(length(cpp_files)) files ($(cached_files) cached)")
    else
        print("  compile: $(length(files_to_compile)) files")
    end

    # Compile in parallel if enabled
    if is_parallel_enabled(config) && length(files_to_compile) > 1

        results = Vector{Tuple{String,Bool,Int}}(undef, length(files_to_compile))
        Threads.@threads for i in 1:length(files_to_compile)
            results[i] = compile_single_to_ir(config, files_to_compile[i], compile_fingerprint)
        end

        println()
        # Check for failures
        failures = [(file, res) for (file, res) in zip(files_to_compile, results) if !res[2]]
        if !isempty(failures)
            for (file, (_, _, exitcode)) in failures
                println("    FAIL: $(basename(file)) (exit $exitcode)")
            end
            error("Compilation failed for $(length(failures)) files")
        end
    else
        # Sequential compilation
        for cpp_file in files_to_compile
            (ir_file, success, exitcode) = compile_single_to_ir(config, cpp_file, compile_fingerprint)
            if !success
                println()
                error("Compilation failed for $cpp_file (exit: $exitcode)")
            end
        end
        println()
    end
    return ir_files
end

# =============================================================================
# LINKING: LLVM IR → Optimized IR
# =============================================================================

# Windows' CreateProcess accepts at most 32,767 characters of command line, and a
# link names every translation unit. llamacpp with ggml-vulkan has 384, whose IR
# paths came to ~39,000 characters from the Hub's checkout depth: llvm-link never
# started, and neither did the clang fallback after it. llvm-link and clang both
# read arguments from a response file (`@file`), so past this many characters the
# inputs go to one. Windows only: Linux's ARG_MAX is about 2 MB.
const _RSP_THRESHOLD = 30_000   # of 32,767: room for the program path and quoting

# What the command line costs: each argument plus a separator, and two quotes
# around one that holds a space.
_cmdline_chars(args) = sum(a -> length(a) + 1 + (occursin(' ', a) ? 2 : 0), args; init=0)

"""
    _args_with_response_file(head, inputs, tail, rsp_path;
                             windows = Sys.iswindows(), threshold = _RSP_THRESHOLD)

`vcat(head, inputs, tail)`, or, on Windows when that would be longer than
`threshold` characters, `vcat(head, ["@rsp_path"], tail)` with `inputs` written to
`rsp_path`. Only the inputs move: callers inspect the flags (`_clang_for_c_bucket`
looks for `-S`, `-c`, `--sysroot`, `-rtlib`).

The two readers tokenize differently. llvm-link, like any LLVM tool built for
Windows, reads a response file by Windows rules, where a backslash is literal. The
clang driver reads it by GNU rules unless it is clang-cl, and there a backslash
escapes the next character, so `C:\\Users\\x.ll` reaches it as `C:Usersx.ll`. One
double-quoted, `/`-separated path per line reads the same to both. An input with a
quote or a line break cannot be written that way, and is refused.
"""
function _args_with_response_file(head::Vector{String}, inputs::Vector{String},
                                  tail::Vector{String}, rsp_path::AbstractString;
                                  windows::Bool = Sys.iswindows(),
                                  threshold::Int = _RSP_THRESHOLD)
    args = vcat(head, inputs, tail)
    (windows && _cmdline_chars(args) > threshold) || return args
    for f in inputs
        any(c -> c in ('"', '\n', '\r'), f) &&
            error("Cannot pass $(repr(f)) to the linker in a response file: " *
                  "the path contains a quote or a line break.")
    end
    mkpath(dirname(rsp_path))
    open(rsp_path, "w") do io
        for f in inputs
            print(io, '"', replace(f, '\\' => '/'), "\"\n")
        end
    end
    return vcat(head, ["@" * rsp_path], tail)
end

"""
Link multiple LLVM IR files and optimize.
Returns linked IR path (String) or original IR files (Vector{String}) on llvm-link failure.
"""
function link_optimize_ir(config::RepliBuildConfig, ir_files::Vector{String}, output_name::String)
    # Link and optimize IR

    if isempty(ir_files)
        error("No IR files to link")
    end

    # Output path
    build_dir = get_build_path(config)
    linked_ir = joinpath(build_dir, output_name * "_linked.ll")

    # In-process vs external pipeline selection.
    #
    # For C, the IR was emitted by the JLL clang, which ships pinned to the same
    # LLVM version as Julia's resident libLLVM (both 18.1.7 on Julia 1.12). That
    # invariant — JLL clang version == Base.libllvm_version — holds across Julia
    # releases because they ship together, so there is no version skew and the
    # in-process libLLVM link/opt path is safe (no silent DWARF-record dropping).
    # It is therefore the default for C. A failure in the in-process path is a
    # hard error, not a silent downgrade, so regressions surface; set
    # `[link] fallback = true` in replibuild.toml to use the external
    # llvm-link/opt pipeline instead. C++ always uses the external pipeline.
    use_inprocess = config.wrap.language == :c && !config.link.fallback

    if use_inprocess
        if !_link_ir_libllvm(ir_files, linked_ir)
            error("In-process libLLVM link failed for C source. " *
                  "Set `[link] fallback = true` in replibuild.toml to use external llvm-link.")
        end
    else
        # C fallback must not pick PATH's LLVM 22 opt against JLL clang 20 IR —
        # that parse-fails inside create_library as "unterminated attribute
        # group" on `nocreateundeforpoison`, which does not name the versions.
        if config.wrap.language == :c && LLVMEnvironment.c_toolchain_bin_dir() === nothing
            major = Base.libllvm_version.major
            error("C `[link] fallback = true` needs llvm-link/opt at Julia's " *
                  "libLLVM $major, not a different major on PATH. Default C is " *
                  "in-process (`fallback = false`). To use the hatch, set " *
                  "REPLIBUILD_LLVM_C_BIN to a matching bin.")
        end
        llvm_link_tool = LLVMEnvironment.resolve_tool("llvm-link", config.wrap.language)
        link_args = _args_with_response_file(["-S"], ir_files, ["-o", linked_ir],
                                             joinpath(build_dir, output_name * "_link.rsp"))
        (output, exitcode) = BuildBridge.execute(llvm_link_tool, link_args)
        _assert_debug_info_kept(output, "llvm-link")

        if exitcode != 0 || !isfile(linked_ir)
            # llvm-link crashed or failed (common with very large IR modules).
            # Fall back: let clang handle linking at binary creation time. A tool
            # that never started did not fail, so that warning says what stopped it.
            if exitcode == BuildBridge.SPAWN_FAILED
                @warn "llvm-link was not run, falling back to clang-based linking: $output"
            else
                @warn "llvm-link failed (exit $exitcode), falling back to clang-based linking"
            end
            return ir_files
        end
    end

    # Optimize if requested
    opt_level = config.link.optimization_level
    if opt_level != "0"

        optimized_ir = joinpath(build_dir, output_name * "_opt.ll")

        if use_inprocess
            if !_optimize_ir_libllvm(linked_ir, optimized_ir, opt_level)
                error("In-process libLLVM opt (default<O$opt_level>) failed for C source. " *
                      "Set `[link] fallback = true` in replibuild.toml to use external opt.")
            end
            linked_ir = optimized_ir
        else
            opt_args = ["-S", "-O$opt_level", linked_ir, "-o", optimized_ir]
            opt_tool = LLVMEnvironment.resolve_tool("opt", config.wrap.language)
            (output, exitcode) = BuildBridge.execute(opt_tool, opt_args)

            if exitcode != 0
                @warn "Optimization failed, using unoptimized IR"
            elseif isfile(optimized_ir)
                linked_ir = optimized_ir
            end
        end
    end

    # Static promotion (llvmcall slicing contract): rename internal statics —
    # functions + mutable globals — to exported `__rb_<lib>_*` symbols, so
    # per-function slices can `declare` and bind to the .so's single copy of
    # every mutable datum. Must run BEFORE the LTO bitcode emission below so
    # the .so and the slice source are the same promoted module. In-process C
    # only; a failure is a hard error (no silent downgrade), matching the
    # link/opt steps. Always clear a stale map first so a knob flip or a
    # fallback build can't serve a previous build's map.
    promoted_map_path = joinpath(build_dir, "promoted_symbols.json")
    aliases_map_path = joinpath(build_dir, "exported_aliases.json")
    rm(promoted_map_path, force=true)
    rm(aliases_map_path, force=true)
    if use_inprocess && config.link.promote_statics
        lib_token = replace(output_name, r"[^A-Za-z0-9]" => "_")
        abi_ir = joinpath(build_dir, output_name * "_abi.ll")
        # With Tier 1 on, every exported definition also gets its `__rb_` alias,
        # the name a slice declares it by (see `_alias_exports!`). Off, the
        # module and the library's symbol table are exactly what they were.
        aliases = config.wrap.tier1.enable ? Dict{String,String}() : nothing
        promoted = _promote_statics_libllvm(linked_ir, abi_ir, lib_token; aliases)
        if promoted === nothing
            error("In-process static promotion failed for C source. " *
                  "Set `[link] promote_statics = false` to skip promotion, or " *
                  "`[link] fallback = true` for the external pipeline.")
        end
        linked_ir = abi_ir
        open(promoted_map_path, "w") do io
            JSON.print(io, promoted, 2)
        end
        if aliases !== nothing
            open(aliases_map_path, "w") do io
                JSON.print(io, aliases, 2)
            end
        end
    end

    # If LTO is enabled, the .so must export every internal symbol the JIT-loaded
    # bitcode references. Source-level `__attribute__((visibility("hidden")))`
    # (e.g. Lua's LUAI_DDEC / LUAI_FUNC) keeps internal globals/functions out of
    # the .so's dynamic symbol table — the JIT then can't resolve them when it
    # processes external references, OR (worse) defines its own private copy
    # that diverges from the .so's view, leading to runtime segfaults.
    # Strip hidden/protected visibility from globals and functions in the linked
    # IR before producing either the .so or the LTO bitcode. Single source of
    # truth: the .so as authoritative data store; the bitcode as inlinable code.
    if config.link.enable_lto
        linked_text = read(linked_ir, String)
        # Globals: `@name = hidden ...` → `@name = ...`
        linked_text = replace(linked_text,
            r"^(@[\w][\w.]*\s*=\s*)(?:hidden|protected)\s+"m => s"\1")
        # Functions: `define hidden ...` / `declare protected ...` → strip token
        linked_text = replace(linked_text,
            r"^((?:define|declare)\s+)(?:hidden|protected)\s+"m => s"\1")
        write(linked_ir, linked_text)
    end

    # If LTO is enabled, also emit a binary bitcode file (.bc)
    # for Julia's LLVM compiler to consume via llvmcall
    if config.link.enable_lto
        output_dir = get_output_path(config)
        mkpath(output_dir)

        lto_ir_text = read(linked_ir, String)
        lto_ir_text = sanitize_ir_for_julia(lto_ir_text)

        filtered_ll_path = joinpath(output_dir, "$(output_name)_lto.ll")
        write(filtered_ll_path, lto_ir_text)

        # Now compile the filtered IR to bitcode
        bitcode_file = joinpath(output_dir, "$(output_name)_lto.bc")
        assemble_bitcode(filtered_ll_path, bitcode_file)
    end

    return linked_ir
end

"""
    _ir_type_end(s, i) -> Union{Int,Nothing}

The byte index just past the LLVM type that starts at `s[i]`: the first space,
tab or comma at bracket depth 0, counting `[ ] { } < > ( )` and skipping quoted
names (`%"class.std::pair<int, int>"`). A type spells its elements with spaces
(`[5 x { i32, [4 x i8], ptr, ptr }]`, `[2 x [3 x i32]]`), so no token or
fixed list of shapes delimits it. `ptr addrspace(N)` is one type. `nothing`
when the brackets do not balance before the line ends.
"""
function _ir_type_end(s::AbstractString, i::Int)
    cu = codeunits(s)
    n = length(cu)
    depth = 0
    j = i
    while j <= n
        c = cu[j]
        if c == UInt8('"')
            q = findnext(==(UInt8('"')), cu, j + 1)
            q === nothing && return nothing
            j = q + 1
            continue
        elseif c in (UInt8('['), UInt8('{'), UInt8('<'), UInt8('('))
            depth += 1
        elseif c in (UInt8(']'), UInt8('}'), UInt8('>'), UInt8(')'))
            depth -= 1
            depth < 0 && return nothing
        elseif depth == 0 && c in (UInt8(' '), UInt8('\t'), UInt8(','))
            break
        end
        j += 1
    end
    depth == 0 || return nothing
    j > i || return nothing
    as = match(r"^\s+addrspace\(\d+\)", SubString(s, j))
    as === nothing || (j += ncodeunits(as.match))
    return j
end

const _IR_GLOBAL_HEAD = r"^(@[\w][\w.]*\s*=\s*)(.*?)(?<![@%.$])\b(global|constant)\b\s+"

"""
    _externalize_global_definition(line) -> String

`@g = <linkage…> global|constant <type> <initializer>…` becomes
`@g = external global|constant <type>`, so a module loaded beside its `.so`
declares the global instead of defining it twice. A line with nothing after
the type is a declaration already and passes through, and so does a hidden,
protected, internal or private definition: the `.so` exports no symbol for it.
A definition whose type cannot be delimited is kept, with a warning naming the
global, never rewritten into a truncated declaration (wasm3's
`@preopen = external global [5`, which no LLVM parses).
"""
function _externalize_global_definition(line::AbstractString)
    gm = match(_IR_GLOBAL_HEAD, line)
    gm === nothing && return line
    occursin(r"\b(?:private|internal|hidden|protected)\b", gm[2]) && return line
    start = gm.offset + ncodeunits(gm.match)
    stop = _ir_type_end(line, start)
    if stop === nothing
        name = strip(rstrip(strip(gm[1]), '='))
        @warn "IR sanitizer: cannot delimit the type of global $name; keeping its definition" maxlog = 20
        return line
    end
    # A definition carries an initializer after the type.
    occursin(r"^\s+\S", SubString(line, stop)) || return line
    type_str = SubString(line, start, prevind(line, stop))
    # What a declaration must keep to name the same object: thread-local
    # storage and the address space. Linkage, visibility, dllexport and
    # unnamed_addr belong to the definition.
    kept = String[]
    tl = match(r"\bthread_local(?:\(\w+\))?", gm[2])
    tl === nothing || push!(kept, tl.match)
    as = match(r"\baddrspace\(\d+\)", gm[2])
    as === nothing || push!(kept, as.match)
    quals = isempty(kept) ? "" : join(kept, " ") * " "
    return "$(gm[1])external $(quals)$(gm[3]) $(type_str)"
end

"""
    sanitize_ir_for_julia(ir_text::String) -> String

Sanitize LLVM IR text for compatibility with Julia's internal LLVM (18).
Strips LLVM 19–22 attributes/instructions, debug metadata, and converts varargs
function bodies to extern declarations (va_start/va_end can't be JIT-compiled).
Uses `inlinehint` (not `alwaysinline`) to avoid recursive inliner explosion on
large modules.

Coverage by LLVM version:
  19: GEP nuw, #dbg_* records, inrange(), captures(), dead_on_unwind,
      initializes(), allocptr, icmp samesign, range(), trunc nuw/nsw,
      zext/uitofp nneg
  20: GEP nusw, or disjoint, fptrunc/fpext fast-math flags, ptrtoaddr→ptrtoint
  21: (covered by 19 patterns — captures/dead_on_unwind/allocptr landed here)
  22: ptrtoaddr instruction

This is the single source of truth for IR compatibility — used by both the C source
LTO pipeline and the MLIR AOT thunks pipeline.
"""
function sanitize_ir_for_julia(ir_text::String)::String
    # ── Phase 1: Multiline patterns (require full-string context) ──────────
    # Rewrite attribute blocks: keep only inlinehint + target-cpu/features
    ir_text = replace(ir_text,
        r"^(attributes\s+#\d+\s*=\s*\{)[^}]*(\})"m => function(m)
            cpu_m  = match(r"\"target-cpu\"=\"([^\"]+)\"", m)
            feat_m = match(r"\"target-features\"=\"([^\"]+)\"", m)
            extras = String[]
            cpu_m  !== nothing && push!(extras, "\"target-cpu\"=\"$(cpu_m[1])\"")
            feat_m !== nothing && push!(extras, "\"target-features\"=\"$(feat_m[1])\"")
            inner = isempty(extras) ? "inlinehint" : "inlinehint " * join(extras, " ")
            hdr_end = findfirst('{', m)
            "$(m[1:hdr_end]) $inner }"
        end)
    # Remove varargs function bodies — va_start/va_end intrinsics can't be JIT-resolved.
    # Two strategies depending on linkage:
    #  * Default-visibility symbols → convert to extern declarations; calls route
    #    through the shared library at runtime.
    #  * Hidden/protected (e.g. Lua's LUAI_FUNC = visibility("internal")) → cannot
    #    become declarations because the .so doesn't export them. Replace the body
    #    with `unreachable` so the symbol remains defined for internal references
    #    but the va_start/va_end calls are gone. The wrapper only calls exported
    #    APIs, so these internal helpers are never actually invoked from Julia.
    ir_text = replace(ir_text,
        r"^(define\s[^\n]*\.\.\.[^\n]*\{)\n(.*?)\n(\})"ms => function(m)
            sig_match = match(r"define\s+(.*?)\s+(@\w+)\s*\(([^)]*)\)", m)
            if sig_match !== nothing
                if occursin(r"\b(?:hidden|protected)\b", sig_match[1])
                    return "$(m[1:findfirst('{', m)])\nentry:\n  unreachable\n}"
                end
                linkage = replace(sig_match[1], r"\b(internal|private)\s*" => "")
                "declare $(linkage) $(sig_match[2])($(sig_match[3]))"
            else
                m
            end
        end)

    # ── Phase 1.5: lifetime intrinsic arity upgrade ───────────────────────
    # System llvm-link (LLVM 22) drops the size argument from
    # `llvm.lifetime.start/end` intrinsics during the link step, producing the
    # 1-arg form that LLVM 22+ uses. Julia 1.12 ships LLVM 18, whose verifier
    # rejects the 1-arg form (it expects `i64 immarg, ptr nocapture`). Restore
    # the 2-arg form using `i64 -1` as the size — LLVM 18's documented sentinel
    # for "unknown/whole-allocation" lifetime markers.
    ir_text = replace(ir_text,
        r"(call\s+void\s+@llvm\.lifetime\.(?:start|end)\.p0\()(ptr\b)" =>
        s"\1i64 -1, \2")
    ir_text = replace(ir_text,
        r"(declare\s+void\s+@llvm\.lifetime\.(?:start|end)\.p0\()(ptr\s+(?:captures\(none\)|nocapture)?)" =>
        s"\1i64 immarg, ptr nocapture")

    # ── Phase 2: Single-pass line processing ───────────────────────────────
    # All per-line transformations and filtering in one streaming pass.
    # Avoids ~15 full-string allocations and 2 split/join cycles from the
    # original implementation. Per-line replace() on SubString only allocates
    # when a pattern actually matches, so most lines pass through cheaply.
    io = IOBuffer(sizehint=sizeof(ir_text))
    first_line = true
    for line in eachsplit(ir_text, '\n')
        # ── Skip lines ──
        # Debug metadata entries (numbered and named)
        if occursin(r"^![0-9]+\s*=\s*(distinct\s+)?!", line)
            continue
        end
        if occursin(r"^![a-zA-Z].*=\s*!\{", line)
            continue
        end
        # #dbg_* record lines (LLVM 19+ debug records)
        if occursin(r"^\s*#dbg_[a-z]+\(", line)
            continue
        end
        # noalias scope intrinsics — reference metadata we've already skipped
        if occursin(r"^\s*(tail\s+)?call void @llvm\.experimental\.noalias\.scope\.decl\(", line)
            continue
        end
        # @llvm.dbg.{declare,value,label} calls — reference numbered metadata
        # nodes we drop above; without the skip, llvm-as fails with
        # "use of undefined metadata !N". Their declarations elsewhere are
        # harmless (unused). Match both `call void @llvm.dbg.X(...)` and the
        # tail-call form.
        if occursin(r"^\s*(tail\s+)?call void @llvm\.dbg\.(?:declare|value|label|addr|assign)\(", line)
            continue
        end
        # va_start/va_end intrinsic declarations are kept: hidden vararg function
        # bodies survive the Phase-1 strip and need these intrinsics for codegen.

        # ── Per-line replacements ──
        # Every keyword strip below is anchored with `(?<![@%.$])` as well as
        # `\b`: `\b` holds between `@` and a letter, so a C function named
        # `range` or `captures` lost its name at every call (`call i32 @`).
        line = replace(line, r"(?<![@%.$])\bnoinline\b" => "")
        line = replace(line, r"(?<![@%.$])\boptnone\b" => "")
        # LLVM 19–22 instruction-level compatibility
        # GEP flags: nuw (19), nusw (20) — strip any combination
        line = replace(line, r"\bgetelementptr inbounds( nuw| nusw)+" => "getelementptr inbounds")
        line = replace(line, r"\bgetelementptr( nuw| nusw)+\b" => "getelementptr")
        line = replace(line, r"(?<![@%.$])\binrange\([^)]*\)\s*" => "")
        line = replace(line, r"(?<![@%.$])\bcaptures\([^)]*\)\s*" => "")
        line = replace(line, r"(?<![@%.$])\bdead_on_unwind\b\s*" => "")
        line = replace(line, r"(?<![@%.$])\binitializes\((?:\([^)]*\),?\s*)+\)\s*" => "")
        line = replace(line, r"(?<![@%.$])\ballocptr\b\s*" => "")
        line = replace(line, r"\bicmp samesign\b" => "icmp")
        line = replace(line, r"(?<![@%.$])\brange\([^)]*\)\s*" => "")
        line = replace(line, r"\btrunc (nuw |nsw )+" => "trunc ")
        line = replace(line, r"\b(uitofp|zext) nneg " => s"\1 ")
        # LLVM 20: disjoint flag on or instruction
        line = replace(line, r"\bor disjoint\b" => "or")
        # LLVM 20: fast-math flags on fptrunc/fpext (pre-18 only supported these on fadd/fmul etc.)
        line = replace(line, r"\bfptrunc\s+(?:(?:fast|nnan|ninf|nsz|arcp|contract|afn|reassoc)\s+)+" => "fptrunc ")
        line = replace(line, r"\bfpext\s+(?:(?:fast|nnan|ninf|nsz|arcp|contract|afn|reassoc)\s+)+" => "fpext ")
        # LLVM 22: ptrtoaddr (ptrtoint without provenance capture) — downgrade to ptrtoint
        line = replace(line, r"(?<![@%.$])\bptrtoaddr\b" => "ptrtoint")
        # Strip metadata references (!dbg, !tbaa, !llvm.loop, etc.)
        line = replace(line, r",?\s*![a-zA-Z_.]+\s+![0-9]+" => "")

        # ── Global variable externalization ──
        # Convert externally-visible global definitions to external declarations.
        # Prevents "Duplicate definition" JIT errors when .so is also loaded via dlopen.
        # EXCEPTION: hidden/protected visibility symbols are not exported by the .so
        # (e.g. Lua's LUAI_DDEC marks tables like luaP_opmodes as visibility("internal")),
        # so externalizing them would create unresolvable JIT references. Keep the
        # definition local to the bitcode — there's no .so symbol to clash with.
        line = _externalize_global_definition(line)

        # ── Write line ──
        if !first_line
            write(io, '\n')
        end
        first_line = false
        write(io, line)
    end

    return String(take!(io))
end

"""
Assemble LLVM IR text (.ll) to bitcode (.bc).
Uses Julia's own libLLVM via the C API (LLVMParseIRInContext + LLVMWriteBitcodeToFile)
to guarantee bitcode version compatibility with Base.llvmcall.
Falls back to Clang_unified_jll, then system llvm-as.
"""
function assemble_bitcode(ll_path::String, bc_path::String)
    # Primary: use Julia's own libLLVM directly — guaranteed version match
    if _assemble_bitcode_libllvm(ll_path, bc_path)
        return
    end

    # Fallback 1: Clang_unified_jll
    try
        Clang_mod = Base.require(Base.PkgId(Base.UUID("40e3b903-d033-50b4-a0cc-940c62c95e31"), "Clang"))
        if isdefined(Clang_mod, :Clang_unified_jll)
            clang_cmd = Clang_mod.Clang_unified_jll.clang()
            cmd = `$(clang_cmd) -Wno-override-module -O0 -x ir -emit-llvm -c $ll_path -o $bc_path`
            run(cmd)
            if isfile(bc_path)
                return
            end
        end
    catch e
        # Silently fallback
    end

    # Fallback 2: system llvm-as
    (bc_out, bc_exit) = BuildBridge.execute("llvm-as", [ll_path, "-o", bc_path])
    if bc_exit != 0
        @warn "Failed to assemble bitcode for LTO: $bc_out"
    end
end

# libLLVM path and handle cached at module level
const _LIBLLVM_PATH = Ref{String}("")
const _LIBLLVM_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

function _get_libllvm_path()
    if isempty(_LIBLLVM_PATH[])
        for lib in Libdl.dllist()
            if occursin("libLLVM", lib)
                _LIBLLVM_PATH[] = lib
                break
            end
        end
    end
    return _LIBLLVM_PATH[]
end

"""Return a cached dlopen handle for Julia's libLLVM. Opened once, never closed (process-lifetime)."""
function _get_libllvm_handle()
    if _LIBLLVM_HANDLE[] == C_NULL
        path = _get_libllvm_path()
        isempty(path) && return C_NULL
        _LIBLLVM_HANDLE[] = Libdl.dlopen(path)
    end
    return _LIBLLVM_HANDLE[]
end

"""
Parse LLVM IR text and write bitcode using Julia's own libLLVM C API.
Returns true on success, false on failure.
"""
function _assemble_bitcode_libllvm(ll_path::String, bc_path::String)::Bool
    h = _get_libllvm_handle()
    h == C_NULL && return false

    ir_text = read(ll_path, String)
    ctx = C_NULL
    mod_ptr = Ref{Ptr{Cvoid}}(C_NULL)
    msg_ptr = Ref{Ptr{UInt8}}(C_NULL)

    # GC.@preserve ir_text: LLVMCreateMemoryBufferWithMemoryRange borrows ir_text's
    # memory (RequiresNullTerminator=0, no copy). The buffer is consumed by
    # LLVMParseIRInContext, but the parser reads from ir_text's backing array
    # during parsing. Without @preserve, GC could collect ir_text mid-parse.
    GC.@preserve ir_text try
        ctx = ccall(Libdl.dlsym(h, :LLVMContextCreate), Ptr{Cvoid}, ())
        ctx == C_NULL && return false

        buf = ccall(Libdl.dlsym(h, :LLVMCreateMemoryBufferWithMemoryRange),
            Ptr{Cvoid},
            (Ptr{UInt8}, Csize_t, Cstring, Cint),
            ir_text, sizeof(ir_text), "lto_ir", 0)
        buf == C_NULL && return false

        # LLVMParseIRInContext takes ownership of buf — do not dispose it
        status = ccall(Libdl.dlsym(h, :LLVMParseIRInContext),
            Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{Ptr{Cvoid}}, Ref{Ptr{UInt8}}),
            ctx, buf, mod_ptr, msg_ptr)

        if status != 0
            if msg_ptr[] != C_NULL
                errmsg = unsafe_string(msg_ptr[])
                ccall(Libdl.dlsym(h, :LLVMDisposeMessage), Cvoid, (Ptr{UInt8},), msg_ptr[])
                @warn "libLLVM IR parse failed: $(first(errmsg, 200))"
            end
            return false
        end

        rc = ccall(Libdl.dlsym(h, :LLVMWriteBitcodeToFile),
            Cint,
            (Ptr{Cvoid}, Cstring),
            mod_ptr[], bc_path)

        if rc != 0
            @warn "libLLVM failed to write bitcode to $bc_path"
            return false
        end

        return isfile(bc_path)
    catch e
        @debug "libLLVM assemble_bitcode failed" exception=(e, catch_backtrace())
        return false
    finally
        if mod_ptr[] != C_NULL
            ccall(Libdl.dlsym(h, :LLVMDisposeModule), Cvoid, (Ptr{Cvoid},), mod_ptr[])
        end
        if ctx != C_NULL
            ccall(Libdl.dlsym(h, :LLVMContextDispose), Cvoid, (Ptr{Cvoid},), ctx)
        end
    end
end

"""
Link multiple `.ll` files in-process via LLVM.jl (`LLVM.link!` over Julia's
resident libLLVM) and write the merged module as text IR to `output_ll`.
Returns true on success, false on any failure (parse, link, write, exception)
so the caller can fall back to external `llvm-link`.

Each input is run through `sanitize_ir_for_julia` before parse, since system
clang may emit attributes/instructions newer than Julia's libLLVM accepts.
`LLVM.link!` consumes the source module — do not dispose it after linking.
"""
function _link_ir_libllvm(ir_files::Vector{String}, output_ll::String)::Bool
    isempty(ir_files) && return false
    try
        LLVM.Context() do _
            # Parse raw, debug-bearing IR. We deliberately do NOT call
            # sanitize_ir_for_julia here: stripping debug metadata pre-link
            # would propagate into the linked output and erase DWARF from the
            # resulting .so, breaking the wrapper generator. If libLLVM rejects
            # the IR (e.g. clang LLVM 22 features on Julia 1.12's libLLVM 18),
            # the parse throws, this function returns false, and the caller
            # falls back to external llvm-link which preserves debug info.
            dst = parse(LLVM.Module, read(first(ir_files), String))
            for path in ir_files[2:end]
                src = parse(LLVM.Module, read(path, String))
                LLVM.link!(dst, src)  # wraps LLVMLinkModules2 — src is consumed
            end
            open(output_ll, "w") do io
                print(io, string(dst))
            end
        end
        return isfile(output_ll)
    catch e
        @debug "LLVM.jl in-process link failed, falling back to external llvm-link" exception=(e, catch_backtrace())
        return false
    end
end

"""
Optimize a `.ll` file in-process via Julia's resident libLLVM, running the new
pass-manager `default<O\$opt_level>` pipeline, and write the optimized module to
`out_ll`. Returns true on success, false on any failure so the caller can decide
whether to hard-error (in-process mode) or fall back to external `opt`.

`opt_level` is the bare level token already stripped of any leading `O`
(`"1"`..`"3"`, `"s"`, `"z"`). Callers skip this for `"0"`. Debug info is
preserved by the default pipeline, so the optimized IR still carries DWARF for
the subsequent `.so` creation.
"""
function _optimize_ir_libllvm(in_ll::String, out_ll::String, opt_level::String)::Bool
    try
        LLVM.Context() do _
            mod = parse(LLVM.Module, read(in_ll, String))
            # Host TargetMachine supplies TTI so target-aware passes in the
            # default pipeline (vectorization cost models, etc.) run correctly.
            tm = LLVM.JITTargetMachine()
            LLVM.run!("default<O$opt_level>", mod, tm)
            open(out_ll, "w") do io
                print(io, string(mod))
            end
        end
        return isfile(out_ll)
    catch e
        @debug "LLVM.jl in-process opt failed" exception=(e, catch_backtrace())
        return false
    end
end

"""
    _promote_statics_libllvm(in_ll, out_ll, lib_token; aliases=nothing) -> Union{Nothing,Dict{String,String}}

Static-promotion pass for the llvmcall slicing contract: rename every
**function or mutable global** that is not already exportable from the `.so`'s
dynamic symbol table — internal/private linkage, OR external linkage with
hidden/protected visibility (e.g. Lua's `LUAI_FUNC` functions like `luaF_close`) —
to an exported `__rb_<lib_token>_<name>` symbol (external linkage, default
visibility). The `.so` built from the promoted module then carries exactly one
copy of every mutable datum and every non-exported function body, which
per-function bitcode slices bind to by `declare` — instead of embedding duplicate
definitions (the cJSON `static global_error` divergence class) or leaving a slice
`declare` unresolvable (the hidden-visibility JIT-deadlock class).

Internal **constant** globals (string literals, lookup tables) are deliberately
left internal: slices may embed constants safely, and promoting them would
bloat the dynamic symbol table for nothing. Constants that are external-but-
hidden are promoted anyway — the Slicer binds those by `declare`, not by
embedding, so they have to be dlsym-able.

With `aliases` a `Dict`, the pass also gives every symbol the module defines
AND exports — external linkage, default visibility: the public API — a second
exported name, the LLVM alias `__rb_<lib_token>_<name>`, and records
`public name => alias name` in it (`_alias_exports!`). A slice that calls another
public function of its own library declares that function, and ORC resolves a
`declare` process-wide, so declaring the public name binds whichever library
defined it first: Julia loads its own `libz.so.1` at startup, and the Hub's zlib
ran `gzclose` as a slice whose `gzclose_w` was Julia's. The alias is a name no
other library carries, so the Slicer declares it instead
(`Slicer._claim_rb_aliases!`). `link_optimize_ir` asks for the aliases only when
`[wrap.tier1] enable` is set: nothing else binds by `declare`, and a tier1-off
build keeps its dynamic symbol table, and so its `LIBRARY_SHA`, as before.

Runs post-optimization so internal linkage still fed IPO, on the exact module
that becomes both the `.so` and the slice source — one truth, bit-identical
code on both sides. Returns the old→new name map (possibly empty), or
`nothing` on failure so the caller can hard-error (in-process doctrine).
"""
function _promote_statics_libllvm(in_ll::String, out_ll::String, lib_token::String;
                                  aliases::Union{Nothing,Dict{String,String}} = nothing)
    local_linkages = (LLVM.API.LLVMInternalLinkage, LLVM.API.LLVMPrivateLinkage)
    hidden_viz     = (LLVM.API.LLVMHiddenVisibility, LLVM.API.LLVMProtectedVisibility)
    # A symbol is dlsym-bindable from a slice only if it reaches the .so's dynamic
    # symbol table: external linkage AND default visibility. So anything a slice may
    # `declare` that is NOT already exportable must be promoted:
    #   * internal/private linkage — static fns, file-local mutable state; OR
    #   * external linkage but hidden/protected visibility — e.g. Lua's LUAI_FUNC
    #     functions (luaF_close &c.), which the optimizer leaves as `define hidden`.
    # That second class is why gating de-hiding on `enable_lto` was wrong: the slicing
    # path runs with enable_lto=false, so those symbols stayed out of the dynsym and a
    # Tier-1 `llvmcall` referencing one deadlocked the JIT (`Symbols not found`).
    # Renaming (not just de-hiding in place) also keeps these internals OUT of the wrap
    # surface: DWARF still names the original, so the `__rb_*` symbol has no DWARF match
    # and `extract_symbols_from_binary` filters it.
    needs_promotion(gv) = LLVM.linkage(gv) in local_linkages || LLVM.visibility(gv) in hidden_viz
    try
        promoted = Dict{String,String}()
        LLVM.Context() do _
            mod = parse(LLVM.Module, read(in_ll, String))

            promote!(gv, old_name) = begin
                LLVM.linkage!(gv, LLVM.API.LLVMExternalLinkage)
                LLVM.visibility!(gv, LLVM.API.LLVMDefaultVisibility)  # into the .so's dynsym
                LLVM.name!(gv, "__rb_$(lib_token)_$(old_name)")
                # Read back: LLVM uniquifies on collision, the map must hold truth.
                promoted[old_name] = LLVM.name(gv)
            end

            for f in LLVM.functions(mod)
                LLVM.isdeclaration(f) && continue        # only definitions (never `declare` externs)
                needs_promotion(f) || continue
                old = LLVM.name(f)
                (isempty(old) || startswith(old, "llvm.") || startswith(old, "__rb_")) && continue
                promote!(f, old)
            end
            for gv in LLVM.globals(mod)
                LLVM.isdeclaration(gv) && continue       # only defined globals
                needs_promotion(gv) || continue
                # INTERNAL constants stay internal: no symbol exists, so the Slicer
                # embeds them, and read-only data has no divergence class. A constant
                # that is external-but-HIDDEN is the opposite case — the Slicer's
                # boundary policy reads "not local linkage" as "a symbol exists" and
                # binds it by `declare`, but hidden keeps it out of the dynsym. Lua's
                # LUAI_DDEF tables (`luaT_typenames_`) are exactly this, and left
                # unpromoted they cost lua_typename/luaL_checktype/luaL_tolstring/
                # luaL_typeerror their Tier-1 slices.
                #
                # `LLVMIsGlobalConstant`, not `LLVM.isconstant`: on LLVM.jl 10 that
                # asks whether a value is an LLVM Constant, which every global is,
                # so no internal mutable static would be promoted.
                (LLVM.API.LLVMIsGlobalConstant(gv) != 0 &&
                 LLVM.linkage(gv) in local_linkages) && continue
                old = LLVM.name(gv)
                (isempty(old) || startswith(old, "llvm.") || startswith(old, "__rb_")) && continue
                promote!(gv, old)
            end

            # After the renames, so a promoted `__rb_*` definition is never
            # aliased a second time and the alias set is exactly the public API.
            aliases === nothing || _alias_exports!(mod, lib_token, aliases)

            open(out_ll, "w") do io
                print(io, string(mod))
            end
        end
        return isfile(out_ll) ? promoted : nothing
    catch e
        @debug "LLVM.jl in-process static promotion failed" exception=(e, catch_backtrace())
        return nothing
    end
end

"""
    _alias_exports!(mod, lib_token, aliases)

Give every exported definition of `mod` — a function or global with external
linkage and default visibility, which is what reaches the dynamic symbol table
under its own name — the exported LLVM alias `__rb_<lib_token>_<name>`, and
record `name => alias name` in `aliases`. The alias is the same address under a
name only this library carries, so a slice can bind the library's own symbol
without sharing a namespace with every other library the process loaded first
(`Slicer._claim_rb_aliases!` turns the slice's `declare @name` into
`declare @__rb_<lib>_name`).

Skipped, and therefore declared by their public name as before: declarations
(the library does not define them, so there is nothing of its own to bind),
weak and `available_externally` definitions (a slice reaching a weak symbol
carries the `:weak` hazard and is demoted), `llvm.*` intrinsics, symbols already
promoted to `__rb_*`, and thread-local globals, which the C API cannot declare an
alias for. The Slicer then keeps a declaration's name, demotes a weak symbol
through its hazard, and refuses a thread-local global. A `dllexport`
definition gets a `dllexport` alias, so on PE the alias reaches the export
directory whenever its aliasee does; on ELF the storage class is a no-op.

The alias name is read back after creation: LLVM uniquifies on collision, and
the map must hold what the module actually says.
"""
function _alias_exports!(mod, lib_token::String, aliases::Dict{String,String})
    for gv in Iterators.flatten((LLVM.functions(mod), LLVM.globals(mod)))
        LLVM.isdeclaration(gv) && continue
        LLVM.linkage(gv) == LLVM.API.LLVMExternalLinkage || continue
        LLVM.visibility(gv) == LLVM.API.LLVMDefaultVisibility || continue
        name = LLVM.name(gv)
        (isempty(name) || startswith(name, "llvm.") || startswith(name, "__rb_")) && continue
        gv isa LLVM.GlobalVariable && LLVM.API.LLVMIsThreadLocal(gv) != 0 && continue
        addrspace = LLVM.API.LLVMGetPointerAddressSpace(LLVM.API.LLVMTypeOf(gv))
        alias = LLVM.API.LLVMAddAlias2(mod, LLVM.API.LLVMGlobalGetValueType(gv), addrspace,
                                       gv, "__rb_$(lib_token)_$(name)")
        LLVM.API.LLVMSetLinkage(alias, LLVM.API.LLVMExternalLinkage)
        LLVM.API.LLVMSetVisibility(alias, LLVM.API.LLVMDefaultVisibility)
        LLVM.API.LLVMSetDLLStorageClass(alias, LLVM.API.LLVMGetDLLStorageClass(gv))
        aliases[name] = _llvm_value_name(alias)
    end
    return aliases
end

"The name of a raw `LLVMValueRef` (an alias has no LLVM.jl wrapper type)."
function _llvm_value_name(ref::LLVM.API.LLVMValueRef)::String
    len = Ref{Csize_t}(0)
    p = LLVM.API.LLVMGetValueName2(ref, len)
    return p == C_NULL ? "" : unsafe_string(Ptr{UInt8}(p), len[])
end

# =============================================================================
# BINARY CREATION: IR → Shared Library / Executable
# =============================================================================

"""
    _pe_export_intent(ir_files) -> (; saw_shim, saw_foreign_export)

Who, if anyone, asked for an explicit PE export.

On PE a DLL keeps mingw's auto-export only while NOTHING is explicitly
exported; one `dllexport` anywhere switches the whole image to explicit-only.
`generate_macro_shims` emits `__declspec(dllexport)` on Windows — it has to, or
no shim reaches the export directory — which makes RepliBuild capable of
erasing the API of a library that exports nothing of its own. This is the
question that decides.

Read off the IR rather than guessed: `dllexport` is a linkage specifier LLVM
prints on the defining line, beside the symbol name, so one pass settles both
"is anything exported" and "is it only ours". Covers the LTO path too, where
every TU has already been merged into one module and a filename test would not.
"""
function _pe_export_intent(ir_files::Vector{String})
    saw_shim = false
    saw_foreign_export = false
    for f in ir_files
        isfile(f) || continue
        # Not `eachline(f)`: that closes the file only once exhausted, so the early
        # return below left it open, and on Windows an open handle blocks deletion.
        io = open(f)
        try
            for line in eachline(io)
                is_shim = occursin("replibuild_shim_", line)
                is_shim && (saw_shim = true)
                if occursin("dllexport", line) && !is_shim
                    saw_foreign_export = true
                end
                (saw_shim && saw_foreign_export) &&
                    return (; saw_shim, saw_foreign_export)
            end
        finally
            close(io)
        end
    end
    return (; saw_shim, saw_foreign_export)
end

"""
    _is_source_location_exhaustion(output) -> Bool

True when clang refused the translation unit because its `SourceLocation` space
ran out. Matched on the diagnostic rather than on a size threshold: the budget
is consumed by DEBUG LOCATIONS, not bytes, so two modules of the same size can
land either side of it depending on how much `-g` detail they carry. A
threshold would be a guess; this is the compiler's own answer.
"""
function _is_source_location_exhaustion(output::AbstractString)::Bool
    return occursin("ran out of source locations", output) ||
           occursin("translation unit is too large", output)
end

"""
    _assert_debug_info_kept(tool_output, stage)

Fail the build when an LLVM tool reports that it dropped a module's debug info.

LLVM verifies debug metadata when it reads IR, and on a failure it prints
`warning: ignoring invalid debug info in <file>`, strips every `!dbg` from that
module, and carries on with exit 0. DWARF is the source of truth for the wrapper,
so every function from that file then wraps from an inferred signature: wrong
types, no error. It happens when IR from one LLVM is read by a newer one with
stricter rules (JLL clang 18 IR into LLVM 23's llvm-link).
"""
function _assert_debug_info_kept(tool_output::AbstractString, stage::AbstractString)
    dropped = [strip(m.captures[1]) for m in eachmatch(r"ignoring invalid debug info in (.+)", tool_output)]
    isempty(dropped) && return nothing
    head = join(first(split(tool_output, '\n'), 6), "\n  ")
    error("""
          $stage dropped the debug info of $(length(dropped)) module(s):
            $(join(dropped, "\n            "))
          Their functions would wrap from inferred signatures (wrong types, no
          error). The IR was produced by a different LLVM than the one reading it;
          every translation unit of a package must come from one toolchain.
          $head""")
end

"""
    _codegen_opt_flags(config) -> Vector{String}

Optimization flags for turning IR into machine code (`create_library` and its
per-TU fallback). The IR is already optimized — per TU by `[compile] flags`, then
linked and run through opt at `[link] optimization_level` — so clang gets that
same `-O` for instruction selection and register allocation, with
`-disable-llvm-passes` so the IR pipeline is not run a second time.

With no `-O` at all, clang compiled optimized IR with the `-O0` backend: FastISel
and the fast register allocator, i.e. spill-heavy code in every shipped library,
and on LLVM 23 a backend crash ("Cannot emit physreg copy instruction") in
llama.cpp's `llama_kv_cache::find_slot` on Windows.
"""
_codegen_opt_flags(config::RepliBuildConfig) =
    ["-O$(config.link.optimization_level)", "-Xclang", "-disable-llvm-passes"]

"""
    _reproducible_link_flags(; windows = Sys.iswindows()) -> Vector{String}

Linker flags under which the same inputs link to the same bytes. Every link of
something RepliBuild ships passes them: `create_library`, `create_executable` and
the AOT thunks library.

A PE image holds its link time twice, in the COFF file header and in the debug
directory, which lld writes whenever the image has debug info (its `.buildid`
record). lld fills both from the clock, so every relink of unchanged IR gave a
different DLL, and with it a different `LIBRARY_SHA`: the wrapper of a rebuilt,
unchanged package differed from the last one in that line alone.
`--no-insert-timestamp` writes 0 there, in lld's MinGW driver and in GNU ld's PE
port (lld-link's `/Brepro` writes a hash of the image instead, and GNU ld has no
such option). The build ID itself is a hash of the image without those fields, so
it already repeated.

ELF has no link time. A build ID, when the linker writes one, is a hash of the
output, so it repeats too.
"""
_reproducible_link_flags(; windows::Bool = Sys.iswindows()) =
    windows ? ["-Wl,--no-insert-timestamp"] : String[]

"""
    _self_binding_link_flags(; windows = Sys.iswindows()) -> Vector{String}

Linker flags that bind a library's references to its own exported symbols to
its own definitions. Every library `create_library` links passes them.

On ELF a library's call to its own exported function goes through the PLT, and
a load of its own exported global through the GOT, and the dynamic linker
resolves both through the process: the global scope first, then the library.
So a library loaded earlier that exports the same name takes the reference
(symbol interposition). Julia loads its own `libz.so.1` at startup, and the
Hub's zlib called Julia's `gzclose_w`, `deflate`, `inflate` and `deflateEnd`
from its own code (`LD_DEBUG=bindings` names the file each symbol bound to):
its `gzclose` handed Julia's zlib a `gz_state` the other zlib laid out, and the
process died of heap corruption. `-Bsymbolic` sets `DF_SYMBOLIC`, under which
the dynamic linker searches the library itself first, so the wrapped library
is one self-contained copy of its code and state, the same copy a Tier 1 slice
binds through the `__rb_` aliases and the wrapper reaches through `dlsym` and
`cglobal`. Nothing RepliBuild builds is linked against by an executable, so
there is no copy relocation for the data half of `-Bsymbolic` to split.

A PE image has no such thing to prevent: a DLL's own calls and loads never go
through its import table, so Windows binds a DLL to itself already.
"""
_self_binding_link_flags(; windows::Bool = Sys.iswindows()) =
    windows ? String[] : ["-Wl,-Bsymbolic"]

"""
    _compile_ir_to_objects(config, ir_files, compiler) -> Vector{String}

Compile each IR module to its own object file, in parallel. Each `clang -c` is
a separate process with a fresh `SourceManager`, which is the entire point —
the source-location budget resets per file.

Objects land beside the IR under `build/obj/` and are reused when newer than
their input, so a re-link after an unrelated change does not recompile all of
them. A failure names the file rather than the whole batch.
"""
function _compile_ir_to_objects(config::RepliBuildConfig, ir_files::Vector{String},
                                compiler::String)::Vector{String}
    obj_dir = joinpath(get_build_path(config), "obj")
    mkpath(obj_dir)

    objs = [joinpath(obj_dir, string(basename(f), ".o")) for f in ir_files]
    todo = [i for i in eachindex(ir_files)
            if !isfile(objs[i]) || mtime(objs[i]) < mtime(ir_files[i])]

    if isempty(todo)
        println("  objects: $(length(objs)) up to date")
        return objs
    end
    println("  objects: compiling $(length(todo)) of $(length(objs)) " *
            "(reusing $(length(objs) - length(todo)))")

    failures = Tuple{String,String}[]
    lk = ReentrantLock()
    done = Threads.Atomic{Int}(0)
    # One process per input, bounded by the thread count. These are external
    # processes, so the concurrency that matters is how many clangs are resident
    # at once — each carries a whole module's debug info.
    asyncmap(todo; ntasks=max(1, Sys.CPU_THREADS)) do i
        (out, code) = _clang_for_c_bucket(compiler, vcat(["-c", "-fPIC"], _codegen_opt_flags(config),
                                                         [ir_files[i], "-o", objs[i]]))
        n = Threads.atomic_add!(done, 1) + 1
        if code != 0
            lock(lk) do
                push!(failures, (ir_files[i], out))
            end
        elseif n % 100 == 0 || n == length(todo)
            println("  objects: $n/$(length(todo))")
        end
        nothing
    end

    if !isempty(failures)
        (f, out) = first(failures)
        error("""
        Per-TU object compilation failed for $(length(failures)) of $(length(todo)) file(s).
        First failure: $f
        $(join(last(split(out, '\n'), 20), "\n"))""")
    end
    return objs
end

# =============================================================================
# libc++ MEMBERS THE STL PATH BINDS MUST BE EXPORTED
# -----------------------------------------------------------------------------
# libc++ 23 opens every `std` namespace block with
#
#   #pragma clang attribute _LibcxxExplicitABIAnnotations.push(
#     __attribute__((__exclude_from_explicit_instantiation__,
#                    __visibility__("hidden"), __abi_tag__(...))), apply_to = function)
#
# (`__configuration/namespace.h`). Every libc++ function without an explicit
# annotation is therefore hidden: on COFF too, where the per-platform
# `_LIBCPP_HIDDEN` is empty, and with no macro to opt out
# (`_LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS` does not reach the pragma). On MinGW
# clang turns hidden into a `-exclude-symbols:` linker directive, so
# `std::vector<int>::vector()` was a `T` in nm and absent from the DLL's export
# directory. The STL thunks could not bind it, the JIT module failed, and every
# Tier-2 function of the library went with it. stl_test: 513 exports and 28/28
# on LLVM 22; 222 exports on LLVM 23, every STL call "Symbol not found".
#
# So, for packages that bind STL members, hidden libc++ definitions are made
# default visibility in the IR before codegen, which is the export table LLVM 22
# produced. Only `define` lines of `std::__N` names change: a package's own
# `-fvisibility=hidden`, clang's helpers (`__clang_call_terminate`) and
# function-local entities (`_ZZ…`) keep their visibility.
# =============================================================================

"A hidden function definition in libc++'s versioned namespace (`std::__1`, `__2`, …)."
const _LIBCXX_HIDDEN_DEFINE = r"^define ((?:\w+ )*?)hidden ([^@]*@_ZN[rVK]*[RO]?St3__[0-9])"

_maybe_hidden_define(line::AbstractString) =
    startswith(line, "define ") && occursin("hidden", line)

"True if the IR file holds at least one hidden libc++ definition. Stops at the first."
function _has_hidden_libcxx(path::String)
    open(path) do io
        for line in eachline(io)
            _maybe_hidden_define(line) && occursin(_LIBCXX_HIDDEN_DEFINE, line) && return true
        end
        return false
    end
end

"""
    _unhide_libcxx_ir(files, out_dir) -> (files_out, n)

Copies of the IR `files` with every hidden libc++ function definition made default
visibility, written to `out_dir`, and how many definitions changed. A file with
none is returned as itself and not copied. A copy at least as new as its input is
reused, which keeps `_compile_ir_to_objects`'s mtime reuse working on the per-TU
fallback. Every other line is copied byte for byte.
"""
function _unhide_libcxx_ir(files::Vector{String}, out_dir::String)
    out = String[]
    total = 0
    for f in files
        dst = joinpath(out_dir, basename(f))
        if isfile(dst) && mtime(dst) >= mtime(f)
            push!(out, dst)
            continue
        end
        if !_has_hidden_libcxx(f)
            push!(out, f)
            continue
        end
        mkpath(out_dir)
        n = 0
        open(f) do src
            open(dst, "w") do io
                for line in eachline(src; keep=true)
                    if _maybe_hidden_define(line)
                        rewritten = replace(line, _LIBCXX_HIDDEN_DEFINE => s"define \1\2")
                        rewritten == line || (n += 1)
                        line = rewritten
                    end
                    write(io, line)
                end
            end
        end
        total += n
        push!(out, dst)
    end
    return (out, total)
end

"""
    _stl_exportable_ir(config, files) -> Vector{String}

`files`, with libc++'s hidden members made exportable when the package binds STL
members, i.e. C++ with `[types] templates`. Nothing else asks a thunk to call
libc++ by name, so every other package keeps the export table it had.
"""
function _stl_exportable_ir(config::RepliBuildConfig, files::Vector{String})
    (config.wrap.language == :c || isempty(config.types.templates)) && return files
    (out, n) = _unhide_libcxx_ir(files, joinpath(get_build_path(config), "libcxx_exported"))
    n > 0 && println("  link: $n libc++ member(s) made exportable for [types] templates")
    return out
end

"""
Create shared library from LLVM IR. Returns path to the library file.

`per_tu_ir` is the UNMERGED per-translation-unit IR list, used only if clang
refuses the (usually merged) `ir_files` for running out of source locations —
see the fallback inside. Callers that already hold that list should pass it;
without it the fallback cannot run, because splitting a merged module is not
possible after the fact.

`extra_link_args` go on the link line after `[link] link_libraries`, on every
attempt. The graph build passes a target's own link inputs there (absolute
library paths and linker flags the build system resolved, which `-l` cannot
spell), its relative rpaths and, on Windows, its import library.
"""
function create_library(config::RepliBuildConfig, ir_files::Union{String,Vector{String}},
                        lib_name::String=""; per_tu_ir::Vector{String}=String[],
                        extra_link_args::Vector{String}=String[])
    # Create shared library

    # Normalize to vector
    files = ir_files isa String ? [ir_files] : ir_files
    for f in files
        if !isfile(f)
            error("IR file not found: $f")
        end
    end

    # The last point before codegen turns libc++'s hidden visibility into
    # `-exclude-symbols:` (see _LIBCXX_HIDDEN_DEFINE).
    files = _stl_exportable_ir(config, files)

    # The forced inline constructors (`generate_inline_constructors`) are inline
    # members, hidden under `-fvisibility=hidden` like any other; they must
    # reach the dynamic table (ELF) or the export directory (PE). On PE the
    # export intent is read BEFORE the rewrite adds our own `dllexport`, so a
    # library that relied on auto-export keeps it (below).
    forced = _forced_constructor_symbols(config)
    pe_intent = (Sys.iswindows() && (!isempty(config.wrap.macros) || !isempty(forced))) ?
                _pe_export_intent(files) : nothing
    if !isempty(forced)
        (files, n) = _forced_exportable_ir(files, forced, joinpath(get_build_path(config), "ctors_exported");
                                           pe = host_record_abi() == :win64)
        n > 0 && println("  link: $n forced inline definition(s) made exportable")
    end

    # Determine library name
    if isempty(lib_name)
        lib_name = get_library_name(config)
    end

    # Output path
    output_dir = get_output_path(config)
    mkpath(output_dir)
    lib_path = joinpath(output_dir, lib_name)

    # Compile IR to shared library.
    #
    # `cmd_tail` is everything after the inputs, kept separate because the
    # inputs can be swapped for per-TU objects on the source-location fallback
    # below, or for a response file, and every other flag must be identical
    # across both attempts.
    cmd_head = [
        "-shared",  # Create shared library
        "-fPIC",    # Position independent code
        _codegen_opt_flags(config)...,   # codegen at the link level, IR passes off
        _reproducible_link_flags()...,   # same IR, same bytes: no PE link time
        _self_binding_link_flags()...,   # its own symbols are its own, whatever loaded first
        _bundled_linker_flags(config)...,  # the C bucket links with the bundled LLD
    ]
    cmd_tail = ["-o", lib_path]

    # A dllexport ANYWHERE turns mingw's auto-export off for the WHOLE image,
    # and the macro-shim TU carries one on Windows now. For a library that
    # exports nothing explicitly and relies on that default, ours alone would
    # therefore ERASE its API: measured on a two-function probe DLL, the export
    # table went from {lib_a, lib_b, shim} to {shim}. `--export-all-symbols` is
    # what the auto-export default expands to, so asking for it back reproduces
    # exactly the set the link would have produced anyway — the CRT stays out,
    # excluded by ld's own list, which is the property _pe_exported_names
    # depends on.
    #
    # Only when WE are the ones introducing the dllexport. A library that
    # already exports explicitly (pcre2, via PCRE2_EXP_DECL) keeps the surface
    # it chose and the shims simply join it; forcing the flag there would
    # publish every internal symbol instead.
    #
    # Gated on `wrap.macros` before the scan, not just on Windows:
    # `_pe_export_intent` reads every line of every IR file, and with no macros
    # configured there is no shim to find — 36 MB of pcre2 IR walked on every
    # link to conclude nothing. `generate_macro_shims` emits RB_SHIM_EXPORT
    # functions only for macros, so the two cannot disagree about whether a shim
    # exists. Its file with `[wrap] shim_defines` alone exports nothing itself,
    # and the auto-export default publishes the implementation it compiles.
    #
    # The forced inline constructors are ours too (`_forced_exportable_ir`
    # marks them `dllexport` on PE), so they ask for the flag on the same
    # terms as the shims.
    if pe_intent !== nothing && (pe_intent.saw_shim || !isempty(forced)) && !pe_intent.saw_foreign_export
        push!(cmd_tail, "-Wl,--export-all-symbols")
    end

    # Add library search paths — AND an rpath for each.
    #
    # `-L` alone answers "where do I find this at LINK time" and says nothing
    # about load time. For a distro library on the default search path that gap
    # never shows; for a library built by another Hub package it is the whole
    # problem, and it fails in the worst possible way — silently, against the
    # wrong copy.
    #
    # Measured: link a .so against packages/glfw/julia/libglfw.so with
    # `-L<dir> -lglfw` and the recorded DT_NEEDED is the bare `libglfw.so`
    # (RepliBuild's output carries no SONAME), so `ldd` resolves it to
    # /usr/lib/libglfw.so — the distro copy, not the one just linked. The build
    # succeeds, the symbols match, and the process ends up with two copies of a
    # library whose state is process-global.
    #
    # So every directory named for linking is also named for loading. `$ORIGIN`
    # comes first so a vendored `lib/` directory (the examples/ layout) resolves
    # siblings without an absolute path; RepliBuild already bakes absolute paths
    # into the wrapper's LIBRARY_PATH, so the rest are consistent with that.
    if !isempty(config.link.link_dirs) && !Sys.iswindows()
        push!(cmd_tail, "-Wl,-rpath,\$ORIGIN")
    end
    for dir in config.link.link_dirs
        push!(cmd_tail, "-L$dir")
        # PE has no RUNPATH; mingw's ld warns and ignores -rpath.
        Sys.iswindows() || push!(cmd_tail, "-Wl,-rpath,$(abspath(dir))")
    end

    # Add link libraries
    for lib in config.link.link_libraries
        push!(cmd_tail, "-l$lib")
    end
    append!(cmd_tail, extra_link_args)

    compiler = config.wrap.language == :c ? "clang" : "clang++"
    # `files` is one merged module unless llvm-link declined to merge, and then it
    # is every translation unit: a response file past the command-line limit.
    rsp_path = joinpath(get_build_path(config), lib_name * ".rsp")
    cmd_args = _args_with_response_file(cmd_head, files, cmd_tail, rsp_path)
    (output, exitcode) = _clang_for_c_bucket(compiler, cmd_args)
    _assert_debug_info_kept(output, "clang (IR -> library)")

    # ── Source-location fallback ─────────────────────────────────────────────
    # Clang re-parses the linked module as ONE translation unit and allocates a
    # `SourceLocation` per debug location. That counter is a 32-bit offset, so a
    # merged `-g -fstandalone-debug` module big enough exhausts it:
    #
    #   fatal error: translation unit is too large for Clang to process:
    #   ran out of source locations
    #
    # Hub onednn hits it at 3.9 GB of linked IR / 21.4M metadata records; the
    # same path ships llamacpp at 519 MB. It is NOT a parallelism limit — the
    # 779 TUs already compiled to IR across every thread — it is one process's
    # budget for source locations.
    #
    # So give each input its own process, which resets that budget: `clang -c`
    # per IR file, then link the objects. Passing the IR VECTOR to one
    # `clang -shared` is not a fix and must not be mistaken for one — that is
    # what the call above already does when llvm-link declines to merge, and it
    # is still a single SourceManager.
    #
    # Debug info is kept. Stripping it to make the module fit would produce a
    # .so the wrapper cannot read, which is the wrong trade at any size.
    if exitcode != 0 && _is_source_location_exhaustion(output)
        # Prefer the unmerged per-TU list when the caller supplied one: `files`
        # here is usually the single merged module, and splitting THAT is not
        # possible — the whole point is to never hand clang the merge.
        split_inputs = _stl_exportable_ir(config, !isempty(per_tu_ir) ? per_tu_ir : files)
        if length(split_inputs) <= 1
            error("""
            Library creation failed: clang ran out of source locations, and there
            is no per-translation-unit IR to fall back to.

            The merged module is too large for one clang process to parse, and
            splitting requires the unmerged .ll files. compile_project passes
            them as `per_tu_ir`; a direct create_library call must do the same.
            $output""")
        end
        @info "link: clang ran out of source locations on the merged module — " *
              "recompiling $(length(split_inputs)) translation unit(s) separately"
        objs = _compile_ir_to_objects(config, split_inputs, compiler)
        cmd_args = _args_with_response_file(cmd_head, objs, cmd_tail, rsp_path)
        (output, exitcode) = _clang_for_c_bucket(compiler, cmd_args)
    end

    if exitcode != 0
        error("Library creation failed: $output")
    end

    if !isfile(lib_path)
        error("Library file not created: $lib_path")
    end

    _assert_library_resolves(lib_path; preload = _link_dir_dlls(config))

    size_mb = round(filesize(lib_path) / 1024 / 1024, digits=2)
    println("  link: $lib_name ($size_mb MB)")

    return lib_path
end

"""
    _assert_library_resolves(lib_path)

Fail the BUILD if the freshly linked `.so` has a symbol nothing can resolve.

`ld -shared` does not require every reference to be satisfied — that is what
makes `DT_NEEDED` linking work — so a missing translation unit produces a
library that links clean, reports success, and wraps normally. It then dies
much later and much further from the cause. Generated wrappers dlopen with
`RTLD_LAZY`, so the failure does not even surface at wrapper load: it surfaces
at the first call to whichever function happened to need the missing symbol.

Found on llama.cpp, where excluding `ggml-backend-dl.cpp` (its name implies a
`GGML_BACKEND_DL` guard that does not exist) left `dl_load_library`,
`dl_get_sym` and `dl_error` undefined. The build printed a 47 MB library and a
success line.

`RTLD_NOW` binds everything eagerly, so the loader itself does the check and
names the first offending symbol. `RTLD_LOCAL` + `dlclose` keeps the probe from
leaking the library into this process's namespace — the same scoping discipline
the Tier-1 slice pre-flight uses.
"""
function _assert_library_resolves(lib_path::AbstractString; preload::Vector{String}=String[])
    # `preload`: the link-dir DLLs a wrapper opens first on Windows
    # (`_link_dir_dlls`); the probe has to see the process the wrapper will.
    deps = Ptr{Cvoid}[]
    try
        for p in preload
            push!(deps, Libdl.dlopen(p, Libdl.RTLD_NOW | Libdl.RTLD_GLOBAL))
        end
        handle = try
            Libdl.dlopen(lib_path, Libdl.RTLD_NOW | Libdl.RTLD_LOCAL)
        catch e
            error(_load_failure_message(lib_path, sprint(showerror, e)))
        end
        Libdl.dlclose(handle)
    finally
        foreach(Libdl.dlclose, deps)
    end
    return nothing
end

"""
    _load_failure_message(lib_path, loader_msg) -> String

Why the freshly linked library would not load, from the loader's own message.
Unresolved symbols are the usual cause, but two Windows failures look the same
from here and need different fixes: an App Control policy (Smart App Control)
refusing a new unsigned DLL, and a DLL it imports that the loader cannot find.
Reporting either as "unresolved symbols" sent people to their TOML.
"""
function _load_failure_message(lib_path::AbstractString, loader_msg::AbstractString)::String
    name = basename(lib_path)
    if occursin("Application Control policy has blocked", loader_msg)
        return """
        Windows refused to load the library it just built: $name
          $loader_msg
        Smart App Control, or another App Control policy, blocks new unsigned
        binaries, and every library RepliBuild builds is one. Event 3077 in Event
        Viewer (Applications and Services Logs > Microsoft > Windows > CodeIntegrity
        > Operational) names the file. RepliBuild cannot sign what it builds;
        whether to change the policy is your decision."""
    end
    if occursin("The specified module could not be found", loader_msg)
        return """
        A library that $name imports could not be found:
          $loader_msg
        Windows looks for it beside the library and on PATH. A C++ library imports
        libc++.dll from C:\\msys64\\clang64\\bin, which must be on PATH; a
        [link] link_libraries DLL in [link] link_dirs is opened first by itself."""
    end
    return """
    Linked library has unresolved symbols: $name
      $loader_msg
    The link step succeeded — `ld -shared` permits undefined symbols —
    but nothing can supply these at load time. Usually a source file
    excluded in replibuild.toml that something still calls, or a
    missing entry in [link] link_libraries."""
end

"""
    _link_dir_dlls(config) -> Vector{String}

On Windows, the DLLs `[link] link_libraries` names inside `[link] link_dirs`, as
absolute paths (relative dirs against the project root); empty elsewhere.

PE has no RUNPATH. `-L../glfw/julia -lglfw` links imgui against the Hub glfw's
`libglfw.dll`, and then the loader cannot find it: "The specified module could
not be found". Windows does bind an import to a module already loaded under that
name, so opening these by absolute path before the library is the PE counterpart
of the rpath ELF gets, and keeps one copy of process-global state like glfw's.
"""
function _link_dir_dlls(config::RepliBuildConfig)::Vector{String}
    Sys.iswindows() || return String[]
    found = String[]
    for dir in config.link.link_dirs
        d = isabspath(dir) ? dir : joinpath(config.project.root, dir)
        for lib in config.link.link_libraries
            for cand in ("lib$lib.dll", "$lib.dll")
                p = normpath(joinpath(d, cand))
                if isfile(p)
                    push!(found, p)
                    break
                end
            end
        end
    end
    return unique(found)
end

"""
Create executable from LLVM IR.
Returns path to executable file.
"""
function create_executable(config::RepliBuildConfig, ir_files::Union{String,Vector{String}}, exe_name::String,
                          link_libraries::Vector{String}=String[],
                          lib_dirs::Vector{String}=String[])
    # Create executable

    files = ir_files isa String ? [ir_files] : ir_files
    for f in files
        if !isfile(f)
            error("IR file not found: $f")
        end
    end

    # Output path
    output_dir = get_output_path(config)
    mkpath(output_dir)
    exe_path = joinpath(output_dir, exe_name)

    # Compile IR to executable
    cmd_tail = vcat(["-o", exe_path], _reproducible_link_flags())

    # Add library directories
    for dir in lib_dirs
        push!(cmd_tail, "-L$dir")
    end

    # Add link libraries (config + additional)
    all_libs = vcat(config.link.link_libraries, link_libraries)
    for lib in all_libs
        push!(cmd_tail, "-l$lib")
    end

    compiler = config.wrap.language == :c ? "clang" : "clang++"
    cmd_args = _args_with_response_file(String[], files, cmd_tail,
                                        joinpath(get_build_path(config), exe_name * ".rsp"))
    (output, exitcode) = _clang_for_c_bucket(compiler, cmd_args)

    if exitcode != 0
        error("Executable creation failed: $output")
    end

    if !isfile(exe_path)
        error("Executable file not created: $exe_path")
    end

    # Make executable
    chmod(exe_path, 0o755)

    size_kb = round(filesize(exe_path) / 1024, digits=1)
    println("  link: $exe_name ($size_kb KB)")

    return exe_path
end

# =============================================================================
# HIGH-LEVEL COMPILATION
# =============================================================================

"""
Auto-detect STL headers needed for template instantiations.
"""
function _auto_detect_stl_headers(templates::Vector{String})::Vector{String}
    headers = Set{String}()
    for t in templates
        if startswith(t, "std::vector")
            push!(headers, "<vector>")
        elseif startswith(t, "std::basic_string") || startswith(t, "std::string")
            push!(headers, "<string>")
        elseif startswith(t, "std::map")
            push!(headers, "<map>")
        elseif startswith(t, "std::unordered_map")
            push!(headers, "<unordered_map>")
        elseif startswith(t, "std::set")
            push!(headers, "<set>")
        elseif startswith(t, "std::unordered_set")
            push!(headers, "<unordered_set>")
        elseif startswith(t, "std::deque")
            push!(headers, "<deque>")
        elseif startswith(t, "std::list")
            push!(headers, "<list>")
        end
    end
    return collect(headers)
end

# =============================================================================
# FORCED ODR-USE OF CONTAINER MEMBERS
# -----------------------------------------------------------------------------
# `template class std::vector<int>;` is not enough to put std::vector<int>'s
# members in the binary, and on libc++ it puts almost none of them there.
#
# libc++ marks nearly every inline member `_LIBCPP_HIDE_FROM_ABI`, which expands
# to `__attribute__((exclude_from_explicit_instantiation))`. That attribute means
# exactly what it says: an explicit instantiation emits NOTHING for that member.
# It exists only where some translation unit actually odr-uses it. There is no
# knob to turn this off — libc++'s __config gates the attribute on
# `__has_attribute`, not on any user-settable macro.
#
# Measured on a real build before this: std::vector<int> carried ONE symbol, its
# destructor, and CppVector had no size() and no push_back() to bind. What did
# survive — size, push_back, operator[] — survived only because the fixture's own
# C++ happened to call them. std::map's at() and count(), which nothing called,
# were simply absent. libstdc++ emits every member on explicit instantiation, so
# none of this was visible on Linux.
#
# The generated functions below are never called. Their only job is to be
# compiled: naming a member odr-uses it, and the symbol lands in the binary.
#
# Every touch is guarded by `if constexpr` on whether the expression is valid for
# that container at all, so ONE generic body covers vector, string, map, set,
# deque and list with no per-container table — std::list has no operator[],
# std::vector no key_type, std::map no push_back, and each simply drops out.
# Arguments are spelled with the container's own nested typedefs for the same
# reason: value_type, key_type and size_type say what the container wants
# without this code having to know which container it is.
#
# `if constexpr` covers the immediate context only. A member whose body hard-
# errors deep inside its own instantiation would still break the TU, so the
# caller syntax-checks the generated file and falls back to the bare
# instantiations if it does not compile. Nothing that builds today can regress.
# =============================================================================

"The C++ preamble: a T& for any T, and the `if constexpr` guard the touches ride on."
const _TEMPLATE_FORCE_PREAMBLE = raw"""
// ── Forced odr-use (see generate_template_instantiations) ───────────────────
// None of this is ever called. It exists so that naming a member odr-uses it,
// which is what makes libc++ emit a member that `template class C;` skips.
#include <type_traits>

namespace replibuild_force {

// A T& for any T, including types that are not default-constructible. Never
// evaluated: every caller below sits in a function that is never called.
// Used for METHOD ARGUMENTS (value_type, key_type), not for constructing the
// container itself — that is `T c;` in the force body, which is also what
// puts a standalone default ctor in the .so (see _template_force_body).
template <class T> T& ref() { return *reinterpret_cast<T*>(1); }

// Runs `f(c)` only if that is a valid expression. The discarded branch of an
// `if constexpr` is not instantiated, so a member the container does not have
// costs nothing and errors nowhere.
template <class F, class C>
inline void touch(F&& f, C& c) {
    if constexpr (std::is_invocable_v<F&, C&>) { f(c); }
}

}  // namespace replibuild_force

// The container's own type, for naming its nested typedefs.
#define RB_SELF std::decay_t<decltype(x)>

// The lambda's trailing return type is what does the SFINAE: if EXPR is not
// valid for this container the lambda is not invocable, and `touch` drops it.
#define RB_TOUCH(EXPR)                                                         \
    ::replibuild_force::touch(                                                 \
        [](auto& x) -> decltype((EXPR), void()) { (void)(EXPR); }, c)
"""

# Every member name `_classify_stl_method` knows how to classify. Element access,
# lookup, insertion and erasure appear twice: once in the sequence spelling (an
# index) and once in the associative one (a key). A container has one or the
# other, never both, and the guard drops whichever it does not have.
const _TEMPLATE_FORCE_EXPRS = [
    # Observers
    "x.size()", "x.length()", "x.empty()", "x.capacity()", "x.max_size()",
    "x.data()", "x.c_str()", "x.begin()", "x.end()", "x.front()", "x.back()",
    # Mutators taking nothing
    "x.clear()", "x.pop_back()",
    # Sized operations
    "x.reserve(0)", "x.resize(0)",
    # Element access
    "x.at(::replibuild_force::ref<const typename RB_SELF::size_type>())",
    "x.at(::replibuild_force::ref<const typename RB_SELF::key_type>())",
    "x[::replibuild_force::ref<const typename RB_SELF::size_type>()]",
    "x[::replibuild_force::ref<const typename RB_SELF::key_type>()]",
    # Lookup
    "x.find(::replibuild_force::ref<const typename RB_SELF::key_type>())",
    "x.count(::replibuild_force::ref<const typename RB_SELF::key_type>())",
    # Insertion and erasure
    "x.push_back(::replibuild_force::ref<const typename RB_SELF::value_type>())",
    "x.insert(::replibuild_force::ref<const typename RB_SELF::value_type>())",
    "x.insert(x.begin(), ::replibuild_force::ref<const typename RB_SELF::value_type>())",
    "x.erase(x.begin())",
    "x.erase(::replibuild_force::ref<const typename RB_SELF::key_type>())",
    # std::string's own
    "x.append(::replibuild_force::ref<const RB_SELF>())",
]

"""
One never-called function per instantiated container, touching every member in
`_classify_stl_method`'s vocabulary.

Default-constructs `T c` rather than taking `T&`. `template class T;` does not
emit the header-inline default ctor as a standalone symbol, and the Julia
factory calls `T()` with no extra args — so a force body that never names
`T t;` leaves `create_std_map_*` memset-zeroing a live `std::map`.

`used` keeps the function through dead-code removal of an unreferenced static.
The anonymous namespace keeps it out of the DLL's export table. `-fno-inline`
on this generated TU (see `_template_tu_compile_flags`) is what stops the
ctor being inlined *into* this function at `-O2`; `noinline` on the function
alone is not enough — clang still inlines `vector()` into a noinline caller.
"""
function _template_force_body(instantiation_type::String, index::Int)::String
    io = IOBuffer()
    println(io, "namespace {")
    println(io, "#if defined(__GNUC__) || defined(__clang__)")
    println(io, "__attribute__((used))")
    println(io, "#endif")
    println(io, "void replibuild_force_$(index)() {")
    println(io, "    $instantiation_type c;")
    for expr in _TEMPLATE_FORCE_EXPRS
        println(io, "    RB_TOUCH($expr);")
    end
    println(io, "}")
    println(io, "}  // namespace")
    return String(take!(io))
end

"Generated instantiation TU (`replibuild_templates.cpp`), not a library source."
_is_template_instantiation_tu(cpp_file::AbstractString)::Bool =
    basename(cpp_file) == "replibuild_templates.cpp"

"""
Extra flags for the generated instantiation TU only.

`-fno-inline` is load-bearing at `-O2`: a `T c;` in the force body still
inlines `vector()` into that function unless inlining is off for the TU.
clang 22 ignores `-fkeep-inline-functions`. This is not a per-file flag on
library sources — it applies only to the file RepliBuild generated.
"""
function _template_tu_compile_flags(cpp_file::AbstractString)::Vector{String}
    _is_generated_force_tu(cpp_file) ? ["-fno-inline"] : String[]
end

"""
Per-file compile fingerprint: the templates TU carries extra flags the others do
not. It is written to the IR's `.key` sidecar, so it is SHA-256 (`_key_feed!`):
under `Base.hash` a Julia switch recompiled the templates TU.
"""
function _file_compile_fingerprint(config_fp::String, cpp_file::AbstractString)::String
    isempty(config_fp) && return config_fp
    if _is_generated_force_tu(cpp_file)
        ctx = SHA.SHA256_CTX()
        _key_feed!(ctx, config_fp)
        _key_feed!(ctx, "template-tu-fno-inline-v1")
        return bytes2hex(SHA.digest!(ctx))
    end
    return config_fp
end

"Write the instantiation TU. `force` adds the odr-use block; without it this is the bare form."
function _write_template_tu(path::String, headers::Vector{String},
                            instantiation_types::Vector{String}, force::Bool)
    open(path, "w") do io
        println(io, "// Auto-generated by RepliBuild to force template instantiations")
        for header in headers
            if startswith(header, "<") || startswith(header, "\\\"")
                println(io, "#include $header")
            else
                println(io, "#include \"$header\"")
            end
        end
        println(io, "")
        for t in instantiation_types
            println(io, "template class $t;")
        end
        # One tag per entry, so the symbol table says how the compiler spells
        # each listed type: `template void replibuild_force::tag<T, N>(T*);`
        # demangles with T in its canonical form (a typedef resolved, default
        # template arguments written out, `Clipper2Lib::Point64` →
        # `Clipper2Lib::Point<long>`), and `_stl_methods_from_nm` reads N and
        # T back to match entry N against nm's spelling of its members.
        println(io, "")
        println(io, "namespace replibuild_force { template <class T, int N> void tag(T*) {} }")
        for (i, t) in enumerate(instantiation_types)
            println(io, "template void replibuild_force::tag<$t, $(i - 1)>($t*);")
        end
        if force
            println(io, "")
            println(io, _TEMPLATE_FORCE_PREAMBLE)
            for (i, t) in enumerate(instantiation_types)
                println(io, "")
                print(io, _template_force_body(t, i))
            end
            println(io, "")
            println(io, "#undef RB_TOUCH")
            println(io, "#undef RB_SELF")
        end
    end
end

"""
Does the generated TU compile? Decides whether the odr-use block can stay.

An unavailable probe compiler counts as NOT compiling: the forcing block is the
optional half of this file, so anything that cannot be verified is dropped
rather than risked.
"""
function _template_tu_compiles(config::RepliBuildConfig, path::String)::Tuple{Bool,String}
    probe = _probe_compiler(:cpp)
    flags = get_compile_flags(config)
    incs  = ["-I$dir" for dir in get_include_dirs(config)]
    defs  = ["-D$(k)=$(v)" for (k, v) in config.compile.defines]
    errbuf = IOBuffer()
    try
        p = run(pipeline(ignorestatus(`$probe -fsyntax-only $flags $incs $defs $path`),
                         stdout = devnull, stderr = errbuf))
        return (p.exitcode == 0, String(take!(errbuf)))
    catch e
        return (false, "probe compiler unavailable: $e")
    end
end

"""
Generate a dummy C++ file to explicitly instantiate requested templates
so they appear in the DWARF metadata.
"""
function generate_template_instantiations(config::RepliBuildConfig, cpp_files::Vector{String})::Vector{String}
    if isempty(config.types.templates)
        return cpp_files
    end

    cache_dir = get_cache_path(config)
    mkpath(cache_dir)
    dummy_file = joinpath(cache_dir, "replibuild_templates.cpp")

    # Merge user-specified headers with auto-detected STL headers
    auto_headers = _auto_detect_stl_headers(config.types.templates)
    all_headers = unique(vcat(config.types.template_headers, auto_headers))

    # STL typedefs that must be expanded to their underlying template class
    # for explicit instantiation to be valid C++.
    stl_typedef_expansions = Dict{String,String}(
        "std::string"  => "std::basic_string<char>",
        "std::wstring" => "std::basic_string<wchar_t>",
        "std::u16string" => "std::basic_string<char16_t>",
        "std::u32string" => "std::basic_string<char32_t>",
    )

    instantiation_types = [get(stl_typedef_expansions, t, t) for t in config.types.templates]

    # Written with the odr-use block, then verified. `template class C;` alone
    # leaves libc++ containers nearly empty (see the section comment above), so
    # the block is what makes the mechanism work there — but it is generated C++
    # over user-chosen element types, and a member that hard-errors inside its
    # own instantiation is outside what `if constexpr` can guard. So it is
    # checked before it is trusted, and dropped if it does not compile: a build
    # that works today cannot start failing because of this.
    _write_template_tu(dummy_file, all_headers, instantiation_types, true)
    (ok, output) = _template_tu_compiles(config, dummy_file)
    if !ok
        _write_template_tu(dummy_file, all_headers, instantiation_types, false)
        @warn """
              Template odr-use forcing did not compile — falling back to bare explicit \
              instantiations. On libstdc++ that is the same thing; on libc++ any container \
              member your own code never calls will be missing from the build, and binding \
              it will fail with "no <method> thunk available".
              """ templates=instantiation_types output=strip(output)
    end

    return vcat(cpp_files, [dummy_file])
end

# =============================================================================
# RECORDS MADE VISIBLE (`[types] records`)
# -----------------------------------------------------------------------------
# Debug info describes the types a translation unit uses. A header-only class
# that no source of the library mentions is in no unit's DWARF, standalone or
# not: Bullet's `btDefaultMotionState`, the class every Bullet application
# passes a body's pose through, is used by nothing under LinearMath,
# BulletCollision or BulletDynamics (2026-11-15). It had no type, no
# constructor and no method in the wrapper, and the Hub compiled a one-line
# file declaring a `btDefaultMotionState *` to make it appear. `[types]
# records` names such classes; a generated unit includes the headers
# (`[wrap] shim_headers`, then `[types] template_headers`) and instantiates
# `replibuild_force::record<T>(T*)` for each, which is a use of `T`. The
# unit's DWARF then carries the class's definition, and the inline-constructor
# pass forces what it declares like any other record's.
#
# It is compiled in the first pass, beside the templates TU. An entry the unit
# cannot compile (a misspelled name, a header not among the includes) is a
# build error that names it; an entry that compiled and still has no
# definition in the metadata (clang's constructor homing under plain `-g`) is
# a warning that says to build with `-fstandalone-debug`.
# =============================================================================

"Generated unit making `[types] records` visible to the DWARF."
_is_visible_records_tu(cpp_file::AbstractString)::Bool =
    basename(cpp_file) == "replibuild_records.cpp"

"""
    _visible_records_unit_text(headers, records) -> String

The unit: one `#include` per header, the helper, and one explicit
instantiation per record, each on its own line.
"""
function _visible_records_unit_text(headers::AbstractVector, records::AbstractVector)::String
    io = IOBuffer()
    println(io, "// Auto-generated by RepliBuild: the classes [types] records names, used once so the DWARF defines them")
    for h in headers
        hs = String(h)
        println(io, (startswith(hs, "<") || startswith(hs, "\"")) ? "#include $hs" : "#include \"$hs\"")
    end
    println(io)
    # `sizeof` asks for the complete type, so a name that is only declared in
    # the headers is a compile error here instead of a silent nothing.
    println(io, "namespace replibuild_force { template <class T> void record(T*) { (void)sizeof(T); } }")
    for r in records
        println(io, "template void replibuild_force::record<$r>($r*);")
    end
    return String(take!(io))
end

"""
    generate_visible_records(config, cpp_files) -> cpp_files

Write the `[types] records` unit under the build cache and append it to
`cpp_files`. The unit is syntax-checked first; when it does not compile, each
record is checked alone and the build stops with the entries that fail and
the compiler's first error for each. Nothing listed, or a C project, returns
`cpp_files` as given.
"""
function generate_visible_records(config::RepliBuildConfig, cpp_files::Vector{String})::Vector{String}
    records = unique(String[String(strip(r)) for r in config.types.records])
    (isempty(records) || config.wrap.language != :cpp) && return cpp_files
    cache_dir = get_cache_path(config)
    mkpath(cache_dir)
    unit_path = joinpath(cache_dir, "replibuild_records.cpp")
    headers = unique(vcat(String[String(h) for h in config.wrap.shim_headers],
                          String[String(h) for h in config.types.template_headers]))
    text = _visible_records_unit_text(headers, records)
    probe = joinpath(cache_dir, "replibuild_records_probe.cpp")
    first_error(out) = (lines = split(out, '\n'); k = findfirst(l -> occursin("error", l), lines);
                        k === nothing ? String(strip(first(out, 200))) : String(strip(lines[k])))
    write(probe, text)
    (ok, out) = _template_tu_compiles(config, probe)
    if !ok
        bad = String[]
        for r in records
            write(probe, _visible_records_unit_text(headers, [r]))
            (rok, rout) = _template_tu_compiles(config, probe)
            rok || push!(bad, "  $r: $(first_error(rout))")
        end
        rm(probe; force = true)
        isempty(bad) && error("[types] records: the generated unit does not compile: $(first_error(out))")
        error("[types] records: $(length(bad)) entr$(length(bad) == 1 ? "y does" : "ies do") not compile " *
              "against the headers in [wrap] shim_headers and [types] template_headers " *
              "($(isempty(headers) ? "none listed" : join(headers, ", "))):\n" * join(bad, "\n") *
              "\nAn entry is a class name as C++ spells it (`ns::Class`), and its header must be among those.")
    end
    rm(probe; force = true)
    _write_if_changed(unit_path, text)
    return vcat(cpp_files, [unit_path])
end

"The bare name DWARF keys a record by: `ns::Outer::Inner<int>` → `Inner<int>`."
function _record_bare_name(spelling::AbstractString)::String
    s = String(strip(spelling))
    depth = 0
    cut = 0
    i = firstindex(s)
    while i <= lastindex(s)
        c = s[i]
        if c == '<'
            depth += 1
        elseif c == '>'
            depth = max(0, depth - 1)
        elseif depth == 0 && c == ':' && i < lastindex(s) && s[nextind(s, i)] == ':'
            cut = nextind(s, i)
        end
        i = nextind(s, i)
    end
    return cut == 0 ? s : String(s[nextind(s, cut):end])
end

"""
    _invisible_records(records, struct_defs) -> Vector{String}

The `[types] records` entries the metadata has no definition for, as written.
"""
_invisible_records(records, struct_defs::AbstractDict) =
    String[String(r) for r in records if !haskey(struct_defs, _record_bare_name(String(r)))]

"Warn, by name, about `[types] records` entries the build compiled and the DWARF still does not define."
function _report_invisible_records(config::RepliBuildConfig, metadata_path::AbstractString)
    (isempty(config.types.records) || config.wrap.language != :cpp) && return String[]
    missing_defs = _invisible_records(config.types.records,
                                      get(_read_json(metadata_path), "struct_definitions", Dict{String,Any}()))
    isempty(missing_defs) || @warn """
        [types] records: $(length(missing_defs)) class(es) compiled into the build and still have no \
        definition in its debug info: $(join(missing_defs, ", ")).
        Under clang's default `-g` a class with a constructor is defined only where a constructor of \
        it is emitted (constructor homing). Add `-fstandalone-debug` (or `-Xclang -fno-use-ctor-homing`) \
        to [compile] flags.
        """
    return missing_defs
end

# =============================================================================
# FORCED INLINE CONSTRUCTORS
# -----------------------------------------------------------------------------
# A constructor defined inline in a header and used by no translation unit of
# the library has no definition in the library: the class DIE declares it
# (`DW_AT_declaration`, with its linkage name), `nm` has no `C1`/`C2` for it,
# and the class cannot be built from Julia. Bullet's `btSphereShape(btScalar)`,
# `btDefaultMotionState(const btTransform&, …)` and
# `btRigidBody::btRigidBodyConstructionInfo(…)` were that shape (2026-11-15),
# and the Hub carried a hand-written placement-new shim per class. Consumer
# compensation is a feature request, so the build now emits those definitions
# itself, the way the templates TU above makes libc++ emit container members:
# a generated unit that includes each record's header and odr-uses the
# constructor through placement new (`replibuild_force::ctor<T, A...>`), so
# clang emits the real `C1`/`C2` definitions into the library. The constructors
# are then definitions in the DWARF and in the symbol table, and nothing
# downstream changes: GeneratorCpp wraps them as any constructor.
#
# The set is declared minus defined, both of which the build has after its
# first link (`_inline_constructor_plan`): every library record's declared
# constructors (`record_abi[name]["constructors"]`) less the ones the symbol
# table defines. So the unit is written after the first metadata extraction,
# and `compile_project` compiles it, relinks, and extracts again. Only a
# declared constructor (a declaration DIE), never an implicit one; a template
# constructor, a deleted one, a private or protected one, and one the unit
# cannot spell are skipped and named. The unit's guard (`is_abstract`,
# `is_constructible`) emits nothing for an abstract class or a constructor it
# cannot call, so the unit always compiles for those; a header that does not
# compile on its own drops its classes with the compiler's first line.
#
# The spelling of the class and of each parameter is the compiler's own: the
# demangled `C4` key (`_demangle`, libstdc++'s `__cxa_demangle` from Julia's
# own runtime, else `c++filt`). DWARF keys records by bare name and a
# declaration's formal parameters carry no names, so the DWARF `c_type`s could
# neither qualify `btRigidBody::btRigidBodyConstructionInfo` nor be relied on
# for the parameter list; the mangled name is the compiler's full answer.
#
# The same unit defines the library's other inline member functions
# (`_inline_method_plan`, 2026-11-15): Bullet's `btRigidBody::applyCentralImpulse`,
# `setLinearVelocity` and `getLinearVelocity` are defined in the class body and
# called by nothing in `src/`, so they had no symbol either. A member function
# has an address, so the unit takes it instead of calling anything.
#
# What the pass cannot see: under constructor homing (clang's default `-g`), a
# class whose constructors are ALL inline and unused is declaration-only in
# every CU's DWARF, so its constructors are not declared anywhere the parser
# can read. `-fstandalone-debug` (Bullet, box2d) makes the definition appear
# and the pass then forces its constructors; see the F4 entry in CLAUDE.md.
# =============================================================================

"Generated unit defining the library's declared-but-undefined inline constructors."
_is_inline_ctor_tu(cpp_file::AbstractString)::Bool =
    basename(cpp_file) == "replibuild_ctors.cpp"

"A unit RepliBuild generated to force definitions (the templates TU, the inline-constructor unit)."
_is_generated_force_tu(cpp_file::AbstractString)::Bool =
    _is_template_instantiation_tu(cpp_file) || _is_inline_ctor_tu(cpp_file) || _is_visible_records_tu(cpp_file)

const _CXA_DEMANGLE = Ref{Ptr{Cvoid}}(C_NULL)
const _CXA_DEMANGLE_LOOKED = Ref(false)

"`__cxa_demangle` from the C++ runtime Julia itself loads, or `C_NULL`. Looked up once."
function _cxa_demangle_ptr()
    _CXA_DEMANGLE_LOOKED[] && return _CXA_DEMANGLE[]
    _CXA_DEMANGLE_LOOKED[] = true
    for lib in ("libstdc++", "libstdc++-6", "libc++abi")
        h = Libdl.dlopen_e(lib)
        h == C_NULL && continue
        p = Libdl.dlsym_e(h, :__cxa_demangle)
        p == C_NULL && continue
        _CXA_DEMANGLE[] = p
        break
    end
    return _CXA_DEMANGLE[]
end

"""
    _demangle(symbol) -> Union{String,Nothing}

The Itanium demangling of `symbol`, or `nothing`. In process through the C++
runtime Julia loads (`__cxa_demangle`), which also reads clang's unified
`C4`/`D4` keys, else through `c++filt`/`llvm-cxxfilt` on PATH. A `C4` key
demangles to the constructor as the compiler spelled it, class and parameters
fully qualified: `btRigidBody::btRigidBodyConstructionInfo::
btRigidBodyConstructionInfo(float, btMotionState*, btCollisionShape*,
btVector3 const&)`.
"""
function _demangle(symbol::AbstractString)
    sym = String(symbol)
    p = _cxa_demangle_ptr()
    if p != C_NULL
        status = Ref{Cint}(0)
        r = ccall(p, Ptr{UInt8}, (Cstring, Ptr{UInt8}, Ptr{Csize_t}, Ref{Cint}),
                  sym, C_NULL, C_NULL, status)
        if r != C_NULL
            s = status[] == 0 ? unsafe_string(r) : nothing
            # The buffer is the C++ runtime's `malloc`. On Linux that is the one
            # libc; on Windows Julia's `libstdc++-6.dll` and Julia share msvcrt,
            # but a few hundred bytes per build are cheaper than being wrong
            # about which CRT, so they are left to the process there.
            Sys.iswindows() || Libc.free(r)
            return s
        end
        return nothing
    end
    for tool in ("c++filt", "llvm-cxxfilt")
        (Tools.known(tool) ? Tools.have(tool) : Sys.which(tool) !== nothing) || continue
        out = try
            strip(read(pipeline(Tools.tool_cmd(`$tool`), stdin = IOBuffer(sym * "\n")), String))
        catch
            ""
        end
        (isempty(out) || out == sym) && continue
        return String(out)
    end
    return nothing
end

"The symbol-table variants of a unified constructor key: `C1`, `C2`, `C3` for `C4`."
function _constructor_variants(c4_key::AbstractString)::Vector{String}
    m = match(r"C4(?=[EI])", c4_key)
    m === nothing && return String[]
    i = m.offset
    return [c4_key[1:prevind(c4_key, i)] * v * c4_key[i+2:end] for v in ("C1", "C2", "C3")]
end

"Split `s` at depth-0 commas (`<>`, `()`, `[]`, `{}` nest), each part stripped."
function _split_depth0_commas(s::AbstractString)::Vector{String}
    out = String[]
    depth = 0
    start = firstindex(s)
    i = start
    n = lastindex(s)
    while i <= n
        c = s[i]
        if c in ('<', '(', '[', '{')
            depth += 1
        elseif c in ('>', ')', ']', '}')
            depth = max(0, depth - 1)
        elseif c == ',' && depth == 0
            push!(out, String(strip(s[start:prevind(s, i)])))
            start = nextind(s, i)
        end
        i = nextind(s, i)
    end
    push!(out, String(strip(s[start:end])))
    return out
end

"""
    _constructor_spelling(demangled) -> Union{Nothing,Tuple{String,Vector{String}}}

The qualified class and the parameter spellings of a demangled constructor,
`("ic::Plain", ["int", "int"])` for `ic::Plain::Plain(int, int)`, or `nothing`
when the unit could not name it in C++: an anonymous namespace, a lambda or a
local class (an enclosing `(`), an ABI tag (`[abi:cxx11]`), a variadic list,
or a name with no class part at all.
"""
function _constructor_spelling(demangled::AbstractString)
    s = String(demangled)
    (open, local_open) = _param_list_paren(s)
    (open == 0 || local_open != 0) && return nothing
    close = _matching_paren(s, open)
    close == 0 && return nothing
    parts = _qualified_name_parts(s)
    length(parts) >= 2 || return nothing
    qualified = join(parts[1:end-1], "::")
    inner = strip(s[nextind(s, open):prevind(s, close)])
    params = isempty(inner) ? String[] : _split_depth0_commas(inner)
    unspellable(t) = occursin("(anonymous namespace)", t) || occursin("lambda", t) ||
                     occursin("[abi:", t) || occursin("...", t) || occursin("{", t) ||
                     occursin("unnamed", t) || isempty(t)
    (unspellable(qualified) || occursin('(', qualified)) && return nothing
    any(unspellable, params) && return nothing
    return (qualified, params)
end

"A standard-library scope (`std`, `std::__cxx11`, `__gnu_cxx`, `std::__1`)."
_is_stdlib_scope(scope) = (s = String(something(scope, "")); isempty(s) ? false :
    (first(split(s, "::")) in ("std", "__gnu_cxx") || startswith(s, "__cxx")))

const _HEADER_EXTENSIONS = (".h", ".hh", ".hpp", ".hxx", ".H", ".inl", ".ipp", ".tcc", ".ixx")

"""
The headers among a record's `decl_files` the unit includes, sorted. All of
them, not one: two records of one bare name (`btHeightfieldTerrainShape::Range`,
`btBatchedConstraints::Range`) merge under it, and a class template's
specialization is declared where the template is while its arguments live
elsewhere, so every header the name was seen in is needed to spell it.
"""
function _inline_ctor_headers_of(decl_files)::Vector{String}
    return sort!(unique!(String[String(f) for f in decl_files
                                if any(e -> endswith(String(f), e), _HEADER_EXTENSIONS) ||
                                   !occursin('.', basename(String(f)))]))
end

"""
    _inline_constructor_plan(struct_defs, record_abi, symbols, exclude_patterns;
                             demangle = _demangle) -> Dict

Declared minus defined: the constructors the library's own records declare
(`record_abi[name]["constructors"]`) that no symbol in `symbols` defines, as
`forced` rows (`class`, `symbol`, `type`, `parameters`, `headers`) the unit
instantiates, and `skipped` rows (`class`, `symbol`, `reason`) the wrap report
names. A toolchain record and a standard-library scope are nobody's to force
and are passed over silently; a library record is skipped with a reason when
its name has distinct definitions, when no header declares it, and per
constructor when it is implicit (`artificial`), deleted, not public, a
constructor template (`C4I…`), dropped by `[wrap] exclude_symbols`, or not
spellable (`_constructor_spelling`). Sorted by class, then symbol.
"""
function _inline_constructor_plan(struct_defs::AbstractDict, record_abi::AbstractDict, symbols,
                                  exclude_patterns::Vector{String}; demangle = _demangle)
    defined = Set{String}(String(get(s, "mangled", "")) for s in symbols if s isa AbstractDict)
    forced = Dict{String,Any}[]
    skipped = Dict{String,Any}[]
    skip(cls, sym, why) = push!(skipped, Dict{String,Any}("class" => cls, "symbol" => sym, "reason" => why))
    for name in sort!(collect(String, keys(struct_defs)))
        startswith(name, "__enum__") && continue
        def = struct_defs[name]
        def isa AbstractDict || continue
        entry = get(record_abi, name, nothing)
        entry isa AbstractDict || continue
        ctors = get(entry, "constructors", nothing)
        (ctors isa AbstractVector && !isempty(ctors)) || continue
        get(def, "toolchain", false) == true && continue
        _is_stdlib_scope(get(def, "scope", "")) && continue
        decl_files = get(def, "decl_files", String[])
        headers = _inline_ctor_headers_of(decl_files)
        record_reason = get(def, "conflict", false) == true ?
                        "the name has distinct definitions in this library" :
                        isempty(headers) ?
                        "declared in no header the unit can include" *
                        (isempty(decl_files) ? "" : " (" * join(decl_files, ", ") * ")") : nothing
        for c in ctors
            c isa AbstractDict || continue
            sym = String(get(c, "symbol", ""))
            isempty(sym) && continue
            variants = _constructor_variants(sym)
            any(in(defined), variants) && continue         # the library defines it
            if record_reason !== nothing
                skip(name, sym, record_reason)
            elseif get(c, "artificial", false) == true
                skip(name, sym, "an implicit constructor, never declared")
            elseif get(c, "deleted", false) == true
                skip(name, sym, "deleted")
            elseif get(c, "accessibility", "public") != "public"
                skip(name, sym, String(c["accessibility"]))
            elseif occursin(r"C4I", sym)
                skip(name, sym, "a constructor template")
            else
                dm = demangle(sym)
                if dm === nothing
                    skip(name, sym, "the symbol does not demangle on this host")
                    continue
                end
                if !isempty(exclude_patterns) && !isempty(variants)
                    kept = _without_excluded_symbols(Set{String}(variants),
                                                     [Dict{String,Any}("mangled" => v, "demangled" => dm) for v in variants],
                                                     exclude_patterns)
                    if length(kept) < length(variants)
                        skip(name, sym, "dropped by [wrap] exclude_symbols")
                        continue
                    end
                end
                spelled = _constructor_spelling(dm)
                if spelled === nothing
                    skip(name, sym, "cannot be spelled in C++: $dm")
                    continue
                end
                push!(forced, Dict{String,Any}("class" => name, "symbol" => sym, "type" => spelled[1],
                                               "parameters" => spelled[2], "headers" => headers))
            end
        end
    end
    return Dict{String,Any}("forced" => forced, "skipped" => skipped)
end

"""
    _method_spelling(demangled) -> Union{Nothing,NamedTuple}

How the forced unit names a demangled member function: `(type, name,
parameters, isconst, structor)`, `("btRigidBody", "applyCentralImpulse",
["btVector3 const&"], false, false)` for `btRigidBody::applyCentralImpulse(
btVector3 const&)`. The parameter list is the LAST paren group, so
`operator()(int) const` reads as `operator()`; the name is cut at the last
depth-0 `::`, or at `::operator` for an operator (a conversion operator's
name holds a type, `operator float*`). `structor` says the name is the
class's own or starts with `~`.

`nothing` when the unit cannot take its address by that spelling: a member
function template or anything else whose demangling opens with a return type
(a depth-0 space before the name), a `volatile` or ref-qualified member, a
variadic list, an anonymous namespace, a lambda or local class, an ABI tag,
or a name with no class part.
"""
function _method_spelling(demangled::AbstractString)
    s = String(strip(demangled))
    isconst = false
    if endswith(s, " const")
        isconst = true
        s = String(strip(s[1:end-6]))
    end
    endswith(s, ")") || return nothing              # volatile, `&`, `&&`, `[abi:…]`
    # The opening paren of the last group.
    depth = 0
    open = 0
    i = lastindex(s)
    while i >= firstindex(s)
        c = s[i]
        if c == ')'
            depth += 1
        elseif c == '('
            depth -= 1
            if depth == 0
                open = i
                break
            end
        end
        i = prevind(s, i)
    end
    open <= firstindex(s) && return nothing
    prefix = s[1:prevind(s, open)]
    inner = strip(s[nextind(s, open):prevind(s, lastindex(s))])
    params = isempty(inner) ? String[] : _split_depth0_commas(inner)
    unspellable(t) = occursin("(anonymous namespace)", t) || occursin("lambda", t) ||
                     occursin("[abi:", t) || occursin("...", t) || occursin("{", t) ||
                     occursin("unnamed", t) || isempty(t)
    any(unspellable, params) && return nothing
    # Depth-0 positions of `::` and of spaces in the prefix (`<>` and `()` nest).
    function depth0(str, pred)
        out = Int[]
        d = 0
        for j in eachindex(str)
            c = str[j]
            if c in ('<', '(', '[', '{')
                d += 1
            elseif c in ('>', ')', ']', '}')
                d = max(0, d - 1)
            elseif d == 0 && pred(str, j)
                push!(out, j)
            end
        end
        return out
    end
    op = findlast("::operator", prefix)
    # An operator's own `<`, `>`, `()` would confuse the depth count, so the
    # class is cut before it and only the class is measured.
    if op !== nothing && !occursin(r"^::operator[\w]", SubString(prefix, first(op))) 
        type = prefix[1:prevind(prefix, first(op))]
        name = prefix[first(op)+2:end]
    else
        cuts = depth0(prefix, (str, j) -> str[j] == ':' && j < lastindex(str) && str[nextind(str, j)] == ':')
        isempty(cuts) && return nothing
        type = prefix[1:prevind(prefix, last(cuts))]
        name = prefix[last(cuts)+2:end]
        occursin('<', name) && return nothing       # a member function template
    end
    (isempty(type) || isempty(name)) && return nothing
    (unspellable(type) || occursin('(', type) || occursin("[abi:", name)) && return nothing
    isempty(depth0(type, (str, j) -> str[j] == ' ')) || return nothing   # a return type
    last_part = let c = depth0(type, (str, j) -> str[j] == ':' && j < lastindex(str) && str[nextind(str, j)] == ':')
        isempty(c) ? type : type[last(c)+2:end]
    end
    bare(t) = (k = findfirst('<', t); k === nothing ? t : t[1:prevind(t, k)])
    structor = startswith(name, "~") || name == bare(last_part)
    return (type = String(type), name = String(name), parameters = params,
            isconst = isconst, structor = structor)
end

"""
    _inline_method_plan(struct_defs, record_abi, symbols, exclude_patterns;
                        demangle = _demangle) -> Dict

Declared minus defined, for member functions: every public, non-virtual
member function a library record declares (`record_abi[name]["methods"]`)
that no symbol in `symbols` defines, as `forced` rows (`class`, `symbol`,
`type`, `name`, `parameters`, `const`, `headers`) whose address the forced
unit takes, and `skipped` rows (`class`, `symbol`, `reason`).

Passed over silently, because they are not the gap: a constructor or
destructor (the constructor plan's, by the demangled name), an implicit or
deleted member, a private or protected one, and a virtual one, which clang
emits with the class's vtable wherever that is emitted. Skipped with a
reason: a record whose name has distinct definitions or that no header
declares, a symbol `[wrap] exclude_symbols` drops, a member the unit
cannot name (`_method_spelling`), and a member of a class template's
specialization (`_TEMPLATE_MEMBER_SKIP`). Whether a forced member has a body in its
header is not known here; `generate_inline_definitions` reads that off the
unit's IR. Sorted by class, then symbol.
"""
# A class template's members are `[types] templates`' to instantiate, all of
# them, per specialization the recipe names. Forced here they are a
# container's whole interface once per element type the library happens to
# use: 2,990 of Bullet's 4,999 undefined inline members were
# `btAlignedObjectArray<T>` and `btAlignedAllocator<T, 16>` over 216
# specializations, against 2,009 members of its classes.
const _TEMPLATE_MEMBER_SKIP = "a member of a class template's specialization (list the specialization in [types] templates to instantiate its members)"

function _inline_method_plan(struct_defs::AbstractDict, record_abi::AbstractDict, symbols,
                             exclude_patterns::Vector{String}; demangle = _demangle)
    defined = Set{String}(String(get(s, "mangled", "")) for s in symbols if s isa AbstractDict)
    forced = Dict{String,Any}[]
    skipped = Dict{String,Any}[]
    skip(cls, sym, why) = push!(skipped, Dict{String,Any}("class" => cls, "symbol" => sym, "reason" => why))
    for name in sort!(collect(String, keys(struct_defs)))
        startswith(name, "__enum__") && continue
        def = struct_defs[name]
        def isa AbstractDict || continue
        entry = get(record_abi, name, nothing)
        entry isa AbstractDict || continue
        methods = get(entry, "methods", nothing)
        (methods isa AbstractVector && !isempty(methods)) || continue
        get(def, "toolchain", false) == true && continue
        _is_stdlib_scope(get(def, "scope", "")) && continue
        decl_files = get(def, "decl_files", String[])
        headers = _inline_ctor_headers_of(decl_files)
        record_reason = get(def, "conflict", false) == true ?
                        "the name has distinct definitions in this library" :
                        isempty(headers) ? "declared in no header the unit can include" : nothing
        for m in methods
            m isa AbstractDict || continue
            sym = String(get(m, "symbol", ""))
            (isempty(sym) || sym in defined) && continue
            (get(m, "artificial", false) == true || get(m, "deleted", false) == true ||
             get(m, "virtual", false) == true || get(m, "accessibility", "public") != "public") && continue
            dm = demangle(sym)
            if dm === nothing
                skip(name, sym, "the symbol does not demangle on this host")
                continue
            end
            spelled = _method_spelling(dm)
            spelled !== nothing && spelled.structor && continue
            if record_reason !== nothing
                skip(name, sym, record_reason)
                continue
            end
            if !isempty(exclude_patterns)
                kept = _without_excluded_symbols(Set{String}([sym]),
                                                 [Dict{String,Any}("mangled" => sym, "demangled" => dm)],
                                                 exclude_patterns)
                if isempty(kept)
                    skip(name, sym, "dropped by [wrap] exclude_symbols")
                    continue
                end
            end
            if spelled === nothing
                skip(name, sym, "cannot be named in C++ (a member template, or a qualifier the unit does not spell)")
                continue
            end
            if occursin('<', spelled.type)
                skip(name, sym, _TEMPLATE_MEMBER_SKIP)
                continue
            end
            push!(forced, Dict{String,Any}("class" => name, "symbol" => sym, "type" => spelled.type,
                                           "name" => spelled.name, "parameters" => spelled.parameters,
                                           "const" => spelled.isconst, "headers" => headers))
        end
    end
    return Dict{String,Any}("forced" => forced, "skipped" => skipped)
end

# =============================================================================
# DEFAULT ARGUMENTS
# -----------------------------------------------------------------------------
# A C++ default argument exists only in the header: the mangled name spells
# every parameter, and clang writes no `DW_AT_default_value`. So
# `btDefaultMotionState(start)` had to be called with its second transform
# spelled out, and `btHingeConstraint::setLimit(lo, hi)` with all five
# arguments (2026-11-15, ObjectSim). The compiler knows the defaults, so the
# forced unit asks it: for a constructor or member function F of N parameters
# and each shorter prefix of them, one function in namespace
# `replibuild_default` makes that call, `self->f(a0, …, ak)`. Where the header
# gives no default the call is a compile error ON THAT ROW, which the unit's
# row tracing drops; where it compiles, clang has filled the defaults in and
# the function is the short call's entry point.
#
# The arguments have exactly F's parameter types, so overload resolution takes
# F or is ambiguous (an error, the row is dropped): F is an identity match on
# every argument and nothing can beat it. A prefix another member of the class
# declares in full is not generated at all; that call is that member's.
#
# The second extraction gives each surviving function F's identity
# (`_apply_default_argument_identity!`): its symbol's demangled name becomes
# `Class::f(prefix)`, its first parameter `this`, so both receiver gates and
# every name derivation treat it as the method it stands for, with fewer
# parameters. On the Itanium ABI a function taking `Class*` first is called
# exactly as a member of `Class` is, by-value returns included (mingw too).
# =============================================================================

"The mangled-name prefix of default-argument function `id`: `_ZN18replibuild_default2d7E`."
_default_shim_prefix(id::Integer) = (n = "d$(id)"; "_ZN18replibuild_default$(length(n))$(n)E")

"""
    _default_argument_plan(struct_defs, record_abi, symbols, exclude_patterns;
                           demangle = _demangle) -> Dict

`candidates`: for every public constructor and member function a library
record declares (`record_abi[name]["methods"]`) and every proper prefix of
its parameters, one row (`id`, `class`, `symbol`, `type`, `name`, `ctor`,
`const`, `parameters`: the prefix, `omitted`, `headers`, `defined`,
`demangled`: `Class::f(prefix)`). Whether the header has defaults for the
omitted parameters is not known here; the unit's compile answers per row.

Left out: an implicit or deleted member, a private or protected one, a
destructor, an operator, a member the unit cannot name, a class template's
specialization, a toolchain or standard-library record, one with distinct
definitions or no header, a symbol `[wrap] exclude_symbols` drops, a prefix
another member of the record declares in full, and a prefix already listed.
`defined` says the symbol table defines the member (a constructor: any of its
variants); the unit keeps an undefined member's rows only when it defines the
member itself. Sorted by class, then symbol, then longest prefix first.
"""
function _default_argument_plan(struct_defs::AbstractDict, record_abi::AbstractDict, symbols,
                                exclude_patterns::Vector{String}; demangle = _demangle)
    defined = Set{String}(String(get(s, "mangled", "")) for s in symbols if s isa AbstractDict)
    rows = Dict{String,Any}[]
    for name in sort!(collect(String, keys(struct_defs)))
        startswith(name, "__enum__") && continue
        def = struct_defs[name]
        def isa AbstractDict || continue
        entry = get(record_abi, name, nothing)
        entry isa AbstractDict || continue
        methods = get(entry, "methods", nothing)
        (methods isa AbstractVector && !isempty(methods)) || continue
        (get(def, "toolchain", false) == true || get(def, "conflict", false) == true) && continue
        _is_stdlib_scope(get(def, "scope", "")) && continue
        headers = _inline_ctor_headers_of(get(def, "decl_files", String[]))
        isempty(headers) && continue
        # Every member's spelling first: a prefix that IS a sibling's whole
        # parameter list is that sibling's call.
        spelled = Tuple{String,Any,Any}[]
        for m in methods
            m isa AbstractDict || continue
            sym = String(get(m, "symbol", ""))
            isempty(sym) && continue
            dm = demangle(sym)
            dm === nothing && continue
            sp = _method_spelling(dm)
            sp === nothing && continue
            push!(spelled, (sym, m, sp))
        end
        full = Set{Any}((sp.type, sp.name, sp.isconst, sp.parameters) for (_, _, sp) in spelled)
        seen = Set{Any}()
        for (sym, m, sp) in spelled
            (get(m, "artificial", false) == true || get(m, "deleted", false) == true ||
             get(m, "accessibility", "public") != "public") && continue
            (startswith(sp.name, "~") || startswith(sp.name, "operator") || occursin('<', sp.type)) && continue
            isempty(sp.parameters) && continue
            ctor = sp.structor
            is_defined = ctor ? any(in(defined), _constructor_variants(sym)) : sym in defined
            if !isempty(exclude_patterns)
                dm = String(demangle(sym))
                names = ctor ? _constructor_variants(sym) : [sym]
                kept = _without_excluded_symbols(Set{String}(names),
                                                 [Dict{String,Any}("mangled" => v, "demangled" => dm) for v in names],
                                                 exclude_patterns)
                length(kept) < length(names) && continue
            end
            for k in (length(sp.parameters)-1):-1:0
                prefix = sp.parameters[1:k]
                key = (sp.type, sp.name, sp.isconst, prefix)
                (key in full || key in seen) && continue
                push!(seen, key)
                push!(rows, Dict{String,Any}(
                    "class" => name, "symbol" => sym, "type" => sp.type, "name" => sp.name, "ctor" => ctor,
                    "const" => sp.isconst, "parameters" => prefix, "full" => sp.parameters,
                    "omitted" => length(sp.parameters) - k,
                    "headers" => headers, "defined" => is_defined,
                    "demangled" => "$(sp.type)::$(sp.name)($(join(prefix, ", ")))" * (sp.isconst ? " const" : "")))
            end
        end
    end
    for (i, r) in enumerate(rows)
        r["id"] = i - 1
    end
    return Dict{String,Any}("candidates" => rows)
end

"The preamble of the unit's default-argument rows."
const _DEFAULT_ARGUMENT_PREAMBLE = raw"""
// ── Default arguments (see _default_argument_plan) ──────────────────────────
// Each function calls one constructor or member function with a prefix of its
// parameters, so the compiler fills in the defaults the header declares. A
// row whose call the header has no defaults for does not compile and is
// removed before this unit is written. `id<T>` lets any type spelling (a
// function pointer, an array) stand in a declarator.
namespace replibuild_force { template <class T> using id = T; }
// A member function's row also names the member through `nonstatic`, which
// has no overload for a plain function pointer: a STATIC member's row does
// not compile, since its function would take an object the member does not.
// Unevaluated, so naming the member defines nothing.
namespace replibuild_force {
template <class... A> struct nonstatic {
    template <class R, class C> static int m(R (C::*)(A...));
    template <class R, class C> static int mc(R (C::*)(A...) const);
};
}
"""

"One default-argument row's line in the unit: the function making the short call."
function _default_argument_line(f::AbstractDict)::String
    id = Int(f["id"])
    params = String[String(p) for p in f["parameters"]]
    decl = join(["replibuild_force::id<$p> a$(i - 1)" for (i, p) in enumerate(params)], ", ")
    args = join(["static_cast<replibuild_force::id<$p>&&>(a$(i - 1))" for (i, p) in enumerate(params)], ", ")
    self = (get(f, "const", false) == true ? "const " : "") * String(f["type"]) * "* self"
    sig = isempty(decl) ? self : "$self, $decl"
    tail = "   // $(f["symbol"]) less $(f["omitted"])"
    if get(f, "ctor", false) == true
        return "namespace replibuild_default { void d$(id)($sig) { ::new (static_cast<void*>(self)) $(f["type"])($args); } }" * tail
    end
    call = "self->$(f["name"])($args)"
    full = join(String[String(p) for p in get(f, "full", params)], ", ")
    member = "replibuild_force::nonstatic<$full>::$(get(f, "const", false) == true ? "mc" : "m")(&$(f["type"])::$(f["name"]))"
    return "namespace replibuild_default { auto d$(id)($sig) -> decltype((void)$member, $call) { return $call; } }" * tail
end

"The C++ preamble of the forced unit's member functions: the overload-selecting address helper."
const _INLINE_METHOD_PREAMBLE = raw"""
// ── Forced inline member functions (see generate_inline_definitions) ────────
// Never called. `args<A...>::m(&T::f)` picks, out of `f`'s overload set, the
// member taking exactly (A...) (a static member through the plain function
// pointer overload) and `mc` the `const` one, with the return type deduced:
// a function's mangled name does not carry it. Taking the address odr-uses
// the member, so clang emits its definition into this unit when the header
// has a body for it, and the `used` constant holding the address keeps that
// definition through every optimisation. C++11.
namespace replibuild_force {

template <class... A> struct args {
    template <class R, class C>
    static constexpr auto m(R (C::*f)(A...)) -> R (C::*)(A...) { return f; }
    template <class R>
    static constexpr auto m(R (*f)(A...)) -> R (*)(A...) { return f; }
    template <class R, class C>
    static constexpr auto mc(R (C::*f)(A...) const) -> R (C::*)(A...) const { return f; }
};

}  // namespace replibuild_force
#define RB_FORCE_MEMBER __attribute__((used)) static const auto
"""

"One forced member function's line in the unit: its address, held by a `used` constant."
function _inline_method_line(i::Integer, f::AbstractDict)::String
    params = join(String[String(p) for p in f["parameters"]], ", ")
    sel = get(f, "const", false) == true ? "mc" : "m"
    return "RB_FORCE_MEMBER replibuild_member_$(i) = replibuild_force::args<$params>::$sel(" *
           "&$(f["type"])::$(f["name"]));   // $(f["symbol"])"
end

"The C++ preamble of the inline-constructor unit: the guarded placement-new helper."
const _INLINE_CTOR_PREAMBLE = raw"""
// ── Forced inline constructors (see generate_inline_constructors) ───────────
// Never called. An explicit instantiation of `ctor<T, A...>` odr-uses T's
// constructor from (A...) through placement new, so clang emits the real
// C1/C2 definitions into this unit. The guard emits nothing for an abstract
// class or a constructor not callable from here (private, protected, deleted,
// or no constructor taking exactly these arguments), so the unit compiles for
// those too. C++11; `-fno-inline` on this unit keeps the constructor a
// definition of its own instead of a body inlined into `go`.
#include <new>
#include <type_traits>

namespace replibuild_force {

template <bool ok, class T, class... A> struct ctor_impl {
    static void go(void* p, A... a) { ::new (p) T(static_cast<A&&>(a)...); }
};
template <class T, class... A> struct ctor_impl<false, T, A...> {
    static void go(void*, A...) {}
};

template <class T, class... A>
void ctor(void* p, A... a) {
    ctor_impl<!std::is_abstract<T>::value && std::is_constructible<T, A...>::value, T, A...>::go(
        p, static_cast<A&&>(a)...);
}

}  // namespace replibuild_force
"""

"""
    _inline_ctor_unit_text(headers, forced) -> String

The unit: one `#include` per header (a name in `<…>` as is, anything else
quoted; absolute paths come from the records' `decl_files`), the preamble, and
one explicit instantiation per forced row, `template void
replibuild_force::ctor<T, A...>(void*, A...);`, the row's symbol beside it.
With `methods` (the inline-method plan's forced rows), the member preamble
and one `RB_FORCE_MEMBER` line per row follow. Every row is exactly one line,
which is how a compiler diagnostic is traced back to its row
(`_unit_row_of_line`).
"""
function _inline_ctor_unit_text(headers::AbstractVector, forced::AbstractVector,
                                methods::AbstractVector = Any[], defaults::AbstractVector = Any[])::String
    io = IOBuffer()
    println(io, "// Auto-generated by RepliBuild: the library's inline constructors and member functions that no translation unit of it defined")
    for h in headers
        hs = String(h)
        println(io, startswith(hs, "<") ? "#include $hs" : "#include \"$hs\"")
    end
    println(io)
    print(io, _INLINE_CTOR_PREAMBLE)
    println(io)
    for f in forced
        params = String[String(p) for p in f["parameters"]]
        args = join(vcat([String(f["type"])], params), ", ")
        sig = join(vcat(["void*"], params), ", ")
        println(io, "template void replibuild_force::ctor<$args>($sig);   // $(f["symbol"])")
    end
    if !isempty(methods)
        println(io)
        print(io, _INLINE_METHOD_PREAMBLE)
        println(io)
        for (i, f) in enumerate(methods)
            println(io, _inline_method_line(i - 1, f))
        end
    end
    if !isempty(defaults)
        println(io)
        print(io, _DEFAULT_ARGUMENT_PREAMBLE)
        println(io)
        for f in defaults
            println(io, _default_argument_line(f))
        end
    end
    return String(take!(io))
end

"The headers the unit includes: `[wrap] shim_headers` first (the library's umbrella, where one is configured), then every forced record's headers."
_inline_ctor_headers(config::RepliBuildConfig, forced::AbstractVector) =
    unique(vcat(String[String(h) for h in config.wrap.shim_headers],
                sort!(unique(String[String(h) for f in forced for h in f["headers"]]))))

"Write `text` to `path` only when it differs, so an unchanged unit keeps its IR cache entry."
function _write_if_changed(path::String, text::String)
    (isfile(path) && read(path, String) == text) && return false
    write(path, text)
    return true
end

"Does `text`, written to `path`, compile (`-fsyntax-only`, the project's flags)? `(ok, first error line)`."
function _unit_compiles(config::RepliBuildConfig, path::String, text::String)
    write(path, text)
    (ok, out) = _template_tu_compiles(config, path)
    lines = split(out, '\n')
    k = findfirst(l -> occursin("error", l), lines)
    msg = k === nothing ? strip(first(out, 200)) : strip(lines[k])
    return (ok, String(msg))
end

"How many rows `generate_inline_definitions` first tries together when it must find the ones that do not compile."
const _ALONE_GROUP_SIZE = 16

"""
    _alone_verdicts(probe, n; group = _ALONE_GROUP_SIZE) -> Vector{Union{Nothing,String}}

For each of `n` rows, what `probe` says of the row tried alone: `nothing` when
it compiles, else its error. `probe(rows, round, k)` compiles the rows whose
indices it is given together (`round` and `k` number the call, for a file
name) and returns `nothing` or an error.

Every compile of the forced unit parses the whole include block, and few rows
fail: 73 of Jolt's 2,139 constructors, which one compile per row turned into
2,139 compiles and most of a ten-minute build (2026-11-17). So the rows are
tried `group` at a time, each round in parallel: a group that compiles clears
every row in it, and one that does not is halved and tried again, down to
single rows. Only a row tried alone is ever given an error, so the verdicts
are those of one compile per row wherever rows that compile together also
compile apart, which holds for explicit instantiations and address-of
initialisers: adding one cannot make another's errors go away. Where it does
not hold, the row is kept, and the unit compiled whole is what decides.
"""
function _alone_verdicts(probe, n::Int; group::Int = _ALONE_GROUP_SIZE)
    bad = Vector{Union{Nothing,String}}(nothing, n)
    groups = Vector{Int}[collect(g) for g in Iterators.partition(1:n, max(group, 1))]
    round = 0
    while !isempty(groups)
        round += 1
        verdicts = Vector{Union{Nothing,String}}(nothing, length(groups))
        let groups = groups, round = round
            Threads.@threads for k in eachindex(groups)
                verdicts[k] = probe(groups[k], round, k)
            end
        end
        next = Vector{Int}[]
        for (k, g) in enumerate(groups)
            verdicts[k] === nothing && continue
            if length(g) == 1
                bad[g[1]] = verdicts[k]
            else
                h = length(g) ÷ 2
                push!(next, g[1:h], g[h+1:end])
            end
        end
        groups = next
    end
    return bad
end

"The sidecar naming the forced definitions' symbols, read at link time (`_forced_constructor_symbols`)."
_forced_symbols_path(config::RepliBuildConfig) = joinpath(get_cache_path(config), "replibuild_ctors.json")

"""
    _unit_probe(config, path, text; ir = nothing) -> (ok, output)

Compile `text`, written to `path`, with the project's flags and no error
limit: `-fsyntax-only`, or to LLVM IR at `ir` (`-O0`, no debug info), which
is what says whether the unit DEFINES a member or only declares it.
"""
function _unit_probe(config::RepliBuildConfig, path::String, text::String; ir::Union{Nothing,String} = nothing)
    write(path, text)
    probe = _probe_compiler(:cpp)
    flags = get_compile_flags(config)
    incs  = ["-I$dir" for dir in get_include_dirs(config)]
    defs  = ["-D$(k)=$(v)" for (k, v) in config.compile.defines]
    mode  = ir === nothing ? ["-fsyntax-only"] : ["-O0", "-g0", "-S", "-emit-llvm", "-o", ir]
    errbuf = IOBuffer()
    try
        p = run(pipeline(ignorestatus(`$probe $flags $incs $defs -ferror-limit=0 $mode $path`),
                         stdout = devnull, stderr = errbuf))
        return (p.exitcode == 0, String(take!(errbuf)))
    catch e
        return (false, "probe compiler unavailable: $e")
    end
end

"""
    _unit_bad_lines(output, path) -> Dict{Int,String}

The lines of the unit at `path` a compiler's diagnostics name, each with the
error it belongs to. An error at a line of the unit is that line's; a note
there (`in instantiation of … requested here`) belongs to the error printed
before it, which sits in a header.
"""
function _unit_bad_lines(output::AbstractString, path::AbstractString)::Dict{Int,String}
    bad = Dict{Int,String}()
    last_error = ""
    unit = basename(path)
    for line in split(output, '\n')
        m = match(r"^(.*?):(\d+):\d+: (fatal error|error|note): (.*)$", line)
        m === nothing && continue
        kind = m.captures[3]
        kind == "note" || (last_error = String(strip(m.captures[4])))
        basename(String(m.captures[1])) == unit || continue
        get!(bad, parse(Int, m.captures[2]), isempty(last_error) ? String(strip(m.captures[4])) : last_error)
    end
    return bad
end

"The row a line of the unit states: `(:ctor, k)`, `(:method, k)` or `(:default, k)`, 1-based in its list, else `nothing`."
function _unit_row_of_line(text::AbstractString)
    rows = Dict{Int,Tuple{Symbol,Int}}()
    nc = 0
    nm = 0
    nd = 0
    for (i, line) in enumerate(split(text, '\n'))
        if startswith(line, "template void replibuild_force::ctor<")
            rows[i] = (:ctor, nc += 1)
        elseif startswith(line, "RB_FORCE_MEMBER replibuild_member_")
            rows[i] = (:method, nm += 1)
        elseif startswith(line, "namespace replibuild_default { ")
            rows[i] = (:default, nd += 1)
        end
    end
    return rows
end

"The names an IR file defines and the names it only declares."
function _ir_defined_declared(ir_path::AbstractString)
    defined = Set{String}()
    declared = Set{String}()
    for line in eachline(ir_path)
        isdef = startswith(line, "define ")
        (isdef || startswith(line, "declare ")) || continue
        m = match(r"@((?:[\w\.\$]|\\[0-9a-fA-F]{2})+|\"[^\"]*\")\(", line)
        m === nothing && continue
        push!(isdef ? defined : declared, String(strip(m.captures[1], '"')))
    end
    return (defined, declared)
end

"""
    generate_inline_definitions(config, cpp_files, plan, mplan) -> (cpp_files, plan, mplan)

Write the forced unit for the constructor plan's and the member-function
plan's `forced` rows (`_inline_constructor_plan`, `_inline_method_plan`) and
the default-argument candidates (`_default_argument_plan`) under the build
cache and append it to `cpp_files`; each plan comes back with the rows the
unit could not keep moved to `skipped`, and the default-argument plan with
the rows that compiled as `bound`, each with its function's symbol (`shim`).

The unit is compiled whole to IR. When that fails, every row a diagnostic
names (`_unit_bad_lines`: each row is one line) is skipped with the
compiler's error and the unit is compiled again. Before any of that the
include block is compiled by itself: when it fails, each header is checked
alone, in parallel, and dropped from every row where it does not compile on
its own (a row left with no header is skipped for that reason, never as a
row that "does not compile"). When the rows' diagnostics name no row, the
same header check runs if it has not, and each remaining constructor is
checked alone, in parallel (in groups first, halved where one fails, so a
row that compiles rarely costs a compile of its own); if that still leaves
errors naming no row, each member function is. From the IR of the unit
that compiles, a member function the unit only DECLARES has no body in its
header (an out-of-line member the library keeps hidden, or one defined in a
file this build does not compile): taking its address would leave the library
with an undefined symbol, so the row is skipped. A C++ package only; nothing
forced returns `cpp_files` as given and removes a stale unit's sidecar. The
symbols of the kept rows are written beside the unit for the link
(`_forced_exportable_ir`).
"""
function generate_inline_definitions(config::RepliBuildConfig, cpp_files::Vector{String},
                                     plan::AbstractDict, mplan::AbstractDict,
                                     dplan::AbstractDict = Dict{String,Any}("candidates" => Any[]))
    copyrows(p, k) = Dict{String,Any}[Dict{String,Any}(f) for f in get(p, k, Any[])]
    forced = copyrows(plan, "forced")
    skipped = copyrows(plan, "skipped")
    mforced = copyrows(mplan, "forced")
    mskipped = copyrows(mplan, "skipped")
    # The default-argument rows (`_default_argument_plan`): most of them are
    # calls the header has no defaults for, dropped below without a word.
    dforced = copyrows(dplan, "candidates")
    result() = (Dict{String,Any}("forced" => forced, "skipped" => skipped),
                Dict{String,Any}("forced" => mforced, "skipped" => mskipped),
                Dict{String,Any}("candidates" => Any[], "bound" => dforced))
    cache_dir = get_cache_path(config)
    mkpath(cache_dir)
    unit_path = joinpath(cache_dir, "replibuild_ctors.cpp")
    rm(_forced_symbols_path(config); force = true)
    if !(config.wrap.language == :cpp && !(isempty(forced) && isempty(mforced) && isempty(dforced)))
        empty!(dforced)
        return (cpp_files, result()...)
    end

    probe = joinpath(cache_dir, "replibuild_ctors_probe.cpp")
    probe_ir = joinpath(cache_dir, "replibuild_ctors_probe.ll")
    shim_headers = String[String(h) for h in config.wrap.shim_headers]
    row(f) = Dict{String,Any}("class" => f["class"], "symbol" => f["symbol"])
    # Keep the rows `reason_of` answers `nothing` for; skip the rest, named.
    function drop!(rows, into, reason_of)
        keep = Dict{String,Any}[]
        for (k, f) in enumerate(rows)
            r = reason_of(k, f)
            r === nothing ? push!(keep, f) : push!(into, merge(row(f), Dict{String,Any}("reason" => r)))
        end
        return keep
    end
    unit_text() = _inline_ctor_unit_text(_inline_ctor_headers(config, vcat(forced, mforced, dforced)),
                                         forced, mforced, dforced)
    # Each row of `rows` compiled alone (`-fsyntax-only`): the compiler's first
    # error line per row, or `nothing`. The rows are tried in groups first
    # (`_alone_verdicts`); a row's verdict and its text are still its own
    # compile's, from the same file name (the error line can name it).
    function alone(rows, ctors::Bool)
        all_headers = _inline_ctor_headers(config, vcat(forced, mforced))
        kind = ctors ? "c" : "m"
        return _alone_verdicts(length(rows)) do g, round, k
            path = joinpath(cache_dir, length(g) == 1 ? "replibuild_ctors_probe_$(kind)_$(g[1]).cpp" :
                                                        "replibuild_ctors_probe_$(kind)_g$(round)_$(k).cpp")
            text = ctors ? _inline_ctor_unit_text(all_headers, rows[g]) :
                           _inline_ctor_unit_text(all_headers, Any[], rows[g])
            (rok, rout) = _unit_probe(config, path, text)
            rm(path; force = true)
            rok && return nothing
            lines = split(rout, '\n')
            e = findfirst(l -> occursin("error", l), lines)
            return e === nothing ? String(strip(first(rout, 200))) : String(strip(lines[e]))
        end
    end
    first_error(out) = (lines = split(out, '\n'); k = findfirst(l -> occursin("error", l), lines);
                        k === nothing ? String(strip(first(out, 200))) : String(strip(lines[k])))
    # Each header alone (the umbrella beside it, as in the unit), in parallel;
    # one that does not compile on its own leaves every row's list, and a row
    # left with none is skipped for that reason. Done once.
    headers_checked = false
    function check_headers!()
        headers_checked && return
        headers_checked = true
        hs = sort!(unique(String[h for f in vcat(forced, mforced, dforced) for h in f["headers"]]))
        verdict = Vector{Union{Nothing,String}}(nothing, length(hs))
        Threads.@threads for i in eachindex(hs)
            path = joinpath(cache_dir, "replibuild_ctors_probe_h_$(i).cpp")
            (hok, hout) = _unit_probe(config, path, _inline_ctor_unit_text(vcat(shim_headers, [hs[i]]), Any[]))
            hok || (verdict[i] = first_error(hout))
            rm(path; force = true)
        end
        bad_header = Dict{String,String}(hs[i] => verdict[i] for i in eachindex(hs) if verdict[i] !== nothing)
        isempty(bad_header) && return
        for f in vcat(forced, mforced, dforced)
            f["was_headers"] = f["headers"]
            f["headers"] = String[h for h in f["headers"] if !haskey(bad_header, h)]
        end
        function no_header(_, f)
            isempty(f["headers"]) || return nothing
            h = first(sort!(String[x for x in f["was_headers"] if haskey(bad_header, x)]))
            return "its header does not compile on its own ($(basename(h)): $(bad_header[h]))"
        end
        forced = drop!(forced, skipped, no_header)
        mforced = drop!(mforced, mskipped, no_header)
        dforced = Dict{String,Any}[f for f in dforced if !isempty(f["headers"])]
        foreach(f -> delete!(f, "was_headers"), vcat(forced, mforced, dforced))
        return
    end
    # The include block by itself, before any row is judged: a header that
    # is half of a pair (Jolt's `WrapShaderBegin.h` opens namespaces its
    # partner closes) leaves every header after it inside its scope, and the
    # errors then land on the lines of rows that are fine. Judged first, each
    # of those rows was skipped as "does not compile" (1,944 of Jolt's
    # constructors, every one its HelloWorld needs) before the per-header
    # check, which ran only once no row was named, could see the culprit.
    let (inc_ok, _) = _unit_probe(config, probe, _inline_ctor_unit_text(
            _inline_ctor_headers(config, vcat(forced, mforced, dforced)), Any[]))
        inc_ok || check_headers!()
    end
    text = unit_text()
    ok = false
    msg = ""
    # What was tried once no diagnostic named a row: 0 nothing yet, 1 the
    # headers and each constructor alone, 2 each member function alone.
    stage = 0
    for _ in 1:12
        (isempty(forced) && isempty(mforced) && isempty(dforced)) && break
        (ok, out) = _unit_probe(config, probe, text; ir = probe_ir)
        ok && break
        bad = _unit_bad_lines(out, probe)
        rows = _unit_row_of_line(text)
        bad_ctor = Dict{Int,String}()
        bad_method = Dict{Int,String}()
        bad_default = Set{Int}()
        for (line, why) in bad
            r = get(rows, line, nothing)
            r === nothing && continue
            r[1] === :default ? push!(bad_default, r[2]) :
                ((r[1] === :ctor ? bad_ctor : bad_method)[r[2]] = why)
        end
        if !isempty(bad_default)
            # No default for the omitted parameters (or an ambiguous call):
            # the expected outcome for most rows, so nothing is recorded.
            dforced = Dict{String,Any}[f for (k, f) in enumerate(dforced) if !(k in bad_default)]
        end
        if !(isempty(bad_ctor) && isempty(bad_method))
            forced = drop!(forced, skipped, (k, _) -> haskey(bad_ctor, k) ? "does not compile: $(bad_ctor[k])" : nothing)
            mforced = drop!(mforced, mskipped, (k, _) -> haskey(bad_method, k) ? "does not compile: $(bad_method[k])" : nothing)
        elseif !isempty(bad_default)
            # Only default-argument rows were named this round.
        elseif stage == 0
            stage = 1
            # No row is named: a header does not compile on its own
            # (`check_headers!`, unless the include block already asked)...
            check_headers!()
            # ...then each remaining constructor alone, against every header
            # that compiles, in parallel: a class template's specialization
            # needs its arguments' headers too. A constructor's error sits
            # inside the placement-new helper and clang prints no note for the
            # explicit instantiation that asked for it, so no line names its
            # row; a member function's address is taken ON its row.
            bc = alone(forced, true)
            forced = drop!(forced, skipped, (k, _) -> bc[k] === nothing ? nothing : "does not compile: $(bc[k])")
        elseif stage == 1
            stage = 2
            bm = alone(mforced, false)
            mforced = drop!(mforced, mskipped, (k, _) -> bm[k] === nothing ? nothing : "does not compile: $(bm[k])")
        else
            lines = split(out, '\n')
            k = findfirst(l -> occursin("error", l), lines)
            msg = k === nothing ? String(strip(first(out, 200))) : String(strip(lines[k]))
            break
        end
        text = unit_text()
    end
    if !ok
        forced = drop!(forced, skipped, (_, _) -> "the unit did not compile: $msg")
        mforced = drop!(mforced, mskipped, (_, _) -> "the unit did not compile: $msg")
        dforced = Dict{String,Any}[]
    end
    if ok && !isempty(mforced)
        # A member the unit only declares has no body in its header.
        (defined, declared) = _ir_defined_declared(probe_ir)
        n = length(mforced)
        mforced = drop!(mforced, mskipped, (_, f) ->
            (String(f["symbol"]) in declared && !(String(f["symbol"]) in defined)) ?
            "declared in its header with no body there" : nothing)
        length(mforced) < n && (text = unit_text())
    end
    if ok && !isempty(dforced)
        # A row stays when the member it calls is defined: by the library, or
        # by this unit (an inline member the call itself made clang emit). A
        # call to a member defined nowhere would leave an undefined symbol.
        # Each kept row learns its function's symbol from the unit's IR.
        (defined, _) = _ir_defined_declared(probe_ir)
        shim_of = Dict{String,String}()
        for d in defined
            m = match(r"^_ZN18replibuild_default\d+d(\d+)E", d)
            m === nothing || (shim_of[String(m.captures[1])] = d)
        end
        n = length(dforced)
        kept = Dict{String,Any}[]
        for f in dforced
            target = String(f["symbol"])
            names = get(f, "ctor", false) == true ? _constructor_variants(target) : [target]
            (get(f, "defined", false) == true || any(in(defined), names)) || continue
            shim = get(shim_of, string(Int(f["id"])), nothing)
            shim === nothing && continue
            f["shim"] = shim
            push!(kept, f)
        end
        dforced = kept
        length(dforced) < n && (text = unit_text())
    end
    rm(probe; force = true)
    rm(probe_ir; force = true)
    (isempty(forced) && isempty(mforced) && isempty(dforced)) && return (cpp_files, result()...)

    _write_if_changed(unit_path, text)
    symbols = sort!(unique(vcat(String[String(f["symbol"]) for f in mforced],
                                String[String(f["shim"]) for f in dforced],
                                [_constructor_variants(String(f["symbol"])) for f in forced]...)))
    open(_forced_symbols_path(config), "w") do io
        JSON.print(io, symbols)
    end
    return (vcat(cpp_files, [unit_path]), result()...)
end

"""
    generate_inline_constructors(config, cpp_files, plan) -> (cpp_files, plan)

`generate_inline_definitions` for a constructor plan alone.
"""
function generate_inline_constructors(config::RepliBuildConfig, cpp_files::Vector{String}, plan::AbstractDict)
    (files, p, _, _) = generate_inline_definitions(config, cpp_files, plan,
                                                   Dict{String,Any}("forced" => Any[], "skipped" => Any[]))
    return (files, p)
end

"The forced definitions' symbols the current build's unit names, or none."
function _forced_constructor_symbols(config::RepliBuildConfig)::Set{String}
    path = _forced_symbols_path(config)
    isfile(path) || return Set{String}()
    names = try
        _read_json(path)
    catch
        return Set{String}()
    end
    return names isa AbstractVector ? Set{String}(String(n) for n in names) : Set{String}()
end

# IR keywords that may precede the type on a `define` line or the aliasee on an
# alias line: linkage, preemption, visibility, DLL storage, thread-local and
# unnamed_addr. `dllexport` is inserted after the last of them.
const _IR_GLOBAL_PREFIX_KEYWORDS = Set([
    "private", "internal", "available_externally", "linkonce", "weak", "common", "appending",
    "extern_weak", "linkonce_odr", "weak_odr", "external", "dso_local", "dso_preemptable",
    "default", "hidden", "protected", "dllimport", "dllexport", "thread_local", "localdynamic",
    "initialexec", "localexec", "unnamed_addr", "local_unnamed_addr"])

"""
    _export_global_line(line, symbols; pe) -> String

`line` with the definition or alias of a symbol in `symbols` made exportable:
`hidden` dropped, and on PE (`pe`) `dllexport` (with `dso_local`, which the
verifier asks of it) inserted after the leading keywords. Any other line, or a
symbol not in the set, comes back as it was.
"""
function _export_global_line(line::AbstractString, symbols::AbstractSet{String}; pe::Bool)
    s = String(line)
    is_define = startswith(s, "define ")
    is_alias = startswith(s, "@") && occursin(" alias ", s)
    (is_define || is_alias) || return s
    name = if is_define
        m = match(r"@((?:[\w\.\$]|\\[0-9a-fA-F]{2})+|\"[^\"]*\")\(", s)
        m === nothing ? nothing : m.captures[1]
    else
        m = match(r"^@((?:[\w\.\$]|\\[0-9a-fA-F]{2})+|\"[^\"]*\") = ", s)
        m === nothing ? nothing : m.captures[1]
    end
    name === nothing && return s
    strip(name, '"') in symbols || return s
    head_end = is_define ? 7 : findfirst(" = ", s).stop
    head = s[1:head_end]
    rest = s[head_end+1:end]
    toks = split(rest, ' '; limit = typemax(Int))
    i = 1
    while i <= length(toks) && toks[i] in _IR_GLOBAL_PREFIX_KEYWORDS
        i += 1
    end
    lead = String[t for t in toks[1:i-1] if t != "hidden"]
    if pe
        # The grammar's order: linkage, preemption, visibility, DLL storage,
        # thread-local, unnamed_addr. `dso_local` after the linkage keyword,
        # `dllexport` before anything of the last two groups.
        linkage_kw = ("private", "internal", "available_externally", "linkonce", "weak", "common",
                      "appending", "extern_weak", "linkonce_odr", "weak_odr", "external")
        tail_kw = ("thread_local", "localdynamic", "initialexec", "localexec",
                   "unnamed_addr", "local_unnamed_addr")
        if !("dso_local" in lead)
            k = findfirst(in(linkage_kw), lead)
            insert!(lead, k === nothing ? 1 : k + 1, "dso_local")
        end
        if !("dllexport" in lead)
            j = findfirst(in(tail_kw), lead)
            insert!(lead, j === nothing ? length(lead) + 1 : j, "dllexport")
        end
    end
    return head * join(vcat(lead, toks[i:end]), ' ')
end

"""
    _forced_exportable_ir(files, symbols, out_dir; pe) -> Vector{String}

Copies of the IR `files` with the forced constructors' definitions
(`symbols`: the `C1`/`C2`/`C3` names of the unit's rows) made exportable
(`_export_global_line`), written to `out_dir`; a file naming none is returned
as itself. Under `-fvisibility=hidden` (or `-fvisibility-inlines-hidden`) the
unit's definitions are hidden like any inline member, and a hidden definition
never reaches the dynamic table (`verify_wrap_surface` R2) or, on PE, the
export directory. The copy is rewritten on every link: its content depends on
the symbol set, not only on the input's mtime.
"""
function _forced_exportable_ir(files::Vector{String}, symbols::AbstractSet{String}, out_dir::String; pe::Bool)
    isempty(symbols) && return (files, 0)
    out = String[]
    total = 0
    # `_export_global_line` returns any line that is not the definition or alias
    # of a symbol in the set as it was, and finds the name with one set lookup.
    # (A scan of every symbol against every `define` line stood in front of it
    # as a filter and could only agree with it: 38 of the 80 seconds of Jolt's
    # second link, 2026-11-17.)
    for f in files
        # A file whose lines the rewrite would leave as they are is returned
        # as itself: a default-visibility build has nothing to change.
        changes = 0
        open(f) do io
            for line in eachline(io)
                _export_global_line(line, symbols; pe = pe) != line && (changes += 1)
            end
        end
        if changes == 0
            push!(out, f)
            continue
        end
        mkpath(out_dir)
        dst = joinpath(out_dir, basename(f))
        open(f) do src
            open(dst, "w") do io
                for line in eachline(src; keep = true)
                    body = chomp(line)
                    nl = line[lastindex(body)+1:end]
                    write(io, _export_global_line(body, symbols; pe = pe), nl)
                end
            end
        end
        total += changes
        push!(out, dst)
    end
    return (out, total)
end


# =============================================================================
# MACRO-SHIM HEADER-COLLISION GUARD
# -----------------------------------------------------------------------------
# The generated macro-shim TU (replibuild_shims.c) is emitted under the build
# cache, NOT inside the vendored source tree, so a bare `#include "foo.h"` cannot
# resolve foo.h as a sibling the way the library's own TUs do — it rides the -I
# search path and can fall through to a system-installed copy of foo.h at a
# DIFFERENT version, silently baking wrong [wrap.macros] values with no other
# symptom. (BLAKE3 hit exactly this: /usr/include/blake3.h from libblake3-1.8.4
# shadowed the vendored 1.8.5 header, so BLAKE3_VERSION_STRING came out "1.8.4".)
# These verify each direct shim include resolves inside the project/dep tree.
# =============================================================================

"The compiler the shim TU is really built with — JLL clang for C (version-locked to
Julia's libLLVM), system clang++ for C++ — so the probe's include resolution matches."
function _probe_compiler(language::Symbol)::Cmd
    if language == :c
        try
            Clang_mod = Base.require(Base.PkgId(
                Base.UUID("40e3b903-d033-50b4-a0cc-940c62c95e31"), "Clang"))
            if isdefined(Clang_mod, :Clang_unified_jll)
                clang_cmd = Clang_mod.Clang_unified_jll.clang()
                # Same sysroot the real compile gets, or the probe would resolve a
                # different set of headers than the build it is vouching for.
                sysroot = _c_bucket_sysroot()
                return isempty(sysroot) ? clang_cmd : `$clang_cmd --sysroot=$sysroot`
            end
        catch
        end
        return `clang`
    else
        return `clang++`
    end
end

"""
    _canon_path(path) -> String

Compare paths in one spelling. On Windows `realpath` hands back `\\` and the
filesystem is case-insensitive, so a Unix-style `startswith(p, root * "/")`
never matches.
"""
_canon_path(p::AbstractString) = Sys.iswindows() ? lowercase(replace(String(p), '\\' => '/')) : String(p)

"""
Return the resolved absolute paths of `shim_file`'s DIRECT includes that fall
OUTSIDE `allowed_roots`. Empty ⇒ every shim header came from the project/dep tree.
A shim header that does not resolve at all (missing) is a compile error, not a
collision, and is deliberately NOT reported here.
"""
function _shim_headers_out_of_tree(shim_file::String, probe::Cmd,
                                   extra_flags::Vector{String},
                                   allowed_roots::Vector{String})::Vector{String}
    # Compare paths in one canonical spelling. On Windows `realpath` hands back
    # `\` separators and the filesystem is case-insensitive, so the containment
    # test below — which was written as `startswith(rp, root * "/")` — could
    # never match: every resolved header looked out-of-tree, and the guard
    # flagged the correctly-vendored header as a collision. That fails the build
    # for exactly the packages the guard exists to protect.
    #
    # Only the comparison is normalized; `offenders` keeps the real path, since
    # that string is what the error message shows the user.
    _canon(p) = _canon_path(p)

    norm_roots = String[]
    for r in allowed_roots
        try
            push!(norm_roots, _canon(realpath(abspath(r))))
        catch
        end
    end

    errbuf = IOBuffer()
    try
        run(pipeline(ignorestatus(`$probe -fsyntax-only -H $extra_flags $shim_file`),
                     stdout = devnull, stderr = errbuf))
    catch
        return String[]   # probe compiler unavailable → never block the build on the guard
    end

    offenders = String[]
    for line in split(String(take!(errbuf)), '\n')
        # `-H` prints the include tree; a SINGLE leading dot marks a direct include
        # of the TU (one of the shim headers). `..`/deeper are transitive includes.
        m = match(r"^\.[ \t]+(.+)$", line)
        m === nothing && continue
        inc = strip(String(m.captures[1]))
        rp = try realpath(inc) catch; abspath(inc) end
        rpc = _canon(rp)
        if !any(root -> rpc == root || startswith(rpc, root * "/"), norm_roots)
            push!(offenders, rp)
        end
    end
    return offenders
end

"""
    _shim_allowed_roots(config) -> Vector{String}

Where a shim header may resolve to: the project root, the build cache (git
clones and unpacked archives live under it), and the path of every local
dependency, which is the project's own by declaration wherever it sits. A
local dependency outside the root used to be refused like a system header
(2026-11-07).
"""
function _shim_allowed_roots(config::RepliBuildConfig)::Vector{String}
    roots = [config.project.root, get_cache_path(config)]
    for dep in values(config.dependencies.items)
        dep.type == "local" && !isempty(dep.path) &&
            push!(roots, abspath(joinpath(config.project.root, dep.path)))
    end
    return roots
end

"""
    verify_shim_headers(config, shim_file)

Hard-error if the generated macro shim resolved any [wrap] shim_header to a path
outside the project/dependency tree (see the section comment above). No-op when the
package declares no macros or shim defines, or no shim headers, or when the shim
file is absent.
"""
function verify_shim_headers(config::RepliBuildConfig, shim_file::String)
    isempty(config.wrap.macros) && isempty(config.wrap.shim_defines) && return
    isempty(config.wrap.shim_headers) && return
    isfile(shim_file) || return

    probe    = _probe_compiler(config.wrap.language)
    incflags = ["-I$dir" for dir in get_include_dirs(config)]
    # -I/-D carried in [compile].flags steer which header wins and which #if branch
    # of it the shim sees; the [compile].defines table feeds the same resolution.
    userID   = filter(f -> startswith(f, "-I") || startswith(f, "-D"),
                      get_compile_flags(config))
    defflags = ["-D$(k)=$(v)" for (k, v) in config.compile.defines]
    extra    = vcat(userID, incflags, defflags)

    offenders = _shim_headers_out_of_tree(shim_file, probe, extra, _shim_allowed_roots(config))
    isempty(offenders) && return

    error("""
    RepliBuild: macro-shim header collision detected.

    The generated macro shim resolved a [wrap] shim_header to a path OUTSIDE the
    project/dependency tree:

    $(join(["    → " * o for o in unique(offenders)], "\n"))

    This compiles the shim against a foreign copy of the header (typically a
    system-installed -dev package at a different version) and bakes wrong values
    into every [wrap.macros] entry, or builds that copy's implementation for
    [wrap] shim_defines — with no other symptom.

    Fix in the package replibuild.toml:
      • reference the header by its in-clone subpath in [wrap] shim_headers
        (e.g. "c/blake3.h" instead of "blake3.h"), or
      • add the directory holding the vendored header to [compile] flags as
        -I<dir> so it precedes the system include path.
    """)
end

"""
    verify_wrap_surface(config, binary_path, metadata_path)

Guard: every function the wrapper is about to emit must be resolvable in the built
library's **dynamic** symbol table, under the exact symbol name the wrapper will
`ccall`.

**The wrap surface and the runtime surface come from two different symbol tables and
nothing has ever checked that they agree.** `extract_compilation_metadata` correlates
DWARF against `extract_symbols_from_binary`, which reads `nm -g` — the full `.symtab`.
The generated wrapper resolves through `dlsym`, which sees only `.dynsym`
(`extract_symbols_nm` uses `nm -D` for exactly this reason). A symbol present in one
and absent from the other is wrapped, documented, exported — and fails at the first
call. Today the two agree everywhere; the check is cheap and the disagreement is
silent, which is the whole argument for asserting it.

**`promote_statics` is what makes them disagree.** It treats hidden visibility as
"internal" (`_promote_statics_libllvm`: `needs_promotion` tests linkage OR
hidden/protected visibility) and renames such definitions to `__rb_<lib>_<name>`,
which by design drops them from the wrap surface — DWARF still names the original, so
the renamed symbol has no DWARF match. That is correct while `hidden` means what the
library's author meant by it.

Compile with `-fvisibility=hidden` and it stops meaning that. Every function the
library did not annotate `visibility("default")` goes hidden — **including public API
whose export macro is a bare `extern`**, which is lua's `LUA_API` and is common. Then
promotion renames the entire API to `__rb_*` and the wrapper generates nothing, from a
build that succeeded and warned about nothing. Reproduced 2026-09-15 on a library-free
three-function fixture through the JLL clang: `annotated_public` (carrying
`__attribute__((visibility("default")))`) survives under its own name; `bare_extern_public`
and `uses_helper` both come back as `__rb_fix_*`. The first of those is API.

`test_cache_invalidation.jl` records the same interaction from the other side —
"`internal_helper` under that flag is external-linkage-and-hidden (the LUAI_FUNC
shape), so static promotion re-exports it as `__rb_cachetest_internal_helper` … assert
names, never export counts, on anything downstream of promotion." **This guard asserts
names.** The one count it reads is the degenerate zero.

Two rules, neither carrying a threshold:

  * **R1 — collapse.** The wrap surface is empty while promotion renamed symbols. That
    is the failure above, observed rather than inferred: something was compiled, and
    all of it left under a different name.
  * **R2 — reachability.** Every wrappable function appears in the dynamic table under
    its `mangled` name. Measured across all 31 built Hub packages 2026-09-15 —
    **0 offenders**, including every C++ package (constructor clones and weak template
    instantiations are all present in `.dynsym`), so a non-empty offender list is news
    rather than noise.

Runs BEFORE `save_project_hash`: a build that fails this must not leave a cache marker
claiming it succeeded, or the next run skips straight past the failure.
"""
function verify_wrap_surface(config::RepliBuildConfig, binary_path::String,
                             metadata_path::Union{Nothing,String})
    config.binary.type == :executable && return
    (isnothing(metadata_path) && return); isfile(metadata_path) || return
    isfile(binary_path) || return

    metadata = try
        _read_json(metadata_path)
    catch e
        @debug "wrap-surface guard: metadata unreadable" exception=e
        return
    end
    functions = get(metadata, "functions", nothing)
    functions isa Vector || return

    # `dlsym` sees the dynamic table on ELF; a PE has none, and the export directory
    # is the equivalent authority there.
    reachable = _runtime_symbol_names(binary_path)
    reachable === nothing && return   # cannot ask the question ⇒ assert nothing
    isempty(reachable) && return

    promoted_count = count(startswith(s, "__rb_") for s in reachable)

    # R1 — the degenerate case, and the only place a count is read.
    #
    # This used to require `promoted_count > 0`, on the theory that promotion was
    # how a surface collapsed. That made the check VACUOUS for the commoner
    # shape: `-fvisibility=hidden` with `promote_statics` OFF hides the API
    # without renaming anything, so `promoted_count` is 0, `functions` is empty,
    # and R2 below — which only inspects functions that exist — asserts nothing
    # about an empty vector. llamacpp went through here with its entire API
    # hidden (GGML_API is a bare `extern` unless GGML_SHARED is defined), linked
    # 101 MB, wrapped zero functions, and exited 0.
    #
    # So the trigger is the empty surface itself. Promotion only selects which
    # advice to give.
    #
    # A templates-only package (`[types] templates` and no source of its own:
    # test_stl_extract's -O2 ctor probe) has no `functions`: its whole surface
    # is the STL method table, which the C++ generator binds as container
    # factories. Its own helpers (`replibuild_force::ref`/`tag`) are filtered
    # out of `functions` since 2026-11-13, so the table is what stands for the
    # surface there, and an empty table is the empty surface again.
    stl_table = get(metadata, "stl_methods", nothing)
    stl_bound = stl_table isa Dict && any(v -> v isa Vector && !isempty(v), values(stl_table))
    if isempty(functions) && !stl_bound
        cause = promoted_count > 0 ? """
        $(promoted_count) of those symbols are `__rb_*` promoted internals, so the
        API was RENAMED away: `-fvisibility=hidden` made every unannotated function
        hidden, and `[link] promote_statics` reads hidden as internal. lua's
        `LUA_API` is the type case.

        Fix in the package replibuild.toml:
          • drop `-fvisibility=hidden` from [compile] flags, or
          • set `[link] promote_statics = false` to stop the renaming (this also
            disables Tier-1 slicing, which is off by default anyway), or
          • if upstream really does annotate its exports, check that the annotation
            reached this TU — a missing -D can turn FOO_API into a bare `extern`.""" : """
        Nothing was renamed — the API is simply not in the dynamic table. With
        `-fvisibility=hidden` in [compile] flags, that means upstream's export
        macro is INERT in this build. Annotations like GGML_API / LLAMA_API /
        LUA_API expand to `__attribute__((visibility("default")))` only inside an
        `#ifdef` on a matching *_SHARED define; undefined, they are a bare
        `extern` and the flag hides them.

        Confirm which you have before changing anything — the annotation being
        present in the header proves nothing:
          echo '#include "<public header>"' | clang++ -E -x c++ <your -D flags> -I<inc> - | grep '<a known function>'
        A live annotation shows __attribute__((visibility("default"))) on that
        line; an inert one shows the bare declaration.

        Fix in the package replibuild.toml:
          • add the defines that activate the macro (e.g. -DGGML_SHARED,
            -DLLAMA_SHARED), or
          • drop `-fvisibility=hidden` from [compile] flags.

        If you reached for the flag to keep generated or vendored symbols OUT of
        the wrapper, `[wrap] exclude_symbols` does that directly and without
        touching visibility."""

        error("""
        RepliBuild: the wrappable surface of $(basename(binary_path)) is EMPTY.

        The library built and exports $(length(reachable)) dynamic symbols, but NOT
        ONE function survived with a name DWARF can match. The wrapper would be
        generated with no functions at all, and would fail at the first call site
        with UndefVarError rather than here.

        $cause
        """)
    end

    # R2 — names, never counts.
    offenders = String[]
    for f in functions
        f isa Dict || continue
        sym = get(f, "mangled", "")
        isempty(sym) && (sym = get(f, "name", ""))
        isempty(sym) && continue
        sym in reachable || push!(offenders, sym)
    end
    isempty(offenders) && return

    shown = first(offenders, 12)
    more  = length(offenders) - length(shown)
    error("""
    RepliBuild: $(length(offenders)) wrappable function(s) are NOT reachable in
    $(basename(binary_path)).

    These names were correlated against DWARF and would be emitted as `ccall` targets,
    but `dlsym` cannot resolve them — they are in the full symbol table and absent from
    the dynamic one. Every call would fail at runtime with an unresolved symbol, and
    the wrapper gives no sign of it at generation time.

    $(join(["    → " * o for o in shown], "\n"))$(more > 0 ? "\n    … and $more more" : "")

    The usual cause is hidden visibility reaching functions that are public API — see
    the `-fvisibility=hidden` × `promote_statics` interaction in this function's
    docstring. Check in the package replibuild.toml:
      • [compile] flags for `-fvisibility=hidden` or a `-D` that turns upstream's
        export macro into a bare `extern`;
      • [link] promote_statics, which renames hidden definitions to `__rb_*`.
    """)
end

"""
Generate a C/C++ shim file to instantiate requested macros as typed functions
so they appear in the DWARF metadata and can be wrapped.

`[wrap] shim_defines` go at the top of the same file, before its `#include`s:
with `STB_IMAGE_IMPLEMENTATION` defined there, the file compiles a single-header
library's implementation section into the library. The file is written when
either key has entries; with no macros it holds only the defines and includes.
"""
function generate_macro_shims(config::RepliBuildConfig, cpp_files::Vector{String})::Vector{String}
    if isempty(config.wrap.macros) && isempty(config.wrap.shim_defines)
        return cpp_files
    end

    cache_dir = get_cache_path(config)
    mkpath(cache_dir)
    ext = config.wrap.language == :c ? "c" : "cpp"
    dummy_file = joinpath(cache_dir, "replibuild_shims.$ext")

    open(dummy_file, "w") do io
        println(io, "// Auto-generated by RepliBuild to wrap macros into typed functions")

        # Here and nowhere else: a -D in [compile] flags would reach every file
        # the build compiles, and a second file including the header would then
        # define every function of the implementation section again.
        if !isempty(config.wrap.shim_defines)
            println(io, "// [wrap] shim_defines")
            for d in config.wrap.shim_defines
                println(io, ConfigurationManager.shim_define_line(d))
            end
        end

        for header in config.wrap.shim_headers
            if startswith(header, "<") || startswith(header, "\\\"")
                println(io, "#include $header")
            else
                println(io, "#include \"$header\"")
            end
        end
        println(io, "")

        # Defines only: the includes above are the whole file.
        isempty(config.wrap.macros) && return

        # A shim has to reach the wrapper's function list, and the two object
        # formats name DIFFERENT mechanisms for that.
        #
        #   ELF — `visibility("default")` is load-bearing: a project built with
        #   `-fvisibility=hidden` (box2d3) turns every shim local otherwise, and
        #   the wrapper's `nm -g --defined-only` list loses all of them.
        #
        #   PE — export is OPT-IN and `visibility("default")` is INERT for it.
        #   The shim was defined in the DLL and absent from the export
        #   directory, which is the list the wrapper reads on Windows
        #   (_pe_exported_names), so every `[wrap.macros]` entry silently
        #   vanished from the generated module. Measured on Hub pcre2: 14 shims
        #   defined by `nm -g`, 0 exported, 0 wrapped.
        #
        # Keyed on `_WIN32` rather than `Sys.iswindows()` because it is the
        # COMPILER's target that decides, not the host running RepliBuild.
        # `used` survives LTO internalization on both.
        println(io, "#if defined(_WIN32)")
        println(io, "#  define RB_SHIM_EXPORT __attribute__((used)) __declspec(dllexport)")
        println(io, "#else")
        println(io, "#  define RB_SHIM_EXPORT __attribute__((used, visibility(\"default\")))")
        println(io, "#endif")
        println(io, "")

        for (macro_name, def) in config.wrap.macros
            ret_type = get(def, "ret", "void")
            args = get(def, "args", String[])
            
            # Create parameter string (e.g. "void* arg0, const char* arg1")
            param_strs = String[]
            arg_names = String[]
            for (i, arg_t) in enumerate(args)
                pname = "arg$(i-1)"
                push!(param_strs, "$arg_t $pname")
                push!(arg_names, pname)
            end
            
            # RB_SHIM_EXPORT — see the per-format reasoning above the loop.
            sig = "RB_SHIM_EXPORT $ret_type replibuild_shim_$macro_name($(join(param_strs, ", ")))"
            println(io, "$sig {")

            # If "args" key is present → function-like macro, emit MACRO(args...)
            # If "args" key is absent → value macro, emit bare MACRO
            if haskey(def, "args")
                call_str = "$macro_name($(join(arg_names, ", ")))"
            else
                call_str = macro_name
            end
            if ret_type == "void"
                println(io, "    $call_str;")
            else
                println(io, "    return $call_str;")
            end
            println(io, "}")
            println(io, "")
        end
    end

    return vcat(cpp_files, [dummy_file])
end

# =============================================================================
# STL METHOD SYMBOL EXTRACTION
# =============================================================================

"""
Normalize an STL container type by stripping default allocator arguments.
E.g., "std::vector<int, std::allocator<int> >" -> "std::vector<int>"
Handles extra spaces that nm -C outputs (e.g., "allocator<int> >").
Also strips the standard library's INLINE NAMESPACE (`__cxx11`, `__1`).
"""
function _normalize_stl_type(demangled_type::String)::String
    t = strip(demangled_type)

    # Strip the standard library's ABI inline namespace.
    #
    # Which one appears is a property of the standard library, not the platform:
    # libstdc++ puts its C++11-ABI types in `std::__cxx11::`, and libc++ puts
    # EVERYTHING in `std::__1::`. A user writes `std::vector<int>` in
    # `[types] templates`, so whichever the toolchain emits has to be folded
    # back to that spelling or the requested instantiation matches nothing.
    #
    # Only `__cxx11` was handled, which was enough for as long as the only
    # toolchain in use was libstdc++. MSYS2 CLANG64 — the environment this
    # Windows port targets, and the only MSYS2 repo shipping MLIR — is a libc++
    # toolchain, so every STL type arrived as `std::__1::vector<int,
    # std::__1::allocator<int>>`, matched no template, and the wrapper silently
    # emitted none of the `create_*` factories. Nothing errored; the functions
    # simply were not there.
    #
    # Applied to the WHOLE string, not just a prefix: the allocator arguments
    # carry the namespace too, and the allocator-stripping patterns below match
    # on `std::allocator<...>` spelled plainly.
    t = replace(t, "std::__cxx11::" => "std::", "std::__1::" => "std::")

    # Normalize spaces around angle brackets for consistent matching
    # nm -C outputs "allocator<int> >" with space before final >
    # Collapse "  >" to ">"
    while contains(t, " >")
        t = replace(t, " >" => ">")
    end

    # std::vector<T, std::allocator<T>> -> std::vector<T>. The arguments are
    # split at depth 0, and the element is normalised in turn: a
    # `std::vector<std::vector<Point<long>, std::allocator<…>>, std::allocator<…>>`
    # (clipper2's `Paths64`) has an allocator at both depths, and a greedy
    # regex took the inner one as part of the element (2026-11-13).
    if startswith(t, "std::vector<") && endswith(t, ">")
        args = _split_template_args_toplevel(t[length("std::vector<")+1:prevind(t, lastindex(t))])
        if !isempty(args)
            elem = _normalize_stl_type(String(rstrip(args[1], [',', ' '])))
            return "std::vector<$elem>"
        end
    end

    # std::basic_string<char, std::char_traits<char>, std::allocator<char>>
    #
    # Anchored on the closing `>`, and that anchor is the whole point. Written as
    # a bare `startswith`, this branch also swallowed basic_string's NESTED
    # helper classes — libc++ has several — so
    #
    #   std::basic_string<char, ...>::__annotation_guard
    #
    # normalized to `std::basic_string<char>` and its members were recorded as
    # the string's own. Dedup keeps the FIRST match per method name, so
    # `~__annotation_guard()` was stored as the string's destructor and the real
    # `~basic_string()` never got in. CppString then destroyed itself by calling
    # a guard's destructor on a string: EXCEPTION_ACCESS_VIOLATION inside
    # __is_long, one frame under ~__annotation_guard.
    #
    # The vector/map/unordered_map branches below are `$`-anchored regexes and
    # were never exposed to this. libstdc++ does not emit these helpers, so the
    # defect was latent on Linux and immediate on libc++.
    if startswith(t, "std::basic_string<char") && endswith(t, ">")
        return "std::basic_string<char>"
    end

    # std::map<K, V, std::less<K>, std::allocator<...>> -> std::map<K, V>, and
    # std::unordered_map<K, V, std::hash<K>, std::equal_to<K>, std::allocator<...>>
    # -> std::unordered_map<K, V>: the first two depth-0 arguments, each
    # normalised in turn.
    for head in ("std::map<", "std::unordered_map<")
        if startswith(t, head) && endswith(t, ">")
            args = _split_template_args_toplevel(t[length(head)+1:prevind(t, lastindex(t))])
            if length(args) >= 2
                k = _normalize_stl_type(String(strip(args[1])))
                v = _normalize_stl_type(String(strip(args[2])))
                return head * k * ", " * v * ">"
            end
        end
    end

    return t
end

"The comma-separated arguments of a template argument list, split at angle-bracket and parenthesis depth 0."
function _split_template_args_toplevel(s::AbstractString)::Vector{String}
    out = String[]
    depth = 0
    start = firstindex(s)
    i = start
    while i <= lastindex(s)
        c = s[i]
        if c == '<' || c == '('
            depth += 1
        elseif c == '>' || c == ')'
            depth -= 1
        elseif c == ',' && depth == 0
            push!(out, String(strip(s[start:prevind(s, i)])))
            start = nextind(s, i)
        end
        i = nextind(s, i)
    end
    push!(out, String(strip(s[start:end])))
    return out
end

"""
Classify an STL method from its demangled name.
Returns (method_name, is_const) or nothing if not a recognized method.
"""
function _classify_stl_method(method_sig::String, container_type::String="")::Union{Tuple{String,Bool}, Nothing}
    sig = strip(method_sig)
    is_map = startswith(container_type, "std::map") || startswith(container_type, "std::unordered_map")

    # An ABI tag is not part of the method's identity, so it is dropped before
    # any of the name tests below.
    #
    # libc++ marks nearly every inline member `_LIBCPP_HIDE_FROM_ABI`, which
    # expands to an `abi_tag` attribute, and the tag mangles into the symbol
    # AFTER the name: `_ZNKSt3__16vectorIiNS_9allocatorIiEEE4sizeB9nqe220108Ev`,
    # demangling to `size[abi:nqe220108]() const`. Every test here is a
    # `startswith(sig, "size(")`, so every tagged member failed to classify and
    # was dropped — std::vector<int> came out of a real build with exactly ONE
    # method (the destructor, which survived only because its test is on the
    # leading `~`, and the tag trails the name). CppVector then had no size and
    # no push_back thunk to call.
    #
    # libstdc++ tags only `std::__cxx11` string/list members, so on Linux this
    # cost a couple of methods and was never traced; on libc++ it costs the
    # whole container. Same treatment as Wrapper/Cpp/IdentifiersCpp.jl, which
    # strips these tags for the same reason.
    sig = replace(sig, r"\[abi:[^\]]*\]" => "")

    # Default constructor only: ClassName(). Allocator/copy/size overloads
    # (`vector(allocator const&)`, `map(less const&, allocator const&)`) are
    # real symbols and used to classify as constructor too — but the Julia
    # factory calls T() with no extra args, so binding any of those is a
    # one-argument invoke of a two-argument callee. Empty parens only.
    if occursin(r"^(vector|basic_string|map|unordered_map|set|deque|list)\(\s*\)", sig)
        return ("constructor", false)
    end

    # Destructor: ~ClassName()
    if startswith(sig, "~")
        return ("destructor", false)
    end

    # Known methods
    is_const = endswith(sig, " const") || endswith(sig, ") const")

    if startswith(sig, "size(")
        return ("size", true)
    elseif startswith(sig, "data(")
        return ("data", is_const)
    elseif startswith(sig, "push_back(")
        return ("push_back", false)
    elseif startswith(sig, "operator[](")
        # Map operator[] takes key by const ref (ptr), vector takes size_t (i64)
        return (is_map ? "map_subscript" : "subscript", is_const)
    elseif startswith(sig, "clear(")
        return ("clear", false)
    elseif startswith(sig, "reserve(")
        return ("reserve", false)
    elseif startswith(sig, "resize(")
        return ("resize", false)
    elseif startswith(sig, "empty(")
        return ("empty", true)
    elseif startswith(sig, "begin(")
        # unordered_map::begin(size_type) is the bucket API, not iterator begin().
        occursin(r"^begin\(\s*\)", replace(sig, r"\s+const$" => "")) || return nothing
        return ("begin", is_const)
    elseif startswith(sig, "end(")
        occursin(r"^end\(\s*\)", replace(sig, r"\s+const$" => "")) || return nothing
        return ("end_", is_const)
    elseif startswith(sig, "c_str(")
        return ("c_str", true)
    elseif startswith(sig, "length(")
        return ("length", true)
    elseif startswith(sig, "append(")
        return ("append", false)
    elseif startswith(sig, "capacity(")
        return ("capacity", true)
    elseif startswith(sig, "back(")
        return ("back", is_const)
    elseif startswith(sig, "front(")
        return ("front", is_const)
    elseif startswith(sig, "pop_back(")
        return ("pop_back", false)
    elseif startswith(sig, "erase(")
        # Iterator overloads take a tree/node iterator, not a key. CppMap.delete!
        # passes a pointer to K. Binding erase(iterator) is a heap smash
        # (_Rb_tree_rebalance_for_erase). The paren-aware split made these
        # visible; they used to be dropped because `std::_Rb_tree_iterator`
        # stole the class/method `::`.
        #
        # BOTH standard libraries' spellings, or the guard is Linux-only:
        # libstdc++ writes `__normal_iterator` / `_Rb_tree_iterator`, libc++
        # writes vector's as `__wrap_iter` — no "iterator" anywhere in it. map
        # is caught either way (libc++'s is `__map_iterator`), so only vector
        # diverged, and only into a thunk nothing currently calls; match both
        # so the two hosts bind the same set rather than resting on that.
        (occursin(r"iterator"i, sig) || occursin("__wrap_iter", sig)) && return nothing
        return ("erase", false)
    elseif startswith(sig, "insert(")
        # Same two spellings as erase above — libc++'s vector iterator is
        # `__wrap_iter`, which `r"iterator"i` does not see.
        (occursin(r"iterator"i, sig) || occursin("__wrap_iter", sig)) && return nothing
        return ("insert", false)
    elseif startswith(sig, "find(")
        return ("find", is_const)
    elseif startswith(sig, "count(")
        return ("count", true)
    elseif startswith(sig, "at(")
        # Map at() takes key by const ref (ptr), vector takes size_t (i64)
        return (is_map ? "map_at" : "at", is_const)
    end

    return nothing
end

"""
    _stl_split_class_method(demangled) -> (class, method_sig) or nothing

Split a demangled `Class::method(args)` on the last `::` that is still the
class/method boundary: angle-bracket depth 0 AND parenthesis depth 0.

A scan that only tracks `<>` takes the last `::` in the whole string, so
constructor arguments like `std::allocator<int>` steal the split:

    std::vector<int, std::allocator<int> >::vector(std::allocator<int> const&)
                                            last :: was here ↑ inside the args
    class  = "std::vector<…>::vector(std"
    method = "allocator<int> const&)"     # dropped: not in the template set

`size()` and `push_back(int&&)` have no `::` in the args, so they happened
to work. Empty-paren `vector()` also worked — which is why the default ctor
was the only constructor extract ever stored, when it was in the .so at all.
"""
function _stl_split_class_method(demangled::AbstractString)::Union{Tuple{String,String}, Nothing}
    angle = 0
    paren = 0
    last_sep = -1
    for i in eachindex(demangled)
        c = demangled[i]
        if c == '<'
            angle += 1
        elseif c == '>'
            angle -= 1
        elseif c == '('
            paren += 1
        elseif c == ')'
            paren -= 1
        elseif angle == 0 && paren == 0 && c == ':' &&
               i < lastindex(demangled) && demangled[nextind(demangled, i)] == ':'
            last_sep = i
        end
    end
    last_sep > 0 || return nothing
    # last_sep is the first ':' of `::`. Class ends the character before;
    # method_sig starts after both colons. Demangled names are ASCII.
    return (String(demangled[1:last_sep-1]), String(demangled[last_sep+2:end]))
end

"""
Extract STL method symbols from the compiled binary.
Uses nm to find mangled names of template-instantiated STL methods, keeping only
the ones the loader can bind (see `_stl_methods_from_nm`).
Returns Dict mapping normalized container type -> vector of method info dicts.
"""
function extract_stl_method_symbols(binary_path::String, templates::Vector{String})::Dict{String,Any}
    if isempty(templates)
        return Dict{String,Any}()
    end

    # Every global symbol, defined (T/W: the library's own, template
    # instantiations among them) AND undefined (U: what the library calls in
    # the C++ runtime), in the symbol table's own order (`-p`) so the two
    # listings pair line by line (`_stl_methods_from_nm`).
    (mangled_output, exitcode1) = BuildBridge.execute("nm", ["-gp", binary_path])
    if exitcode1 != 0
        @warn "nm failed for STL symbol extraction"
        return Dict{String,Any}()
    end

    (demangled_output, exitcode2) = BuildBridge.execute("nm", ["-gpC", binary_path])
    if exitcode2 != 0
        @warn "nm demangled failed for STL symbol extraction"
        return Dict{String,Any}()
    end

    # An undefined member of a listed container is bindable when the library's
    # OWN handle resolves it: glibc's dlsym searches a library's dependency
    # tree, so libstdc++'s `basic_string<char>` members answer through the
    # library that links them. A PE handle resolves only the DLL's own exports,
    # so there the probe resolves none and nothing changes.
    undefined = Set{String}(_nm_undefined_names(mangled_output))
    resolvable = isempty(undefined) ? Set{String}() : _resolved_through_library(binary_path, undefined)

    (stl_methods, unbindable) = _stl_methods_from_nm(mangled_output, demangled_output, templates;
                                                     runtime = _runtime_symbol_names(binary_path),
                                                     resolvable = resolvable)
    runtime_n = sum(count(m -> get(m, "runtime_library", false) == true, ms) for ms in values(stl_methods); init = 0)
    runtime_n == 0 || println("  STL: $runtime_n member(s) of [types] templates bind from the C++ runtime " *
                              "library through the library's own handle")
    if !isempty(unbindable)
        shown = join(first(unbindable, 8), "\n  ")
        more = length(unbindable) > 8 ? "\n  … and $(length(unbindable) - 8) more" : ""
        @warn "STL: $(length(unbindable)) member(s) of [types] templates are in the symbol " *
              "table but neither among the library's runtime symbols nor resolvable through " *
              "its handle, so no thunk can bind them; the wrapper leaves them out:\n  $shown$more"
    end
    # An entry with no table is never silent: clipper2's
    # `std::vector<Clipper2Lib::Point64>` and fmt's `fmt::basic_memory_buffer<char>`
    # both produced none, and nothing said so (2026-11-13).
    for (entry, why) in _stl_table_gaps(templates, stl_methods)
        @warn "STL: [types] templates entry $(repr(entry)) yields no container table: $why"
    end
    return stl_methods
end

"The STL containers the method table and the wrapper's factories know."
const _STL_TABLE_CONTAINERS = ("std::vector<", "std::basic_string<", "std::map<", "std::unordered_map<")

"""
    _stl_table_gaps(templates, stl_methods) -> Vector{Tuple{String,String}}

Each `[types] templates` entry with no method table, with why. A template
outside `_STL_TABLE_CONTAINERS` (`fmt::basic_memory_buffer<char>`) is
instantiated for its DWARF alone and gets no factory, which `[types]
templates` is also for; one of the known containers with no table had no
member in the symbol table under the compiler's spelling of it.
"""
function _stl_table_gaps(templates, stl_methods)::Vector{Tuple{String,String}}
    gaps = Tuple{String,String}[]
    for t in templates
        key = t == "std::string" ? "std::basic_string<char>" : String(t)
        (haskey(stl_methods, key) && !isempty(stl_methods[key])) && continue
        haskey(stl_methods, String(t)) && !isempty(stl_methods[String(t)]) && continue
        if any(p -> startswith(key, p), _STL_TABLE_CONTAINERS)
            push!(gaps, (String(t), "no member of it is in the library's symbol table under the compiler's " *
                                    "spelling (the instantiation TU tags each entry; is the type complete " *
                                    "where the TU includes it?)"))
        else
            push!(gaps, (String(t), "it is not a container the STL table knows (std::vector, std::string, " *
                                    "std::map, std::unordered_map); it was instantiated so its DWARF exists, " *
                                    "which is all [types] templates does for it, and no factory is emitted"))
        end
    end
    return gaps
end

"""
    _stl_template_tags(demangled_output) -> Dict{Int,String}

Entry index → the compiler's canonical spelling of the listed type, read off
the `replibuild_force::tag<T, N>(T*)` instantiations the templates TU emits
(`_write_template_tu`), each normalised as the method table normalises nm's
container spellings. A listed `std::vector<Clipper2Lib::Point64>` is spelled
`std::vector<Clipper2Lib::Point<long>, std::allocator<…>>` on every member
symbol, which no string match on the recipe's typedef could find.
"""
function _stl_template_tags(demangled_output::AbstractString)::Dict{Int,String}
    tags = Dict{Int,String}()
    marker = "replibuild_force::tag<"
    for line in eachsplit(demangled_output, '\n')
        i = findfirst(marker, line)
        i === nothing && continue
        start = last(i) + 1
        depth = 1
        j = start
        stop = 0
        while j <= lastindex(line)
            c = line[j]
            if c == '<'
                depth += 1
            elseif c == '>'
                depth -= 1
                depth == 0 && (stop = j; break)
            end
            j = nextind(line, j)
        end
        stop == 0 && continue
        inner = String(line[start:prevind(line, stop)])
        k = findlast(',', inner)
        k === nothing && continue
        n = tryparse(Int, strip(inner[nextind(inner, k):end]))
        n === nothing && continue
        tags[n] = _normalize_stl_type(String(strip(inner[1:prevind(inner, k)])))
    end
    return tags
end

"""
    _nm_undefined_names(nm_output) -> Vector{String}

The names of the undefined (`U`) symbols in an `nm -g` listing, their version
suffix cut (`_ZNSt…@GLIBCXX_3.4.21` is how nm spells a reference to a versioned
symbol; `dlsym` takes the bare name).
"""
function _nm_undefined_names(nm_output::AbstractString)::Vector{String}
    return [name for (typ, _, name) in _nm_symbol_lines(nm_output) if typ == "U"]
end

"""
    _nm_symbol_lines(nm_output) -> Vector{NTuple{3,String}}

Every code or undefined symbol of an `nm -g` listing as `(type, address,
name)`: a defined code symbol (`T`/`W`/`t`/`w`, after a hex address) with its
address, an undefined one (`U`, no address) with `""`. The name is everything
after the type, so a demangled name keeps its spaces, and an undefined name
loses its version suffix (`@GLIBCXX_3.4.21`; demangled lines carry it too).
Lines of other kinds (data, weak undefined `w` without an address) are left out
of both listings alike, so the two pair by position.
"""
function _nm_symbol_lines(nm_output::AbstractString)::Vector{NTuple{3,String}}
    out = NTuple{3,String}[]
    for line in eachsplit(nm_output, '\n')
        parts = split(strip(line))
        length(parts) >= 2 || continue
        if parts[1] == "U"
            name = join(parts[2:end], " ")
            push!(out, ("U", "", String(replace(name, r"@[A-Za-z0-9_.]+$" => ""))))
        elseif length(parts) >= 3 && String(parts[2]) in ("T", "W", "t", "w") &&
               occursin(r"^[0-9a-fA-F]+$", parts[1])
            push!(out, (String(parts[2]), String(parts[1]), join(parts[3:end], " ")))
        end
    end
    return out
end

"""
    _resolved_through_library(lib_path, names) -> Set{String}

The `names` that `dlsym` resolves through `lib_path`'s OWN handle: the library
is opened `RTLD_NOW | RTLD_LOCAL`, asked, and closed again, the Tier 1
pre-flight's discipline (`_tier1_preflight!`), never `RTLD_DEFAULT`, whose
answer could be another copy of the runtime (Julia's own libstdc++: the zlib
lesson). An empty set when the library will not open; the caller then has
today's table.
"""
function _resolved_through_library(lib_path::AbstractString, names)::Set{String}
    out = Set{String}()
    handle = try
        Libdl.dlopen(lib_path, Libdl.RTLD_NOW | Libdl.RTLD_LOCAL)
    catch e
        @warn "STL: could not open $(basename(lib_path)) to probe its runtime symbols" exception = e
        return out
    end
    try
        for n in names
            Libdl.dlsym(handle, n; throw_error = false) === nothing || push!(out, String(n))
        end
    finally
        Libdl.dlclose(handle)
    end
    return out
end

"""
    _stl_methods_from_nm(mangled_output, demangled_output, templates; runtime=nothing)
        -> (stl_methods, unbindable)

The STL method table, read from `nm -g --defined-only` output (mangled and `-C`).

`runtime` is `_runtime_symbol_names` of the same binary. A symbol outside it is
skipped: its thunk would call a name the loader cannot bind, and one such thunk
fails the library's whole JIT module, so every Tier-2 function of the library
goes down with it. That is what libc++ 23 did on Windows — its members were
`T` in nm and absent from the export directory. `nothing` keeps nm's answer.

`unbindable` names each `Container::method` that had symbols but no bindable
variant, for the caller to report. A method whose other variant binds (a C1
constructor beside an exported C2) is not unbindable.

`resolvable` is the set of UNDEFINED symbols (`U` lines, version suffix cut)
the library's own handle resolves (`_resolved_through_library`): a member of a
listed container the library calls in the C++ runtime rather than
instantiating itself. Such a member enters the table marked
`"runtime_library" => true`, and the JIT binds it at the address the handle
gives (`JITManager`); an undefined member outside `resolvable` is unbindable.
Undefined lines have no address, so they pair with their mangled names by
position, which needs both listings in the symbol table's own order
(`nm -gp`, `nm -gpC`); listings that do not line up (the address-keyed shape
the older callers and tests pass) keep the address map and see no undefined
symbol (2026-11-13).
"""
function _stl_methods_from_nm(mangled_output::AbstractString, demangled_output::AbstractString,
                              templates::Vector{String};
                              runtime::Union{Nothing,AbstractSet{<:AbstractString}}=nothing,
                              resolvable::Union{Nothing,AbstractSet{<:AbstractString}}=nothing)
    # Build address -> mangled mapping (include T and W symbol types)
    address_to_mangled = Dict{String,String}()
    address_to_type = Dict{String,String}()
    for line in split(mangled_output, '\n')
        line = strip(line)
        isempty(line) && continue
        parts = split(line)
        if length(parts) >= 3 && parts[2] in ("T", "W", "t", "w")
            address_to_mangled[parts[1]] = parts[3]
            address_to_type[parts[1]] = parts[2]
        end
    end

    # Every code (T/W/t/w, with an address) or undefined (U, no address) line
    # of a listing as `(type, address, name)`, the version suffix cut from an
    # undefined name. The two listings pair by position when they line up
    # (same count, same type at each line: `nm -gp` and `nm -gpC`); only then
    # is an undefined symbol reachable, since it has no address to pair by.
    m_syms = _nm_symbol_lines(mangled_output)
    d_syms = _nm_symbol_lines(demangled_output)
    positional = length(m_syms) == length(d_syms) &&
                 all(i -> m_syms[i][1] == d_syms[i][1], eachindex(m_syms))

    # Build normalized template set for matching
    template_set = Set{String}()
    for t in templates
        push!(template_set, t)
        # Also add common aliases
        if t == "std::string"
            push!(template_set, "std::basic_string<char>")
        elseif t == "std::basic_string<char>"
            push!(template_set, "std::string")
        end
    end
    # The compiler's spelling of each entry (`_stl_template_tags`), so a member
    # spelled `std::vector<Clipper2Lib::Point<long>>` matches the entry written
    # `std::vector<Clipper2Lib::Point64>`. The table is keyed by the entry as
    # written, which names the factory; each member records the canonical
    # spelling (`container`), which the generator reads for the element type.
    canonical_to_entry = Dict{String,String}()
    for (n, canon) in _stl_template_tags(demangled_output)
        1 <= n + 1 <= length(templates) || continue
        # Only a container the table knows: a tagged `fmt::basic_memory_buffer`
        # has `size()` and `data()` too, and would otherwise get a table no
        # factory reads and thunks nothing calls, instead of the warning that
        # says what [types] templates does for it.
        any(p -> startswith(canon, p), _STL_TABLE_CONTAINERS) || continue
        entry = String(templates[n + 1])
        entry == "std::string" && (entry = "std::basic_string<char>")
        canon == entry && continue
        canonical_to_entry[canon] = entry
    end

    # Debug: print what we're looking for
    @debug "STL template set: $template_set"

    stl_methods = Dict{String,Any}()
    skipped = Set{Tuple{String,String}}()   # (container, method) with an unbindable symbol

    for (i, (sym_type, address, demangled)) in enumerate(d_syms)
        is_undefined = sym_type == "U"
        # An undefined symbol is a candidate only when it can be paired and
        # the handle resolves it.
        is_undefined && !(positional && resolvable !== nothing) && continue

        # Match pattern: ContainerType::MethodSignature. Paren-aware: see
        # `_stl_split_class_method` — a `<>`-only scan walks into ctor args.
        class_method = _stl_split_class_method(demangled)
        class_method === nothing && continue
        container_full, method_sig = class_method

        # Normalize the container type
        container_normalized = _normalize_stl_type(container_full)

        # Check if this container matches any configured template, as written
        # or in the compiler's spelling of it.
        canonical = container_normalized
        if haskey(canonical_to_entry, container_normalized)
            container_normalized = canonical_to_entry[container_normalized]
        elseif !(container_normalized in template_set)
            continue
        end

        # Classify the method
        classification = _classify_stl_method(method_sig, container_normalized)
        isnothing(classification) && continue

        (method_name, is_const) = classification

        # Get mangled name: its own line's when the listings line up, else
        # the one at its address.
        mangled = positional ? m_syms[i][3] : get(address_to_mangled, address, "")
        isempty(mangled) && continue

        # For constructors/destructors, accept C1/C2 and D1/D2 variants
        # (no filtering — we'll deduplicate later by keeping the first match per method)
        if method_name == "destructor"
            # Skip D0 (deleting destructor) — it calls operator delete which we don't want
            if occursin("D0", mangled)
                continue
            end
        end

        # Before the dedup below, so an unbindable variant cannot shadow a
        # bindable one (C1 first in nm, only C2 exported). A defined symbol
        # must be among the library's runtime symbols; an undefined one must
        # resolve through its handle.
        if is_undefined
            if !(mangled in resolvable)
                push!(skipped, (container_normalized, method_name))
                continue
            end
        elseif runtime !== nothing && !(mangled in runtime)
            push!(skipped, (container_normalized, method_name))
            continue
        end

        # Store (deduplicate: keep first match per method name per container)
        if !haskey(stl_methods, container_normalized)
            stl_methods[container_normalized] = Dict{String,Any}[]
        end

        # Check if we already have this method (dedup overloads — keep first)
        existing_methods = Set(get(m, "method", "") for m in stl_methods[container_normalized])
        if method_name in existing_methods
            continue
        end

        entry = Dict{String,Any}(
            "mangled" => mangled,
            "demangled" => demangled,
            "method" => method_name,
            "is_const" => is_const,
            "full_signature" => method_sig
        )
        canonical == container_normalized || (entry["container"] = canonical)
        # Lives in the C++ runtime library, not in this one: the JIT binds it
        # at the address the library's handle resolves (`JITManager`).
        is_undefined && (entry["runtime_library"] = true)
        push!(stl_methods[container_normalized], entry)
    end

    kept(c, m) = any(x -> get(x, "method", "") == m, get(stl_methods, c, Dict{String,Any}[]))
    unbindable = sort!([string(c, "::", m) for (c, m) in skipped if !kept(c, m)])
    return (stl_methods, unbindable)
end

"""
Complete compilation workflow: discover sources, compile, link, create binary.
This is the main entry point for building a project.
"""
function compile_project(config::RepliBuildConfig)
    println("RepliBuild | $(config.project.name)")

    start_time = time()

    # The inputs this build is OF, hashed before anything reads them, and the
    # value saved at the end. Hashing again at the end recorded whatever was on
    # disk THEN: a file saved while the build ran (llama.cpp spends 18 minutes
    # in DWARF extraction) was recorded as built, and every later build said
    # "project unchanged" over a library compiled from the version before.
    project_hash = compute_project_hash(config)

    # Fast path: check project-level content hash cache
    if is_project_cache_valid(config; hash=project_hash)
        output_dir = get_output_path(config)
        lib_name = get_library_name(config)
        library_path = joinpath(output_dir, lib_name)
        elapsed = round(time() - start_time, digits=4)
        println("  cache: project unchanged ($(elapsed)s)")
        return library_path
    end

    # A miss: the marker goes before any artifact is touched. Kept, it is the
    # LAST successful build's hash, and a build that fails after the link (the
    # wrap-surface guard, the load check, DWARF extraction) leaves a new
    # library beside it. Reverting the change that broke the build restored
    # those inputs, the old hash matched, and the fast path returned the failed
    # build's library.
    rm(_project_hash_path(config); force=true)

    # Get source files (from config or discovery)
    cpp_files = get_source_files(config)

    # Generate template instantiations
    cpp_files = generate_template_instantiations(config, cpp_files)

    # ...and the unit that makes `[types] records` visible to the DWARF.
    cpp_files = generate_visible_records(config, cpp_files)

    # The forced-constructor sidecar belongs to THIS build's second link (step
    # 4b); a stale one must not shape the first.
    rm(_forced_symbols_path(config); force=true)

    # Generate macro shims
    cpp_files = generate_macro_shims(config, cpp_files)

    # Guard: the shim TU must not have resolved any shim_header to a system copy
    # (a different-version collision would silently corrupt every [wrap.macros]).
    verify_shim_headers(config, joinpath(get_cache_path(config),
        "replibuild_shims.$(config.wrap.language == :c ? "c" : "cpp")"))

    if isempty(cpp_files)
        # Not "run discover() first": this is usually reached after discover
        # ran and found nothing, or after a git dependency failed to clone.
        @warn """No source files to compile for $(config.project.name).
        [compile] source_files is empty and the scan of $(config.project.root) found none.
        If the sources come from a [dependencies] entry, look above for
        "Failed to clone dependency". Otherwise add sources under the project and
        run RepliBuild.discover(path; force=true)."""
        return nothing
    end

    println("  sources: $(length(cpp_files))  includes: $(length(get_include_dirs(config)))")

    # Step 1: Compile C++ → IR
    ir_files = compile_to_ir(config, cpp_files)

    # Step 2: Link & optimize IR
    output_name = config.project.name
    linked_ir = link_optimize_ir(config, ir_files, output_name)

    # Step 3: Create binary (library or executable)
    binary_path = if config.binary.type == :executable
        create_executable(config, linked_ir, config.project.name)
    else
        # `ir_files` is handed along as the per-TU fallback. link_optimize_ir
        # normally returns ONE merged module, and when that module is too big
        # for clang's SourceLocation space the only way through is to give each
        # translation unit its own process — which needs the unmerged list.
        create_library(config, linked_ir; per_tu_ir=ir_files)
    end

    # Step 4: Extract and save compilation metadata. `nounwind` is read off the
    # exact module(s) the binary was linked from — the one merged module, or the
    # per-TU list when llvm-link declined to merge.
    nounwind = _nounwind_definitions(linked_ir isa String ? [linked_ir] : linked_ir)
    # So is each definition's `byval` slot alignment (`_byval_definitions`).
    byval = _byval_definitions(linked_ir isa String ? [linked_ir] : linked_ir)
    metadata_path = save_compilation_metadata(config, cpp_files, binary_path;
                                              nounwind=nounwind, byval=byval)

    # Step 4b: the inline constructors the library declares and no translation
    # unit of it defined (`inline_constructors` in the metadata, computed from
    # this first extraction: declared minus defined). A generated unit odr-uses
    # each through placement new (`generate_inline_constructors`); it is
    # compiled beside the others (every other IR is a cache hit), the library
    # is relinked with it and the metadata extracted again, now with those
    # constructors as definitions. The second extraction carries the plan with
    # each row's outcome instead of recomputing it.
    first_meta = _read_json(metadata_path)
    empty_plan() = Dict{String,Any}("forced" => Any[], "skipped" => Any[])
    plan = something(get(first_meta, "inline_constructors", nothing), empty_plan())
    # ...and the same for its other inline member functions (`inline_methods`):
    # the unit takes each one's address.
    mplan = something(get(first_meta, "inline_methods", nothing), empty_plan())
    # ...and the calls that leave default arguments out (`default_arguments`):
    # one function per call the header has defaults for.
    dplan = something(get(first_meta, "default_arguments", nothing), Dict{String,Any}("candidates" => Any[]))
    if config.wrap.language == :cpp && config.binary.type != :executable &&
       plan isa AbstractDict && mplan isa AbstractDict && dplan isa AbstractDict &&
       !(isempty(get(plan, "forced", Any[])) && isempty(get(mplan, "forced", Any[])) &&
         isempty(get(dplan, "candidates", Any[])))
        (cpp_files_b, plan, mplan, dplan) = generate_inline_definitions(config, cpp_files, plan, mplan, dplan)
        if length(cpp_files_b) > length(cpp_files)
            nforced = length(plan["forced"])
            nforced > 0 &&
                println("  ctors: $(nforced) inline constructor(s) of $(length(unique(String[f["class"] for f in plan["forced"]]))) class(es) " *
                        "declared in headers and defined nowhere: compiling their definitions")
            nmeth = length(mplan["forced"])
            nmeth > 0 &&
                println("  methods: $(nmeth) inline member function(s) of $(length(unique(String[f["class"] for f in mplan["forced"]]))) class(es) " *
                        "declared in headers and defined nowhere: compiling their definitions")
            ndef = length(dplan["bound"])
            ndef > 0 &&
                println("  defaults: $(ndef) call(s) of $(length(unique(String[f["symbol"] for f in dplan["bound"]]))) function(s) " *
                        "with default arguments left out: compiling their entry points")
            cpp_files = cpp_files_b
            ir_files = compile_to_ir(config, cpp_files)
            linked_ir = link_optimize_ir(config, ir_files, output_name)
            binary_path = create_library(config, linked_ir; per_tu_ir=ir_files)
            nounwind = _nounwind_definitions(linked_ir isa String ? [linked_ir] : linked_ir)
            byval = _byval_definitions(linked_ir isa String ? [linked_ir] : linked_ir)
        end
        # Extracted again even when no unit was written: the plans now carry
        # the rows the unit refused, which the wrap report names.
        metadata_path = save_compilation_metadata(config, cpp_files, binary_path;
                                                  nounwind=nounwind, byval=byval,
                                                  inline_constructors=plan, inline_methods=mplan,
                                                  default_arguments=dplan)
    end

    # A `[types] records` entry the DWARF still does not define is named.
    _report_invisible_records(config, metadata_path)

    # Guard: the functions metadata says are wrappable must actually be reachable
    # through the dynamic symbol table the generated wrapper will dlsym against.
    # BEFORE save_project_hash — a build that fails this must not leave a cache
    # marker claiming it succeeded.
    verify_wrap_surface(config, binary_path, metadata_path)

    # Step 5: Save project hash for future cache hits: the one taken before the
    # build, which is what it compiled.
    save_project_hash(config; hash=project_hash)

    # Timed HERE, not before step 4. DWARF extraction is part of the build and on
    # a large library it dominates it: llama.cpp linked at 73s and then spent 18
    # further minutes parsing a 252 MB readelf dump, while this line claimed
    # "done: 73.69s". Anyone profiling the pipeline off that number would look in
    # exactly the wrong place.
    elapsed = round(time() - start_time, digits=2)
    println("  done: $(elapsed)s")

    return binary_path
end

"""
    ingest_library(config) -> String

BYOB ("bring your own build") path: skip the whole Clang/LLVM compile pipeline and
ingest a pre-built `.so` produced by upstream's own build system. Runs DWARF metadata
extraction on the binary and produces `compilation_metadata.json` next to a copy of
the library inside the project's output dir, so `wrap()` finds it where it expects.

Ingested libraries dispatch through Tier 3 (`ccall`) only — there's no LTO bitcode to
enable Tier 1 `llvmcall`, and no per-file IR for Tier 2 thunks.

Requires `config.ingest !== nothing` (set by a `[ingest]` section in replibuild.toml).
"""
function ingest_library(config::RepliBuildConfig)::String
    ingest = config.ingest
    ingest === nothing && error("ingest_library called without [ingest] section in config")

    println("RepliBuild | $(config.project.name) [ingest]")

    # Guard both entry paths: RepliBuild.ingest() warns at scaffold time, but a
    # hand-written [ingest] TOML lands here directly. The C++ API surface of an
    # ingested binary is not supported — dialect marshalling/thunks only exist
    # in the source-build pipeline (issue #4: the generated C++ wrappers were
    # unusable and the user had no signal why).
    if config.wrap.language == :cpp
        @warn """
        Ingest mode with language=\"cpp\": the C++ API surface (classes, methods, templates,
        virtual dispatch) of an ingested binary is NOT supported — at best the extern \"C\"
        surface will be usable. Build from source (discover/build/wrap) for C++ libraries,
        or ingest a C API variant with language=\"c\"."""
    end

    start_time = time()

    src_lib = isabspath(ingest.library) ? ingest.library : abspath(joinpath(config.project.root, ingest.library))
    if !isfile(src_lib)
        error("Ingest library not found: $src_lib")
    end

    # Cheap DWARF presence check. It gates `extract_compilation_metadata`, which
    # would otherwise emit a wrapper full of untyped void* signatures on a
    # stripped library — so it must agree with what extraction can actually read,
    # and extraction is GNU readelf and nothing else (see the note on
    # `extract_dwarf_return_types` for why the llvm-dwarfdump fallback there was
    # removed: it parsed a valid dump into zero functions and shipped guesses).
    #
    # This check claimed to "mirror the tool order" and did not. It carried an
    # llvm-dwarfdump fallback the extractor does not have — a gate more
    # permissive than the thing it gates — and that fallback was UNREACHABLE
    # besides: it sat in a `catch`, but `BuildBridge.execute` never throws
    # (`_run_command_impl` returns `("Error: …", 1)` for every exception,
    # including a missing binary). So it never ran once, and the only thing it
    # did was make two derivations of one fact disagree on paper. Removed.
    #
    # The three outcomes are now distinguished, because they need different
    # answers from the user and the old code collapsed all of them into
    # "rebuild with -g".
    dumper = _dwarf_dumper()
    dumper === nothing && error(_no_dwarf_dumper_error())

    (sections, sections_code) = BuildBridge.execute(dumper.tool, [dumper.sections, src_lib])
    if sections_code != 0
        # The container could not be opened. This USED to be the PE case as
        # well — the gate ran `readelf -S`, readelf is ELF-only, and a
        # mingw-w64 .dll carrying perfectly good DWARF was refused for its
        # container rather than its debug info. `_dwarf_dumper` now hands back
        # GNU objdump on Windows, which reads PE/COFF, so a .dll gets past here
        # and is parsed by the same code path as an ELF .so.
        error("""
            $(dumper.tool) cannot read $src_lib as an object file.
            RepliBuild reads DWARF through GNU binutils, which handles ELF and
            PE/COFF. A file it cannot open is either not an object file or is a
            container binutils does not support (e.g. Mach-O).
            $(dumper.tool) said: $(strip(sections))""")
    end
    if !occursin(".debug_info", sections)
        error("No DWARF debug info found in $src_lib — rebuild upstream with -g (or pass a debug build)")
    end

    # Copy the .so into output_dir under the name wrap() expects, so the existing
    # wrapper-generation pipeline (which looks at joinpath(output_dir, lib_name))
    # finds it without modification.
    output_dir = get_output_path(config)
    mkpath(output_dir)
    dst_lib = joinpath(output_dir, get_library_name(config))
    if abspath(src_lib) != abspath(dst_lib)
        cp(src_lib, dst_lib; force=true, follow_symlinks=true)
    end

    # Merge ingest.headers into compile.include_dirs so extract_compilation_metadata
    # can find headers for noexcept scanning + downstream Clang.jl type extraction.
    if !isempty(ingest.headers)
        merged_includes = unique(vcat(config.compile.include_dirs, ingest.headers))
        config = with_include_dirs(config, merged_includes)
    end

    # No source files in ingest mode — extract_compilation_metadata tolerates an empty
    # vector (the C++ noexcept scan iterates include_dirs independently).
    save_compilation_metadata(config, String[], dst_lib)

    elapsed = round(time() - start_time, digits=2)
    println("  done: $(elapsed)s")

    return dst_lib
end

# =============================================================================
# METADATA EXTRACTION - The Key to Automatic Wrapping
# =============================================================================

"""
    _scan_noexcept_functions(source_files, include_dirs) -> Set{String}

BARE names that appear with a `noexcept` specifier in the sources and the top level
of the include dirs. A candidate set only: see `_nounwind_definitions` for why a
name alone must never route a function, and `extract_compilation_metadata` for the
intersection that does.
"""
function _scan_noexcept_functions(source_files::Vector{String},
                                  include_dirs)::Set{String}
    noexcept_names = Set{String}()

    # Collect all files to scan: source files + headers in include dirs
    files_to_scan = copy(source_files)
    for dir in include_dirs
        isdir(dir) || continue
        for f in readdir(dir, join=true)
            isfile(f) || continue
            if endswith(f, ".h") || endswith(f, ".hpp") || endswith(f, ".hxx")
                push!(files_to_scan, f)
            end
        end
    end

    # Regex: function_name followed by ( ... ) and then noexcept. It also matches
    # `noexcept(false)` and knows nothing about scope, which is why its answer is
    # only a CANDIDATE, confirmed per mangled symbol by `_nounwind_definitions`.
    noexcept_re = r"(\w+)\s*\([^)]*\)\s*(?:const\s*)?noexcept\b"

    for filepath in files_to_scan
        isfile(filepath) || continue
        try
            content = read(filepath, String)
            for m in eachmatch(noexcept_re, content)
                push!(noexcept_names, m.captures[1])
            end
        catch
            # Ignore unreadable files
        end
    end

    return noexcept_names
end

"""
    _nounwind_definitions(ir_files) -> Set{String}

Symbols DEFINED in `ir_files` whose LLVM function attributes include `nounwind`.
This is what decides whether a C++ function may be called through a plain
`ccall` (Tier 3) instead of an exception-catching thunk (Tier 2).

`nounwind` is the compiler's own answer to the question the router asks: can an
exception leave this function? clang stamps it on every definition declared
`noexcept` (a throw inside one calls `std::terminate` in the callee, as C++
requires, so the call site sees no unwind whichever tier calls it). It also
stamps it on definitions the optimizer PROVES cannot unwind. Either way a plain
`ccall` is exactly as safe as the thunk. It is read from the linked, optimized
module that becomes the `.so`, keyed on the mangled name, so it describes the
code that actually runs.

It confirms `_scan_noexcept_functions`, which on its own did the routing. That
scan regex-matches `name(...) noexcept` in source text and returns BARE names.
Every function sharing a bare name with a noexcept one (`B::get` vs
`A::get() const noexcept`, every constructor of a class with a `noexcept` move
constructor, anything named `size`/`what`/`swap`) was then routed to Tier 3. The regex also matched `noexcept(false)`. A throw through such
a `ccall` has no landing pad, so the process aborted (exit 134). Found by the
2026-09-26 audit.

An attribute group (`#N`) is resolved in the file that defines it: LLVM numbers
groups per module, and a library linked from several IR files (llvm-link
declined to merge, a graph library's per-TU list) has a `#0` in each. One table
across files let the last file's `#0` decide for all of them.
"""
function _nounwind_definitions(ir_files)::Set{String}
    unquote(n) = startswith(n, "\"") ?
        replace(n[2:end-1], r"\\([0-9A-Fa-f]{2})" => h -> string(Char(parse(UInt8, h[2:3], base=16)))) : n
    has_nounwind(attrs) = occursin(r"(?:^|\s)nounwind(?:\s|$)", attrs)

    out = Set{String}()
    for f in ir_files
        isfile(f) || continue
        # Per file: `#N` names a group of this module only (docstring).
        group_nounwind = Dict{String,Bool}()
        defs = Tuple{String,String}[]     # (symbol, text after the parameter list)
        for line in eachline(f)
            if startswith(line, "define ")
                m = match(r"@(\"(?:[^\"\\]|\\.)*\"|[-a-zA-Z$._][-a-zA-Z$._0-9]*)\(", line)
                m === nothing && continue
                # Skip to the parameter list's closing paren: parameter attributes
                # carry parens of their own (`dereferenceable(8)`, `align(16)`),
                # and function attributes only follow the whole list.
                i = m.offset + ncodeunits(m.match)
                depth = 1
                while i <= ncodeunits(line) && depth > 0
                    c = codeunit(line, i)
                    c == UInt8('(') && (depth += 1)
                    c == UInt8(')') && (depth -= 1)
                    i += 1
                end
                push!(defs, (unquote(String(m.captures[1])), line[i:end]))
            elseif startswith(line, "attributes #")
                g = match(r"^attributes (#\d+) = \{(.*)\}", line)
                g === nothing || (group_nounwind[g.captures[1]] = has_nounwind(g.captures[2]))
            end
        end
        for (name, tail) in defs
            if has_nounwind(tail) || any(r -> get(group_nounwind, r.match, false), eachmatch(r"#\d+", tail))
                push!(out, name)
            end
        end
    end
    return out
end

"""
    _byval_definitions(ir_files) -> Dict{String,Vector{Dict{String,Any}}}

For every symbol DEFINED in `ir_files`, its `byval` parameters in order: the
record each one carries (`"record"`, the LLVM type's name without its
`%struct.`/`%union.`/`%class.` prefix, quotes and the `.N` suffix llvm-link adds
to a duplicate) and the alignment of its stack slot (`"align"`). A definition
with no `byval` parameter maps to an empty list; a function the IR only declares
is absent.

This is the compiler's own answer to where a MEMORY-class record goes when it
is passed by value on SysV: clang stamps `byval(%struct.T) align N` with the
record's alignment, and the X86 backend places the slot at that alignment.
Julia's `ccall` lowering writes no `align` on its `byval`, so the backend uses
the Julia stand-in's (8 for any struct of fields), and a record C aligns to 16
lands one slot early whenever a stack-resident argument precedes it (W2,
2026-10-24). DWARF cannot always say which alignment C used: a `packed` record
whose members happen to sit at their natural offsets, or `#pragma pack(8)`
capping an `__int128` member, has the same DIEs as the unpacked record, and an
alignment only a typedef carries (`typedef struct S T
__attribute__((aligned(16)))`) is not in DWARF at all. The IR attribute is
exact in every one of those cases. GeneratorC matches these against the
record parameters its own classifier calls MEMORY (`_c_byval_facts`), and
passes the record through a type of that alignment where it can
(`_c_byval_param_shape`).

Read from the same `define` lines as `_nounwind_definitions`. A `byval` with
no `align` is not recorded: the function then has no fact, and the generator
falls back to DWARF. Win64 definitions carry no `byval` at all (a record goes
by reference to a copy), so every function there maps to an empty list.
"""
function _byval_definitions(ir_files)::Dict{String,Vector{Dict{String,Any}}}
    unquote(n) = startswith(n, "\"") ?
        replace(n[2:end-1], r"\\([0-9A-Fa-f]{2})" => h -> string(Char(parse(UInt8, h[2:3], base=16)))) : n
    out = Dict{String,Vector{Dict{String,Any}}}()
    for f in ir_files
        isfile(f) || continue
        for line in eachline(f)
            startswith(line, "define ") || continue
            m = match(r"@(\"(?:[^\"\\]|\\.)*\"|[-a-zA-Z$._][-a-zA-Z$._0-9]*)\(", line)
            m === nothing && continue
            # The parameter list: from after `@name(` to its matching paren.
            i = m.offset + ncodeunits(m.match)
            start = i
            depth = 1
            while i <= ncodeunits(line) && depth > 0
                c = codeunit(line, i)
                c == UInt8('(') && (depth += 1)
                c == UInt8(')') && (depth -= 1)
                i += 1
            end
            depth == 0 || continue
            params = line[start:i-2]
            facts = Dict{String,Any}[]
            complete = true
            for p in _ir_split_params(params)
                bv = match(r"\bbyval\(", p)
                bv === nothing && continue
                # The type inside `byval(...)`, with its own parens balanced.
                j = bv.offset + ncodeunits(bv.match)
                tstart = j
                d = 1
                while j <= ncodeunits(p) && d > 0
                    c = codeunit(p, j)
                    c == UInt8('(') && (d += 1)
                    c == UInt8(')') && (d -= 1)
                    j += 1
                end
                al = match(r"\balign (\d+)\b", p[j:end])
                if d != 0 || al === nothing
                    complete = false
                    break
                end
                push!(facts, Dict{String,Any}("record" => _ir_record_name(p[tstart:j-2]),
                                              "align" => parse(Int, al.captures[1])))
            end
            complete || continue
            out[unquote(String(m.captures[1]))] = facts
        end
    end
    return out
end

# The record an LLVM struct type name spells: `%struct.D4A` is `D4A`,
# `%struct.state.3` (llvm-link's rename of a duplicate) is `state`,
# `%"class.ns::T"` is `ns::T`, and `%union.U` is `U`. Anything else (an
# anonymous literal `{ i64, i64 }`, an array) is returned as written.
function _ir_record_name(t::AbstractString)::String
    s = String(strip(t))
    startswith(s, "%") || return s
    s = s[2:end]
    if startswith(s, "\"") && endswith(s, "\"")
        s = s[2:end-1]
    end
    s = replace(s, r"^(?:struct|union|class)\." => "")
    return replace(s, r"\.\d+$" => "")
end

# Split an LLVM `define` line's parameter list on the commas between
# parameters: those at bracket depth 0, counting `(`/`[`/`{`/`<` and their
# closers. An attribute's argument (`byval(%struct.T)`, `dereferenceable(8)`)
# and a literal type (`{ i64, i64 }`, `<2 x double>`) carry commas of their own.
function _ir_split_params(s::AbstractString)::Vector{String}
    out = String[]
    depth = 0
    buf = IOBuffer()
    for c in s
        if c in ('(', '[', '{', '<')
            depth += 1
        elseif c in (')', ']', '}', '>')
            depth -= 1
        end
        if c == ',' && depth == 0
            push!(out, String(take!(buf)))
        else
            write(buf, c)
        end
    end
    push!(out, String(take!(buf)))
    return out
end

# =============================================================================
# WHAT A PE BINARY ACTUALLY EXPORTS
# -----------------------------------------------------------------------------
# `nm -g --defined-only` is not a library's API on Windows, and the gap is not
# cosmetic.
#
# mingw links its C runtime, startup code and unwinder STATICALLY into every
# DLL. Those objects carry defined global symbols, so nm reports them exactly
# like the library's own functions: snprintf, memcpy, atexit, fprintf, abort,
# _CRT_INIT, operator new. On ELF the same code lives in a shared libc and is
# only ever REFERENCED, so nm's answer and the library's API are the same list
# and nothing had to distinguish them.
#
# The consequence is not a longer list, it is a broken build. `snprintf` was
# recorded as an exported function of test/c_test, RepliBuild generated a thunk
# for it, and the JIT then had to resolve `snprintf` at materialisation. It
# cannot: the symbol is in the COFF symbol table but NOT in the DLL's export
# directory, so GetProcAddress — and therefore dlsym, and therefore ORC's
# process search — does not find it. The session died with
#
#     JIT session error: Symbols not found: [ snprintf ]
#
# and the run then hung rather than failing.
#
# PE answers the question exactly, in the export directory, which is also the
# only thing GetProcAddress will ever consult — so it is the right authority in
# the strongest sense: a symbol outside it cannot be called through the library
# even in principle. Measured across the fixtures: c_test 115 nm symbols / 28
# exports, mi_test 133/43, vi_test 142/56, stl_test 691/513, and zero CRT
# symbols in any export table.
#
# ELF's equivalent is the dynamic symbol table, which is what `nm -D` reads —
# and `nm -D` is an error on PE ("File format has no dynamic symbol table"),
# which is why the sites that used it never worked on Windows.
# =============================================================================

"""
    _nm_dynamic_names(nm_output) -> Set{String}

The names in `nm -D --defined-only` output (mangled, never `--demangle`), with any
ELF symbol-version suffix cut. GNU nm prints a versioned definition as
`deflate@@ZLIB_1.2.0` (or `name@VER` for a non-default version), but the loader
binds, and DWARF names, `deflate`. Left on, every symbol of a library built with
a version script looked absent. A mangled name never contains `@`, so cutting at
the first one is exact; demangled output can (`basic_format_arg@fmt`), which is
why this takes mangled output only.
"""
function _nm_dynamic_names(nm_output::AbstractString)
    names = Set{String}()
    for line in eachsplit(nm_output, '\n')
        parts = split(strip(line))
        length(parts) >= 3 || continue
        push!(names, String(first(split(parts[3], '@'))))
    end
    return names
end

"""
    _pe_crt_dll(binary_path) -> Union{Nothing,String}

The C runtime a PE binary allocates from, as the DLL to resolve `free` in:
`"ucrtbase.dll"` when it imports the UCRT (`api-ms-win-crt-*`, which every
CLANG64-built library does), `"msvcrt.dll"` when it imports that, `nothing` off
Windows or when the import table cannot be read.

Julia itself is linked against msvcrt.dll, so `Libc.free` on memory such a library
malloc'd frees into the wrong heap and the process dies with
STATUS_HEAP_CORRUPTION. Generated wrappers free through this DLL instead
(`libc_free`).
"""
function _pe_crt_dll(binary_path::String)
    Sys.iswindows() || return nothing
    out = try
        readchomp(pipeline(ignorestatus(Tools.tool_cmd(`$(_gnu_objdump()) -p $binary_path`)), stderr = devnull))
    catch
        return nothing
    end
    imports = Set(lowercase(m.captures[1]) for m in eachmatch(r"DLL Name:\s*(\S+)", out))
    any(d -> startswith(d, "api-ms-win-crt-") || d == "ucrtbase.dll", imports) && return "ucrtbase.dll"
    "msvcrt.dll" in imports && return "msvcrt.dll"
    return nothing
end

"""
    _runtime_defined_names(library_path) -> Union{Set{String},Nothing}

Every symbol the loader can actually resolve in `library_path`, or `nothing`
when that cannot be determined. PE's export directory answers this on Windows;
`nm -D` reads ELF's dynamic symbol table elsewhere, and is a hard ERROR rather
than an empty answer on PE — hence the split.

`nothing` and an empty set mean different things and callers must not conflate
them: `nothing` is "could not look", which has to fail safe, while an empty set
is a real answer about a library that exports nothing.

Here in the Builder because `parse_dwarf_dump` asks it too: a toolchain record
keeps its layout only for what an exported function or global crosses by value.
"""
function _runtime_defined_names(library_path::AbstractString)
    isfile(library_path) || return nothing
    pe = _pe_exported_names(library_path)
    pe === nothing || return pe
    (out, code) = try
        BuildBridge.execute("nm", ["-D", "--defined-only", library_path])
    catch
        return nothing
    end
    code == 0 || return nothing
    return _nm_dynamic_names(out)
end

"""
    _without_excluded_symbols(exported, symbols, patterns) -> AbstractSet{String}

`exported` less every symbol that `[wrap] exclude_symbols` drops, matched the
way wrap matches it (`Wrapper._apply_symbol_filter!`). A function, from
`symbols`, on its mangled spelling, its demangled spelling, or its function
name (`extract_function_name`, which `parse_function_signatures` records as
`name`). A global on the exported name itself: `symbols` holds code symbols
only (`extract_symbols_from_binary`), and an unmangled global's symbol is its
name, which is what wrap matches a global on.

What wrap will not bind seeds no toolchain record. glfw is built without
`-fvisibility=hidden`, so its internal state global `_glfw` is exported, and
its TOML drops `_glfw*`; the by-value walk from that global still kept glibc's
`pthread_mutex_t`, `__pthread_mutex_s` and `re_pattern_buffer` and the
kernel's `input_absinfo`, and Glfw.jl grew them and the `_GLFW*` records that
hold them (2026-11-08). `patterns` empty returns `exported` itself.
"""
function _without_excluded_symbols(exported::AbstractSet{String}, symbols, patterns::Vector{String})
    isempty(patterns) && return exported
    (matches, _) = _glob_matcher(patterns)
    dropped = Set{String}()
    for s in symbols
        mangled = get(s, "mangled", nothing)
        demangled = get(s, "demangled", nothing)
        mangled isa AbstractString || continue
        if matches(mangled) ||
           (demangled isa AbstractString && (matches(demangled) || matches(extract_function_name(String(demangled)))))
            push!(dropped, String(mangled))
        end
    end
    for name in exported                      # globals: data symbols are not in `symbols`
        matches(name) && push!(dropped, name)
    end
    return isempty(dropped) ? exported : setdiff(exported, dropped)
end

"""
    _pe_exported_names(binary_path) -> Union{Nothing,Set{String}}

The names in a PE binary's export directory, or `nothing` when the question
cannot be answered — not Windows, no GNU objdump, or a binary with no export
table at all.

`nothing` means "do not filter", so a missing tool degrades to the old
behaviour rather than silently emptying a library's API.
"""
function _pe_exported_names(binary_path::String)
    Sys.iswindows() || return nothing
    dumper = _dwarf_dumper()
    dumper === nothing && return nothing

    out = try
        readchomp(pipeline(ignorestatus(Tools.tool_cmd(`$(dumper.tool) -p $binary_path`)), stderr = devnull))
    catch
        return nothing
    end

    names = Set{String}()
    started = false
    for line in split(out, '\n')
        if !started
            occursin("[Ordinal/Name Pointer] Table", line) && (started = true)
            continue
        end
        # Rows are `[   0] +base[   1]  0000 aabb_area`; older binutils drop the
        # +base/hint columns. Taking the last field covers both.
        m = match(r"^\s*\[\s*\d+\]\s+(.*\S)\s*$", line)
        if m === nothing
            # objdump prints an `Ordinal Hint Name` header under the marker
            # before the first row, so a non-row line only ends the table once
            # rows have started. Breaking on the header instead made the answer
            # always empty, which reads as "could not ask" and silently
            # restores the unfiltered behaviour this exists to replace.
            isempty(names) && continue
            break
        end
        push!(names, String(last(split(m.captures[1]))))
    end

    return isempty(names) ? nothing : names
end

"""
    _runtime_symbol_names(binary_path) -> Union{Nothing,Set{String}}

The names a loader can bind in `binary_path`: the PE export directory on Windows,
the dynamic symbol table (`nm -D`) elsewhere. `nothing` when neither could be
read, which callers treat as "do not filter".

This is the set a thunk's target has to be in. `nm -g` is not: on PE it reads the
COFF symbol table, where a function the DLL does not export is still a global `T`.
"""
function _runtime_symbol_names(binary_path::String)
    pe = _pe_exported_names(binary_path)
    pe === nothing || return pe
    (out, rc) = BuildBridge.execute("nm", ["-D", "--defined-only", binary_path])
    rc == 0 || return nothing
    return _nm_dynamic_names(out)
end

"""
    _nm_code_lines(output) -> Vector{NTuple{3,String}}

The `(address, type, name)` of every code symbol in an `nm` listing: types
`T` (text) and `W` (weak), which cover C++ template instantiations and inline
functions whose definitions live in headers, and the local-linkage `t`/`w`
nm reports for static functions, so a static C function exported through
`-fvisibility=default` still surfaces. The name is everything after the type,
so a demangled name keeps its spaces.
"""
function _nm_code_lines(output::AbstractString)::Vector{NTuple{3,String}}
    out = NTuple{3,String}[]
    for line in split(output, '\n')
        line = strip(line)
        isempty(line) && continue
        parts = split(line)
        length(parts) >= 3 || continue
        String(parts[2]) in ("T", "W", "t", "w") || continue
        push!(out, (String(parts[1]), String(parts[2]), join(parts[3:end], " ")))
    end
    return out
end

"""
    _pair_nm_outputs(mangled_output, demangled_output) -> Vector{NTuple{4,String}}

Every code symbol as `(address, type, mangled, demangled)`, from the two
listings `nm -gp --defined-only` (mangled) and `nm -gpC --defined-only`
(demangled) of one binary. With `-p` (no sort) both print the symbol table in
its own order, so line *i* of one is line *i* of the other, and each symbol
keeps its own two names.

Pairing by ADDRESS, as this did until 2026-11-13, gave every symbol at an
address the mangled name of whichever symbol nm listed last there. Clang
emits constructors and destructors as aliases (`-mconstructor-aliases`): D1
is D2's alias always, and at `-O1` and above a derived class's trivial
destructor is an alias of its base's, so tinyxml2's `~XMLComment`,
`~XMLUnknown`, `~XMLDeclaration` and `~XMLNode` D1/D2 were eight names at one
address. The metadata then carried `XMLNode`'s D2 symbol four times, once per
class, demangled as each derived destructor and with each derived class as
its `class`, and the derived classes' own symbols not at all:
`XMLComment_destroy_XMLComment` called `_ZN8tinyxml27XMLNodeD2Ev` under
`this::Ptr{XMLNode}`. The address map stays as the fallback for an `nm` whose
two listings do not line up (a count or an address differs), with a warning;
there, as before, a `__rb_` alias (`_alias_exports!`, Tier 1) never displaces
the real name at its address and any other name may.
"""
function _pair_nm_outputs(mangled_output::AbstractString, demangled_output::AbstractString)
    mangled = _nm_code_lines(mangled_output)
    demangled = _nm_code_lines(demangled_output)
    aligned = length(mangled) == length(demangled) &&
              all(i -> mangled[i][1] == demangled[i][1] && mangled[i][2] == demangled[i][2],
                  eachindex(mangled))
    if aligned
        return [(d[1], d[2], m[3], d[3]) for (m, d) in zip(mangled, demangled)]
    end
    @warn "nm's mangled and demangled listings do not line up ($(length(mangled)) and " *
          "$(length(demangled)) code symbols); pairing them by address, where aliased " *
          "symbols (constructors, destructors) share one name"
    address_to_mangled = Dict{String,String}()
    for (address, _, name) in mangled
        prev = get(address_to_mangled, address, nothing)
        if prev === nothing || !startswith(name, "__rb_") || startswith(prev, "__rb_")
            address_to_mangled[address] = name
        end
    end
    return [(address, typ, get(address_to_mangled, address, name), name) for (address, typ, name) in demangled]
end

"""
    _order_like_listing(pairs, sorted_output) -> Vector{NTuple{4,String}}

`pairs` (`_pair_nm_outputs`) in the order of `sorted_output`, nm's own sorted
demangled listing (`nm -gC --defined-only`, no `-p`): the order the metadata's
`functions` have always had, which every emitted wrapper follows, and which
nm collates by the locale (`en_US.UTF-8` folds case and skips punctuation),
so nothing here reproduces it. Each sorted line takes the next pair with its
address, type and demangled name; among aliases of one demangled name (D1 and
D2, C1 and C2 at one address) the smaller mangled name comes first, which is
the order the address map gave them. It only places them: which variant a
wrapper binds is decided by the variant (GeneratorCpp's `_structor_bindings`),
no longer by `_dedup_method_chunks` keeping the last one, which was D2 and C2.
A pair no sorted line claims is appended, sorted the same way, with a warning.
"""
function _order_like_listing(pairs::AbstractVector{NTuple{4,String}}, sorted_output::AbstractString)
    queues = Dict{NTuple{3,String},Vector{NTuple{4,String}}}()
    for p in pairs
        push!(get!(() -> NTuple{4,String}[], queues, (p[1], p[2], p[4])), p)
    end
    for q in values(queues)
        sort!(q; by = p -> p[3])
    end
    out = NTuple{4,String}[]
    for key in _nm_code_lines(sorted_output)
        q = get(queues, key, nothing)
        (q === nothing || isempty(q)) && continue
        push!(out, popfirst!(q))
    end
    left = sort!([p for q in values(queues) for p in q]; by = p -> (p[4], p[3]))
    isempty(left) || @warn "$(length(left)) code symbol(s) of nm's unsorted listing are not in its sorted one; appended"
    append!(out, left)
    return out
end

"""
Extract symbol information from compiled binary using nm.
Returns vector of symbol dictionaries with mangled/demangled names.
"""
function extract_symbols_from_binary(binary_path::String)
    # Two listings in the symbol table's own order (`-p`), so they pair line
    # by line (`_pair_nm_outputs`), and the sorted demangled listing for the
    # order (`_order_like_listing`): nm's default sort collates the demangled
    # names when `-C` is given and the mangled ones otherwise, and the
    # metadata's order is the wrapper's.
    (mangled_output, exitcode1) = BuildBridge.execute("nm", ["-gp", "--defined-only", binary_path])
    if exitcode1 != 0
        @warn "nm command failed: $mangled_output"
        return Dict{String,Any}[]
    end

    (demangled_output, exitcode2) = BuildBridge.execute("nm", ["-gpC", "--defined-only", binary_path])
    if exitcode2 != 0
        @warn "nm demangled command failed: $demangled_output"
        return Dict{String,Any}[]
    end

    (sorted_output, exitcode3) = BuildBridge.execute("nm", ["-gC", "--defined-only", binary_path])
    if exitcode3 != 0
        @warn "nm sorted command failed: $sorted_output"
        return Dict{String,Any}[]
    end

    # On PE, nm's answer includes the statically-linked C runtime and is not
    # this library's API — see _pe_exported_names for what that cost. `nothing`
    # means the question could not be asked, and then nothing is screened out.
    exported = _pe_exported_names(binary_path)

    # Build symbols array with both mangled and demangled
    symbols = Dict{String,Any}[]
    paired = _order_like_listing(_pair_nm_outputs(mangled_output, demangled_output), sorted_output)
    for (address, symbol_type, mangled_name, demangled_name) in paired
        # Filter out libstdc++/libc++ internals that get inlined into the
        # binary as weak symbols. These come from #include <vector> et al.
        # — they're "defined-only" because the inliner emitted them, not
        # because the user wrote them. Wrapping them produces signatures
        # referencing types we never declared (e.g. `std::less<void>`).
        # Deliberate STL container wrapping goes through extract_stl_method_symbols
        # under [types].templates, not this general path.
        if _is_stdlib_internal(mangled_name, demangled_name)
            continue
        end

        # Not in the PE export directory ⇒ not API, whatever nm says.
        # This is the screen that keeps mingw's statically-linked CRT
        # out: snprintf and friends are defined globals in the COFF
        # symbol table and absent from the export directory, so nothing
        # can reach them through the library — GetProcAddress consults
        # only that directory. Wrapping one produced a thunk the JIT
        # could not resolve, and the failure hung the run.
        if exported !== nothing && !(mangled_name in exported)
            continue
        end

        # Promoted statics (`__rb_<lib>_*`, static-promotion pass) are
        # slice-binding surface, not API: they carry no matching DWARF
        # entry under the promoted name, so wrapping them would emit
        # zero-arg garbage. The wrapper never sees them; slices bind by
        # symbol at JIT time.
        if startswith(mangled_name, "__rb_") || startswith(demangled_name, "__rb_")
            continue
        end

        # The templates TU's own helpers (`replibuild_force::ref<T>`,
        # `replibuild_force::tag<T, N>`; `_TEMPLATE_FORCE_PREAMBLE`,
        # `_write_template_tu`) are never API: `ref` returns a
        # reference to address 1 and exists to be named, `tag` to be
        # read off the symbol table. They were wrapped as functions
        # (clipper2's wrapper carried five) until 2026-11-13.
        if occursin("replibuild_force::", demangled_name)
            continue
        end

        # Itanium adjustor thunks are vtable-slot entry points, not API:
        # they fix up `this` on entry (a fixed subtraction, or a vcall
        # offset read through the vptr) and tail-jump to the real method.
        # Nothing should reach one BY NAME — an override is reached
        # through the vtable (the vcall producer) or after an explicit
        # upcast (`Derived_as_Base2`).
        #
        # Same reasoning as `__rb_*` above, and the same failure: no
        # DWARF subprogram describes a thunk, so `parse_function_signatures`
        # infers the signature from the demangled string — where the
        # "class" is the demangler's PHRASE ("non-virtual thunk to
        # Derived"), not a class name. No aggregate is named that, so
        # neither receiver gate (`FunctionGen._has_receiver`,
        # `GeneratorCpp`'s `struct_types` check) synthesizes `this`, and
        # the wrapper emits a zero-argument function calling a method that
        # needs `this` in rdi. Proven live 2026-08-13, not latent:
        # `ViTest.virtual_thunk_to_Diamond_tag()` was exported and
        # SIGSEGV'd — the virtual-thunk form dereferences the garbage
        # `this` twice (`mov (%rdi),%rax; mov -0x20(%rax),%rax`) to read
        # the vcall offset out of the vptr.
        if _is_itanium_thunk(mangled_name, demangled_name)
            continue
        end

        push!(symbols, Dict(
            "mangled" => mangled_name,
            "demangled" => demangled_name,
            "type" => symbol_type,
            "address" => address
        ))
    end

    return symbols
end

"""
    _is_itanium_thunk(mangled, demangled) -> Bool

Return true for Itanium ABI thunk symbols — `_ZTh` (non-virtual / this-adjusting),
`_ZTv` (virtual), `_ZTc` (covariant return).

These belong to the mangling grammar's `<special-name>` production, so a `_ZT`
prefix can never collide with an ordinary function; the remaining `_ZT*` special
names (`_ZTV` vtable, `_ZTI`/`_ZTS` typeinfo, `_ZTT` VTT, `_ZTC` construction
vtable) are **data** symbols and never reach this filter — the caller admits only
nm types `T`/`W`/`t`/`w`. Verified against the MI and VI fixtures: every `_ZT*`
symbol with a code type is a thunk, and every vtable/typeinfo one is `D`/`R`/`V`.

The demangled phrases are checked as well because the caller falls back to the
demangled string when the address→mangled lookup misses, and because an ingested
binary may be stripped of the mangled form.
"""
function _is_itanium_thunk(mangled::AbstractString, demangled::AbstractString)::Bool
    startswith(mangled, "_ZTh") && return true    # non-virtual thunk to X
    startswith(mangled, "_ZTv") && return true    # virtual thunk to X
    startswith(mangled, "_ZTc") && return true    # covariant return thunk to X
    startswith(demangled, "non-virtual thunk to ") && return true
    startswith(demangled, "virtual thunk to ") && return true
    startswith(demangled, "covariant return thunk to ") && return true
    return false
end

"""
    _is_stdlib_internal(mangled, demangled) -> Bool

Return true for symbols that live in the C++ standard library's namespaces
(`std`, `__gnu_cxx`, `__cxxabiv1`). These are inlined-into-the-binary as weak
symbols but are not part of the user's API.

The Itanium ABI mangled name is the precise signal: anything starting with
`_ZSt`, `_ZNSt`, `_ZNKSt`, `_ZNVSt`, `_ZNKVSt` is in the std namespace;
`_ZN9__gnu_cxx`/`_ZN10__cxxabiv1` cover libstdc++ extensions and the C++ ABI
runtime. The demangled name is the fallback for symbols whose mangled name
didn't survive (e.g. plain C exports, vtable/typeinfo strings).
"""
function _is_stdlib_internal(mangled::AbstractString, demangled::AbstractString)::Bool
    # Itanium ABI prefix checks — these are structural, not heuristic
    startswith(mangled, "_ZSt") && return true       # std::foo
    startswith(mangled, "_ZNSt") && return true      # nested std::Bar::foo
    startswith(mangled, "_ZNKSt") && return true     # const member in std
    startswith(mangled, "_ZNVSt") && return true     # volatile member in std
    startswith(mangled, "_ZNKVSt") && return true    # const volatile member in std
    # `_ZZ` prefix marks symbols defined inside another function's body
    # (e.g. nested helper types in libstdc++ inline templates). When the
    # enclosing function lives in std, treat the inner symbol as stdlib too.
    startswith(mangled, "_ZZSt") && return true
    startswith(mangled, "_ZZNSt") && return true
    startswith(mangled, "_ZZNKSt") && return true
    startswith(mangled, "_ZZNVSt") && return true
    startswith(mangled, "_ZN9__gnu_cxx") && return true
    startswith(mangled, "_ZNK9__gnu_cxx") && return true
    startswith(mangled, "_ZZN9__gnu_cxx") && return true
    startswith(mangled, "_ZN10__cxxabiv1") && return true
    startswith(mangled, "_ZNK10__cxxabiv1") && return true
    startswith(mangled, "__cxa_") && return true     # Itanium ABI runtime helpers

    # vtable / typeinfo entries appear in the demangled output without parens
    startswith(demangled, "vtable for std::") && return true
    startswith(demangled, "typeinfo for std::") && return true
    startswith(demangled, "typeinfo name for std::") && return true
    startswith(demangled, "vtable for __cxxabiv1::") && return true
    startswith(demangled, "typeinfo for __cxxabiv1::") && return true

    return false
end

"""
Extract mangled symbol name from nm output.
"""
function extract_mangled_name(nm_output::String, demangled::String)::String
    # Try to find the mangled version
    for line in split(nm_output, '\n')
        if contains(line, " T ")
            parts = split(strip(line))
            if length(parts) >= 3
                # parts[3] is the mangled name
                return parts[3]
            end
        end
    end
    return demangled  # Fallback
end

# =============================================================================
# DWARF DEBUG INFO PARSING - Type Extraction
# =============================================================================

# Comprehensive C/C++ → Julia type map. Hoisted to module scope so it is
# allocated once at module load, not once per call to dwarf_type_to_julia.
const _DWARF_TYPE_MAP = Dict{String,String}(
    # Void
    "void" => "Cvoid",

    # Boolean
    "bool" => "Bool",
    "_Bool" => "Bool",

    # Character types
    "char" => "Cchar",
    "signed char" => "Cchar",
    "unsigned char" => "Cuchar",
    "wchar_t" => "Cwchar_t",
    "char8_t" => "UInt8",
    "char16_t" => "UInt16",
    "char32_t" => "UInt32",

    # Standard integers (platform-dependent sizes)
    "short" => "Cshort",
    "short int" => "Cshort",
    "signed short" => "Cshort",
    "signed short int" => "Cshort",
    "unsigned short" => "Cushort",
    "unsigned short int" => "Cushort",

    "int" => "Cint",
    "signed int" => "Cint",
    "signed" => "Cint",
    "unsigned int" => "Cuint",
    "unsigned" => "Cuint",

    "long" => "Clong",
    "long int" => "Clong",
    "signed long" => "Clong",
    "signed long int" => "Clong",
    "unsigned long" => "Culong",
    "unsigned long int" => "Culong",

    "long long" => "Clonglong",
    "long long int" => "Clonglong",
    "signed long long" => "Clonglong",
    "signed long long int" => "Clonglong",
    "unsigned long long" => "Culonglong",
    "unsigned long long int" => "Culonglong",

    # Fixed-width integers (stdint.h)
    "int8_t" => "Int8",
    "int16_t" => "Int16",
    "int32_t" => "Int32",
    "int64_t" => "Int64",
    "uint8_t" => "UInt8",
    "uint16_t" => "UInt16",
    "uint32_t" => "UInt32",
    "uint64_t" => "UInt64",

    # Floating point
    "float" => "Cfloat",
    "double" => "Cdouble",
    "long double" => "NTuple{2, UInt64}",  # 80-bit extended precision in 16-byte slot

    # Size types
    "size_t" => "Csize_t",
    "ssize_t" => "Cssize_t",
    "ptrdiff_t" => "Cptrdiff_t",
    "intptr_t" => "Cintptr_t",
    "uintptr_t" => "Cuintptr_t",

    # Time types
    "time_t" => "Ctime_t",

    # Special types. DWARF names the unsigned one `unsigned __int128` (clang)
    # or `__int128 unsigned` (GCC); `__uint128_t` is a typedef of it.
    "__int128" => "Int128",
    "__int128_t" => "Int128",
    "unsigned __int128" => "UInt128",
    "__int128 unsigned" => "UInt128",
    "__uint128_t" => "UInt128",

    # C11 _Complex types
    "_Complex float" => "ComplexF32",
    "_Complex double" => "ComplexF64",
    "complex float" => "ComplexF32",
    "complex double" => "ComplexF64",
    "float _Complex" => "ComplexF32",
    "double _Complex" => "ComplexF64",
    # DWARF spells every width `complex`; `_complex_base_name` rewrites it
    # to one of these before the lookup.
    "complex float" => "ComplexF32",
    "complex double" => "ComplexF64",
    "complex long double" => "ComplexF64",
)

"""
    _complex_base_name(nbytes) -> String

DWARF names every C `_Complex` width `complex`. The byte size is the element
width doubled: 8 is `float`, 16 is `double`, 32 is `long double`. An unknown
size stays `complex`, which maps to `Any` and is refused rather than guessed.
"""
function _complex_base_name(nbytes::Int)::String
    nbytes == 8 && return "complex float"
    nbytes == 16 && return "complex double"
    nbytes == 32 && return "complex long double"
    return "complex"
end

"""
Comprehensive C/C++ type to Julia type mapping.
Handles all standard C/C++ types including sized integers, pointers, and qualifiers.
"""
function dwarf_type_to_julia(c_type::AbstractString)::String
    # Clean up type (remove extra spaces, trailing qualifiers)
    c_type = strip(c_type)

    # Direct match
    if haskey(_DWARF_TYPE_MAP, c_type)
        return _DWARF_TYPE_MAP[c_type]
    end

    # Handle pointer types (T*)
    if endswith(c_type, "*")
        # Strip pointer and get base type
        base_type = strip(replace(c_type, r"\*$" => ""))
        base_type = replace(base_type, r"\bconst\b" => "")
        base_type = replace(base_type, r"\bvolatile\b" => "")
        base_type = strip(base_type)

        # Special case: char* / i8* → Cstring (null-terminated C strings). Only
        # plain `char` is text: `signed char*` is bytes, `Ptr{Int8}` on every
        # host (`Cchar` is `UInt8` on AArch64), as `int8_t*` and `unsigned
        # char*` already are; a `Cstring` return copied {-1, -2, 0} into a
        # `String`. Metadata written before says `Cstring`, which the wrap
        # corrects from the return's `c_type` (`Wrapper._signed_char_returns!`).
        if base_type == "char" || base_type == "i8"
            return "Cstring"
        end
        base_type == "signed char" && return "Ptr{Int8}"

        # void* → Ptr{Cvoid}
        if base_type == "void" || isempty(base_type)
            return "Ptr{Cvoid}"
        end

        # Nested pointers: T** → Ptr{Ptr{...}}
        if endswith(base_type, "*")
            inner = dwarf_type_to_julia(base_type)
            return "Ptr{$inner}"
        end

        # Resolve base type through the type map
        julia_base = dwarf_type_to_julia(base_type)
        if julia_base != "Any"
            return "Ptr{$julia_base}"
        end

        # Unknown base type — keep as Ptr{Cvoid}
        return "Ptr{Cvoid}"
    end

    # Handle reference types (T& and T&&: both are an address at the ABI level —
    # stripping one `&` from `T&&` gave `Ref{Ref{T}}`)
    if endswith(c_type, "&")
        base_type = strip(replace(c_type, r"&&?$" => ""))
        base_type = replace(base_type, r"\bconst\b" => "")
        base_type = replace(base_type, r"\bvolatile\b" => "")
        base_type = strip(base_type)

        julia_base = dwarf_type_to_julia(base_type)
        if julia_base != "Any"
            return "Ref{$julia_base}"
        end
        return "Ref{Cvoid}"
    end

    # Handle const/volatile qualifiers
    if startswith(c_type, "const ") || startswith(c_type, "volatile ")
        clean_type = replace(c_type, "const " => "")
        clean_type = replace(clean_type, "volatile " => "")
        return dwarf_type_to_julia(strip(clean_type))
    end

    # Unknown type - return Any for safety
    return "Any"
end

"""
    as_return_julia_type(c_type, mapped) -> String

Adjust a mapped Julia type for a RETURN position.

Only C++ reference types differ. A `T&` is an ADDRESS at the ABI level — the
callee returns a pointer and the reference-ness is a source-language fiction —
but both type mappers render it `Ref{T}`, because in a PARAMETER position that
is correct *and* ergonomic: the caller passes a `Ref` and Julia converts it to a
pointer at the boundary.

In a return position `Ref{T}` is neither. `JITManager._invoke_call` allocates the
result buffer as `Ref{T}()`, so a `Ref{Cvoid}` return asks for
`Ref{Ref{Cvoid}}()` — an undefined reference whose `unsafe_convert` raises
`UndefRefError` *before the call is ever made*. `ImGui::GetIO()`
(`ImGuiIO&`, DWARF size 8) died exactly there.

Reported as `Ref{Nothing}` because `Cvoid === Nothing`, which reads like a
void-return bug and is why this was mis-filed as one.
"""
function as_return_julia_type(c_type::AbstractString, mapped::AbstractString)::String
    endswith(strip(String(c_type)), "&") || return String(mapped)
    m = match(r"^Ref\{(.*)\}$", String(mapped))
    return m === nothing ? String(mapped) : "Ptr{$(m.captures[1])}"
end

# C/C++ → byte size. `long` and `wchar_t` follow the host ABI (Julia's Clong /
# Cwchar_t); everything else is the same on LP64 and LLP64 x86_64.
# `get_type_size` feeds packed-detection / `_is_struct_unsafe`. A miss returns
# 0, which poisons those checks — so DWARF spellings like `long int` have to
# land in this table, not fall through.
const _C_LONG_BYTES = sizeof(Clong)
const _C_WCHAR_BYTES = sizeof(Cwchar_t)
const _C_TYPE_SIZE_MAP = Dict{String,Int}(
    "void" => 0,
    "bool" => 1, "_Bool" => 1,
    "char" => 1, "signed char" => 1, "unsigned char" => 1,
    "int8_t" => 1, "uint8_t" => 1,
    "short" => 2, "unsigned short" => 2,
    "int16_t" => 2, "uint16_t" => 2,
    "int" => 4, "unsigned int" => 4,
    "int32_t" => 4, "uint32_t" => 4,
    "long" => _C_LONG_BYTES, "unsigned long" => _C_LONG_BYTES,
    "Clong" => _C_LONG_BYTES, "Culong" => _C_LONG_BYTES,
    "long long" => 8, "unsigned long long" => 8,
    "int64_t" => 8, "uint64_t" => 8,
    "float" => 4,
    "double" => 8,
    "long double" => 16,
    "size_t" => 8, "ssize_t" => 8,
    "ptrdiff_t" => 8,
    "intptr_t" => 8, "uintptr_t" => 8,
    "wchar_t" => _C_WCHAR_BYTES, "Cwchar_t" => _C_WCHAR_BYTES,
    "char8_t" => 1, "char16_t" => 2, "char32_t" => 4,
    "__int128" => 16, "__uint128_t" => 16,
    "__int128_t" => 16, "unsigned __int128" => 16, "__int128 unsigned" => 16,
)

"""
    _c_type_size_key(c_type) -> String

Strip cv-qualifiers and fold DWARF/clang spellings (`long int` → `long`) so
the size table has one entry per ABI type.
"""
function _c_type_size_key(c_type::AbstractString)::String
    s = replace(strip(String(c_type)), r"\b(const|volatile|mutable)\b\s*" => "")
    s = strip(replace(s, r"\s+" => " "))
    s = replace(s, r"\blong int\b" => "long")
    s = replace(s, r"\bshort int\b" => "short")
    s = replace(s, r"^signed long$" => "long")
    s = replace(s, r"^signed short$" => "short")
    s = replace(s, r"^signed int$" => "int")
    s = replace(s, r"^signed$" => "int")
    return s
end

"""
Get type size in bytes from C/C++ type name.
"""
function get_type_size(c_type::AbstractString)::Int
    stripped = _c_type_size_key(c_type)

    # Pointers/references are word-sized on every x86_64 ABI we target.
    if endswith(stripped, "*") || endswith(stripped, "&")
        return 8
    end

    return get(_C_TYPE_SIZE_MAP, stripped, 0)
end

"""
    _die_byte_size(ref, type_refs; base_sizes) -> Int

Byte size of the type DIE at `ref` in `parse_dwarf_dump`'s DIE table,
following the chain DWARF spells a type with: a typedef or a `const`,
`volatile`, `restrict` or `_Atomic` qualifier to its target; an enum, struct,
class or union to its `DW_AT_byte_size`; an array to its element times its
extents; a pointer or reference to 8; a base type to its own `DW_AT_byte_size`
(`base_sizes`, by DIE offset), else `get_type_size` of its name. `0` when the
chain ends somewhere unmeasured.

`get_type_size` answers only for spelled primitives. A member whose type is an
enum, a typedef of a record, or an array of records came out with size 0, and
GeneratorCpp's layout proof then gave up and emitted the enclosing record as a
byte blob.
"""
function _die_byte_size(ref, type_refs; base_sizes::AbstractDict = Dict{String,Int}(),
                        visited::Set{String} = Set{String}())::Int
    (ref isa AbstractString && !(ref in visited)) || return 0
    push!(visited, ref)
    info = get(type_refs, ref, nothing)
    info isa AbstractString && return get(base_sizes, ref, get_type_size(info))
    info isa Dict || return 0
    kind = get(info, "kind", nothing)
    kind in ("pointer", "reference", "rvalue_reference") && return 8
    if kind in ("struct", "class", "union", "enum")
        bs = get(info, "byte_size", nothing)
        bs === nothing && return 0
        s = string(bs)
        return something(startswith(s, "0x") ? tryparse(Int, s[3:end]; base = 16) : tryparse(Int, s), 0)
    end
    follow(r) = _die_byte_size(r, type_refs; base_sizes, visited)
    kind in ("typedef", "const", "volatile", "restrict", "atomic") && return follow(get(info, "target", nothing))
    if kind == "array"
        # A vector DIE's own byte_size wins over lanes × element: clang writes
        # it when the lanes do not fill the vector (`ext_vector_type(3)` of
        # float is 16 bytes, not 12), and nothing else carries that fact.
        bs = get(info, "byte_size", nothing)
        if bs !== nothing
            s = string(bs)
            n = startswith(s, "0x") ? tryparse(Int, s[3:end]; base = 16) : tryparse(Int, s)
            n !== nothing && n > 0 && return n
        end
        dims = get(info, "dimensions", [])
        isempty(dims) && return 0
        return follow(get(info, "element_type", nothing)) * prod(Int.(dims))
    end
    return 0
end

function _is_gnu_binutils(version_out::AbstractString)::Bool
    # LLVM's llvm-objdump has printed `compatible with GNU objdump`. A
    # substring `GNU` accepts that dialect, which `parse_dwarf_dump` cannot
    # read. GNU binutils identifies itself as `GNU Binutils` on Debian, Arch
    # and MSYS2 (`GNU nm (GNU Binutils for Debian) 2.42`), and by the program's
    # own line everywhere: Fedora builds print `GNU nm version 2.46.1-1.fc44`
    # with no `Binutils` in it (issue #68), so every binutils program name is
    # accepted at the start of a line. LLVM's tools never start one with `GNU`.
    occursin("GNU Binutils", version_out) ||
        occursin(r"(?m)^GNU (objdump|readelf|nm|ar|objcopy|strip|ranlib|size|strings)\b", version_out)
end

"""
    _dumper_candidates() -> Vector{String}

The programs `_dwarf_dumper` tries, in order: `REPLIBUILD_OBJDUMP` when set,
then `readelf` and `objdump` on PATH, and on Windows the GNU objdump of each
GCC-flavoured MSYS2 tree by absolute path (CLANG64's bare `objdump` is
llvm-objdump).
"""
function _dumper_candidates()::Vector{String}
    candidates = String[]
    env_override = get(ENV, "REPLIBUILD_OBJDUMP", "")
    isempty(env_override) || push!(candidates, env_override)
    if Sys.iswindows()
        # MSYS2 ships GNU binutils in the GCC-flavoured trees; CLANG64's own
        # `objdump` is llvm-objdump and will fail the GNU check, so the bare
        # name is tried first only for the case where a GNU one is on PATH.
        push!(candidates, "objdump")
        msys_root = get(ENV, "MSYS2_ROOT", "C:/msys64")
        for envdir in ("mingw64", "ucrt64", "mingw32")
            push!(candidates, joinpath(msys_root, envdir, "bin", "objdump.exe"))
        end
    else
        push!(candidates, "readelf", "objdump")
    end
    return candidates
end

"""
    _dumper_candidate_report() -> Vector{String}

One line per `_dumper_candidates()` entry saying what `--version` did: the
exit code and the first line it printed, or that it did not start. The
doctor shows these when no candidate was accepted, so a report of "no GNU
binutils" carries what RepliBuild saw instead of leaving the user to guess
(issue #68: binutils installed, every candidate refused, no way to tell why).
"""
function _dumper_candidate_report()::Vector{String}
    lines = String[]
    for tool in _dumper_candidates()
        (out, ec) = try
            BuildBridge.execute(tool, ["--version"])
        catch e
            (sprint(showerror, e), 1)
        end
        first_line = strip(first(split(out, '\n'; limit=2)))
        isempty(first_line) && (first_line = "(no output)")
        if ec == BuildBridge.SPAWN_FAILED
            push!(lines, "$tool: did not start — $first_line")
        else
            verdict = ec == 0 ? (_is_gnu_binutils(out) ? "accepted" : "not GNU binutils") : "exit $ec"
            push!(lines, "$tool: $verdict — `$first_line`")
        end
    end
    return lines
end

"""
    _gnu_objdump() -> String

A GNU binutils `objdump`, or the bare name when none can be identified.

Separate from `_dwarf_dumper` because that one may legitimately answer
`readelf` (it does on Linux, where readelf is preferred), and readelf cannot
disassemble. Same trap underneath, though: `Debug.disassemble` passes
`--disassemble=<symbol>`, which is GNU spelling that llvm-objdump rejects
outright (`unknown argument`), and in a CLANG64 shell the bare `objdump` IS
llvm-objdump. `_is_gnu_binutils` is what actually decides; the GCC-flavoured
MSYS2 trees are named because that is where a GNU one lives on Windows.

Falls back to the bare name rather than `nothing` so a machine with no objdump
at all fails by naming the tool, which is what the caller reports.
"""
function _gnu_objdump()::String
    candidates = String[]
    env_override = get(ENV, "REPLIBUILD_OBJDUMP", "")
    isempty(env_override) || push!(candidates, env_override)
    push!(candidates, "objdump")
    if Sys.iswindows()
        msys_root = get(ENV, "MSYS2_ROOT", "C:/msys64")
        for envdir in ("mingw64", "ucrt64", "mingw32")
            push!(candidates, joinpath(msys_root, envdir, "bin", "objdump.exe"))
        end
    end
    for tool in candidates
        (out, ec) = try
            BuildBridge.execute(tool, ["--version"])
        catch
            ("", 1)
        end
        ec == 0 || continue
        _is_gnu_binutils(out) && return tool
    end
    return "objdump"
end


"""
    _dwarf_dumper() -> NamedTuple or nothing

The GNU binutils program used to dump DWARF, plus how it spells its flags.
`nothing` when no acceptable one is installed.

`parse_dwarf_dump` is a GNU **binutils**-format parser: it keys every DIE on
`<level><offset>: Abbrev Number: N (DW_TAG_*)`. Two binutils programs print
exactly that, and they print it from the same source file (binutils' `dwarf.c`),
so their output is not merely similar — it is the same text:

    readelf --debug-dump=info   ==   objdump --dwarf=info

Verified byte-identical over the DIE lines of one object, for `info` and for
`rawline`. What they do NOT share is the set of containers they can open, and
that is the only reason to choose between them:

  * `readelf` is a standalone ELF reader. It cannot open a PE/COFF `.dll` at all.
  * `objdump` goes through BFD, so it reads `pei-x86-64` as readily as ELF.

Hence Windows uses objdump. Linux keeps readelf, which is what the reference
host has always run; there is nothing to gain by moving a proven path.

**The `--version` check is load-bearing, not a formality.** In an MSYS2 CLANG64
environment the names `objdump` and `readelf` on PATH are `llvm-objdump` and
`llvm-readelf`, and LLVM's dump format is a different dialect entirely — no
level, no abbrev number. Accepting one would not fail loudly: `parse_dwarf_dump`
would match nothing, return zero functions, and every signature would fall back
to `parameters_source: "inferred"` — the exact silent-guess failure the
llvm-dwarfdump fallback was deleted for. So a candidate is used only if it
identifies itself as GNU.

Set `REPLIBUILD_OBJDUMP` to an explicit path to override the search.
"""
function _dwarf_dumper()
    for tool in _dumper_candidates()
        (out, ec) = try
            BuildBridge.execute(tool, ["--version"])
        catch
            ("", 1)
        end
        ec == 0 || continue
        _is_gnu_binutils(out) || continue   # rejects llvm-objdump / llvm-readelf

        # readelf spells it `--debug-dump=info` and lists sections with `-S`;
        # objdump spells the same dump `--dwarf=info`, and sections with `-h`.
        is_readelf = occursin("readelf", lowercase(basename(tool)))
        return (tool     = tool,
                dwarf    = is_readelf ? "--debug-dump=" : "--dwarf=",
                sections = is_readelf ? "-S" : "-h")
    end
    return nothing
end

"""
    _no_dwarf_dumper_error() -> String

Shared text for "no GNU DWARF dumper", so the ingest gate and the extractor
cannot drift into describing the same missing tool two different ways.
"""
function _no_dwarf_dumper_error()
    if Sys.iswindows()
        """
        No GNU binutils DWARF dumper found — DWARF extraction cannot run.

        RepliBuild parses GNU binutils' dump format specifically. On Windows the
        container is PE/COFF, which GNU `readelf` cannot open at all, so the
        dumper must be GNU `objdump` (BFD-based, reads PE and prints the same
        format from the same code).

        The `objdump` in an MSYS2 CLANG64 shell is llvm-objdump, which prints a
        DIFFERENT dialect and is deliberately rejected — accepting it would
        yield zero parsed functions and a wrapper full of guessed signatures.

        Install GNU binutils:   pacman -S mingw-w64-x86_64-binutils
        Or point at one:        REPLIBUILD_OBJDUMP=/path/to/objdump.exe

        That is the MINGW64 package on purpose, and it is the fix even though
        you are in a CLANG64 shell: MSYS2 ships GNU binutils only in the
        GCC-flavoured trees, and both resolvers look for
        `$(get(ENV, "MSYS2_ROOT", "C:/msys64"))/{mingw64,ucrt64,mingw32}/bin/objdump.exe`
        by absolute path, so PATH and the active MSYSTEM do not matter. Do NOT
        "correct" this to a mingw-w64-clang-x86_64-* package — CLANG64's
        objdump is llvm-objdump, which is the thing being rejected."""
    else
        """
        GNU readelf/objdump not found — DWARF extraction cannot run.
        RepliBuild's DWARF parser reads GNU binutils' format specifically; no
        other dumper is substitutable. Install binutils."""
    end
end

"""
    _dwarf_file_tables(dumper, binary_path) -> Vector{Dict{Int,String}}

One `file index => resolved path` table per compilation unit, in CU order,
parsed from `readelf --debug-dump=rawline`.

`DW_AT_decl_file` is an index into the CU's own line-program file table, so the
tables are **CU-relative** — index 2 names a different file in every CU, and a
single flattened map would mis-attribute every type in a multi-CU library. CU
order is the only thing tying the two dumps together: readelf emits line
programs and DIEs in the same section order.

Names are resolved against the Directory Table, itself relative to directory 0
(the compilation directory) when an entry is not absolute.
"""
function _dwarf_file_tables(dumper, binary_path::AbstractString)::Vector{Dict{Int,String}}
    (out, ec) = BuildBridge.execute(dumper.tool, [dumper.dwarf * "rawline", binary_path])
    ec == 0 || return Vector{Dict{Int,String}}()

    tables = Vector{Dict{Int,String}}()
    dirs = Dict{Int,String}()
    mode = :none
    # DWARF 4 and earlier print `Entry Dir Time Size Name`; DWARF 5 prints
    # `Entry Dir [MD5] Name`. Read off each table's own column header: taken
    # as the DWARF 5 shape, a DWARF 4 row's name came out as `0\t0\t<name>`
    # and no declaration file of a `-gdwarf-4` build matched anything.
    legacy_columns = false
    # readelf renders a string-table indirection as `(indirect line string,
    # offset: 0x..): <value>` — the value is everything after the last `): `.
    strip_indirect(s) = begin
        m = match(r"\):\s*(.*)$", s)
        String(strip(m === nothing ? s : m.captures[1]))
    end

    for line in split(out, '\n')
        l = strip(line)
        if occursin("The Directory Table", l)
            dirs = Dict{Int,String}(); mode = :dirs; continue
        elseif occursin("The File Name Table", l)
            push!(tables, Dict{Int,String}()); mode = :files; legacy_columns = false; continue
        elseif isempty(l) || startswith(l, "Entry") || occursin("Offset:", l)
            mode == :files && isempty(l) && (mode = :none)
            mode == :files && startswith(l, "Entry") &&
                (legacy_columns = occursin("Time", l) && occursin("Size", l))
            continue
        end

        if mode == :dirs
            m = match(r"^(\d+)\s+(.*)$", l)
            m === nothing && continue
            dirs[parse(Int, m.captures[1])] = strip_indirect(String(m.captures[2]))
        elseif mode == :files && !isempty(tables)
            # Entry, Dir, [MD5,] Name — MD5 is present only when the producer emitted it.
            m = legacy_columns ? match(r"^(\d+)\s+(\d+)\s+\d+\s+\d+\s+(.*)$", l) :
                                 match(r"^(\d+)\s+(\d+)\s+(?:0x[0-9a-fA-F]+\s+)?(.*)$", l)
            m === nothing && continue
            idx = parse(Int, m.captures[1]); dir = parse(Int, m.captures[2])
            name = strip_indirect(String(m.captures[3]))
            isempty(name) && continue
            base = get(dirs, dir, "")
            path = if startswith(name, "/")
                name
            elseif startswith(base, "/") || isempty(base)
                isempty(base) ? name : joinpath(base, name)
            else
                # Directory itself is relative to the compilation directory (dir 0).
                joinpath(get(dirs, 0, ""), base, name)
            end
            tables[end][idx] = path
        end
    end
    return tables
end

"""
    _is_system_decl_file(path) -> Bool

True when `path` is a toolchain or system header rather than project code.

A **blocklist**, deliberately, not a project-root allowlist: dropping a type the
project actually declares would produce an undeclared name in a signature, while
keeping a system type that slipped through costs only noise. So the failure mode
is biased toward keeping too much.
"""
function _is_system_decl_file(path::AbstractString)::Bool
    isempty(path) && return false
    p = _canon_path(path)
    for pat in ("/usr/include", "/usr/lib/gcc", "/usr/lib/clang", "/usr/lib64/gcc",
                "/usr/local/include", "/include/c++/", "/lib/clang/", "/lib64/gcc/",
                # MSYS2 / WinSDK — after `_canon_path`, backslashes are `/`.
                # `/include/c++/` already catches libc++ under clang64; these
                # catch the C library and compiler resource headers that do not
                # sit under `c++/` or `lib/clang/`.
                "/clang64/include", "/ucrt64/include", "/mingw64/include",
                "/mingw32/include", "/program files/llvm/include",
                "/windows kits/")
        occursin(pat, p) && return true
    end
    return false
end

"""
    _dwarf_format_mismatch_error(dumper, binary_path, nbytes) -> String

The empty-parse guard. Interpolating a removed local (`readelf_tool`) used to
throw `UndefVarError` on this path — hiding the mismatch this exists to name.
"""
function _dwarf_format_mismatch_error(dumper, binary_path::AbstractString, nbytes::Integer)::String
    flag = dumper.dwarf * "info"
    """
    DWARF parse produced no functions from a $(round(nbytes / 1024)) KB dump of
    $(basename(binary_path)).
    The dump is non-empty, so this is a format mismatch rather than a
    library without debug info. `parse_dwarf_dump` expects GNU binutils
    `$flag` output; check that `$(dumper.tool)` is GNU binutils
    (objdump/readelf) and not a differently-formatted stand-in
    (llvm-objdump, llvm-readelf, llvm-dwarfdump)."""
end

"""
Extract return types and struct definitions from DWARF debug info.
Returns: (return_types, struct_defs, global_vars, typedef_table, record_abi)
  - return_types: Dict{mangled_name => {c_type, julia_type, size}}
  - struct_defs: Dict{struct_name => {members: [{name, type, offset}]}}
  - record_abi: Dict{record_name => {byte_size, pass, empty}} for by-value records
"""
function extract_dwarf_return_types(binary_path::String;
                                    exported::Union{Nothing,AbstractSet{String}} = nothing)::Tuple{Dict{String,Dict{String,Any}}, Dict{String,Dict{String,Any}}, Dict{String,Any}, Dict{String,String}, Dict{String,Dict{String,Any}}}
    # GNU binutils format only — `_dwarf_dumper()` picks readelf (ELF) or
    # objdump (PE). Both print from binutils' dwarf.c; llvm-dwarfdump is a
    # different dialect and is rejected there. A fallback that returns empty
    # ships guessed signatures, so: no fallback.
    dumper = nothing
    if _dwarf_reader() === :libdw
        # In process: libdw walks the DIE tree and LibDW writes it in the
        # dumper's format, so the parser below is the same one either way.
        # `DumpLines` produces the lines as the parser asks for them, so
        # the dump is never in memory.
        output = LibDW.DumpLines(binary_path)
        file_tables = try
            LibDW.file_tables(binary_path)
        catch e
            @debug "DWARF file tables unavailable; system-header filter disabled" exception=e
            Vector{Dict{Int,String}}()
        end
    else
        dumper = _dwarf_dumper()
        dumper === nothing && error(_no_dwarf_dumper_error())

        (output, exitcode) = BuildBridge.execute(dumper.tool, [dumper.dwarf * "info", binary_path])
        exitcode == 0 || error("$(dumper.tool) $(dumper.dwarf)info failed on $binary_path:\n$output")

        # DW_AT_decl_file is a per-CU INDEX in readelf's output (llvm-dwarfdump
        # resolves it to a path; readelf does not), so the line-program file tables
        # are a second, separate dump. Failure here is non-fatal: without tables the
        # provenance filter simply does not run and extraction behaves as before.
        file_tables = try
            _dwarf_file_tables(dumper, binary_path)
        catch e
            @debug "DWARF file tables unavailable; system-header filter disabled" exception=e
            Vector{Dict{Int,String}}()
        end
    end

    (return_types, struct_defs, global_vars, typedefs, record_abi) = parse_dwarf_dump(output; file_tables=file_tables,
                                                                                      exported=exported)

    # A non-empty dump that yields nothing is a dialect or format mismatch, not
    # a library without debug info — and it is indistinguishable downstream from
    # "this library has no functions", which is how the dead fallback above went
    # unnoticed. Fail here instead of emitting a wrapper of guesses.
    dump_bytes = output isa LibDW.DumpLines ? output.bytes : length(output)
    if isempty(return_types) && dump_bytes > 4096
        dumper === nothing &&
            error("libdw read $dump_bytes bytes of DWARF from $binary_path and the parser " *
                  "found no function in it. Set REPLIBUILD_DWARF=readelf to read it with the " *
                  "GNU dumper instead, and report the library.")
        error(_dwarf_format_mismatch_error(dumper, binary_path, dump_bytes))
    end

    return (return_types, struct_defs, global_vars, typedefs, record_abi)
end

"""
    _dwarf_reader() -> Symbol

What reads a library's DWARF: `:readelf` (the GNU dumper `_dwarf_dumper`
finds, as a subprocess) or `:libdw` (elfutils' libdw, in this process, through
`LibDW`). `REPLIBUILD_DWARF` chooses; unset, it is `:readelf`. libdw reads ELF
only, so asking for it anywhere else, or where it cannot be loaded, is an
error and never a silent fall back to the other reader: the two are meant to
give one answer, and a difference must be seen.
"""
function _dwarf_reader()::Symbol
    v = lowercase(strip(get(ENV, "REPLIBUILD_DWARF", "")))
    (isempty(v) || v == "readelf") && return :readelf
    v == "libdw" || error("REPLIBUILD_DWARF is $(repr(v)); it takes `readelf` or `libdw`")
    Sys.islinux() || error("REPLIBUILD_DWARF=libdw: libdw reads ELF, and this host's libraries are not ELF")
    LibDW.available() || error("REPLIBUILD_DWARF=libdw: libdw (Elfutils_jll) cannot be loaded on this host")
    return :libdw
end

"""
    _arity_sample(msgs; limit=20) -> String

Render an arity-mismatch list for a log message, capped.

The under-count list is per-symbol and, on a C++ library, effectively unbounded:
llama.cpp produced **73,570 lines** in a single `@warn` — every inlined
libstdc++ destructor and lambda with a declaration-only DIE — which buried the
compile and link output that actually needed reading. The count is the signal;
the names are a sample. Sorted first so the sample is stable across runs.
"""
function _arity_sample(msgs::Vector{String}; limit::Int=20)::String
    sorted = sort(msgs)
    shown = join(first(sorted, limit), "\n  ")
    length(sorted) <= limit && return shown
    return "$shown\n  … and $(length(sorted) - limit) more (of $(length(sorted)) total)"
end

"""
    check_param_arity!(return_types, die_param_counts)

Fail loudly when an extracted signature disagrees with the DIE tree it came from.

The parameter list drives the emitted `ccall` argument tuple, so a count that
does not match the subprogram DIE's own `DW_TAG_formal_parameter` children is a
wrong-ABI call the caller cannot see — it type-checks, links, and corrupts the
stack at runtime. `die_param_counts` is the independent witness: a plain count
off the readelf tree depth, not derived from the name/type extraction path.

Over-count (a phantom parameter, as in the `free_opaque` leak) always aborts.
Under-count is reported as a warning: a declaration-only DIE legitimately
carries unnamed parameters that nothing can be extracted from, and the
definition DIE elsewhere in the dump is what supplies the real list.
"""
function check_param_arity!(return_types::Dict{String,Dict{String,Any}}, die_param_counts::Dict{String,Int})
    phantom = String[]
    dropped = String[]

    for (key, info) in return_types
        haskey(die_param_counts, key) || continue  # never opened as a subprogram DIE
        extracted = length(get(info, "parameters", []))
        in_die = die_param_counts[key]
        extracted == in_die && continue
        msg = "$key: extracted $extracted, DIE tree has $in_die"
        extracted > in_die ? push!(phantom, msg) : push!(dropped, msg)
    end

    if !isempty(dropped)
        @warn """
        DWARF parameter extraction dropped parameters present in the DIE tree.
        Wrappers for these functions will be called with too few arguments:
          $(_arity_sample(dropped))"""
    end

    if !isempty(phantom)
        error("""
        DWARF parameter extraction produced phantom parameters not in the DIE tree.
        This emits a ccall with the wrong argument tuple — refusing to continue:
          $(_arity_sample(phantom))
        The DIE tree is authoritative; the extraction walk mis-attributed \
        parameters from a neighbouring DIE.""")
    end

    return nothing
end

"""
    _dwarf_count(line) -> Union{Int,Nothing}

The unsigned constant of an attribute line, `: 19` or `: 0x10000`. GNU readelf
prints a constant by the size it is stored in: one or two bytes in decimal,
four or eight in hex. An array bound read as decimal only took the `0` of
`0x10000`, and every array of 65536 elements or more (lz4's hash tables,
miniz's dictionaries, pcre2's frames) had no elements and no size.
"""
function _dwarf_count(line::AbstractString)::Union{Int,Nothing}
    m = match(r":\s*(0x[0-9a-fA-F]+|\d+)", line)
    m === nothing && return nothing
    v = m.captures[1]
    return startswith(v, "0x") ? parse(Int, v[3:end]; base = 16) : parse(Int, v)
end

# The lines of a dump, one at a time: of a dumper's text without a vector of
# them, or straight off libdw's walk (`LibDW.DumpLines`), which never holds
# the text at all. The parser makes three passes, so a source is iterated
# three times.
_dump_lines(output::AbstractString) = eachsplit(output, '\n')
_dump_lines(output::LibDW.DumpLines) = output

"""
    parse_dwarf_dump(output) -> (return_types, struct_defs, global_vars, typedef_table, record_abi)

Parse a GNU `readelf --debug-dump=info` dump into the type/signature tables.

Split out from [`extract_dwarf_return_types`](@ref) so DIE-tree attribution can
be pinned with synthetic dumps — no compiler, binary, or fixture build needed.
See `test/test_dwarf_attribution.jl`.

`file_tables` (one `index => path` map per CU, from [`_dwarf_file_tables`](@ref))
enables the system-header provenance filter: a type declared in a toolchain
header is dropped instead of emitted. Omitted — as every synthetic-dump test
does — the filter simply does not run, and parsing behaves exactly as before.
"""
function parse_dwarf_dump(output::Union{AbstractString, LibDW.DumpLines};
                          file_tables::Vector{Dict{Int,String}}=Vector{Dict{Int,String}}(),
                          exported::Union{Nothing,AbstractSet{String}}=nothing
                         )::Tuple{Dict{String,Dict{String,Any}}, Dict{String,Dict{String,Any}}, Dict{String,Any}, Dict{String,String}, Dict{String,Dict{String,Any}}}
    # Which CU we are inside, 1-based; readelf emits line programs and DIEs in the
    # same section order, so the Nth "Compilation Unit @ offset" uses the Nth table.
    cu_index = 0
    cu_comp_dir = ""
    return_types = Dict{String,Dict{String,Any}}()

    # Parse DWARF output to extract type information
    # Strategy: Find DW_TAG_subprogram entries, extract linkage_name and return type

    current_function = nothing
    current_linkage_name = nothing
    type_refs = Dict{String,Any}()  # offset => type_name (String) or type_info (Dict)
    base_byte_size = Dict{String,Int}()  # base-type offset => DW_AT_byte_size

    # First pass: Build type reference table
    # We need to handle: base types, pointer types, const types, reference types
    offset_to_kind = Dict{String,Symbol}()  # Track what kind each offset is
    current_type_offset = nothing  # Track current type/struct being processed
    current_struct_context = nothing  # Track parent struct for members

    # The open DW_TAG_subroutine_type, its depth, and whether the open DIE is one
    # of its formal parameters (a direct child). Its parameter types are
    # collected here, in dump order, and not in pass 2: pass 2 resolves each
    # signature as it meets it, and clang emits a subroutine type after the
    # first function that takes it, so a list built there would be complete for
    # some signatures and empty for others. Pass 2 never collected one at all,
    # so every function pointer was `function_ptr(R)*`.
    current_subroutine_offset = nothing
    current_subroutine_depth = nothing
    subroutine_param_open = false

    # Depth-indexed context: readelf DIE headers carry the tree depth
    # (`<2><2331>: ...`). A member/enumerator/inheritance DIE at depth d
    # belongs to the type DIE last seen at depth d-1. The flat
    # current_struct_context alone mis-attributes members that FOLLOW a
    # nested type definition (e.g. `Type m_type; enum Type {...}; float
    # m_radius;` — clang emits the nested enum DIE between the two members,
    # the context flipped to the enum and never restored, and m_radius
    # silently vanished from the class; found via box2d b2Shape).
    context_by_depth = Dict{Int,String}()
    function set_ctx_at_depth!(depth::Int, off::String)
        context_by_depth[depth] = off
        for k in collect(keys(context_by_depth))
            k > depth && delete!(context_by_depth, k)
        end
    end
    parent_ctx(depth::Union{Int,Nothing}) =
        depth === nothing ? current_struct_context :
        get(context_by_depth, depth - 1, current_struct_context)
    function die_depth(line)
        m = match(r"<(\d+)><[^>]+>", line)
        m === nothing ? nothing : parse(Int, m.captures[1])
    end

    # The DIE a record, enum or namespace is declared in, for the `scope` the
    # metadata gives each record (`_scope_of`): the namespace or record last
    # opened one level up, provided it is also the DIE last opened there. A
    # subprogram sets no context, so a record local to a function would read a
    # stale namespace or record at its parent depth; the header tracker below
    # catches that and the record gets no scope (`missing`, unknown) rather
    # than a wrong one. `nothing` is the global scope, at depth 1.
    last_die_by_depth = Dict{Int,String}()
    function scope_parent(depth::Union{Int,Nothing})
        depth === nothing && return missing
        depth <= 1 && return nothing
        c = get(context_by_depth, depth - 1, nothing)
        c === nothing && return missing
        return get(last_die_by_depth, depth - 1, nothing) == c ? c : missing
    end

    # Flush a completed member into its parent struct's members array.
    # Called on each new DW_TAG_* to ensure all attributes (name, type, offset)
    # are collected before the member dict is copied to the parent.
    function flush_pending_member()
        if !haskey(type_refs, "_pending_member_offset")
            return
        end
        mo = type_refs["_pending_member_offset"]
        if haskey(type_refs, mo) && isa(type_refs[mo], Dict) && get(type_refs[mo], "kind", "") == "member"
            mi = type_refs[mo]
            parent_off = get(mi, "parent", nothing)
            if !isnothing(parent_off) && haskey(type_refs, parent_off) &&
               isa(type_refs[parent_off], Dict) && haskey(type_refs[parent_off], "members")
                # A NAMELESS member is kept here rather than dropped: a C11
                # anonymous `union {...};` / `struct {...};` member has no
                # DW_AT_name, and dropping it lost the entire embedded tree.
                # It gets a synthetic field name later (the anonymous-aggregate
                # pass, which runs once the whole dump is parsed and the member's
                # type DIE — emitted AFTER the member that references it — can
                # finally be inspected); anything still nameless there is purged.
                if !isnothing(mi["type"])
                    existing_names = [m["name"] for m in type_refs[parent_off]["members"]]
                    if isnothing(mi["name"]) || !(mi["name"] in existing_names)
                        member_dict = Dict{String, Any}(
                            "name" => mi["name"],
                            "type" => mi["type"],
                            "offset" => mi["offset"]
                        )
                        for bf_key in ["bit_size", "data_bit_offset", "bit_offset_legacy", "alignment"]
                            if haskey(mi, bf_key)
                                member_dict[bf_key] = mi[bf_key]
                            end
                        end
                        push!(type_refs[parent_off]["members"], member_dict)
                    end
                end
            end
        end
        delete!(type_refs, "_pending_member_offset")
    end

    for line in _dump_lines(output)
        line = strip(line)

        # CU boundary — DW_AT_decl_file indices are CU-relative, so the table must
        # advance with them.
        if startswith(line, "Compilation Unit @")
            cu_index += 1
            cu_comp_dir = ""
        end
        # The unit's compilation directory: what a relative file name in its
        # line table is relative to. DWARF 5 puts that directory in the table
        # (entry 0) and the names come out absolute; DWARF 4 does not.
        if contains(line, "DW_AT_comp_dir")
            cm = match(r"DW_AT_comp_dir\s*:\s*(?:\([^)]*\):\s*)?(.*)$", line)
            cm === nothing || (cu_comp_dir = String(strip(cm.captures[1])))
        end

        # Track ANY tag to avoid attribute pollution
        # Tags have format: <level><offset>: Abbrev Number: N (DW_TAG_*)
        if contains(line, "DW_TAG_")
            # Flush any pending member before starting a new tag
            flush_pending_member()
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                tag_offset = "0x" * offset_match.captures[1]
                type_refs["last_tag_offset"] = tag_offset  # Always update for ANY tag
                let d = die_depth(line)
                    d === nothing || (last_die_by_depth[d] = tag_offset)
                end
            end
        end

        # A DIE header, terminators included, ends the previous DIE's attributes,
        # and a header back at the subroutine type's depth or above ends its
        # children. A direct child is a parameter, or `...`.
        let hm = match(r"^<(\d+)><[^>]+>", line)
            if hm !== nothing
                subroutine_param_open = false
                hdr_depth = parse(Int, hm.captures[1])
                if current_subroutine_depth !== nothing && hdr_depth <= current_subroutine_depth
                    current_subroutine_offset = nothing
                    current_subroutine_depth = nothing
                elseif current_subroutine_depth !== nothing && hdr_depth == current_subroutine_depth + 1
                    if contains(line, "DW_TAG_formal_parameter")
                        subroutine_param_open = true
                    elseif contains(line, "DW_TAG_unspecified_parameters")
                        type_refs[current_subroutine_offset]["variadic"] = true
                    end
                end
            end
        end
        if subroutine_param_open && contains(line, "DW_AT_type")
            pm = match(r"<(0x[^>]+)>", line)
            pm === nothing || push!(type_refs[current_subroutine_offset]["parameters"], String(pm.captures[1]))
            subroutine_param_open = false
        end

        # Record where a type was DECLARED. This is the only provenance DWARF
        # carries, and it is what separates a type the project defines from one
        # `#include <iostream>` dragged in: hello_world.cpp declares ZERO types,
        # yet six reach the wrapper (`max_align_t`, `__mbstate_t`, `ldiv_t`, …)
        # along with accessors and exports for them. Functions never had this
        # problem because they are gated by `nm --defined-only`; types had no
        # gate at all.
        if contains(line, "DW_AT_decl_file") && !isempty(file_tables) &&
           haskey(type_refs, "last_tag_offset")
            fm = match(r"DW_AT_decl_file\s*:?\s*(\d+)", line)
            if !isnothing(fm) && 1 <= cu_index <= length(file_tables)
                tgt = type_refs["last_tag_offset"]
                if haskey(type_refs, tgt) && isa(type_refs[tgt], Dict)
                    path = get(file_tables[cu_index], parse(Int, fm.captures[1]), "")
                    (!isempty(path) && !isabspath(path) && !isempty(cu_comp_dir)) &&
                        (path = joinpath(cu_comp_dir, path))
                    isempty(path) || (type_refs[tgt]["decl_file_path"] = path)
                end
            end
        end

        # Extract base type definitions (DW_TAG_base_type)
        # Example: <1><27>: Abbrev Number: 2 (DW_TAG_base_type)
        if contains(line, "DW_TAG_base_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                type_refs[current_type_offset] = "unknown"
                offset_to_kind[current_type_offset] = :base
            end
        end

        # Extract pointer type definitions (DW_TAG_pointer_type)
        # Example: <1><9f6>: Abbrev Number: 40 (DW_TAG_pointer_type)
        #          <9f7>   DW_AT_type        : <0x41>
        if contains(line, "DW_TAG_pointer_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                type_refs[current_type_offset] = Dict{String,Any}("kind" => "pointer", "target" => nothing)
                offset_to_kind[current_type_offset] = :pointer
            end
        end

        # Extract const type definitions (DW_TAG_const_type)
        # Example: <1><41>: Abbrev Number: 5 (DW_TAG_const_type)
        #          <42>   DW_AT_type        : <0x46>
        if contains(line, "DW_TAG_const_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                type_refs[current_type_offset] = Dict{String,Any}("kind" => "const", "target" => nothing)
                offset_to_kind[current_type_offset] = :const
            end
        end

        # Extract reference type definitions (DW_TAG_reference_type)
        if contains(line, "DW_TAG_reference_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                type_refs[current_type_offset] = Dict{String,Any}("kind" => "reference", "target" => nothing)
                offset_to_kind[current_type_offset] = :reference
            end
        end

        # `T&&` (DW_TAG_rvalue_reference_type). An address at the ABI level, exactly
        # like `T&`, and it resolves to `T&&` so every mapper sees a reference. It
        # was unmodelled, so a move constructor's `T&&` parameter resolved to
        # "unknown": `Any` in the signature, and an untyped slot in the thunk.
        # `contains(line, "DW_TAG_reference_type")` above does not match this tag.
        if contains(line, "DW_TAG_rvalue_reference_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                type_refs[current_type_offset] = Dict{String,Any}("kind" => "rvalue_reference", "target" => nothing)
                offset_to_kind[current_type_offset] = :reference
            end
        end

        # Qualifiers that change nothing at the call boundary: `restrict`,
        # `volatile`, `_Atomic`. Each is its own DIE in the type chain
        # (`a: restrict → pointer → const → int`), exactly like `const`, and
        # each is resolved straight through to its target (see resolve_type).
        #
        # Without an entry here the DIE has no `type_refs` slot at all, so
        # `resolve_type` answers "unknown" for everything behind it. For
        # `restrict` that is the PARAMETER itself — the qualifier sits on the
        # pointer — so `int *restrict a` mapped to `Any`, and `Any` in a ccall
        # argument tuple passes a boxed `jl_value_t*`: the callee read Julia's
        # object header as its array. Silent garbage on Tier 3, and
        # `Malformed llvmcall` on Tier 1. Hub cglm shipped 58 of these
        # (`glmc_mat4_make(const float *__restrict src, …)`).
        #
        # `_Atomic T` is transparent only for scalars ≤ 8 bytes, which is where
        # x86-64 gives it T's size and alignment. An `_Atomic` struct can be
        # padded larger than T; its enclosing layout still comes out right,
        # because member offsets and `byte_size` are read from DWARF, not
        # recomputed from the member types.
        let qm = match(r"\(DW_TAG_(restrict|volatile|atomic)_type\)", line)
            if !isnothing(qm)
                offset_match = match(r"<\d+><([^>]+)>", line)
                if !isnothing(offset_match)
                    current_type_offset = "0x" * offset_match.captures[1]
                    type_refs[current_type_offset] = Dict{String,Any}(
                        "kind" => String(qm.captures[1]), "target" => nothing)
                    offset_to_kind[current_type_offset] = :qualifier
                end
            end
        end

        # Extract struct type definitions (DW_TAG_structure_type)
        # Example: <1><1f5f3e>: DW_TAG_structure_type
        #          DW_AT_name: "Vector3d"
        #          DW_AT_byte_size: 0x18
        if contains(line, "DW_TAG_structure_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                current_struct_context = current_type_offset  # Set context for members
                set_ctx_at_depth!(something(die_depth(line), 1), current_type_offset)
                # Store struct info as a dict with members array
                type_refs[current_type_offset] = Dict{String,Any}(
                    "kind" => "struct",
                    "name" => "unknown_struct",
                    "members" => [],
                    "scope_parent" => scope_parent(die_depth(line))
                )
                offset_to_kind[current_type_offset] = :struct
            end
        end

        # Extract union type definitions (DW_TAG_union_type)
        if contains(line, "DW_TAG_union_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                current_struct_context = current_type_offset  # Set context for members
                set_ctx_at_depth!(something(die_depth(line), 1), current_type_offset)
                # Store union info as a dict with members array
                type_refs[current_type_offset] = Dict{String,Any}(
                    "kind" => "union",
                    "name" => "unknown_union",
                    "members" => [],
                    "scope_parent" => scope_parent(die_depth(line))
                )
                offset_to_kind[current_type_offset] = :struct  # Treat as struct for generic handling
            end
        end

        # Extract class type definitions (DW_TAG_class_type)
        # Similar to struct, C++ distinguishes but for our purposes treat the same
        if contains(line, "DW_TAG_class_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                current_struct_context = current_type_offset  # Set context for members
                set_ctx_at_depth!(something(die_depth(line), 1), current_type_offset)
                # Store class info as a dict with members array
                type_refs[current_type_offset] = Dict{String,Any}(
                    "kind" => "class",
                    "name" => "unknown_class",
                    "members" => [],
                    "scope_parent" => scope_parent(die_depth(line))
                )
                offset_to_kind[current_type_offset] = :class
            end
        end

        # Extract enum type definitions (DW_TAG_enumeration_type)
        # Example: <1><abc>: DW_TAG_enumeration_type
        #          DW_AT_name: "Color"
        #          DW_AT_type: <0x27>  (underlying type, e.g., int)
        #          DW_AT_byte_size: 0x04
        if contains(line, "DW_TAG_enumeration_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                current_struct_context = current_type_offset  # Use for enumerators
                set_ctx_at_depth!(something(die_depth(line), 1), current_type_offset)
                # Store enum info as a dict with enumerators array
                type_refs[current_type_offset] = Dict{String,Any}(
                    "kind" => "enum",
                    "name" => "unknown_enum",
                    "underlying_type" => nothing,
                    "byte_size" => nothing,
                    "enumerators" => [],
                    "scope_parent" => scope_parent(die_depth(line))
                )
                offset_to_kind[current_type_offset] = :enum
            end
        end

        # Extract array type definitions (DW_TAG_array_type)
        # Example: <1><def>: DW_TAG_array_type
        #          DW_AT_type: <0x27>  (element type)
        if contains(line, "DW_TAG_array_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                # Store array info with element type and dimensions
                type_refs[current_type_offset] = Dict{String,Any}(
                    "kind" => "array",
                    "element_type" => nothing,
                    "dimensions" => []
                )
                offset_to_kind[current_type_offset] = :array
            end
        end

        # Extract array subrange (dimensions)
        # Example: <2><xyz>: DW_TAG_subrange_type
        #          DW_AT_type: <0x123>  (index type)
        #          DW_AT_upper_bound: 8  (for array[9])
        if contains(line, "DW_TAG_subrange_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                subrange_offset = "0x" * offset_match.captures[1]
                type_refs[subrange_offset] = Dict{String,Any}(
                    "kind" => "subrange",
                    "upper_bound" => nothing,
                    "parent" => current_type_offset  # Link to parent array
                )
                offset_to_kind[subrange_offset] = :subrange
            end
        end

        # Extract enumerator (enum member)
        # Example: <2><ghi>: DW_TAG_enumerator
        #          DW_AT_name: "Red"
        #          DW_AT_const_value: 0
        if contains(line, "DW_TAG_enumerator")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                enumerator_offset = "0x" * offset_match.captures[1]
                type_refs[enumerator_offset] = Dict{String,Any}(
                    "kind" => "enumerator",
                    "name" => nothing,
                    "value" => nothing,
                    "parent" => parent_ctx(die_depth(line))  # Track which enum this belongs to
                )
                offset_to_kind[enumerator_offset] = :enumerator
            end
        end

        # Extract subroutine type (function pointer signature)
        # Example: <1><jkl>: DW_TAG_subroutine_type
        #          DW_AT_type: <0x27>  (return type)
        #          <2><jkm>: DW_TAG_formal_parameter  (a parameter; see above)
        if contains(line, "DW_TAG_subroutine_type")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                current_subroutine_offset = current_type_offset  # Track for param collection
                current_subroutine_depth = something(die_depth(line), 1)
                # Store function signature info
                type_refs[current_type_offset] = Dict{String,Any}(
                    "kind" => "subroutine",
                    "return_type" => nothing,
                    "parameters" => []
                )
                offset_to_kind[current_type_offset] = :subroutine
            end
        end

        # Extract typedef (type aliases)
        # Example: <1><390>: Abbrev Number: 34 (DW_TAG_typedef)
        #          <391>   DW_AT_type        : <0x398>
        #          <395>   DW_AT_name        : int32_t
        if contains(line, "DW_TAG_typedef")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_type_offset = "0x" * offset_match.captures[1]
                # Store typedef info - will resolve to underlying type
                type_refs[current_type_offset] = Dict{String,Any}(
                    "kind" => "typedef",
                    "name" => nothing,
                    "target" => nothing
                )
                offset_to_kind[current_type_offset] = :typedef
            end
        end

        # Extract inheritance (base class)
        # Example: <2><218>: DW_TAG_inheritance
        #          <219>   DW_AT_type        : <0x135>  (base class ref)
        #          <21d>   DW_AT_data_member_location: 0
        #          <21e>   DW_AT_accessibility: DW_ACCESS_public
        if contains(line, "DW_TAG_inheritance")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                inheritance_offset = "0x" * offset_match.captures[1]
                type_refs[inheritance_offset] = Dict{String,Any}(
                    "kind" => "inheritance",
                    "base_type" => nothing,
                    "offset" => 0,
                    "accessibility" => "public",  # default
                    "parent" => parent_ctx(die_depth(line))
                )
                offset_to_kind[inheritance_offset] = :inheritance
            end
        end

        # Extract template type parameter
        # Example: <2><2448>: DW_TAG_template_type_parameter
        #          <2449>   DW_AT_type        : <0x1d9>  (actual type in instantiation)
        #          <244d>   DW_AT_name        : T
        if contains(line, "DW_TAG_template_type_param")   # binutils: _param; llvm-dwarfdump: _parameter (a prefix match takes both)
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                template_offset = "0x" * offset_match.captures[1]
                type_refs[template_offset] = Dict{String,Any}(
                    "kind" => "template_type",
                    "name" => nothing,
                    "type" => nothing,
                    "parent" => parent_ctx(die_depth(line))
                )
                offset_to_kind[template_offset] = :template_type
            end
        end

        # Extract template value parameter
        # Example: <2><247b>: DW_TAG_template_value_parameter
        #          <247c>   DW_AT_type        : <0x1d9>
        #          <2480>   DW_AT_name        : N
        #          <2481>   DW_AT_const_value : 10
        if contains(line, "DW_TAG_template_value_param")  # both spellings, as above
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                template_offset = "0x" * offset_match.captures[1]
                type_refs[template_offset] = Dict{String,Any}(
                    "kind" => "template_value",
                    "name" => nothing,
                    "type" => nothing,
                    "value" => nothing,
                    "parent" => parent_ctx(die_depth(line))
                )
                offset_to_kind[template_offset] = :template_value
            end
        end

        # Extract namespace
        # Example: <1><8bf>: DW_TAG_namespace
        #          <8c0>   DW_AT_name        : math
        if contains(line, "DW_TAG_namespace")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                ns_offset = "0x" * offset_match.captures[1]
                type_refs[ns_offset] = Dict{String,Any}(
                    "kind" => "namespace",
                    "name" => nothing,
                    "scope_parent" => scope_parent(die_depth(line))
                )
                offset_to_kind[ns_offset] = :namespace
                # A namespace is a scope the records under it are declared in.
                set_ctx_at_depth!(something(die_depth(line), 1), ns_offset)
            end
        end

        # Extract type name for base types, structs, classes, enums, and new types
        # Example: <28>   DW_AT_name        : (indexed string: 0x3): int
        if contains(line, "DW_AT_name") && haskey(type_refs, "last_tag_offset")
            tag_offset = type_refs["last_tag_offset"]
            if haskey(offset_to_kind, tag_offset) && offset_to_kind[tag_offset] in [:base, :struct, :class, :enum, :enumerator, :typedef, :template_type, :template_value, :namespace]
                name_match_quotes = match(r"DW_AT_name\s*\(\"([^\"]+)\"\)", line)
                type_name = ""
                if !isnothing(name_match_quotes)
                    type_name = String(name_match_quotes.captures[1])
                else
                    name_match = match(r"DW_AT_name\s*:\s*(?:\([^)]*\):\s*)?(.*)$", line)
                    if !isnothing(name_match)
                        type_name = String(strip(name_match.captures[1]))
                    end
                end
                
                if !isempty(type_name)
                    if offset_to_kind[tag_offset] == :base
                        # Base types are stored as simple strings. DWARF names
                        # every `_Complex` width `complex`; the byte size (which
                        # may already have been seen) is what distinguishes them.
                        if type_name == "complex"
                            type_name = _complex_base_name(get(base_byte_size, tag_offset, 0))
                        end
                        type_refs[tag_offset] = type_name
                    elseif offset_to_kind[tag_offset] in [:struct, :class, :enum, :typedef, :template_type, :template_value, :namespace]
                        # These types are dicts - update the name field
                        if haskey(type_refs, tag_offset) && isa(type_refs[tag_offset], Dict)
                            type_refs[tag_offset]["name"] = type_name
                        end
                    elseif offset_to_kind[tag_offset] == :enumerator
                        # Enumerator: store name
                        if haskey(type_refs, tag_offset) && isa(type_refs[tag_offset], Dict)
                            type_refs[tag_offset]["name"] = type_name
                        end
                    end
                end
            end
        end

        # Extract DW_AT_type for pointer/const/reference/enum/array/subroutine/typedef/template/inheritance types
        # Example: <9f7>   DW_AT_type        : <0x41>
        if contains(line, "DW_AT_type") && haskey(type_refs, "last_tag_offset")
            tag_offset = type_refs["last_tag_offset"]
            if haskey(offset_to_kind, tag_offset) && offset_to_kind[tag_offset] in [:pointer, :const, :reference, :qualifier, :enum, :array, :subroutine, :typedef, :template_type, :template_value, :inheritance]
                type_match = match(r"<(0x[^>]+)>", line)
                if !isnothing(type_match)
                    target_offset = String(type_match.captures[1])  # Convert SubString to String
                    if haskey(type_refs, tag_offset) && isa(type_refs[tag_offset], Dict)
                        # Pointer/const/reference/qualifier: target
                        if offset_to_kind[tag_offset] in [:pointer, :const, :reference, :qualifier]
                            type_refs[tag_offset]["target"] = target_offset
                        # Enum: underlying type
                        elseif offset_to_kind[tag_offset] == :enum
                            type_refs[tag_offset]["underlying_type"] = target_offset
                        # Array: element type
                        elseif offset_to_kind[tag_offset] == :array
                            type_refs[tag_offset]["element_type"] = target_offset
                        # Subroutine: return type
                        elseif offset_to_kind[tag_offset] == :subroutine
                            type_refs[tag_offset]["return_type"] = target_offset
                        # Typedef: target type
                        elseif offset_to_kind[tag_offset] == :typedef
                            type_refs[tag_offset]["target"] = target_offset
                        # Template type/value parameter: actual type
                        elseif offset_to_kind[tag_offset] in [:template_type, :template_value]
                            type_refs[tag_offset]["type"] = target_offset
                        # Inheritance: base class type
                        elseif offset_to_kind[tag_offset] == :inheritance
                            type_refs[tag_offset]["base_type"] = target_offset
                        end
                    end
                end
            end
        end

        # Inheritance attributes beyond the base type ref: the base subobject's
        # byte offset in the derived object (DW_AT_data_member_location — the
        # whole point of multiple inheritance; non-zero for every non-primary
        # base) and virtual-base marking (DW_AT_virtuality). Without the offset
        # every base was silently recorded at 0.
        if haskey(type_refs, "last_tag_offset")
            tag_offset = type_refs["last_tag_offset"]
            if haskey(offset_to_kind, tag_offset) && offset_to_kind[tag_offset] == :inheritance &&
               haskey(type_refs, tag_offset) && isa(type_refs[tag_offset], Dict)
                inh_info = type_refs[tag_offset]

                if contains(line, "DW_AT_data_member_location")
                    if contains(line, "byte block")
                        # Virtual base: the location is a DWARF EXPRESSION, not a
                        # constant — readelf renders it like
                        #   7 byte block: ... (DW_OP_dup; DW_OP_deref;
                        #     DW_OP_constu: 24; DW_OP_minus; DW_OP_deref; DW_OP_plus)
                        # i.e. "read the vptr, read the vbase-offset entry N bytes
                        # BELOW the vtable address point, add it to this". Record
                        # the vtable-relative position (negative); the static
                        # "offset" field stays 0 — there IS no static offset.
                        # (Without the byte-block guard the constant regex below
                        # matched the "7" of "7 byte block" — a bogus offset 7.)
                        #
                        # The constant has TWO spellings and both occur in the
                        # wild: clang emits DW_OP_constu N, gcc emits the
                        # single-byte DW_OP_lit<N> form for N <= 31. libstdc++
                        # is gcc-built and uses DW_OP_lit24, so matching only
                        # constu left the key unset — and an unset key reads as
                        # `nothing` in GeneratorCpp's _collect_upcasts!, which
                        # silently skipped the upcast. DWARFParser.jl carries
                        # the same pair of spellings; both parsers must agree.
                        expr_match = match(r"DW_OP_dup[;,].*DW_OP_deref[;,].*(?:DW_OP_constu:?\s*(0x[0-9a-fA-F]+|\d+)|DW_OP_lit(\d+))\s*[;,].*DW_OP_minus[;,].*DW_OP_deref[;,].*DW_OP_plus", line)
                        if !isnothing(expr_match)
                            val_str = expr_match.captures[1] !== nothing ?
                                      expr_match.captures[1] : expr_match.captures[2]
                            n = startswith(val_str, "0x") ?
                                parse(Int, val_str[3:end], base=16) :
                                parse(Int, val_str)
                            inh_info["vbase_vtable_offset"] = -n
                        end
                    else
                        loc_match = match(r"DW_AT_data_member_location\s*:\s*(0x[0-9a-fA-F]+|\d+)", line)
                        if !isnothing(loc_match)
                            val_str = loc_match.captures[1]
                            inh_info["offset"] = startswith(val_str, "0x") ?
                                parse(Int, val_str[3:end], base=16) :
                                parse(Int, val_str)
                        end
                    end
                end

                # Virtual base: offset is not a static layout fact (it lives in
                # the vtable) — consumers must reject, not trust "offset".
                if contains(line, "DW_AT_virtuality") &&
                   (contains(line, "virtual") || match(r"DW_AT_virtuality\s*:\s*(0x0?[12]|[12])\b", line) !== nothing)
                    inh_info["virtual"] = true
                end
            end
        end

        # Extract DW_AT_byte_size for enums, structs, classes, unions, and base types.
        # A base type named `complex` is both `_Complex float` (8) and
        # `_Complex double` (16); the name alone cannot tell them apart, so the
        # size is recorded and the stored name is rewritten once both are known.
        if contains(line, "DW_AT_byte_size") && haskey(type_refs, "last_tag_offset")
            tag_offset = type_refs["last_tag_offset"]
            if haskey(offset_to_kind, tag_offset) && offset_to_kind[tag_offset] == :base
                size_match = match(r":\s*(0x[0-9a-fA-F]+|\d+)", line)
                if !isnothing(size_match)
                    val_str = size_match.captures[1]
                    nbytes = startswith(val_str, "0x") ?
                             parse(Int, val_str[3:end], base=16) : parse(Int, val_str)
                    base_byte_size[tag_offset] = nbytes
                    if get(type_refs, tag_offset, "") == "complex"
                        type_refs[tag_offset] = _complex_base_name(nbytes)
                    end
                end
            elseif haskey(offset_to_kind, tag_offset) && offset_to_kind[tag_offset] in [:enum, :struct, :class, :union, :array]
                # Match value after colon. An array DIE carries a byte_size only
                # when its lanes do not fill it: a SIMD vector padded to a power
                # of two (`ext_vector_type(3)` of float is 16 bytes).
                size_match = match(r":\s*(0x[0-9a-fA-F]+|\d+)", line)
                if !isnothing(size_match)
                    val_str = size_match.captures[1]
                    if haskey(type_refs, tag_offset) && isa(type_refs[tag_offset], Dict)
                        if !startswith(val_str, "0x")
                            val = parse(Int, val_str)
                            type_refs[tag_offset]["byte_size"] = "0x" * string(val, base=16)
                        else
                            type_refs[tag_offset]["byte_size"] = val_str
                        end
                    end
                end
            end
        end

        # DW_AT_declaration on an enum DIE: an opaque declaration (`enum class E
        # : int;`), which has a size and no enumerators. An enum DEFINED with
        # none (`enum class byte : unsigned char {};`) has no such attribute,
        # and only that one gets an entry (see the enum definitions below).
        if contains(line, "DW_AT_declaration") && haskey(type_refs, "last_tag_offset")
            tag_offset = type_refs["last_tag_offset"]
            if get(offset_to_kind, tag_offset, nothing) === :enum &&
               isa(get(type_refs, tag_offset, nothing), Dict)
                type_refs[tag_offset]["declaration"] = true
            end
        end

        # DW_AT_calling_convention on a record DIE is clang's own answer to "is
        # this type trivial for the purposes of calls": 5 (DW_CC_pass_by_value)
        # travels in registers or on the stack like a C struct, 4
        # (DW_CC_pass_by_reference) travels as a pointer to a caller-owned
        # temporary. Readelf spells it `DW_AT_calling_convention: 5	(pass by
        # value)`. C records carry none. The same attribute on a subprogram means
        # something else, so only a record DIE records it.
        if contains(line, "DW_AT_calling_convention") && haskey(type_refs, "last_tag_offset")
            tag_offset = type_refs["last_tag_offset"]
            if get(offset_to_kind, tag_offset, nothing) in (:struct, :class) &&
               isa(get(type_refs, tag_offset, nothing), Dict)
                ccm = match(r"DW_AT_calling_convention\s*:?\s*(\d+)", line)
                ccm === nothing ||
                    (type_refs[tag_offset]["calling_convention"] = parse(Int, ccm.captures[1]))
            end
        end

        # DW_AT_alignment: an alignment the source asked for, `aligned(N)` on a
        # record and `_Alignas(N)` on a member. Clang also puts it on a member
        # whose type is an over-aligned typedef (the typedef itself carries
        # none), and emits it in DWARF 4 too. A record's alignment decides how a
        # hidden return buffer must be aligned (GeneratorC `_c_record_align`).
        # Natural alignment has no attribute: `long double` and vectors are
        # found from their types.
        if contains(line, "DW_AT_alignment") && haskey(type_refs, "last_tag_offset")
            tag_offset = type_refs["last_tag_offset"]
            if get(offset_to_kind, tag_offset, nothing) in (:struct, :class, :member) &&
               isa(get(type_refs, tag_offset, nothing), Dict)
                am = match(r"DW_AT_alignment\s*:\s*(0x[0-9a-fA-F]+|\d+)", line)
                if am !== nothing
                    v = String(am.captures[1])
                    type_refs[tag_offset]["alignment"] =
                        startswith(v, "0x") ? parse(Int, v[3:end], base=16) : parse(Int, v)
                end
            end
        end

        # DW_AT_GNU_vector: the array is a vector (`vector_size`, `__m128`),
        # aligned to its size where an array is aligned to its element.
        if contains(line, "DW_AT_GNU_vector") && haskey(type_refs, "last_tag_offset")
            tag_offset = type_refs["last_tag_offset"]
            if get(offset_to_kind, tag_offset, nothing) == :array &&
               isa(get(type_refs, tag_offset, nothing), Dict)
                type_refs[tag_offset]["vector"] = true
            end
        end

        # Extract DW_AT_const_value for enumerators and template value parameters
        if contains(line, "DW_AT_const_value") && haskey(type_refs, "last_tag_offset")
            tag_offset = type_refs["last_tag_offset"]
            if haskey(offset_to_kind, tag_offset) && offset_to_kind[tag_offset] in [:enumerator, :template_value]
                # Value can be decimal or hex (0x...)
                value_match = match(r":\s*(-?(?:0x[0-9a-fA-F]+|\d+))", line)
                if !isnothing(value_match)
                    if haskey(type_refs, tag_offset) && isa(type_refs[tag_offset], Dict)
                        val_str = value_match.captures[1]
                        val = if startswith(val_str, "0x") || startswith(val_str, "0X")
                            parse(Int128, val_str[3:end], base=16)
                        elseif startswith(val_str, "-0x") || startswith(val_str, "-0X")
                            -parse(Int128, val_str[4:end], base=16)
                        else
                            parse(Int128, val_str)
                        end
                        type_refs[tag_offset]["value"] = val
                    end
                end
            end
        end

        # Extract DW_AT_upper_bound for array subranges (DWARF 4 and earlier)
        if contains(line, "DW_AT_upper_bound") && haskey(type_refs, "last_tag_offset")
            tag_offset = type_refs["last_tag_offset"]
            if haskey(offset_to_kind, tag_offset) && offset_to_kind[tag_offset] == :subrange
                # Upper bound can be decimal or hex: readelf prints a constant
                # stored in four or eight bytes as `0x10000`, and read as
                # decimal that was 0, so an array of 65536 elements or more
                # had none (`_dwarf_count`).
                bound = _dwarf_count(line)
                bound_match = bound === nothing ? nothing : (captures = (string(bound),),)
                if !isnothing(bound_match)
                    if haskey(type_refs, tag_offset) && isa(type_refs[tag_offset], Dict)
                        # Upper bound is size-1, so actual size is upper_bound + 1
                        type_refs[tag_offset]["upper_bound"] = parse(Int, bound_match.captures[1])

                        # Add dimension to parent array
                        parent_offset = get(type_refs[tag_offset], "parent", nothing)
                        if !isnothing(parent_offset) && haskey(type_refs, parent_offset) &&
                           isa(type_refs[parent_offset], Dict) && haskey(type_refs[parent_offset], "dimensions")
                            size = parse(Int, bound_match.captures[1]) + 1  # Array[9] has upper_bound=8
                            push!(type_refs[parent_offset]["dimensions"], size)
                        end
                    end
                end
            end
        end

        # Extract DW_AT_count for array subranges (DWARF 5+, more common)
        if contains(line, "DW_AT_count") && haskey(type_refs, "last_tag_offset")
            tag_offset = type_refs["last_tag_offset"]
            if haskey(offset_to_kind, tag_offset) && offset_to_kind[tag_offset] == :subrange
                # Count is the actual array size, decimal or hex (`_dwarf_count`).
                count_value = _dwarf_count(line)
                if !isnothing(count_value)
                    if haskey(type_refs, tag_offset) && isa(type_refs[tag_offset], Dict)
                        # Count is the actual size
                        size = count_value
                        type_refs[tag_offset]["count"] = size

                        # Add dimension to parent array
                        parent_offset = get(type_refs[tag_offset], "parent", nothing)
                        if !isnothing(parent_offset) && haskey(type_refs, parent_offset) &&
                           isa(type_refs[parent_offset], Dict) && haskey(type_refs[parent_offset], "dimensions")
                            push!(type_refs[parent_offset]["dimensions"], size)
                        end
                    end
                end
            end
        end

        # Extract struct/class members (DW_TAG_member)
        # Example: <2><aa>: DW_TAG_member
        #          DW_AT_name: "x"
        #          DW_AT_type: <0xbd>
        #          DW_AT_data_member_location: 0x00
        if contains(line, "DW_TAG_member")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                member_offset = "0x" * offset_match.captures[1]
                type_refs[member_offset] = Dict{String,Any}(
                    "kind" => "member",
                    "name" => nothing,
                    "type" => nothing,
                    "offset" => nothing,
                    "parent" => parent_ctx(die_depth(line))  # Track which struct/class this belongs to
                )
                offset_to_kind[member_offset] = :member
                type_refs["_pending_member_offset"] = member_offset
            end
        end

        # Extract member attributes
        if haskey(type_refs, "last_tag_offset")
            tag_offset = type_refs["last_tag_offset"]
            if haskey(offset_to_kind, tag_offset) && offset_to_kind[tag_offset] == :member
                member_info = type_refs[tag_offset]

                # Member name
                if contains(line, "DW_AT_name")
                    name_match = match(r":\s*([^:]+)\s*$", line)
                    if !isnothing(name_match)
                        member_info["name"] = String(strip(name_match.captures[1]))
                    end
                end

                # Member type reference
                if contains(line, "DW_AT_type")
                    type_match = match(r"<(0x[^>]+)>", line)
                    if !isnothing(type_match)
                        member_info["type"] = String(type_match.captures[1])
                    end
                end

                # Member byte offset within struct
                if contains(line, "DW_AT_data_member_location")
                    # Match explicit attribute name and value
                    offset_match = match(r"DW_AT_data_member_location\s*:\s*(0x[0-9a-fA-F]+|\d+)", line)
                    if !isnothing(offset_match)
                        val_str = offset_match.captures[1]
                        # Normalize to hex string if decimal
                        if !startswith(val_str, "0x")
                            val = parse(Int, val_str)
                            member_info["offset"] = "0x" * string(val, base=16)
                        else
                            member_info["offset"] = val_str
                        end
                    end
                end

                # Bitfield: DW_AT_bit_size (number of bits in bitfield member)
                if contains(line, "DW_AT_bit_size")
                    bit_size_match = match(r"DW_AT_bit_size\s*:\s*(\d+)", line)
                    if !isnothing(bit_size_match)
                        member_info["bit_size"] = parse(Int, bit_size_match.captures[1])
                    end
                end

                # Bitfield: DW_AT_data_bit_offset (DWARF 4+ — absolute bit offset from struct start)
                if contains(line, "DW_AT_data_bit_offset")
                    # Hex past 65535 (a bitfield 8 KiB into its record), like any
                    # constant readelf prints from four bytes (`_dwarf_count`).
                    dbo = _dwarf_count(line)
                    isnothing(dbo) || (member_info["data_bit_offset"] = dbo)
                end

                # Bitfield: DW_AT_bit_offset (DWARF 2/3 — offset from MSB of containing unit)
                if contains(line, "DW_AT_bit_offset") && !contains(line, "DW_AT_data_bit_offset")
                    bo_match = match(r"DW_AT_bit_offset\s*:\s*(\d+)", line)
                    if !isnothing(bo_match)
                        member_info["bit_offset_legacy"] = parse(Int, bo_match.captures[1])
                    end
                end

                # Member addition to parent is deferred to flush_pending_member()
                # which runs at the next DW_TAG_* boundary to ensure all attributes are collected.
            end

            # Extract enumerator attributes and add to parent enum
            if haskey(offset_to_kind, tag_offset) && offset_to_kind[tag_offset] == :enumerator
                enumerator_info = type_refs[tag_offset]

                # When we have all info, add to parent enum
                parent_offset = get(enumerator_info, "parent", nothing)
                if !isnothing(parent_offset) && haskey(type_refs, parent_offset) &&
                   isa(type_refs[parent_offset], Dict) && haskey(type_refs[parent_offset], "enumerators")
                    if !isnothing(enumerator_info["name"]) && !isnothing(enumerator_info["value"])
                        # Only add if we haven't already added this enumerator
                        existing_names = [e["name"] for e in type_refs[parent_offset]["enumerators"]]
                        if !(enumerator_info["name"] in existing_names)
                            push!(type_refs[parent_offset]["enumerators"], Dict(
                                "name" => enumerator_info["name"],
                                "value" => enumerator_info["value"]
                            ))
                        end
                    end
                end
            end
        end
    end

    # Flush any remaining pending member after the last line
    flush_pending_member()

    # Debug: Show type refs collected
    base_count = count(k -> haskey(offset_to_kind, k) && offset_to_kind[k] == :base && isa(type_refs[k], String), keys(type_refs))
    pointer_count = count(k -> haskey(offset_to_kind, k) && offset_to_kind[k] == :pointer, keys(type_refs))
    struct_count = count(k -> haskey(offset_to_kind, k) && offset_to_kind[k] == :struct && isa(type_refs[k], Dict), keys(type_refs))
    class_count = count(k -> haskey(offset_to_kind, k) && offset_to_kind[k] == :class && isa(type_refs[k], Dict), keys(type_refs))
    enum_count = count(k -> haskey(offset_to_kind, k) && offset_to_kind[k] == :enum && isa(type_refs[k], Dict), keys(type_refs))
    array_count = count(k -> haskey(offset_to_kind, k) && offset_to_kind[k] == :array && isa(type_refs[k], Dict), keys(type_refs))
    subroutine_count = count(k -> haskey(offset_to_kind, k) && offset_to_kind[k] == :subroutine && isa(type_refs[k], Dict), keys(type_refs))

    # Count total members/enumerators extracted
    total_members = 0
    total_enumerators = 0
    for (k, v) in type_refs
        if isa(v, Dict)
            if haskey(v, "members")
                total_members += length(v["members"])
            end
            if haskey(v, "enumerators")
                total_enumerators += length(v["enumerators"])
            end
        end
    end

    # Every pass below that can let one DIE win over another walks the DIEs in
    # dump order, never in `type_refs`' own order: that is the hash order of the
    # offset strings, which changes with the Julia release and with every shift
    # of the offsets. See `_die_order`.
    pass1_order = _die_order(type_refs)

    # Propagate typedef names to anonymous structs, unions, and enums
    # C-style code often defines structs as: typedef struct { ... } Name;
    # `typedef struct { … } A, B;` gives the struct the first name declared.
    for offset in pass1_order
        type_info = type_refs[offset]
        if isa(type_info, Dict) && get(type_info, "kind", "") == "typedef"
            typedef_name = get(type_info, "name", "")
            target_ref = get(type_info, "target", nothing)
            if !isempty(typedef_name) && !isnothing(target_ref) && haskey(type_refs, target_ref)
                target_info = type_refs[target_ref]
                if isa(target_info, Dict)
                    target_kind = get(target_info, "kind", "")
                    target_name = get(target_info, "name", "")
                    if target_kind in ["struct", "class", "union", "enum"] && 
                       (isempty(target_name) || startswith(target_name, "unknown_"))
                        target_info["name"] = typedef_name
                    end
                end
            end
        end
    end

    # Name anonymous aggregates after the member that embeds them.
    #
    # Both `union { ... } u;` and a C11 anonymous `union { ... };` produce an
    # aggregate DIE with no DW_AT_name. Such a DIE was dropped outright (the
    # struct_defs export filters `unknown_union`/`unknown_struct`) and the
    # member embedding it resolved to "unknown"/Any, so the WHOLE enclosing
    # struct degraded to an opaque byte blob even though DWARF carries the
    # complete member tree — toml_datum_t came out as 40 bytes with zero named
    # fields (2026-08-01). Naming them `<parent>_<member>` turns them into
    # ordinary named types the rest of the pipeline already knows how to carry.
    #
    # A nameless C11 member also gets a synthetic FIELD name (`_anonN`) so it
    # survives `flush_pending_member`, and is flagged `anonymous_member` — its
    # members are injected into the parent's namespace by C, which is what lets
    # the generator forward `parent.x` to the embedded aggregate.
    #
    # Parent-first and iterated to a fixed point, so a nested anonymous
    # aggregate inherits its parent's already-synthesized name.
    _anon_agg_names = Set(["", "unknown", "unknown_struct", "unknown_class", "unknown_union"])
    _named_ok(n) = isa(n, String) && !(n in _anon_agg_names)
    _is_anon_agg(ti) = isa(ti, Dict) &&
        get(ti, "kind", "") in ["struct", "class", "union"] &&
        !_named_ok(get(ti, "name", ""))

    taken_type_names = Set{String}()
    for (_, ti) in type_refs
        if isa(ti, Dict) && get(ti, "kind", "") in ["struct", "class", "union", "enum"]
            n = get(ti, "name", "")
            _named_ok(n) && push!(taken_type_names, n)
        end
    end
    # Synthetic names must already be in the form the C generator's
    # `_sanitize_c_type_name` would produce — it collapses `_+` to `_`, so a
    # raw `Parent` + `_` + `_anon1` would be REWRITTEN to `Parent_anon1` at
    # emission while the dependency graph still keyed on the unsanitized
    # spelling. The edge then missed, the topological sort emitted the parent
    # BEFORE the type it embeds, and the parent fell back to a byte blob.
    function synth_type_name(parent::AbstractString, suffix::AbstractString)
        s = replace(string(parent) * "_" * string(suffix), r"[^A-Za-z0-9_]" => "_")
        s = replace(s, r"_+" => "_")
        return String(rstrip(s, '_'))
    end
    function claim_type_name!(base::String)
        cand = base
        i = 2
        while cand in taken_type_names
            cand = "$(base)_$(i)"
            i += 1
        end
        push!(taken_type_names, cand)
        return cand
    end

    # Every CU that includes the header declares the same anonymous aggregate
    # again, and each gets its own DIE — mpack_tag_t's union appeared in six,
    # which without this minted mpack_tag_t_v through _v_6, five of them dead
    # types in the wrapper. Structurally identical aggregates under the same
    # parent name are the same C type, so they share one name.
    function agg_signature(ti)
        parts = String[string(get(ti, "kind", ""), "|", get(ti, "byte_size", "?"))]
        for m in get(ti, "members", [])
            push!(parts, string(get(m, "name", "?"), "@", get(m, "offset", "?"),
                                ":", get(m, "bit_size", "")))
        end
        return join(parts, ",")
    end
    claimed_by_sig = Dict{String,String}()

    for _round in 1:16
        progressed = false
        # Dump order: the synthetic names and their collision suffixes depend
        # on which parent claims a name first. Sorting the offset STRINGS was
        # stable for one dump, but not across an upstream edit: "0xff00" sorts
        # after "0x10000", so an offset gaining a digit reordered two parents.
        for parent_off in pass1_order
            ti = type_refs[parent_off]
            (isa(ti, Dict) && haskey(ti, "members")) || continue
            pname = get(ti, "name", "")
            _named_ok(pname) || continue          # parent still anonymous — wait a round
            anon_idx = 0
            for m in ti["members"]
                mname = get(m, "name", nothing)
                isnothing(mname) && (anon_idx += 1)
                tref = get(m, "type", nothing)
                isnothing(tref) && continue
                child = get(type_refs, tref, nothing)
                _is_anon_agg(child) || continue
                # Field name vs TYPE name suffix: a C11 anonymous member gets the
                # field `_anonN` but the type `Parent_anonN` — one leading
                # underscore, because the sanitizer would collapse two anyway.
                suffix = if isnothing(mname)
                    mname = "_anon$(anon_idx)"
                    m["name"] = mname
                    m["anonymous_member"] = true
                    "anon$(anon_idx)"
                else
                    string(mname)
                end
                base = synth_type_name(pname, suffix)
                sigkey = base * "\0" * agg_signature(child)
                child["name"] = get!(claimed_by_sig, sigkey) do
                    claim_type_name!(base)
                end
                child["anonymous"] = true
                # The type is reachable only through this parent, so the
                # parent's provenance is its own (see `drops_type`).
                child["anon_parent"] = parent_off
                progressed = true
            end
        end
        progressed || break
    end

    # Members that stayed nameless are not anonymous aggregates (unnamed
    # bitfield padding, `int : 3;`). They were dropped before this pass existed
    # and are dropped here — a member with no name and no embedded type has
    # nothing to expose.
    for (_, ti) in type_refs
        if isa(ti, Dict) && haskey(ti, "members")
            filter!(m -> !isnothing(get(m, "name", nothing)), ti["members"])
        end
    end

    # Types collected from DWARF

    # Collect all struct/class/enum names for type resolution
    struct_names = Set{String}()
    enum_names = Set{String}()

    for (offset, type_info) in type_refs
        if isa(type_info, Dict)
            kind = get(type_info, "kind", "")
            name = get(type_info, "name", "")

            # Unions belong here too: a union-typed member used to resolve to
            # "unknown" and land in the wrapper as `Any`, dropping it from the
            # enclosing struct entirely.
            if kind in ["struct", "class", "union"] && _named_ok(name)
                push!(struct_names, name)
            elseif kind == "enum" && !isempty(name) && name != "unknown" && name != "unknown_enum"
                push!(enum_names, name)
            end
        end
    end

    # Helper function to resolve type references (follows pointer/const/reference chains)
    function resolve_type(type_ref::String, type_refs::Dict, visited::Set{String}=Set{String}())::String
        # Avoid infinite loops
        if type_ref in visited
            return "unknown"
        end
        push!(visited, type_ref)

        if !haskey(type_refs, type_ref)
            # Type not found in our table - likely a forward reference or external type
            return "unknown"
        end

        type_info = type_refs[type_ref]

        # Base type - just return the name
        if isa(type_info, String)
            return type_info
        end

        # Pointer/const/reference/enum/array/subroutine type - follow the chain
        if isa(type_info, Dict)
            kind = get(type_info, "kind", nothing)

            # Struct, class or union - return the name. Unions were missing
            # here, so EVERY union-typed member (named or anonymous) resolved
            # to "unknown" and was typed `Any` by the caller.
            if kind in ["struct", "class", "union"]
                return get(type_info, "name", "unknown")
            end

            # Enum - return the name
            if kind == "enum"
                return get(type_info, "name", "unknown_enum")
            end

            # Array - return element_type[size] or element_type[size1][size2]...
            if kind == "array"
                element_type_ref = get(type_info, "element_type", nothing)
                dimensions = get(type_info, "dimensions", [])

                if !isnothing(element_type_ref)
                    element_type = resolve_type(element_type_ref, type_refs, visited)
                    if !isempty(dimensions)
                        # Build array notation: type[size1][size2]...
                        dim_str = join(["[$d]" for d in dimensions], "")
                        return element_type * dim_str
                    else
                        # No dimensions specified (incomplete DWARF)
                        return element_type * "[]"
                    end
                end
                return "unknown[]"
            end

            # Subroutine (function pointer) - return detailed signature with parameters
            if kind == "subroutine"
                return_type_ref = get(type_info, "return_type", nothing)
                param_refs = get(type_info, "parameters", [])

                # Each branch gets its own copy of `visited`, which guards one
                # chain against cycles: shared, the second `int` of `int (*)(int,
                # int)` was a cycle, and resolved to "unknown".
                ret_type_str = if !isnothing(return_type_ref)
                    resolve_type(return_type_ref, type_refs, copy(visited))
                else
                    "void"
                end

                # Parameter type refs, in dump order (collected in pass 1)
                param_type_strs = String[]
                for param_ref in param_refs
                    push!(param_type_strs, resolve_type(param_ref, type_refs, copy(visited)))
                end
                get(type_info, "variadic", false) == true && push!(param_type_strs, "...")

                # Build signature: function_ptr(return_type; param1, param2, ...)
                # Using semicolon to separate return from params for easier parsing
                if isempty(param_type_strs)
                    return "function_ptr($ret_type_str)"
                else
                    param_list = join(param_type_strs, ", ")
                    return "function_ptr($ret_type_str; $param_list)"
                end
            end

            # Typedef - special handling for well-known types
            if kind == "typedef"
                typedef_name = get(type_info, "name", "")

                # Preserve well-known typedefs that have portable Julia mappings
                # These should NOT be resolved to their underlying type
                well_known_typedefs = [
                    "size_t", "ssize_t", "ptrdiff_t",
                    "intptr_t", "uintptr_t",
                    "int8_t", "int16_t", "int32_t", "int64_t",
                    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
                    "wchar_t", "char16_t", "char32_t"
                ]

                if typedef_name in well_known_typedefs
                    # Preserve the typedef name for portable mapping
                    return typedef_name
                end

                # For other typedefs, resolve to underlying type
                target_ref = get(type_info, "target", nothing)
                if !isnothing(target_ref)
                    resolved_target = resolve_type(target_ref, type_refs, visited)
                    if resolved_target == "unknown_struct" && !isempty(typedef_name)
                        # If the target is an anonymous struct, use the typedef name instead
                        return typedef_name
                    end
                    return resolved_target
                else
                    # If no target, return the typedef name itself
                    return typedef_name == "" ? "unknown" : typedef_name
                end
            end

            target = get(type_info, "target", nothing)

            # restrict / volatile / _Atomic: transparent. The qualifier carries no
            # ABI information, and spelling it into the type string would hand every
            # downstream mapper a prefix only some of them strip. A qualifier DIE
            # with no DW_AT_type qualifies `void` (`volatile void *`), exactly as a
            # pointer DIE with none points at it.
            if kind in ("restrict", "volatile", "atomic")
                return isnothing(target) ? "void" : resolve_type(target, type_refs, visited)
            end

            if isnothing(target)
                # A DIE with no DW_AT_type points at, or qualifies, `void`. clang
                # writes `const void *` as a pointer to a const_type with no type,
                # so that const is `const void`, spelled like every other `const T`
                # (the mappers strip the qualifier and see `void*`). llama.cpp's
                # tables had 1257 `unknown*` parameters from this one case. A
                # reference to nothing cannot be written in C or C++.
                kind == "pointer" && return "void*"
                kind == "const" && return "const void"
                return "unknown"
            end

            # Recursively resolve the target type
            target_type = resolve_type(target, type_refs, visited)

            # Build the full type string
            if kind == "pointer"
                return target_type * "*"
            elseif kind == "const"
                return "const " * target_type
            elseif kind == "reference"
                return target_type * "&"
            elseif kind == "rvalue_reference"
                return target_type * "&&"
            end
        end

        return "unknown"
    end

    # Second pass: Extract function return types and parameters
    current_function_offset = nothing
    current_function_name = nothing
    current_function_linkage = nothing
    current_function_level = nothing
    function_processed = false  # State flag: true once we leave function's own level
    params_for_this_function = []  # Collect parameters for current function

    # Track formal parameters (similar to struct members)
    current_param_offset = nothing
    current_param_level = nothing  # depth of the open DW_TAG_formal_parameter DIE

    # Handle to the parameter dict most recently pushed, live only for the
    # remainder of that parameter's own DIE. A parameter is pushed as soon as its
    # DW_AT_type is seen, but clang emits DW_AT_artificial AFTER the type, so the
    # receiver flag arrives once the parameter context has already been closed —
    # and it must stay closed, because an unnamed parameter holding its context
    # open past its own terminator is exactly the phantom-parameter leak fixed
    # 2026-07-26. Cleared on every DW_TAG_ line so it can never reach another DIE.
    pending_param_dict = nothing

    # DIE-tree arity witness: how many DW_TAG_formal_parameter children the
    # subprogram DIE actually has, counted straight off the readelf tree depth
    # and independent of whether name/type extraction succeeded for each one.
    # Compared against the extracted list by check_param_arity! below — a
    # disagreement means the emitted ccall signature does not match the DIE
    # tree, which is a silent wrong-call bug, so it fails loudly.
    formal_params_in_die = 0
    die_param_counts = Dict{String,Int}()  # function key => DIE-tree arity

    # `DW_AT_object_pointer` exists only on a definition DIE of an instance
    # method. A declaration of the same method does not carry it, so absence
    # there must NOT be read as "static" — that would strip `this` from every
    # out-of-line method. A definition with code and no object pointer is a
    # static member (or a free function). Recorded only from such a definition.
    current_fn_is_declaration = false
    current_fn_has_code = false
    current_fn_saw_object_pointer = false
    # `DW_AT_inline` marks the abstract definition of an inline method. It has
    # the signature and (for an instance method) `DW_AT_object_pointer`, but no
    # `low_pc` — the concrete copy carries the code and only
    # `DW_AT_abstract_origin`. Recording "no object pointer" from that concrete
    # copy would mark every inline instance method static.
    current_fn_is_inline = false
    current_fn_is_concrete = false
    # A declaration DIE's own facts, read by the inline-constructor plan
    # (`_inline_constructor_plan`): `DW_AT_artificial` (an implicit member),
    # `DW_AT_deleted`, and `DW_AT_accessibility` as written; absent, the
    # record's default applies (private in a `class`, public in a `struct`),
    # which `_record_declared_constructors!` fills in from the record's kind.
    current_fn_is_artificial = false
    current_fn_is_deleted = false
    current_fn_accessibility = nothing
    current_fn_is_virtual = false
    declaration_facts = Dict{String,Dict{String,Any}}()
    # The DIE's own `DW_AT_linkage_name`, before `DW_AT_specification` replaces
    # it with the declaration's. Clang gives each emitted constructor and
    # destructor VARIANT a definition DIE of its own, named for the variant
    # (`C1`, `C2`, `D0`, `D1`, `D2`) and specifying the one declaration, whose
    # name is the unified `C4`/`D4`. The base-object variants (`C2`, `D2`) of a
    # class with a virtual base take one parameter more than the declaration
    # has, the VTT, right after `this`; no other variant does.
    current_function_own_linkage = nothing
    # variant symbol => (key, params, count), and declaration key => its
    # parameter count: read after the walk, where a variant with one parameter
    # more than its declaration gets an entry of its own and the unified key
    # keeps the declared list.
    variant_definitions = Dict{String,Any}()
    declared_param_counts = Dict{String,Int}()
    # Keys whose parameters came from a DEFINITION (named, `this` included). A
    # declaration of the same key that is read afterwards (a later unit's, or
    # the class's own when it follows the definition) has unnamed parameters
    # and must not replace them.
    definition_params = Set{String}()

    # subprogram DIE offset => its DW_AT_linkage_name, so a definition DIE can be
    # resolved back to the declaration that owns the mangled name via
    # DW_AT_specification (see the handler below). It is filled before the
    # walk (below): a declaration usually precedes its out-of-line definition
    # in a unit, but not always.
    linkage_by_offset = Dict{String,String}()

    # NEW: Track global variables
    current_variable_offset = nothing

    last_seen_level = nothing  # Track last level for attribute lines

    # Attach collected parameters to the function entry and close the function
    # context. Runs when the subprogram DIE closes (a sibling or shallower DIE
    # arrives, or the dump ends) rather than whenever the next subprogram
    # happens to appear, so a parameter list can never be mutated after it has
    # been recorded.
    function finalize_current_function!()
        isnothing(current_function_offset) && return
        function_key = !isnothing(current_function_linkage) ? current_function_linkage : current_function_name

        if !isnothing(function_key)
            # Snapshot: `params_for_this_function` is rebound per function, but
            # storing the live array made every later push retro-mutate an
            # already-finished function (that is how a struct member landed on
            # free_opaque as a phantom second parameter).
            params = copy(params_for_this_function)

            # What the post-pass needs to tell a base-object variant's VTT
            # from a declared parameter (`variant_definitions`): how many
            # parameters the declaration has, and each variant's own list.
            if current_fn_is_declaration
                declared_param_counts[function_key] = formal_params_in_die
            elseif current_function_own_linkage !== nothing && current_function_own_linkage != function_key
                variant_definitions[current_function_own_linkage] =
                    (key = function_key, params = params, count = formal_params_in_die)
            end

            die_param_counts[function_key] = max(get(die_param_counts, function_key, 0), formal_params_in_die)

            if !haskey(return_types, function_key)
                # No DW_AT_type at the function's own level = void return
                prev_noexcept = false
                if haskey(type_refs, current_function_offset) && isa(type_refs[current_function_offset], Dict)
                    prev_noexcept = get(type_refs[current_function_offset], "is_noexcept", false)
                end
                return_types[function_key] = Dict(
                    "c_type" => "void",
                    "julia_type" => "Cvoid",
                    "size" => 0,
                    "is_noexcept" => prev_noexcept,
                    "parameters" => params
                )
            elseif (!isempty(params) || isempty(get(return_types[function_key], "parameters", []))) &&
                   !(current_fn_is_declaration && function_key in definition_params)
                # A declaration-only DIE (unnamed parameters, nothing extracted)
                # must not clobber parameters already recovered from the
                # definition DIE, whichever of the two is read first.
                return_types[function_key]["parameters"] = params
            end
            (!current_fn_is_declaration && !current_fn_is_concrete && !isempty(params)) &&
                push!(definition_params, function_key)

            # Definition only. A later declaration of the same key must not
            # erase the answer, and a declaration must not invent `false`.
            # An inline method's abstract DIE (`DW_AT_inline`, no low_pc) is
            # that definition; the concrete `DW_AT_abstract_origin` copy is not.
            if !current_fn_is_declaration && !current_fn_is_concrete &&
               (current_fn_has_code || current_fn_saw_object_pointer || current_fn_is_inline)
                return_types[function_key]["has_object_pointer"] = current_fn_saw_object_pointer
            end

            # The in-class declaration's facts (one per key; a class declares a
            # member once). A definition DIE says nothing here.
            if current_fn_is_declaration
                declaration_facts[function_key] = Dict{String,Any}(
                    "artificial" => current_fn_is_artificial,
                    "deleted" => current_fn_is_deleted,
                    "accessibility" => current_fn_accessibility,
                    "virtual" => current_fn_is_virtual)
            end
        end

        current_function_offset = nothing
        current_fn_is_declaration = false
        current_fn_has_code = false
        current_fn_saw_object_pointer = false
        current_fn_is_inline = false
        current_fn_is_concrete = false
        current_fn_is_artificial = false
        current_fn_is_deleted = false
        current_fn_accessibility = nothing
        current_fn_is_virtual = false
        current_function_name = nothing
        current_function_linkage = nothing
        current_function_own_linkage = nothing
        current_function_level = nothing
        function_processed = false
        params_for_this_function = []
        formal_params_in_die = 0
    end

    # The DIE open at each depth, as (offset, tag): a subprogram's parent is the
    # one a level up. Read for one fact, the record a destructor is declared in
    # (`destructor_of`, keyed by the record's DIE offset), which the record loop
    # below files under the record's name for `record_abi` (`string_layout`).
    die_at_depth = Dict{Int,Tuple{String,String}}()
    current_function_parent = nothing
    destructor_of = Dict{String,String}()
    # ...and the constructors declared in it (`constructors_of`, every `C4`/`C1`/
    # `C2` linkage), for the one `std::string` building from a `const char*`
    # and a length (`_string_constructor`, 2026-11-15).
    constructors_of = Dict{String,Vector{String}}()
    # ...and every member function declared in it (`methods_of`), for the
    # inline-method plan (`_inline_method_plan`), which tells a constructor or
    # destructor from a method by the demangled name, not here.
    methods_of = Dict{String,Vector{String}}()

    # Every subprogram's linkage name by DIE offset, BEFORE the walk: a
    # definition names its declaration by offset (`DW_AT_specification`), and
    # the declaration can come later in the unit. Clang 22 put a class's DIE
    # after the out-of-line definitions of its members for one fixture, so
    # each definition's reference pointed forward, the definition was dropped
    # whole, and every member of the class kept the declaration's facts
    # alone: no parameter names, and a static member wrapped with a receiver
    # (CI's LLVM 22 cells, 2026-11-15; clang 18 and 23 emit the class first).
    let sub = nothing
        for line in _dump_lines(output)
            h = match(r"^\s*<\d+><([0-9a-fA-F]+)>: Abbrev Number", line)
            if h !== nothing
                sub = occursin("DW_TAG_subprogram", line) ? "0x" * h.captures[1] : nothing
            elseif sub !== nothing && occursin("DW_AT_linkage_name", line)
                m = match(r":\s*([^:\s]+)\s*$", line)
                m === nothing || (linkage_by_offset[sub] = String(strip(m.captures[1])))
            end
        end
    end

    for line in _dump_lines(output)
        line = strip(line)

        # Extract nesting level from DWARF format: <level><offset>
        # Tags have: <level><offset>: ...
        # Attributes have: <offset>   DW_AT_...
        level_match = match(r"^\s*<(\d+)><", line)
        if !isnothing(level_match)
            last_seen_level = parse(Int, level_match.captures[1])
            dm = match(r"^<(\d+)><([0-9a-fA-F]+)>: Abbrev Number: \d+ \((DW_TAG_\w+)\)", line)
            if dm !== nothing
                d = parse(Int, dm.captures[1])
                die_at_depth[d] = ("0x" * dm.captures[2], String(dm.captures[3]))
                for k in collect(keys(die_at_depth))
                    k > d && delete!(die_at_depth, k)
                end
            end
        end
        # Use last seen level for attributes (which don't have level prefix)
        current_level = last_seen_level

        # Close DIE contexts this line has left behind.
        #
        # readelf prints a depth on every DIE header, including the
        # `Abbrev Number: 0` sibling-list terminators, so a header at depth L
        # ends every DIE at depth >= L. Without this the flat context scalars
        # stayed live long after their DIE closed: `current_param_offset`
        # survived the end of its DW_TAG_formal_parameter and vacuumed up the
        # DW_AT_name/DW_AT_type of an unrelated DIE later in the CU. In
        # c_abomination the unnamed parameter of the `InnerFunc`
        # DW_TAG_subroutine_type stayed open across the struct that follows
        # it, swallowed `SelfReferential`'s first member, and pushed
        # ("next", SelfReferential*) onto the still-aliased parameter array of
        # the last subprogram — free_opaque got a phantom second argument and
        # a two-argument ccall. Same shape as the nested-type member
        # mis-attribution fixed in pass 1 (context_by_depth above).
        if !isnothing(level_match)
            die_level = last_seen_level
            if !isnothing(current_param_level) && die_level <= current_param_level
                current_param_offset = nothing
                current_param_level = nothing
            end
            if !isnothing(current_function_level) && die_level <= current_function_level
                finalize_current_function!()
            end
        end

        # Generic context reset for major tags to prevent leakage
        if contains(line, "DW_TAG_") && !contains(line, "DW_TAG_variable") && 
           !contains(line, "DW_TAG_subprogram") && !contains(line, "DW_TAG_formal_parameter") &&
           !contains(line, "DW_TAG_unspecified_parameters")
            # If we hit a tag we don't explicitly handle as a "child" container in this pass
            # we should reset the variable context.
            # Base types, const types, etc. are handled in Pass 1, but we see them here too.
            current_variable_offset = nothing
        end

        # Detect when we've left the function's own level (entered child tags).
        # The function's own attributes are done, so a missing DW_AT_type means
        # a void return. Parameters are attached by finalize_current_function!
        # when the DIE closes, not here.
        if !isnothing(current_function_offset) && !isnothing(current_level) && !isnothing(current_function_level)
            if current_level > current_function_level && !function_processed
                function_key = !isnothing(current_function_linkage) ? current_function_linkage : current_function_name
                if !isnothing(function_key) && !haskey(return_types, function_key)
                    return_types[function_key] = Dict(
                        "c_type" => "void",
                        "julia_type" => "Cvoid",
                        "size" => 0,
                        "parameters" => []
                    )
                end
                function_processed = true
            end
        end

        # A new DIE ends the previous parameter's attribute run — see
        # `pending_param_dict`. Must precede every DW_TAG_ handler below.
        if contains(line, "DW_TAG_")
            pending_param_dict = nothing
        elseif contains(line, "DW_AT_artificial") && !isnothing(pending_param_dict)
            # DW_AT_artificial seen AFTER DW_AT_type (clang's order): the parameter
            # is already pushed, so patch it. Only a position-0 parameter still
            # carrying its synthesized name is renamed — a programmer-named
            # parameter keeps its name, and a non-receiver artificial parameter
            # (e.g. a lambda's captured-frame slot) is never at position 0 of a
            # method, so it keeps argN.
            if get(pending_param_dict, "position", -1) == 0 &&
               startswith(String(get(pending_param_dict, "name", "")), "arg")
                pending_param_dict["name"] = "this"
            end
            pending_param_dict = nothing
        end

        # Detect function start (DW_TAG_subprogram)
        if contains(line, "DW_TAG_subprogram")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                # A subprogram nested inside another subprogram (local class,
                # lambda body) does not trip the depth-close above, so flush the
                # enclosing function here rather than dropping its parameters.
                finalize_current_function!()

                current_function_offset = "0x" * offset_match.captures[1]
                current_function_level = current_level
                current_function_parent = current_level === nothing ? nothing :
                                          get(die_at_depth, current_level - 1, nothing)
                current_function_name = nothing
                current_function_linkage = nothing
                current_function_own_linkage = nothing
                function_processed = false  # Reset flag for new function
                params_for_this_function = []  # Reset for the new function
                formal_params_in_die = 0
                current_fn_is_declaration = false
                current_fn_has_code = false
                current_fn_saw_object_pointer = false
                current_fn_is_inline = false
                current_fn_is_concrete = false
                current_fn_is_artificial = false
                current_fn_is_deleted = false
                current_fn_accessibility = nothing
                current_fn_is_virtual = false

                # RESET VARIABLE CONTEXT to prevent leakage
                current_variable_offset = nothing
            end
        end
        
        # NEW: Detect global variable (DW_TAG_variable)
        if contains(line, "DW_TAG_variable")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_variable_offset = "0x" * offset_match.captures[1]
                
                type_refs[current_variable_offset] = Dict{String,Any}(
                    "kind" => "variable",
                    "name" => nothing,
                    "type" => nothing,
                    "external" => false,
                    "declaration" => false
                )
                offset_to_kind[current_variable_offset] = :variable

                # Ensure we don't leak param context
                current_param_offset = nothing
                current_param_level = nothing
            end
        end
        
        # NEW: Detect varargs (DW_TAG_unspecified_parameters)
        if contains(line, "DW_TAG_unspecified_parameters")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                # If we are inside a function context, mark it as varargs
                if !isnothing(current_function_offset) && !isnothing(current_level) &&
                   !isnothing(current_function_level) && current_level > current_function_level
                   
                   # Update type_refs
                   if !haskey(type_refs, current_function_offset)
                       type_refs[current_function_offset] = Dict{String,Any}()
                   end
                   if isa(type_refs[current_function_offset], Dict)
                       type_refs[current_function_offset]["is_vararg"] = true
                   end
                   
                   # Update return_types if function is already registered
                   function_key = !isnothing(current_function_linkage) ? current_function_linkage : current_function_name
                   if !isnothing(function_key) && haskey(return_types, function_key)
                       return_types[function_key]["is_vararg"] = true
                   end
                end
            end
        end

        # Detect formal parameter. Only a subprogram's are read here; a
        # DW_TAG_subroutine_type's were collected in pass 1.
        # Example: <2><7a>: DW_TAG_formal_parameter
        if contains(line, "DW_TAG_formal_parameter")
            offset_match = match(r"<\d+><([^>]+)>", line)
            if !isnothing(offset_match)
                current_param_offset = "0x" * offset_match.captures[1]
                current_param_level = current_level

                # Witness the DIE-tree arity. Only direct children of the
                # subprogram DIE are its parameters — a DW_TAG_subroutine_type's
                # own parameters live outside the function subtree entirely.
                if !isnothing(current_function_level) && !isnothing(current_level) &&
                   current_level == current_function_level + 1
                    formal_params_in_die += 1
                end

                # RESET VARIABLE CONTEXT
                current_variable_offset = nothing

                type_refs[current_param_offset] = Dict{String,Any}(
                    "kind" => "parameter",
                    "name" => nothing,
                    "type" => nothing,
                    "position" => length(params_for_this_function),
                    # DW_AT_artificial marks a parameter the compiler synthesized
                    # rather than the programmer writing it — for a member function
                    # that is the implicit `this`. It is the only way to recognise a
                    # receiver on a DECLARATION DIE, where the parameter has a type
                    # but no name, so it decides the synthesized name below.
                    "artificial" => false
                )
                offset_to_kind[current_param_offset] = :parameter
            end
        end
        
        # Process Global Variable Attributes
        if !isnothing(current_variable_offset) && haskey(type_refs, current_variable_offset) &&
           haskey(offset_to_kind, current_variable_offset) && offset_to_kind[current_variable_offset] == :variable
           
           var_info = type_refs[current_variable_offset]
           
           if contains(line, "DW_AT_name")
               name_match = match(r":\s*([^:]+)\s*$", line)
               if !isnothing(name_match)
                   var_info["name"] = String(strip(name_match.captures[1]))
               end
           end
           
           if contains(line, "DW_AT_type")
               type_match = match(r"<(0x[^>]+)>", line)
               if !isnothing(type_match)
                   var_info["type"] = String(type_match.captures[1])
               end
           end
           
           if contains(line, "DW_AT_external")
               # Usually DW_AT_external : 1
               var_info["external"] = true
           end
           
           if contains(line, "DW_AT_declaration")
               var_info["declaration"] = true
           end
        end

        # Only process attributes at the function level (not in child tags like parameters)
        in_function_context = !isnothing(current_function_offset) &&
                             !isnothing(current_level) &&
                             !isnothing(current_function_level) &&
                             current_level == current_function_level

        # Extract function name (for C functions without mangling)
        # Example: <2f>   DW_AT_name        : (indexed string: 0xc): test_sin
        if contains(line, "DW_AT_name") && in_function_context && isnothing(current_function_name)
            name_match_quotes = match(r"DW_AT_name\s*\(\"([^\"]+)\"\)", line)
            if !isnothing(name_match_quotes)
                current_function_name = String(name_match_quotes.captures[1])
            else
                name_match = match(r"DW_AT_name\s*:\s*(?:\([^)]*\):\s*)?(.*)$", line)
                if !isnothing(name_match)
                    current_function_name = String(strip(name_match.captures[1]))
                end
            end
        end

        # Extract linkage name (mangled name for C++ functions)
        # Example: <5c>   DW_AT_linkage_name: (indexed string: 0x8): _ZN10Calculator5powerEdd
        # Definition vs declaration, for the object-pointer fact below.
        # `low_pc` / `ranges` / `high_pc` mean this DIE has a body. Clang puts
        # `DW_AT_object_pointer` on that body for an instance method and omits
        # it for a static member.
        if in_function_context && contains(line, "DW_AT_declaration")
            current_fn_is_declaration = true
        end
        # The subprogram's OWN attributes only: a formal parameter's lines sit a
        # level deeper, so `this`'s `DW_AT_artificial` is not the function's.
        if in_function_context && contains(line, "DW_AT_artificial")
            current_fn_is_artificial = true
        end
        if in_function_context && contains(line, "DW_AT_deleted")
            current_fn_is_deleted = true
        end
        if in_function_context && contains(line, "DW_AT_accessibility")
            # `DW_AT_accessibility: 1	(public)`: 1 public, 2 protected, 3 private.
            am = match(r"DW_AT_accessibility\s*:\s*(\d+)", line)
            if am !== nothing
                current_fn_accessibility = get(("public", "protected", "private"),
                                               parse(Int, am.captures[1]), nothing)
            end
        end
        if in_function_context && contains(line, "DW_AT_object_pointer")
            current_fn_saw_object_pointer = true
        end
        if in_function_context && (contains(line, "DW_AT_low_pc") ||
                                   contains(line, "DW_AT_high_pc") ||
                                   contains(line, "DW_AT_ranges"))
            current_fn_has_code = true
        end
        if in_function_context && contains(line, "DW_AT_inline")
            current_fn_is_inline = true
        end
        if in_function_context && contains(line, "DW_AT_abstract_origin")
            current_fn_is_concrete = true
        end

        if contains(line, "DW_AT_linkage_name") && in_function_context
            # Extract just the mangled name after the last colon
            linkage_match = match(r":\s*([^:\s]+)\s*$", line)
            if !isnothing(linkage_match)
                current_function_linkage = String(strip(linkage_match.captures[1]))  # Convert SubString to String
                current_function_own_linkage = current_function_linkage
                if !isnothing(current_function_offset)
                    linkage_by_offset[current_function_offset] = current_function_linkage
                end
                # A destructor declared in a record: `D4` is clang's unified
                # spelling, `D1`/`D2` a variant's.
                if current_function_parent !== nothing &&
                   current_function_parent[2] in ("DW_TAG_class_type", "DW_TAG_structure_type")
                    occursin(r"D[124]Ev$", current_function_linkage) &&
                        get!(destructor_of, current_function_parent[1], current_function_linkage)
                    occursin(r"C[124]E", current_function_linkage) &&
                        push!(get!(() -> String[], constructors_of, current_function_parent[1]), current_function_linkage)
                    push!(get!(() -> String[], methods_of, current_function_parent[1]), current_function_linkage)
                end
            end
        end

        # Follow DW_AT_specification to the declaration that owns the mangled name.
        #
        # An out-of-line C++ method definition carries NEITHER DW_AT_linkage_name
        # NOR DW_AT_name — only a back-reference to its in-class declaration:
        #
        #   <2><b3> DW_TAG_subprogram        <- declaration: linkage_name; params UNNAMED
        #   <1><cc> DW_TAG_subprogram        <- definition: low_pc, object_pointer,
        #             DW_AT_specification: <0xb3>    params NAMED (incl. `this`)
        #
        # Without this link `function_key` is `nothing` for the definition DIE, so
        # finalize_current_function! discarded it whole and the only signature ever
        # recorded was the declaration's — whose parameters are all unnamed and are
        # therefore dropped by the name gate on the parameter push. That is the
        # single source of BOTH the "extracted 0, DIE tree has N" arity warnings and
        # the generic argN parameter names in generated wrappers, and it is why a
        # real method's `this` never survived extraction (found via Dear ImGui,
        # 2026-08-09).
        #
        # Adopting the declaration's key makes the definition merge onto it. The
        # non-empty-wins rule already in finalize_current_function! then prefers the
        # definition's real parameter list over the declaration's empty one,
        # regardless of which DIE closes first.
        if contains(line, "DW_AT_specification") && in_function_context
            spec_match = match(r"DW_AT_specification\s*:\s*<(0x[0-9a-fA-F]+)>", line)
            if !isnothing(spec_match)
                spec_target = String(spec_match.captures[1])
                if haskey(linkage_by_offset, spec_target)
                    current_function_linkage = linkage_by_offset[spec_target]
                end
            end
        end

        # (noexcept is NOT read from DWARF: there is no DW_AT_noexcept, and a
        # substring match on "noexcept" here flagged any function whose name
        # contained it. Unwinding is decided from the IR — `_nounwind_definitions`.)

        # Extract virtuality
        # Example: <60>   DW_AT_virtuality  : 1 (virtual)
        if contains(line, "DW_AT_virtuality") && in_function_context
            # Virtuality is usually 1 (virtual) or 2 (pure virtual)
            virt_match = match(r":\s*(\d+)", line)
            if !isnothing(virt_match)
                is_virtual = parse(Int, virt_match.captures[1]) > 0
                current_fn_is_virtual = is_virtual
                
                # Store this temporarily in type_refs for the current function offset
                if !isnothing(current_function_offset)
                    if !haskey(type_refs, current_function_offset)
                        type_refs[current_function_offset] = Dict{String,Any}()
                    end
                    if isa(type_refs[current_function_offset], Dict)
                        type_refs[current_function_offset]["is_virtual"] = is_virtual
                    end
                end
            end
        end

        # Extract function return type reference
        # CRITICAL: Only match DW_AT_type at the function's own level, not from parameters!
        if contains(line, "DW_AT_type") && in_function_context && !function_processed
            type_match = match(r"<(0x[^>]+)>", line)
            if !isnothing(type_match)
                type_ref = String(type_match.captures[1])  # Convert SubString to String
                # Resolve the type reference (follows pointer/const/reference chains)
                c_type = resolve_type(type_ref, type_refs)

                # Map to Julia type using comprehensive mapping
                # Check if it's a struct/class type first
                julia_type = if haskey(type_refs, type_ref) && isa(type_refs[type_ref], Dict) &&
                               get(type_refs[type_ref], "kind", nothing) in ["struct", "class"]
                    # It's a struct/class - use the struct name as the Julia type
                    c_type
                else
                    # Return position: `T&` must be a pointer, not a Ref — see
                    # as_return_julia_type.
                    as_return_julia_type(c_type, dwarf_type_to_julia(c_type))
                end
                type_size = get_type_size(c_type)

                # Use linkage name (C++) or fall back to function name (C)
                function_key = !isnothing(current_function_linkage) ? current_function_linkage : current_function_name

                # Check for virtuality, varargs, and noexcept stored earlier
                is_virtual = false
                is_vararg = false
                is_noexcept = false
                if !isnothing(current_function_offset) && haskey(type_refs, current_function_offset) &&
                   isa(type_refs[current_function_offset], Dict)
                    is_virtual = get(type_refs[current_function_offset], "is_virtual", false)
                    is_vararg = get(type_refs[current_function_offset], "is_vararg", false)
                    is_noexcept = get(type_refs[current_function_offset], "is_noexcept", false)
                end

                # Store if we have a function identifier
                if !isnothing(function_key)
                    # This entry is rebuilt by every DIE of the key that has a
                    # return type, the in-class declaration of a LATER compile
                    # unit included. `has_object_pointer` is a definition's
                    # fact, recorded when that DIE closed; a declaration that
                    # follows must not erase it. It did: a static member with
                    # a non-void return, declared again by a unit after the
                    # one defining it, lost its `false` and was wrapped with a
                    # receiver it does not take. Which unit comes first
                    # follows the IR files' names, so it showed on some hosts
                    # only (CI's LLVM 22 cells, 2026-11-15).
                    kept_object_pointer = haskey(return_types, function_key) ?
                        get(return_types[function_key], "has_object_pointer", nothing) : nothing
                    # ...and so are a definition's parameters (`definition_params`).
                    kept_params = (haskey(return_types, function_key) && function_key in definition_params) ?
                        get(return_types[function_key], "parameters", nothing) : nothing
                    return_types[function_key] = Dict(
                        "c_type" => c_type,
                        "julia_type" => julia_type,
                        "size" => type_size,
                        "is_virtual" => is_virtual, # NEW: Store virtuality
                        "is_vararg" => is_vararg,   # NEW: Store varargs status
                        "is_noexcept" => is_noexcept, # NEW: Store noexcept status
                        "parameters" => params_for_this_function  # NEW: Use params_for_this_function
                    )
                    # A SIMD vector returned by value (`_vector_size`).
                    r_vec = _vector_size(type_ref, type_refs; base_sizes = base_byte_size)
                    r_vec === nothing || (return_types[function_key]["vector_size"] = r_vec)
                    kept_object_pointer isa Bool &&
                        (return_types[function_key]["has_object_pointer"] = kept_object_pointer)
                    kept_params === nothing || (return_types[function_key]["parameters"] = kept_params)
                end
                function_processed = true  # Mark as processed
            end
        end

        # Extract parameter attributes (name and type).
        #
        # A parameter belongs to the function only if its DIE is a DIRECT child
        # of the subprogram. The old `level > function_level` test admitted any
        # deeper DIE anywhere later in the CU, which is what let a
        # DW_TAG_subroutine_type's parameter — and through it a struct member —
        # be attributed to a function it has nothing to do with.
        param_is_direct_child = !isnothing(current_param_level) &&
                                !isnothing(current_function_level) &&
                                current_param_level == current_function_level + 1

        if !isnothing(current_param_offset) && haskey(type_refs, current_param_offset) &&
           haskey(offset_to_kind, current_param_offset) && offset_to_kind[current_param_offset] == :parameter &&
           param_is_direct_child

            param_info = type_refs[current_param_offset]

            # DW_AT_artificial seen BEFORE DW_AT_type — the push below reads it.
            if contains(line, "DW_AT_artificial")
                param_info["artificial"] = true
            end

            # Extract parameter name
            if contains(line, "DW_AT_name")
                name_match_quotes = match(r"DW_AT_name\s*\(\"([^\"]+)\"\)", line)
                if !isnothing(name_match_quotes)
                    param_info["name"] = String(name_match_quotes.captures[1])
                else
                    name_match = match(r"DW_AT_name\s*:\s*(?:\([^)]*\):\s*)?(.*)$", line)
                    if !isnothing(name_match)
                        param_info["name"] = String(strip(name_match.captures[1]))
                    end
                end
            end

            # Extract parameter type
            if contains(line, "DW_AT_type")
                type_match = match(r"<(0x[^>]+)>", line)
                if !isnothing(type_match)
                    param_info["type"] = String(type_match.captures[1])

                    # A parameter's ABI identity is its TYPE and POSITION; the name
                    # is documentation. Requiring a name here dropped every
                    # parameter of every declaration-only DIE — which is where the
                    # implicit `this` lives, and which is the whole of a function
                    # whose definition is in another TU. That produced signatures
                    # short by one or more arguments (the "extracted 0, DIE tree
                    # has N" warnings) and, for a method, a receiver the generators
                    # then had to guess at. Keep it and synthesize a name.
                    #
                    # DW_AT_artificial distinguishes a compiler-synthesized
                    # parameter from a programmer-written one — on a member
                    # function that is exactly the implicit `this`. Naming it
                    # "this" is load-bearing: both generators skip receiver
                    # synthesis when `params[1]["name"] == "this"`, so calling it
                    # `arg0` would let them inject a SECOND receiver.
                    type_ref = param_info["type"]
                    c_type = resolve_type(type_ref, type_refs)
                    julia_type = cpp_to_julia_type(c_type, struct_names, enum_names)

                    pos = get(param_info, "position", length(params_for_this_function))
                    resolved_name = if !isnothing(param_info["name"])
                        param_info["name"]
                    elseif get(param_info, "artificial", false) && pos == 0
                        "this"
                    else
                        "arg$(pos)"
                    end

                    param_dict = Dict(
                        "name" => resolved_name,
                        "c_type" => c_type,
                        "julia_type" => julia_type,
                        "position" => param_info["position"]
                    )
                    # A SIMD vector by value (`_vector_size`): its `c_type` is
                    # the array of its lanes.
                    p_vec = _vector_size(type_ref, type_refs; base_sizes = base_byte_size)
                    p_vec === nothing || (param_dict["vector_size"] = p_vec)

                    # If this is a function pointer, preserve the full signature
                    if startswith(c_type, "function_ptr(")
                        param_dict["function_pointer_signature"] = c_type
                        # …and the typedef it is declared with, which is what
                        # its docstring's `@cfunction` example comes from.
                        fp_td = _function_pointer_typedef(type_ref, type_refs)
                        fp_td === nothing || (param_dict["function_pointer_typedef"] = fp_td)
                    end

                    push!(params_for_this_function, param_dict)
                    pending_param_dict = param_dict  # for a trailing DW_AT_artificial

                    # Reset param offset to avoid re-processing
                    current_param_offset = nothing
                    current_param_level = nothing
                end
            end
        end
    end

    # Close the last function — a dump that ends without a terminator at the
    # subprogram's depth leaves it open.
    finalize_current_function!()

    if isempty(return_types)
        @warn "No DWARF return type info found (compile with -g flag)"
    end

    # A constructor's or destructor's variants share one declaration and so
    # one key, and every definition's parameters were merged into it: the last
    # variant read decided what all of them took. A base-object variant of a
    # class with a virtual base takes the VTT after `this`, so by the order
    # clang emitted them in, a complete-object destructor that takes one
    # argument was `(this, vtt)`, or a base-object constructor that takes two
    # was `(this)`. Such a variant is the definition with exactly one
    # parameter more than its declaration. It gets an entry under its own
    # symbol, the declaration's facts with its own list and the VTT marked
    # `artificial`, and the unified key keeps (or gets back) the declared
    # list. After the walk, because a declaration may follow its definitions.
    for own in sort!(collect(keys(variant_definitions)))
        v = variant_definitions[own]
        declared = get(declared_param_counts, v.key, nothing)
        unified = get(return_types, v.key, nothing)
        (declared === nothing || unified === nothing || haskey(return_types, own)) && continue
        (v.count == declared + 1 && length(v.params) == v.count && v.count >= 2 &&
         get(v.params[1], "name", "") == "this") || continue
        vtt = copy(v.params[2])
        vtt["artificial"] = true
        entry = copy(unified)
        entry["parameters"] = Any[v.params[1], vtt, v.params[3:end]...]
        return_types[own] = entry
        die_param_counts[own] = v.count
        if length(get(unified, "parameters", [])) == v.count
            unified["parameters"] = Any[(q = copy(p); q["position"] = i - 1; q) for (i, p) in
                                        enumerate(p for (j, p) in enumerate(v.params) if j != 2)]
        end
        die_param_counts[v.key] = declared
    end

    # Every extracted signature must match the DIE tree it came from.
    check_param_arity!(return_types, die_param_counts)

    # Children indexed by parent offset, built ONCE. The inheritance and
    # template-param lookups below are per-struct questions ("which DIEs name me
    # as parent?") that were each answered by rescanning all of `type_refs` —
    # `aggregates × DIEs × 2`, quadratic in the binary. Measured on Hub curl:
    # 196841 DIEs, 6594 aggregates, 415s to parse, of which nearly all was two
    # scans that found NOTHING — curl is C, so it has zero DW_TAG_inheritance and
    # zero template params. Every C library paid the full bill to build two empty
    # arrays. 415s → 3.5s with this index, byte-identical output.
    #
    # Built in dump order (pass 2 added parameter and variable DIEs, so the
    # order is taken again), which puts each parent's children in declaration
    # order: template parameters keep the order they are written in.
    die_order = _die_order(type_refs)
    children_by_parent = Dict{Any,Vector{Any}}()
    for off in die_order
        die = type_refs[off]
        isa(die, Dict) || continue
        parent = get(die, "parent", nothing)
        parent === nothing && continue
        push!(get!(() -> Any[], children_by_parent, parent), die)
    end

    # A type the provenance gate drops: declared in a toolchain header, or an
    # anonymous type named after a parent the gate drops. The second is how a
    # lambda's closure type arrives. It is declared at the lambda, in the
    # project's source, but it is reached only through the libc++ record that
    # stores it (`__func<(lambda at <path>:<line>:<col>), …>`), and its name,
    # taken from that record, carries the lambda's path and line. Emitted, it
    # was a dead type that changed name whenever the checkout moved or the
    # source above the lambda gained a line. Measured on llama.cpp: 53 of
    # them, and no function, global or other struct reached one.
    function drops_type(ti)
        ti isa Dict || return false
        _is_system_decl_file(get(ti, "decl_file_path", "")) && return true
        parent = get(ti, "anon_parent", nothing)
        return parent !== nothing && drops_type(get(type_refs, parent, nothing))
    end

    # Extract struct definitions with member information.
    #
    # Keys are the bare DWARF name, as everything downstream keys them
    # (`resolve_type`, member and signature `c_type`s, a method's innermost
    # class scope), so one name can have several definitions: the same type in
    # many compile units, spelled differently by C and C++ (`_Bool`/`bool`, an
    # anonymous member `_anon1` against a named `data`), with or without a
    # vtable home's `__clang_vtable`; or distinct types that share a name (a
    # pimpl `impl` per class). Every definition is collected and one is kept by
    # content (`_canonical_definition`). The last one written used to win, in
    # `type_refs`' hash order, so the choice changed with the Julia release.
    struct_defs = Dict{String,Dict{String,Any}}()
    struct_candidates = Dict{String,Vector{Dict{String,Any}}}()
    toolchain_candidates = Dict{String,Vector{Dict{String,Any}}}()   # dropped by provenance
    decl_files_of = Dict{String,Vector{String}}()                    # headers, per bare name
    record_facts = Dict{String,Dict{String,Any}}()   # → record_abi, see below
    destructor_by_name = Dict{String,String}()       # → record_abi's string_layout
    constructors_by_name = Dict{String,Vector{String}}()
    methods_by_name = Dict{String,Vector{String}}()
    record_kind_by_name = Dict{String,String}()      # "class" or "struct": a member's default accessibility
    for offset in die_order
        type_info = type_refs[offset]
        if isa(type_info, Dict) && get(type_info, "kind", nothing) in ["struct", "class", "union"]
            struct_name = get(type_info, "name", "unknown")
            if struct_name != "unknown" && struct_name != "unknown_struct" && struct_name != "unknown_class" && struct_name != "unknown_union"
                let dk = get(destructor_of, offset, nothing)
                    dk === nothing || get!(destructor_by_name, String(struct_name), dk)
                end
                let ck = get(constructors_of, offset, nothing)
                    if ck !== nothing
                        append!(get!(() -> String[], constructors_by_name, String(struct_name)), ck)
                        get!(record_kind_by_name, String(struct_name), String(type_info["kind"]))
                    end
                end
                let mk = get(methods_of, offset, nothing)
                    if mk !== nothing
                        append!(get!(() -> String[], methods_by_name, String(struct_name)), mk)
                        get!(record_kind_by_name, String(struct_name), String(type_info["kind"]))
                    end
                end
                # Every named DEFINITION (a declaration has no byte_size), before
                # any provenance or member filter: calling a function that takes
                # `std::string_view` by value needs the ABI facts of a type the
                # API tables below deliberately leave out.
                if haskey(type_info, "byte_size")
                    has_base = any(d -> isa(d, Dict) && get(d, "kind", nothing) == "inheritance",
                                   get(children_by_parent, offset, ()))
                    _merge_record_facts!(record_facts, String(struct_name), Dict{String,Any}(
                        "byte_size" => type_info["byte_size"],
                        "pass" => _record_pass_convention(get(type_info, "calling_convention", nothing)),
                        # No data member and no base: the record has no bytes a
                        # callee reads. A union is never counted, and a class
                        # with virtuals has its `_vptr` member.
                        "empty" => type_info["kind"] != "union" &&
                                   isempty(get(type_info, "members", [])) && !has_base))
                end

                # Resolve member types to Julia types
                resolved_members = []
                for member in get(type_info, "members", [])
                    member_type_ref = get(member, "type", nothing)
                    if !isnothing(member_type_ref)
                        c_type = resolve_type(member_type_ref, type_refs)
                        julia_type = cpp_to_julia_type(c_type, struct_names, enum_names)
                        # `get_type_size` only knows primitives; any other
                        # member's size comes from its type's DIE chain
                        # (`_die_byte_size`): an embedded record, an enum, a
                        # typedef of either, an array of anything (`float v[3]`
                        # carries no byte_size of its own). A size of 0 stops the
                        # emitters' running offset, and GeneratorCpp's layout
                        # proof then emits the whole record as a byte blob.
                        m_size = get_type_size(c_type)
                        m_size == 0 &&
                            (m_size = _die_byte_size(member_type_ref, type_refs; base_sizes = base_byte_size))
                        if m_size == 0
                            am = match(r"^(.+?)((?:\[\d+\])+)$", c_type)
                            if am !== nothing
                                es = get_type_size(strip(String(am.captures[1])))
                                dims = [parse(Int, d.captures[1]) for d in eachmatch(r"\[(\d+)\]", String(am.captures[2]))]
                                if es > 0 && !isempty(dims)
                                    m_size = es * prod(dims)
                                end
                            end
                        end
                        member_resolved = Dict(
                            "name" => get(member, "name", "unknown"),
                            "c_type" => c_type,
                            "julia_type" => julia_type,
                            "size" => m_size,
                            "offset" => let v = get(member, "offset", nothing); isnothing(v) ? nothing : v end
                        )
                        # Propagate bitfield attributes if present
                        for bf_key in ["bit_size", "data_bit_offset", "bit_offset_legacy"]
                            if haskey(member, bf_key)
                                member_resolved[bf_key] = member[bf_key]
                            end
                        end
                        # C11 anonymous member: C injects its members into the
                        # enclosing scope, so the generator forwards them.
                        if get(member, "anonymous_member", false) == true
                            member_resolved["anonymous_member"] = true
                        end
                        # The alignment the member asked for (DW_AT_alignment),
                        # or a vector's, which is its size and carries no
                        # attribute (an array of vectors too). Natural
                        # alignment is left to the reader.
                        m_align = get(member, "alignment", nothing)
                        m_vec = _vector_size(member_type_ref, type_refs; base_sizes = base_byte_size, arrays = true)
                        if m_vec !== nothing
                            member_resolved["vector_size"] = m_vec
                            ispow2(m_vec) && (m_align = max(something(m_align, 1), m_vec))
                        elseif m_align !== nothing && m_align == m_size && ispow2(m_size) && endswith(c_type, "]")
                            # An array aligned to its own size (cglm's `vec4`,
                            # `float[4]` aligned 16) is what a vector looked like
                            # in metadata from before `vector_size`. `null` says
                            # this one was looked at and is not a vector.
                            member_resolved["vector_size"] = nothing
                        end
                        m_align === nothing || (member_resolved["alignment"] = m_align)
                        push!(resolved_members, member_resolved)
                    end
                end

                # A toolchain record with no member of its own has an entry only
                # when its bytes come from its non-virtual bases (libstdc++'s
                # `std::tuple` and `_Tuple_impl`, `std::optional` and
                # `_Optional_payload`, libc++'s `__tuple_impl`):
                # `_promote_toolchain_records!` lays it out from them.
                from_bases = isempty(resolved_members) && drops_type(type_info) &&
                    any(d -> isa(d, Dict) && get(d, "kind", nothing) == "inheritance" &&
                             get(d, "virtual", false) != true,
                        get(children_by_parent, offset, ()))
                # Every DEFINED library record has an entry, members or not
                # (2026-11-14): a derived class that adds methods and no field
                # (Bullet's `btBoxShape`, `btQuaternion`; 39 of its 603 records),
                # a stateless functor, an interface. Without one the wrapper had
                # no `STRUCT_SIZES` entry to size the class's constructor
                # storage by, every class deriving from it lost the bases above
                # it (`btConvexInternalShape` carried a 32-byte pad where
                # `btCollisionShape`'s fields belong), and graph mode could not
                # see what it derives from. The entry carries `members = []` and
                # `base_classes`; GeneratorCpp flattens the bases into fields,
                # and the thunk side reads a record's class from its bases'
                # leaves. A declaration has no `byte_size`; a size-0 record is
                # GNU C's empty struct, which has nothing to size; an anonymous
                # record exists to be embedded; a lambda's closure type cannot be
                # named. A virtual base is flattened by nobody, so such a record
                # gets its entry and keeps every refusal it had.
                library_defined = isempty(resolved_members) && !drops_type(type_info) &&
                    _memberless_record_entry(String(struct_name), type_info)
                if !isempty(resolved_members) || from_bases || library_defined
                    struct_def = Dict{String,Any}(
                        "kind" => get(type_info, "kind", "struct"),
                        "byte_size" => get(type_info, "byte_size", "0x0"),
                        "members" => resolved_members
                    )
                    # Synthesized name (see the anonymous-aggregate pass) — the
                    # generator emits these as ABI-exact embeddable types rather
                    # than as user-facing API surface.
                    if get(type_info, "anonymous", false) == true
                        struct_def["anonymous"] = true
                    end
                    # `aligned(N)` on the record (DW_AT_alignment); see above.
                    haskey(type_info, "alignment") && (struct_def["alignment"] = type_info["alignment"])

                    # Add inheritance information if available
                    base_classes = []
                    for inh_info in get(children_by_parent, offset, ())
                        if isa(inh_info, Dict) && get(inh_info, "kind", nothing) == "inheritance"
                            base_type_ref = get(inh_info, "base_type", nothing)
                            if !isnothing(base_type_ref)
                                base_type = resolve_type(base_type_ref, type_refs)
                                base_entry = Dict{String,Any}(
                                    "type" => base_type,
                                    "accessibility" => get(inh_info, "accessibility", "public"),
                                    # Byte offset of this base's subobject in the
                                    # derived object (0 for the primary base)
                                    "offset" => get(inh_info, "offset", 0),
                                    "virtual" => get(inh_info, "virtual", false)
                                )
                                # Virtual base: static offset is meaningless; the
                                # object's vtable holds the real offset at this
                                # (negative) byte position from the address point.
                                if haskey(inh_info, "vbase_vtable_offset")
                                    base_entry["vbase_vtable_offset"] = inh_info["vbase_vtable_offset"]
                                end
                                push!(base_classes, base_entry)
                            end
                        end
                    end
                    if !isempty(base_classes)
                        # type_refs is a Dict — iteration order is arbitrary.
                        # Sort by subobject offset (primary base first) so
                        # consumers see a deterministic, layout-ordered list.
                        sort!(base_classes, by = b -> (get(b, "offset", 0), get(b, "type", "")))
                        struct_def["base_classes"] = base_classes
                    end

                    # The namespaces and records this one is declared in
                    # (`boost::core`, `std::__1`, "" at global scope), when the
                    # tree settled it. The C++ generator's STL screen matches
                    # bare names, which is all DWARF keys records by; this is
                    # what tells `boost::core::basic_string_view<char>` from
                    # `std::string_view`.
                    let sp = get(type_info, "scope_parent", missing)
                        if sp !== missing
                            sc = _scope_of(sp, type_refs)
                            sc === nothing || (struct_def["scope"] = sc)
                        end
                    end

                    # Add template parameters if available
                    template_params = []
                    for tmpl_info in get(children_by_parent, offset, ())
                        if isa(tmpl_info, Dict)
                            if get(tmpl_info, "kind", nothing) == "template_type"
                                type_ref = get(tmpl_info, "type", nothing)
                                param_type = !isnothing(type_ref) ? resolve_type(type_ref, type_refs) : "Any"
                                push!(template_params, Dict(
                                    "kind" => "type",
                                    "name" => get(tmpl_info, "name", "T"),
                                    "type" => param_type
                                ))
                            elseif get(tmpl_info, "kind", nothing) == "template_value"
                                type_ref = get(tmpl_info, "type", nothing)
                                param_type = !isnothing(type_ref) ? resolve_type(type_ref, type_refs) : "Int"
                                push!(template_params, Dict(
                                    "kind" => "value",
                                    "name" => get(tmpl_info, "name", "N"),
                                    "type" => param_type,
                                    "value" => get(tmpl_info, "value", nothing)
                                ))
                            end
                        end
                    end
                    if !isempty(template_params)
                        struct_def["template_params"] = template_params
                    end

                    # Provenance gate: a type declared in a toolchain header is
                    # not part of this library's API. Skipping it here — at the
                    # emission point, on the ACTUAL record being written — rather
                    # than post-filtering keeps the intent/output split that has
                    # bitten the generators repeatedly. Its layout is kept aside:
                    # the library may hold or pass it by value (`std::pair`), and
                    # `_promote_toolchain_records!` brings back exactly those.
                    # Where this definition was declared, beside the definition and
                    # not in it: `_canonical_definition` orders definitions by their
                    # text, and a header path must not decide which one is kept.
                    let p = String(get(type_info, "decl_file_path", ""))
                        isempty(p) || push!(get!(() -> String[], decl_files_of, struct_name), replace(p, '\\' => '/'))
                    end
                    if drops_type(type_info)
                        push!(get!(() -> Dict{String,Any}[], toolchain_candidates, struct_name), struct_def)
                        continue
                    end

                    push!(get!(() -> Dict{String,Any}[], struct_candidates, struct_name), struct_def)
                end
            end
        end
    end
    for (struct_name, defs) in struct_candidates
        struct_defs[struct_name] = _canonical_definition(defs)
        _record_provenance!(struct_defs[struct_name], defs, get(decl_files_of, struct_name, String[]))
    end
    toolchain_defs = Dict{String,Dict{String,Any}}(name => _canonical_definition(defs)
                                                   for (name, defs) in toolchain_candidates)
    for (name, defs) in toolchain_candidates
        _record_provenance!(toolchain_defs[name], defs, get(decl_files_of, name, String[]))
    end
    # An empty base holds no bytes, and a record with no members has no entry of
    # its own (`Tag` in `struct P : Tag { double a, b; }`, libstdc++'s
    # `__pair_base` under every `std::pair`). Say so on the base entry, or the
    # SysV classifier meets a name it cannot read and traps every by-value
    # crossing of the derived record. Only a 1-byte one: an `alignas` empty
    # base raises the derived record's alignment, which DWARF does not repeat.
    for defs in (struct_defs, toolchain_defs), (_, def) in defs
        for b in get(def, "base_classes", ())
            fk = _key_in(String(get(b, "type", "")), record_facts)
            fk === nothing && continue
            facts = record_facts[fk]
            get(facts, "empty", false) == true && _definition_byte_size(facts) == 1 &&
                (b["empty"] = true)
        end
    end


    # Extract enum definitions with enumerator information. Keyed and chosen
    # like the records above: a nested enum's bare name (`Type`, `State`) can
    # name several enums.
    enum_defs = Dict{String,Dict{String,Any}}()
    enum_candidates = Dict{String,Vector{Dict{String,Any}}}()
    toolchain_enum_candidates = Dict{String,Vector{Dict{String,Any}}}()   # by `__enum__<name>`
    enum_scopes_of = Dict{String,Vector{String}}()
    enum_decl_files_of = Dict{String,Vector{String}}()
    for offset in die_order
        type_info = type_refs[offset]
        if isa(type_info, Dict) && get(type_info, "kind", nothing) == "enum"
            enum_name = get(type_info, "name", "unknown")
            if enum_name != "unknown" && enum_name != "unknown_enum"
                # Get underlying type
                underlying_type_ref = get(type_info, "underlying_type", nothing)
                underlying_c_type = "int"  # Default
                if !isnothing(underlying_type_ref)
                    underlying_c_type = resolve_type(underlying_type_ref, type_refs)
                end
                underlying_julia_type = cpp_to_julia_type(underlying_c_type, struct_names, enum_names)

                # Get enumerators
                enumerators = get(type_info, "enumerators", [])

                # An enum DEFINED with no enumerators is a type too: C++'s
                # `enum class byte : unsigned char {};` (`std::byte`), a
                # library's `enum class Handle : uint32_t {};`. It crosses a
                # call as its underlying integer like any enum, and without an
                # entry it was a record with no layout: `struct byte end` and
                # an ABI trap. An opaque declaration (`enum class E : int;`)
                # says neither what it is an integer of nor what it holds, and
                # gets none. One a toolchain header declares is set aside like
                # a toolchain record, and comes back only where the library
                # names it (`_promote_toolchain_records!`).
                empty_definition = isempty(enumerators) && !isnothing(underlying_type_ref) &&
                                   get(type_info, "declaration", false) != true &&
                                   !isnothing(get(type_info, "byte_size", nothing))
                set_aside = empty_definition && drops_type(type_info)
                if !isempty(enumerators) || empty_definition
                    enum_key = set_aside ? "__enum__" * enum_name : enum_name
                    push!(get!(() -> Dict{String,Any}[], set_aside ? toolchain_enum_candidates : enum_candidates,
                               enum_key), Dict{String,Any}(
                        "kind" => "enum",
                        "underlying_type" => underlying_c_type,
                        "julia_type" => underlying_julia_type,
                        "byte_size" => get(type_info, "byte_size", "0x04"),
                        "enumerators" => enumerators
                    ))
                    # Scope and header beside the definition (see the records above).
                    let sp = get(type_info, "scope_parent", missing)
                        sc = sp === missing ? nothing : _scope_of(sp, type_refs)
                        sc === nothing || push!(get!(() -> String[], enum_scopes_of, enum_key), sc)
                    end
                    let p = String(get(type_info, "decl_file_path", ""))
                        isempty(p) || push!(get!(() -> String[], enum_decl_files_of, enum_key), replace(p, '\\' => '/'))
                    end
                end
            end
        end
    end
    for (enum_name, defs) in enum_candidates
        enum_defs[enum_name] = _canonical_definition(defs)
        _record_provenance!(enum_defs[enum_name], get(enum_scopes_of, enum_name, String[]),
                            get(enum_decl_files_of, enum_name, String[]))
    end
    # The toolchain's enumerator-less enums wait beside its records, under the
    # key they will have in `struct_defs`.
    for (enum_key, defs) in toolchain_enum_candidates
        toolchain_defs[enum_key] = _canonical_definition(defs)
        _record_provenance!(toolchain_defs[enum_key], get(enum_scopes_of, enum_key, String[]),
                            get(enum_decl_files_of, enum_key, String[]))
    end


    # Store enum_defs in struct_defs with special marker (for now, to maintain API compatibility)
    # Later we can extend the return type
    for (enum_name, enum_info) in enum_defs
        struct_defs["__enum__" * enum_name] = enum_info
    end

    # Toolchain records the library holds or passes BY VALUE get their layout
    # back (`_promote_toolchain_records!`). Here, not in a later pass: the
    # return repair below sizes a by-value return from `struct_defs`, and a
    # `std::pair` return left at size 0 is as unclassifiable as one typed `Any`.
    # Seeds are what the wrapper can reach, the dynamic-symbol gate every other
    # wrapped thing passes: an exported function (a constructor or destructor
    # through its unified C4/D4 DWARF key, `_structor_dwarf_key`) and an exported
    # global, less the standard library's own (`_is_stdlib_internal`, which
    # the function list screens with too). A library record seeds nothing by
    # itself; one reached by value is walked (`_promote_toolchain_records!`).
    # Without the gate, ggml-vulkan's hidden C++ promoted 123 Vulkan-Hpp records
    # (vk::Event, vk::Semaphore, …) that no exported function names (2026-11-08).
    # `exported === nothing` (no binary to ask, as in a parser test) falls back
    # to every function that is not the standard library's own.
    seed_fn = if exported === nothing
        Set(String(k) for k in keys(return_types) if !_is_stdlib_internal(String(k), ""))
    else
        # Exported AND not the standard library's own, the two screens the
        # function list passes: libstdc++ gives `namespace std` default
        # visibility, so `std::map<…>::find` is in the dynamic table of a
        # -fvisibility=hidden build and seeded 55 `_Rb_tree_iterator`s and
        # friends for a llama.cpp no wrapper function returns one from.
        Set{String}(something(haskey(return_types, e) ? e : nothing, _structor_dwarf_key(e, return_types), "")
                    for e in exported if !_is_stdlib_internal(e, ""))
    end
    toolchain_seeds = String[]
    for (fkey, info) in return_types
        (isa(info, Dict) && String(fkey) in seed_fn) || continue
        push!(toolchain_seeds, String(get(info, "c_type", "")))
        for p in get(info, "parameters", [])
            push!(toolchain_seeds, String(get(p, "c_type", "")))
        end
    end
    for offset in die_order
        info = type_refs[offset]
        if isa(info, Dict) && get(info, "kind", nothing) == "variable" &&
           get(info, "external", false) == true && !get(info, "declaration", false) &&
           !_is_system_decl_file(get(info, "decl_file_path", "")) &&
           (exported === nothing || String(get(info, "name", "")) in exported)
            type_ref = get(info, "type", nothing)
            isnothing(type_ref) || push!(toolchain_seeds, resolve_type(type_ref, type_refs))
        end
    end
    # The 1-byte empty records, the rule the empty-base marking above uses: a
    # union may hold one beside its real members (libstdc++'s `_Storage` holds
    # `_Empty_byte` beside the value of a `std::optional`).
    empty_records = Set{String}(k for (k, f) in record_facts
                                if get(f, "empty", false) == true && _definition_byte_size(f) == 1)
    _promote_toolchain_records!(struct_defs, toolchain_defs, toolchain_seeds; empty_records)

    # ---- Repair by-value AGGREGATE return types that resolved to `Any` ----
    #
    # The in-loop return mapping ("Extract function return type reference",
    # above) decides "is this an aggregate?" from the DIE kind of the type the
    # subprogram POINTS AT — `kind in ["struct", "class"]` — not from the
    # RESOLVED type. `typedef struct mpack_tag_t mpack_tag_t;`, the ordinary C
    # spelling, makes that DIE a **typedef**, so the aggregate branch missed and
    # the name-only fallback mapper returned "Any" with size 0. That test also
    # never had a "union" branch — the same omission fixed in `resolve_type` on
    # 2026-08-01 for member resolution.
    #
    # The asymmetry is the tell: the PARAMETER path (~line 4229) resolves the
    # type and then maps it with the struct/enum-aware `cpp_to_julia_type`, so
    # the very same type is typed correctly as an argument and `Any` as a
    # return. `mpack_tag_cmp(mpack_tag_t, mpack_tag_t) -> int` extracted
    # perfectly while `mpack_tag_make_nil() -> mpack_tag_t` did not.
    #
    # Downstream an `Any` return is not a cosmetic defect: the function is
    # dropped rather than wrapped (an emitted one would trip
    # `_assert_no_any_ccall_return` and refuse the WHOLE wrapper), so the symbol
    # silently has no binding at all. Measured over the Hub before this fix:
    # 27 unreachable exports in mpack, every `mpack_tag_*` returning
    # `mpack_tag_t` by value.
    #
    # Deliberately a POST-PASS, not an inline fix. `struct_defs` is only
    # complete after the parse loop, and a subprogram DIE can precede the DIE of
    # the aggregate it returns, so an in-loop name test would be
    # order-dependent. And it only ever rewrites "Any": it cannot change a
    # return type that already mapped, so the blast radius is exactly the set of
    # functions that previously had no wrapper. In particular the `char*` →
    # `Cstring` mapping (and the whole Cstring return policy built on it) is
    # untouched, which a wholesale swap to `cpp_to_julia_type` would have broken.
    # TWO defects, not one — the size half also bites the DIRECT spelling.
    # When the DW_AT_type does point straight at the structure the kind test
    # matches and `julia_type` comes out right, but the size beside it is
    # `get_type_size(c_type)`, which only knows primitives and answers 0 for
    # every aggregate. A by-value struct return whose size is 0 is just as
    # unclassifiable downstream as one typed `Any`, so both fields are repaired
    # and they are repaired independently.
    for (_fname, _info) in return_types
        isa(_info, Dict) || continue
        _jt = get(_info, "julia_type", "")
        _sz = get(_info, "size", 0)
        # Nothing to do unless the type is unresolved or the size unmeasured.
        (_jt == "Any" || _sz == 0) || continue
        _ct = String(get(_info, "c_type", ""))
        # Pointers and arrays already map through the normal path; only a
        # by-value aggregate or enum can land here. `void` has size 0 too, and
        # falls out below because it is not a key in struct_defs.
        (isempty(_ct) || occursin("*", _ct) || occursin("[", _ct)) && continue
        _key = haskey(struct_defs, _ct) ? _ct :
               (haskey(struct_defs, "__enum__" * _ct) ? "__enum__" * _ct : nothing)
        _key === nothing && continue
        _jt == "Any" && (_info["julia_type"] = _ct)
        if _sz == 0
            _bs = get(struct_defs[_key], "byte_size", nothing)
            if !isnothing(_bs)
                _info["size"] = try
                    _s = string(_bs)
                    startswith(_s, "0x") ? parse(Int, _s[3:end], base=16) : parse(Int, _s)
                catch
                    0
                end
            end
        end
    end
    
    # Process extracted variables into a clean dictionary. A C++17 inline
    # variable is defined in every compile unit that uses it, and C and C++
    # spell a type differently, so the entry is chosen like a record's.
    global_vars = Dict{String,Any}()
    global_candidates = Dict{String,Vector{Dict{String,Any}}}()
    for offset in die_order
        info = type_refs[offset]
        if isa(info, Dict) && get(info, "kind", nothing) == "variable" &&
           get(info, "external", false) == true && !get(info, "declaration", false)

           name = get(info, "name", "unknown")
           type_ref = get(info, "type", nothing)

           if name != "unknown" && !isnothing(type_ref)
               c_type = resolve_type(type_ref, type_refs)
               julia_type = cpp_to_julia_type(c_type, struct_names, enum_names)

               push!(get!(() -> Dict{String,Any}[], global_candidates, name), Dict{String,Any}(
                   "name" => name,
                   "c_type" => c_type,
                   "julia_type" => julia_type
               ))
           end
        end
    end
    for (name, defs) in global_candidates
        global_vars[name] = _canonical_definition(defs)
    end


    # Build typedef resolution table: typedef name -> resolved Julia type. A
    # member typedef's bare name (`value_type`, `size_type`) names one typedef
    # per class, so of several targets the table keeps one by content too.
    typedef_table = Dict{String,String}()
    for offset in die_order
        type_info = type_refs[offset]
        if isa(type_info, Dict) && get(type_info, "kind", nothing) == "typedef"
            td_name = get(type_info, "name", "")
            if !isempty(td_name)
                target_ref = get(type_info, "target", nothing)
                if !isnothing(target_ref)
                    c_type = resolve_type(target_ref, type_refs)
                    julia_type = cpp_to_julia_type(c_type, struct_names, enum_names)
                    if julia_type != "Any"
                        prev = get(typedef_table, td_name, nothing)
                        (prev === nothing || julia_type < prev) && (typedef_table[td_name] = julia_type)
                    end
                end
            end
        end
    end

    # ABI facts for every record a function takes or returns BY VALUE. The
    # Tier-2 thunk needs them for exactly the records `struct_defs` cannot
    # describe: an empty class (no members, so no entry), a filtered toolchain
    # type (`std::string_view`), a record only ever declared here (constructor
    # homing). See `FunctionGen.by_value_crossing`, which is the consumer.
    record_abi = Dict{String,Dict{String,Any}}()
    for (_, info) in return_types
        isa(info, Dict) || continue
        for t in Iterators.flatten(((get(info, "c_type", ""),),
                                    (get(p, "c_type", "") for p in get(info, "parameters", []))))
            name = _by_value_record_name(String(t))
            name === nothing && continue
            facts = get(record_facts, name, nothing)
            facts === nothing || (record_abi[name] = facts)
        end
    end
    # A `std::basic_string` crossed by value: how to read it and end it
    # (`_string_layout`), for GeneratorCpp's decoder (`StdStrings`).
    for name in collect(keys(record_abi))
        startswith(name, "basic_string<") || continue
        entry = Dict{String,Any}(record_abi[name])
        entry["string_layout"] = _string_layout(name, toolchain_defs, struct_defs,
                                                get(destructor_by_name, name, nothing);
                                                byte_size_of = t -> (fk = _key_in(String(t), record_facts);
                                                                     fk === nothing ? nothing :
                                                                     _definition_byte_size(record_facts[fk])),
                                                constructors = get(constructors_by_name, name, String[]),
                                                params_of = sym -> (e = get(return_types, sym, nothing);
                                                                    e === nothing ? nothing :
                                                                    [String(get(q, "c_type", "")) for q in get(e, "parameters", [])]))
        record_abi[name] = entry
    end

    # The constructors each library record declares, with the declaration
    # DIE's facts, for the inline-constructor plan (`_inline_constructor_plan`).
    _record_declared_constructors!(record_abi, struct_defs, constructors_by_name,
                                   record_kind_by_name, declaration_facts)
    # ...and all its member functions, for the inline-method plan.
    _record_declared_constructors!(record_abi, struct_defs, methods_by_name,
                                   record_kind_by_name, declaration_facts; key = "methods")

    _drop_force_helper_records!(struct_defs, record_abi)

    return (return_types, struct_defs, global_vars, typedef_table, record_abi)
end

"""
    _drop_force_helper_records!(struct_defs, record_abi)

Remove the records of namespace `replibuild_force`: the helpers of the units
RepliBuild generates (`ctor_impl<true, T, A...>`, `args<A...>`). They exist
to be instantiated, like the helper functions `extract_symbols_from_binary`
filters, and are not the library's: as memberless records they had an entry,
and every wrapper of a package with a forced constructor carried a
`struct ctor_impl_true_…` per constructor.
"""
function _drop_force_helper_records!(struct_defs::AbstractDict, record_abi::AbstractDict)
    for name in collect(keys(struct_defs))
        def = struct_defs[name]
        def isa AbstractDict || continue
        scope = get(def, "scope", nothing)
        scope isa AbstractString || continue
        (scope == "replibuild_force" || startswith(scope, "replibuild_force::")) || continue
        delete!(struct_defs, name)
        delete!(record_abi, name)
    end
    return struct_defs
end

"""
    _record_declared_constructors!(record_abi, struct_defs, constructors_by_name,
                                   kind_by_name, declaration_facts)

On each LIBRARY record's `record_abi` entry, `constructors`: one row per
constructor its definition declares (`symbol`, the unified `C4` DWARF key),
with the declaration DIE's facts: `artificial` (an implicit member clang
declared for it), `deleted`, and `accessibility` (`DW_AT_accessibility` as
written, else the record's default: private in a `class`, public in a
`struct`). With `key = "methods"` the same rows are written for every member
function the record declares (`methods`, with `virtual` added; constructors
and destructors among them, which `_inline_method_plan` passes over). The parameters are not recorded: a declaration's formal parameters
carry no names, so the parser keeps none, and the symbol spells them anyway
(`_constructor_spelling`). Toolchain records put back by promotion and enum
keys get no rows; a record with no `record_abi` entry (a declaration) gets
one only for this key. Rows are sorted by symbol, so the metadata is stable.
"""
function _record_declared_constructors!(record_abi::AbstractDict, struct_defs::AbstractDict,
                                        constructors_by_name::AbstractDict, kind_by_name::AbstractDict,
                                        declaration_facts::AbstractDict; key::String = "constructors")
    for (name, def) in struct_defs
        startswith(name, "__enum__") && continue
        get(def, "toolchain", false) == true && continue
        syms = get(constructors_by_name, name, nothing)
        syms === nothing && continue
        default_access = get(kind_by_name, name, "struct") == "class" ? "private" : "public"
        rows = Dict{String,Any}[]
        for sym in sort!(unique(String[String(s) for s in syms]))
            facts = get(declaration_facts, sym, nothing)
            facts === nothing && continue           # no declaration DIE seen for it
            push!(rows, Dict{String,Any}(
                "symbol" => sym,
                "artificial" => get(facts, "artificial", false) == true,
                "deleted" => get(facts, "deleted", false) == true,
                "accessibility" => something(get(facts, "accessibility", nothing), default_access)))
            # A member function's row also says whether it is virtual: a
            # virtual one is emitted with its class's vtable, never forced.
            key == "methods" && (rows[end]["virtual"] = get(facts, "virtual", false) == true)
        end
        isempty(rows) && continue
        entry = Dict{String,Any}(get(record_abi, name, Dict{String,Any}()))
        entry[key] = rows
        record_abi[name] = entry
    end
    return record_abi
end

"""
    _apply_default_argument_identity!(functions, bound) -> functions

Finish what the demangled-name rewrite began for each default-argument
function (`bound`: its symbol → its plan row): the first parameter, the
object the call is made on, is named `this` and `has_object_pointer` is
`true`, so both receiver gates see a method with the receiver DWARF supplied;
the other parameters take the names the member's own definition gives them,
where it has one; and the entry says what it is (`default_arguments`: the
member's symbol and how many arguments it leaves out). A static member's
row does not compile in the unit (`nonstatic`), on any toolchain; a function
whose member's definition says it is static is removed here as well. The
DWARF fact alone was not enough: clang 22 does not record it for every
static member, and the short call of one reached the wrapper (CI, 2026-11-15).
"""
function _apply_default_argument_identity!(functions::AbstractVector, bound::AbstractDict)
    isempty(bound) && return functions
    by_symbol = Dict{String,Any}(String(get(f, "mangled", "")) => f for f in functions)
    drop = Set{String}()
    for f in functions
        row = get(bound, String(get(f, "mangled", "")), nothing)
        row === nothing && continue
        target_sym = String(row["symbol"])
        names = get(row, "ctor", false) == true ? reverse(_constructor_variants(target_sym)) : [target_sym]
        k = findfirst(n -> haskey(by_symbol, n), names)
        target = k === nothing ? nothing : by_symbol[names[k]]
        params = get(f, "parameters", Any[])
        # The function's own DWARF must have supplied the object and the
        # prefix; an inferred list would make `this` out of a real argument.
        if length(params) != length(row["parameters"]) + 1 || get(f, "parameters_source", "") != "dwarf" ||
           (target !== nothing && get(target, "has_object_pointer", nothing) === false)
            push!(drop, String(f["mangled"]))
            continue
        end
        params = Any[Dict{String,Any}(p) for p in params]
        params[1]["name"] = "this"
        if target !== nothing
            tp = [p for p in get(target, "parameters", Any[]) if get(p, "name", "") != "this"]
            for i in 2:length(params)
                (i - 1) <= length(tp) || break
                n = String(get(tp[i-1], "name", ""))
                isempty(n) || (params[i]["name"] = n)
            end
        end
        f["parameters"] = params
        f["has_object_pointer"] = true
        f["default_arguments"] = Dict{String,Any}("of" => target_sym, "omitted" => Int(row["omitted"]))
    end
    isempty(drop) || filter!(f -> !(String(get(f, "mangled", "")) in drop), functions)
    return functions
end

"""
    _record_pass_convention(cc) -> Union{String,Nothing}

DWARF's `DW_AT_calling_convention` on a record, as `"value"` (5,
DW_CC_pass_by_value) or `"reference"` (4, DW_CC_pass_by_reference). `nothing`
when the attribute is absent, which is every C record.
"""
_record_pass_convention(cc) = cc == 5 ? "value" : cc == 4 ? "reference" : nothing

"""
    _string_constructor(constructors, params_of) -> Union{Nothing,Tuple{String,Bool}}

Of a `basic_string`'s declared constructors (their `C4`/`C1`/`C2` linkage
names), the one that builds from a character pointer and a length:
`(const CharT*, size_type)` (libc++ declares it), else
`(const CharT*, size_type, const allocator&)` (libstdc++ declares only this;
the allocator is an empty class, so the caller passes a byte). `params_of`
gives a linkage name's parameter C types as the DWARF declares them, `this`
first, or `nothing`. The symbol as declared (`C4`, the unified spelling the
DWARF carries, or a variant); the wrapper spells the complete-object `C1`.
`(symbol, takes_allocator)`, or `nothing` when the record declares neither.
"""
function _string_constructor(constructors::AbstractVector, params_of)
    is_char_ptr(t) = (s = strip(replace(String(t), r"\bconst\b" => "")); occursin(r"^(?:char|wchar_t|char(?:8|16|32)_t)\s*\*$", s))
    is_size(t) = strip(String(t)) in ("size_t", "size_type", "unsigned long", "unsigned long long", "unsigned int",
                                      "unsigned long int", "unsigned", "uint64_t", "uint32_t", "std::size_t")
    is_alloc(t) = (s = strip(String(t)); occursin(r"^const\s+allocator<.*>\s*&$", s) || occursin(r"allocator<.*>\s*const\s*&$", s))
    with_alloc = nothing
    for sym in constructors
        ps = params_of(String(sym))
        ps === nothing && continue
        args = length(ps) >= 1 ? ps[2:end] : ps          # drop `this`
        if length(args) == 2 && is_char_ptr(args[1]) && is_size(args[2])
            return (String(sym), false)
        elseif length(args) == 3 && is_char_ptr(args[1]) && is_size(args[2]) && is_alloc(args[3])
            with_alloc === nothing && (with_alloc = (String(sym), true))
        end
    end
    return with_alloc
end

"""
    _string_layout(name, toolchain_defs, struct_defs, destructor) -> Dict

What a Julia caller needs to read a `std::basic_string` returned by value and to
end it, from the record's own DWARF: `"leaves"`, every member at depth with its
dotted `path` (`_M_dataplus._M_p`), `c_type`, absolute `offset` in bytes (or,
for a bitfield, `bit_size` and absolute `data_bit_offset`), through nested
records, unions and non-empty bases; `"char_type"`, the first template type
argument; `"destructor"`, the linkage name of the destructor declared in the
record (clang's unified `D4`, as written), or `nothing`. A part that cannot be
walked (a virtual base, a record with no definition that is not one empty
byte, nesting past 12) leaves
`"leaves"` empty and says why in `"unread"`, and the generator keeps the
stand-in. Which leaves mean what is the reader's business
(`StdStrings.string_reader`), not the build's: libstdc++ and libc++ spell the
string differently, and a layout this does not recognise must be refused
there, not guessed here.
"""
function _string_layout(name::AbstractString, toolchain_defs::AbstractDict, struct_defs::AbstractDict,
                        destructor::Union{Nothing,AbstractString}; byte_size_of = t -> nothing,
                        constructors::AbstractVector = String[], params_of = sym -> nothing)
    lookup(t) = (k = _key_in(String(t), toolchain_defs); k !== nothing ? toolchain_defs[k] :
                 (k2 = _key_in(String(t), struct_defs); k2 === nothing ? nothing : struct_defs[k2]))
    out = Dict{String,Any}("leaves" => Any[], "destructor" => destructor, "char_type" => nothing)
    # The constructor a Julia `String` is built into a slot with, for a
    # parameter (`StdStrings.string_arg`); `nothing` when the record declares
    # none of that shape, and the wrapper then keeps the stand-in.
    ctor = _string_constructor(constructors, params_of)
    out["constructor"] = ctor === nothing ? nothing : ctor[1]
    out["constructor_allocator"] = ctor === nothing ? nothing : ctor[2]
    def = lookup(name)
    if def === nothing
        out["unread"] = "no definition of `$(name)` in the DWARF"
        return out
    end
    for tp in get(def, "template_params", ())
        if get(tp, "kind", "") == "type"
            out["char_type"] = String(get(tp, "type", ""))
            break
        end
    end
    off(v) = v isa Integer ? Int(v) : (s = string(v); startswith(s, "0x") ? parse(Int, s[3:end]; base = 16) : parse(Int, s))
    leaves = Any[]
    reason = Ref{Union{Nothing,String}}(nothing)
    function walk(d, base::Int, prefix::String, depth::Int)
        depth > 12 && (reason[] = "nesting deeper than 12 in `$(name)`"; return false)
        for m in get(d, "members", ())
            mname = String(get(m, "name", ""))
            o = get(m, "offset", nothing)
            if o === nothing
                haskey(m, "data_bit_offset") || continue          # a static: no storage
                push!(leaves, Dict{String,Any}("path" => prefix * mname, "c_type" => String(get(m, "c_type", "")),
                                               "bit_size" => m["bit_size"],
                                               "data_bit_offset" => 8base + Int(m["data_bit_offset"])))
                continue
            end
            at = base + off(o)
            ct = String(get(m, "c_type", ""))
            held = strip(replace(ct, r"^(?:(?:const|volatile)\s+)+" => ""))
            sub = (occursin(r"[*&\[]", _outside_template_args(held)) ? nothing : lookup(held))
            if sub !== nothing && haskey(sub, "members")
                walk(sub, at, prefix * mname * ".", depth + 1) || return false
            else
                push!(leaves, Dict{String,Any}("path" => prefix * mname, "c_type" => ct, "offset" => at,
                                               "size" => get(m, "size", 0)))
            end
        end
        for b in get(d, "base_classes", ())
            get(b, "empty", false) == true && continue
            bt = String(get(b, "type", ""))
            get(b, "virtual", false) == true && (reason[] = "virtual base `$(bt)`"; return false)
            bd = lookup(bt)
            # No definition and one byte: no data member (one would have given
            # it a definition), so an empty base, whatever it derives from
            # (`allocator<char>` over `__new_allocator<char>`). No definition
            # and no size either (libstdc++ declares `extern template class
            # allocator<char>`, and clang then emits it as a declaration-only
            # DIE under some header versions: Ubuntu 24.04's libstdc++ 13,
            # 2026-11-14): the base contributes no bytes when the derived
            # record's own members already cover its whole byte_size, since
            # base and member subobjects never overlap. A base the members
            # leave room for may carry data we cannot see, and stays a refusal.
            bd === nothing && byte_size_of(bt) == 1 && continue
            bd === nothing && _members_cover_record(d) && continue
            bd === nothing && (reason[] = "base `$(bt)` has no definition and `$(name)`'s members leave room for it"; return false)
            walk(bd, base + off(get(b, "offset", 0)), prefix, depth + 1) || return false
        end
        return true
    end
    if walk(def, 0, "", 0)
        out["leaves"] = leaves
    else
        out["unread"] = something(reason[], "unreadable")
    end
    return out
end

"""
    _function_pointer_typedef(type_ref, type_refs) -> Union{String,Nothing}

The typedef a function-pointer parameter is declared with:
`ggml_backend_eval_callback` for `ggml_backend_eval_callback callback`. Of a
chain (`typedef cb_t my_cb;`) the innermost, the typedef of the pointer itself,
which is the one a header spells `typedef R (*name)(…);`. `nothing` for a
parameter declared without one (`void (*cb)(int)`).

`resolve_type` resolves through the typedef, so the parameter's `c_type`
(`function_ptr(R; P1, P2)*`) has lost the name. A docstring's `@cfunction`
example comes from this typedef's header signature (`_callback_typedef_signature`),
and from that `c_type` when the headers spell none (`_dwarf_callback_signature`).
"""
function _function_pointer_typedef(type_ref, type_refs)
    ref = type_ref
    name = nothing
    for _ in 1:16                                  # typedef chains are short; no cycles
        t = get(type_refs, ref, nothing)
        t isa Dict || return nothing
        kind = get(t, "kind", nothing)
        if kind == "pointer"
            return name
        elseif kind == "typedef"
            n = get(t, "name", nothing)
            n isa AbstractString && !isempty(n) && (name = String(n))
        elseif !(kind in ("const", "volatile", "restrict"))
            return nothing
        end
        ref = get(t, "target", nothing)
        ref === nothing && return nothing
    end
    return nothing
end

"""
    _vector_size(type_ref, type_refs; base_sizes, arrays = false) -> Union{Int,Nothing}

The bytes of the SIMD vector (`__attribute__((vector_size))`, `__m128`) a DWARF
type reference names: an array DIE with `DW_AT_GNU_vector`, reached through
typedefs and cv-qualifiers, and with `arrays` also through ordinary arrays (a
member `v4f lanes[2]` holds two 16-byte vectors). `nothing` for anything else.

DWARF spells a vector as the array of its lanes, and `resolve_type` keeps that
spelling (`float[4]`), but a vector is aligned to its size, which no attribute
states, and x86-64 SysV classifies it by its size, never lane by lane: `<1 x
float>` is INTEGER, `<2 x int>` SSE, `<4 x float>` one SSE register. The
metadata records it as `vector_size` on the member, parameter or return.
"""
function _vector_size(type_ref, type_refs; base_sizes::AbstractDict = Dict{String,Int}(),
                      arrays::Bool = false)::Union{Int,Nothing}
    ref = type_ref
    for _ in 1:16                                  # typedef chains are short; no cycles
        t = get(type_refs, ref, nothing)
        t isa Dict || return nothing
        kind = get(t, "kind", nothing)
        if kind == "array"
            if get(t, "vector", false) == true
                n = _die_byte_size(ref, type_refs; base_sizes)
                return n > 0 ? n : nothing
            end
            arrays || return nothing
            ref = get(t, "element_type", nothing)
        elseif kind in ("typedef", "const", "volatile")
            ref = get(t, "target", nothing)
        else
            return nothing
        end
        ref === nothing && return nothing
    end
    return nothing
end

"""
    _merge_record_facts!(facts, name, entry)

Records are keyed by name, like `struct_defs`, and one name can have several
definitions — the same type in many compile units, which agree, or two scopes
reusing a name, which may not. Agreement keeps the entry. Disagreement marks it
`"conflict"`, and a conflicted record is never treated as known.
"""
function _merge_record_facts!(facts::Dict{String,Dict{String,Any}}, name::String,
                              entry::Dict{String,Any})
    prev = get(facts, name, nothing)
    if prev === nothing
        facts[name] = entry
    elseif get(prev, "conflict", false) != true &&
           any(k -> get(prev, k, nothing) != get(entry, k, nothing), ("byte_size", "pass", "empty"))
        prev["conflict"] = true
    end
    return facts
end

"""
    _die_order(type_refs) -> Vector{String}

The DIE offsets among `type_refs`' keys, in dump order (numeric offset order).

`type_refs` is a `Dict` keyed by offset strings, so iterating it gives the hash
order of those strings. That order changes with the Julia release (1.12 and 1.13
hash strings differently) and with any shift of the offsets, which every upstream
edit causes. A pass in which one DIE can win over another iterates this instead.
Keys that are not offsets (`"last_tag_offset"`) are left out.
"""
function _die_order(type_refs::AbstractDict)::Vector{String}
    keyed = Tuple{UInt64,String}[]
    sizehint!(keyed, length(type_refs))
    for k in keys(type_refs)
        (k isa AbstractString && startswith(k, "0x")) || continue
        v = tryparse(UInt64, SubString(k, 3); base = 16)
        v === nothing || push!(keyed, (v, String(k)))
    end
    sort!(keyed; by = first)
    return String[k for (_, k) in keyed]
end

"""
    _canonical_text(x) -> String

`x` as text, with every `Dict`'s keys sorted: equal contents give equal text on
every Julia release, whatever order the keys were inserted in. The content that
`_canonical_definition` orders definitions by.
"""
_canonical_text(x::AbstractDict) =
    string("{", join((string(repr(string(k)), "=>", _canonical_text(x[k]))
                      for k in sort!(collect(keys(x)); by = string)), ","), "}")
_canonical_text(x::AbstractVector) = string("[", join((_canonical_text(v) for v in x), ","), "]")
_canonical_text(x) = repr(x)

"""
    _layout_signature(def) -> String

What two definitions under one name must share to be one type, as opposed to
spellings of it. For a record: kind, size, alignment, where each placed member
sits and how big it is, and where each base sits. Member names, member types and
the members with no place (a static, a vtable home's `__clang_vtable`) are left
out: those are what C and C++ compile units spell differently. For an enum: size
and enumerators. Anything else has no layout and gives `""`.
"""
function _layout_signature(def::AbstractDict)::String
    if haskey(def, "members")
        placed = [(get(m, "offset", nothing), get(m, "size", nothing),
                   get(m, "bit_size", nothing), get(m, "data_bit_offset", nothing))
                  for m in def["members"]
                  if get(m, "offset", nothing) !== nothing || haskey(m, "data_bit_offset")]
        bases = [(get(b, "offset", 0), get(b, "virtual", false)) for b in get(def, "base_classes", [])]
        return _canonical_text(Any[get(def, "kind", ""), get(def, "byte_size", nothing),
                                   get(def, "alignment", nothing), placed, bases])
    elseif haskey(def, "enumerators")
        return _canonical_text(Any[get(def, "byte_size", nothing),
                                   [(get(e, "name", nothing), get(e, "value", nothing))
                                    for e in def["enumerators"]]])
    end
    return ""
end

"""
    _scope_of(parent, type_refs) -> Union{String,Nothing}

The `::`-joined names of the namespaces and records enclosing a DIE whose
`scope_parent` is `parent`: `""` at global scope (`parent === nothing`),
`"boost::core"`, `"std::__1"`, `"Outer"` for a nested record. `nothing` when
any link of the chain is unknown (`missing`) or is not a namespace or record,
so a partial chain is never reported as the whole one: `"chrono"` for
`std::__1::chrono` would read as a library's own namespace.
"""
function _scope_of(parent, type_refs::AbstractDict)
    parts = String[]
    p = parent
    seen = Set{Any}()
    while p !== nothing
        (p === missing || p in seen) && return nothing
        push!(seen, p)
        die = get(type_refs, p, nothing)
        die isa AbstractDict || return nothing
        kind = get(die, "kind", nothing)
        kind in ("namespace", "struct", "class", "union") || return nothing
        name = get(die, "name", nothing)
        pushfirst!(parts, kind == "namespace" && (name === nothing || isempty(string(name))) ?
                          "(anonymous namespace)" : string(name))
        p = get(die, "scope_parent", missing)
    end
    return join(parts, "::")
end

"""
    _record_provenance!(def, scopes_or_defs, decl_files) -> def

Where the definitions merged under one bare name were declared, on the kept
entry: `scopes`, the sorted distinct enclosing scopes (`""` at global scope;
each definition's `scope`, or the strings given), and `decl_files`, the sorted
distinct `/`-separated header paths. Neither key is written when nothing is
known, so a reader that needs them fails safe. Graph mode's cross-library
type identity reads `scopes` (two libraries' `Config` are one type only when
their scopes meet) and reports `decl_files`; nothing else does.
"""
function _record_provenance!(def::AbstractDict, scopes_or_defs::AbstractVector, decl_files::AbstractVector)
    scopes = String[]
    for x in scopes_or_defs
        if x isa AbstractDict
            haskey(x, "scope") && push!(scopes, String(x["scope"]))
        else
            push!(scopes, String(x))
        end
    end
    isempty(scopes) || (def["scopes"] = sort!(unique!(scopes)))
    isempty(decl_files) || (def["decl_files"] = sort!(unique!(String[String(f) for f in decl_files])))
    return def
end

"""
    _canonical_definition(defs) -> Dict

The definition `parse_dwarf_dump` keeps when one bare name has several: the one
whose `_canonical_text` sorts first. It depends on the definitions alone, not on
the order they were found in, so neither the Julia release, nor the DIE offsets,
nor the order of the compile units can move it.

Definitions that agree on `_layout_signature` are one type, differently spelled,
and any of them is right. Definitions that disagree are distinct types sharing a
name (a pimpl `impl` per class), and a bare-name key can describe only one: the
kept one is marked `"conflict"`. It is still the layout every reference by that
name gets, as it was before, so no call may pass it by value
(`FunctionGen.conflicted_within`).

A conflicted entry also records `"conflict_sizes"`, the distinct byte sizes of
all its definitions in ascending order. An enum crosses a call as an integer of
its width, so one whose definitions differ only in their enumerators still
crosses correctly; the sizes say whether they do.
"""
function _canonical_definition(defs::AbstractVector)
    length(defs) == 1 && return defs[1]
    best = defs[argmin(map(_canonical_text, defs))]
    if length(unique(map(_layout_signature, defs))) > 1
        best["conflict"] = true
        best["conflict_sizes"] = sort!(unique!(Int[s for s in map(_definition_byte_size, defs)
                                                   if s !== nothing]))
    end
    return best
end

# A record's or enum's `byte_size` (`"0x18"`) as an Int; `nothing` without one.
"""
    _members_cover_record(def) -> Bool

Do `def`'s own byte-offset members tile `[0, byte_size)` without a gap? Then no
base subobject has room: every base is empty (the empty-base optimisation),
whatever the DWARF says or fails to say about it. Bitfields are not counted.
"""
function _members_cover_record(def::AbstractDict)::Bool
    total = _definition_byte_size(def)
    total === nothing && return false
    iv = Tuple{Int,Int}[]
    for m in get(def, "members", ())
        o = get(m, "offset", nothing)
        o === nothing && continue
        sz = get(m, "size", 0)
        sz isa Integer && sz > 0 || continue
        s = string(o)
        a = startswith(s, "0x") ? parse(Int, s[3:end]; base = 16) : parse(Int, s)
        push!(iv, (a, a + Int(sz)))
    end
    isempty(iv) && return false
    sort!(iv)
    cur = 0
    for (a, b) in iv
        a > cur && return false
        cur = max(cur, b)
    end
    return cur >= total
end

"""
    _memberless_record_entry(name, type_info) -> Bool

Whether a library record with no member of its own gets a `struct_definitions`
entry: a definition (it has a `byte_size`) of more than zero bytes (GNU C's
empty struct has nothing to size), not an anonymous record (it exists to be
embedded) and not a lambda's closure type (nothing can name it). Every such
record does (2026-11-14): see the parser's note at its call.
"""
function _memberless_record_entry(name::AbstractString, type_info::AbstractDict)::Bool
    return something(_definition_byte_size(type_info), 0) > 0 &&
           get(type_info, "anonymous", false) != true &&
           !startswith(name, "(lambda")
end

function _definition_byte_size(def::AbstractDict)
    bs = get(def, "byte_size", nothing)
    bs === nothing && return nothing
    s = string(bs)
    return startswith(s, "0x") ? tryparse(Int, s[3:end]; base = 16) : tryparse(Int, s)
end

"""
    _by_value_record_name(c_type) -> Union{String,Nothing}

The record name a by-value parameter or return names, or `nothing` for a
pointer, reference, function pointer, array or `void`. Only TOP-LEVEL cv is
removed: `fstring<const char *>` keeps the `const` inside its argument list,
which is part of the name the record is keyed under.
"""
function _by_value_record_name(c_type::AbstractString)
    t = strip(c_type)
    t = replace(t, r"^(?:(?:const|volatile)\s+)+" => "")
    t = replace(t, r"(?:\s+(?:const|volatile))+$" => "")
    (isempty(t) || t in ("void", "unknown")) && return nothing
    occursin(r"[*&(\[]", t) && return nothing
    return String(t)
end

# The key `t` is filed under in `table`: `t`, or `t` with leading scopes cut at
# depth 0 of its template arguments (`_scope_suffixes`), since tables are keyed
# by the bare DWARF name; `nothing` when there is none.
function _key_in(t::AbstractString, table::AbstractDict)
    for c in _scope_suffixes(t)
        haskey(table, c) && return c
    end
    return nothing
end

_member_offset_int(v) = v isa Integer ? Int(v) :
    (s = string(v); startswith(s, "0x") ? parse(Int, s[3:end]; base = 16) : parse(Int, s))

"""
    _laid_out_from_bases(def, lookup, own_members) -> Dict

A copy of the toolchain record `def` whose members are `own_members(def)` plus
every member of every non-empty, non-virtual base `lookup(type)` finds, at
depth, each at the base's offset plus its own (a bitfield's `data_bit_offset`
moved by as many bytes), sorted by offset; its `base_classes` keeps the empty
bases and any base `lookup` does not know, as before. A name two members share
is the first's in offset order and `<name>_1`, `<name>_2` after it. When a base
deeper down cannot be looked up, the copy keeps `def`'s own members and bases:
a layout missing bytes is worse than the old refusal-free shape.
"""
function _laid_out_from_bases(def::AbstractDict, lookup, own_members)
    out = Dict{String,Any}(def)
    shifted(m, d) = begin
        m2 = Dict{String,Any}(m)
        o = get(m, "offset", nothing)
        o === nothing || (m2["offset"] = "0x" * string(_member_offset_int(o) + d; base = 16))
        haskey(m, "data_bit_offset") && (m2["data_bit_offset"] = Int(m["data_bit_offset"]) + 8d)
        m2
    end
    function flatten(d::AbstractDict, at::Int)
        ms = Any[shifted(m, at) for m in own_members(d)]
        for b in get(d, "base_classes", ())
            get(b, "empty", false) == true && continue
            get(b, "virtual", false) == true && return nothing
            bd = lookup(String(get(b, "type", "")))
            bd === nothing && return nothing
            sub = flatten(bd, at + _member_offset_int(get(b, "offset", 0)))
            sub === nothing && return nothing
            append!(ms, sub)
        end
        return ms
    end

    members = Any[shifted(m, 0) for m in own_members(def)]
    kept = Any[]
    for b in get(def, "base_classes", ())
        bd = (get(b, "empty", false) == true || get(b, "virtual", false) == true) ? nothing :
             lookup(String(get(b, "type", "")))
        if bd === nothing
            push!(kept, b)
            continue
        end
        sub = flatten(bd, _member_offset_int(get(b, "offset", 0)))
        if sub === nothing
            out["members"] = own_members(def)
            return out
        end
        append!(members, sub)
    end
    # Offset order; a bitfield by its bits, a static (no offset, no bits) last.
    pos(m) = get(m, "offset", nothing) !== nothing ? 8 * _member_offset_int(m["offset"]) :
             haskey(m, "data_bit_offset") ? Int(m["data_bit_offset"]) : typemax(Int)
    sort!(members; by = pos, alg = Base.Sort.DEFAULT_STABLE)
    seen = Dict{String,Int}()
    for m in members
        n = String(get(m, "name", ""))
        k = get(seen, n, 0)
        seen[n] = k + 1
        k == 0 || (m["name"] = "$(n)_$(k)")
    end
    out["members"] = members
    if isempty(kept)
        delete!(out, "base_classes")
    else
        out["base_classes"] = kept
    end
    return out
end

"""
    _promote_toolchain_records!(struct_defs, toolchain_defs, seed_types; empty_records) -> Vector{String}

Put back into `struct_defs` the toolchain records the library holds or passes by
value, and return their keys, sorted. Each one is marked `"toolchain" => true`.

`parse_dwarf_dump`'s provenance gate keeps a record declared in a toolchain header
out of `struct_defs`: it is not the library's API. But a function that returns
`std::pair<double, double>` needs that pair's layout to be called, and without it
both generators emit an ABI trap (`FunctionGen.by_value_crossing` says
`:opaque`). So does a library record holding one by value, whose member was a
byte blob and whose by-value crossings were traps. `record_abi` carried the
pair's size and pass convention all along; only its members were missing.

A record is reached from `seed_types` (the by-value returns, parameters and
globals of what the library exports; `parse_dwarf_dump` gates them on the
dynamic symbol table) and from what a reached record holds by value, through
members (arrays included) and bases, whether that record is the library's own
(`Span { std::pair<int, int> range; … }` returned by value) or one promoted
here. A library record nothing exported crosses by value promotes nothing. It is promoted only
when its whole by-value closure has a layout: every member a scalar (a
`complex float`/`double`/`long double` included: the parser reads the width off
the base type's `byte_size`, `_complex_base_name`), pointer, reference, enum,
STL container (held as bytes, as in a library record), library record or
promotable toolchain record, and every base one of those last two or marked
`"empty"` (`std::pair`'s `__pair_base`). Anything else stays out, and its
crossings stay traps: a virtual base, a conflicted entry, a member nothing can
lay out.

**A toolchain record is laid out from its bases** (`_laid_out_from_bases`): a
promoted one gets every member of every non-empty, non-virtual base, at depth,
at the base's offset plus the member's, in place of that base, and its
`base_classes` keeps only the empty ones. libstdc++'s `std::tuple` and
`std::optional` have no member of their own (the parser gives such a toolchain
record an entry for this), so they had no layout; and every reader of a
definition (StructGen, the SysV classifier, both generators) reads members at
offsets, so a flattened record needs nothing new downstream. A name two
members share (each `_Head_base`'s `_M_head_impl`) is kept by the first in
offset order; the later ones are `<name>_1`, `<name>_2`. A union drops a member
whose type is a 1-byte empty record (`empty_records`; `_Storage`'s
`_Empty_byte` beside a `std::optional`'s value): it overlaps the others and
holds no bytes they do not.

**A toolchain enum with no enumerators** (`std::byte`, `std::align_val_t`;
`toolchain_defs["__enum__<name>"]`) comes back the same way, as
`struct_defs["__enum__<name>"]` marked `"toolchain"`: reached by value it is
the integer the call crosses, and a member of that type is complete. It is
also put back where a seed or a walked record names it behind a pointer or
reference, which a record is not: the enum is the type that pointer is typed
with (`std::byte*` is `Ptr{byte}`), and it holds nothing whose layout could be
missing. One no exported function, exported global or reached record names
stays out. (A toolchain enum WITH enumerators was never set aside.)

Never promoted: an STL container, which keeps its own path (pointer in, byte blob
out; `stl_container_size`), and a name on `INTERNAL_TYPE_BLOCKLIST`, which no
generator emits. A library record of the same name keeps its entry.
"""
function _promote_toolchain_records!(struct_defs::AbstractDict, toolchain_defs::AbstractDict,
                                     seed_types; empty_records = Set{String}())::Vector{String}
    isempty(toolchain_defs) && return String[]
    promotable(k) = !(k in INTERNAL_TYPE_BLOCKLIST) && stl_container_size(k) == 0
    is_union(def) = get(def, "kind", "struct") == "union"
    # The members a definition contributes: all of them, less a union's empty
    # record members (see above).
    empty_record(t) = any(c -> c in empty_records, _scope_suffixes(String(t)))
    own_members(def) = is_union(def) ?
        [m for m in get(def, "members", ()) if !empty_record(get(m, "c_type", ""))] :
        collect(get(def, "members", ()))

    # The type `t` holds in place: `nothing` for a pointer, reference or function
    # pointer (it holds an address), else the spelling with top-level cv and
    # array extents removed.
    function held(t::AbstractString)
        s = strip(replace(String(t), r"^(?:(?:const|volatile)\s+)+" => ""))
        s = strip(replace(s, r"(?:\s+(?:const|volatile))+$" => ""))
        m = match(r"^(.*?)\s*((?:\[\d*\])+)$", s)
        m === nothing || (s = strip(String(m.captures[1])))
        # The declarators outside the template arguments: `pair<int*, bool>`
        # holds a record, `pair<int, bool>*` an address.
        (occursin(r"[*&(]", _outside_template_args(s)) || startswith(s, "function_ptr")) && return nothing
        return String(s)
    end

    verdict = Dict{String,Bool}()
    function complete(key::String, stack::Vector{String})
        haskey(verdict, key) && return verdict[key]
        key in stack && return false   # nothing holds itself by value; a cycle is a misparse
        push!(stack, key)
        def = toolchain_defs[key]
        ok = get(def, "conflict", false) != true &&
             all(m -> holds_ok(String(get(m, "c_type", "")), stack), own_members(def)) &&
             all(b -> get(b, "virtual", false) != true &&
                      (get(b, "empty", false) == true || base_ok(String(get(b, "type", "")), stack)),
                 get(def, "base_classes", ()))
        pop!(stack)
        verdict[key] = ok
        return ok
    end
    function holds_ok(t::String, stack)
        e = held(t)
        e === nothing && return true
        (isempty(e) || e in ("unknown", "Any", "void")) && return false
        get_type_size(e) > 0 && return true
        # `_complex_base_name`'s spellings, which only a sized base type gets.
        e in ("complex float", "complex double", "complex long double") && return true
        cuts = _scope_suffixes(e)
        any(c -> haskey(struct_defs, "__enum__" * c) || haskey(toolchain_defs, "__enum__" * c), cuts) && return true
        stl_container_size(e) > 0 && return true
        _key_in(e, struct_defs) !== nothing && return true
        tk = _key_in(e, toolchain_defs)
        return tk !== nothing && promotable(tk) && complete(tk, stack)
    end
    function base_ok(t::String, stack)
        isempty(t) && return false
        _key_in(t, struct_defs) !== nothing && return true
        tk = _key_in(t, toolchain_defs)
        return tk !== nothing && promotable(tk) && complete(tk, stack)
    end

    promoted = String[]
    walked = Set{String}()
    # A record in `struct_defs` reached by value: what it holds is reached too.
    function walk(key::String)
        key in walked && return
        push!(walked, key)
        def = struct_defs[key]
        for m in get(def, "members", ())
            reach(String(get(m, "c_type", "")))
        end
        for b in get(def, "base_classes", ())
            reach(String(get(b, "type", "")))
        end
    end
    # A toolchain enum with no enumerators that `e` names (`std::byte`): put
    # back whole, since an enum holds nothing. `true` when `e` is an enum.
    function reach_enum(e::AbstractString)
        for c in _scope_suffixes(e)
            ek = "__enum__" * c
            haskey(struct_defs, ek) && return true
            (haskey(toolchain_defs, ek) && !(c in INTERNAL_TYPE_BLOCKLIST)) || continue
            def = Dict{String,Any}(toolchain_defs[ek])
            def["toolchain"] = true
            struct_defs[ek] = def
            push!(promoted, ek)
            return true
        end
        return false
    end
    function reach(t::AbstractString)
        e = held(t)
        if e === nothing
            # An address. A record behind it is not held, and stays out; an
            # enum is the type the pointer is typed with (`std::byte*` is
            # `Ptr{byte}`), which the wrapper can only spell if it emits it.
            s = replace(String(t), r"\b(?:const|volatile)\b" => "")
            s = replace(s, r"(?:\[\d*\])+\s*$" => "")
            occursin('(', s) || reach_enum(strip(replace(s, r"[*&\s]+$" => "")))
            return
        end
        isempty(e) && return
        stl_container_size(e) > 0 && return
        lk = _key_in(e, struct_defs)
        lk === nothing || return walk(lk)                # the library's own, or done
        tk = _key_in(e, toolchain_defs)
        tk === nothing && reach_enum(e) && return
        (tk === nothing || !promotable(tk) || !complete(tk, String[])) && return
        def = _laid_out_from_bases(toolchain_defs[tk], k -> begin
                  bk = _key_in(k, toolchain_defs)
                  bk === nothing ? nothing : toolchain_defs[bk]
              end, own_members)
        def["toolchain"] = true
        struct_defs[tk] = def
        push!(promoted, tk)
        walk(tk)
    end

    for t in seed_types
        reach(String(t))
    end
    return sort!(promoted)
end

"""
Extract compilation metadata from source files and binary.
This is the core of automatic wrapper generation!
"""
function extract_compilation_metadata(config::RepliBuildConfig, source_files::Vector{String},
                                      binary_path::String;
                                      nounwind::Union{Nothing,Set{String}}=nothing,
                                      byval::Union{Nothing,Dict{String,Vector{Dict{String,Any}}}}=nothing,
                                      inline_constructors::Union{Nothing,AbstractDict}=nothing,
                                      inline_methods::Union{Nothing,AbstractDict}=nothing,
                                      default_arguments::Union{Nothing,AbstractDict}=nothing)::Dict{String,Any}
    # Extract compilation metadata
    symbols = extract_symbols_from_binary(binary_path)

    # Extract return types and struct definitions from DWARF debug info (if available)
    # What the loader can resolve: a toolchain record keeps its layout only for an
    # exported function's or global's by-value crossing (`_promote_toolchain_records!`).
    # Less what `[wrap] exclude_symbols` drops: wrap will not bind it, so it seeds
    # nothing (`_without_excluded_symbols`). `nothing` ("could not look") stays
    # `nothing`, and the parser's fallback seeds every non-std function.
    exported = _runtime_defined_names(binary_path)
    exported === nothing ||
        (exported = _without_excluded_symbols(exported, symbols, config.wrap.exclude_symbols))
    (dwarf_return_types, struct_defs, global_vars, typedef_table, record_abi) =
        extract_dwarf_return_types(binary_path; exported = exported)

    # Types are keyed by bare name, so distinct types sharing one (a nested
    # `impl` per class) keep one definition between them; say which names.
    conflicted = sort!([replace(k, r"^__enum__" => "enum ") for (k, v) in struct_defs
                        if get(v, "conflict", false) == true])
    isempty(conflicted) || println("  metadata: ", length(conflicted) == 1 ? "1 type name has" :
                                   "$(length(conflicted)) type names have", " distinct definitions, ",
                                   "one kept for each: ", join(first(conflicted, 8), ", "),
                                   length(conflicted) > 8 ? ", …" : "")
    toolchain = sort!([replace(k, r"^__enum__" => "enum ") for (k, v) in struct_defs
                       if get(v, "toolchain", false) == true])
    isempty(toolchain) || println("  metadata: ", length(toolchain), " toolchain record(s) held or passed ",
                                  "by value, layout kept: ", join(first(toolchain, 6), ", "),
                                  length(toolchain) > 6 ? ", …" : "")

    # Collect struct/enum names for type resolution in function signatures
    sig_struct_names = Set{String}()
    sig_enum_names = Set{String}()
    for (name, info) in struct_defs
        if startswith(name, "__enum__")
            enum_name = replace(name, "__enum__" => "")
            push!(sig_enum_names, enum_name)
        else
            push!(sig_struct_names, name)
        end
    end

    # A default-argument function (`_default_argument_plan`) takes the identity
    # of the member it calls with fewer arguments: its symbol demangles as
    # `Class::f(prefix)`, so name, class and every derivation downstream are
    # the member's own.
    bound_defaults = default_arguments === nothing ? Dict{String,Any}() :
        Dict{String,Any}(String(r["shim"]) => r for r in get(default_arguments, "bound", Any[])
                         if r isa AbstractDict && haskey(r, "shim"))
    for sym in symbols
        r = get(bound_defaults, String(get(sym, "mangled", "")), nothing)
        r === nothing || (sym["demangled"] = String(r["demangled"]))
    end

    # Parse function signatures (basic type inference from symbol names)
    functions = parse_function_signatures(symbols, sig_struct_names, sig_enum_names)

    # Merge DWARF return types and parameters into function metadata (overrides inference)
    address_of = Dict{String,String}(String(s["mangled"]) => String(get(s, "address", "")) for s in symbols)
    by_address = Dict{String,Vector{String}}()      # the exported functions at each address
    for (name, address) in address_of
        push!(get!(() -> String[], by_address, address), name)
    end
    for func in functions
        mangled = func["mangled"]
        # Constructors and destructors are keyed differently on the two sides:
        # the symbol table carries C1/C2/C3 and D0/D1/D2, and clang writes the
        # unified C4/D4 into DW_AT_linkage_name. See `_structor_dwarf_key`.
        dwarf_key = haskey(dwarf_return_types, mangled) ? mangled :
                    _structor_dwarf_key(mangled, dwarf_return_types)
        # A symbol no DIE describes is an alias when a described function has
        # its address, and then has that function's signature
        # (`_alias_signature_key`). Any other stays `inferred`, which both
        # generators refuse to call (`Wrapper._lacks_debug_info`).
        if dwarf_key === nothing
            twin = _alias_signature_key(mangled, get(address_of, mangled, ""), by_address, dwarf_return_types)
            twin === nothing || ((func["alias_of"], dwarf_key) = twin)
        end
        if dwarf_key !== nothing
            dwarf_info = dwarf_return_types[dwarf_key]

            # Merge return type (only the return type fields, not parameters)
            func["return_type"] = Dict(
                "c_type" => get(dwarf_info, "c_type", "void"),
                "julia_type" => get(dwarf_info, "julia_type", "Cvoid"),
                "size" => get(dwarf_info, "size", 0)
            )
            haskey(dwarf_info, "vector_size") && (func["return_type"]["vector_size"] = dwarf_info["vector_size"])
            func["return_type_source"] = "dwarf"
            
            # Merge is_vararg
            func["is_vararg"] = get(dwarf_info, "is_vararg", false)

            # Present only when a definition DIE was seen. `true` is an instance
            # method, `false` is static (no receiver). Absent → the gates keep
            # their class-table heuristic.
            if haskey(dwarf_info, "has_object_pointer")
                func["has_object_pointer"] = dwarf_info["has_object_pointer"]
            end

            # Merge parameters if available from DWARF (at function level, not in return_type)
            if haskey(dwarf_info, "parameters") && !isempty(dwarf_info["parameters"])
                func["parameters"] = dwarf_info["parameters"]
                func["parameters_source"] = "dwarf"
            else
                func["parameters_source"] = "inferred"
            end
        else
            func["return_type_source"] = "inferred"
            func["parameters_source"] = "inferred"
            func["is_vararg"] = false
        end
    end
    # Say which symbols had no DIE: the aliases that took a signature, and the
    # ones the wrapper will refuse.
    let aliases = sort!(["$(f["mangled"]) = $(f["alias_of"])" for f in functions if haskey(f, "alias_of")]),
        refused = sort!([String(f["mangled"]) for f in functions if f["return_type_source"] == "inferred"])
        isempty(aliases) || println("  metadata: ", length(aliases), " alias(es) with no debug information of their ",
                                    "own take their aliasee's signature: ", join(first(aliases, 6), ", "),
                                    length(aliases) > 6 ? ", …" : "")
        isempty(refused) || println("  metadata: ", length(refused), " exported function(s) have no debug ",
                                    "information, and the wrapper will refuse to call them: ",
                                    join(first(refused, 6), ", "), length(refused) > 6 ? ", …" : "")
    end

    # "Can an exception leave this function?" — the question the tier router asks
    # (DispatchLogic.is_ccall_safe) and the thunk generator asks (FunctionGen:
    # try_call vs call). A function counts as noexcept only when BOTH:
    #
    #   * its bare name was declared `noexcept` somewhere in the sources — the
    #     candidate set (`_scan_noexcept_functions`), and
    #   * its own mangled DEFINITION is `nounwind` in the IR the binary was linked
    #     from (`_nounwind_definitions`) — the proof.
    #
    # The proof removes every false positive of the name scan: a bare-name
    # collision, `noexcept(false)`, a throwing constructor of a class with a
    # noexcept move constructor. A definition that can unwind is never `nounwind`.
    # The candidate set keeps the routing from GROWING: `nounwind` alone would also
    # send every function the optimizer merely proved non-throwing to Tier 3, and
    # the Tier-3 emitter has type gaps the thunks tolerate (box2d `AddType`'s
    # nested enum, fmt's inferred constructor signatures). Widening is a separate
    # change, after those gaps are closed. No IR (ingest mode) means nothing is
    # noexcept, so all C++ goes to Tier 2, the router's documented safe default.
    # A function with C linkage in a C++ library is routed by the proof alone
    # (`nounwind`, which `_mark_noexcept!` records beside `is_noexcept`): ccall
    # is where those already were, so that only narrows.
    candidates = config.wrap.language == :c ? Set{String}() :
                 _scan_noexcept_functions(source_files, get_include_dirs(config))
    _apply_default_argument_identity!(functions, bound_defaults)
    _mark_noexcept!(functions, candidates, nounwind)
    _mark_byval_params!(functions, byval)

    # The inline constructors the library declares and does not define, for the
    # forced unit (`generate_inline_constructors`): computed from THIS binary on
    # the first extraction, and on the second carried over from the first with
    # each forced row's outcome (`emitted`: the symbol table now defines it; the
    # unit's guard declined the rest). A C bucket has none.
    plan = if inline_constructors !== nothing
        p = Dict{String,Any}(inline_constructors)
        defined = Set{String}(String(get(s, "mangled", "")) for s in symbols)
        p["forced"] = Any[(r = Dict{String,Any}(f);
                           r["emitted"] = any(in(defined), _constructor_variants(String(r["symbol"])));
                           r) for f in get(p, "forced", Any[])]
        p
    elseif config.wrap.language == :cpp
        _inline_constructor_plan(struct_defs, record_abi, symbols, config.wrap.exclude_symbols)
    else
        Dict{String,Any}("forced" => Any[], "skipped" => Any[])
    end
    # The same for the other inline member functions (`_inline_method_plan`):
    # `emitted` says the symbol table now defines the member.
    mplan = if inline_methods !== nothing
        p = Dict{String,Any}(inline_methods)
        defined = Set{String}(String(get(s, "mangled", "")) for s in symbols)
        p["forced"] = Any[(r = Dict{String,Any}(f); r["emitted"] = String(r["symbol"]) in defined; r)
                          for f in get(p, "forced", Any[])]
        p
    elseif config.wrap.language == :cpp
        _inline_method_plan(struct_defs, record_abi, symbols, config.wrap.exclude_symbols)
    else
        Dict{String,Any}("forced" => Any[], "skipped" => Any[])
    end

    # The calls with default arguments left out: on the first extraction the
    # candidates the unit will try, on the second the ones it compiled, each
    # with whether its function is in the symbol table and bound.
    dargs = if default_arguments !== nothing
        bound_now = Set{String}(String(f["mangled"]) for f in functions if haskey(f, "default_arguments"))
        Dict{String,Any}("candidates" => Any[],
                         "bound" => Any[(r = Dict{String,Any}(f); r["emitted"] = String(get(r, "shim", "")) in bound_now; r)
                                        for f in get(default_arguments, "bound", Any[])])
    elseif config.wrap.language == :cpp
        _default_argument_plan(struct_defs, record_abi, symbols, config.wrap.exclude_symbols)
    else
        Dict{String,Any}("candidates" => Any[], "bound" => Any[])
    end

    # Build type registry (basic types + inferred types)
    type_registry = build_type_registry(functions)

    # Gather compiler info. The C bucket emits IR with Julia's JLL clang and links
    # via in-process libLLVM — both are Base.libllvm_version (the JLL-clang ==
    # libllvm_version invariant), NOT the system llvm-config/clang. Only the
    # C++/external-fallback path uses the system toolchain. Recording system
    # llvm-config for an in-process C build is misleading — it falsely implies a
    # version skew (and once sent a debugging session chasing one), so report the
    # toolchain that actually produced the IR.
    uses_julia_llvm = config.wrap.language == :c && !config.link.fallback

    llvm_version = if uses_julia_llvm
        string(Base.libllvm_version)
    else
        try
            (out, _) = BuildBridge.execute("llvm-config", ["--version"])
            strip(out)
        catch
            "unknown"
        end
    end

    clang_version = if uses_julia_llvm
        "clang $(Base.libllvm_version) (Julia JLL)"
    else
        try
            compiler_cmd = config.wrap.language == :c ? "clang" : "clang++"
            (out, _) = BuildBridge.execute(compiler_cmd, ["--version"])
            first(split(out, '\n'))
        catch
            "unknown"
        end
    end

    # Extract STL method symbols if templates are configured
    stl_methods = Dict{String,Any}()
    if !isempty(config.types.templates)
        stl_methods = extract_stl_method_symbols(binary_path, config.types.templates)
        if !isempty(stl_methods)
            stl_count = sum(length(v) for v in values(stl_methods))
            println("  metadata: $(length(stl_methods)) STL containers, $stl_count methods")
        end
    end

    metadata = Dict{String,Any}(
        "timestamp" => string(now()),
        "project" => config.project.name,
        "binary_path" => binary_path,
        "binary_type" => string(config.binary.type),

        # Source information
        "source_files" => source_files,
        "include_dirs" => [abspath(d) for d in get_include_dirs(config)],
        "compile_flags" => get_compile_flags(config),

        # Symbols and functions
        "symbols" => symbols,
        "functions" => functions,
        "globals" => global_vars,  # NEW: Global variables
        "function_count" => length(functions),

        # Type mappings
        "type_registry" => type_registry,
        "struct_definitions" => struct_defs,  # Struct member layout from DWARF
        "record_abi" => record_abi,            # By-value records: size, pass convention, empty
        "typedef_table" => typedef_table,      # Typedef name -> Julia type resolution
        # Inline constructors declared in headers and defined by no translation
        # unit: the rows the forced unit defines and the ones it skipped, with
        # reasons (`_inline_constructor_plan`, `generate_inline_constructors`).
        "inline_constructors" => plan,
        # ...and the other inline member functions, the same way
        # (`_inline_method_plan`, `generate_inline_definitions`).
        "inline_methods" => mplan,
        # ...and the calls that leave default arguments out
        # (`_default_argument_plan`): `bound` rows name each function's symbol.
        "default_arguments" => dargs,
        # Function-pointer c_types list their parameters (`function_ptr(R; P1, P2)*`).
        # Older metadata spells every one `function_ptr(R)*`, so the generators
        # build a callback's `@cfunction` example from it only when this is set.
        "function_pointer_parameters" => true,
        # A symbol with no DIE was looked up by address (`_alias_signature_key`),
        # so one still `inferred` here is no alias of a described function.
        # Older metadata never looked: the refusal says to rebuild.
        "alias_signatures" => true,

        # STL container method symbols (for JIT thunk generation)
        "stl_methods" => stl_methods,

        # Compiler information
        "compiler_info" => Dict(
            "llvm_version" => llvm_version,
            "clang_version" => clang_version,
            "target_triple" => _detect_target_triple(),
            "optimization_level" => config.link.optimization_level,
            "lto_enabled" => config.link.enable_lto
        ),

        # Language information
        "language" => config.wrap.language == :c ? "c" : "c++",

        # Metadata version (for future compatibility)
        "metadata_version" => "1.0"
    )

    return metadata
end

"""
    _structor_dwarf_key(mangled, dwarf_table) -> Union{String,Nothing}

The DWARF key for a constructor or destructor SYMBOL, or `nothing`.

The symbol table has the Itanium variants: `C1`/`C2`/`C3` (complete, base,
allocating) and `D0`/`D1`/`D2` (deleting, complete, base). clang writes the
unified `C4`/`D4` spelling into `DW_AT_linkage_name`, one DIE for every variant.
The exact-name join therefore missed EVERY constructor and destructor. Their
signatures fell back to `parse_parameters` on the demangled string, where a
by-value struct is only a name. msdfgen's `LinearSegment(Vector2, Vector2,
EdgeColor)` took `Any` arguments and stored garbage points.

Each marker position in the symbol is tried, and only a rewrite DWARF actually
has is accepted. A marker-shaped substring inside an identifier cannot produce a
real key. Exactly one hit is required. More than one is refused as ambiguous
rather than guessed.
"""
function _structor_dwarf_key(mangled::AbstractString, dwarf_table::AbstractDict)
    startswith(mangled, "_Z") || return nothing
    hits = String[]
    for m in eachmatch(r"(C[123]|D[012])(?=[EIB])", mangled)
        unified = m.match[1] == 'C' ? "C4" : "D4"
        key = mangled[1:prevind(mangled, m.offset)] * unified * mangled[m.offset+2:end]
        haskey(dwarf_table, key) && push!(hits, key)
    end
    return length(hits) == 1 ? hits[1] : nothing
end

"""
    _alias_signature_key(mangled, address, by_address, dwarf_table)
        -> Union{Nothing,Tuple{String,String}}

For a symbol no DIE describes: `(aliasee, key)`, a function at the same
`address` that `dwarf_table` describes and its key there; `nothing` when no
symbol at the address is described. `by_address` holds the library's exported
functions (`extract_symbols_from_binary`).

`int aliased_name(int) __attribute__((alias("aliased_real")))` is a second name
for `aliased_real`'s code and clang writes no DIE for it. Its signature was
then inferred from the demangled name, which for a C symbol is the name: no
parameters and `Cint`, a call that passes nothing and returns whatever the
callee computed from a stale register (2026-11-17). The code at an address is
one function's whatever it is called, so the described function's signature is
the alias's, exactly; that is the one use an address has here (a symbol's own
two names are still paired by position, `_pair_nm_outputs`). Several described
names at one address (constructor variants, identical functions a linker
folded) are the same code too, and the smallest name is taken so the answer
does not move. A constructor or destructor name is looked up as the merge
does (`_structor_dwarf_key`).

Only an exported name is a key into `dwarf_table`. A local symbol's is not:
two files may each define a `static int impl`, with different signatures, and
the table keeps one entry per name. So an alias whose aliasee is `static` (or
hidden, or a static the in-process C build promoted to `__rb_<lib>_<name>`)
finds nothing here and is refused, not given a namesake's signature. Matching
those needs the DIE's own address (`DW_AT_low_pc`), which the parser does not
read.
"""
function _alias_signature_key(mangled::AbstractString, address::AbstractString,
                              by_address::AbstractDict, dwarf_table::AbstractDict)
    for name in sort!(unique(get(by_address, address, String[])))
        name == mangled && continue
        key = haskey(dwarf_table, name) ? name : _structor_dwarf_key(name, dwarf_table)
        key === nothing || return (String(name), String(key))
    end
    return nothing
end

"""
    _mark_noexcept!(functions, candidates, nounwind)

Set `is_noexcept` on every function: `true` only when its bare `name` is a
`noexcept` candidate from the sources AND its `mangled` definition is `nounwind`
in the IR. `nounwind === nothing` (no IR, i.e. ingest) marks nothing, which sends
all C++ to Tier 2. The reasoning is in `extract_compilation_metadata`, and the
test is `test_noexcept_routing.jl`.

The proof by itself is recorded too, as `nounwind`, whenever there was IR to
read. It is what routes a function with C linkage in a C++ library
(`DispatchLogic.is_ccall_safe`): `extern "C"` changes a function's name, not
what its body may throw, and no source spelling says which of them can
(`test_c_linkage_unwind.jl`). No IR writes no key, and a reader takes a missing
key as "not proven". An alias (`alias_of`, `_alias_signature_key`) has no
definition of its own: its code is its aliasee's, so the aliasee's is the one
asked about.
"""
function _mark_noexcept!(functions, candidates::Set{String},
                         nounwind::Union{Nothing,Set{String}})
    for func in functions
        proven = nounwind !== nothing &&
                 get(func, "alias_of", get(func, "mangled", "")) in nounwind
        func["is_noexcept"] = proven && get(func, "name", "") in candidates
        nounwind === nothing || (func["nounwind"] = proven)
    end
    return functions
end

"""
    _mark_byval_params!(functions, byval)

Record on every function the IR defines the `byval` parameters of its
definition, as `byval_params`: a list of `{"record", "align"}` in parameter
order (`_byval_definitions`), empty when the definition passes nothing `byval`.
A function the IR only declares, and every function when there is no IR
(`byval === nothing`: ingest), gets no key, which the generator reads as "not
recorded" and decides from DWARF alone (`_c_byval_facts`). Metadata from
before this key has the same effect, so an over-aligned record parameter is
callable only after a rebuild. An alias (`alias_of`) takes its aliasee's
definition, whose signature it has.
"""
function _mark_byval_params!(functions, byval::Union{Nothing,Dict{String,Vector{Dict{String,Any}}}})
    byval === nothing && return functions
    for func in functions
        facts = get(byval, get(func, "alias_of", get(func, "mangled", "")), nothing)
        facts === nothing || (func["byval_params"] = Any[copy(f) for f in facts])
    end
    return functions
end

"""
Parse function signatures from symbol information.
Infers parameter types and return types from demangled names.
"""
function parse_function_signatures(symbols::Vector{Dict{String,Any}},
                                   struct_names::Set{String}=Set{String}(),
                                   enum_names::Set{String}=Set{String}())::Vector{Dict{String,Any}}
    functions = Dict{String,Any}[]

    for sym in symbols
        demangled = sym["demangled"]

        # Skip vtables, typeinfo, etc.
        if startswith(demangled, "vtable") || startswith(demangled, "typeinfo")
            continue
        end

        func_info = Dict{String,Any}(
            "name" => extract_function_name(demangled),
            "mangled" => sym["mangled"],
            "demangled" => demangled,
            "return_type" => infer_return_type(demangled),
            "parameters" => parse_parameters(demangled, struct_names, enum_names),
            "is_method" => contains(_name_prefix(demangled), "::"),
            "class" => extract_class_name(demangled),
            "exported" => true
        )

        push!(functions, func_info)
    end

    return functions
end

"""
Return the index of the first '(' at angle-bracket depth 0, or 0 if not found.
This correctly handles template parameters like `std::vector<int, std::allocator<int>>`.
"""
function _find_toplevel_paren(s::String)::Int
    depth = 0
    for (i, c) in enumerate(s)
        if c == '<'
            depth += 1
        elseif c == '>'
            depth -= 1
        elseif c == '(' && depth == 0
            return i
        end
    end
    return 0
end

# First '(' at or after byte index `from` that is outside `<>` and `{}`. Angle
# depth is unclamped, exactly as in `_find_toplevel_paren`, so `operator<` hides
# every later '(' here too. Braces are what `{lambda(int)#1}` needs.
function _next_toplevel_paren(s::String, from::Int)::Int
    angle = 0
    brace = 0
    i = from
    n = lastindex(s)
    while i <= n
        c = s[i]
        if c == '<'
            angle += 1
        elseif c == '>'
            angle -= 1
        elseif c == '{'
            brace += 1
        elseif c == '}'
            brace = max(0, brace - 1)
        elseif c == '(' && angle == 0 && brace == 0
            return i
        end
        i = nextind(s, i)
    end
    return 0
end

# The ')' closing the '(' at `open`, counting parens only (demangler output
# balances them; it does not balance `<>`). 0 if unbalanced.
function _matching_paren(s::String, open::Int)::Int
    depth = 0
    i = open
    n = lastindex(s)
    while i <= n
        c = s[i]
        if c == '('
            depth += 1
        elseif c == ')'
            depth -= 1
            depth == 0 && return i
        end
        i = nextind(s, i)
    end
    return 0
end

# What GNU prints between an enclosing function's `)` and the `::` of a local
# entity: its cv- and ref-qualifiers (`A::f() const &::{lambda()#1}`).
const _LOCAL_SCOPE_TAIL = r"^(?:\s*(?:const|volatile|restrict|&&|&))*::"

"""
    _param_list_paren(demangled) -> (open, local)

`open` is the byte index of the `(` that opens the function's OWN parameter
list, 0 if there is none. `local` is the `(` of the outermost enclosing
function when the symbol is a local entity, else 0.

`_find_toplevel_paren` (the first `(` outside `<>`) is that paren except where
the GNU demangler puts a paren group in front of it at depth 0. Two shapes
occur, and both are skipped:

  * **A `decltype (…)` return type.** `decltype ({parm#1}(0))
    fmt::v12::loc_value::visit<…>(…)` cut at the decltype's `(` gave the prefix
    `"decltype "`, and the return-type strip left name and class `""`.
  * **A parenthesized scope**, i.e. a group followed by cv/ref qualifiers and
    `::`. A local entity's enclosing function
    (`write_fixed<…>(…)::{lambda(…)#3}::operator()(…) const`) is one, and cutting
    there gave every lambda in `write_fixed` that function's name and scope, so
    they collapsed onto one Julia name. `(anonymous namespace)::` is the other.

After a skipped group the search resumes past it, ignoring the `(` inside
`{lambda(…)#N}`. `operator()`'s own `()` is part of its name, so what decides
there is the group after it. The name still cuts at `operator(`, which is the
spelling every `operator()` already has (name `"operator"`).

Every other shape gets the first candidate back unchanged: a group followed by
anything else, no later `(`, unbalanced parens, and `operator<`'s open `<`
(0, as before). The first candidate is `_find_toplevel_paren`'s answer, except
that a `(` inside `{…}` does not count. Only a name with a `{…}` scope before
any paren can tell the two apart (a namespace-scope lambda,
`f::{lambda()#1}::operator()() const`). Hub names depend on that answer.
"""
function _param_list_paren(s::String)::Tuple{Int,Int}
    first_open = _next_toplevel_paren(s, firstindex(s))
    open = first_open
    local_open = 0
    while open != 0
        close = _matching_paren(s, open)
        close == 0 && break
        group_end = close
        if close == nextind(s, open) &&
           occursin(r"(?:^|[^\w])operator$", SubString(s, 1, prevind(s, open)))
            after = nextind(s, close)
            (after <= lastindex(s) && s[after] == '(') || return (open, local_open)
            group_end = _matching_paren(s, after)
            group_end == 0 && break
        end
        if occursin(r"(?:^|[^\w])decltype\s*$", SubString(s, 1, prevind(s, open)))
            resume = nextind(s, group_end)
        else
            m = match(_LOCAL_SCOPE_TAIL, SubString(s, nextind(s, group_end)))
            m === nothing && return (open, local_open)
            if local_open == 0 && SubString(s, open, close) != "(anonymous namespace)"
                local_open = open
            end
            resume = nextind(s, group_end) + ncodeunits(m.match)
        end
        open = _next_toplevel_paren(s, resume)
    end
    return (first_open, 0)
end

"""
Return the function-name prefix of a demangled signature, i.e. the portion before
its own parameter list (see [`_param_list_paren`]).  For `"Calculator::compute(int)"`
this is `"Calculator::compute"`; for `"sum_vector(std::vector<int>)"` this is `"sum_vector"`.
"""
function _name_prefix(demangled::String)::String
    p, _ = _param_list_paren(demangled)
    return p == 0 ? demangled : demangled[1:prevind(demangled, p)]
end

"""
    _strip_return_type(prefix) -> String

Drop a leading return type from a demangled function-name prefix:
`"bool gguf_reader::read<int>"` → `"gguf_reader::read<int>"`.

Itanium mangles a function **template's** return type into the symbol, so the
demangler prints it — ordinary functions carry none. The return type ends at the
last space outside `<>`/`()`/`{}`, which is enough even when it is itself qualified
(`std::enable_if<std::is_integral<unsigned int>::value, bool>::type`), or is a
`decltype (…)`. `{}` is for `{unnamed type#1}`.

A **conversion operator** is the only C++ name that legitimately contains a
top-level space (`operator bool`, `operator void (*)(…)`, `operator new`), and
cutting at its space would eat the name. Those bail out unchanged, so the result
is the historical behaviour rather than a new wrong answer — pugixml has 7.
"""
function _strip_return_type(prefix::AbstractString)::String
    occursin("operator", prefix) && return String(prefix)
    depth = 0
    cut = 0
    i = firstindex(prefix)
    n = lastindex(prefix)
    while i <= n
        c = prefix[i]
        if c == '<' || c == '(' || c == '{'
            depth += 1
        elseif c == '>' || c == ')' || c == '}'
            depth = max(0, depth - 1)
        elseif depth == 0 && c == ' '
            cut = i
        end
        i = nextind(prefix, i)
    end
    cut == 0 && return String(prefix)
    return String(strip(prefix[nextind(prefix, cut):end]))
end

"""
    _qualified_name_parts(demangled) -> Vector{String}

The `::`-separated components of a demangled signature's qualified name, with any
leading return type removed and template arguments left intact:
`"bool gguf_reader::read<std::vector<int> >(…)"` → `["gguf_reader", "read<std::vector<int> >"]`.

`extract_class_name` and `extract_function_name` both derive from this one
function, so they cannot disagree about where the scope ends.

The obvious `split(prefix, "::")` this replaces got two things wrong, both live
in the Hub until 2026-08-13:

  * **Template arguments contain `::`.** A flat split cuts inside them, so
    `llama_model_loader::get_arr<std::vector<int, std::allocator<int> > >` yielded
    class `"bool llama_model_loader::get_arr<std::vector<int, std"` and name
    `"allocator<int> > >"`. Splitting is angle-bracket- and paren-depth aware
    (depth clamps at 0 so `operator>` / `operator->` cannot drive it negative).

  * **The return type of a function template.** The scope came out
    `"bool gguf_reader"`; no aggregate is named that, so neither receiver gate
    (`FunctionGen._has_receiver`, `GeneratorCpp`'s `struct_types` check)
    synthesized `this` and every argument shifted one register — the silent
    direction. Measured: **62 methods gained a receiver, 0 lost one** across
    box2d (4), llamacpp (48), pugixml (8), tinyxml2 (2).

A **local entity** keeps its enclosing function's signature in the scope, as
the demangler prints it (`class` is the demangler's prefix):
`"fmt::v12::detail::write_fixed<…>(…)::{lambda(fmt::v12::basic_appender<char>)#3}::operator()(…) const"`
→ class `"fmt::v12::detail::write_fixed<…>(…)::{lambda(fmt::v12::basic_appender<char>)#3}"`,
name `"operator"`. Only the outermost name can carry a return type
(`void foo()::Bar::baz<int>()`), and the enclosing function's qualifiers
(`A::f() const::…`) contain a depth-0 space, so the strip is applied to the part
before the enclosing function's `(` only. See [`_param_list_paren`] for the
`decltype` return type and the local-entity cut.
"""
function _qualified_name_parts(demangled::AbstractString)::Vector{String}
    s = String(demangled)
    open, local_open = _param_list_paren(s)
    prefix = if local_open == 0
        _strip_return_type(open == 0 ? s : s[1:prevind(s, open)])
    else
        _strip_return_type(s[1:prevind(s, local_open)]) * s[local_open:prevind(s, open)]
    end
    out = String[]
    depth = 0
    start = firstindex(prefix)
    i = start
    n = lastindex(prefix)
    while i <= n
        c = prefix[i]
        if c == '<' || c == '(' || c == '{'
            depth += 1
        elseif c == '>' || c == ')' || c == '}'
            depth = max(0, depth - 1)
        elseif depth == 0 && c == ':' && i < n && prefix[nextind(prefix, i)] == ':'
            push!(out, String(prefix[start:prevind(prefix, i)]))
            i = nextind(prefix, i)
            start = nextind(prefix, i)
        end
        i = nextind(prefix, i)
    end
    push!(out, String(prefix[start:end]))
    return out
end

"""
Extract function name from demangled signature.
Example: "Calculator::compute(int, int, char)" -> "compute"
Example: "sum_vector(std::vector<int, std::allocator<int> > const&)" -> "sum_vector"
Example: "bool gguf_reader::read<std::vector<int> >(...)" -> "read<std::vector<int> >"
"""
function extract_function_name(demangled::String)::String
    return String(strip(_qualified_name_parts(demangled)[end]))
end

"""
Extract class name from method signature (only considers '::' in the function-name prefix).
Example: "Calculator::compute(int, int)" -> "Calculator"
Example: "sum_vector(std::vector<int> const&)"  -> ""  (free function)
Example: "bool gguf_reader::read<int>(...)"     -> "gguf_reader"  (NOT "bool gguf_reader")
"""
function extract_class_name(demangled::String)::String
    parts = _qualified_name_parts(demangled)
    length(parts) >= 2 || return ""
    return join(parts[1:end-1], "::")
end

"""
Infer return type from demangled function signature using pattern matching.
This is a fallback when DWARF debug info is unavailable for a given function.
"""
function infer_return_type(demangled::String)::Dict{String,Any}
    name = extract_function_name(demangled)
    class = extract_class_name(demangled)

    # Constructors and destructors return void
    if !isempty(class)
        bare_class = last(split(class, "::"))
        if name == bare_class || name == "~$bare_class"
            return Dict("c_type" => "void", "julia_type" => "Cvoid", "size" => 0)
        end
    end

    # Operator patterns with known return types
    if startswith(name, "operator")
        op = replace(name, "operator" => "")
        op = strip(op)
        # Boolean operators
        if op in ["==", "!=", "<", ">", "<=", ">=", "!", "&&", "||", "bool"]
            return Dict("c_type" => "bool", "julia_type" => "Bool", "size" => 1)
        end
        # Assignment/compound assignment returns reference (effectively void for FFI)
        if op in ["=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="]
            return Dict("c_type" => "void", "julia_type" => "Cvoid", "size" => 0)
        end
        # new/new[] return pointer
        if startswith(op, "new")
            return Dict("c_type" => "void*", "julia_type" => "Ptr{Cvoid}", "size" => sizeof(Ptr))
        end
        # delete/delete[] return void
        if startswith(op, "delete")
            return Dict("c_type" => "void", "julia_type" => "Cvoid", "size" => 0)
        end
    end

    # Common naming conventions
    lower_name = lowercase(name)

    # void patterns: set_, init_, destroy_, free_, clear_, reset_, print_, write_, close_
    for prefix in ["set_", "init_", "destroy_", "free_", "clear_", "reset_",
                    "print_", "write_", "close_", "release_", "cleanup_", "remove_",
                    "delete_", "push_", "pop_", "insert_", "erase_", "swap_"]
        if startswith(lower_name, prefix)
            return Dict("c_type" => "void", "julia_type" => "Cvoid", "size" => 0)
        end
    end

    # bool patterns: is_, has_, can_, should_, contains_, empty_, valid_
    for prefix in ["is_", "has_", "can_", "should_", "contains_", "empty_", "valid_",
                    "check_", "exists_", "equals_", "matches_"]
        if startswith(lower_name, prefix)
            return Dict("c_type" => "bool", "julia_type" => "Bool", "size" => 1)
        end
    end

    # size/count patterns → size_t
    for prefix in ["size", "count", "length", "num_", "get_size", "get_count", "get_length",
                    "get_num", "capacity"]
        if startswith(lower_name, prefix)
            return Dict("c_type" => "size_t", "julia_type" => "Csize_t", "size" => sizeof(Csize_t))
        end
    end

    # Pointer-returning patterns: create_, alloc_, malloc_, new_, get_ptr, make_
    for prefix in ["create_", "alloc_", "malloc_", "new_", "make_", "open_"]
        if startswith(lower_name, prefix)
            return Dict("c_type" => "void*", "julia_type" => "Ptr{Cvoid}", "size" => sizeof(Ptr))
        end
    end

    # String-returning patterns
    for prefix in ["to_string", "get_name", "get_string", "get_path", "get_message",
                    "get_error", "strerror", "name", "what"]
        if startswith(lower_name, prefix) || lower_name == prefix
            return Dict("c_type" => "const char*", "julia_type" => "Cstring", "size" => sizeof(Ptr))
        end
    end

    # Default fallback: int (most common return type in C APIs)
    return Dict("c_type" => "int", "julia_type" => "Cint", "size" => 4)
end

"""
Find the matching `)` for the first top-level `(` in `s`, scanning at angle-bracket
depth 0 so template parameters like `std::vector<int, std::allocator<int>>` don't
cause a premature match. Returns (open_index, close_index) byte offsets, or
(0, 0) if no balanced pair exists at the top level.
"""
function _find_toplevel_paren_pair(s::AbstractString)::Tuple{Int,Int}
    open_idx = _find_toplevel_paren(String(s))
    open_idx == 0 && return (0, 0)
    angle = 0
    paren = 0
    for i in eachindex(s)
        i < open_idx && continue
        c = s[i]
        if c == '<'
            angle += 1
        elseif c == '>'
            angle -= 1
        elseif angle == 0 && c == '('
            paren += 1
        elseif angle == 0 && c == ')'
            paren -= 1
            paren == 0 && return (open_idx, i)
        end
    end
    return (0, 0)
end

"""
Split a parameter list at commas that sit at angle-bracket depth 0, so
`std::map<int, std::vector<int>>, int` splits into two parts, not four.
"""
function _split_toplevel_commas(s::AbstractString)::Vector{String}
    parts = String[]
    depth = 0
    start = firstindex(s)
    for i in eachindex(s)
        c = s[i]
        if c == '<'
            depth += 1
        elseif c == '>'
            depth -= 1
        elseif c == ',' && depth == 0
            push!(parts, String(s[start:prevind(s, i)]))
            start = nextind(s, i)
        end
    end
    if start <= lastindex(s)
        push!(parts, String(s[start:lastindex(s)]))
    end
    return parts
end

"""
Parse parameter types from demangled signature.
Example: "add(int, int)" -> [{"type": "int", "julia_type": "Cint"}, ...]
Template-aware: angle brackets in nested types do not confuse the splitter.
"""
function parse_parameters(demangled::String,
                         struct_names::Set{String}=Set{String}(),
                         enum_names::Set{String}=Set{String}())::Vector{Dict{String,Any}}
    params = Dict{String,Any}[]

    (open_idx, close_idx) = _find_toplevel_paren_pair(demangled)
    open_idx == 0 && return params

    inner = demangled[nextind(demangled, open_idx):prevind(demangled, close_idx)]
    param_types = _split_toplevel_commas(inner)

    for (i, ptype) in enumerate(param_types)
        ptype = strip(ptype)
        if isempty(ptype)
            continue
        end

        julia_type = cpp_to_julia_type(ptype, struct_names, enum_names)

        push!(params, Dict(
            "name" => "arg$i",
            "c_type" => ptype,
            "julia_type" => julia_type,
            "position" => i-1
        ))
    end

    return params
end

"""
Convert C++ type to Julia type.
Basic type mapping - can be enhanced with type registry.

# Arguments
- `cpp_type`: C++ type string
- `struct_names`: Set of known struct/class names from DWARF
- `enum_names`: Set of known enum names from DWARF
"""
const _CPP_TO_JULIA_TYPE_MAP = Dict{String,String}(
    "int" => "Cint",
    "unsigned int" => "Cuint",
    "long" => "Clong",
    "unsigned long" => "Culong",
    "short" => "Cshort",
    "unsigned short" => "Cushort",
    "char" => "UInt8",
    "unsigned char" => "UInt8",
    "float" => "Cfloat",
    "double" => "Cdouble",
    "bool" => "Bool",
    "_Bool" => "Bool",
    "void" => "Cvoid",

    # NO `char*` / `const char*` ENTRY, DELIBERATELY. Two used to sit here
    # mapping to `Cstring`, and they were unreachable: `cpp_to_julia_type` peels
    # trailing `*` BEFORE consulting this map, so a pointer never reaches a
    # pointer-spelled key. They read as if this map unified the two type
    # mappings, and it does not.
    #
    # It must not. `dwarf_type_to_julia` (RETURN types) maps `char*` to
    # `Cstring`, which is what the whole Cstring return policy is built on —
    # `_cstring_wrapper_pair`, the `_ptr` siblings, `_assert_cstring_policy`.
    # This map serves PARAMETERS and gives `Ptr{UInt8}`. Two answers to one C
    # spelling, on purpose, because they answer different questions. Re-adding a
    # `char*` key here would not change behaviour (still unreachable) but would
    # re-plant the idea that these should agree. See the 2026-08-20 note.

    # DWARF's CANONICAL INTEGER SPELLINGS. DWARF emits the expanded form
    # (`long int`, `short int`, `signed char`), and this map only had the short
    # ones — so those parameters resolved to `Any` while the return path typed
    # them correctly. Measured across the Hub: 11 parameters affected, so the
    # gap is narrow, but every one of these is an ordinary integer with nothing
    # subtle about its ABI.
    "signed" => "Cint",
    "signed int" => "Cint",
    "unsigned" => "Cuint",
    "signed char" => "Cchar",
    "short int" => "Cshort",
    "signed short" => "Cshort",
    "signed short int" => "Cshort",
    "unsigned short int" => "Cushort",
    "long int" => "Clong",
    "signed long" => "Clong",
    "signed long int" => "Clong",
    "unsigned long int" => "Culong",
    "long long int" => "Clonglong",
    "signed long long" => "Clonglong",
    "signed long long int" => "Clonglong",
    "unsigned long long int" => "Culonglong",
    "wchar_t" => "Cwchar_t",
    "char8_t" => "UInt8",
    "char16_t" => "UInt16",
    "char32_t" => "UInt32",
    "time_t" => "Ctime_t",

    # `__int128` and `unsigned __int128` (GCC's DWARF: `__int128 unsigned`;
    # `__int128_t`/`__uint128_t` are typedefs of them) are INTEGER class in
    # both eightbytes. `Int128`/`UInt128` are the Julia types, and `ccall`
    # passes one as C does only while both halves fit in registers; the C
    # generator refuses the positions where it does not
    # (`Wrapper._c_int128_misplaced`), and a thunk passes `i128`.
    "__int128" => "Int128",
    "__int128_t" => "Int128",
    "unsigned __int128" => "UInt128",
    "__int128 unsigned" => "UInt128",
    "__uint128_t" => "UInt128",

    # `long double` is DELIBERATELY ABSENT. It is x87 80-bit, SysV class X87,
    # passed on the x87 stack — NOT in the integer registers an `NTuple{2,UInt64}`
    # would claim. The return map answers `NTuple{2, UInt64}`, which is a storage
    # size, not a calling convention. Typing it here would convert a loud `Any`
    # into a silently mis-passed argument, which is strictly worse. 6 parameters
    # in fmt hit this and should keep failing visibly until the classifier grows
    # an X87 path.
    "size_t" => "Csize_t",
    "ssize_t" => "Cssize_t",
    "ptrdiff_t" => "Cptrdiff_t",
    "intptr_t" => "Cintptr_t",
    "uintptr_t" => "Cuintptr_t",
    "int8_t" => "Int8",
    "uint8_t" => "UInt8",
    "int16_t" => "Int16",
    "uint16_t" => "UInt16",
    "int32_t" => "Int32",
    "uint32_t" => "UInt32",
    "int64_t" => "Int64",
    "uint64_t" => "UInt64",
    "long long" => "Clonglong",
    "unsigned long long" => "Culonglong",
    # C11 _Complex types
    "_Complex float" => "ComplexF32",
    "_Complex double" => "ComplexF64",
    "complex float" => "ComplexF32",
    "complex double" => "ComplexF64",
    "float _Complex" => "ComplexF32",
    "double _Complex" => "ComplexF64",
    "complex float" => "ComplexF32",
    "complex double" => "ComplexF64",
    "complex long double" => "ComplexF64",
)

const _CPP_INTERNAL_TYPE_BLOCKLIST = Set{String}([
    "__va_list_tag", "__mbstate_t", "_va_list_tag", "_mbstate_t", "max_align_t",
])

function cpp_to_julia_type(cpp_type::AbstractString,
                           struct_names::Set{String}=Set{String}(),
                           enum_names::Set{String}=Set{String}())::String
    # Strip qualifiers
    cpp_type = strip(cpp_type)
    cpp_type = replace(cpp_type, r"^const\s+" => "")
    cpp_type = replace(cpp_type, r"^volatile\s+" => "")

    # Clean up internal compiler types to avoid leaking them
    base_type_for_check = strip(replace(cpp_type, r"[*&\[\]\s\d]+$" => ""))
    if base_type_for_check in _CPP_INTERNAL_TYPE_BLOCKLIST
        cpp_type = replace(cpp_type, base_type_for_check => "void")
    end

    type_map = _CPP_TO_JULIA_TYPE_MAP

    # Strip _Atomic qualifier — same ABI layout as the underlying type
    cpp_type = replace(cpp_type, r"\b_Atomic\b\s*" => "")
    cpp_type = strip(cpp_type)

    # Handle arrays: type[size] or type[size1][size2]
    # Examples: "double[3]" -> "NTuple{3, Cdouble}"
    #           "int[3][4]" -> "NTuple{12, Cint}"  (flattened)
    # The extents are the trailing run only. An element can hold brackets of
    # its own: `function_ptr(void; int[4]*)*[2]` is two function pointers, and
    # `int[4]*[2]` two pointers, where the first `[` gave eight of each.
    if contains(cpp_type, "[") && contains(cpp_type, "]")
        # Extract element type and dimensions
        array_match = match(r"^(.+?)((?:\[\d+\])+)$", cpp_type)
        if !isnothing(array_match)
            element_type_str = strip(array_match.captures[1])
            dims_str = array_match.captures[2]

            # Parse dimensions: [3][4] -> [3, 4]
            dims = Int[]
            for dim_match in eachmatch(r"\[(\d+)\]", dims_str)
                push!(dims, parse(Int, dim_match.captures[1]))
            end

            if !isempty(dims)
                # Calculate total size (product of dimensions)
                total_size = prod(dims)

                # Map element type to Julia (with struct/enum awareness)
                element_julia_type = cpp_to_julia_type(element_type_str, struct_names, enum_names)

                # Return as NTuple (flattened)
                return "NTuple{$total_size, $element_julia_type}"
            end
        end
    end

    # Handle function pointers (and pointers to function pointers)
    # First, strip trailing '*' if present, and then check for "function_ptr("
    temp_cpp_type = replace(cpp_type, r"\*+$" => "") # Temporarily strip pointers for the check
    if startswith(temp_cpp_type, "function_ptr(")
        return "Ptr{Cvoid}"
    end

    # Check if it's a user-defined struct/class (before pointer handling)
    base_type = replace(cpp_type, r"[*&\s]+$" => "")  # Strip pointer/ref/spaces
    base_type = strip(base_type)

    if base_type in struct_names
        # Count pointer levels: sqlite3** → 2, sqlite3* → 1
        ptr_suffix = match(r"(\*+)\s*$", cpp_type)
        if !isnothing(ptr_suffix)
            ptr_count = length(ptr_suffix.captures[1])
            result = base_type
            for _ in 1:ptr_count
                result = "Ptr{$result}"
            end
            return result
        elseif contains(cpp_type, "&")
            return "Ref{$base_type}"
        else
            return base_type  # Use struct name directly
        end
    end

    # An enum BY VALUE is used by name (the type the wrapper emits for it). A
    # pointer or reference to one is an address, typed by the branches below
    # like a pointer to any scalar: `E*` is `Ptr{E}`, `E**` `Ptr{Ptr{E}}`,
    # `E&` and `const E&` `Ref{E}`. Returning the name for those too wrapped
    # `void out(enum E *)` as taking the enum's VALUE, and C wrote through it
    # as an address.
    if cpp_type in enum_names
        return String(cpp_type)
    end

    # Handle pointers (for non-struct types)
    if endswith(cpp_type, "*")
        # Map the base type first, then wrap in Ptr{}
        base = strip(cpp_type[1:end-1])
        base_julia = cpp_to_julia_type(base, struct_names, enum_names)
        # If base mapped to a specific type, use it; otherwise fall back to Ptr{Cvoid}.
        # An enum maps to its own name.
        if base_julia != "Any" && (base_julia != base || base in enum_names)
            return "Ptr{$base_julia}"
        else
            return "Ptr{Cvoid}"
        end
    end

    # Handle references (for non-struct types). `T&&` is one reference, not two.
    if endswith(cpp_type, "&")
        base = strip(replace(cpp_type, r"&&?$" => ""))
        return "Ref{$(cpp_to_julia_type(base, struct_names, enum_names))}"
    end

    # Fallback to type map or Any
    return get(type_map, cpp_type, "Any")
end

"""
Build type registry from functions.
Collects all unique types used in the codebase.
"""
function build_type_registry(functions::Vector{Dict{String,Any}})::Dict{String,Any}
    registry = Dict{String,Any}()

    # Add standard types
    standard_types = Dict(
        "int" => Dict("size" => 4, "alignment" => 4, "julia_type" => "Cint"),
        "float" => Dict("size" => 4, "alignment" => 4, "julia_type" => "Cfloat"),
        "double" => Dict("size" => 8, "alignment" => 8, "julia_type" => "Cdouble"),
        "char" => Dict("size" => 1, "alignment" => 1, "julia_type" => "UInt8"),
        "bool" => Dict("size" => 1, "alignment" => 1, "julia_type" => "Bool"),
        "void" => Dict("size" => 0, "alignment" => 0, "julia_type" => "Cvoid")
    )

    merge!(registry, standard_types)

    # Collect types from functions
    for func in functions
        # Add return type
        if haskey(func, "return_type")
            rt = func["return_type"]
            if haskey(rt, "c_type")
                registry[rt["c_type"]] = rt
            end
        end

        # Add parameter types
        if haskey(func, "parameters")
            for param in func["parameters"]
                if haskey(param, "c_type")
                    registry[param["c_type"]] = Dict(
                        "julia_type" => get(param, "julia_type", "Any"),
                        "inferred" => true
                    )
                end
            end
        end
    end

    return registry
end

"""
Save compilation metadata to JSON file next to binary.
This enables automatic wrapper generation!
"""
function save_compilation_metadata(config::RepliBuildConfig, source_files::Vector{String},
                                   binary_path::String;
                                   nounwind::Union{Nothing,Set{String}}=nothing,
                                   byval::Union{Nothing,Dict{String,Vector{Dict{String,Any}}}}=nothing,
                                   inline_constructors::Union{Nothing,AbstractDict}=nothing,
                                   inline_methods::Union{Nothing,AbstractDict}=nothing,
                                   default_arguments::Union{Nothing,AbstractDict}=nothing)::String
    # Extract metadata
    metadata = extract_compilation_metadata(config, source_files, binary_path;
                                            nounwind=nounwind, byval=byval,
                                            inline_constructors=inline_constructors,
                                            inline_methods=inline_methods,
                                            default_arguments=default_arguments)

    # Static-promotion map (old static name → exported __rb_* symbol), written
    # by link_optimize_ir when the promotion pass ran. Slices resolve their
    # declarations through this map; tooling uses it to de-mystify nm output.
    promoted_map_path = joinpath(get_build_path(config), "promoted_symbols.json")
    if isfile(promoted_map_path)
        metadata["promoted_symbols"] = _read_json(promoted_map_path)
    end
    # Exported-alias map (public name → exported `__rb_*` alias), written only
    # when `[wrap.tier1] enable` asked for it. A slice declares the library's
    # own symbols by these names; `nm` shows each beside its aliasee.
    aliases_map_path = joinpath(get_build_path(config), "exported_aliases.json")
    if isfile(aliases_map_path)
        metadata["exported_aliases"] = _read_json(aliases_map_path)
    end

    # Save to JSON file next to binary
    output_dir = get_output_path(config)
    metadata_path = joinpath(output_dir, "compilation_metadata.json")

    open(metadata_path, "w") do io
        JSON.print(io, metadata, 2)  # Pretty print with indent=2
    end

    nfuncs = length(get(metadata, "functions", []))
    # struct_definitions also stores enums under "__enum__<name>" keys — exclude them
    # from the type count so the diagnostic matches what was actually wrapped.
    nstructs = count(k -> !startswith(k, "__enum__"),
                     keys(get(metadata, "struct_definitions", Dict())))
    println("  dwarf: $(nfuncs) functions, $(nstructs) types")
    return metadata_path
end

end # module Compiler
