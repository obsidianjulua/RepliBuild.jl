#!/usr/bin/env julia
# RepliBuild.jl — Registry Test Suite
#
# Lightweight tests for `Pkg.test()` / AutoMerge CI.
# No C++ toolchain required — validates package loading, types, and API surface.
# For full integration tests: julia --project=. test/devtests.jl

using Test
using RepliBuild

# The registry and the Julia environments are the user's. `register` writes to
# ~/.replibuild and adds packages to the default Julia environment, so the suite
# points both somewhere else first; a test that wants either names it.
ENV["REPLIBUILD_HOME"] = mktempdir()
ENV["REPLIBUILD_JULIA_ENV"] = "none"

@testset "RepliBuild.jl" begin

    @testset "Package loads" begin
        # VERSION const must track Project.toml — the pre-3.0.0 three-way
        # version drift started exactly here (hardcoded literals go stale)
        @test RepliBuild.VERSION == pkgversion(RepliBuild)
        @test isdefined(RepliBuild, :discover)
        @test isdefined(RepliBuild, :build)
        @test isdefined(RepliBuild, :wrap)
        @test isdefined(RepliBuild, :clean)
        @test isdefined(RepliBuild, :info)
        @test isdefined(RepliBuild, :check_environment)
        @test isdefined(RepliBuild, :search)
    end

    @testset "ConfigurationManager" begin
        cfg = RepliBuild.ConfigurationManager

        mktempdir() do dir
            toml_path = joinpath(dir, "replibuild.toml")
            write(toml_path, """
            [project]
            name = "test_pkg"
            uuid = "00000000-0000-0000-0000-000000000000"
            root = "."

            [compile]
            source_files = ["src/foo.cpp"]
            include_dirs = ["include"]
            flags = ["-std=c++17"]

            [link]
            enable_lto = false
            optimization_level = "2"

            [binary]
            type = "shared"

            [wrap]
            style = "module"
            """)

            config = cfg.load_config(toml_path)
            @test config.project.name == "test_pkg"
            @test "-std=c++17" in config.compile.flags
            @test config.link.optimization_level == "2"
            @test config.binary.type == :shared
        end
    end

    @testset "DWARFParser types" begin
        dp = RepliBuild.DWARFParser

        vm = dp.VirtualMethod("foo", "_ZN1A3fooEv", 0, "int", String[])
        @test vm.name == "foo"
        @test vm.slot == 0

        ci = dp.ClassInfo("A", 0, String[], [vm], dp.MemberInfo[], 8)
        @test ci.name == "A"
        @test length(ci.virtual_methods) == 1
    end

    @testset "JLCSIRGenerator" begin
        dp = RepliBuild.DWARFParser
        gen = RepliBuild.JLCSIRGenerator

        vm = dp.VirtualMethod("bar", "_ZN1B3barEv", 0, "void", ["int"])
        ci = dp.ClassInfo("B", 0, String[], [vm], dp.MemberInfo[], 8)

        ir = gen.generate_type_info_ir("B", ci, UInt64(0x1000))
        @test contains(ir, "jlcs.type_info")
        @test contains(ir, "\"B\"")

        ir_m = gen.generate_virtual_method_ir(vm, UInt64(0x2000))
        @test contains(ir_m, "thunk__ZN1B3barEv")
    end

    @testset "MLIR native library" begin
        mlir = RepliBuild.MLIRNative
        if isfile(mlir.libJLCS)
            ctx = mlir.create_context()
            @test ctx != C_NULL

            mod = mlir.parse_module(ctx, """
                module { func.func @id(%x: i32) -> i32 { return %x : i32 } }
            """)
            @test mod != C_NULL

            mlir.destroy_context(ctx)
        else
            @info "libJLCS not found — skipping MLIR native tests (Tier 1 still works)"
        end
    end

    @testset "Environment doctor" begin
        status = RepliBuild.check_environment(verbose=false)
        @test hasproperty(status, :ready) || hasfield(typeof(status), :ready)
    end
end

# ── Ingest-mode unit tests (no C++ toolchain required) ───────────────────────

include(joinpath(@__DIR__, "test_ingest.jl"))

# ── Registry unit tests (no C++ toolchain required) ──────────────────────────

include(joinpath(@__DIR__, "test_registry.jl"))

# ── clean(toml; deps=false) keeps the vendored git clones (no toolchain) ─────

include(joinpath(@__DIR__, "test_clean.jl"))

# ── discover with no system LLVM (no toolchain) ──────────────────────────────
# A C project needs only Julia, but discover's optional dependency walk and its
# toolchain warm-up raised "LLVM Toolchain not found" with no LLVM installed.
# The first GitHub CI run hit it in test_registry's c_test integration.

include(joinpath(@__DIR__, "test_discover_without_llvm.jl"))

# ── User-facing messages say what is true (no toolchain) ─────────────────────

include(joinpath(@__DIR__, "test_user_messages.jl"))

# ── Varargs @ccall emission regression (no toolchain required) ───────────────

include(joinpath(@__DIR__, "test_varargs_emission.jl"))

# ── User-intent TOML preservation across re-discovery (no toolchain required) ─

include(joinpath(@__DIR__, "test_toml_preservation.jl"))

# ── [compile] visibility: the declared form of -fvisibility=hidden (no toolchain)
# Parsed, validated, consumed through get_compile_flags so it reaches the compile
# fingerprint, serialized, and preserved across a forced re-discovery. Each of
# those fails SILENTLY if it regresses — a key that does not reach the fingerprint
# serves IR built under the other visibility, and a key dropped on a round trip
# restores the library's internals to the wrapper's API.

include(joinpath(@__DIR__, "test_compile_visibility.jl"))

# ── C-generator policy regressions: cstring_owned, macro-shim visibility,
#    blob param trap, bitfield byte-span (no toolchain required) ──────────────

include(joinpath(@__DIR__, "test_c_generator_policies.jl"))

# ── Tier 3 by-value record classification, per target ABI (no toolchain) ────
# Where C returns a record (hidden pointer or registers) decides the ccall
# shape, and a wrong answer shifts every argument with no error (C1). Both
# conventions are checked on any host; the live calls are in devtests.

include(joinpath(@__DIR__, "test_c_sret_classification.jl"))

# ── Record layouts: natural or not, for C and C++ routing (no toolchain) ─────
# One answer for "does a Julia struct of these fields reproduce the C layout?",
# from the member types' alignment. GeneratorCpp keeps ≤16-byte byte blobs off
# the ccall path.

include(joinpath(@__DIR__, "test_record_natural_layout.jl"))

# ── C++ records emitted as `mutable struct` (no toolchain) ──────────────────
# A named union or bitfield record moves by value only through its bytes:
# the Tier 2 sret return (abi_fuzz X1), member accessors (X5), and bitfields
# wider than 32 bits (X6).

include(joinpath(@__DIR__, "test_cpp_mutable_records.jl"))

# ── What a C++ finalizer calls (no toolchain) ───────────────────────────────
# A Managed<X> or factory struct frees its object with the library's deleter
# function (a static member counts) or the deleting destructor (D0), never a
# destructor that only ends the object's lifetime or a method that takes
# `this`; with neither, no owning type. A factory makes the class it returns,
# not one its name invents.

include(joinpath(@__DIR__, "test_cpp_finalizers.jl"))

# ── DWARF member sizes through typedefs, enums, qualifiers, arrays (no toolchain)
# A member typed as an enum or a typedef'd record came out with size 0, and the
# C++ layout proof turned the whole record into a byte blob. A real GNU dump.

include(joinpath(@__DIR__, "test_dwarf_member_sizes.jl"))

# ── Macro-shim header-collision guard (library-free fixture; needs clang) ────

include(joinpath(@__DIR__, "test_shim_header_guard.jl"))

# ── [wrap] shim_defines: a header's implementation section from the TOML ─────
# The defines go at the top of the shim file and nowhere else; the file is
# written without macros; bad entries are refused at load; the header guard
# runs for a defines-only shim (needs the JLL clang, like the guard above).

include(joinpath(@__DIR__, "test_shim_defines.jl"))

# ── STL force-TU default ctor + paren-aware Class::method split ──────────────
# Library-free traces of the two Hub stl wrap defects; optional -O2 emission
# probe needs clang++ and self-skips without it.

include(joinpath(@__DIR__, "test_stl_extract.jl"))

# ── Supplementary header enums come from the project's headers (no toolchain) ─
# The Clang.jl enum walk took every EnumDecl in the translation unit, so a
# header including <system_error> brought libstdc++'s enums into the wrapper,
# and which ones depended on the running Julia's libclang. Declarations outside
# the named headers' directories and the include dirs are dropped.

include(joinpath(@__DIR__, "test_header_enum_provenance.jl"))

# ── Git dependency cache is version-aware (local git upstream; needs git) ────

include(joinpath(@__DIR__, "test_dep_cache.jl"))

# ── Git dependency submodules (local git repos over file://; needs git) ──────
# `submodules = [...]` fetches exactly the listed paths at the commits the
# superproject records; each is a source root of the dependency. A submodule off
# its recorded commit is a hard error, and the key survives discover(force=true).

include(joinpath(@__DIR__, "test_dep_submodules.jl"))

# ── Wrapper type-binding guard (no toolchain required) ───────────────────────
# A type used in a foreign-call signature but never declared by the module is
# an UndefVarError at include time — the whole wrapper dies, not one function.
# Library-free traces over the checker plus its refusal-to-write behaviour.

include(joinpath(@__DIR__, "test_wrapper_type_bindings.jl"))

# ── Emitted struct size == DWARF byte_size (no toolchain required) ───────────
# A Tier-2 MEMORY-class struct return is stored straight into the caller's
# `Ref{T}`, so an emitted body one byte too large writes past a live Julia
# object. Measures the emitted type string; needs no MLIR or clang.

include(joinpath(@__DIR__, "test_struct_layout.jl"))

# ── Template-spelled members keep their size (no toolchain required) ─────────
# A member typed `Vec<int, 3>` or libc++'s `array<int, 3ULL>` was emitted as an
# 8-byte `!llvm.ptr`, so a by-value Tier-2 return lost every later member.

include(joinpath(@__DIR__, "test_template_members.jl"))

# ── Byte-blob struct setters + the Base-name namespace guard (no toolchain) ──
# Blob structs had accessors in one direction only, so a param struct the
# library builds was read-only — the only path into llama.cpp runs through one.

include(joinpath(@__DIR__, "test_blob_setters.jl"))

# ── Export hygiene (no toolchain) ────────────────────────────────────────────
# The consumer-side half of the Base-name problem. `_assert_base_calls_qualified`
# stops the wrapper breaking ITSELF; this stops it breaking whoever `using`s it,
# which is silent until their first failure path runs.

include(joinpath(@__DIR__, "test_export_hygiene.jl"))

# ── char* return policy + the tier-independence guard (no toolchain) ─────────
# The policy lived on the ccall path only, so the C++ MLIR-dispatch branch —
# which skips that path entirely — handed back bare pointers and silently
# discarded any [wrap.cstring_owned] deallocator. One derivation now, and a
# guard on the write path so a tier cannot decide presentation again.

include(joinpath(@__DIR__, "test_cstring_policy.jl"))

# ── Dispatch + layout introspection (no toolchain) ───────────────────────────
# Twelve Hub consumers hand-rolled `kernel_emits_llvmcall` against a private
# kernel, and four more re-parsed compilation_metadata.json for sizes/offsets.
# The wrapper emits both now; these pin the emitters and execute what they emit.

include(joinpath(@__DIR__, "test_introspection.jl"))

# ── Enumerator named `_` (no toolchain) ──────────────────────────────────────
# `enum class inner_blk_t { _, _4a, … }` is legal C++ and DWARF records the `_`
# faithfully. `_sanitize_cpp_type_name` collapsed it to "_" and rstrip'd it to
# EMPTY, emitting `@enum inner_blk_t::Cint begin` + ` = 0` — a syntax error
# that cost onednn's whole 271k-line wrapper (2026-09-20). The C sanitizer had
# an empty guard and the C++ one did not. Repairing it to `_` is not enough:
# Julia's all-underscore identifiers are write-only, so the value-0 member — the
# C++ default — would be unreferenceable.

include(joinpath(@__DIR__, "test_enum_underscore_name.jl"))

# ── Two enums, one member set (no toolchain) ─────────────────────────────────
# DWARF records a std::map's `key_type`/`value_type`/`mapped_type` typedefs as
# separate enums carrying the real enum's enumerators. Both were emitted, the
# later `@enum` rebound every member, and Julia allowed it silently — onednn's
# `dnnl_f32` became a `key_type`, so every ccall taking `dnnl_data_type_t`
# raised MethodError while the module loaded and its suite stayed 15/15
# (2026-09-20). Identical member sets are now aliased, not re-emitted.

include(joinpath(@__DIR__, "test_enum_alias_collision.jl"))

# ── Function names: one derivation, total (no toolchain) ─────────────────────
# fmt instantiates a template on a lambda's closure type, which GNU demangles as
# `…::{lambda(fmt::v12::basic_appender<char>)#1}&>`. The function-name
# replace-list had no catch-all, so `{`/`}`/`#` reached 58 definitions and
# `_assert_wrapper_parses` refused the whole module (2026-09-23). The same wrap
# emitted 8 empty names as `function (this, vis)`, anonymous functions that
# parse and bind nothing. The list was four inline copies plus a fifth; it is
# now `_julia_function_name`, driven here through both generators.

include(joinpath(@__DIR__, "test_function_name_derivation.jl"))

# ── Name and class of local entities and decltype returns (no toolchain) ─────
# The build cut a demangled name at its first `(`. For a lambda that is the
# enclosing function's parameter list, so fmt's write_fixed lambdas took
# write_fixed's name and 10 of them became unreachable. imgui's static lambda
# invokers took a real struct's class and gained a phantom `this`. A
# `decltype (…)` return type left name and class empty (2026-09-24).

include(joinpath(@__DIR__, "test_local_entity_names.jl"))

# ── Symbol hygiene (no toolchain) ────────────────────────────────────────────
# Itanium thunk symbols (`_ZTh`/`_ZTv`/`_ZTc`) have no DWARF subprogram, so
# their "class" is inferred as the demangler's phrase ("non-virtual thunk to
# Derived") and neither receiver gate gives them `this`. They shipped as
# exported zero-argument wrappers that SIGSEGV'd on call (2026-08-13).

include(joinpath(@__DIR__, "test_symbol_hygiene.jl"))

# ── Config surface (no toolchain) ────────────────────────────────────────────
# A TOML key that parses and does nothing is worse than an absent one: it is
# documented, users set it, and nothing reads it. `[ingest] extra_link_libs` was
# documented as loading libraries and only ever serialized — the test that
# "covered" it asserted the value round-tripped through the PARSER, which is
# true whether or not the feature exists. Every field must now be consumed or
# explicitly reserved, and the feature is asserted by EXECUTING what is emitted.

include(joinpath(@__DIR__, "test_config_surface.jl"))

# ── [wrap] exclude_symbols — bounding the wrapper surface ────────────────────
# Glob compilation, the metadata filter, stale-pattern reporting, and config
# round-trip. Pure Julia: no toolchain, no binary.
include(joinpath(@__DIR__, "test_exclude_symbols.jl"))

# ── [wrap] surface_types_only — bounding the wrapper TYPE surface ────────────
# The other half of the same question. exclude_symbols bounds which functions
# the wrapper binds; this bounds which types it defines, and for a C++ library
# with vendored internals those diverge wildly — onednn's DWARF carries 2974
# types and its API reaches 79. The excess is not bloat, it is what breaks the
# module at `include`. Opt-in, because md4c's reachable set is genuinely 2 of
# 36: callback signatures erase their types and it ships 231/231 on the rest.
include(joinpath(@__DIR__, "test_surface_types.jl"))

# ── JSON reads go through _read_json (no toolchain) ──────────────────────────
# A leaked mmap is free on POSIX, where a mapped file still unlinks, and fatal
# on Windows, where it blocks deleting the file. `clean()` failed on a build
# tree RepliBuild had just produced, and the file left behind was
# compilation_metadata.json — not the .dll anyone would have suspected. JSON
# 0.21 maps by default; JSON 1.x rejects the `use_mmap=false` that fixed it.
# The guard is textual because the defect is: a second read path is the
# failure mode, and no single run — Linux, or one JSON major — observes both.

include(joinpath(@__DIR__, "test_json_mmap_hygiene.jl"))

# ── C++ personality is one fact in two languages (no toolchain) ──────────────
# The dialect emits llvm.invoke from its C++ passes AND from the Julia IR
# generator; neither can see the other, so the personality name is written down
# twice. Itanium unwinding calls it __gxx_personality_v0, mingw-w64 x86-64
# unwinds with SEH and calls it __gxx_personality_seh0, and the Windows C++
# runtime exports only the latter — so a drift costs one undefined symbol at
# link time, on one platform, and no Linux run can see it.

include(joinpath(@__DIR__, "test_cxx_personality.jl"))

# ── The C bucket's compiler needs a libc on Windows (toolchain) ──────────────
# .c goes through Clang_unified_jll's clang so its IR is version-locked to
# Julia's libLLVM. That JLL is a bare compiler artifact: right triple, no libc.
# On Linux it silently falls through to /usr/include; on Windows there is
# nothing to fall through to, and every C file dies on its first system
# #include. Only a Windows run can observe it.

include(joinpath(@__DIR__, "test_c_bucket_sysroot.jl"))

# ── The C bucket's clang call drains its output while it runs (JLL clang) ────
# Read after `run`, so more than ~64 KiB of diagnostics blocked clang on a full
# pipe and hung the build silently. Runs in a SIGKILL-bounded child, because the
# regression is a hang.

include(joinpath(@__DIR__, "test_c_bucket_pipe.jl"))

# ── One C type, two tiers, two widths (no toolchain) ─────────────────────────
# C long is 32 bits on Win64 (LLP64) and 64 on Unix64 (LP64) — the one integer
# whose width the word size does not settle. Wrapper emits Julia Clong and is
# right for free; the IRGen producers hardcoded i64 in two hand-filled tables.
# A disagreement raises no error anywhere: the thunk just reads a different
# number of bytes than the caller wrote. The guard compares the two tiers
# rather than restating the constant, so it also catches the next type added
# to one table and not the other.

include(joinpath(@__DIR__, "test_llp64_widths.jl"))

# ── Host-format leftovers the Windows port still had open (no toolchain) ─────
# Library extension, system-header provenance on MSYS2/WinSDK paths, GNU
# binutils identity (not a `GNU` substring), the empty-DWARF diagnostic, wrap_basic
# path escaping, and LLVM_CONFIG without `.exe`.

include(joinpath(@__DIR__, "test_windows_port.jl"))

# ── The thunk manifest is written in JIT mode too (no toolchain) ─────────────
# 1c4255b put the write inside the AOT branch. A failed AOT link falls back to a
# JIT wrap, which then recorded nothing, so the next build emitted the same
# unfiltered thunk set and failed again.

include(joinpath(@__DIR__, "test_thunk_manifest.jl"))

# ── libc++ members the STL path binds are exported (no toolchain) ────────────
# libc++ 23 hides every function it does not annotate, with a pragma that
# applies on COFF too; on MinGW that kept std::vector's members out of the DLL's
# exports, and one unbindable STL thunk failed the library's whole JIT module.
# create_library un-hides them for [types] templates packages, and the STL table
# keeps only symbols the loader can bind.

include(joinpath(@__DIR__, "test_libcxx_exports.jl"))

# ── [platform.<host>] overlays (no toolchain) ────────────────────────────────
# One config builds on Linux and Windows: per-host link libraries, defines and
# dependency excludes merge over the base; an unknown platform name is an error.

include(joinpath(@__DIR__, "test_platform_overlay.jl"))

# ── Config paths are portable (no toolchain) ─────────────────────────────────
# Relative compile paths resolve against the root, not the cwd; save_config
# (and so discover) writes paths under the project relative, with `/`.

include(joinpath(@__DIR__, "test_portable_paths.jl"))

# ── IR -> machine code at the link level (no toolchain) ──────────────────────
# create_library passed no -O, so every library came out of the -O0 backend;
# LLVM 23 crashed there on llama.cpp. Codegen now uses [link]
# optimization_level with the IR pipeline disabled.

include(joinpath(@__DIR__, "test_codegen_flags.jl"))

# ── Link inputs past one Windows command line (no toolchain) ─────────────────
# CreateProcess takes at most 32,767 characters, and llamacpp with ggml-vulkan
# (384 TUs) linked with ~39,000 from the Hub's checkout depth: llvm-link never
# started, the warning blamed it with "exit 1", and the clang fallback died with
# ENAMETOOLONG. Long input lists now go to a response file, and a program that
# never started is reported as BuildBridge.SPAWN_FAILED. The real tools read
# those files in test_link_cmdline_limit.jl (devtests).

include(joinpath(@__DIR__, "test_response_files.jl"))

# ── LLVM tool table probes only what get_tool looks up (no toolchain) ────────
# A 51-name probe at init listed clang-format, mlir-*, FileCheck, clang-20 —
# none of which get_tool or execute ever asked for. BuildBridge.execute
# falls back to the bare command when the table misses, so the prune is
# safe; this file pins the list and that fallback, including that the two
# discover_llvm_tools functions no longer share a name.

include(joinpath(@__DIR__, "test_llvm_tool_table.jl"))

# ── LLVM prefix search: version gating and newest-wins (no toolchain) ────────
# The hand-written prefix ladder ran /usr/lib/llvm-20 down to -15 — every entry
# below the 21 minimum, 21+ absent — and the loop version-checked nothing, so
# the first prefix carrying clang++/llvm-config won regardless of how old it
# was. Driven over fabricated prefixes, so it asserts the rule rather than
# whatever this machine has installed.

include(joinpath(@__DIR__, "test_llvm_prefix_search.jl"))

# ── The toolchain environment rides on the command, not on ENV (no toolchain) ─
# `with_llvm_env` saved, set and restored the process-global ENV around every
# tool BuildBridge ran, and the per-TU compiles run concurrently. Overlapping
# cycles do not nest, so a parallel C++ build deleted the user's
# LD_LIBRARY_PATH and CPATH. The devtests verify children then lost the path
# libJLCS needed (2026-10-02). The toolchain is fabricated, and the children
# are julia.

include(joinpath(@__DIR__, "test_llvm_env_isolation.jl"))

# ── Toolchain advice has one source (no toolchain) ───────────────────────────
# The LLVM floor was stated three times and the install lines four; build.sh
# alone carried two Arch spellings and three Ubuntu ones, including a
# `mlir-21-dev` next to a `${LLVM_MIN_MAJOR}` interpolation. This asserts one
# definition each — not that the advice is right, which no test can know.

include(joinpath(@__DIR__, "test_toolchain_advice.jl"))

# ── Project cache key covers what the compiler reads (no toolchain) ──────────
# The include-dir walk was `readdir`, top level only, so a header one directory
# down could be edited freely while the build reported "project unchanged"
# against modified sources — and `include/<lib>/<lib>.h` is the ordinary C
# layout. Also pins what must stay OUT: .git, build/, cache dirs, non-headers.

include(joinpath(@__DIR__, "test_project_hash.jl"))

# ── Switching toolchains rebuilds what the switch affects (no toolchain) ─────
# A Julia switch changes a C project's compiler (the JLL clang is pinned to
# Julia's libLLVM) and its Tier 1 slices; a wrapper built under 1.13 failed under
# 1.12 at the first call. The keys carry the toolchain, the slice cache is per
# libLLVM major, and a kernel keeps its ccall body under another major (run
# against the C runtime's `abs`).

include(joinpath(@__DIR__, "test_toolchain_switch.jl"))

# ── noexcept routing: a bare name is a candidate, `nounwind` is the proof ────
# A C++ throw through a Tier-3 ccall aborts the process. The routing used to
# trust a regex that returns BARE names, so `B::get` rode along with
# `A::get() noexcept` (and `noexcept(false)` matched too). The IR's `nounwind`
# on the mangled definition now has to agree. Pure text, no toolchain.

include(joinpath(@__DIR__, "test_noexcept_routing.jl"))

# ── By-value records at the Tier-2 thunk boundary (no toolchain) ────────────
# An empty class, a declared-only type, a filtered std type or a template
# spelling used to reach the thunk as `!llvm.ptr`. `record_abi` (DWARF calling
# convention + emptiness) and `FunctionGen.by_value_crossing` decide each one;
# the wrapper traps what the thunk cannot type. Verbatim readelf, text thunks.

include(joinpath(@__DIR__, "test_by_value_crossing.jl"))

# ── Packed C++ structs hold the C layout (no toolchain) ─────────────────────
# `__attribute__((packed))` / `#pragma pack`: the wrapper proves each struct's
# Julia layout against DWARF (blob when it cannot), the thunk reads Julia's
# value at DWARF offsets, StructGen gives the classifier a packed body.

include(joinpath(@__DIR__, "test_packed_layout.jl"))

# ── Tier 2 thunk bodies for byte regions and #pragma pack (no toolchain) ────
# A union or bitfield record crosses as a packed struct of its bytes, never an
# array (abi_fuzz X2); a member is measured as what is emitted (X8).

include(joinpath(@__DIR__, "test_thunk_byte_regions.jl"))

# ── SysV record classes from DWARF facts (no toolchain) ─────────────────────
# `FunctionGen.sysv_record_abi` classifies a by-value record for x86-64 SysV
# from its metadata, not from the body StructGen emits, and states the verdict
# on the call (`sysv_args`, `sysv_ret`) for the dialect's buildCallShape. One
# synthetic record per rule; the live calls are test_sysv_live.jl.

include(joinpath(@__DIR__, "test_sysv_record_abi.jl"))

# ── SIMD vectors by value (no toolchain) ────────────────────────────────────
# DWARF spells a vector as the array of its lanes, and both tiers passed it
# lane by lane: `{<4 x float>}` in two XMM registers, a bare vector as
# `!llvm.array`, and a vector return as `ccall(…, float[4], …)`, which failed
# the whole wrapper's load. A vendored GNU objdump dump, both conventions.

include(joinpath(@__DIR__, "test_simd_vectors.jl"))

# ── Emission does not follow Dict order, the Julia release or DIE offsets ───
# DWARF extraction kept the last definition in a Dict's hash order, GeneratorCpp
# emitted Managed blocks in one, callback examples followed one, and IR file
# names carried `Base.hash`: the same source wrapped differently on Julia 1.12
# and 1.13, and every upstream bump re-rolled it. Synthetic DWARF, text output.

include(joinpath(@__DIR__, "test_emission_determinism.jl"))

# ── Consumers of a function-pointer c_type with a parameter list ────────────
# DWARF function pointers had no parameter lists until the parser collected a
# subroutine type's parameters. With them a c_type holds brackets, words and
# text that its consumers once never saw: array extents, the float-member
# check, surface reachability and the callback docstring.

include(joinpath(@__DIR__, "test_function_pointer_types.jl"))

# ── A name several types share never crosses by value (no toolchain) ────────
# Types are keyed by bare name and one definition is kept per name, so a pimpl
# `impl` per class all got one layout, by-value crossings included. Each such
# crossing (or a record holding one) is now a trap in the thunk generator, both
# Julia generators and the varargs path; an enum only when its widths differ.
# Synthetic DWARF for two units, text output.

include(joinpath(@__DIR__, "test_conflicted_names.jl"))

# ── Build-graph contract (no toolchain) ──────────────────────────────────────
# The data the CMake reader fills and the graph executor compiles from, and
# the one ordering rule. Hand-built graphs.

include(joinpath(@__DIR__, "test_build_graph.jl"))

# ── CMake File API reader (no toolchain) ─────────────────────────────────────
# Vendored codemodel-v2 replies (a fixture project and libcurl), regenerated by
# test/tools/gen_cmake_graph_fixtures.jl, read into the contract above.

include(joinpath(@__DIR__, "test_cmake_graph.jl"))

# ── Graph executor planning (no toolchain) ───────────────────────────────────
# What `GraphBuild` compiles with which flags, what folds into which library,
# what each library links against, and what it refuses, on hand-built graphs.
# Nothing is compiled; test_graph_build_live.jl (devtests) builds and calls one.

include(joinpath(@__DIR__, "test_graph_build.jl"))

# ── Build-time generators of a graph (no toolchain) ─────────────────────────
# Which steps `GraphGenerators` asks the project's ninja to run so a graph's
# generated sources exist, in what order, and when it refuses (a tool the
# project compiles), read off a fake of ninja's tools. The live run is
# test_graph_generators_live.jl (devtests §21).

include(joinpath(@__DIR__, "test_graph_generators.jl"))

# ── Graph mode: a [graph] project end to end (no toolchain) ──────────────────
# The [graph] section, the library module names (`Core` → `RCore`), the graph
# module's text in load order, the link and metadata cache keys, what a build
# asks the generators for, the registry's inputs and vendored files, and the
# Windows preload of a graph library's dependencies, on hand-built graphs. The
# live run is test_graph_mode_live.jl (devtests §22).

include(joinpath(@__DIR__, "test_graph_mode.jl"))

# ── Version is one number, read two ways (no toolchain) ──────────────────────
# `RepliBuild.VERSION` is derived from Project.toml, so this is not tautological:
# `pkgversion` answers from Julia's own package resolution, an independent path.
# Agreement means the file the module read and the file Julia loaded are the same
# one — which is exactly what breaks when a package is dev'd from one checkout
# and loaded from another.
#
# It matters because VERSION feeds `_generator_fingerprint`, which gates the
# registry build cache: a wrong version there serves a stale-codegen wrapper
# instead of rebuilding it.

@testset "Version has a single source of truth" begin
    # Explicit: runtests.jl loads only Test + RepliBuild, and a name that happens
    # to resolve through a dependency today is one refactor from an
    # `UndefVarError` that a try/catch would swallow into a vacuous pass.
    import TOML

    declared = VersionNumber(TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))["version"])
    @test RepliBuild.VERSION == declared
    @test RepliBuild.VERSION == pkgversion(RepliBuild)

    # A hardcoded literal would satisfy the checks above on the day it was
    # written and drift the next time Project.toml moved. Assert the derivation
    # itself, so re-introducing a literal fails here rather than silently later.
    src = read(joinpath(@__DIR__, "..", "src", "RepliBuild.jl"), String)
    @test occursin("VersionNumber(TOML.parsefile(_PROJECT_TOML)", src)
    @test occursin("include_dependency(_PROJECT_TOML)", src)
    @test !occursin(r"const VERSION = v\"", src)
end

# ── Suite wiring guard (no toolchain required) ───────────────────────────────
# A test file that no suite includes is a test that silently never runs. That
# is exactly what happened to test_tier1_dispatch.jl: it shipped with the M3
# slicing commit wired into neither suite and went unnoticed until an audit
# ran it by hand (2026-07-25, 42/42 on first execution). Nothing structural
# prevented it, so: every test_*.jl in test/ must be included by runtests.jl
# or devtests.jl. Adding a file and forgetting to wire it now fails here,
# naming the file.

@testset "Every test file is wired into a suite" begin
    # Collect what the suites actually `include`, NOT every mention of a
    # filename — a comment naming a file (this block names one) must not count
    # as wiring it, or the guard silently passes on the case it exists for.
    function included_basenames(path)
        names = Set{String}()
        for line in eachline(path)
            stripped = lstrip(line)
            (startswith(stripped, '#') || !occursin("include(", stripped)) && continue
            for m in eachmatch(r"\"([^\"]+\.jl)\"", stripped)
                push!(names, basename(m.captures[1]))
            end
        end
        return names
    end

    wired = union(included_basenames(joinpath(@__DIR__, "runtests.jl")),
                  included_basenames(joinpath(@__DIR__, "devtests.jl")))

    # DELIBERATELY UNWIRED — Tier 1 (llvmcall + bitcode slicing) is quarantined.
    # It is the most experimental thing in the repo, it is off by default
    # (`[wrap.tier1] enable = false`, and every Hub config pins
    # `enable_lto = false`), and it ships as a side project rather than as a
    # supported tier. Running its suite means rebuilding slice_test three times
    # for a path nothing takes, so devtests no longer does.
    #
    # This list is the difference between "quarantined" and "silently rotted",
    # which is the exact failure this whole testset exists to prevent —
    # test_tier1_dispatch.jl once shipped wired into neither suite and went
    # unnoticed until an audit. Naming them keeps that impossible: a NEW test
    # file still fails the guard until someone consciously puts it here, and
    # the two asserts below stop this list itself from going stale.
    #
    # To run them (needs the toolchain; each rebuilds test/slice_test/):
    #     julia --project=. test/test_static_promotion.jl   # M1 promotion
    #     julia --project=. test/test_slicer.jl             # M2 slicing
    #     julia --project=. test/test_tier1_dispatch.jl     # M3 dispatch
    # Note they build from test/slice_test/replibuild.toml, which is gitignored
    # and regenerated by nothing — see CLAUDE.md; they are machine-local today.
    # test_tier1_alias_binding.jl is NOT in this list: it is wired into devtests
    # (§7e) on purpose, because it guards the binding of a slice's declares for
    # the Hub recipes that do enable Tier 1.
    experimental = Set([
        "test_static_promotion.jl",
        "test_slicer.jl",
        "test_tier1_dispatch.jl",
    ])

    test_files = sort!(filter(f -> startswith(f, "test_") && endswith(f, ".jl"),
                              readdir(@__DIR__)))
    @test length(test_files) > 10        # readdir sanity — the scan found the suite

    # A quarantined file that no longer exists means the list is stale and is
    # now hiding nothing — fail rather than carry a fiction.
    missing_exempt = filter(f -> !(f in test_files), collect(experimental))
    @test isempty(missing_exempt)

    # …and one that IS wired means the quarantine leaked: devtests would be
    # paying for it while this list claims it does not run.
    leaked = filter(f -> f in wired, collect(experimental))
    @test isempty(leaked)

    unwired = filter(f -> !(f in wired) && !(f in experimental), test_files)
    isempty(unwired) || @warn "Test files included by no suite — they never run" unwired
    @test isempty(unwired)

    # ── ONE SUITE OWNS EACH FILE ─────────────────────────────────────────────
    # "Wired into a suite" was the only rule, so a file could sit in BOTH and
    # satisfy it. `test_registry.jl` did: no suite owned it, it ran twice for
    # anyone running both, and the question "is this a CI test or a toolchain
    # test?" had no answer in the tree — which is the same ambiguity that lets a
    # toolchain dependency drift into CI unnoticed.
    #
    # The split is the contract: runtests.jl is what runs with no C/C++
    # toolchain, devtests.jl is everything that needs one. A file belongs to
    # exactly one of them. If a genuine reason to share ever appears, name the
    # file in SHARED with that reason — the point is that it becomes a decision
    # someone made rather than an accident nobody noticed.
    SHARED = Dict{String,String}()

    ci_files  = included_basenames(joinpath(@__DIR__, "runtests.jl"))
    dev_files = included_basenames(joinpath(@__DIR__, "devtests.jl"))
    overlap   = filter(f -> !haskey(SHARED, f), collect(intersect(ci_files, dev_files)))
    isempty(overlap) || @warn "Included by BOTH suites — give it one owner, or declare it in SHARED with a reason" overlap
    @test isempty(overlap)

    # A stale SHARED entry is a claim about a file that no longer overlaps.
    stale_shared = filter(f -> !(f in intersect(ci_files, dev_files)), collect(keys(SHARED)))
    @test isempty(stale_shared)

    # The scan has to be seeing both suites for any of the above to mean
    # anything — an empty side would make the intersection trivially empty.
    @test length(ci_files)  > 5
    @test length(dev_files) > 5
end

# ── No included file can end the suite early ─────────────────────────────────
# `exit(0)` inside an `include` ends the whole run with a SUCCESS status: every
# file after it silently never runs and the suite still reports green. Six
# toolchain files used it as their "libJLCS not built" skip, and
# test_mlir_templates.jl ran first in the libJLCS group, so a machine without
# the dialect skipped most of devtests and passed. They wrap their body in
# `if requires(...)` now (test/support/TestSupport.jl).
#
# Checked on the parsed AST, not the text: a subprocess driver held in a string
# literal (test_jlcs_invariants.jl has two) legitimately calls `exit`, and a
# comment explaining this rule mentions it. What counts is code that runs at
# include time — top level, testset bodies, `if`/`let`/`for`/`try` blocks and
# `do` blocks — not the body of a function, macro or lambda.

@testset "No include-time exit in a suite-included file" begin
    # Full paths, from every quoted segment of an include line:
    # include(joinpath(TEST_DIR, "callback_test", "test_exceptions.jl")).
    function included_paths(suite)
        paths = String[]
        for line in eachline(suite)
            s = lstrip(line)
            (startswith(s, '#') || !occursin("include(", s)) && continue
            parts = [m.captures[1] for m in eachmatch(r"\"([^\"]+)\"", s)]
            (isempty(parts) || !endswith(parts[end], ".jl")) && continue
            push!(paths, joinpath(@__DIR__, parts...))
        end
        return paths
    end

    is_exit(f) = f === :exit || f == :(Base.exit)
    is_short_def(ex) = ex.head === :(=) && ex.args[1] isa Expr &&
        (ex.args[1].head in (:call, :where) ||
         (ex.args[1].head === :(::) && ex.args[1].args[1] isa Expr &&
          ex.args[1].args[1].head === :call))

    # Walks in source order, carrying the last LineNumberNode so a hit names
    # its line. Returns that line.
    function exits!(hits, ex, line)
        ex isa LineNumberNode && return ex.line
        ex isa Expr || return line
        (ex.head in (:function, :macro, :->, :quote) || is_short_def(ex)) && return line
        if ex.head === :do
            # `f() do x … end` runs its lambda now; scan the body, not the head.
            line = exits!(hits, ex.args[1], line)
            return exits!(hits, ex.args[2].args[2], line)
        end
        ex.head === :call && is_exit(ex.args[1]) && push!(hits, line)
        for a in ex.args
            line = exits!(hits, a, line)
        end
        return line
    end

    function include_time_exits(src::AbstractString)
        ast = Meta.parseall(src)
        bad = filter(a -> a isa Expr && a.head in (:incomplete, :error), ast.args)
        isempty(bad) || error("does not parse: $(first(bad))")
        hits = Int[]
        exits!(hits, ast, 0)
        return hits
    end

    # The walker itself, on shapes whose answer is known. Without these the
    # empty result below could just mean the walk never reached anything.
    @test include_time_exits("exit(0)") == [1]
    @test include_time_exits("if !ok\n    @info \"skip\"\n    exit(0)\nend") == [3]
    @test include_time_exits("@testset \"t\" begin\n  Base.exit(1)\nend") == [2]
    @test include_time_exits("mktempdir() do d\n  exit(0)\nend") == [2]
    @test isempty(include_time_exits("f() = exit(0)"))
    @test isempty(include_time_exits("function g()\n exit(1)\nend"))
    @test isempty(include_time_exits("h = x -> exit(x)"))
    @test isempty(include_time_exits("d = \"\"\"\nexit(3)\n\"\"\""))
    @test isempty(include_time_exits("# exit(0)"))

    files = unique(vcat(included_paths(joinpath(@__DIR__, "runtests.jl")),
                        included_paths(joinpath(@__DIR__, "devtests.jl"))))
    # A path that does not resolve means the reconstruction is wrong, and a
    # file it cannot read is a file it silently did not check.
    @test all(isfile, files)
    @test length(files) > 40

    offenders = Dict{String,Vector{Int}}()
    for f in filter(isfile, files)
        hits = include_time_exits(read(f, String))
        isempty(hits) || (offenders[relpath(f, @__DIR__)] = hits)
    end
    isempty(offenders) || @warn "Include-time exit — use `if requires(...)` from test/support/TestSupport.jl" offenders
    @test isempty(offenders)
end

# ── Every test import is declared, so `Pkg.test()` works ─────────────────────
# `Pkg.test` builds an environment from [deps] plus the [targets] test extras.
# A test file importing anything outside that set loads fine under
# `julia --project=. test/runtests.jl` — the package's own deps are direct there
# — and dies under `Pkg.test()`, which is how CI invokes it. That is exactly
# what happened: test_struct_layout.jl imports Logging, which was neither a dep
# nor an extra, so `Pkg.test()` ran 21 testsets and then errored with
# "Package Logging not found in current path" while the same suite was green
# the way it gets run by hand.
#
# Checking the declaration rather than shelling out to Pkg.test keeps this cheap
# and keeps the failure legible: it names the package and the file.

@testset "Test imports are declared for Pkg.test" begin
    proj = TOML.parsefile(joinpath(dirname(@__DIR__), "Project.toml"))
    deps   = Set(keys(get(proj, "deps", Dict{String,Any}())))
    extras = Set(keys(get(proj, "extras", Dict{String,Any}())))
    target = Set(get(get(proj, "targets", Dict{String,Any}()), "test", String[]))

    # An extra that is not in the test target is not in the test environment.
    @test issubset(target, extras)

    available = union(deps, target, Set(["RepliBuild"]))

    test_files = filter(f -> startswith(f, "test_") && endswith(f, ".jl"),
                        readdir(@__DIR__))
    @test length(test_files) > 10

    undeclared = Dict{String,Vector{String}}()
    for f in test_files
        for line in eachline(joinpath(@__DIR__, f))
            m = match(r"^\s*(?:using|import)\s+([A-Za-z][A-Za-z0-9_]*)", line)
            m === nothing && continue
            pkg = m.captures[1]
            # `using .Foo` / submodules of the package under test are not deps.
            (pkg in available || startswith(pkg, "RepliBuild")) && continue
            push!(get!(() -> String[], undeclared, pkg), f)
        end
    end

    isempty(undeclared) || @warn "Imported by tests but absent from [deps] and [targets].test — Pkg.test() will fail" undeclared
    @test isempty(undeclared)

    # The scan must actually be finding imports, or the assert above is vacuous
    # — the same "asserting into an empty loop" failure this suite guards
    # elsewhere.
    seen = Set{String}()
    for f in test_files, line in eachline(joinpath(@__DIR__, f))
        m = match(r"^\s*(?:using|import)\s+([A-Za-z][A-Za-z0-9_]*)", line)
        m === nothing || push!(seen, m.captures[1])
    end
    @test "Test" in seen
    @test length(seen) >= 5
end
