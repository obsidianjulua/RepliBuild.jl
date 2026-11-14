# test/suites.jl — the one list of every test file: which suite owns it, what
# area of the code it covers, and what it needs built first.
#
# Three suites:
#   pkg   `runtests.jl`: what `Pkg.test()` and registration need — the package
#         loads, VERSION has one source, the config surface, the environment
#         doctor, and the structural guards over this list. Seconds, no toolchain.
#   unit  `unit.jl`: every toolchain-free regression test, in one process.
#   dev   `devtests.jl`: everything that compiles, links or JITs, each file in a
#         process of its own, in parallel (`run.jl`).
#
# A file is in exactly one suite (the guard in runtests.jl reads this list, not
# the suites' include lines). `areas` says which parts of `src/` a file
# exercises, so `run.jl --changed` can run what a diff touches:
#
#   :build    the build engine: Compiler (less DWARF), BuildBridge, LLVMEnvironment,
#             Discovery, DependencyResolver, ConfigurationManager, PackageRegistry,
#             SysConfigGen, VisibilityProbe, ThunkBuilder, the package root
#   :dwarf    DWARF extraction: Compiler's parser, DWARFParser
#   :c        GeneratorC and the shared generator code (Utils, Generator, DispatchLogic)
#   :cpp      GeneratorCpp, the shared generator code, IRGen, Debug
#   :dialect  src/mlir and IRGen: the JLCS dialect and the IR it is fed
#   :graph    BuildGraph, CMakeGraph, GraphGenerators, GraphBuild, GraphMode
#
# `areas_of_path` maps a changed path to areas; a file is selected when it
# shares an area with a changed path. A path no rule knows selects everything.
#
# `locks` are the fixture directories a file builds in or reads from: two jobs
# sharing a lock never run at the same time. `after` names jobs that must have
# finished (and passed) first: a fixture job (`fixture:<name>`) builds and
# verifies one of the discover-driven fixture projects, and a file that reads
# that fixture's wrapper lists it.

const TEST_DIR = @__DIR__

struct TestFile
    file::String
    suite::Symbol
    areas::Vector{Symbol}
    locks::Vector{String}
    after::Vector{String}
    timeout::Int
end

T(file, suite, areas; locks = String[], after = String[], timeout = 1800) =
    TestFile(file, suite, Symbol[areas...], String[locks...], String[after...], timeout)

# The discover-driven fixture projects: discover(force) → curated config → build
# → wrap → verify.jl in a child. Each is one job, `fixture:<name>`, that locks
# its own directory. `test/support/fixture.jl <name>` runs one.
struct Fixture
    name::String
    label::String
    areas::Vector{Symbol}
    curated::Dict{String,Any}   # [section][key] = value, seeded after discover
end

const FIXTURES = Fixture[
    Fixture("stress_test", "Stress Test (numerics, vtable, RAII, MLIR)", [:cpp, :dialect], Dict()),
    Fixture("mi_test", "Multiple Inheritance (two-base layout, upcasts)", [:cpp, :dialect], Dict()),
    Fixture("vi_test", "Virtual Inheritance (diamond, dynamic vbase upcasts)", [:cpp, :dialect], Dict()),
    Fixture("stl_test", "STL Templates (vector, string, map)", [:cpp, :dialect],
            Dict("types" => Dict("templates" => ["std::vector<int>", "std::string", "std::map<int, int>"],
                                 "template_headers" => ["<vector>", "<string>", "<map>"]))),
    Fixture("c_test", "C Fundamentals (structs, enums, LTO, packed, unions)", [:c, :build], Dict()),
    Fixture("c_abomination_test", "C Edge Cases (opaque structs, nested callbacks)", [:c], Dict()),
    Fixture("callback_test", "Callbacks (Julia ↔ C++)", [:cpp, :dialect], Dict()),
]

const TESTS = TestFile[
    # ── unit: toolchain-free regression tests (runtests.jl's old include list) ──
    T("test_ingest.jl", :unit, [:build]),
    T("test_registry.jl", :unit, [:build]),
    T("test_clean.jl", :unit, [:build]),
    T("test_discover_without_llvm.jl", :unit, [:build]),
    T("test_user_messages.jl", :unit, [:build]),
    T("test_varargs_emission.jl", :unit, [:c]),
    T("test_toml_preservation.jl", :unit, [:build]),
    T("test_compile_visibility.jl", :unit, [:build]),
    T("test_c_generator_policies.jl", :unit, [:c]),
    T("test_c_sret_classification.jl", :unit, [:c]),
    T("test_record_natural_layout.jl", :unit, [:c, :cpp]),
    T("test_cpp_mutable_records.jl", :unit, [:cpp]),
    T("test_cpp_finalizers.jl", :unit, [:cpp]),
    T("test_dwarf_member_sizes.jl", :unit, [:dwarf]),
    T("test_shim_header_guard.jl", :unit, [:build, :c]),
    T("test_shim_defines.jl", :unit, [:build, :c]),
    T("test_stl_extract.jl", :unit, [:cpp]),
    T("test_header_enum_provenance.jl", :unit, [:build, :c, :cpp]),
    T("test_dep_cache.jl", :unit, [:build]),
    T("test_dep_submodules.jl", :unit, [:build]),
    T("test_git_progress.jl", :unit, [:build]),
    T("test_dep_archive.jl", :unit, [:build]),
    T("test_wrapper_type_bindings.jl", :unit, [:c, :cpp]),
    T("test_struct_layout.jl", :unit, [:dialect, :cpp]),
    T("test_template_members.jl", :unit, [:dialect, :cpp]),
    T("test_toolchain_records.jl", :unit, [:dwarf, :cpp]),
    T("test_std_strings.jl", :unit, [:dwarf, :cpp]),
    T("test_record_scope.jl", :unit, [:dwarf, :cpp]),
    T("test_integer_width_overloads.jl", :unit, [:cpp]),
    T("test_tier2_typed_dispatch.jl", :unit, [:cpp]),
    T("test_cpp_return_resolution.jl", :unit, [:cpp]),
    T("test_typed_pointer_returns.jl", :unit, [:c, :cpp]),
    T("test_tier2_scalar_widths.jl", :unit, [:cpp]),
    T("test_blob_setters.jl", :unit, [:c, :cpp]),
    T("test_export_hygiene.jl", :unit, [:c, :cpp]),
    T("test_cstring_policy.jl", :unit, [:c, :cpp]),
    T("test_introspection.jl", :unit, [:c, :cpp]),
    T("test_enum_underscore_name.jl", :unit, [:c, :cpp]),
    T("test_enum_alias_collision.jl", :unit, [:c, :cpp]),
    T("test_open_enums.jl", :unit, [:c, :cpp]),
    T("test_enumerator_collisions.jl", :unit, [:c, :cpp]),
    T("test_function_name_derivation.jl", :unit, [:c, :cpp]),
    T("test_local_entity_names.jl", :unit, [:dwarf]),
    T("test_symbol_hygiene.jl", :unit, [:build, :cpp]),
    T("test_config_surface.jl", :unit, [:build]),
    T("test_exclude_symbols.jl", :unit, [:build, :c, :cpp]),
    T("test_surface_types.jl", :unit, [:c, :cpp]),
    T("test_json_mmap_hygiene.jl", :unit, [:build]),
    T("test_cxx_personality.jl", :unit, [:dialect]),
    T("test_c_bucket_sysroot.jl", :unit, [:build]),
    T("test_c_bucket_pipe.jl", :unit, [:build]),
    T("test_llp64_widths.jl", :unit, [:dialect, :cpp]),
    T("test_windows_port.jl", :unit, [:build, :cpp]),
    T("test_thunk_manifest.jl", :unit, [:cpp]),
    T("test_jit_runtime_symbols.jl", :unit, [:build, :cpp]),
    T("test_libcxx_exports.jl", :unit, [:build, :cpp]),
    T("test_platform_overlay.jl", :unit, [:build]),
    T("test_portable_paths.jl", :unit, [:build]),
    T("test_codegen_flags.jl", :unit, [:build]),
    T("test_response_files.jl", :unit, [:build]),
    T("test_llvm_tool_table.jl", :unit, [:build]),
    T("test_llvm_prefix_search.jl", :unit, [:build]),
    T("test_llvm_env_isolation.jl", :unit, [:build]),
    T("test_toolchain_advice.jl", :unit, [:build]),
    T("test_project_hash.jl", :unit, [:build]),
    T("test_compile_never_ran.jl", :unit, [:build]),
    T("test_toolchain_switch.jl", :unit, [:build, :c]),
    T("test_noexcept_routing.jl", :unit, [:cpp]),
    T("test_by_value_crossing.jl", :unit, [:dialect, :cpp]),
    T("test_packed_layout.jl", :unit, [:dialect, :cpp]),
    T("test_thunk_byte_regions.jl", :unit, [:dialect, :cpp]),
    T("test_sysv_record_abi.jl", :unit, [:dialect, :cpp]),
    T("test_aapcs64_record_abi.jl", :unit, [:dialect, :c, :cpp]),
    T("test_simd_vectors.jl", :unit, [:dwarf, :c, :cpp]),
    T("test_emission_determinism.jl", :unit, [:dwarf, :c, :cpp]),
    T("test_function_pointer_types.jl", :unit, [:c, :cpp]),
    T("test_conflicted_names.jl", :unit, [:dwarf, :c, :cpp]),
    T("test_build_graph.jl", :unit, [:graph]),
    T("test_cmake_graph.jl", :unit, [:graph]),
    T("test_graph_build.jl", :unit, [:graph]),
    T("test_graph_generators.jl", :unit, [:graph]),
    T("test_graph_mode.jl", :unit, [:graph]),
    T("test_cross_library_types.jl", :unit, [:graph, :c, :cpp]),

    # ── dev: compiles, links or JITs ─────────────────────────────────────────
    # The pipeline end to end on stress_test, and ingest on c_test's library.
    T("test_pipeline_live.jl", :dev, [:build], locks = ["stress_test"], after = ["fixture:stress_test"]),
    T("test_jit_runtime_symbols_live.jl", :dev, [:build, :cpp], locks = ["stl_test"], after = ["fixture:stl_test"]),
    T("test_ingest_live.jl", :dev, [:build, :c], locks = ["c_test"], after = ["fixture:c_test"]),
    T("test_registry_local_first.jl", :dev, [:build], locks = ["c_test"], after = ["fixture:c_test"]),
    # The dialect: templates, exceptions, unwinding, AOT, invariants, producers, ABI.
    T("test_mlir_templates.jl", :dev, [:dialect]),
    T(joinpath("callback_test", "test_exceptions.jl"), :dev, [:cpp, :dialect],
      locks = ["callback_test"], after = ["fixture:callback_test"]),
    T("test_jit_unwind.jl", :dev, [:dialect]),
    T("test_aot_thunks.jl", :dev, [:cpp, :dialect], locks = ["aot_test"]),
    T("test_jlcs_invariants.jl", :dev, [:dialect]),
    T("test_win64_abi.jl", :dev, [:dialect]),
    T("test_jlcs_producers.jl", :dev, [:dialect]),
    T("test_struct_abi.jl", :dev, [:dialect, :cpp]),
    T("test_sysv_abi_oracle.jl", :dev, [:dialect]),
    T("test_aapcs64_abi_spec.jl", :dev, [:dialect]),
    T("test_sysv_live.jl", :dev, [:dialect, :cpp], locks = ["sysv_live_test"]),
    T("test_simd_vectors_live.jl", :dev, [:dwarf, :c, :cpp], locks = ["simd_live_test"]),
    T("test_multilib_jit.jl", :dev, [:cpp], locks = ["mi_test", "vi_test"],
      after = ["fixture:mi_test", "fixture:vi_test"]),
    T("test_debug_inspection.jl", :dev, [:cpp], locks = ["mi_test", "vi_test"],
      after = ["fixture:mi_test", "fixture:vi_test"]),
    # The build engine, live.
    T("test_wrap_surface_guard.jl", :dev, [:build]),
    T("test_visibility_probe.jl", :dev, [:build]),
    T("test_c_inprocess.jl", :dev, [:build, :c], locks = ["c_test"], after = ["fixture:c_test"]),
    T("test_tier1_alias_binding.jl", :dev, [:build], locks = ["tier1_alias_test"]),
    T("test_shim_defines_live.jl", :dev, [:build, :c], locks = ["shim_defines_test"]),
    T("test_cache_invalidation.jl", :dev, [:build]),
    T("test_build_markers_live.jl", :dev, [:build]),
    T("test_reproducible_link.jl", :dev, [:build]),
    T("test_dwarf_attribution.jl", :dev, [:dwarf]),
    T("test_sysconfiggen.jl", :dev, [:build]),
    T("test_link_cmdline_limit.jl", :dev, [:build]),
    # By-value records, live, on both tiers.
    T("test_abi_nested.jl", :dev, [:c], locks = ["abi_nested_test"]),
    T("test_c_sret_abi.jl", :dev, [:c], locks = ["c_sret_test"]),
    T("test_cpp_record_routing.jl", :dev, [:cpp], locks = ["cpp_routing_test"]),
    T("test_sret_in_caller_buffer.jl", :dev, [:dialect, :cpp], locks = ["sret_inplace_test"]),
    T("test_determinism_fixture.jl", :dev, [:dwarf, :c, :cpp], locks = ["determinism_test"]),
    T("test_conflict_trap_fixture.jl", :dev, [:dwarf, :c, :cpp],
      locks = ["conflict_trap_test", "conflict_trap_c_test"]),
    T("test_cpp_finalizers_live.jl", :dev, [:cpp], locks = ["finalizer_test"]),
    T("test_toolchain_byvalue_live.jl", :dev, [:dwarf, :cpp], locks = ["toolchain_byvalue_test"]),
    T("test_convenience_overloads.jl", :dev, [:cpp], locks = ["convenience_overload_test"]),
    T("test_anonymous_unions.jl", :dev, [:c], locks = ["c_abomination_test"], after = ["fixture:c_abomination_test"]),
    # Graph mode.
    T("test_graph_build_live.jl", :dev, [:graph], locks = ["graph_build_test"]),
    T("test_graph_generators_live.jl", :dev, [:graph], locks = ["graph_generators_test"]),
    T("test_graph_mode_live.jl", :dev, [:graph], locks = ["graph_mode_test"]),
    T("test_cross_library_types_live.jl", :dev, [:graph, :cpp], locks = ["xlib_test"]),
]

# Quarantined: Tier 1's own suites (CLAUDE.md). Wired into no suite on purpose;
# runtests.jl checks they exist and that nothing lists them.
const EXPERIMENTAL = ["test_static_promotion.jl", "test_slicer.jl", "test_tier1_dispatch.jl"]

# The whole repository is one change for these: a diff that touches them runs
# every suite it asks for.
const EVERYTHING_PATHS = ["Project.toml", "Manifest.toml", "deps/", "test/support/", "test/run.jl",
                          "test/suites.jl", "test/runtests.jl", "test/unit.jl", "test/devtests.jl",
                          ".github/", "src/RepliBuild.jl"]
# Paths no test depends on: documentation, the changelog, the rulebook, repo metadata.
const NOTHING_PATHS = ["docs/", "README.md", "CHANGELOG.md", "CLAUDE.md", "LICENSE", ".gitignore",
                       ".claude/", ".gitattributes", "examples/"]

const GRAPH_FILES = ("src/Builder/BuildGraph.jl", "src/Builder/CMakeGraph.jl", "src/Builder/GraphGenerators.jl",
                     "src/Builder/GraphBuild.jl", "src/Builder/GraphMode.jl")
const DWARF_FILES = ("src/Builder/Compiler.jl", "src/Builder/DWARFParser.jl")

"""
    areas_of_path(path) -> Vector{Symbol} | :all | Symbol[]

The areas a change to `path` (repository-relative, `/`-separated) exercises.
`:all` for a path everything depends on, an empty vector for a path no test
needs to run for (docs, the Hub guide). Graph files are `:graph` alone; the
DWARF sources are `:dwarf` and both generators (they read what they parse);
the other builder files are `:build`; the shared generator code is both
generators; IRGen and the dialect are `:dialect` and `:cpp`.
"""
function areas_of_path(path::AbstractString)
    p = replace(String(path), '\\' => '/')
    any(e -> p == e || startswith(p, e), EVERYTHING_PATHS) && return :all
    any(e -> p == e || startswith(p, e), NOTHING_PATHS) && return Symbol[]
    p in GRAPH_FILES && return [:graph]
    p in DWARF_FILES && return [:dwarf, :build, :c, :cpp]
    startswith(p, "src/Builder/") && return [:build]
    startswith(p, "src/Wrapper/C/") && return [:c]
    startswith(p, "src/Wrapper/Cpp/") && return [:cpp]
    startswith(p, "src/Wrapper/") && return [:c, :cpp]
    startswith(p, "src/IRGen/") && return [:dialect, :cpp]
    startswith(p, "src/mlir/") && return [:dialect, :cpp]
    startswith(p, "src/Debug/") && return [:cpp]
    startswith(p, "src/") && return :all
    endswith(p, ".md") && return Symbol[]
    startswith(p, "test/") && return Symbol[]      # test files select themselves (run.jl)
    return :all
end
