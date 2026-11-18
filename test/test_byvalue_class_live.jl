# =============================================================================
# A class with a destructor, taken by value through a thunk, live (2026-11-17)
#
# `struct Res { int *p; Res(const Res&); ~Res(); }; int take(Res r);` and
# nothing else: `wrap` succeeded and `using` the wrapper killed Julia. The
# thunk for `take` copy-constructs a temporary and ends it (`jlcs.ctor_call`,
# `jlcs.scope`), naming the complete constructor and destructor (`C1`, `D1`);
# their only declarations were their own thunks', the wrapper binds `C2` and
# `D2`, and a thunk the wrapper does not bind is not emitted. The module
# called symbols it never declared, and MLIR's translation to LLVM IR
# dereferenced null.
#
# Behind that: the thunk read "no copy-constructor symbol" as "the copy is
# trivial", so a move-only class was passed as a second copy of its bytes and
# that copy destroyed, which freed the caller's object; and a class in a
# namespace was never found for its parameter at all, so it went as the bytes
# behind the slot with no constructor and no destructor.
#
# This builds test/byvalue_class_test, wraps it, and from a child loads the
# wrapper as wrap left it (manifest included) and passes each class by value:
# a copy that leaves the caller's object alone, a move that ends the resource
# once, the bytes only for a trivial copy, and an ABI trap that says so where
# no temporary can be made. A second child does the same through AOT thunks,
# and a third hands the JIT the module as it used to be generated and expects
# one library's failed initialization, by name, from a live process.
# Needs clang++ and libJLCS.
# =============================================================================

using Test
using Libdl
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const BYVALUE_CLASS_DIR = joinpath(@__DIR__, "byvalue_class_test")
const BYVALUE_CLASS_PROBE = joinpath(BYVALUE_CLASS_DIR, "probe_bv.jl")
const BYVALUE_CLASS_PROBES = ["thunks_loaded", "copy_keeps_the_callers", "move_consumes_once",
    "no_copy_no_move_is_refused", "private_copy_is_refused", "trivial_copy_is_bytes",
    "defaulted_copy_is_bytes", "implicit_nontrivial_copy_is_refused", "copy_constructor_runs",
    "virtual_destructor", "implicit_virtual_destructor", "aligned_temporary", "nested_class", "namespaced_copy", "namespaced_move", "template_class",
    "shared_name_is_refused", "two_temporaries"]

# Regenerated each run: it holds absolute, machine-local paths.
function byvalue_class_toml(aot::Bool)
    path = joinpath(BYVALUE_CLASS_DIR, "replibuild.toml")
    write(path, """
    [project]
    name = "byvalue_class"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(BYVALUE_CLASS_DIR))"

    [compile]
    flags = ["-std=c++17", "-fPIC"]
    source_files = ["$(escape_string(joinpath(BYVALUE_CLASS_DIR, "src", "bv.cpp")))"]
    include_dirs = ["$(escape_string(joinpath(BYVALUE_CLASS_DIR, "include")))"]
    aot_thunks = $(aot)

    [link]
    enable_lto = false
    optimization_level = "0"

    [binary]
    type = "shared"

    [wrap]
    language = "cpp"

    [types]
    strictness = "warn"
    allow_unknown_structs = true

    [cache]
    enabled = false
    """)
    return path
end

function byvalue_class_check_probe(r)
    v = probe_verdicts(r.output)
    @test r.ok
    @test haskey(v, "PROBE_DONE")
    for n in BYVALUE_CLASS_PROBES
        @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
    end
    @test sort!(collect(setdiff(keys(v), ("PROBE_DONE",)))) == sort(BYVALUE_CLASS_PROBES)
end

if requires("a class taken by value through a thunk, live", :libJLCS, :clangxx)
@testset "A class taken by value through a thunk, live" begin
    for d in ("build", "julia", ".replibuild_cache", ".debug")
        p = joinpath(BYVALUE_CLASS_DIR, d)
        ispath(p) && RepliBuild._rm_tree(p)
    end
    toml_path = byvalue_class_toml(false)
    redirect_stdout(devnull) do
        RepliBuild.build(toml_path)
    end
    wrapper = RepliBuild.wrap(toml_path)
    @test isfile(wrapper)
    out = dirname(wrapper)
    lib = joinpath(out, "libbyvalue_class." * Libdl.dlext)
    @test isfile(lib)
    txt = read(wrapper, String)
    md = RepliBuild._read_json(joinpath(out, "compilation_metadata.json"))

    @testset "the wrapper: a call where a temporary can be made, a trap that says why where it cannot" begin
        body(name) = (m = match(Regex("\nfunction " * name * "\\([^\n]*\n(.*?)\nend\n", "s"), txt);
                      m === nothing ? "no function $name" : String(m.captures[1]))
        for name in ("take_res", "consume_uniq", "take_plain", "take_dflt", "take_tracked", "take_poly", "take_wide", "take_vkid",
                     "take_inner", "ns_take_nres", "ns_consume_nuniq", "ns_take_box", "take_both")
            @test (name, occursin("ABI Safety Trap", body(name)), occursin("_mlir_ciface_", body(name))) == (name, false, true)
        end
        for (name, part) in (("take_pinned", "its copy constructor is deleted and it has no move constructor"),
                             ("take_guarded", "its copy constructor is private"),
                             ("take_holder", "its copy constructor is implicit and not trivial"),
                             ("take_twin", "distinct classes in this library are named `Twin` (one::Twin; two::Twin)"))
            b = body(name)
            @test (name, occursin("ABI Safety Trap", b), occursin("cannot be passed by value", b), occursin(part, b)) ==
                  (name, true, true, true)
        end
    end

    @testset "the thunk module wrap's manifest asks for declares everything it calls, once" begin
        manifest = Set{String}(RepliBuild._read_json(joinpath(out, "thunk_manifest.json"))["function_thunks"])
        vt = RepliBuild.DWARFParser.parse_vtables(lib)
        J = RepliBuild.JLCSIRGenerator
        called(ir) = (s = Set{String}();
                      for m in eachmatch(r"dtors\(\[([^\]]*)\]\)", ir), x in eachmatch(r"@([^\s,\]]+)", m.captures[1])
                          push!(s, String(x.captures[1]))
                      end;
                      for m in eachmatch(r"jlcs\.ctor_call @([^\s(]+)\(", ir)
                          push!(s, String(m.captures[1]))
                      end; s)
        own = RepliBuild.Compiler._runtime_symbol_names(lib)
        @test own !== nothing
        own = something(own, Set{String}())
        # VKid's destructor is implicit and virtual: its deleting variant is in
        # the library, and its temporary is ended by the complete-object one
        # where a toolchain emits it, else by the base-object one (clang on
        # ELF emits no D1 for it). Never by D0.
        @test "_ZN4VKidD0Ev" in own
        vkid_dtor = "_ZN4VKidD1Ev" in own ? "_ZN4VKidD1Ev" : "_ZN4VKidD2Ev"
        @test vkid_dtor in own
        @info "VKid's temporary is ended by" vkid_dtor
        # The complete constructors and destructors of every class a thunk
        # makes a temporary of. Itanium names, on every host.
        want = Set(["_ZN3ResC1ERKS_", "_ZN3ResD1Ev", "_ZN4UniqC1EOS_", "_ZN4UniqD1Ev", "_ZN5PlainD1Ev", "_ZN4DfltD1Ev",
                    "_ZN4VKidC1ERKS_", vkid_dtor,
                    "_ZN7TrackedC1ERKS_", "_ZN7TrackedD1Ev", "_ZN4PolyC1ERKS_", "_ZN4PolyD1Ev",
                    "_ZN4WideC1ERKS_", "_ZN4WideD1Ev",
                    "_ZN5Outer5InnerC1ERKS0_", "_ZN5Outer5InnerD1Ev", "_ZN2ns4NResC1ERKS0_", "_ZN2ns4NResD1Ev",
                    "_ZN2ns5NUniqC1EOS0_", "_ZN2ns5NUniqD1Ev", "_ZN2ns3BoxIiEC1ERKS1_", "_ZN2ns3BoxIiED1Ev"])
        # As wrap left the manifest; with every function thunked; and with the
        # temporaries' own constructors and destructors in the manifest, as a
        # wrapper that binds the complete-object variants has them. Each is
        # declared once in all three: by its thunk where it has one, else by
        # the module. Twice is a redefinition, and no thunk of the library.
        for needed in (manifest, nothing, union(manifest, want))
            ir = J.generate_jlcs_ir(vt, deepcopy(md); needed_symbols = needed === nothing ? nothing : copy(needed))
            @test called(ir) == want
            @test J.undeclared_callees(ir) == String[]
            counts = [(s, count("func.func private @$(s)(", ir) + count("llvm.func @$(s)(", ir)) for s in sort!(collect(want))]
            @test counts == [(s, 1) for s in sort!(collect(want))]
            names = [String(m.captures[1]) for m in eachmatch(r"(?:func\.func(?: private)?|llvm\.func) @([^\s(]+)\(", ir)]
            @test allunique(names)
            # No scope ends a temporary with a deleting destructor.
            @test !occursin(r"dtors\(\[[^\]]*D0Ev", ir)
        end
        # Every temporary is aligned as an object: 16 where DWARF says no more.
        ir = J.generate_jlcs_ir(vt, deepcopy(md); needed_symbols = copy(manifest))
        allocas = [String(m.match) for m in eachmatch(r"%raii_tmp_\d+ = llvm\.alloca [^\n]*", ir)]
        @test length(allocas) >= length(want) ÷ 2
        @test all(a -> occursin("{alignment = 16}", a), allocas)
        # Each is the library's own, which is what the pre-flight asks.
        @test issubset(want, own)
        callees = RepliBuild.JITManager._module_callees(md, manifest)
        @test issubset(want, callees)
    end

    @testset "every class, from a child: the wrapper loads and each temporary is made once and ended once" begin
        byvalue_class_check_probe(run_isolated(`$BYVALUE_CLASS_PROBE $wrapper jit`; timeout = 900))
    end

    # The module as it used to be generated: the declarations of the
    # temporaries' constructors and destructors taken back out. The generator
    # no longer writes it, so the check in front of the JIT is driven directly.
    # Without that check this child dies in `create_jit`.
    @testset "a module calling an undeclared symbol fails one library's JIT, by name, and the process lives" begin
        script = """
        using RepliBuild
        const J = RepliBuild.JLCSIRGenerator
        function J.generate_jlcs_ir(vt::RepliBuild.DWARFParser.VtableInfo, md::AbstractDict; kw...)
            ir = invoke(J.generate_jlcs_ir, Tuple{RepliBuild.DWARFParser.VtableInfo, Any}, vt, md; kw...)
            i = findfirst("// Constructors and destructors of by-value temporaries", ir)
            i === nothing && return ir
            j = findnext("\\n\\n", ir, i.stop)
            return ir[1:prevind(ir, i.start)] * ir[j.stop:end]
        end
        const M = include($(repr(wrapper)))
        check(name, ok, detail) = println("PROBE ", name, ": ", ok ? "PASS" : "FAIL " * string(detail))
        eng = only(RepliBuild.JITManager.GLOBAL_JIT.engines)
        msg = eng.init_error === nothing ? "no init error" : sprint(showerror, eng.init_error)
        check("init_failed_not_crashed", eng.init_error !== nothing && eng.jit_engine === nothing, msg)
        check("names_the_symbols", occursin("never declares", msg) && occursin("_ZN3ResD1Ev", msg) &&
                                   occursin("_ZN4UniqC1EOS_", msg), msg)
        err = try
            M.take_res(M.Res(C_NULL)); "returned"
        catch e
            sprint(showerror, e)
        end
        check("calls_raise", occursin("init failed", err) || occursin("JIT", err), err)
        println("PROBE_DONE")
        """
        r = run_isolated(`-e $script`; timeout = 900)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        for n in ("init_failed_not_crashed", "names_the_symbols", "calls_raise")
            @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
        end
    end

    @testset "the same through AOT thunks: wrap compiles the module, and a child calls through it" begin
        aot_toml = byvalue_class_toml(true)
        r = run_isolated(`$BYVALUE_CLASS_PROBE $aot_toml aot`; timeout = 1200)
        byvalue_class_check_probe(r)
        thunks = joinpath(out, "libbyvalue_class_thunks." * Libdl.dlext)
        @test isfile(thunks)
        aot_txt = isfile(wrapper) ? read(wrapper, String) : ""
        @test occursin("JITManager.invoke_aot_ptr(", aot_txt)
        @test !occursin("initialize_global_jit", aot_txt)
    end
end
end # requires
