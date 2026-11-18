# =============================================================================
# Pure virtual member functions, live (2026-11-18)
#
# test_pure_virtual_methods.jl (unit) checks what the parser, the build's plan,
# the thunk generator and the wrapper generator make of a vendored dump. This
# builds test/pure_virtual_test with the system compiler, wraps it, and calls
# it from a child, once through the JIT and once with `[compile] aot_thunks`:
#
#   * `Animal_name(p)` is "dog", "puppy", "bird", "duck" for the four objects
#     `make_animal` returns as `Animal*`: the function has no symbol, and its
#     thunk reads the object's vtable. The factory struct has the proxy.
#   * An override that throws arrives as `JITManager.CxxException`.
#   * A private or protected pure virtual, a variadic one, one of a class in
#     an unnamed namespace and a pure virtual destructor get no wrapper; one
#     that takes or returns a record by value is
#     a refusal that says so; so is the constructor of every abstract class,
#     which leaves its storage untouched.
#   * The build's surface guard does not look for a pure virtual's name, and
#     does not count one as a surface.
#
# Nothing here loads the fixture's libraries in this process.
# Needs clang++ and libJLCS.
# =============================================================================

using Test
using RepliBuild
import JSON
import Libdl

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())

const PURE_VIRTUAL_DIR = joinpath(@__DIR__, "pure_virtual_test")

# The pure virtuals the wrapper binds, by the declaration's linkage name
# (Itanium on every host), and the ones the build passes over.
const PV_LIVE_BOUND = ["_ZNK6Animal4nameEv", "_ZNK6Animal5speedEd", "_ZNK6Animal4rankEihb", "_ZN6Animal6renameEPKc",
                       "_ZNK6Animal5riskyEi", "_ZN8Listener8on_eventEi", "_ZNK8Listener8priorityEv",
                       "_ZNK6Logger5linesEv", "_ZNK5Shape4areaEv", "_ZNK5Shape5sidesEv",
                       "_ZNK3zoo6Keeper5feedsERK6Animal", "_ZNK3zoo6Keeper9favouriteEv"]
const PV_LIVE_REFUSED = ["_ZNK6Animal4spanEv", "_ZNK6Animal5weighE4Span", "_ZNK6Animal4blobEv"]
const PV_LIVE_SKIPPED = ["_ZNK6Animal6secretEv" => "protected", "_ZNK6Animal6hiddenEv" => "private",
                         "_ZN6Logger3logEPKcz" => "variadic", "_ZN5TokenD4Ev" => "a destructor",
                         "_ZNK12_GLOBAL__N_14Step4stepEi" => "its class has internal linkage"]

const PV_LIVE_PROBES = ["name_through_the_base", "scalar_in_scalar_out", "integers_narrower_than_int",
                        "void_return_and_pointer_argument", "exception_from_the_override",
                        "defined_members_unchanged", "factory_struct_proxies",
                        "private_protected_variadic_and_internal_have_no_wrapper",
                        "pure_virtual_destructor_is_bound_by_symbol_only", "record_by_value_is_refused",
                        "abstract_class_is_not_constructed", "concrete_subclass_built_in_place",
                        "interface_with_no_symbol_at_all", "pure_virtual_with_a_definition",
                        "namespace_reference_bool_and_typed_pointer", "nothing_leaked"]

function pv_live_toml(path; aot::Bool)
    # Regenerated each run: it holds absolute, machine-local paths.
    write(path, """
    [project]
    name = "pvirt"
    root = "$(escape_string(PURE_VIRTUAL_DIR))"

    [compile]
    flags = ["-std=c++17", "-fPIC"]
    source_files = ["$(escape_string(joinpath(PURE_VIRTUAL_DIR, "src", "pv.cpp")))"]
    include_dirs = ["$(escape_string(joinpath(PURE_VIRTUAL_DIR, "include")))"]
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

# The exported names of a library, read from the file.
function pv_live_exports(path)
    names = RepliBuild.Compiler._pe_exported_names(path)
    names === nothing || return names
    return RepliBuild.Compiler._nm_dynamic_names(read(RepliBuild.Tools.tool_cmd(`nm -D --defined-only $path`), String))
end

function pv_live_child(wrapper, mode, probes)
    r = run_isolated(`$(joinpath(PURE_VIRTUAL_DIR, "probe_pure_virtual.jl")) $wrapper $mode`)
    v = probe_verdicts(r.output)
    @test r.ok
    @test haskey(v, "PROBE_DONE")
    for n in probes
        @test (mode, n, get(v, n, "no PROBE line")) == (mode, n, "PASS")
    end
    # No case the script prints is left unasserted.
    @test sort!(collect(setdiff(keys(v), ("PROBE_DONE",)))) == sort(probes)
    return r.output
end

if requires("pure virtual member functions, live", :libJLCS, :clangxx)
@testset "Pure virtual member functions, live" begin
    for name in ("build", "julia", ".replibuild_cache", ".debug")
        p = joinpath(PURE_VIRTUAL_DIR, name)
        ispath(p) && RepliBuild._rm_tree(p)
    end
    toml_path = pv_live_toml(joinpath(PURE_VIRTUAL_DIR, "replibuild.toml"); aot = false)
    # The build's surface guard runs here: R2 would name every pure virtual
    # as unreachable if it looked for one.
    redirect_stdout(devnull) do
        RepliBuild.build(toml_path)
    end
    wrapper = RepliBuild.wrap(toml_path)
    @test isfile(wrapper)
    code = read(wrapper, String)
    out = dirname(wrapper)
    library = joinpath(out, "libpvirt." * Libdl.dlext)
    md_path = joinpath(out, "compilation_metadata.json")
    md = RepliBuild._read_json(md_path)
    fns = Dict{String,Any}(String(f["mangled"]) => f for f in md["functions"])
    thunks = Set{String}(String.(RepliBuild._read_json(joinpath(out, "thunk_manifest.json"))["function_thunks"]))
    exported = pv_live_exports(library)
    body(name) = (m = match(Regex("^function $name\\([^\\n]*\\)[^\\n]*\\n.*?^end", "ms"), code); m === nothing ? "" : String(m.match))
    defined(name) = occursin(Regex("^function $name\\(", "m"), code)

    @testset "the fixture has the shapes it is built for" begin
        # No pure virtual is a symbol, but the one that is defined as well.
        for sym in vcat(PV_LIVE_BOUND, PV_LIVE_REFUSED, first.(PV_LIVE_SKIPPED))
            @test (sym, sym in exported) == (sym, false)
        end
        @test "_ZNK4Base3tagEv" in exported
        # An abstract class has its base-object constructor only.
        @test "_ZN6AnimalC2Ei" in exported && !("_ZN6AnimalC1Ei" in exported)
        @test "_ZN6SquareC1Ei" in exported
        # `Listener` and `Logger` define no member function at all, and their
        # DIEs are still definitions, with the members' declarations.
        for cls in ("Animal", "Listener", "Logger", "Token", "Shape", "Poly", "Base", "Keeper")
            rows = get(get(md["record_abi"], cls, Dict()), "methods", Any[])
            @test (cls, any(r -> get(r, "pure", false) == true, rows) || cls == "Poly") == (cls, true)
        end
    end

    @testset "metadata: a row per pure virtual, and the abstract records" begin
        plan = md["vtable_methods"]
        @test sort!([String(r["symbol"]) for r in plan["bound"]]) == sort(vcat(PV_LIVE_BOUND, PV_LIVE_REFUSED))
        skipped = Dict(String(r["symbol"]) => String(r["reason"]) for r in plan["skipped"])
        @test sort!(collect(keys(skipped))) == sort(first.(PV_LIVE_SKIPPED))
        for (sym, why) in PV_LIVE_SKIPPED
            @test (sym, startswith(get(skipped, sym, ""), why)) == (sym, true)
        end
        checked = 0
        for sym in vcat(PV_LIVE_BOUND, PV_LIVE_REFUSED)
            f = get(fns, sym, nothing)
            @test (sym, f !== nothing) == (sym, true)
            f === nothing && continue
            @test (sym, get(f, "vtable_only", false), get(f, "vtable_slot", nothing) isa Integer) == (sym, true, true)
            @test (sym, String(f["parameters"][1]["name"]), f["has_object_pointer"]) == (sym, "this", true)
            checked += 1
        end
        @test checked == length(PV_LIVE_BOUND) + length(PV_LIVE_REFUSED)
        # Every other row is a symbol of the library, as before.
        others = [String(f["mangled"]) for f in md["functions"] if get(f, "vtable_only", false) != true]
        @test !isempty(others) && issubset(others, exported)
        @test get(fns["_ZNK4Base3tagEv"], "vtable_only", false) == false
        # The slot is the declaring class's own.
        @test fns["_ZNK6Animal4nameEv"]["vtable_slot"] == 2 && fns["_ZNK5Shape5sidesEv"]["vtable_slot"] == 3
        left(cls) = get(get(md["record_abi"], cls, Dict{String,Any}()), "pure_virtuals", nothing)
        @test left("Shape") == ["Shape::area() const", "Shape::sides() const"]
        @test left("Poly") == ["Shape::area() const"]
        @test left("Token") == ["Token::~Token()"]
        @test left("Listener") == ["Listener::on_event(int)", "Listener::priority() const"]
        @test "Animal::name() const" in left("Animal") && "Animal::hidden() const" in left("Animal")
        for cls in ("Dog", "Puppy", "Bird", "Duck", "Coin", "Square", "Derived")
            @test (cls, left(cls)) == (cls, nothing)
        end
    end

    @testset "the surface guard: no name to find, and no surface to count" begin
        cfg = RepliBuild.ConfigurationManager.load_config(toml_path)
        @test RepliBuild.Compiler.verify_wrap_surface(cfg, library, md_path) === nothing
        # A library that exports no function at all is empty, whatever its
        # interfaces declare.
        mktempdir() do dir
            only_pure = Dict{String,Any}(md)
            only_pure["functions"] = Any[f for f in md["functions"] if get(f, "vtable_only", false) == true]
            @test length(only_pure["functions"]) == length(PV_LIVE_BOUND) + length(PV_LIVE_REFUSED)
            path = joinpath(dir, "compilation_metadata.json")
            open(io -> JSON.print(io, only_pure), path, "w")
            err = try
                RepliBuild.Compiler.verify_wrap_surface(cfg, library, path); nothing
            catch e
                sprint(showerror, e)
            end
            @test err !== nothing && occursin("is EMPTY", err)
        end
    end

    @testset "the wrapper: a thunk for each, a refusal where there is none" begin
        for sym in PV_LIVE_BOUND
            @test (sym, sym in thunks) == (sym, true)
            @test (sym, occursin("\"_mlir_ciface_$(sym)_thunk\"", code)) == (sym, true)
        end
        for sym in vcat(PV_LIVE_BOUND, PV_LIVE_REFUSED, first.(PV_LIVE_SKIPPED))
            @test (sym, occursin(":$sym,", code)) == (sym, false)       # no ccall names one
        end
        for (name, sym) in (("Animal_span", "_ZNK6Animal4spanEv"), ("Animal_weigh", "_ZNK6Animal5weighE4Span"),
                            ("Animal_blob", "_ZNK6Animal4blobEv"))
            @test (name, occursin("ABI Safety Trap: cannot call '$name'", body(name))) == (name, true)
            @test (name, occursin("It is pure virtual and", body(name))) == (name, true)
            @test (sym, sym in thunks) == (sym, false)
        end
        for name in ("Animal_secret", "Animal_hidden", "Logger_log")
            @test (name, defined(name)) == (name, false)
        end
        # The string pair of a `const char*` return, and the factory struct's proxy.
        @test defined("Animal_name") && defined("Animal_name_ptr")
        @test occursin(r"^name\(obj::Animal\) = .*\bAnimal_name\(obj\.handle\)$"m, code)
        # No constructor of an abstract class is bound.
        for (name, sym) in (("Animal_Animal", "_ZN6AnimalC2Ei"), ("Shape_Shape", "_ZN5ShapeC2Ev"),
                            ("Poly_Poly", "_ZN4PolyC2Ei"), ("Token_Token", "_ZN5TokenC2Ev"))
            @test (name, occursin("is an abstract class", body(name))) == (name, true)
            @test (sym, sym in thunks) == (sym, false)
        end
        @test "_ZN6SquareC1Ei" in thunks && "_ZN3DogC1Ev" in thunks
        # A virtual with a symbol keeps its own thunk.
        @test "_ZNK6Animal4legsEv" in thunks && "_ZNK4Base3tagEv" in thunks
    end

    @testset "from a child, through the JIT" begin
        # The child's engine writes the module it was given under `.debug/`,
        # named by its content and only when no file of that name exists; an
        # earlier run's are removed so the one read below is this run's.
        rm(joinpath(PURE_VIRTUAL_DIR, ".debug"); recursive = true, force = true)
        output = pv_live_child(wrapper, "jit", vcat(["jit_engine_initialized"], PV_LIVE_PROBES))
        # The pre-flight asked the library for no pure virtual.
        @test !occursin("resolve neither", output)
        @test !occursin("nothing pre-flighted", output)
    end

    @testset "the thunk module: a vtable call, and no callee declared" begin
        dir = joinpath(PURE_VIRTUAL_DIR, ".debug", "mlir")
        files = isdir(dir) ? filter(f -> endswith(f, ".mlir"), readdir(dir; join = true)) : String[]
        @test length(files) == 1
        ir = isempty(files) ? "" : read(first(files), String)
        thunk(sym) = (i = findfirst("func.func @$(sym)_thunk(", ir); i === nothing ? "" :
                      ir[first(i):last(something(findnext("\n}\n", ir, last(i)), i))])
        checked = 0
        for sym in PV_LIVE_BOUND
            t = thunk(sym)
            @test (sym, count("\"jlcs.vcall\"", t)) == (sym, 1)
            @test (sym, occursin("slot = $(fns[sym]["vtable_slot"]) : i64", t)) == (sym, true)
            @test (sym, occursin("_call", t)) == (sym, false)
            @test (sym, occursin("func.func private @$sym(", ir)) == (sym, false)
            checked += 1
        end
        @test checked == length(PV_LIVE_BOUND)
        for sym in PV_LIVE_REFUSED
            @test (sym, thunk(sym)) == (sym, "")
        end
        @test RepliBuild.JLCSIRGenerator.undeclared_callees(ir) == String[]
        # A virtual with a symbol is declared, and dispatched the same way.
        @test occursin("func.func private @_ZNK6Animal4legsEv(", ir)
        @test occursin("\"jlcs.vcall\"", thunk("_ZNK6Animal4legsEv"))
        @test occursin("\"jlcs.vcall\"", thunk("_ZNK4Base3tagEv"))
    end

    @testset "with AOT thunks" begin
        pv_live_toml(toml_path; aot = true)
        aot_wrapper = RepliBuild.wrap(toml_path)
        @test isfile(aot_wrapper)
        text = read(aot_wrapper, String)
        thunks_lib = joinpath(out, "libpvirt_thunks." * Libdl.dlext)
        @test isfile(thunks_lib)
        defined_thunks = isfile(thunks_lib) ? pv_live_exports(thunks_lib) : Set{String}()
        @test occursin("JITManager.invoke_aot_ptr(", text) && !occursin("JITManager.invoke(", text)
        slots = Set(String(s.captures[1]) for s in eachmatch(r"\(_FPTR_\w+, \"(_mlir_ciface_\w+_thunk)\"\)", text))
        checked = 0
        for sym in PV_LIVE_BOUND
            @test (sym, "_mlir_ciface_$(sym)_thunk" in slots) == (sym, true)            # bound by the wrapper
            @test (sym, "_mlir_ciface_$(sym)_thunk" in defined_thunks) == (sym, true)   # defined by the thunks library
            checked += 1
        end
        @test checked == length(PV_LIVE_BOUND)
        for sym in PV_LIVE_REFUSED
            @test (sym, "_mlir_ciface_$(sym)_thunk" in defined_thunks) == (sym, false)
        end
        pv_live_child(aot_wrapper, "aot", vcat(["aot_thunks_resolved"], PV_LIVE_PROBES))
    end
end
end # requires
