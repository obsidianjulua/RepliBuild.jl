# =============================================================================
# Constructors, destructors and virtual methods of real classes, live (2026-11-17)
#
# test_structor_variants.jl and test_virtual_dispatch.jl (unit) check what the
# parser, the generator and the thunk producer emit from synthetic input. This
# builds test/virtual_class_test with the system compiler, wraps it, and calls
# it from a child:
#
#   * A class with a virtual base (`Left`, `Right`, `Both` over `Root`) is built
#     in place by `<X>_<X>(p, …)` and ended by `<X>_destroy_<X>(p)`: the
#     complete-object constructor and destructor. Bound to the base-object
#     ones, as every class was, the constructor ran with no VTT and the child
#     segfaulted, and the destructors' wrappers took `(this, vtt)`.
#   * `Dog`, `Puppy`, `Duck` and `Mallard` add no data member to their bases.
#     Their virtual methods go through the vtable (`Dog_name(puppy)` is
#     "puppy"), and so does a `noexcept` virtual, which a `ccall` could carry,
#     and the methods of `Bird`, which the library's second file only declares.
#
# Needs clang++ and libJLCS.
# =============================================================================

using Test
using RepliBuild
import Libdl

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const VIRTUAL_CLASS_DIR = joinpath(@__DIR__, "virtual_class_test")

if requires("constructor/destructor variants and virtual dispatch, live", :libJLCS, :clangxx)
@testset "Constructor/destructor variants and virtual dispatch, live" begin
    toml_path = joinpath(VIRTUAL_CLASS_DIR, "replibuild.toml")
    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "vclass"
    root = "$(escape_string(VIRTUAL_CLASS_DIR))"

    [compile]
    flags = ["-std=c++17", "-fPIC"]
    source_files = ["$(escape_string(joinpath(VIRTUAL_CLASS_DIR, "src", "vclass.cpp")))",
                    "$(escape_string(joinpath(VIRTUAL_CLASS_DIR, "src", "vclass_late.cpp")))"]
    include_dirs = ["$(escape_string(joinpath(VIRTUAL_CLASS_DIR, "include")))"]

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
    # The build's own warnings are part of what is checked: the VTT of a
    # base-object variant used to be reported as a dropped parameter.
    (build_logs, _) = Test.collect_test_logs() do
        redirect_stdout(devnull) do
            RepliBuild.build(toml_path)
        end
    end
    wrapper = RepliBuild.wrap(toml_path)
    @test isfile(wrapper)
    code = read(wrapper, String)
    out = dirname(wrapper)
    library = joinpath(out, "libvclass." * Libdl.dlext)
    md = RepliBuild._read_json(joinpath(out, "compilation_metadata.json"))
    fns = Dict{String,Any}(String(f["mangled"]) => f for f in md["functions"])
    thunks = Set{String}(String.(RepliBuild._read_json(joinpath(out, "thunk_manifest.json"))["function_thunks"]))
    body(name) = (m = match(Regex("^function $name\\([^\\n]*\\)\\n.*?^end", "ms"), code); m === nothing ? "" : String(m.match))
    sigs(name) = [String(m.captures[1]) for m in eachmatch(Regex("^function $name\\(([^\\n]*)\\)\$", "m"), code)]
    # The base-object variants whose complete-object twin the library has too.
    twins = sort!([m for m in keys(fns) if occursin(r"[CD]2E", m) &&
                   haskey(fns, replace(m, r"C2E" => "C1E", r"D2E" => "D1E"))])

    @testset "the fixture has the shapes it is built for" begin
        # Both variants of the classes with a virtual base, and for the
        # abstract ones only the base-object constructor.
        for sym in ("_ZN4BothC1Ev", "_ZN4BothC2Ev", "_ZN4BothC1Eii", "_ZN4BothC2Eii", "_ZN4LeftC1Ei", "_ZN4LeftC2Ei",
                    "_ZN4LeftD0Ev", "_ZN4LeftD1Ev", "_ZN4LeftD2Ev", "_ZN4BothD0Ev", "_ZN4BothD1Ev",
                    "_ZN6SketchC2Ev", "_ZN5PlainC2Ei")
            @test (sym, haskey(fns, sym)) == (sym, true)
        end
        @test !haskey(fns, "_ZN6SketchC1Ev") && !haskey(fns, "_ZN5PlainC1Ei")
        sd = md["struct_definitions"]
        @test all(c -> get(sd[c], "members", nothing) == [], ("Dog", "Puppy", "Duck", "Mallard"))
        @test get(fns["_ZNK3Dog4rankEv"], "is_noexcept", false) == true
        # Bird has a DIE in each of the two units, and one is a declaration.
        dump = read(RepliBuild.Tools.tool_cmd(`llvm-dwarfdump --debug-info $library`), String)
        birds = [m.match for m in eachmatch(r"DW_TAG_structure_type\n(?:\s+DW_AT_(?!name)[^\n]*\n)*\s+DW_AT_name\s+\(\"Bird\"\)\n(?:\s+DW_AT_[^\n]*\n)*", dump)]
        @test length(birds) == 2
        declared = count(b -> occursin("DW_AT_declaration", b), birds)
        # Clang's default off Windows (constructor homing); elsewhere say
        # what this toolchain wrote, since the checks below hold either way.
        Sys.iswindows() ? (@info "Bird DIEs that are declarations" declared) : @test declared == 1
        # The class table keeps the definition and every virtual method.
        bird = RepliBuild.DWARFParser.parse_vtables(library).classes["Bird"]
        @test bird.size == 24 && [m.name for m in bird.members] == ["wingspan"]
        @test issubset(["name", "speed"], [m.name for m in bird.virtual_methods])
    end

    @testset "metadata: each variant has its own parameters" begin
        names(sym) = [String(p["name"]) for p in fns[sym]["parameters"]]
        checked = 0
        for (sym, want) in (("_ZN4BothC1Ev", ["this"]), ("_ZN4BothC2Ev", ["this", "vtt"]),
                            ("_ZN4BothC1Eii", ["this", "l_", "b_"]), ("_ZN4BothC2Eii", ["this", "vtt", "l_", "b_"]),
                            ("_ZN4LeftC1Ei", ["this", "l_"]), ("_ZN4LeftC2Ei", ["this", "vtt", "l_"]),
                            ("_ZN4LeftD0Ev", ["this"]), ("_ZN4LeftD1Ev", ["this"]), ("_ZN4LeftD2Ev", ["this", "vtt"]),
                            ("_ZN4BothD0Ev", ["this"]), ("_ZN4BothD1Ev", ["this"]),
                            ("_ZN6SketchC2Ev", ["this", "vtt"]),
                            # No virtual base: nothing to add.
                            ("_ZN4RootC1Ev", ["this"]), ("_ZN4RootC2Ev", ["this"]), ("_ZN5PlainC2Ei", ["this", "v_"]))
            @test (sym, haskey(fns, sym) ? names(sym) : nothing) == (sym, want)
            checked += 1
        end
        @test checked == 15
        # The VTT is marked, and nothing else is.
        marked = sort!([String(f["mangled"]) for f in md["functions"]
                        if any(p -> get(p, "artificial", false) == true, f["parameters"])])
        @test !isempty(marked)
        @test all(m -> occursin(r"^_ZN\d+\w+?[CD]2E", m), marked)
        @test all(m -> fns[m]["parameters"][2]["name"] == "vtt" && get(fns[m]["parameters"][2], "artificial", false) == true, marked)
        # No "dropped parameters" warning for a variant's VTT.
        @test !any(l -> occursin("dropped parameters present in the DIE tree", string(l.message)), build_logs)
    end

    @testset "the wrapper binds the complete-object constructor and destructor" begin
        want = (("Both_Both", "this::Any", "_ZN4BothC1Ev"), ("Both_Both", "this::Any, l_::Integer, b_::Integer", "_ZN4BothC1Eii"),
                ("Both_destroy_Both", "this::Any", "_ZN4BothD1Ev"), ("Both_delete_Both", "this::Any", "_ZN4BothD0Ev"),
                ("Left_Left", "this::Any", "_ZN4LeftC1Ev"), ("Left_Left", "this::Any, l_::Integer", "_ZN4LeftC1Ei"),
                ("Left_destroy_Left", "this::Any", "_ZN4LeftD1Ev"), ("Left_delete_Left", "this::Any", "_ZN4LeftD0Ev"),
                ("Right_Right", "this::Any", "_ZN5RightC1Ev"), ("Solid_Solid", "this::Any", "_ZN5SolidC1Ev"),
                # No virtual base: C1 where the library has it, else C2.
                ("Root_Root", "this::Any", "_ZN4RootC1Ev"), ("Root_destroy_Root", "this::Any", "_ZN4RootD1Ev"),
                ("Plain_Plain", "this::Any, v_::Integer", "_ZN5PlainC2Ei"))
        checked = 0
        for (name, sig, sym) in want
            @test (name, sig in sigs(name)) == (name, true)
            i = findfirst("function $name($sig)\n", code)
            fn = i === nothing ? "" : code[first(i):last(something(findnext("\nend", code, last(i)), i))]
            @test (name, sig, occursin("\"_mlir_ciface_$(sym)_thunk\"", fn)) == (name, sig, true)
            @test (sym, sym in thunks) == (sym, true)
            checked += 1
        end
        @test checked == length(want)
        @test sort!(sigs("Both_Both")) == ["this::Any", "this::Any, l_::Integer, b_::Integer"]
        for name in ("Both_destroy_Both", "Both_delete_Both", "Left_destroy_Left", "Left_delete_Left",
                     "Right_destroy_Right", "Right_delete_Right")
            @test (name, sigs(name)) == (name, ["this::Any"])
        end
        # No function takes a VTT, and no thunk is compiled for a base-object
        # variant that has a complete-object twin.
        @test isempty([m.match for m in eachmatch(r"^function \w+\([^\n]*\bvtt\b[^\n]*\)$"m, code)])
        @test length(twins) >= 8
        @test isempty(intersect(twins, thunks))
        @test !any(m -> occursin("_mlir_ciface_$(m)_thunk", code), twins)
        # The abstract class with a virtual base: a refusal that says why.
        @test occursin("only the base-object constructor of `Sketch` (`_ZN6SketchC2Ev`)", body("Sketch_Sketch"))
        @test !("_ZN6SketchC2Ev" in thunks)
    end

    @testset "the wrapper sends every virtual method to a thunk" begin
        for (name, sym) in (("Dog_name", "_ZNK3Dog4nameEv"), ("Dog_speed", "_ZNK3Dog5speedEd"),
                            ("Dog_rank", "_ZNK3Dog4rankEv"), ("Animal_rank", "_ZNK6Animal4rankEv"),
                            ("Duck_name", "_ZNK4Duck4nameEv"), ("Duck_depth", "_ZNK4Duck5depthEv"))
            @test (name, sym in thunks) == (name, true)
            @test (name, occursin("ccall((:$sym,", code)) == (name, false)
        end
    end

    @testset "from a child: built, called, ended" begin
        # The child's engine writes the module it was given under `.debug/`,
        # named by its content and only when no file of that name exists; an
        # earlier run's are removed so the one read below is this run's.
        rm(joinpath(VIRTUAL_CLASS_DIR, ".debug"); recursive = true, force = true)
        r = run_isolated(`$(joinpath(VIRTUAL_CLASS_DIR, "probe_virtual_classes.jl")) $wrapper`)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        names = ["both_constructed_in_place", "both_constructed_with_arguments", "left_and_right_constructed_in_place",
                 "delete_takes_the_object_alone", "no_wrapper_takes_a_vtt", "solid_is_built_through_its_abstract_base",
                 "abstract_with_a_virtual_base_is_refused", "abstract_without_a_virtual_base_is_bound",
                 "memberless_class_dispatches", "memberless_class_with_two_bases_dispatches",
                 "class_declared_in_a_later_unit_dispatches", "noexcept_virtual_dispatches"]
        for n in names
            @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
        end
        @test count(n -> haskey(v, n), names) == length(names)
    end

    @testset "the thunk module: a vcall for each, a type_info for each class named" begin
        # The module the child's engine was given, as it wrote it for the
        # debugger (`.debug/mlir/`, emptied before the child ran).
        dir = joinpath(VIRTUAL_CLASS_DIR, ".debug", "mlir")
        files = isdir(dir) ? filter(f -> endswith(f, ".mlir"), readdir(dir; join = true)) : String[]
        @test length(files) == 1
        ir = isempty(files) ? "" : read(first(files), String)
        thunk(sym) = (i = findfirst("func.func @$(sym)_thunk(", ir); i === nothing ? "" :
                      ir[first(i):last(something(findnext("\n}\n", ir, last(i)), i))])
        checked = 0
        for (sym, cls) in (("_ZNK3Dog4nameEv", "Dog"), ("_ZNK3Dog5speedEd", "Dog"), ("_ZNK3Dog4rankEv", "Dog"),
                           ("_ZNK5Puppy4nameEv", "Puppy"), ("_ZNK4Duck4nameEv", "Duck"), ("_ZNK4Duck5depthEv", "Duck"),
                           ("_ZNK7Mallard4nameEv", "Mallard"), ("_ZNK6Animal4rankEv", "Animal"),
                           ("_ZNK6Animal4nameEv", "Animal"), ("_ZNK4Bird4nameEv", "Bird"), ("_ZNK4Bird5speedEd", "Bird"))
            t = thunk(sym)
            @test (sym, occursin("\"jlcs.vcall\"", t) && occursin("class_name = @$cls,", t)) == (sym, true)
            @test (sym, occursin("callee = @$sym", t)) == (sym, false)
            checked += 1
        end
        @test checked == 11
        described = Set(String(m.captures[1]) for m in eachmatch(r"jlcs\.type_info \"([^\"]+)\"", ir))
        named = Set(String(m.captures[1]) for m in eachmatch(r"class_name = @(\w+)", ir))
        @test issubset(Set(["Dog", "Puppy", "Duck", "Mallard"]), named)
        @test issubset(named, described)
        # Constructors and destructors are direct calls on the exact class,
        # and a complete-object destructor is the one-operand `dtor_call`.
        @test occursin("callee = @_ZN4BothC1Ev", thunk("_ZN4BothC1Ev"))
        @test occursin("jlcs.dtor_call @_ZN4BothD1Ev(", thunk("_ZN4BothD1Ev"))
        @test occursin("jlcs.dtor_call @_ZN4BothD0Ev(", thunk("_ZN4BothD0Ev"))
        @test !any(m -> occursin("@$(m)_thunk", ir), twins)
    end
end
end # requires
