#!/usr/bin/env julia
# test/test_stl_table_matching.jl — a [types] templates entry matches the
# compiler's spelling of its members, and an entry with no table says so
# (2026-11-13).
#
# clipper2 lists `std::vector<Clipper2Lib::Point64>`; `Point64` is a typedef of
# `Point<long>`, and every member symbol demangles as
# `std::vector<Clipper2Lib::Point<long>, std::allocator<…>>`, so the string
# match on the recipe's spelling found nothing and the package got no factory,
# silently. fmt lists `fmt::basic_memory_buffer<char>`, which the STL table does
# not know at all, and got nothing with the same silence. The templates TU now
# tags each entry (`template void replibuild_force::tag<T, N>(T*);`), whose
# demangled symbol carries T in the compiler's spelling, so the table maps the
# spelling back to the entry (`_stl_template_tags`); an entry that still yields
# no table is a warning naming it and why (`_stl_table_gaps`); and the factory's
# element type comes from the compiler's spelling and must be a struct the
# wrapper emitted (`_stl_vector_element_type`), never `_UnsafeUnknown`.
#
# No toolchain: nm-shaped text and synthetic metadata. The live half is the
# stl_test fixture (`std::vector<stl_test::Pt64>`, verify.jl).
#
# Negative checks: return an empty Dict from `_stl_template_tags` and the
# first testset fails on the Point64 table; make `_stl_vector_element_type`
# return its `infer` answer unconditionally and the factory testset fails.

using Test
using Libdl
using RepliBuild

const _SM_C = RepliBuild.Compiler
const _SM_W = RepliBuild.Wrapper

const _SM_V = "std::vector<Clipper2Lib::Point<long>, std::allocator<Clipper2Lib::Point<long> > >"

@testset "[types] templates entries match the compiler's spelling" begin
    @testset "the tag maps an entry to its canonical spelling" begin
        d = """
        0000000000031000 W void replibuild_force::tag<$(_SM_V), 0>($(_SM_V)*)
        0000000000031010 W void replibuild_force::tag<std::vector<std::vector<Clipper2Lib::Point<long>, std::allocator<Clipper2Lib::Point<long> > >, std::allocator<std::vector<Clipper2Lib::Point<long>, std::allocator<Clipper2Lib::Point<long> > > > >, 1>(std::vector<std::vector<Clipper2Lib::Point<long>, std::allocator<Clipper2Lib::Point<long> > >, std::allocator<std::vector<Clipper2Lib::Point<long>, std::allocator<Clipper2Lib::Point<long> > > > >*)
        0000000000031020 W void replibuild_force::tag<fmt::v12::basic_memory_buffer<char, 500ul, fmt::v12::detail::allocator<char> >, 2>(fmt::v12::basic_memory_buffer<char, 500ul, fmt::v12::detail::allocator<char> >*)
        0000000000031030 W Clipper2Lib::Point<long> const& replibuild_force::ref<Clipper2Lib::Point<long> const>()
        """
        tags = _SM_C._stl_template_tags(d)
        @test tags[0] == "std::vector<Clipper2Lib::Point<long>>"
        @test tags[1] == "std::vector<std::vector<Clipper2Lib::Point<long>>>"
        @test tags[2] == "fmt::v12::basic_memory_buffer<char, 500ul, fmt::v12::detail::allocator<char>>"
        @test length(tags) == 3                                     # `ref` is not a tag
    end

    @testset "the table is keyed by the entry as written and records the spelling" begin
        mangled = """
        0000000000031000 W _ZN16replibuild_force3tagISt6vectorIN11Clipper2Lib5PointIlEESaIS3_EELi0EEEvPT_
        0000000000040000 W _ZNSt6vectorIN11Clipper2Lib5PointIlEESaIS2_EEC2Ev
        0000000000040010 W _ZNKSt6vectorIN11Clipper2Lib5PointIlEESaIS2_EE4sizeEv
        0000000000040020 W _ZNSt6vectorIN11Clipper2Lib5PointIlEESaIS2_EE9push_backERKS2_
        0000000000040030 W _ZNSt6vectorIN11Clipper2Lib5PointIlEESaIS2_EEixEm
        0000000000040040 W _ZNSt6vectorIN11Clipper2Lib5PointIlEESaIS2_EED2Ev
        0000000000040050 W _ZNSt6vectorIiSaIiEE4sizeEv
        """
        demangled = """
        0000000000031000 W void replibuild_force::tag<$(_SM_V), 0>($(_SM_V)*)
        0000000000040000 W $(_SM_V)::vector()
        0000000000040010 W $(_SM_V)::size() const
        0000000000040020 W $(_SM_V)::push_back(Clipper2Lib::Point<long> const&)
        0000000000040030 W $(_SM_V)::operator[](unsigned long)
        0000000000040040 W $(_SM_V)::~vector()
        0000000000040050 W std::vector<int, std::allocator<int> >::size() const
        """
        # The tags index the recipe's own list: entry 0 is the Point64 vector.
        tmpl = ["std::vector<Clipper2Lib::Point64>", "std::vector<int>"]
        (table, unbindable) = _SM_C._stl_methods_from_nm(mangled, demangled, tmpl; runtime = nothing)
        @test isempty(unbindable)
        @test sort(collect(keys(table))) == ["std::vector<Clipper2Lib::Point64>", "std::vector<int>"]   # the entries as written
        ms = table["std::vector<Clipper2Lib::Point64>"]
        @test Set(m["method"] for m in ms) == Set(["constructor", "size", "push_back", "subscript", "destructor"])
        @test all(m -> m["container"] == "std::vector<Clipper2Lib::Point<long>>", ms)
        @test Dict(m["method"] => m["mangled"] for m in ms)["size"] == "_ZNKSt6vectorIN11Clipper2Lib5PointIlEESaIS2_EE4sizeEv"
        @test isempty(_SM_C._stl_table_gaps(tmpl, table))
        # Without the tag, the typedef spelling matches nothing, and the gap says so.
        (t0, _) = _SM_C._stl_methods_from_nm(mangled, replace(demangled, "replibuild_force::tag" => "x"), tmpl; runtime = nothing)
        @test !haskey(t0, "std::vector<Clipper2Lib::Point64>") && haskey(t0, "std::vector<int>")
        gaps = _SM_C._stl_table_gaps(tmpl, t0)
        @test length(gaps) == 1 && gaps[1][1] == "std::vector<Clipper2Lib::Point64>" && occursin("no member of it", gaps[1][2])
        # An entry spelled as the compiler spells it matches as before, with no `container`.
        @test [m["method"] for m in table["std::vector<int>"]] == ["size"] && !haskey(table["std::vector<int>"][1], "container")
    end

    @testset "a tagged non-STL template gets no table, and is named" begin
        B = "fmt::v12::basic_memory_buffer<char, 500ul, fmt::v12::detail::allocator<char> >"
        mangled = """
        0000000000031000 W _ZN16replibuild_force3tagIN3fmt3v1219basic_memory_bufferIcLm500ENS1_6detail9allocatorIcEEEELi0EEEvPT_
        0000000000040000 W _ZNK3fmt3v1219basic_memory_bufferIcLm500ENS0_6detail9allocatorIcEEE4sizeEv
        0000000000040010 W _ZN3fmt3v1219basic_memory_bufferIcLm500ENS0_6detail9allocatorIcEEE6resizeEm
        """
        demangled = """
        0000000000031000 W void replibuild_force::tag<$B, 0>($B*)
        0000000000040000 W $B::size() const
        0000000000040010 W $B::resize(unsigned long)
        """
        tmpl = ["fmt::basic_memory_buffer<char>"]
        (t, u) = _SM_C._stl_methods_from_nm(mangled, demangled, tmpl; runtime = nothing)
        @test isempty(t) && isempty(u)
        gaps = _SM_C._stl_table_gaps(tmpl, t)
        @test length(gaps) == 1 && occursin("not a container the STL table knows", gaps[1][2])
    end

    @testset "an entry outside the table's containers is named, not silent" begin
        gaps = _SM_C._stl_table_gaps(["fmt::basic_memory_buffer<char>", "std::string", "std::map<int, int>"],
                                     Dict{String,Any}("std::basic_string<char>" => Any[Dict("method" => "size")]))
        @test [g[1] for g in gaps] == ["fmt::basic_memory_buffer<char>", "std::map<int, int>"]
        @test occursin("not a container the STL table knows", gaps[1][2]) && occursin("DWARF", gaps[1][2])
        @test occursin("no member of it", gaps[2][2])
        @test isempty(_SM_C._stl_table_gaps(String[], Dict{String,Any}()))
    end

    @testset "the factory's element is the emitted struct, never an unknown" begin
        emitted = Set(["Point_long", "Vec3"])
        infer = t -> t == "int" ? "Cint" : "_UnsafeUnknown"
        f = _SM_W._stl_vector_element_type
        @test f("Clipper2Lib::Point<long>", emitted, infer) == "Point_long"
        @test f("Point<long>", emitted, infer) == "Point_long"
        @test f("int", emitted, infer) == "Cint"
        @test f("Clipper2Lib::Point64", emitted, infer) === nothing          # the typedef names no emitted struct
        @test f("Nowhere::Thing", emitted, infer) === nothing

        # End to end: synthetic metadata with the clipper2 shape gives a
        # CppVector{Point_long} factory; an element nothing emitted gives none.
        dir = mktempdir()
        write(joinpath(dir, "replibuild.toml"), """
        [project]
        name = "stlmatch"
        root = "$(escape_string(dir))"
        [wrap]
        language = "cpp"
        [types]
        strictness = "warn"
        allow_unknown_structs = true
        templates = ["std::vector<Clipper2Lib::Point64>", "std::vector<Clipper2Lib::Ghost>"]
        [cache]
        enabled = false
        """)
        cfg = RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml"))
        libref = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
        row(method, mangled, container) = Dict{String,Any}("method" => method, "mangled" => mangled,
            "demangled" => "$container::$method", "is_const" => false, "full_signature" => method, "container" => container)
        meta = Dict{String,Any}(
            "functions" => Any[],
            "globals" => Dict{String,Any}(),
            "struct_definitions" => Dict{String,Any}(
                "Point<long>" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x10", "scope" => "Clipper2Lib",
                    "members" => Any[Dict{String,Any}("name" => "x", "c_type" => "long", "julia_type" => "Clong", "offset" => "0x0", "size" => 8),
                                     Dict{String,Any}("name" => "y", "c_type" => "long", "julia_type" => "Clong", "offset" => "0x8", "size" => 8)])),
            "stl_methods" => Dict{String,Any}(
                "std::vector<Clipper2Lib::Point64>" => Any[
                    row("constructor", "_ZNSt6vectorIN11Clipper2Lib5PointIlEESaIS2_EEC2Ev", "std::vector<Clipper2Lib::Point<long>>"),
                    row("size", "_ZNKSt6vectorIN11Clipper2Lib5PointIlEESaIS2_EE4sizeEv", "std::vector<Clipper2Lib::Point<long>>")],
                "std::vector<Clipper2Lib::Ghost>" => Any[
                    row("size", "_ZNKSt6vectorIN11Clipper2Lib5GhostESaIS1_EE4sizeEv", "std::vector<Clipper2Lib::Ghost>")]),
            "record_abi" => Dict{String,Any}())
        code = @test_logs (:warn, r"no CppVector factory for \"std::vector<Clipper2Lib::Ghost>\"") match_mode = :any begin
            r = _SM_W.generate_introspective_module_cpp(cfg, libref, meta, "StlMatch", _SM_W.create_type_registry(cfg), true)
            r isa Tuple ? String(first(r)) : String(r)
        end
        @test occursin("function create_std_vector_Clipper2Lib_Point64()", code)
        @test occursin("CppVector{Point_long}(buf, true, _THUNKS_STD_VECTOR_CLIPPER2LIB_POINT64", code)
        @test !occursin("create_std_vector_Clipper2Lib_Ghost", code)
        @test !occursin("_UnsafeUnknown}(buf", code)
        ex = Meta.parseall(code)
        @test !any(a -> a isa Expr && a.head in (:incomplete, :error), ex.args)
    end
end
