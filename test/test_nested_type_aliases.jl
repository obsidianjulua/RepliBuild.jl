# =============================================================================
# Class-nested types under their qualified names (2026-11-15)
#
# DWARF keys a record by its bare name, so a type declared inside a class is
# emitted under its own name (`ClosestRayResultCallback`, and that key in
# STRUCT_SIZES) while its functions carry the class
# (`btCollisionWorld_ClosestRayResultCallback_…`). The C++ wrapper now also
# binds the type under the name its functions use, and keys its layout by
# both (Wrapper._nested_type_aliases). Additive: the bare name is unchanged.
#
# Toolchain-free: which records get an alias and which do not, the section's
# text, and the layout tables. test_nested_type_aliases_live.jl builds
# test/nested_type_test and uses the names.
# =============================================================================

using Test
using RepliBuild

const _NA = RepliBuild.Wrapper

@testset "Class-nested type aliases" begin
    san = _NA._sanitize_cpp_type_name
    rec(scope; kw...) = merge(Dict{String,Any}("kind" => "struct", "scope" => scope, "scopes" => Any[scope],
                                               "byte_size" => "0x8",
                                               "members" => Any[Dict{String,Any}("name" => "a", "offset" => 4)]),
                              Dict{String,Any}(String(k) => v for (k, v) in kw))
    ds = Dict{String,Any}(
        "World" => rec(""),
        "RayResult" => rec("World"),                              # nested in a class
        "Callback" => rec("World"),
        "Info" => rec("World::Callback"),                         # two deep
        "Mark" => rec("World"),                                   # alias is a function's name
        "Plain" => rec("nt"),                                     # a namespace only
        "Box" => rec("nt"),
        "Lid" => rec("nt::Box"),                                  # a class inside a namespace
        "Node" => rec("Tree<ns::K>"),                             # a template outer
        "Tree<ns::K>" => rec(""),
        "Twice" => rec("World"; conflict = true),                 # distinct definitions
        "Shared" => rec("World"; scopes = Any["World", "nt::Box"]),   # one name, two scopes
        "Orphan" => rec("Gone"),                                  # the outer has no entry
        "NoScope" => delete!(delete!(rec(""), "scope"), "scopes"),
        "Unemitted" => rec("World"),
        "Old" => delete!(rec("World"), "scopes"),                 # metadata with `scope` alone
        "__enum__Kind" => Dict{String,Any}("kind" => "enum", "scope" => "World"))
    emitted = Set(["World", "RayResult", "Callback", "Info", "Mark", "Plain", "Box", "Lid", "Node",
                   "Tree_ns_K", "Twice", "Shared", "Orphan", "NoScope", "Old"])
    taken = Set(["World_Mark", "World_cast"])

    @testset "which records answer to a qualified name" begin
        aliases = _NA._nested_type_aliases(ds, emitted, san, taken)
        @test aliases == ["Tree_ns_K_Node" => "Node", "World_Callback" => "Callback",
                          "World_Callback_Info" => "Info", "World_Old" => "Old",
                          "World_RayResult" => "RayResult", "nt_Box_Lid" => "Lid"]
        names = first.(aliases)
        # The alias is the prefix the type's functions already have.
        @test _NA._julia_function_name("World::RayResult_hit", "_ZNK5World9RayResult3hitEv") == "World_RayResult_hit"
        # Not emitted: a name the module binds already, a namespace-only scope,
        # a conflicted name, a name with two scopes, an outer with no entry,
        # no scope at all, a record the wrapper does not emit, an enum.
        for absent in ("World_Mark", "nt_Plain", "nt_Box", "World_Twice", "World_Shared", "nt_Box_Shared",
                       "Gone_Orphan", "NoScope", "World_Unemitted", "World_Kind", "World")
            @test (absent, absent in names) == (absent, false)
        end
        # An alias equal to an emitted type's own name is that type's, not an alias.
        ds2 = Dict{String,Any}("A" => rec(""), "B" => rec("A"), "A::B" => rec(""))
        @test _NA._nested_type_aliases(ds2, Set(["A", "B", "A_B"]), san, Set{String}()) == Pair{String,String}[]
        # Nothing nested: nothing, and the section is empty text.
        flat = Dict{String,Any}("A" => rec(""), "B" => rec("ns"))
        @test isempty(_NA._nested_type_aliases(flat, Set(["A", "B"]), san, Set{String}()))
        @test _NA._nested_type_alias_chunk(Pair{String,String}[]) == ""
    end

    @testset "the innermost scope component keeps its template arguments" begin
        f = _NA._innermost_scope_component
        @test f("World") == "World"
        @test f("nt::Box") == "Box"
        @test f("World::Callback") == "Callback"
        @test f("ns::Tree<ns::K, std::less<ns::K>>") == "Tree<ns::K, std::less<ns::K>>"
        @test f("Tree<ns::K>") == "Tree<ns::K>"
    end

    @testset "the section and the layout tables" begin
        aliases = _NA._nested_type_aliases(ds, emitted, san, taken)
        chunk = _NA._nested_type_alias_chunk(aliases)
        @test occursin("\nconst World_RayResult = RayResult\n", chunk)
        @test occursin("\nconst nt_Box_Lid = Lid\n", chunk)
        @test count(r"^const "m, chunk) == length(aliases) == 6
        # Parses, and binds nothing but the aliases.
        ex = Meta.parseall(chunk)
        @test !any(a -> a isa Expr && a.head in (:incomplete, :error), ex.args)
        layout = _NA._layout_chunk(ds, emitted, san; aliases = aliases)
        @test occursin("    :RayResult => 8,\n", layout)
        @test occursin("    :World_RayResult => 8,\n", layout)
        @test occursin("    :World_RayResult => Dict{Symbol,Int}(:a => 4),\n", layout)
        @test occursin("    :Tree_ns_K_Node => 8,\n", layout)
        # The bare names come first, as they always were; the aliases follow.
        @test findfirst(":World => 8", layout).start < findfirst(":Tree_ns_K_Node => 8", layout).start
        # No aliases: the tables are byte for byte what they were.
        @test _NA._layout_chunk(ds, emitted, san) == _NA._layout_chunk(ds, emitted, san; aliases = Pair{String,String}[])
        @test !occursin("World_RayResult", _NA._layout_chunk(ds, emitted, san))
    end
end
