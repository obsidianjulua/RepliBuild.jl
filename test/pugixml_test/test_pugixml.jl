#!/usr/bin/env julia
# Integration test: RepliBuild wraps pugixml 1.15 (C++ XML library)
#
# pugixml is a pure C++ library — no extern "C" API.  All class methods are
# C++ mangled symbols.  RepliBuild wraps them directly via DWARF introspection.
#
# Navigation functions (first_child, child, attribute, ...) return xml_node /
# xml_attribute *by value* (tiny 8-byte structs) and are dispatched via the
# MLIR JIT tier in the generated wrapper.  Because libJLCS.so is not always
# present, this test calls them through direct ccall with the mangled symbol,
# which also exercises the correctness of the generated struct layout.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using RepliBuild
using Test

const TEST_DIR   = @__DIR__
const SETUP_FILE = joinpath(TEST_DIR, "setup.jl")
const TOML_FILE  = joinpath(TEST_DIR, "replibuild.toml")
const WRAP_FILE  = joinpath(TEST_DIR, "julia", "PugixmlTest.jl")

@testset "pugixml Integration" begin

    println("\n" * "="^70)
    println("Building and Wrapping pugixml 1.15 (C++)...")
    println("="^70)

    # =========================================================================
    # Build
    # =========================================================================
    @testset "Build" begin
        include(SETUP_FILE)
        library_path = RepliBuild.build(TOML_FILE; clean=true)
        @test isfile(library_path)
        @test filesize(library_path) > 100_000
        println("Library: $(library_path) ($(round(filesize(library_path)/1024, digits=0)) KB)")
    end

    # =========================================================================
    # Wrap
    # =========================================================================
    @testset "Wrap" begin
        RepliBuild.wrap(TOML_FILE)
        @test isfile(WRAP_FILE)
        lines = countlines(WRAP_FILE)
        @test lines > 1000
        println("Wrapper: $(WRAP_FILE) ($(lines) lines)")
    end

end  # outer testset (Build + Wrap)

# =========================================================================
# Load the generated wrapper at top level (required for const + using)
# =========================================================================
include(WRAP_FILE)
using .PugixmlTest
const S   = PugixmlTest
const LIB = S.LIBRARY_PATH

# Mangled symbols for C++ methods that return xml_node / xml_attribute by value
# (8-byte structs, returned in a single register on x86-64 SysV ABI).
const SYM_DOC_ELEM    = :_ZNK4pugi12xml_document16document_elementEv
const SYM_FIRST_CHILD = :_ZNK4pugi8xml_node11first_childEv
const SYM_CHILD       = :_ZNK4pugi8xml_node5childEPKc
const SYM_ATTRIB      = :_ZNK4pugi8xml_node9attributeEPKc
const SYM_NEXT_SIB    = :_ZNK4pugi8xml_node12next_siblingEv
const SYM_LOAD_STR    = :_ZN4pugi12xml_document11load_stringEPKcj

# pugixml's xml_node and xml_attribute are value types (8 bytes = 1 pointer).
# To pass them to wrapped functions that take Ptr{xml_node}, wrap in Ref{}.
node_ref(n) = Ref(n)
attr_ref(a) = Ref(a)

# Allocate + default-construct an xml_document on the Julia heap.
function make_doc()
    ref = Ref(S.xml_document())
    S.pugi_xml_document_xml_document(ref)
    return ref
end

destroy_doc(ref) = S.pugi_xml_document_destroy_xml_document(ref)

function load_xml(doc_ref, xml::String)
    ccall((SYM_LOAD_STR, LIB), S.xml_parse_result,
          (Ptr{S.xml_document}, Cstring, Cuint), doc_ref, xml, Cuint(0))
end

doc_elem(doc_ref) =
    ccall((SYM_DOC_ELEM, LIB), S.xml_node, (Ptr{S.xml_document},), doc_ref)

first_child(node) =
    ccall((SYM_FIRST_CHILD, LIB), S.xml_node, (Ptr{S.xml_node},), node_ref(node))

child_named(node, name::String) =
    ccall((SYM_CHILD, LIB), S.xml_node, (Ptr{S.xml_node}, Cstring), node_ref(node), name)

attrib(node, name::String) =
    ccall((SYM_ATTRIB, LIB), S.xml_attribute,
          (Ptr{S.xml_node}, Cstring), node_ref(node), name)

next_sib(node) =
    ccall((SYM_NEXT_SIB, LIB), S.xml_node, (Ptr{S.xml_node},), node_ref(node))

# =========================================================================
# Runtime tests (use the loaded wrapper)
# =========================================================================
@testset "pugixml Runtime" begin

    @testset "xml_document lifecycle" begin
        doc = make_doc()
        @test doc[] isa S.xml_document
        destroy_doc(doc)
        println("xml_document lifecycle: OK")
    end

    @testset "load_string parse result" begin
        doc    = make_doc()
        result = load_xml(doc, "<root/>")
        @test result isa S.xml_parse_result
        @test result.status == S.status_ok
        @test S.pugi_xml_parse_result_operator_bool(Ref(result))
        desc = S.pugi_xml_parse_result_description(Ref(result))
        @test desc == "No error"
        println("load_string: status=", result.status, " desc='", desc, "'")
        destroy_doc(doc)
    end

    @testset "document element name and type" begin
        doc  = make_doc()
        load_xml(doc, "<library version=\"2\"><book id=\"1\">Julia</book></library>")
        root = doc_elem(doc)
        @test !S.pugi_xml_node_empty(node_ref(root))
        @test S.pugi_xml_node_type(node_ref(root)) == S.node_element
        name = S.pugi_xml_node_name(node_ref(root))
        @test name == "library"
        println("Root node: <", name, ">")
        destroy_doc(doc)
    end

    @testset "attribute access" begin
        doc  = make_doc()
        load_xml(doc, "<library version=\"2\"><book id=\"1\">Julia</book></library>")
        root = doc_elem(doc)
        attr = attrib(root, "version")
        @test !S.pugi_xml_attribute_empty(attr_ref(attr))
        @test S.pugi_xml_attribute_name(attr_ref(attr))  == "version"
        @test S.pugi_xml_attribute_value(attr_ref(attr)) == "2"
        println("Attribute: version=2")
        destroy_doc(doc)
    end

    @testset "child navigation" begin
        doc  = make_doc()
        load_xml(doc, "<root><a>alpha</a><b>beta</b><c>gamma</c></root>")
        root = doc_elem(doc)

        node_a = first_child(root)
        @test S.pugi_xml_node_name(node_ref(node_a))        == "a"
        @test S.pugi_xml_node_child_value(node_ref(node_a)) == "alpha"

        node_b = next_sib(node_a)
        @test S.pugi_xml_node_name(node_ref(node_b))        == "b"
        @test S.pugi_xml_node_child_value(node_ref(node_b)) == "beta"

        node_c = next_sib(node_b)
        @test S.pugi_xml_node_name(node_ref(node_c))        == "c"
        @test S.pugi_xml_node_child_value(node_ref(node_c)) == "gamma"

        node_b2 = child_named(root, "b")
        @test S.pugi_xml_node_name(node_ref(node_b2)) == "b"

        println("Children: a=alpha, b=beta, c=gamma — OK")
        destroy_doc(doc)
    end

    @testset "nested document traversal" begin
        xml = """<catalog><item id="42" cat="sci-fi"><title>Dune</title></item><item id="7" cat="fantasy"><title>Rings</title></item></catalog>"""
        doc    = make_doc()
        result = load_xml(doc, xml)
        @test result.status == S.status_ok

        catalog = doc_elem(doc)
        @test S.pugi_xml_node_name(node_ref(catalog)) == "catalog"

        item1 = first_child(catalog)
        @test S.pugi_xml_attribute_value(attr_ref(attrib(item1, "id")))  == "42"
        @test S.pugi_xml_attribute_value(attr_ref(attrib(item1, "cat"))) == "sci-fi"
        @test S.pugi_xml_node_child_value(node_ref(child_named(item1, "title"))) == "Dune"

        item2 = next_sib(item1)
        @test S.pugi_xml_attribute_value(attr_ref(attrib(item2, "id"))) == "7"
        @test S.pugi_xml_node_child_value(node_ref(child_named(item2, "title"))) == "Rings"

        println("Catalog: item1=Dune (id=42), item2=Rings (id=7) — OK")
        destroy_doc(doc)
    end

    @testset "invalid XML parse error" begin
        doc    = make_doc()
        result = load_xml(doc, "<unclosed>")
        @test result.status != S.status_ok
        @test !S.pugi_xml_parse_result_operator_bool(Ref(result))
        println("Invalid XML status: ", result.status, " — OK")
        destroy_doc(doc)
    end

    @testset "empty node sentinel" begin
        doc  = make_doc()
        load_xml(doc, "<root/>")
        root = doc_elem(doc)

        missing_node = child_named(root, "no_such_child")
        @test S.pugi_xml_node_empty(node_ref(missing_node))
        @test S.pugi_xml_node_type(node_ref(missing_node)) == S.node_null

        miss_attr = attrib(root, "no_such_attr")
        @test S.pugi_xml_attribute_empty(attr_ref(miss_attr))

        println("Empty node sentinel: OK")
        destroy_doc(doc)
    end

    @testset "multiple documents" begin
        doc1 = make_doc(); load_xml(doc1, "<a/>")
        doc2 = make_doc(); load_xml(doc2, "<b/>")
        @test S.pugi_xml_node_name(node_ref(doc_elem(doc1))) == "a"
        @test S.pugi_xml_node_name(node_ref(doc_elem(doc2))) == "b"
        destroy_doc(doc1); destroy_doc(doc2)
        println("Multiple documents: OK")
    end

end  # @testset "pugixml Runtime"

println()
println("=" ^ 70)
println("All pugixml tests passed!")
println("Julia → RepliBuild → pugixml C++ library: working end-to-end.")
println("=" ^ 70)
