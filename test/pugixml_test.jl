using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using RepliBuild

# Load the generated wrapper
include(joinpath(@__DIR__, "..", "hub", "packages", "pugixml", "julia", "Pugixml.jl"))
using .Pugixml

function run_pugixml_test()
    println("--- Testing Pugixml wrapper ---")
    
    # 1. Instantiate the document manually
    doc_ptr = Base.Libc.malloc(sizeof(Pugixml.xml_document))
    
    # Initialize via constructor
    Pugixml.pugi_xml_document_xml_document(doc_ptr)
    println("Document initialized.")
    
    # 2. Parse a string
    xml_content = "<?xml version=\"1.0\"?><test>Hello</test>"
    
    # arg1: string pointer, arg2: options (0 for default)
    # This will hit the Tier 2 MLIR JIT as it returns xml_parse_result by value
    res = Pugixml.pugi_xml_document_load_string(doc_ptr, pointer(xml_content), 0)
    
    # Check result
    success = Pugixml.pugi_xml_parse_result_operator_bool(Ref(res))
    println("Parse result success: ", success)
    
    if success
        println("Success! Pugixml parsed the string successfully via JIT.")
    else
        desc = Pugixml.pugi_xml_parse_result_description(Ref(res))
        println("Failed to parse. Reason: ", unsafe_string(desc))
    end
    
    # Clean up document
    Pugixml.pugi_xml_document_destroy_xml_document(doc_ptr)
    Base.Libc.free(doc_ptr)
end

run_pugixml_test()
