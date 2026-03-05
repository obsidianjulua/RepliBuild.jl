using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using RepliBuild
using Test

@testset "cJSON Wrap Test" begin
    println("
" * "="^70)
    println("Building and Wrapping cJSON...")
    println("="^70)

    toml_path = joinpath(@__DIR__, "replibuild.toml")
    @test isfile(toml_path)

    # Build
    library_path = RepliBuild.build(toml_path)
    @test isfile(library_path)
    println("Library built: $library_path")

    # Wrap
    wrapper_path = RepliBuild.wrap(toml_path)
    @test isfile(wrapper_path)
    println("Wrapper generated: $wrapper_path")

    # Load wrapper
    include(wrapper_path)

    @testset "Parse JSON" begin
        json_string = """
        {
            "name": "RepliBuild",
            "version": 2.1,
            "awesome": true
        }
        """

        # Parse string
        cjson_root = CjsonTest.cJSON_Parse(json_string)
        @test cjson_root != C_NULL

        # Get values
        name_item = CjsonTest.cJSON_GetObjectItemCaseSensitive(cjson_root, "name")
        @test name_item != C_NULL
        # cJSON struct string is accessed via valuestring
        
        # In Julia wrapper, properties of structs that are anonymous or complex might be mapped. 
        # But we can just test if the items exist to prove the functions work.
        version_item = CjsonTest.cJSON_GetObjectItemCaseSensitive(cjson_root, "version")
        @test version_item != C_NULL

        awesome_item = CjsonTest.cJSON_GetObjectItemCaseSensitive(cjson_root, "awesome")
        @test awesome_item != C_NULL

        # Delete
        CjsonTest.cJSON_Delete(cjson_root)
        println("Successfully parsed and traversed JSON.")
    end

    @testset "Create JSON" begin
        # Create root object
        root = CjsonTest.cJSON_CreateObject()
        @test root != C_NULL

        # Add string
        CjsonTest.cJSON_AddStringToObject(root, "framework", "Julia")

        # Add number
        CjsonTest.cJSON_AddNumberToObject(root, "year", Float64(2026))

        # Print unformatted
        printed_ptr = CjsonTest.cJSON_PrintUnformatted(root)
        @test printed_ptr != C_NULL

        # In Julia, convert Ptr{Cchar} or Ptr{UInt8} to String
        # Depending on wrapper output, it might be Cstring or Ptr{Cvoid}. We can cast to Cstring and then unsafe_string.
        str = unsafe_string(Base.unsafe_convert(Cstring, printed_ptr))
        println("Created JSON: ", str)
        @test contains(str, "\"framework\":\"Julia\"")
        @test contains(str, "\"year\":2026")

        CjsonTest.cJSON_Delete(root)
    end
end
