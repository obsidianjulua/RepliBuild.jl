#!/usr/bin/env julia
# test_type_validation.jl - Test type validation and strictness modes

using Test
using RepliBuild
using RepliBuild.Wrapper: TypeRegistry, TypeStrictness, STRICT, WARN, PERMISSIVE
using RepliBuild.Wrapper: create_type_registry, infer_julia_type
using RepliBuild.Wrapper: is_struct_like, is_enum_like, is_function_pointer_like
using RepliBuild.ConfigurationManager: RepliBuildConfig

@testset "Type Heuristics" begin
    @testset "is_struct_like" begin
        # Should recognize struct-like names
        @test is_struct_like("Matrix3x3") == true
        @test is_struct_like("Grid") == true
        @test is_struct_like("ComplexType") == true
        @test is_struct_like("My_Struct") == true
        @test is_struct_like("CamelCase") == true

        # Should reject primitives and special cases
        @test is_struct_like("int") == false
        @test is_struct_like("double") == false
        @test is_struct_like("void") == false
        @test is_struct_like("") == false

        # Should handle pointers/references
        @test is_struct_like("Matrix3x3*") == true
        @test is_struct_like("Grid&") == true
    end

    @testset "is_function_pointer_like" begin
        # Should recognize function pointer syntax
        @test is_function_pointer_like("int (*callback)(double, double)") == true
        @test is_function_pointer_like("void (*cleanup)()") == true
        @test is_function_pointer_like("int (*)(double)") == true

        # Should reject non-function-pointers
        @test is_function_pointer_like("int*") == false
        @test is_function_pointer_like("Matrix3x3") == false
        @test is_function_pointer_like("std::function<int()>") == false
    end
end

@testset "Type Strictness Modes" begin
    # Create a minimal config for testing
    using RepliBuild.ConfigurationManager: load_config
    using TOML

    # Create temporary config file
    temp_config = joinpath(tempdir(), "test_replibuild_$(time()).toml")
    open(temp_config, "w") do io
        write(io, """
        [project]
        name = "TypeValidationTest"

        [[sources]]
        path = "test.cpp"
        """)
    end

    config = load_config(temp_config)

    @testset "PERMISSIVE mode" begin
        registry = create_type_registry(config,
            strictness=PERMISSIVE,
            allow_unknown_structs=false)  # Disable to get pure "Any" behavior

        # Unknown types should return "Any" without warning/error
        @test infer_julia_type(registry, "UnknownType") == "Any"
        @test infer_julia_type(registry, "MyCustomStruct") == "Any"
        @test infer_julia_type(registry, "SomeEnum") == "Any"
    end

    @testset "WARN mode with unknown structs allowed" begin
        registry = create_type_registry(config,
            strictness=WARN,
            allow_unknown_structs=true)

        # Unknown struct-like types should use C++ name and warn
        result = @test_logs (:warn, r"Treating unknown type.*MyStruct") begin
            infer_julia_type(registry, "MyStruct", context="test")
        end
        @test result == "MyStruct"

        # Known types should work normally
        @test infer_julia_type(registry, "int") == "Cint"
        @test infer_julia_type(registry, "double") == "Cdouble"
    end

    @testset "WARN mode with function pointers allowed" begin
        registry = create_type_registry(config,
            strictness=WARN,
            allow_function_pointers=true)

        # Function pointers should map to Ptr{Cvoid} and warn
        result = @test_logs (:warn, r"function pointer.*Ptr\{Cvoid\}") begin
            infer_julia_type(registry, "int (*callback)(double)", context="test")
        end
        @test result == "Ptr{Cvoid}"
    end

    @testset "STRICT mode" begin
        registry = create_type_registry(config,
            strictness=STRICT,
            allow_unknown_structs=false,
            allow_unknown_enums=false,
            allow_function_pointers=false)

        # Unknown types should throw helpful error
        @test_throws ErrorException infer_julia_type(registry, "UnknownType", context="test parameter")

        # Error message should be helpful
        try
            infer_julia_type(registry, "UnknownType", context="test parameter")
            @test false  # Should not reach here
        catch e
            msg = sprint(showerror, e)
            @test occursin("UnknownType", msg)
            @test occursin("test parameter", msg)
            @test occursin("Suggestions", msg)
        end

        # Known types should still work
        @test infer_julia_type(registry, "int") == "Cint"
        @test infer_julia_type(registry, "double*") == "Ptr{Cdouble}"
        @test infer_julia_type(registry, "char*") == "Cstring"
    end
end

@testset "Context-Aware Error Messages" begin
    using RepliBuild.ConfigurationManager: load_config

    # Create temporary config file
    temp_config = joinpath(tempdir(), "test_replibuild_ctx_$(time()).toml")
    open(temp_config, "w") do io
        write(io, """
        [project]
        name = "ContextTest"

        [[sources]]
        path = "test.cpp"
        """)
    end

    config = load_config(temp_config)
    registry = create_type_registry(config,
        strictness=STRICT,
        allow_unknown_structs=false,  # Ensure errors are thrown
        allow_unknown_enums=false,
        allow_function_pointers=false)

    # Test that context flows through to error messages
    @testset "Parameter context" begin
        try
            infer_julia_type(registry, "BadType", context="parameter 1 of function foo")
            @test false  # Should not reach here
        catch e
            msg = sprint(showerror, e)
            @test occursin("parameter 1 of function foo", msg)
        end
    end

    @testset "Nested type context" begin
        try
            # This should fail on the unknown element type
            infer_julia_type(registry, "UnknownElem*", context="return type")
            @test false  # Should not reach here
        catch e
            msg = sprint(showerror, e)
            # Context should mention it's a pointer base type
            @test occursin("pointer", msg) || occursin("UnknownElem", msg)
        end
    end
end

@testset "Known Type Mappings" begin
    using RepliBuild.ConfigurationManager: load_config

    # Create temporary config file
    temp_config = joinpath(tempdir(), "test_replibuild_known_$(time()).toml")
    open(temp_config, "w") do io
        write(io, """
        [project]
        name = "KnownTypesTest"

        [[sources]]
        path = "test.cpp"
        """)
    end

    config = load_config(temp_config)
    registry = create_type_registry(config, strictness=STRICT)

    @testset "Primitive types" begin
        @test infer_julia_type(registry, "void") == "Cvoid"
        @test infer_julia_type(registry, "bool") == "Bool"
        @test infer_julia_type(registry, "char") == "Cchar"
        @test infer_julia_type(registry, "int") == "Cint"
        @test infer_julia_type(registry, "unsigned int") == "Cuint"
        @test infer_julia_type(registry, "long") == "Clong"
        @test infer_julia_type(registry, "float") == "Cfloat"
        @test infer_julia_type(registry, "double") == "Cdouble"
        @test infer_julia_type(registry, "size_t") == "Csize_t"
    end

    @testset "Fixed-width integer types" begin
        @test infer_julia_type(registry, "int8_t") == "Int8"
        @test infer_julia_type(registry, "uint8_t") == "UInt8"
        @test infer_julia_type(registry, "int32_t") == "Int32"
        @test infer_julia_type(registry, "uint64_t") == "UInt64"
    end

    @testset "Pointer types" begin
        @test infer_julia_type(registry, "void*") == "Ptr{Cvoid}"
        @test infer_julia_type(registry, "char*") == "Cstring"
        @test infer_julia_type(registry, "int*") == "Ptr{Cint}"
        @test infer_julia_type(registry, "double*") == "Ptr{Cdouble}"
    end

    @testset "Const qualifiers" begin
        @test infer_julia_type(registry, "const int") == "Cint"
        @test infer_julia_type(registry, "const char*") == "Cstring"
        @test infer_julia_type(registry, "const double*") == "Ptr{Cdouble}"
    end

    @testset "Reference types" begin
        @test infer_julia_type(registry, "int&") == "Ref{Cint}"
        @test infer_julia_type(registry, "double&") == "Ref{Cdouble}"
    end

    @testset "Array types" begin
        @test infer_julia_type(registry, "int[10]") == "NTuple{10,Cint}"
        @test infer_julia_type(registry, "double[9]") == "NTuple{9,Cdouble}"
        @test infer_julia_type(registry, "char[256]") == "NTuple{256,Cchar}"
    end

    @testset "STL template types" begin
        @test infer_julia_type(registry, "std::string") == "String"

        # Note: These will fail with STRICT mode since template args need mapping
        # In real usage, allow_unknown_structs would handle this
    end
end

@testset "Custom Type Mappings" begin
    using RepliBuild.ConfigurationManager: load_config

    # Create temporary config file
    temp_config = joinpath(tempdir(), "test_replibuild_custom_$(time()).toml")
    open(temp_config, "w") do io
        write(io, """
        [project]
        name = "CustomTypesTest"

        [[sources]]
        path = "test.cpp"
        """)
    end

    config = load_config(temp_config)

    # Create registry with custom types
    custom = Dict(
        "MyCustomType" => "MyJuliaType",
        "ErrorCode" => "Cint",
        "Handle" => "Ptr{Cvoid}"
    )

    registry = create_type_registry(config,
        custom_types=custom,
        strictness=STRICT)

    @testset "Custom mappings work" begin
        @test infer_julia_type(registry, "MyCustomType") == "MyJuliaType"
        @test infer_julia_type(registry, "ErrorCode") == "Cint"
        @test infer_julia_type(registry, "Handle") == "Ptr{Cvoid}"
    end

    @testset "Custom mappings in pointers" begin
        @test infer_julia_type(registry, "MyCustomType*") == "Ptr{MyJuliaType}"
        @test infer_julia_type(registry, "ErrorCode*") == "Ptr{Cint}"
    end
end

println("\n✓ All type validation tests passed!")
