#!/usr/bin/env julia
# Precompilation statements for RepliBuild
# This file helps PackageCompiler.jl create an optimized sysimage

using RepliBuild

println("Running RepliBuild precompilation...")

# Core initialization functions
RepliBuild.info()
RepliBuild.help()

# BuildBridge functions - command checking
RepliBuild.BuildBridge.command_exists("gcc")
RepliBuild.BuildBridge.command_exists("clang")
RepliBuild.BuildBridge.command_exists("julia")
try
    RepliBuild.BuildBridge.find_executable("gcc")
catch e
    # Expected to fail sometimes, that's ok
end

# JuliaWrapItUp core functions - identifier handling
RepliBuild.JuliaWrapItUp.make_julia_identifier("test_function")
RepliBuild.JuliaWrapItUp.make_julia_identifier("MyClass::method")
RepliBuild.JuliaWrapItUp.make_julia_identifier("operator+")
RepliBuild.JuliaWrapItUp.make_julia_identifier("123invalid")
RepliBuild.JuliaWrapItUp.make_julia_identifier("while")  # reserved word

# Module name generation
RepliBuild.JuliaWrapItUp.generate_module_name("testlib")
RepliBuild.JuliaWrapItUp.generate_module_name("libmath")
RepliBuild.JuliaWrapItUp.generate_module_name("lib_crypto_ssl")

# Binary type identification
RepliBuild.JuliaWrapItUp.identify_binary_type("/usr/lib/libc.so")
RepliBuild.JuliaWrapItUp.identify_binary_type("test.a")
RepliBuild.JuliaWrapItUp.identify_binary_type("test.o")
RepliBuild.JuliaWrapItUp.identify_binary_type("test.dylib")

# Configuration creation
wrapper_config = RepliBuild.JuliaWrapItUp.create_default_wrapper_config()

# Type registry loading
type_registry = RepliBuild.JuliaWrapItUp.load_type_registry(wrapper_config)

# Create a temporary wrapper config file for testing
temp_dir = mktempdir()
temp_config = joinpath(temp_dir, "test_wrapper_config.toml")
try
    RepliBuild.JuliaWrapItUp.save_wrapper_config(wrapper_config, temp_config)

    # Now create a BinaryWrapper using the temporary config
    test_wrapper = RepliBuild.JuliaWrapItUp.BinaryWrapper(temp_config)

    # Type inference with different types
    RepliBuild.JuliaWrapItUp.infer_julia_type(test_wrapper, "int")
    RepliBuild.JuliaWrapItUp.infer_julia_type(test_wrapper, "void*")
    RepliBuild.JuliaWrapItUp.infer_julia_type(test_wrapper, "const char*")
    RepliBuild.JuliaWrapItUp.infer_julia_type(test_wrapper, "float")
    RepliBuild.JuliaWrapItUp.infer_julia_type(test_wrapper, "double")
    RepliBuild.JuliaWrapItUp.infer_julia_type(test_wrapper, "uint64_t")
    RepliBuild.JuliaWrapItUp.infer_julia_type(test_wrapper, "std::string")
    RepliBuild.JuliaWrapItUp.infer_julia_type(test_wrapper, "int*")
    RepliBuild.JuliaWrapItUp.infer_julia_type(test_wrapper, "const int&")
    RepliBuild.JuliaWrapItUp.infer_julia_type(test_wrapper, nothing)

catch e
    @warn "Wrapper instance creation skipped: $e"
finally
    rm(temp_dir, recursive=true, force=true)
end

# Symbol parsing - various C++ signatures
RepliBuild.JuliaWrapItUp.parse_symbol_signature("myfunction(int, float)")
RepliBuild.JuliaWrapItUp.parse_symbol_signature("std::vector<int>::push_back(int const&)")
RepliBuild.JuliaWrapItUp.parse_symbol_signature("operator+(MyClass const&, MyClass const&)")
RepliBuild.JuliaWrapItUp.parse_symbol_signature("simple_func")
RepliBuild.JuliaWrapItUp.parse_symbol_signature("void function()")
RepliBuild.JuliaWrapItUp.parse_symbol_signature("int* getPointer(const char*)")

# Parameter parsing
RepliBuild.JuliaWrapItUp.parse_single_parameter("const int value")
RepliBuild.JuliaWrapItUp.parse_single_parameter("float* ptr")
RepliBuild.JuliaWrapItUp.parse_single_parameter("std::string&")
RepliBuild.JuliaWrapItUp.parse_single_parameter("volatile int x")
RepliBuild.JuliaWrapItUp.parse_single_parameter("double")

# Parameter list parsing
RepliBuild.JuliaWrapItUp.parse_parameter_list("int a, float b, const char* c")
RepliBuild.JuliaWrapItUp.parse_parameter_list("std::vector<int> v, std::map<std::string, int> m")
RepliBuild.JuliaWrapItUp.parse_parameter_list("void")
RepliBuild.JuliaWrapItUp.parse_parameter_list("")

# Templates functions
try
    RepliBuild.Templates.detect_project_type([])
    RepliBuild.Templates.detect_project_type(["main.cpp", "test.h"])
    RepliBuild.Templates.detect_project_type(["main.c", "utils.c"])
catch e
    @warn "Template functions skipped: $e"
end

# Test isexecutable helper
try
    RepliBuild.JuliaWrapItUp.isexecutable("/usr/bin/gcc")
    RepliBuild.JuliaWrapItUp.isexecutable("/usr/lib/libc.so")
catch e
    # May fail on some systems
end

println("✅ Precompilation statements executed successfully")
