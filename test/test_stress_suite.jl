#!/usr/bin/env julia
# Aggressive RepliBuild Test Suite - Stress Testing with Real Scenarios

using Test
using RepliBuild

println("="^80)
println("🔥 REPLIBUILD STRESS TEST SUITE 🔥")
println("="^80)
println()

# Test 1: Simple CMake project with dependencies
@testset "Test 1: CMake + zlib dependency" begin
    println("\n[TEST 1] CMake project using zlib...")

    test_dir = mktempdir()

    # Create source file that uses zlib
    open(joinpath(test_dir, "main.cpp"), "w") do io
        write(io, """
        #include <iostream>
        #include <zlib.h>

        int main() {
            std::cout << "zlib version: " << ZLIB_VERSION << std::endl;

            // Test compression
            const char* data = "Hello, RepliBuild!";
            uLong compressed_size = compressBound(strlen(data));
            Bytef* compressed = new Bytef[compressed_size];

            int result = compress(compressed, &compressed_size,
                                (const Bytef*)data, strlen(data));

            if (result == Z_OK) {
                std::cout << "Compression successful!" << std::endl;
                delete[] compressed;
                return 0;
            }

            delete[] compressed;
            return 1;
        }
        """)
    end

    # Create CMakeLists.txt
    open(joinpath(test_dir, "CMakeLists.txt"), "w") do io
        write(io, """
        cmake_minimum_required(VERSION 3.10)
        project(ZlibTest)

        set(CMAKE_CXX_STANDARD 17)

        find_package(ZLIB REQUIRED)

        add_executable(zlib_test main.cpp)
        target_link_libraries(zlib_test ZLIB::ZLIB)
        """)
    end

    # Create replibuild.toml
    open(joinpath(test_dir, "replibuild.toml"), "w") do io
        write(io, """
        [project]
        name = "ZlibTest"
        version = "0.1.0"

        [dependencies]
        modules = ["Zlib"]

        [build]
        system = "cmake"
        build_dir = "build"
        cmake_options = ["-DCMAKE_BUILD_TYPE=Release"]

        [output]
        julia_module_name = "ZlibTest"
        """)
    end

    println("   Project created at: $test_dir")

    # Test build system detection
    detected = RepliBuild.BuildSystemDelegate.detect_build_system(test_dir)
    @test detected == RepliBuild.BuildSystemDelegate.CMAKE
    println("   ✅ Build system detected: $detected")

    # Test module resolution
    zlib_module = RepliBuild.ModuleRegistry.resolve_module("Zlib")
    if zlib_module !== nothing
        println("   ✅ Zlib module resolved")
        @test true
    else
        println("   ⚠️  Zlib module not resolved (may need installation)")
        @test_skip "Zlib module resolution"
    end

    println("   📁 Test project: $test_dir")
end

# Test 2: Missing header error - test error learning
@testset "Test 2: Error learning with missing header" begin
    println("\n[TEST 2] Testing error learning with missing header...")

    test_dir = mktempdir()

    # Create source with missing header
    open(joinpath(test_dir, "bad.cpp"), "w") do io
        write(io, """
        #include <iostream>
        #include <nonexistent_header.h>  // This will fail

        int main() {
            std::cout << "This won't compile" << std::endl;
            return 0;
        }
        """)
    end

    # Create simple Makefile
    open(joinpath(test_dir, "Makefile"), "w") do io
        write(io, """
        CXX = g++
        CXXFLAGS = -std=c++17

        bad: bad.cpp
        \t\$(CXX) \$(CXXFLAGS) -o bad bad.cpp

        .PHONY: clean
        clean:
        \trm -f bad
        """)
    end

    # Try to compile and capture error
    try
        cd(test_dir) do
            output = read(`make`, String)
        end
    catch e
        error_msg = string(e)
        println("   ✅ Caught compilation error (expected)")

        # Test error pattern detection
        pattern = RepliBuild.ErrorLearning.detect_error_pattern(error_msg)
        println("   Detected pattern: $pattern")
        @test !isnothing(pattern)

        # Test error database
        db_path = joinpath(test_dir, "test_errors.db")
        db = RepliBuild.ErrorLearning.init_db(db_path)

        (error_id, pattern_name, desc) = RepliBuild.ErrorLearning.record_error(
            db, "make", error_msg
        )

        @test error_id > 0
        println("   ✅ Error recorded in database (ID: $error_id)")

        # Test fix suggestions
        suggestions = RepliBuild.ErrorLearning.suggest_fixes(db, error_msg)
        println("   Fix suggestions: $(length(suggestions))")
        @test suggestions isa Vector
    end

    println("   📁 Test project: $test_dir")
end

# Test 3: Multi-library project
@testset "Test 3: Complex multi-library project" begin
    println("\n[TEST 3] Complex project with multiple dependencies...")

    test_dir = mktempdir()

    # Create source using multiple libraries
    open(joinpath(test_dir, "complex.cpp"), "w") do io
        write(io, """
        #include <iostream>
        #include <vector>
        #include <string>
        #include <cmath>

        // Simulated complex computation
        class DataProcessor {
        public:
            std::vector<double> process(const std::vector<double>& data) {
                std::vector<double> result;
                for (double val : data) {
                    result.push_back(std::sqrt(val * val + 1.0));
                }
                return result;
            }
        };

        int main() {
            std::cout << "Complex multi-library test" << std::endl;

            DataProcessor proc;
            std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
            auto result = proc.process(data);

            std::cout << "Processed " << result.size() << " items" << std::endl;
            return 0;
        }
        """)
    end

    # Create CMakeLists.txt
    open(joinpath(test_dir, "CMakeLists.txt"), "w") do io
        write(io, """
        cmake_minimum_required(VERSION 3.10)
        project(ComplexTest)

        set(CMAKE_CXX_STANDARD 17)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)

        add_executable(complex complex.cpp)
        target_link_libraries(complex m)  # Math library
        """)
    end

    # Test detection
    detected = RepliBuild.BuildSystemDelegate.detect_build_system(test_dir)
    @test detected == RepliBuild.BuildSystemDelegate.CMAKE
    println("   ✅ Build system detected: CMAKE")

    # Test if cmake is available
    if success(`which cmake`)
        println("   ✅ cmake available")
        @test true
    else
        println("   ⚠️  cmake not available")
        @test_skip "cmake availability"
    end

    println("   📁 Test project: $test_dir")
end

# Test 4: Makefile project with custom rules
@testset "Test 4: Custom Makefile project" begin
    println("\n[TEST 4] Custom Makefile with multiple targets...")

    test_dir = mktempdir()

    # Create multiple source files
    open(joinpath(test_dir, "util.cpp"), "w") do io
        write(io, """
        #include "util.h"
        #include <iostream>

        void print_message(const char* msg) {
            std::cout << "Message: " << msg << std::endl;
        }
        """)
    end

    open(joinpath(test_dir, "util.h"), "w") do io
        write(io, """
        #ifndef UTIL_H
        #define UTIL_H

        void print_message(const char* msg);

        #endif
        """)
    end

    open(joinpath(test_dir, "main.cpp"), "w") do io
        write(io, """
        #include "util.h"

        int main() {
            print_message("RepliBuild test");
            return 0;
        }
        """)
    end

    # Create complex Makefile
    open(joinpath(test_dir, "Makefile"), "w") do io
        write(io, """
        CXX = g++
        CXXFLAGS = -std=c++17 -Wall -O2

        OBJS = main.o util.o
        TARGET = testapp

        all: \$(TARGET)

        \$(TARGET): \$(OBJS)
        \t\$(CXX) \$(CXXFLAGS) -o \$(TARGET) \$(OBJS)

        main.o: main.cpp util.h
        \t\$(CXX) \$(CXXFLAGS) -c main.cpp

        util.o: util.cpp util.h
        \t\$(CXX) \$(CXXFLAGS) -c util.cpp

        clean:
        \trm -f \$(OBJS) \$(TARGET)

        .PHONY: all clean
        """)
    end

    # Test detection
    detected = RepliBuild.BuildSystemDelegate.detect_build_system(test_dir)
    @test detected == RepliBuild.BuildSystemDelegate.MAKE
    println("   ✅ Build system detected: MAKE")

    println("   📁 Test project: $test_dir")
end

# Test 5: pkg-config integration
@testset "Test 5: pkg-config integration" begin
    println("\n[TEST 5] Testing pkg-config integration...")

    # Test available packages
    if success(`which pkg-config`)
        common_packages = ["zlib", "sqlite3", "libpng", "libcurl"]
        found_packages = String[]

        for pkg in common_packages
            if success(`pkg-config --exists $pkg`)
                push!(found_packages, pkg)

                # Get package info
                version = readchomp(`pkg-config --modversion $pkg`)
                cflags = try readchomp(`pkg-config --cflags $pkg`) catch; "" end
                libs = readchomp(`pkg-config --libs $pkg`)

                println("   ✅ $pkg v$version")
                println("      CFLAGS: $(isempty(cflags) ? "(none)" : cflags)")
                println("      LIBS: $libs")
            end
        end

        @test length(found_packages) > 0
        println("   Found $(length(found_packages)) packages via pkg-config")
    else
        println("   ⚠️  pkg-config not available")
        @test_skip "pkg-config availability"
    end
end

# Test 6: Module resolution stress test
@testset "Test 6: Module resolution stress test" begin
    println("\n[TEST 6] Testing module resolution for all available modules...")

    modules = RepliBuild.ModuleRegistry.list_modules()
    println("   Found $(length(modules)) modules")

    resolved_count = 0
    failed_count = 0

    for mod in modules
        mod_name = mod isa String ? mod : (mod isa RepliBuild.ModuleRegistry.ModuleInfo ? mod.name : string(mod))

        try
            info = RepliBuild.ModuleRegistry.resolve_module(mod_name)
            if info !== nothing
                resolved_count += 1
                println("   ✅ $mod_name v$(info.version) - resolved")
            else
                failed_count += 1
                println("   ⚠️  $mod_name - failed to resolve (returned nothing)")
            end
        catch e
            failed_count += 1
            println("   ❌ $mod_name - error: $(typeof(e))")
        end
    end

    @test resolved_count > 0
    println("   Resolved: $resolved_count / $(length(modules)), Failed: $failed_count")
end

# Test 7: Error statistics
@testset "Test 7: Error learning statistics" begin
    println("\n[TEST 7] Testing error learning statistics...")

    db_path = tempname() * ".db"
    db = RepliBuild.ErrorLearning.init_db(db_path)

    # Add some test errors
    errors_data = [
        ("g++ test.cpp", "error: 'iostream' file not found"),
        ("g++ test2.cpp", "undefined reference to `pthread_create'"),
        ("g++ test3.cpp", "error: 'boost/filesystem.hpp' file not found"),
    ]

    for (cmd, error_msg) in errors_data
        RepliBuild.ErrorLearning.record_error(db, cmd, error_msg)
    end

    # Get statistics
    stats = RepliBuild.ErrorLearning.get_error_stats(db)

    @test haskey(stats, "total_errors")
    @test stats["total_errors"] == 3

    println("   ✅ Recorded $(stats["total_errors"]) errors")
    println("   Total fixes: $(stats["total_fixes"])")
    println("   Success rate: $(stats["success_rate"])")

    # Export to markdown
    md_path = tempname() * ".md"
    RepliBuild.ErrorLearning.export_to_markdown(db, md_path)

    @test isfile(md_path)
    println("   ✅ Exported to markdown: $md_path")

    rm(db_path)
    rm(md_path)
end

# Test 8: Build system delegate creation
@testset "Test 8: Build system delegate creation" begin
    println("\n[TEST 8] Testing build system delegate creation...")

    test_cases = [
        ("cmake", RepliBuild.BuildSystemDelegate.CMAKE),
        ("make", RepliBuild.BuildSystemDelegate.MAKE),
        ("CMAKE", RepliBuild.BuildSystemDelegate.CMAKE),
        ("Make", RepliBuild.BuildSystemDelegate.MAKE),
    ]

    for (input, expected) in test_cases
        result = RepliBuild.BuildSystemDelegate.parse_build_system_string(input)
        @test result == expected
        println("   ✅ '$input' → $result")
    end
end

println()
println("="^80)
println("🎉 STRESS TEST SUITE COMPLETE 🎉")
println("="^80)
