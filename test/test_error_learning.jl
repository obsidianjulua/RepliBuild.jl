#!/usr/bin/env julia
# Test error learning system

using Test
using RepliBuild
using SQLite
using DataFrames

@testset "Error Learning System" begin
    # Test database creation
    @testset "Database Creation" begin
        db_path = tempname() * ".db"

        try
            # Initialize database
            db = RepliBuild.ErrorLearning.init_db(db_path)

            @test isfile(db_path)
            @test !isnothing(db)

            # Verify tables exist
            tables = DataFrame(SQLite.DBInterface.execute(db, "SELECT name FROM sqlite_master WHERE type='table'"))
            @test "compilation_errors" in tables.name
            @test "error_patterns" in tables.name
            @test "error_fixes" in tables.name

            println("✅ Database created successfully at: $db_path")
        finally
            isfile(db_path) && rm(db_path)
        end
    end

    # Test error recording
    @testset "Error Recording" begin
        db_path = tempname() * ".db"

        try
            db = RepliBuild.ErrorLearning.init_db(db_path)

            # Record a sample error
            error_output = """
            error: 'iostream' file not found
            #include <iostream>
                     ^~~~~~~~~~
            1 error generated.
            """

            (error_id, pattern_name, description) = RepliBuild.ErrorLearning.record_error(
                db,
                "clang++ -o test test.cpp",
                error_output
            )

            @test error_id > 0
            @test pattern_name == "missing_header"
            @test !isempty(description)

            println("✅ Error recorded with ID: $error_id, Pattern: $pattern_name")

            # Verify error was stored
            errors = DataFrame(SQLite.DBInterface.execute(db,
                "SELECT * FROM compilation_errors WHERE id = $error_id"))
            @test nrow(errors) == 1
            @test errors.pattern_name[1] == "missing_header"

        finally
            isfile(db_path) && rm(db_path)
        end
    end

    # Test error pattern detection
    @testset "Error Pattern Detection" begin
        test_cases = [
            ("'iostream' file not found", "missing_header"),
            ("undefined reference to `pthread_create'", "missing_library"),
            ("no member named 'foo' in namespace 'std'", "wrong_namespace"),
            ("expected ';' after expression", "syntax_error"),
            ("undefined symbol: _ZN3std", "abi_mismatch"),
        ]

        for (error_msg, expected_pattern) in test_cases
            pattern = RepliBuild.ErrorLearning.detect_error_pattern(error_msg)
            @test pattern == expected_pattern
            println("✅ Pattern detection: '$error_msg' → $pattern")
        end
    end

    # Test fix recording
    @testset "Fix Recording" begin
        db_path = tempname() * ".db"

        try
            db = RepliBuild.ErrorLearning.init_db(db_path)

            # Record error
            (error_id, _, _) = RepliBuild.ErrorLearning.record_error(
                db,
                "clang++ test.cpp",
                "error: 'iostream' file not found"
            )

            # Record successful fix
            RepliBuild.ErrorLearning.record_fix(
                db,
                error_id,
                "Added -I/usr/include to compiler flags",
                "clang++ -I/usr/include test.cpp",
                "added_include_path",
                true  # successful
            )

            # Verify fix was recorded
            fixes = DataFrame(SQLite.DBInterface.execute(db,
                "SELECT * FROM error_fixes WHERE error_id = $error_id"))
            @test nrow(fixes) == 1
            @test fixes.successful[1] == 1

            println("✅ Fix recorded for error $error_id")

        finally
            isfile(db_path) && rm(db_path)
        end
    end

    # Test fix suggestions
    @testset "Fix Suggestions" begin
        db_path = tempname() * ".db"

        try
            db = RepliBuild.ErrorLearning.init_db(db_path)

            # Create similar errors with successful fixes
            for i in 1:3
                (error_id, _, _) = RepliBuild.ErrorLearning.record_error(
                    db,
                    "clang++ test.cpp",
                    "error: 'boost/filesystem.hpp' file not found"
                )

                RepliBuild.ErrorLearning.record_fix(
                    db,
                    error_id,
                    "Added Boost to dependencies",
                    "clang++ test.cpp -lboost_filesystem",
                    "added_library",
                    true
                )
            end

            # Get suggestions for similar error
            suggestions = RepliBuild.ErrorLearning.suggest_fixes(
                db,
                "error: 'boost/system.hpp' file not found"
            )

            @test length(suggestions) > 0
            @test any(s -> contains(s["description"], "Boost"), suggestions)

            println("✅ Got $(length(suggestions)) fix suggestions")
            for (i, sug) in enumerate(suggestions)
                println("   $i. $(sug["description"]) (confidence: $(sug["confidence"]))")
            end

        finally
            isfile(db_path) && rm(db_path)
        end
    end

    # Test error statistics
    @testset "Error Statistics" begin
        db_path = tempname() * ".db"

        try
            db = RepliBuild.ErrorLearning.init_db(db_path)

            # Add multiple errors
            for i in 1:5
                RepliBuild.ErrorLearning.record_error(
                    db,
                    "clang++ test$i.cpp",
                    "error: random error $i"
                )
            end

            # Get stats
            stats = RepliBuild.ErrorLearning.get_error_stats(db)

            @test stats[:total_errors] == 5
            @test stats[:total_fixes] == 0
            @test stats[:unique_patterns] >= 0

            println("✅ Statistics: $(stats[:total_errors]) errors, $(stats[:total_fixes]) fixes")

        finally
            isfile(db_path) && rm(db_path)
        end
    end

    # Test markdown export
    @testset "Markdown Export" begin
        db_path = tempname() * ".db"
        md_path = tempname() * ".md"

        try
            db = RepliBuild.ErrorLearning.init_db(db_path)

            # Add some test data
            (error_id, _, _) = RepliBuild.ErrorLearning.record_error(
                db,
                "clang++ test.cpp",
                "error: 'iostream' file not found",
                project_path=pwd()
            )

            RepliBuild.ErrorLearning.record_fix(
                db,
                error_id,
                "Added standard include path",
                "clang++ -I/usr/include test.cpp",
                "added_include",
                true
            )

            # Export to markdown
            RepliBuild.ErrorLearning.export_to_markdown(db, md_path)

            @test isfile(md_path)

            content = read(md_path, String)
            @test contains(content, "# RepliBuild Error Learning Log")
            @test contains(content, "iostream")

            println("✅ Markdown exported to: $md_path")
            println("   File size: $(filesize(md_path)) bytes")

        finally
            isfile(db_path) && rm(db_path)
            isfile(md_path) && rm(md_path)
        end
    end

    # Test integration with BuildBridge
    @testset "BuildBridge Integration" begin
        db_path = tempname() * ".db"

        try
            # Get error DB through BuildBridge
            db = RepliBuild.BuildBridge.get_error_db(db_path)
            @test !isnothing(db)

            # Test export functions
            export_path = tempname() * ".md"
            RepliBuild.BuildBridge.export_error_log(db_path, export_path)
            @test isfile(export_path)

            # Test stats
            stats = RepliBuild.BuildBridge.get_error_stats(db_path)
            @test haskey(stats, :total_errors)

            println("✅ BuildBridge integration working")
            rm(export_path)

        finally
            isfile(db_path) && rm(db_path)
        end
    end
end

println("\n" * "="^60)
println("Error Learning System Tests Complete!")
println("="^60)
