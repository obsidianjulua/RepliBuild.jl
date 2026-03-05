using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using RepliBuild
using Test

@testset "SQLite Integration" begin
    println("\n" * "="^70)
    println("Building and Wrapping SQLite 3.49.1...")
    println("="^70)

    toml_path = joinpath(@__DIR__, "replibuild.toml")
    @test isfile(toml_path)

    # =========================================================================
    # Build
    # =========================================================================
    @testset "Build" begin
        library_path = RepliBuild.build(toml_path, clean=true)
        @test isfile(library_path)
        @test filesize(library_path) > 1_000_000  # sqlite .so is > 1 MB
        println("Library built: $library_path ($(round(filesize(library_path)/1024/1024, digits=2)) MB)")
    end

    # =========================================================================
    # Wrap
    # =========================================================================
    @testset "Wrap" begin
        wrapper_path = RepliBuild.wrap(toml_path)
        @test isfile(wrapper_path)
        lines = readlines(wrapper_path)
        @test length(lines) > 1000   # substantial wrapper
        # Check critical functions were wrapped
        src = join(lines, "\n")
        for fn in ["sqlite3_open", "sqlite3_exec", "sqlite3_prepare_v2",
                   "sqlite3_step", "sqlite3_finalize", "sqlite3_close",
                   "sqlite3_column_int", "sqlite3_column_text",
                   "sqlite3_libversion", "sqlite3_errmsg"]
            @test occursin(fn, src)
        end
        println("Wrapper generated: $wrapper_path ($(length(lines)) lines)")
    end

    # =========================================================================
    # Load wrapper
    # =========================================================================
    wrapper_path = joinpath(@__DIR__, "julia", "SqliteTest.jl")
    @test isfile(wrapper_path)
    include(wrapper_path)
    S = Main.SqliteTest   # alias

    # SQLite result codes
    SQLITE_OK   = 0
    SQLITE_ROW  = 100
    SQLITE_DONE = 101

    # =========================================================================
    # Test 1: Version string
    # =========================================================================
    @testset "Version" begin
        ver = S.sqlite3_libversion()
        @test ver isa String
        @test startswith(ver, "3.")
        vernum = S.sqlite3_libversion_number()
        @test vernum >= 3_040_000   # 3.40.0+
        println("SQLite version: $ver  ($vernum)")
    end

    # =========================================================================
    # Helper: open an in-memory database, run tests, close
    # =========================================================================
    function with_db(f)
        ppDb = Ref{Ptr{S.sqlite3}}(C_NULL)
        rc = S.sqlite3_open(":memory:", ppDb)
        @test rc == SQLITE_OK
        db = ppDb[]
        @test db != C_NULL
        try
            f(db)
        finally
            S.sqlite3_close(db)
        end
    end

    # =========================================================================
    # Test 2: Open / close in-memory database
    # =========================================================================
    @testset "Open/close" begin
        with_db(db -> @test db != C_NULL)
        println("In-memory database: open/close OK")
    end

    # =========================================================================
    # Test 3: sqlite3_exec — CREATE TABLE + INSERT
    # =========================================================================
    @testset "exec CREATE/INSERT" begin
        with_db() do db
            rc = S.sqlite3_exec(db,
                "CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT, val REAL);",
                C_NULL, C_NULL, C_NULL)
            @test rc == SQLITE_OK

            rc = S.sqlite3_exec(db,
                "INSERT INTO t VALUES (1,'alpha',1.5),(2,'beta',2.5),(3,'gamma',3.5);",
                C_NULL, C_NULL, C_NULL)
            @test rc == SQLITE_OK

            # COUNT
            ppStmt = Ref{Ptr{S.sqlite3_stmt}}(C_NULL)
            rc = S.sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM t;", -1, ppStmt, C_NULL)
            @test rc == SQLITE_OK
            stmt = ppStmt[]
            @test S.sqlite3_step(stmt) == SQLITE_ROW
            @test S.sqlite3_column_int(stmt, 0) == 3
            S.sqlite3_finalize(stmt)
        end
        println("exec CREATE/INSERT: OK")
    end

    # =========================================================================
    # Test 4: Prepared statement — SELECT with typed column reads
    # =========================================================================
    @testset "SELECT rows" begin
        with_db() do db
            S.sqlite3_exec(db,
                "CREATE TABLE nums(id INT, label TEXT, score REAL);",
                C_NULL, C_NULL, C_NULL)
            S.sqlite3_exec(db,
                "INSERT INTO nums VALUES(10,'ten',3.14),(20,'twenty',2.71),(30,'thirty',1.41);",
                C_NULL, C_NULL, C_NULL)

            ppStmt = Ref{Ptr{S.sqlite3_stmt}}(C_NULL)
            rc = S.sqlite3_prepare_v2(db,
                "SELECT id, label, score FROM nums ORDER BY id;", -1, ppStmt, C_NULL)
            @test rc == SQLITE_OK
            stmt = ppStmt[]

            expected = [(10,"ten",3.14), (20,"twenty",2.71), (30,"thirty",1.41)]
            for (eid, elabel, escore) in expected
                @test S.sqlite3_step(stmt) == SQLITE_ROW
                @test S.sqlite3_column_int(stmt, 0) == eid
                raw_text = S.sqlite3_column_text(stmt, 1)
                @test unsafe_string(raw_text) == elabel
                @test S.sqlite3_column_double(stmt, 2) ≈ escore atol=1e-9
            end
            @test S.sqlite3_step(stmt) == SQLITE_DONE
            S.sqlite3_finalize(stmt)
        end
        println("SELECT rows: OK")
    end

    # =========================================================================
    # Test 5: Parameterized bind (prevent SQL injection)
    # =========================================================================
    @testset "Bind parameters" begin
        with_db() do db
            S.sqlite3_exec(db,
                "CREATE TABLE users(name TEXT, age INT);",
                C_NULL, C_NULL, C_NULL)

            ppStmt = Ref{Ptr{S.sqlite3_stmt}}(C_NULL)
            rc = S.sqlite3_prepare_v2(db,
                "INSERT INTO users(name, age) VALUES(?, ?);", -1, ppStmt, C_NULL)
            @test rc == SQLITE_OK
            stmt = ppStmt[]

            data = [("Alice", 30), ("Bob", 25), ("Carol", 35)]
            for (name, age) in data
                S.sqlite3_bind_text(stmt, 1, name, length(name), C_NULL)
                S.sqlite3_bind_int(stmt, 2, age)
                @test S.sqlite3_step(stmt) == SQLITE_DONE
                S.sqlite3_reset(stmt)
            end
            S.sqlite3_finalize(stmt)

            # Verify
            ppStmt2 = Ref{Ptr{S.sqlite3_stmt}}(C_NULL)
            S.sqlite3_prepare_v2(db,
                "SELECT name, age FROM users ORDER BY age;", -1, ppStmt2, C_NULL)
            stmt2 = ppStmt2[]

            expected_ordered = [("Bob",25), ("Alice",30), ("Carol",35)]
            for (ename, eage) in expected_ordered
                @test S.sqlite3_step(stmt2) == SQLITE_ROW
                @test unsafe_string(S.sqlite3_column_text(stmt2, 0)) == ename
                @test S.sqlite3_column_int(stmt2, 1) == eage
            end
            @test S.sqlite3_step(stmt2) == SQLITE_DONE
            S.sqlite3_finalize(stmt2)
        end
        println("Bind parameters: OK")
    end

    # =========================================================================
    # Test 6: Transactions — commit and rollback
    # =========================================================================
    @testset "Transactions" begin
        with_db() do db
            S.sqlite3_exec(db, "CREATE TABLE acct(balance INT);", C_NULL, C_NULL, C_NULL)
            S.sqlite3_exec(db, "INSERT INTO acct VALUES(100);", C_NULL, C_NULL, C_NULL)

            # Committed transaction
            S.sqlite3_exec(db, "BEGIN;", C_NULL, C_NULL, C_NULL)
            S.sqlite3_exec(db, "UPDATE acct SET balance = balance + 50;", C_NULL, C_NULL, C_NULL)
            S.sqlite3_exec(db, "COMMIT;", C_NULL, C_NULL, C_NULL)

            ppStmt = Ref{Ptr{S.sqlite3_stmt}}(C_NULL)
            S.sqlite3_prepare_v2(db, "SELECT balance FROM acct;", -1, ppStmt, C_NULL)
            stmt = ppStmt[]
            @test S.sqlite3_step(stmt) == SQLITE_ROW
            @test S.sqlite3_column_int(stmt, 0) == 150
            S.sqlite3_finalize(stmt)

            # Rolled-back transaction
            S.sqlite3_exec(db, "BEGIN;", C_NULL, C_NULL, C_NULL)
            S.sqlite3_exec(db, "UPDATE acct SET balance = 0;", C_NULL, C_NULL, C_NULL)
            S.sqlite3_exec(db, "ROLLBACK;", C_NULL, C_NULL, C_NULL)

            ppStmt2 = Ref{Ptr{S.sqlite3_stmt}}(C_NULL)
            S.sqlite3_prepare_v2(db, "SELECT balance FROM acct;", -1, ppStmt2, C_NULL)
            stmt2 = ppStmt2[]
            @test S.sqlite3_step(stmt2) == SQLITE_ROW
            @test S.sqlite3_column_int(stmt2, 0) == 150   # unchanged
            S.sqlite3_finalize(stmt2)
        end
        println("Transactions: OK")
    end

    # =========================================================================
    # Test 7: sqlite3_exec callback — iterate rows via C callback
    # =========================================================================
    @testset "exec callback" begin
        with_db() do db
            S.sqlite3_exec(db,
                "CREATE TABLE items(v INT); INSERT INTO items VALUES(7),(14),(21);",
                C_NULL, C_NULL, C_NULL)

            results = Int[]
            function row_cb(pArg::Ptr{Cvoid}, ncols::Cint, cols::Ptr{Ptr{UInt8}}, names::Ptr{Ptr{UInt8}})::Cint
                val_ptr = unsafe_load(cols, 1)  # first (and only) column
                push!(results, parse(Int, unsafe_string(val_ptr)))
                return Cint(0)
            end

            cb = @cfunction($row_cb, Cint, (Ptr{Cvoid}, Cint, Ptr{Ptr{UInt8}}, Ptr{Ptr{UInt8}}))
            rc = S.sqlite3_exec(db, "SELECT v FROM items ORDER BY v;", cb, C_NULL, C_NULL)
            @test rc == SQLITE_OK
            @test results == [7, 14, 21]
        end
        println("exec callback: OK")
    end

    # =========================================================================
    # Test 8: Error handling — invalid SQL returns an error code
    # =========================================================================
    @testset "Error handling" begin
        with_db() do db
            ppStmt = Ref{Ptr{S.sqlite3_stmt}}(C_NULL)
            rc = S.sqlite3_prepare_v2(db, "THIS IS NOT SQL", -1, ppStmt, C_NULL)
            @test rc != SQLITE_OK
            msg = S.sqlite3_errmsg(db)
            @test msg isa String
            @test !isempty(msg)
            println("  Error message: \"$msg\"")
        end
        println("Error handling: OK")
    end

    # =========================================================================
    # Test 9: last_insert_rowid
    # =========================================================================
    @testset "last_insert_rowid" begin
        with_db() do db
            S.sqlite3_exec(db, "CREATE TABLE r(id INTEGER PRIMARY KEY AUTOINCREMENT, v INT);",
                C_NULL, C_NULL, C_NULL)
            S.sqlite3_exec(db, "INSERT INTO r(v) VALUES(99);", C_NULL, C_NULL, C_NULL)
            rowid = S.sqlite3_last_insert_rowid(db)
            @test rowid == 1
            S.sqlite3_exec(db, "INSERT INTO r(v) VALUES(100);", C_NULL, C_NULL, C_NULL)
            @test S.sqlite3_last_insert_rowid(db) == 2
        end
        println("last_insert_rowid: OK")
    end

    # =========================================================================
    # Test 10: Disk-backed database (real file I/O)
    # =========================================================================
    @testset "Disk database" begin
        db_path = tempname() * ".sqlite3"
        try
            ppDb = Ref{Ptr{S.sqlite3}}(C_NULL)
            @test S.sqlite3_open(db_path, ppDb) == SQLITE_OK
            db = ppDb[]
            S.sqlite3_exec(db, "CREATE TABLE data(x INT);", C_NULL, C_NULL, C_NULL)
            S.sqlite3_exec(db, "INSERT INTO data VALUES(42);", C_NULL, C_NULL, C_NULL)
            S.sqlite3_close(db)

            # Re-open and verify data persisted
            @test S.sqlite3_open(db_path, ppDb) == SQLITE_OK
            db2 = ppDb[]
            ppStmt = Ref{Ptr{S.sqlite3_stmt}}(C_NULL)
            S.sqlite3_prepare_v2(db2, "SELECT x FROM data;", -1, ppStmt, C_NULL)
            stmt = ppStmt[]
            @test S.sqlite3_step(stmt) == SQLITE_ROW
            @test S.sqlite3_column_int(stmt, 0) == 42
            S.sqlite3_finalize(stmt)
            S.sqlite3_close(db2)
            println("  Database file: $db_path")
        finally
            isfile(db_path) && rm(db_path)
        end
        println("Disk database: OK")
    end

    println("\n" * "="^70)
    println("All SQLite tests passed!")
    println("Julia → RepliBuild → SQLite C library: working end-to-end.")
    println("="^70)
end
