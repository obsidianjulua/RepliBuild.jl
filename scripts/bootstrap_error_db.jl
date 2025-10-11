#!/usr/bin/env julia
# Bootstrap the error knowledge database with common patterns

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# Install SQLite if needed
println("📦 Checking dependencies...")
try
    using SQLite
    println("✅ SQLite already installed")
catch
    println("📥 Installing SQLite...")
    Pkg.add("SQLite")
    println("✅ SQLite installed")
end

# Load RepliBuild modules
include(joinpath(@__DIR__, "..", "src", "BuildBridge.jl"))
using .BuildBridge

println("🚀 Bootstrapping RepliBuild Error Knowledge Database")
println("=" ^ 60)

# Initialize database
db_path = get(ENV, "REPLIBUILD_ERROR_DB", "replibuild_errors.db")
println("Database path: $db_path")

db = get_error_db(db_path)
println("✅ Database initialized")

# Bootstrap with common errors
println("\n📚 Loading common error patterns...")
bootstrap_common_errors(db)

println("\n✅ Bootstrap complete!")
println("=" ^ 60)
println("\n📊 Database Statistics:")

# Query statistics using SQLite
import SQLite

error_count_result = SQLite.DBInterface.execute(db.conn, "SELECT COUNT(*) as count FROM error_patterns")
error_count = first(error_count_result).count

fix_count_result = SQLite.DBInterface.execute(db.conn, "SELECT COUNT(*) as count FROM error_fixes")
fix_count = first(fix_count_result).count

println("  • Error patterns: $error_count")
println("  • Known fixes: $fix_count")

if db.use_embeddings
    println("  • Semantic search: ✅ Enabled (using EngineT)")
else
    println("  • Semantic search: ⚠️  Disabled (EngineT not available)")
end

println("\n💡 Usage:")
println("  julia> using RepliBuild")
println("  julia> output, exitcode, attempts, suggestions = BuildBridge.compile_with_learning(")
println("           \"clang++\", [\"myfile.cpp\", \"-o\", \"myfile.o\"])")
println("\n  The system will automatically suggest or apply fixes for known errors!")
