#!/usr/bin/env julia
# ErrorLearning.jl - Compilation error pattern learning and fix suggestions
# Stores errors and successful fixes in SQLite database for future reference

module ErrorLearning

using SQLite
using DBInterface
using DataFrames
using Dates

# ============================================================================
# DATABASE SCHEMA
# ============================================================================

"""
Initialize or connect to the error learning database
"""
function init_db(db_path::String="replibuild_errors.db")
    db = SQLite.DB(db_path)

    # Create errors table
    DBInterface.execute(db, """
        CREATE TABLE IF NOT EXISTS compilation_errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            command TEXT NOT NULL,
            error_output TEXT NOT NULL,
            error_pattern TEXT,
            project_path TEXT,
            file_path TEXT
        )
    """)

    # Create fixes table
    DBInterface.execute(db, """
        CREATE TABLE IF NOT EXISTS error_fixes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            error_id INTEGER,
            timestamp TEXT NOT NULL,
            fix_description TEXT NOT NULL,
            fix_action TEXT NOT NULL,
            fix_type TEXT NOT NULL,
            success INTEGER NOT NULL,
            FOREIGN KEY (error_id) REFERENCES compilation_errors(id)
        )
    """)

    # Create pattern index for faster lookups
    DBInterface.execute(db, """
        CREATE INDEX IF NOT EXISTS idx_error_pattern
        ON compilation_errors(error_pattern)
    """)

    return db
end

# ============================================================================
# ERROR PATTERN DETECTION
# ============================================================================

"""
Common compiler error patterns
"""
const ERROR_PATTERNS = [
    (r"fatal error: '([^']+)' file not found", "missing_header", "Missing header file: \$1"),
    (r"error: no such file or directory: '([^']+)'", "missing_file", "File not found: \$1"),
    (r"undefined reference to [`']([^'`]+)'", "undefined_symbol", "Undefined symbol: \$1"),
    (r"error: use of undeclared identifier '([^']+)'", "undeclared_identifier", "Undeclared: \$1"),
    (r"error: expected ';' (.*)", "missing_semicolon", "Missing semicolon"),
    (r"error: no matching function for call to '([^']+)'", "no_matching_function", "No match: \$1"),
    (r"error: cannot convert '(.*)' to '(.*)'", "type_conversion", "Type mismatch: \$1 -> \$2"),
    (r"ld: library not found for -l(\w+)", "missing_library", "Missing library: lib\$1"),
    (r"error: unknown type name '([^']+)'", "unknown_type", "Unknown type: \$1"),
    (r"/usr/bin/ld: cannot find -l(\w+)", "linker_library_missing", "Linker missing: lib\$1")
]

"""
Extract error pattern from compiler output
"""
function detect_error_pattern(error_output::String)
    for (regex, pattern_name, description) in ERROR_PATTERNS
        m = match(regex, error_output)
        if !isnothing(m)
            # Replace capture groups in description
            desc = description
            for i in 1:length(m.captures)
                desc = replace(desc, "\$$i" => something(m.captures[i], ""))
            end
            return (pattern_name, desc, m.captures)
        end
    end
    return ("unknown", "Unknown error", String[])
end

"""
Extract file path from error output
"""
function extract_file_path(error_output::String)
    # Try to find file:line:column pattern
    m = match(r"^([^:]+\.(cpp|cc|c|h|hpp)):(\d+):(\d+):", error_output)
    if !isnothing(m)
        return m.captures[1]
    end

    # Try to find just file path
    m = match(r"^([^:]+\.(cpp|cc|c|h|hpp)):", error_output)
    if !isnothing(m)
        return m.captures[1]
    end

    return ""
end

# ============================================================================
# ERROR RECORDING
# ============================================================================

"""
Record a compilation error in the database
"""
function record_error(db::SQLite.DB, command::String, error_output::String;
                     project_path::String="", file_path::String="")
    (pattern_name, description, captures) = detect_error_pattern(error_output)

    # Extract file path if not provided
    if isempty(file_path)
        file_path = extract_file_path(error_output)
    end

    timestamp = string(now())

    DBInterface.execute(db, """
        INSERT INTO compilation_errors
        (timestamp, command, error_output, error_pattern, project_path, file_path)
        VALUES (?, ?, ?, ?, ?, ?)
    """, [timestamp, command, error_output, pattern_name, project_path, file_path])

    # Get the inserted row ID
    result = DataFrame(DBInterface.execute(db, "SELECT last_insert_rowid() as id"))
    error_id = result.id[1]

    return (error_id, pattern_name, description)
end

"""
Record a fix attempt for an error
"""
function record_fix(db::SQLite.DB, error_id::Int, fix_description::String,
                   fix_action::String, fix_type::String, success::Bool)
    timestamp = string(now())
    success_int = success ? 1 : 0

    DBInterface.execute(db, """
        INSERT INTO error_fixes
        (error_id, timestamp, fix_description, fix_action, fix_type, success)
        VALUES (?, ?, ?, ?, ?, ?)
    """, [error_id, timestamp, fix_description, fix_action, fix_type, success_int])
end

# ============================================================================
# FIX SUGGESTIONS
# ============================================================================

"""
Find similar errors in the database
"""
function find_similar_errors(db::SQLite.DB, error_pattern::String,
                            error_output::String; limit::Int=5)
    # First try exact pattern match
    result = DataFrame(DBInterface.execute(db, """
        SELECT DISTINCT e.id, e.error_output, e.error_pattern, e.command
        FROM compilation_errors e
        WHERE e.error_pattern = ?
        LIMIT ?
    """, [error_pattern, limit]))

    if size(result, 1) > 0
        return result
    end

    # Fallback: fuzzy text search (simple substring match)
    # Extract key error terms
    error_terms = extract_error_keywords(error_output)

    if !isempty(error_terms)
        # Build a LIKE query for each term
        conditions = join(["error_output LIKE ?" for _ in error_terms], " OR ")
        patterns = ["%$term%" for term in error_terms]

        result = DataFrame(DBInterface.execute(db, """
            SELECT DISTINCT e.id, e.error_output, e.error_pattern, e.command
            FROM compilation_errors e
            WHERE $conditions
            LIMIT ?
        """, [patterns..., limit]))

        return result
    end

    return nothing
end

"""
Extract keywords from error message for fuzzy matching
"""
function extract_error_keywords(error_output::String)
    keywords = String[]

    # Extract identifiers in quotes
    for m in eachmatch(r"'([^']+)'", error_output)
        push!(keywords, m.captures[1])
    end

    # Extract file names
    for m in eachmatch(r"(\w+\.(cpp|h|hpp|cc))", error_output)
        push!(keywords, m.captures[1])
    end

    return unique(keywords)
end

"""
Get successful fixes for similar errors
"""
function get_successful_fixes(db::SQLite.DB, error_ids::Vector{Int})
    if isempty(error_ids)
        return nothing
    end

    placeholders = join(["?" for _ in error_ids], ",")

    result = DataFrame(DBInterface.execute(db, """
        SELECT f.error_id, f.fix_description, f.fix_action, f.fix_type,
               COUNT(*) as usage_count,
               SUM(f.success) as success_count
        FROM error_fixes f
        WHERE f.error_id IN ($placeholders) AND f.success = 1
        GROUP BY f.fix_description, f.fix_action, f.fix_type
        ORDER BY success_count DESC, usage_count DESC
    """, error_ids))

    return result
end

"""
Suggest fixes for a compilation error
"""
function suggest_fixes(db::SQLite.DB, error_output::String;
                      project_path::String="", top_n::Int=3)
    # Detect error pattern
    (pattern_name, description, captures) = detect_error_pattern(error_output)

    # Find similar errors
    similar_errors = find_similar_errors(db, pattern_name, error_output)

    if isnothing(similar_errors) || size(similar_errors, 1) == 0
        return generate_default_suggestions(pattern_name, captures)
    end

    # Get error IDs
    error_ids = similar_errors.id

    # Get successful fixes
    fixes = get_successful_fixes(db, error_ids)

    if isnothing(fixes) || size(fixes, 1) == 0
        return generate_default_suggestions(pattern_name, captures)
    end

    # Build suggestions
    suggestions = []
    for i in 1:min(top_n, size(fixes, 1))
        confidence = fixes.success_count[i] / fixes.usage_count[i]
        push!(suggestions, Dict(
            "description" => fixes.fix_description[i],
            "action" => fixes.fix_action[i],
            "type" => fixes.fix_type[i],
            "confidence" => confidence,
            "usage_count" => fixes.usage_count[i]
        ))
    end

    return suggestions
end

"""
Generate default fix suggestions based on error pattern
"""
function generate_default_suggestions(pattern_name::String, captures::Vector)
    suggestions = []

    if pattern_name == "missing_header"
        header = !isempty(captures) ? captures[1] : "header"
        push!(suggestions, Dict(
            "description" => "Add include directory containing $header",
            "action" => "add_include_dir",
            "type" => "config_change",
            "confidence" => 0.7,
            "usage_count" => 0
        ))
        push!(suggestions, Dict(
            "description" => "Install missing package containing $header",
            "action" => "install_package",
            "type" => "system_change",
            "confidence" => 0.5,
            "usage_count" => 0
        ))
    elseif pattern_name == "undefined_symbol"
        symbol = !isempty(captures) ? captures[1] : "symbol"
        push!(suggestions, Dict(
            "description" => "Add library containing $symbol",
            "action" => "add_library",
            "type" => "config_change",
            "confidence" => 0.6,
            "usage_count" => 0
        ))
    elseif pattern_name == "missing_library"
        lib = !isempty(captures) ? captures[1] : "library"
        push!(suggestions, Dict(
            "description" => "Install lib$lib or add library search path",
            "action" => "add_lib_path",
            "type" => "config_change",
            "confidence" => 0.7,
            "usage_count" => 0
        ))
    else
        push!(suggestions, Dict(
            "description" => "Check compiler flags and source code",
            "action" => "manual_review",
            "type" => "manual",
            "confidence" => 0.3,
            "usage_count" => 0
        ))
    end

    return suggestions
end

# ============================================================================
# STATISTICS AND REPORTING
# ============================================================================

"""
Get error statistics from database
"""
function get_error_stats(db::SQLite.DB)
    total_errors = DataFrame(DBInterface.execute(db, "SELECT COUNT(*) as count FROM compilation_errors")).count[1]
    total_fixes = DataFrame(DBInterface.execute(db, "SELECT COUNT(*) as count FROM error_fixes")).count[1]
    successful_fixes = DataFrame(DBInterface.execute(db, "SELECT COUNT(*) as count FROM error_fixes WHERE success = 1")).count[1]

    # Get most common error patterns
    patterns = DataFrame(DBInterface.execute(db, """
        SELECT error_pattern, COUNT(*) as count
        FROM compilation_errors
        GROUP BY error_pattern
        ORDER BY count DESC
        LIMIT 10
    """))

    return Dict(
        "total_errors" => total_errors,
        "total_fixes" => total_fixes,
        "successful_fixes" => successful_fixes,
        "success_rate" => total_fixes > 0 ? successful_fixes / total_fixes : 0.0,
        "common_patterns" => patterns
    )
end

"""
Export errors to readable format for Obsidian
"""
function export_to_markdown(db::SQLite.DB, output_path::String="error_log.md")
    stats = get_error_stats(db)

    content = """
    # RepliBuild Compilation Error Log

    Generated: $(now())

    ## Statistics

    - Total Errors: $(stats["total_errors"])
    - Total Fix Attempts: $(stats["total_fixes"])
    - Successful Fixes: $(stats["successful_fixes"])
    - Success Rate: $(round(stats["success_rate"] * 100, digits=2))%

    ## Common Error Patterns

    """

    for row in eachrow(stats["common_patterns"])
        content *= "- **$(row.error_pattern)**: $(row.count) occurrences\n"
    end

    content *= "\n## Recent Errors\n\n"

    # Get recent errors with fixes
    recent = DataFrame(DBInterface.execute(db, """
        SELECT e.timestamp, e.error_pattern, e.command, e.error_output,
               GROUP_CONCAT(f.fix_description, '; ') as fixes,
               GROUP_CONCAT(f.success, ',') as fix_results
        FROM compilation_errors e
        LEFT JOIN error_fixes f ON e.id = f.error_id
        GROUP BY e.id
        ORDER BY e.timestamp DESC
        LIMIT 20
    """))

    for row in eachrow(recent)
        content *= """
        ### $(row.timestamp)

        **Pattern**: `$(row.error_pattern)`

        **Command**: `$(row.command)`

        **Error**:
        ```
        $(row.error_output)
        ```

        """

        if !ismissing(row.fixes)
            content *= "**Fixes Attempted**: $(row.fixes)\n"
            content *= "**Results**: $(row.fix_results)\n"
        end

        content *= "\n---\n\n"
    end

    open(output_path, "w") do f
        write(f, content)
    end

    println("✅ Exported error log to: $output_path")
end

# ============================================================================
# EXPORTS
# ============================================================================

export
    # Database
    init_db,

    # Error recording
    record_error,
    record_fix,
    detect_error_pattern,

    # Fix suggestions
    suggest_fixes,
    find_similar_errors,
    get_successful_fixes,

    # Statistics
    get_error_stats,
    export_to_markdown

end # module ErrorLearning
