# Error Learning System

RepliBuild's error learning system captures, analyzes, and learns from build errors to provide better diagnostics over time.

## Overview

The error learning system:

- **Captures** all compilation and linking errors
- **Categorizes** errors by type and pattern
- **Stores** errors in SQLite database
- **Suggests** solutions based on past resolutions
- **Exports** errors to markdown for documentation

## How It Works

### Error Capture

Every compilation error is captured:

```
error: 'iostream' file not found
```

Stored as:

```julia
Dict(
    :error_type => "missing_header",
    :file => "src/main.cpp",
    :line => 1,
    :message => "'iostream' file not found",
    :timestamp => now(),
    :resolved => false
)
```

### Pattern Recognition

Common error patterns:

| Pattern | Type | Solution |
|---------|------|----------|
| `file not found` | `missing_header` | Add include directory |
| `undefined reference` | `missing_library` | Add library to link |
| `no member named` | `wrong_namespace` | Check namespace |
| `expected ';'` | `syntax_error` | Fix syntax |

### Solution Suggestions

When an error matches a known pattern:

```
❌ Error: 'opencv2/opencv.hpp' file not found

💡 Suggestion: This looks like a missing OpenCV header.
   Try adding OpenCV to your dependencies:

   [dependencies]
   modules = ["OpenCV"]

   Or install OpenCV:
   Ubuntu: sudo apt-get install libopencv-dev
   macOS: brew install opencv
```

## Using the Error Learning System

### Automatic Error Logging

Errors are logged automatically:

```julia
using RepliBuild

try
    RepliBuild.compile()
catch e
    # Error is automatically logged to database
    rethrow(e)
end
```

### Export Error Log

```julia
# Export errors to markdown
RepliBuild.export_errors("error_log.md")
```

Generated `error_log.md`:

```markdown
# RepliBuild Error Log

## Error Statistics

- Total errors: 42
- Unique errors: 15
- Resolved: 12 (80%)
- Unresolved: 3 (20%)

## Recent Errors

### Error: Missing Header 'boost/filesystem.hpp'
**Type:** missing_header
**File:** src/utils.cpp:5
**Date:** 2024-01-15 14:23:45
**Status:** ✅ Resolved

**Solution:**
Added Boost module to dependencies.

### Error: undefined reference to `pthread_create'
**Type:** missing_library
**File:** (linker)
**Date:** 2024-01-15 15:10:22
**Status:** ✅ Resolved

**Solution:**
Added pthread to link_libs.

### Error: expected ';' after expression
**Type:** syntax_error
**File:** src/calculator.cpp:42
**Date:** 2024-01-15 16:05:11
**Status:** ✅ Resolved

**Solution:**
Fixed missing semicolon.
```

### Query Error Database

```julia
# Get error statistics
stats = RepliBuild.get_error_stats()

println("Total errors: ", stats[:total])
println("Resolved: ", stats[:resolved])
println("Unresolved: ", stats[:unresolved])

# Get errors by type
missing_headers = RepliBuild.query_errors(type="missing_header")
for error in missing_headers
    println("- $(error.message)")
end
```

## Error Categories

### Missing Header

**Pattern:** `'header.h' file not found`

**Common causes:**
- Missing include directory
- Missing dependency
- Typo in filename

**Solutions:**
```toml
[compilation]
include_dirs = ["/path/to/includes"]

[dependencies]
modules = ["LibraryName"]
```

### Missing Library

**Pattern:** `undefined reference to 'function'`

**Common causes:**
- Missing link library
- Wrong library name
- Library not installed

**Solutions:**
```toml
[compilation]
link_libs = ["pthread", "m"]

[dependencies]
modules = ["Library"]
```

### Wrong Namespace

**Pattern:** `no member named 'X' in namespace 'Y'`

**Common causes:**
- Wrong namespace
- Missing using directive
- Wrong header included

**Solutions:**
```cpp
using namespace CorrectNamespace;
// or
CorrectNamespace::function();
```

### ABI Mismatch

**Pattern:** `undefined symbol: _ZN...`

**Common causes:**
- C++ standard mismatch
- Different compiler
- Wrong library version

**Solutions:**
```toml
[compilation]
cxx_standard = "c++17"  # Match library
```

### Template Errors

**Pattern:** `candidate template ignored`

**Common causes:**
- Wrong template arguments
- Missing template specialization
- Template deduction failed

**Solutions:**
```cpp
// Explicit template arguments
function<int>(value);
```

## Database Schema

Error database location: `~/.replibuild/error_db.sqlite`

### Tables

**errors:**
```sql
CREATE TABLE errors (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    error_type TEXT,
    file TEXT,
    line INTEGER,
    message TEXT,
    full_output TEXT,
    project TEXT,
    resolved BOOLEAN,
    resolution TEXT,
    resolution_time DATETIME
);
```

**error_patterns:**
```sql
CREATE TABLE error_patterns (
    id INTEGER PRIMARY KEY,
    pattern TEXT UNIQUE,
    error_type TEXT,
    suggested_solution TEXT,
    occurrence_count INTEGER,
    success_rate REAL
);
```

### Direct Database Access

```julia
using SQLite

db = SQLite.DB(joinpath(RepliBuild.get_replibuild_dir(), "error_db.sqlite"))

# Query recent errors
recent = SQLite.query(db, """
    SELECT * FROM errors
    WHERE timestamp > datetime('now', '-7 days')
    ORDER BY timestamp DESC
    LIMIT 10
""")

for row in recent
    println("$(row.timestamp): $(row.message)")
end
```

## Machine Learning Features

### Pattern Learning

The system learns new patterns:

```julia
# After resolving an error manually
error_id = 42
resolution = "Added OpenCV module to dependencies"

RepliBuild.mark_error_resolved(error_id, resolution)

# System learns: errors matching this pattern can be resolved this way
```

### Solution Ranking

Solutions are ranked by success rate:

```julia
# Query suggested solutions
suggestions = RepliBuild.get_error_suggestions(error_message)

for (i, suggestion) in enumerate(suggestions)
    println("$i. $(suggestion.solution) (success rate: $(suggestion.success_rate)%)")
end
```

### Automated Fixes

For high-confidence patterns:

```julia
# Attempt automatic fix
if RepliBuild.can_auto_fix(error)
    println("Attempting automatic fix...")
    RepliBuild.apply_fix(error)
    RepliBuild.compile()  # Retry
end
```

## Configuration

Configure error learning in `~/.replibuild/config.toml`:

```toml
[error_learning]
# Enable error learning
enabled = true

# Automatically export errors
auto_export = false
export_path = "~/replibuild_errors.md"

# Auto-fix confidence threshold (0-1)
auto_fix_threshold = 0.9

# Keep errors for N days
retention_days = 90

# Max database size (MB)
max_db_size = 100
```

## Best Practices

### 1. Review Error Log Periodically

```julia
# Weekly review
RepliBuild.export_errors("weekly_errors.md")
# Analyze patterns and update documentation
```

### 2. Mark Resolutions

```julia
# When you fix an error, mark it
error_id = get_last_error_id()
RepliBuild.mark_error_resolved(error_id, "Solution description")
```

### 3. Share Error Patterns

```julia
# Export patterns for team
RepliBuild.export_error_patterns("team_patterns.json")

# Import on other machine
RepliBuild.import_error_patterns("team_patterns.json")
```

### 4. Clean Old Errors

```julia
# Remove errors older than 90 days
RepliBuild.cleanup_old_errors(days=90)
```

## Integration with Issue Tracking

### Export to GitHub Issues

```julia
function create_github_issue(error)
    title = "Build Error: $(error.error_type)"
    body = """
    ## Error Details

    **File:** $(error.file):$(error.line)
    **Message:** $(error.message)

    **Full Output:**
    ```
    $(error.full_output)
    ```

    **Timestamp:** $(error.timestamp)
    """

    # Use GitHub API to create issue
    # ... GitHub API call ...
end

# Export unresolved errors to issues
unresolved = RepliBuild.query_errors(resolved=false)
for error in unresolved
    create_github_issue(error)
end
```

### Obsidian Integration

```julia
# Export for Obsidian
RepliBuild.export_errors("~/Obsidian/RepliBuild/errors.md")
```

Generates Obsidian-compatible markdown with tags:

```markdown
# Error: Missing Boost Header

#replibuild #error #boost #missing-header

**Status:** ✅ Resolved
**Date:** 2024-01-15

## Details
...

## Resolution
Added Boost to dependencies.

## Related
[[Boost Installation Guide]]
[[Common Build Errors]]
```

## Troubleshooting

### Database Locked

**Error:** `database is locked`

**Solution:**
```julia
# Close all connections
# Restart Julia
```

### Database Corrupted

**Solution:**
```julia
# Backup and reset
db_path = joinpath(RepliBuild.get_replibuild_dir(), "error_db.sqlite")
cp(db_path, "error_db_backup.sqlite")
rm(db_path)

# Database recreated on next error
```

### Too Many Errors

**Solution:**
```julia
# Clear old errors
RepliBuild.cleanup_old_errors(days=30)

# Or reset database
RepliBuild.reset_error_database()
```

## Next Steps

- **[Daemons](daemons.md)**: Daemon system for faster builds
- **[LLVM Toolchain](llvm-toolchain.md)**: Toolchain management
- **[API Reference](../api/advanced.md)**: Advanced API
