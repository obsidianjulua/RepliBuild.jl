#!/usr/bin/env julia
# Downloads SQLite 3.49.1 amalgamation source for RepliBuild wrapping

using Downloads

# SQLite version 3.49.1 → amalgamation ID 3490100
const SQLITE_VERSION = "3.49.1"
const SQLITE_YEAR = "2025"
const SQLITE_ID = "3490100"
const SQLITE_URL = "https://www.sqlite.org/$(SQLITE_YEAR)/sqlite-amalgamation-$(SQLITE_ID).zip"
const TEST_DIR = @__DIR__
const SRC_DIR = joinpath(TEST_DIR, "src")
const INCLUDE_DIR = joinpath(TEST_DIR, "include")

function setup()
    # Skip if source already present
    if isfile(joinpath(SRC_DIR, "sqlite3.c"))
        println("SQLite source already present, skipping download.")
        return
    end

    zipfile = joinpath(tempdir(), "sqlite-amalgamation-$(SQLITE_ID).zip")
    extract_dir = joinpath(tempdir(), "sqlite-amalgamation-$(SQLITE_ID)")

    # Download
    println("Downloading SQLite $(SQLITE_VERSION)...")
    Downloads.download(SQLITE_URL, zipfile)
    println("Downloaded to: $zipfile")

    # Extract
    println("Extracting...")
    run(`unzip -o -q $zipfile -d $(tempdir())`)

    # Copy source
    println("Copying source files to $SRC_DIR ...")
    mkpath(SRC_DIR)
    for f in ["sqlite3.c"]
        src = joinpath(extract_dir, f)
        if isfile(src)
            cp(src, joinpath(SRC_DIR, f), force=true)
        else
            @warn "Source file not found: $f"
        end
    end

    # Copy headers
    println("Copying header files to $INCLUDE_DIR ...")
    mkpath(INCLUDE_DIR)
    for f in ["sqlite3.h", "sqlite3ext.h"]
        src = joinpath(extract_dir, f)
        if isfile(src)
            cp(src, joinpath(INCLUDE_DIR, f), force=true)
        else
            @warn "Header file not found: $f"
        end
    end

    # Cleanup
    rm(zipfile, force=true)
    rm(extract_dir, recursive=true, force=true)

    println("SQLite $(SQLITE_VERSION) source ready.")
    println("  Sources:  $(length(readdir(SRC_DIR))) files in src/")
    println("  Headers:  $(length(readdir(INCLUDE_DIR))) files in include/")
end

setup()
