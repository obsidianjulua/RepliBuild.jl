#!/usr/bin/env julia
# test/duktape_test/setup.jl — Download the Duktape amalgamation
#
# Duktape is a lightweight embeddable JavaScript engine (pure C).
# We use the amalgamated (single-file) distribution for simplicity.

const DUKTAPE_VERSION = "2.7.0"
const DUKTAPE_URL = "https://duktape.org/duktape-$(DUKTAPE_VERSION).tar.xz"

const SCRIPT_DIR = @__DIR__
const SRC_DIR = joinpath(SCRIPT_DIR, "src")
const MARKER = joinpath(SRC_DIR, ".duktape_$(DUKTAPE_VERSION)")

function setup_duktape()
    if isfile(MARKER)
        println("  duktape $(DUKTAPE_VERSION) already present")
        return true
    end

    println("  downloading duktape $(DUKTAPE_VERSION)...")
    mkpath(SRC_DIR)

    tarball = joinpath(tempdir(), "duktape-$(DUKTAPE_VERSION).tar.xz")
    try
        run(`curl -fsSL -o $tarball $DUKTAPE_URL`)
    catch e
        @warn "Failed to download Duktape: $e"
        return false
    end

    # Extract the amalgamation files we need
    extract_dir = joinpath(tempdir(), "duktape-extract")
    mkpath(extract_dir)
    run(`tar -xf $tarball -C $extract_dir`)

    # The archive extracts to duktape-<version>/
    src_root = joinpath(extract_dir, "duktape-$(DUKTAPE_VERSION)", "src")
    if !isdir(src_root)
        @warn "Unexpected archive layout; looking for src/ directory"
        return false
    end

    # Copy amalgamation files
    for f in ["duktape.c", "duktape.h", "duk_config.h"]
        src = joinpath(src_root, f)
        dst = joinpath(SRC_DIR, f)
        if isfile(src)
            cp(src, dst; force=true)
        else
            @warn "Missing expected file: $f"
            return false
        end
    end

    # Write marker
    write(MARKER, "$(DUKTAPE_VERSION)\n")

    # Clean up
    rm(tarball; force=true)
    rm(extract_dir; recursive=true, force=true)

    println("  duktape $(DUKTAPE_VERSION) ready")
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    setup_duktape() || error("Duktape setup failed")
end
