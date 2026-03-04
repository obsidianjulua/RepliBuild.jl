#!/usr/bin/env julia
# Downloads Duktape 2.7.0 amalgamation source for RepliBuild wrapping

using Downloads

const DUK_VERSION = "2.7.0"
const DUK_URL = "https://duktape.org/duktape-$(DUK_VERSION).tar.xz"
const TEST_DIR = @__DIR__
const SRC_DIR = joinpath(TEST_DIR, "src")
const INCLUDE_DIR = joinpath(TEST_DIR, "include")

function setup()
    # Skip if source already present
    if isfile(joinpath(SRC_DIR, "duktape.c"))
        println("Duktape source already present, skipping download.")
        return
    end

    tarball = joinpath(tempdir(), "duktape-$(DUK_VERSION).tar.xz")
    extract_dir = joinpath(tempdir(), "duktape-$(DUK_VERSION)")

    # Download
    println("Downloading Duktape $(DUK_VERSION)...")
    Downloads.download(DUK_URL, tarball)
    println("Downloaded to: $tarball")

    # Extract
    println("Extracting...")
    run(`tar -xJf $tarball -C $(tempdir())`)

    duk_src = joinpath(extract_dir, "src")

    # Copy amalgamation source
    println("Copying source files to $SRC_DIR ...")
    mkpath(SRC_DIR)
    for f in ["duktape.c"]
        src = joinpath(duk_src, f)
        if isfile(src)
            cp(src, joinpath(SRC_DIR, f), force=true)
        else
            @warn "Source file not found: $f"
        end
    end

    # Copy header files
    println("Copying header files to $INCLUDE_DIR ...")
    mkpath(INCLUDE_DIR)
    for f in ["duktape.h", "duk_config.h"]
        src = joinpath(duk_src, f)
        if isfile(src)
            cp(src, joinpath(INCLUDE_DIR, f), force=true)
        else
            @warn "Header file not found: $f"
        end
    end

    # Cleanup
    rm(tarball, force=true)
    rm(extract_dir, recursive=true, force=true)

    println("Duktape $(DUK_VERSION) source ready.")
    println("  Sources:  $(length(readdir(SRC_DIR))) files in src/")
    println("  Headers:  $(length(readdir(INCLUDE_DIR))) files in include/")
end

setup()
