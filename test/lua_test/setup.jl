#!/usr/bin/env julia
# Downloads and extracts Lua 5.4.7 source for RepliBuild wrapping

using Downloads

const LUA_VERSION = "5.4.7"
const LUA_URL = "https://www.lua.org/ftp/lua-$(LUA_VERSION).tar.gz"
const TEST_DIR = @__DIR__
const SRC_DIR = joinpath(TEST_DIR, "src")
const INCLUDE_DIR = joinpath(TEST_DIR, "include")

# Core VM + auxiliary + stdlib source files (exclude lua.c and luac.c)
const LUA_SOURCES = [
    "lapi.c", "lcode.c", "lctype.c", "ldebug.c", "ldo.c", "ldump.c",
    "lfunc.c", "lgc.c", "llex.c", "lmem.c", "lobject.c", "lopcodes.c",
    "lparser.c", "lstate.c", "lstring.c", "ltable.c", "ltm.c",
    "lundump.c", "lvm.c", "lzio.c",
    "lauxlib.c", "linit.c",
    "lbaselib.c", "lcorolib.c", "ldblib.c", "liolib.c", "lmathlib.c",
    "loadlib.c", "loslib.c", "lstrlib.c", "ltablib.c", "lutf8lib.c",
]

function setup()
    # Skip if source already present
    if isfile(joinpath(SRC_DIR, "lapi.c"))
        println("Lua source already present, skipping download.")
        return
    end

    tarball = joinpath(tempdir(), "lua-$(LUA_VERSION).tar.gz")
    extract_dir = joinpath(tempdir(), "lua-$(LUA_VERSION)")

    # Download
    println("Downloading Lua $(LUA_VERSION)...")
    Downloads.download(LUA_URL, tarball)
    println("Downloaded to: $tarball")

    # Extract
    println("Extracting...")
    run(`tar -xzf $tarball -C $(tempdir())`)

    lua_src = joinpath(extract_dir, "src")

    # Copy source files
    println("Copying source files to $SRC_DIR ...")
    mkpath(SRC_DIR)
    for f in LUA_SOURCES
        src = joinpath(lua_src, f)
        if isfile(src)
            cp(src, joinpath(SRC_DIR, f), force=true)
        else
            @warn "Source file not found: $f"
        end
    end

    # Copy all header files
    println("Copying header files to $INCLUDE_DIR ...")
    mkpath(INCLUDE_DIR)
    for f in readdir(lua_src)
        if endswith(f, ".h")
            cp(joinpath(lua_src, f), joinpath(INCLUDE_DIR, f), force=true)
        end
    end

    # Cleanup
    rm(tarball, force=true)
    rm(extract_dir, recursive=true, force=true)

    println("Lua $(LUA_VERSION) source ready.")
    println("  Sources:  $(length(readdir(SRC_DIR))) files in src/")
    println("  Headers:  $(length(readdir(INCLUDE_DIR))) files in include/")
end

setup()
