#!/usr/bin/env julia
# Downloads pugixml v1.15 source for RepliBuild wrapping

using Downloads

const PUGI_VERSION = "1.15"
const PUGI_URL = "https://github.com/zeux/pugixml/releases/download/v$(PUGI_VERSION)/pugixml-$(PUGI_VERSION).tar.gz"
const TEST_DIR  = @__DIR__
const SRC_DIR   = joinpath(TEST_DIR, "src")
const INC_DIR   = joinpath(TEST_DIR, "include")

function setup()
    if isfile(joinpath(SRC_DIR, "pugixml.cpp"))
        println("pugixml source already present, skipping download.")
        return
    end

    tarball = joinpath(tempdir(), "pugixml-$(PUGI_VERSION).tar.gz")
    println("Downloading pugixml $(PUGI_VERSION)...")
    Downloads.download(PUGI_URL, tarball)

    extract_dir = joinpath(tempdir(), "pugixml-$(PUGI_VERSION)")
    mkpath(extract_dir)
    run(`tar -xzf $tarball -C $extract_dir --strip-components=1`)

    mkpath(SRC_DIR)
    mkpath(INC_DIR)

    cp(joinpath(extract_dir, "src", "pugixml.cpp"), joinpath(SRC_DIR, "pugixml.cpp"))
    cp(joinpath(extract_dir, "src", "pugixml.hpp"), joinpath(INC_DIR, "pugixml.hpp"))
    cp(joinpath(extract_dir, "src", "pugiconfig.hpp"), joinpath(INC_DIR, "pugiconfig.hpp"))

    rm(tarball; force=true)
    rm(extract_dir; recursive=true, force=true)
    println("pugixml $(PUGI_VERSION) ready.")
end

setup()
