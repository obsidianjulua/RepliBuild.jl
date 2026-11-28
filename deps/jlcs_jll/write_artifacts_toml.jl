# Write the `JLCS` entries of the package's Artifacts.toml from the tarballs
# build_tarballs.jl left in ./products.
#
#     julia deps/jlcs_jll/write_artifacts_toml.jl <download URL prefix>
#
# The prefix is where the tarballs will be served from, without the file name:
# for a GitHub release, https://github.com/<owner>/<repo>/releases/download/<tag>.
# The entries are lazy, so nothing is downloaded at `Pkg.add`; RepliBuild
# fetches the library the first time Tier 2 needs it (src/Builder/JLCSLibrary.jl).
#
# Each entry is the tarball's own identity: the git tree hash of its contents
# and the sha256 of the file. Upload exactly the files hashed here.
#
# Side effect, on purpose: each tarball is unpacked into this depot's artifact
# store (that is how its tree hash is computed), so on the machine that built
# them the artifact is installed and `REPLIBUILD_JLCS=artifact` can be tested
# before anything is uploaded.

using Pkg, Pkg.Artifacts, SHA
using Base.BinaryPlatforms

length(ARGS) == 1 || error("usage: write_artifacts_toml.jl <download URL prefix>")
prefix = rstrip(ARGS[1], '/')

products = joinpath(@__DIR__, "products")
toml = normpath(joinpath(@__DIR__, "..", "..", "Artifacts.toml"))

# Only the dialect's current version (src/mlir/JLCS_VERSION): ./products keeps
# what earlier versions built.
version = strip(read(normpath(joinpath(@__DIR__, "..", "..", "src", "mlir", "JLCS_VERSION")), String))
tarballs = filter(f -> startswith(f, "JLCS.v$version.") && endswith(f, ".tar.gz"), readdir(products))
isempty(tarballs) && error("no JLCS.v$version.*.tar.gz in $products; run build_tarballs.jl first")

for file in sort(tarballs)
    # JLCS.v<major>.<minor>.<patch>.<platform triplet>.tar.gz
    m = match(r"^JLCS\.v\d+\.\d+\.\d+\.(.+)\.tar\.gz$", file)
    m === nothing && error("cannot read a platform out of $file")
    platform = parse(Platform, m[1])
    path = joinpath(products, file)
    tree = create_artifact(dir -> Pkg.PlatformEngines.unpack(path, dir))
    sha = bytes2hex(open(sha256, path))
    bind_artifact!(toml, "JLCS", tree; platform, lazy = true, force = true,
                   download_info = [("$prefix/$file", sha)])
    println(rpad(triplet(platform), 28), " tree ", bytes2hex(tree.bytes), "  sha256 ", sha)
end
println("wrote ", toml)
