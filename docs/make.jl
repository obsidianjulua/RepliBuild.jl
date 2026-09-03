using Documenter
push!(LOAD_PATH,"../src/")
using RepliBuild

makedocs(;
    modules=[RepliBuild],
    authors="John <archjulialua@gmail.com>",
    repo=Documenter.Remotes.GitHub("obsidianjulua", "RepliBuild.jl"),
    sitename="RepliBuild.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://obsidianjulua.github.io/RepliBuild.jl/stable",
        edit_link="main",
        assets=String[],
        collapselevel=1,
    ),
    pages=[
        "Home" => "index.md",
        "Install" => "install.md",
        "Wrap a library" => "guide.md",
        "Edit the TOML" => "config.md",
        "Call a wrapper" => "calling.md",
        "Ship a package" => "using-wrappers.md",
        "Registry" => "use.md",
        "API" => "api.md",
        "Troubleshooting" => "troubleshooting.md",
        "Changelog" => "release-notes.md",
        "Developer" => [
            "Overview" => "developer.md",
            "JLCS / MLIR" => "mlir.md",
            "The inheritance ABI" => "inheritance-abi.md",
            "Internals" => "internals.md",
            "Tier 1 (experimental)" => "tier1.md",
        ],
    ],
    warnonly=true,
)

deploydocs(;
    repo="github.com/obsidianjulua/RepliBuild.jl",
    devbranch="main",
    versions=["stable" => "v^", "v#.#", "dev" => "dev"],
)
