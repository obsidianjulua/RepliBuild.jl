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
        "Set up" => "install.md",
        "Write the recipe" => [
            "Wrap your first library" => "guide.md",
            "What goes in the TOML" => "toml.md",
            "Every key" => "config.md",
            "Where the source comes from" => "dependencies.md",
            "Several libraries" => "libraries.md",
            "CMake projects" => "graph.md",
            "Rust crates" => "rust.md",
            "Header-only libraries" => "header-only.md",
        ],
        "Use the wrapper" => [
            "Call the wrapper" => "calling.md",
            "Load it anywhere" => "use.md",
            "Build a package on it" => "using-wrappers.md",
            "Make it fast" => "performance.md",
            "Debug it" => "debugging.md",
        ],
        "Troubleshooting" => "troubleshooting.md",
        "Support and versions" => "support.md",
        "API" => "api.md",
        "Changelog" => "release-notes.md",
        "Internals (contributors)" => [
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
