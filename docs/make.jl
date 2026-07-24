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
        canonical="https://obsidianjulua.github.io/RepliBuild.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Guide" => [
            "Workflow" => "guide.md",
            "Configuration (replibuild.toml)" => "config.md",
            "Using a Wrapper" => "using-wrappers.md",
        ],
        "API Reference" => "api.md",
        "Advanced" => [
            "The Inheritance ABI" => "inheritance-abi.md",
            "Internals & Dispatch" => "internals.md",
        ],
        "Release Notes" => "release-notes.md",
    ],
    warnonly=true,
)

deploydocs(;
    repo="github.com/obsidianjulua/RepliBuild.jl",
    devbranch="main",
)
