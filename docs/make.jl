using Documenter
push!(LOAD_PATH,"../src/")
using RepliBuild

makedocs(;
    modules=[RepliBuild],
    authors="John <archjulialua@gmail.com>",
    repo="https://github.com/obsidianjulua/RepliBuild.jl/blob/{commit}{path}#{line}",
    sitename="RepliBuild.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://obsidianjulua.github.io/RepliBuild.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Architecture" => "architecture.md",
        "User Guide" => [
            "Workflow" => "guide.md",
            "Configuration" => "config.md",
        ],
        "API Reference" => "api.md",
        "Advanced" => [
            "MLIR / JLCS Dialect" => "mlir.md",
            "Introspection Tools" => "introspect.md",
            "Benchmarks" => "benchmarks.md",
        ],
        "Internals" => "internals.md",
    ],
    warnonly=true,
)

deploydocs(;
    repo="github.com/obsidianjulua/RepliBuild.jl",
    devbranch="main",
)
