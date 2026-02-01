using Documenter
push!(LOAD_PATH,"../src/")
using RepliBuild

makedocs(;
    modules=[RepliBuild],
    authors="grim <archjulialua@gmail.com>",
    repo="https://github.com/grim/RepliBuild.jl/blob/{commit}{path}#{line}",
    sitename="RepliBuild.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://grim.github.io/RepliBuild.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "User Guide" => [
            "Workflow" => "guide.md",
            "Configuration" => "config.md",
        ],
        "API Reference" => "api.md",
        "Advanced" => [
            "Introspection Tools" => "introspect.md",
            "MLIR / JLCS" => "mlir.md",
        ],
        "Internals" => "internals.md",
    ],
    warnonly=true,
)

deploydocs(;
    repo="github.com/grim/RepliBuild.jl",
    devbranch="main",
)
