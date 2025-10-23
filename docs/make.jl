using Documenter
using RepliBuild

makedocs(
    sitename = "RepliBuild.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://obsidianjulua.github.io/RepliBuild.jl",
        assets = String[],
    ),
    modules = [RepliBuild],
    checkdocs = :none,  # Don't error on missing internal function docs
    pages = [
        "Home" => "index.md",
        "Getting Started" => [
            "Installation" => "getting-started/installation.md",
            "Quick Start" => "getting-started/quickstart.md",
            "Project Structure" => "getting-started/project-structure.md",
        ],
        "User Guide" => [
            "C++ to Julia Workflow" => "guide/cpp-workflow.md",
            "Binary Wrapping" => "guide/binary-wrapping.md",
            "Build System Integration" => "guide/build-systems.md",
            "Configuration Files" => "guide/configuration.md",
            "Module System" => "guide/modules.md",
            "CMake Projects" => "guide/cmake-import.md",
        ],
        "Examples" => [
            "Simple C++ Library" => "examples/simple-cpp.md",
            "Qt Application" => "examples/qt-app.md",
            "Wrapping Existing Binaries" => "examples/binary-wrap.md",
            "Multi-Module Project" => "examples/multi-module.md",
        ],
        "API Reference" => [
            "Core Functions" => "api/core.md",
            "Build System" => "api/build-system.md",
            "Module Registry" => "api/modules.md",
            "Advanced" => "api/advanced.md",
        ],
        "Advanced Topics" => [
            "Daemon System" => "advanced/daemons.md",
            "Error Learning" => "advanced/error-learning.md",
            "LLVM Toolchain" => "advanced/llvm-toolchain.md",
        ],
    ]
)

deploydocs(
    repo = "github.com/obsidianjulua/RepliBuild.jl.git",
    devbranch = "main",
)
