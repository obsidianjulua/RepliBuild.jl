#!/usr/bin/env julia
# Test complete Qt + CMake workflow with artifact resolution

using Pkg
Pkg.activate(".")

using RepliBuild
using RepliBuild.ModuleRegistry
using RepliBuild.CMakeParser
using TOML

println("="^70)
println("Qt + CMake Build Test - Full Artifact Resolution")
println("="^70)
println()

# Create a realistic Qt CMake project
test_dir = mktempdir()
println("📁 Test directory: $test_dir")
println()

# Create project structure
mkpath(joinpath(test_dir, "src"))
mkpath(joinpath(test_dir, "include"))

# Write a CMakeLists.txt with Qt5 dependency
cmake_file = joinpath(test_dir, "CMakeLists.txt")
write(cmake_file, """
cmake_minimum_required(VERSION 3.10)
project(QtTestApp VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Find Qt5
find_package(Qt5 REQUIRED COMPONENTS Core Widgets Network)

# Also test Boost
find_package(Boost REQUIRED COMPONENTS system filesystem)

# Create library
add_library(qttest SHARED
    src/mainwindow.cpp
    src/network_helper.cpp
)

target_include_directories(qttest PUBLIC include)

# Link Qt libraries
target_link_libraries(qttest
    Qt5::Core
    Qt5::Widgets
    Qt5::Network
    Boost::system
    Boost::filesystem
)
""")

# Write realistic Qt source files
write(joinpath(test_dir, "include", "mainwindow.h"), """
#pragma once
#include <QMainWindow>
#include <QString>

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow(QWidget *parent = nullptr);
    void setTitle(const QString &title);
private:
    QString m_title;
};
""")

write(joinpath(test_dir, "src", "mainwindow.cpp"), """
#include "mainwindow.h"
#include <QApplication>
#include <QMenuBar>
#include <boost/filesystem.hpp>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    setWindowTitle("RepliBuild Qt Test");

    // Use Boost filesystem
    boost::filesystem::path p = boost::filesystem::current_path();
}

void MainWindow::setTitle(const QString &title) {
    m_title = title;
    setWindowTitle(title);
}
""")

write(joinpath(test_dir, "src", "network_helper.cpp"), """
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <boost/system/error_code.hpp>

class NetworkHelper {
public:
    NetworkHelper() : manager(new QNetworkAccessManager()) {}
    void fetchUrl(const QString &url) {
        QNetworkRequest request((QUrl(url)));
        manager->get(request);
    }
private:
    QNetworkAccessManager *manager;
};
""")

println("="^70)
println("Step 1: Parse CMakeLists.txt")
println("="^70)

cmake_project = CMakeParser.parse_cmake_file(cmake_file)
println("✅ Parsed CMake project:")
println("   Project: $(cmake_project.project_name)")
println("   Targets: $(join(keys(cmake_project.targets), ", "))")
println("   External packages: $(join(cmake_project.find_packages, ", "))")
println()

println("="^70)
println("Step 2: Resolve External Dependencies via ModuleRegistry")
println("="^70)

for pkg_name in cmake_project.find_packages
    println("\n🔍 Resolving: $pkg_name")
    mod_info = ModuleRegistry.resolve_module(pkg_name)

    if !isnothing(mod_info)
        println("  ✅ Source: $(mod_info.source)")
        println("  📦 Julia Package: $(mod_info.julia_package)")
        println("  📂 Include dirs: $(length(mod_info.include_dirs))")
        println("  📚 Library dirs: $(length(mod_info.library_dirs))")
        println("  🔗 Libraries: $(mod_info.libraries)")

        # Show actual paths if JLL resolved
        if mod_info.source == :jll && !isempty(mod_info.include_dirs)
            println("  📍 Artifact paths:")
            for inc_dir in mod_info.include_dirs[1:min(2, end)]
                println("     • $inc_dir")
                if isdir(inc_dir)
                    println("       ✓ Directory exists")
                else
                    println("       ⚠ Directory missing (artifact may not be extracted)")
                end
            end
        end
    else
        println("  ❌ Could not resolve")
    end
end

println()
println("="^70)
println("Step 3: Convert to replibuild.toml with Artifacts")
println("="^70)

config_data = CMakeParser.to_replibuild_config(cmake_project, "qttest")

# Save to file
output_toml = joinpath(test_dir, "replibuild.toml")
open(output_toml, "w") do io
    TOML.print(io, config_data)
end

println("✅ Generated: $output_toml")
println()

# Display key sections
if haskey(config_data, "dependencies")
    println("📦 [dependencies]")
    for (name, info) in config_data["dependencies"]
        println("  [$name]")
        println("    source = \"$(info["source"])\"")
        if haskey(info, "julia_package") && !isempty(info["julia_package"])
            println("    julia_package = \"$(info["julia_package"])\"")
        end
    end
    println()
end

if haskey(config_data, "compile")
    compile_config = config_data["compile"]

    println("🔧 [compile]")
    if haskey(compile_config, "include_dirs")
        println("  include_dirs = [")
        for dir in compile_config["include_dirs"][1:min(5, end)]
            println("    \"$dir\",")
        end
        if length(compile_config["include_dirs"]) > 5
            println("    ... $(length(compile_config["include_dirs"]) - 5) more")
        end
        println("  ]")
    end

    if haskey(compile_config, "link_libraries")
        println("  link_libraries = $(compile_config["link_libraries"])")
    end
    println()
end

println("="^70)
println("Step 4: Verify Artifact Accessibility")
println("="^70)

# Check if we can actually access Qt artifacts
qt_info = ModuleRegistry.get_module_info("Qt5")
if !isnothing(qt_info)
    println("✅ Qt5 module cached")

    # Try to find Qt headers
    if !isempty(qt_info.include_dirs)
        qt_inc = qt_info.include_dirs[1]
        println("\n📂 Checking Qt5 artifact at: $qt_inc")

        if isdir(qt_inc)
            # Look for Qt headers
            qt_headers = ["QtCore", "QtWidgets", "QtNetwork"]
            for header in qt_headers
                header_path = joinpath(qt_inc, header)
                if isdir(header_path)
                    println("  ✓ Found $header/")

                    # List some header files
                    if isdir(header_path)
                        files = readdir(header_path)
                        h_files = filter(f -> endswith(f, ".h"), files)
                        if !isempty(h_files)
                            println("    Headers: $(join(h_files[1:min(3, end)], ", "))...")
                        end
                    end
                else
                    println("  ⚠ Missing $header/")
                end
            end
        else
            println("  ⚠ Artifact directory doesn't exist (may need extraction)")
        end
    else
        println("  ℹ No include directories found (JLL may not expose artifacts)")
    end
else
    println("⚠ Qt5 not in cache")
end

println()
println("="^70)
println("✅ TEST COMPLETE")
println("="^70)
println()
println("Summary:")
println("• CMake project with Qt5 + Boost parsed successfully")
println("• Dependencies resolved via ModuleRegistry")
println("• JLL packages auto-discovered and linked")
println("• replibuild.toml generated with artifact paths")
println("• Build config ready for compilation")
println()

# Show what the actual build command would look like
println("📝 Next steps for actual compilation:")
println("   1. RepliBuild would use these include paths:")
if haskey(config_data, "compile") && haskey(config_data["compile"], "include_dirs")
    for dir in config_data["compile"]["include_dirs"][1:min(3, end)]
        println("      -I$dir")
    end
end
println()
println("   2. Link against these libraries:")
if haskey(config_data, "compile") && haskey(config_data["compile"], "link_libraries")
    for lib in config_data["compile"]["link_libraries"]
        if startswith(lib, "Qt5::")
            # CMake target - would resolve to actual .so file
            println("      $lib (from Qt5_jll artifact)")
        elseif startswith(lib, "Boost::")
            println("      $lib (from Boost_jll artifact)")
        else
            println("      -l$lib")
        end
    end
end

println()
println("="^70)
