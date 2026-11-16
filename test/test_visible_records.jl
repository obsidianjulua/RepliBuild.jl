# =============================================================================
# Records made visible: `[types] records` (2026-11-15)
#
# A header-only class that no source of the library mentions is in no
# translation unit's debug info, so the wrapper has no type for it and the
# inline-constructor pass no declaration to force (Bullet's
# btDefaultMotionState; the Hub compiled a one-line file to make it appear).
# `[types] records` names such classes and a generated unit uses each once
# (Compiler.generate_visible_records).
#
# Toolchain-free: the key's way through the config (load, save, the default,
# forced re-discovery), the unit's text, the bare name a spelling is looked up
# by, and which entries a metadata does not define.
# test_visible_records_live.jl builds test/visible_record_test.
# =============================================================================

using Test
using RepliBuild
using TOML

const _VR = RepliBuild.Compiler
const _VRC = RepliBuild.ConfigurationManager

@testset "[types] records" begin

    @testset "the key loads, saves, defaults to empty and survives discover(force=true)" begin
        mktempdir() do dir
            toml = joinpath(dir, "replibuild.toml")
            write(toml, """
            [project]
            name = "vr"

            [wrap]
            language = "cpp"

            [types]
            records = ["btDefaultMotionState", "ns::Outer::Inner"]
            """)
            cfg = _VRC.load_config(toml)
            @test cfg.types.records == ["btDefaultMotionState", "ns::Outer::Inner"]
            # `save_config` writes the file the config was loaded from.
            _VRC.save_config(cfg)
            @test TOML.parsefile(toml)["types"]["records"] == ["btDefaultMotionState", "ns::Outer::Inner"]
            @test _VRC.load_config(toml).types.records == ["btDefaultMotionState", "ns::Outer::Inner"]
            # Absent: empty, and not written back.
            write(toml, "[project]\nname = \"vr\"\n")
            cfg0 = _VRC.load_config(toml)
            @test cfg0.types.records == String[]
            _VRC.save_config(cfg0)
            @test !haskey(get(TOML.parsefile(toml), "types", Dict()), "records")
        end
        @test ("types", "records") in RepliBuild.Discovery.PRESERVED_TOML_KEYS
    end

    @testset "the unit: the headers, one use per record, each on its own line" begin
        text = _VR._visible_records_unit_text(["btBulletDynamicsCommon.h", "<vector>", "\"quoted.h\""],
                                              ["btDefaultMotionState", "ns::Outer::Inner"])
        @test occursin("#include \"btBulletDynamicsCommon.h\"\n#include <vector>\n#include \"quoted.h\"\n", text)
        # `sizeof` asks for the complete type: a declared-only name is an error.
        @test occursin("namespace replibuild_force { template <class T> void record(T*) { (void)sizeof(T); } }\n", text)
        @test occursin("\ntemplate void replibuild_force::record<btDefaultMotionState>(btDefaultMotionState*);\n", text)
        @test occursin("\ntemplate void replibuild_force::record<ns::Outer::Inner>(ns::Outer::Inner*);\n", text)
        @test count("template void replibuild_force::record<", text) == 2
        # Never a `template class T;`: that is invalid C++ for a class that is
        # not a template, which is why `[types] templates` could not do this.
        @test !occursin("template class", text)
        # A generated force unit like the other two: same flags, same fingerprint.
        @test _VR._is_visible_records_tu("/x/.replibuild_cache/replibuild_records.cpp")
        @test !_VR._is_visible_records_tu("/x/src/records.cpp")
        @test _VR._is_generated_force_tu("/x/.replibuild_cache/replibuild_records.cpp")
        @test _VR._file_compile_fingerprint("abc", "/x/replibuild_records.cpp") ==
              _VR._file_compile_fingerprint("abc", "/x/replibuild_templates.cpp")
    end

    @testset "a spelling is looked up by the bare name DWARF keys records by" begin
        bare = _VR._record_bare_name
        @test bare("btDefaultMotionState") == "btDefaultMotionState"
        @test bare("vr::Unused") == "Unused"
        @test bare("ns::Outer::Inner") == "Inner"
        @test bare("  ns::C  ") == "C"
        # Template arguments keep their own `::`.
        @test bare("ns::Box<std::pair<int, ns::V>>") == "Box<std::pair<int, ns::V>>"
        @test bare("ns::Box<ns::V>::Node") == "Node"
        sd = Dict{String,Any}("Unused" => Dict{String,Any}(), "Inner" => Dict{String,Any}())
        @test _VR._invisible_records(["vr::Unused", "vr::Outer::Inner"], sd) == String[]
        @test _VR._invisible_records(["vr::Unused", "vr::Gone", "Other"], sd) == ["vr::Gone", "Other"]
        @test _VR._invisible_records(String[], sd) == String[]
    end

    @testset "nothing listed, or a C project: no unit" begin
        mktempdir() do dir
            toml = joinpath(dir, "replibuild.toml")
            write(toml, "[project]\nname = \"vr\"\nroot = \"$(escape_string(dir))\"\n\n[wrap]\nlanguage = \"cpp\"\n")
            files = ["/x/a.cpp"]
            @test _VR.generate_visible_records(_VRC.load_config(toml), files) === files
            write(toml, "[project]\nname = \"vr\"\nroot = \"$(escape_string(dir))\"\n\n[wrap]\nlanguage = \"c\"\n\n" *
                        "[types]\nrecords = [\"S\"]\n")
            cfg = _VRC.load_config(toml)
            @test _VR.generate_visible_records(cfg, files) === files
            @test _VR._report_invisible_records(cfg, joinpath(dir, "no_metadata.json")) == String[]
        end
    end
end
