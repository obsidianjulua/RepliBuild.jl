#pragma once
// An export header (clipper2's clipper.export.h): it declares no class and
// defines the library's C entry points. A recipe lists it in `[wrap]
// shim_headers`, which puts it first in every unit RepliBuild generates.
extern "C" int export_version() { return 7; }
