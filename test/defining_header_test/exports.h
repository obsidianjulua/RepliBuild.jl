#pragma once
// An export header (clipper2's clipper.export.h): it declares no class and
// defines the library's C entry points. A recipe lists it in `[wrap]
// shim_headers`, after the header it needs, which puts both first in every
// unit RepliBuild generates. Like clipper2's, it does not compile by itself:
// it uses `Widget` and leaves including widget.h to whoever includes it.
extern "C" int export_version() { return 7; }
extern "C" int export_widget(const Widget *w) { return w->v; }
