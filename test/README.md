# DWARF Extraction Test

Compare RepliBuild's DWARF extraction against full dwarfdump output to find gaps.

## Quick Start

```bash
julia --project=.. test_dwarf.jl
```

## Output

**Console**: Coverage 71.9%, supported tags ✅, high-priority missing 🎯

**Files**:
- `replibuild_extraction.json` - What RepliBuild extracted (40KB)
- `dwarfdump_full.txt` - Full DWARF dump from llvm-dwarfdump (267KB)
- `readelf_full.txt` - Full DWARF dump from readelf (253KB)
- `tag_comparison.txt` - Tag frequency analysis (1.6KB)
- `FINDINGS.md` - Analysis and recommendations

## Key Findings

**Current Coverage**: 71.9% (867/1206 DWARF tag instances)

**Top Missing Features** (see FINDINGS.md):
1. **typedef** (63 instances) - type aliases lost
2. **imported_declaration** (221 instances) - using declarations
3. **template_type_parameter** - generic types incomplete
4. **inheritance** - class hierarchies missing
5. **namespace** - scoping info lost

**We Extract Well**: Functions (239), structs (9), enums (3), primitives, pointers, arrays
