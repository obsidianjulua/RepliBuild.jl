/* Scale + fan-out fixture for the Slicer (M2). Added 2026-08-19.
 *
 * REPLACES a testset that read the RepliBuild-Hub lua build by absolute path.
 * That coupling made a core-engine test depend on whichever library version was
 * built in another repo: lua 5.5.1 turned luaL_openlibs into
 * `#define luaL_openlibs(L) luaL_openselectedlibs(L, ~0, 0)`, so the plain
 * symbol stopped existing, the slice was refused "function not found in module",
 * and test_slicer.jl went red for a reason with nothing to do with slicing.
 * Hub rebuilds are the integration test; this keeps the MECHANIC testable alone.
 *
 * The property under test is DECLARATIONS-ONLY EXTRACTION: a slice carries the
 * target's own body and nothing else, however much of the library the target
 * reaches. st_sc_hub's transitive closure is this entire translation unit, so a
 * slicer that followed reachability would emit all of it; a correct one emits
 * one `define` plus `declare`s and stays kilobytes. That is what lua_pcallk and
 * luaL_openlibs were standing in for, now stated as a ratio the test can measure
 * instead of a library it has to find.
 *
 * `noinline` throughout is LOAD-BEARING, not stylistic. [link]
 * optimization_level is "2", and without it the leaves fold into the mids and
 * the mids into the hub — the fan-out this file exists to create would be
 * optimized away and the test would pass vacuously against an empty call graph.
 * test_slicer.jl asserts the call edges survived rather than trusting they did.
 *
 * Every leaf returns a DISTINCT constant so identical-code folding cannot merge
 * them into one function and quietly collapse the fan-out the same way.
 *
 * Closed form, for an oracle independent of the other tier:
 *   leaf_i(v) = 3v + (100 + i),  i = 0..63
 *   hub(v)    = sum(leaf_i(v))  = 192v + 8416
 * Tier-1 and Tier-3 agreeing only proves coherence; this proves correctness.
 */

#include "slice_test.h"

#define ST_SC_FN __attribute__((noinline))

ST_SC_FN long st_sc_leaf_00(long v) { return v * 3 + 100; }
ST_SC_FN long st_sc_leaf_01(long v) { return v * 3 + 101; }
ST_SC_FN long st_sc_leaf_02(long v) { return v * 3 + 102; }
ST_SC_FN long st_sc_leaf_03(long v) { return v * 3 + 103; }
ST_SC_FN long st_sc_leaf_04(long v) { return v * 3 + 104; }
ST_SC_FN long st_sc_leaf_05(long v) { return v * 3 + 105; }
ST_SC_FN long st_sc_leaf_06(long v) { return v * 3 + 106; }
ST_SC_FN long st_sc_leaf_07(long v) { return v * 3 + 107; }
ST_SC_FN long st_sc_leaf_08(long v) { return v * 3 + 108; }
ST_SC_FN long st_sc_leaf_09(long v) { return v * 3 + 109; }
ST_SC_FN long st_sc_leaf_10(long v) { return v * 3 + 110; }
ST_SC_FN long st_sc_leaf_11(long v) { return v * 3 + 111; }
ST_SC_FN long st_sc_leaf_12(long v) { return v * 3 + 112; }
ST_SC_FN long st_sc_leaf_13(long v) { return v * 3 + 113; }
ST_SC_FN long st_sc_leaf_14(long v) { return v * 3 + 114; }
ST_SC_FN long st_sc_leaf_15(long v) { return v * 3 + 115; }
ST_SC_FN long st_sc_leaf_16(long v) { return v * 3 + 116; }
ST_SC_FN long st_sc_leaf_17(long v) { return v * 3 + 117; }
ST_SC_FN long st_sc_leaf_18(long v) { return v * 3 + 118; }
ST_SC_FN long st_sc_leaf_19(long v) { return v * 3 + 119; }
ST_SC_FN long st_sc_leaf_20(long v) { return v * 3 + 120; }
ST_SC_FN long st_sc_leaf_21(long v) { return v * 3 + 121; }
ST_SC_FN long st_sc_leaf_22(long v) { return v * 3 + 122; }
ST_SC_FN long st_sc_leaf_23(long v) { return v * 3 + 123; }
ST_SC_FN long st_sc_leaf_24(long v) { return v * 3 + 124; }
ST_SC_FN long st_sc_leaf_25(long v) { return v * 3 + 125; }
ST_SC_FN long st_sc_leaf_26(long v) { return v * 3 + 126; }
ST_SC_FN long st_sc_leaf_27(long v) { return v * 3 + 127; }
ST_SC_FN long st_sc_leaf_28(long v) { return v * 3 + 128; }
ST_SC_FN long st_sc_leaf_29(long v) { return v * 3 + 129; }
ST_SC_FN long st_sc_leaf_30(long v) { return v * 3 + 130; }
ST_SC_FN long st_sc_leaf_31(long v) { return v * 3 + 131; }
ST_SC_FN long st_sc_leaf_32(long v) { return v * 3 + 132; }
ST_SC_FN long st_sc_leaf_33(long v) { return v * 3 + 133; }
ST_SC_FN long st_sc_leaf_34(long v) { return v * 3 + 134; }
ST_SC_FN long st_sc_leaf_35(long v) { return v * 3 + 135; }
ST_SC_FN long st_sc_leaf_36(long v) { return v * 3 + 136; }
ST_SC_FN long st_sc_leaf_37(long v) { return v * 3 + 137; }
ST_SC_FN long st_sc_leaf_38(long v) { return v * 3 + 138; }
ST_SC_FN long st_sc_leaf_39(long v) { return v * 3 + 139; }
ST_SC_FN long st_sc_leaf_40(long v) { return v * 3 + 140; }
ST_SC_FN long st_sc_leaf_41(long v) { return v * 3 + 141; }
ST_SC_FN long st_sc_leaf_42(long v) { return v * 3 + 142; }
ST_SC_FN long st_sc_leaf_43(long v) { return v * 3 + 143; }
ST_SC_FN long st_sc_leaf_44(long v) { return v * 3 + 144; }
ST_SC_FN long st_sc_leaf_45(long v) { return v * 3 + 145; }
ST_SC_FN long st_sc_leaf_46(long v) { return v * 3 + 146; }
ST_SC_FN long st_sc_leaf_47(long v) { return v * 3 + 147; }
ST_SC_FN long st_sc_leaf_48(long v) { return v * 3 + 148; }
ST_SC_FN long st_sc_leaf_49(long v) { return v * 3 + 149; }
ST_SC_FN long st_sc_leaf_50(long v) { return v * 3 + 150; }
ST_SC_FN long st_sc_leaf_51(long v) { return v * 3 + 151; }
ST_SC_FN long st_sc_leaf_52(long v) { return v * 3 + 152; }
ST_SC_FN long st_sc_leaf_53(long v) { return v * 3 + 153; }
ST_SC_FN long st_sc_leaf_54(long v) { return v * 3 + 154; }
ST_SC_FN long st_sc_leaf_55(long v) { return v * 3 + 155; }
ST_SC_FN long st_sc_leaf_56(long v) { return v * 3 + 156; }
ST_SC_FN long st_sc_leaf_57(long v) { return v * 3 + 157; }
ST_SC_FN long st_sc_leaf_58(long v) { return v * 3 + 158; }
ST_SC_FN long st_sc_leaf_59(long v) { return v * 3 + 159; }
ST_SC_FN long st_sc_leaf_60(long v) { return v * 3 + 160; }
ST_SC_FN long st_sc_leaf_61(long v) { return v * 3 + 161; }
ST_SC_FN long st_sc_leaf_62(long v) { return v * 3 + 162; }
ST_SC_FN long st_sc_leaf_63(long v) { return v * 3 + 163; }

ST_SC_FN long st_sc_mid_0(long v) { return st_sc_leaf_00(v) + st_sc_leaf_01(v) + st_sc_leaf_02(v) + st_sc_leaf_03(v) + st_sc_leaf_04(v) + st_sc_leaf_05(v) + st_sc_leaf_06(v) + st_sc_leaf_07(v); }
ST_SC_FN long st_sc_mid_1(long v) { return st_sc_leaf_08(v) + st_sc_leaf_09(v) + st_sc_leaf_10(v) + st_sc_leaf_11(v) + st_sc_leaf_12(v) + st_sc_leaf_13(v) + st_sc_leaf_14(v) + st_sc_leaf_15(v); }
ST_SC_FN long st_sc_mid_2(long v) { return st_sc_leaf_16(v) + st_sc_leaf_17(v) + st_sc_leaf_18(v) + st_sc_leaf_19(v) + st_sc_leaf_20(v) + st_sc_leaf_21(v) + st_sc_leaf_22(v) + st_sc_leaf_23(v); }
ST_SC_FN long st_sc_mid_3(long v) { return st_sc_leaf_24(v) + st_sc_leaf_25(v) + st_sc_leaf_26(v) + st_sc_leaf_27(v) + st_sc_leaf_28(v) + st_sc_leaf_29(v) + st_sc_leaf_30(v) + st_sc_leaf_31(v); }
ST_SC_FN long st_sc_mid_4(long v) { return st_sc_leaf_32(v) + st_sc_leaf_33(v) + st_sc_leaf_34(v) + st_sc_leaf_35(v) + st_sc_leaf_36(v) + st_sc_leaf_37(v) + st_sc_leaf_38(v) + st_sc_leaf_39(v); }
ST_SC_FN long st_sc_mid_5(long v) { return st_sc_leaf_40(v) + st_sc_leaf_41(v) + st_sc_leaf_42(v) + st_sc_leaf_43(v) + st_sc_leaf_44(v) + st_sc_leaf_45(v) + st_sc_leaf_46(v) + st_sc_leaf_47(v); }
ST_SC_FN long st_sc_mid_6(long v) { return st_sc_leaf_48(v) + st_sc_leaf_49(v) + st_sc_leaf_50(v) + st_sc_leaf_51(v) + st_sc_leaf_52(v) + st_sc_leaf_53(v) + st_sc_leaf_54(v) + st_sc_leaf_55(v); }
ST_SC_FN long st_sc_mid_7(long v) { return st_sc_leaf_56(v) + st_sc_leaf_57(v) + st_sc_leaf_58(v) + st_sc_leaf_59(v) + st_sc_leaf_60(v) + st_sc_leaf_61(v) + st_sc_leaf_62(v) + st_sc_leaf_63(v); }

/* Reaches every function above. Its own body stays small — that gap is
 * exactly what the slice-size assertion measures. */
long st_sc_hub(long v) { return st_sc_mid_0(v) + st_sc_mid_1(v) + st_sc_mid_2(v) + st_sc_mid_3(v) + st_sc_mid_4(v) + st_sc_mid_5(v) + st_sc_mid_6(v) + st_sc_mid_7(v); }
