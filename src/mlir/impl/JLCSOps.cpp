//===- JLCSOps.cpp - JLCS dialect operations -----------------------------===//
//
// Operation implementations and verifiers for JLCS dialect
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

#include "JLCSDialect.h"
#include "JLCSOps.h"
#include "JLCSTypes.h"

using namespace mlir;
using namespace mlir::jlcs;

//===----------------------------------------------------------------------===//
// Builders
//===----------------------------------------------------------------------===//
//
// Eight ops declare `skipDefaultBuilders = 1` plus a bodyless
// `OpBuilder<(ins ...)>`. TableGen emits a DECLARATION for each of those and
// expects the body here; none was ever written, so every one was an undefined
// symbol in libJLCS.so. They are referenced even though no hand-written code
// calls them: MLIR's ODS generates a `static OpTy create(OpBuilder&, Location,
// ...)` wrapper per builder, and that wrapper calls `build`.
//
// It stayed invisible because MLIRNative.jl reaches the library through
// `ccall((:sym, path), ...)`, which binds lazily — an undefined symbol nothing
// calls is never resolved. `dlopen(..., RTLD_NOW)` refuses the library
// outright. The producers all build IR as TEXT and parse it, which is why the
// dialect works at all today and why no test noticed.
//
// Defaulted attributes (TypeInfoOp's base tables, VirtualCallOp's this_offset)
// are deliberately NOT set here: `populateDefaultProperties` fills them when
// the properties storage is initialized, which is also why the generated
// builders for the non-skipping ops only assign their explicit arguments.
// Assigning them here would duplicate that and drift from the .td defaults.

void TypeInfoOp::build(OpBuilder &, OperationState &odsState, StringAttr typeName,
                       TypeAttr structType, StringAttr superType,
                       StringAttr destructorName) {
    auto &props = odsState.getOrAddProperties<Properties>();
    props.typeName = typeName;
    props.structType = structType;
    props.superType = superType;
    props.destructorName = destructorName;
}

void GetFieldOp::build(OpBuilder &, OperationState &odsState, Value structValue,
                       IntegerAttr fieldOffset, Type resultType) {
    odsState.addOperands(structValue);
    odsState.getOrAddProperties<Properties>().fieldOffset = fieldOffset;
    odsState.addTypes(resultType);
}

void SetFieldOp::build(OpBuilder &, OperationState &odsState, Value structValue,
                       Value newValue, IntegerAttr fieldOffset) {
    odsState.addOperands({structValue, newValue});
    odsState.getOrAddProperties<Properties>().fieldOffset = fieldOffset;
    // No results: `let results = (outs);`
}

void VirtualCallOp::build(OpBuilder &, OperationState &odsState,
                          SymbolRefAttr class_name, ValueRange args,
                          IntegerAttr vtable_offset, IntegerAttr slot,
                          Type resultType) {
    odsState.addOperands(args);
    auto &props = odsState.getOrAddProperties<Properties>();
    props.class_name = class_name;
    props.vtable_offset = vtable_offset;
    props.slot = slot;
    // The result is `Optional<AnyType>`, so a void virtual method passes a null
    // Type and the op gets zero results. Adding a null type instead would build
    // an op whose result exists with no type — the verifier reports that far
    // from where it was made.
    if (resultType)
        odsState.addTypes(resultType);
}

void LoadArrayElementOp::build(OpBuilder &, OperationState &odsState, Value view,
                               ValueRange indices, Type resultType) {
    odsState.addOperands(view);
    odsState.addOperands(indices);
    odsState.addTypes(resultType);
}

void StoreArrayElementOp::build(OpBuilder &, OperationState &odsState, Value value,
                                Value view, ValueRange indices) {
    odsState.addOperands(value);
    odsState.addOperands(view);
    odsState.addOperands(indices);
}

void MarshalArgOp::build(OpBuilder &, OperationState &odsState, Value srcPtr,
                         ArrayAttr memberTypes, ArrayAttr juliaOffsets,
                         Type resultType) {
    odsState.addOperands(srcPtr);
    auto &props = odsState.getOrAddProperties<Properties>();
    props.memberTypes = memberTypes;
    props.juliaOffsets = juliaOffsets;
    odsState.addTypes(resultType);
}

void MarshalRetOp::build(OpBuilder &, OperationState &odsState, Value packedValue,
                         IntegerAttr numMembers, Type resultType) {
    odsState.addOperands(packedValue);
    odsState.getOrAddProperties<Properties>().numMembers = numMembers;
    odsState.addTypes(resultType);
}

//===----------------------------------------------------------------------===//
// Operation implementations
//===----------------------------------------------------------------------===//

// Verifiers for the two ops whose malformed IR used to SIGSEGV during
// lowering instead of diagnosing (test_jlcs_invariants.jl A2/B2). Producers
// co-generate the paired lists, so these guard hand-written IR and future
// producers.

LogicalResult ScopeOp::verify() {
    size_t nPtrs = getManagedPtrs().size();
    size_t nDtors = getDestructors().size();
    if (nPtrs != nDtors)
        return emitOpError() << "manages " << nPtrs << " pointer(s) but lists "
                             << nDtors << " destructor(s); the lists must pair "
                             << "1:1 (destructors run in reverse order over "
                             << "managed_ptrs)";
    for (Attribute attr : getDestructors())
        if (!isa<FlatSymbolRefAttr>(attr))
            return emitOpError() << "destructors entries must be flat symbol "
                                 << "references, got " << attr;
    return success();
}

LogicalResult VirtualCallOp::verify() {
    if (getArgs().empty())
        return emitOpError() << "requires at least the object pointer as "
                             << "args[0]; vtable_offset/this_offset are read "
                             << "relative to it";
    return success();
}

LogicalResult TypeInfoOp::verify() {
    size_t nNames = getBaseNames().size();
    size_t nOffs = getBaseOffsets().size();
    if (nNames != nOffs)
        return emitOpError() << "baseNames has " << nNames
                             << " entrie(s) but baseOffsets has " << nOffs
                             << "; each base class needs exactly one "
                             << "subobject offset";
    for (Attribute attr : getBaseNames())
        if (!isa<StringAttr>(attr))
            return emitOpError() << "baseNames entries must be string "
                                 << "attributes, got " << attr;
    for (Attribute attr : getBaseOffsets())
        if (!isa<IntegerAttr>(attr))
            return emitOpError() << "baseOffsets entries must be integer "
                                 << "attributes, got " << attr;

    size_t nVNames = getVbaseNames().size();
    size_t nVOffs = getVbaseVtableOffsets().size();
    if (nVNames != nVOffs)
        return emitOpError() << "vbaseNames has " << nVNames
                             << " entrie(s) but vbaseVtableOffsets has "
                             << nVOffs << "; each virtual base needs exactly "
                             << "one vtable-relative offset position";
    for (Attribute attr : getVbaseNames())
        if (!isa<StringAttr>(attr))
            return emitOpError() << "vbaseNames entries must be string "
                                 << "attributes, got " << attr;
    for (Attribute attr : getVbaseVtableOffsets())
        if (!isa<IntegerAttr>(attr))
            return emitOpError() << "vbaseVtableOffsets entries must be "
                                 << "integer attributes, got " << attr;
    return success();
}

LogicalResult MarshalArgOp::verify() {
    size_t nTypes = getMemberTypes().size();
    size_t nOffs = getJuliaOffsets().size();
    if (nTypes != nOffs)
        return emitOpError() << "memberTypes has " << nTypes
                             << " entrie(s) but juliaOffsets has " << nOffs
                             << "; each member needs exactly one offset";
    for (Attribute attr : getMemberTypes())
        if (!isa<TypeAttr>(attr))
            return emitOpError() << "memberTypes entries must be type "
                                 << "attributes, got " << attr;
    for (Attribute attr : getJuliaOffsets())
        if (!isa<IntegerAttr>(attr))
            return emitOpError() << "juliaOffsets entries must be integer "
                                 << "attributes, got " << attr;
    return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated operation definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "JLCSOps.cpp.inc"
