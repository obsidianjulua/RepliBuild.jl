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
