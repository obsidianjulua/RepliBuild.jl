//===- JLCSDialect.cpp - JLCS dialect implementation ---------------------===//

#include "JLCSDialect.h"
#include "JLCSOps.h"
#include "JLCSTypes.h"

// Include the generated dialect class BEFORE defining methods that depend on it.
#include "JLCSDialect.cpp.inc"

using namespace mlir;
using namespace mlir::jlcs;

void JLCSDialect::initialize()
{
    // Register generated types
    addTypes<
#define GET_TYPEDEF_LIST
#include "JLCSTypes.cpp.inc"
        >();

    // Register generated ops
    addOperations<
#define GET_OP_LIST
#include "JLCSOps.cpp.inc"
        >();
}

// Use the generated parser/printer helpers where available.
// Parse a type token (delegates to generated parser functions).
Type JLCSDialect::parseType(DialectAsmParser& parser) const
{
    StringRef mnemonic;
    if (parser.parseKeyword(&mnemonic))
        return Type();
    // generatedTypeParser is provided by TableGen when -gen-type-decls/defs were run
    return generatedTypeParser(parser, mnemonic);
}

void JLCSDialect::printType(Type type, DialectAsmPrinter& printer) const
{
    // generatedTypePrinter is provided by TableGen when -gen-type-decls/defs were run
    if (failed(generatedTypePrinter(type, printer)))
        printer << "<invalid jlcs type>";
}

// Minimal attribute stubs (no custom attributes yet)
Attribute JLCSDialect::parseAttribute(DialectAsmParser& parser, Type type) const
{
    parser.emitError(parser.getNameLoc(), "JLCS dialect: no attributes implemented");
    return Attribute();
}

void JLCSDialect::printAttribute(Attribute attr, DialectAsmPrinter& printer) const
{
    (void)attr;
    // nothing for now
}

// C API: keep the exported registration helper for Julia bindings.
extern "C" {
void registerJLCSDialect(MlirContext context)
{
    mlir::MLIRContext* ctx = unwrap(context);
    mlir::DialectRegistry registry;
    registry.insert<mlir::jlcs::JLCSDialect>();
    ctx->appendDialectRegistry(registry);
    ctx->loadDialect<mlir::jlcs::JLCSDialect>();
}
} // extern "C"
//
