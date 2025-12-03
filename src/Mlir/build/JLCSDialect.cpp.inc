/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: JLCSDialect.td                                                       *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::jlcs::JLCSDialect)
namespace mlir {
namespace jlcs {

JLCSDialect::JLCSDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<JLCSDialect>())
    
     {
  
  initialize();
}

JLCSDialect::~JLCSDialect() = default;

} // namespace jlcs
} // namespace mlir
