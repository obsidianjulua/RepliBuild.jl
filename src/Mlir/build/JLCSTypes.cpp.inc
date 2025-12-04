/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::jlcs::CStructType,
::mlir::jlcs::ArrayViewType

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::jlcs::CStructType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::jlcs::CStructType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::jlcs::ArrayViewType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::jlcs::ArrayViewType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)    .Case<::mlir::jlcs::CStructType>([&](auto t) {
      printer << ::mlir::jlcs::CStructType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::jlcs::ArrayViewType>([&](auto t) {
      printer << ::mlir::jlcs::ArrayViewType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace jlcs {
namespace detail {
struct CStructTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<StringAttr, ::llvm::ArrayRef<Type>, ArrayAttr>;
  CStructTypeStorage(StringAttr juliaTypeName, ::llvm::ArrayRef<Type> fieldTypes, ArrayAttr fieldOffsets) : juliaTypeName(std::move(juliaTypeName)), fieldTypes(std::move(fieldTypes)), fieldOffsets(std::move(fieldOffsets)) {}

  KeyTy getAsKey() const {
    return KeyTy(juliaTypeName, fieldTypes, fieldOffsets);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (juliaTypeName == std::get<0>(tblgenKey)) && (fieldTypes == std::get<1>(tblgenKey)) && (fieldOffsets == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static CStructTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto juliaTypeName = std::move(std::get<0>(tblgenKey));
    auto fieldTypes = std::move(std::get<1>(tblgenKey));
    auto fieldOffsets = std::move(std::get<2>(tblgenKey));
    fieldTypes = allocator.copyInto(fieldTypes);
    return new (allocator.allocate<CStructTypeStorage>()) CStructTypeStorage(std::move(juliaTypeName), std::move(fieldTypes), std::move(fieldOffsets));
  }

  StringAttr juliaTypeName;
  ::llvm::ArrayRef<Type> fieldTypes;
  ArrayAttr fieldOffsets;
};
} // namespace detail
CStructType CStructType::get(::mlir::MLIRContext *context, StringAttr juliaTypeName, ::llvm::ArrayRef<Type> fieldTypes, ArrayAttr fieldOffsets) {
  return Base::get(context, std::move(juliaTypeName), std::move(fieldTypes), std::move(fieldOffsets));
}

::mlir::Type CStructType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<StringAttr> _result_juliaTypeName;
  ::mlir::FailureOr<::llvm::SmallVector<Type>> _result_fieldTypes;
  ::mlir::FailureOr<ArrayAttr> _result_fieldOffsets;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'juliaTypeName'
  _result_juliaTypeName = ::mlir::FieldParser<StringAttr>::parse(odsParser);
  if (::mlir::failed(_result_juliaTypeName)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse CStructType parameter 'juliaTypeName' which is to be a `StringAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  // Parse literal '['
  if (odsParser.parseLSquare()) return {};

  // Parse variable 'fieldTypes'
  _result_fieldTypes = ::mlir::FieldParser<::llvm::SmallVector<Type>>::parse(odsParser);
  if (::mlir::failed(_result_fieldTypes)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse CStructType parameter 'fieldTypes' which is to be a `::llvm::ArrayRef<Type>`");
    return {};
  }
  // Parse literal ']'
  if (odsParser.parseRSquare()) return {};
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  // Parse literal '['
  if (odsParser.parseLSquare()) return {};

  // Parse variable 'fieldOffsets'
  _result_fieldOffsets = ::mlir::FieldParser<ArrayAttr>::parse(odsParser);
  if (::mlir::failed(_result_fieldOffsets)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse CStructType parameter 'fieldOffsets' which is to be a `ArrayAttr`");
    return {};
  }
  // Parse literal ']'
  if (odsParser.parseRSquare()) return {};
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_juliaTypeName));
  assert(::mlir::succeeded(_result_fieldTypes));
  assert(::mlir::succeeded(_result_fieldOffsets));
  return CStructType::get(odsParser.getContext(),
      StringAttr((*_result_juliaTypeName)),
      ::llvm::ArrayRef<Type>((*_result_fieldTypes)),
      ArrayAttr((*_result_fieldOffsets)));
}

void CStructType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getJuliaTypeName());
  odsPrinter << ",";
  odsPrinter << ' ' << "[";
  odsPrinter.printStrippedAttrOrType(getFieldTypes());
  odsPrinter << "]";
  odsPrinter << ",";
  odsPrinter << ' ' << "[";
  odsPrinter.printStrippedAttrOrType(getFieldOffsets());
  odsPrinter << "]";
  odsPrinter << ">";
}

StringAttr CStructType::getJuliaTypeName() const {
  return getImpl()->juliaTypeName;
}

::llvm::ArrayRef<Type> CStructType::getFieldTypes() const {
  return getImpl()->fieldTypes;
}

ArrayAttr CStructType::getFieldOffsets() const {
  return getImpl()->fieldOffsets;
}

} // namespace jlcs
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::jlcs::CStructType)
namespace mlir {
namespace jlcs {
namespace detail {
struct ArrayViewTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type, unsigned>;
  ArrayViewTypeStorage(Type elementType, unsigned rank) : elementType(std::move(elementType)), rank(std::move(rank)) {}

  KeyTy getAsKey() const {
    return KeyTy(elementType, rank);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (elementType == std::get<0>(tblgenKey)) && (rank == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static ArrayViewTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto elementType = std::move(std::get<0>(tblgenKey));
    auto rank = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<ArrayViewTypeStorage>()) ArrayViewTypeStorage(std::move(elementType), std::move(rank));
  }

  Type elementType;
  unsigned rank;
};
} // namespace detail
ArrayViewType ArrayViewType::get(::mlir::MLIRContext *context, Type elementType, unsigned rank) {
  return Base::get(context, std::move(elementType), std::move(rank));
}

::mlir::Type ArrayViewType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<Type> _result_elementType;
  ::mlir::FailureOr<unsigned> _result_rank;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'elementType'
  _result_elementType = ::mlir::FieldParser<Type>::parse(odsParser);
  if (::mlir::failed(_result_elementType)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ArrayViewType parameter 'elementType' which is to be a `Type`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'rank'
  _result_rank = ::mlir::FieldParser<unsigned>::parse(odsParser);
  if (::mlir::failed(_result_rank)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ArrayViewType parameter 'rank' which is to be a `unsigned`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_elementType));
  assert(::mlir::succeeded(_result_rank));
  return ArrayViewType::get(odsParser.getContext(),
      Type((*_result_elementType)),
      unsigned((*_result_rank)));
}

void ArrayViewType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getElementType());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getRank());
  odsPrinter << ">";
}

Type ArrayViewType::getElementType() const {
  return getImpl()->elementType;
}

unsigned ArrayViewType::getRank() const {
  return getImpl()->rank;
}

} // namespace jlcs
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::jlcs::ArrayViewType)

#endif  // GET_TYPEDEF_CLASSES

