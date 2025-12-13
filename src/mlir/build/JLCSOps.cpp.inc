/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: JLCS.td                                                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::jlcs::GetFieldOp,
::mlir::jlcs::FFECallOp,
::mlir::jlcs::LoadArrayElementOp,
::mlir::jlcs::SetFieldOp,
::mlir::jlcs::StoreArrayElementOp,
::mlir::jlcs::TypeInfoOp,
::mlir::jlcs::VirtualCallOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace jlcs {

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_JLCS1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_JLCS2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of any type, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_JLCS3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::IndexType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of index, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_JLCS1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_JLCS1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_JLCS1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_JLCS2(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::StringAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: string attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_JLCS2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_JLCS2(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_JLCS3(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: any type attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_JLCS3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_JLCS3(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_JLCS4(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::SymbolRefAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: symbol reference attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_JLCS4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_JLCS4(attr, attrName, [op]() {
    return op->emitOpError();
  });
}
} // namespace jlcs
} // namespace mlir
namespace mlir {
namespace jlcs {

//===----------------------------------------------------------------------===//
// ::mlir::jlcs::GetFieldOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetFieldOpGenericAdaptorBase::GetFieldOpGenericAdaptorBase(GetFieldOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

uint64_t GetFieldOpGenericAdaptorBase::getFieldOffset() {
  auto attr = getFieldOffsetAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
GetFieldOpAdaptor::GetFieldOpAdaptor(GetFieldOp op) : GetFieldOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult GetFieldOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_fieldOffset = getProperties().fieldOffset; (void)tblgen_fieldOffset;
  if (!tblgen_fieldOffset) return emitError(loc, "'jlcs.get_field' op ""requires attribute 'fieldOffset'");

  if (tblgen_fieldOffset && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_fieldOffset))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_fieldOffset).getType().isSignlessInteger(64)))))
    return emitError(loc, "'jlcs.get_field' op ""attribute 'fieldOffset' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

::llvm::LogicalResult GetFieldOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.fieldOffset;
       auto attr = dict.get("fieldOffset");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `fieldOffset` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute GetFieldOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.fieldOffset;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("fieldOffset",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code GetFieldOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.fieldOffset.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> GetFieldOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "fieldOffset")
      return prop.fieldOffset;
  return std::nullopt;
}

void GetFieldOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "fieldOffset") {
       prop.fieldOffset = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.fieldOffset)>>(value);
       return;
    }
}

void GetFieldOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.fieldOffset) attrs.append("fieldOffset", prop.fieldOffset);
}

::llvm::LogicalResult GetFieldOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getFieldOffsetAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_JLCS1(attr, "fieldOffset", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult GetFieldOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.fieldOffset)))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetFieldOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.fieldOffset);
}

uint64_t GetFieldOp::getFieldOffset() {
  auto attr = getFieldOffsetAttr();
  return attr.getValue().getZExtValue();
}

void GetFieldOp::setFieldOffset(uint64_t attrValue) {
  getProperties().fieldOffset = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

GetFieldOp GetFieldOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, Value structValue, IntegerAttr fieldOffset, Type resultType) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, structValue, fieldOffset, resultType);
  auto __res__ = ::llvm::dyn_cast<GetFieldOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

GetFieldOp GetFieldOp::create(::mlir::ImplicitLocOpBuilder &builder, Value structValue, IntegerAttr fieldOffset, Type resultType) {
  return create(builder, builder.getLoc(), structValue, fieldOffset, resultType);
}

::llvm::LogicalResult GetFieldOp::verifyInvariantsImpl() {
  auto tblgen_fieldOffset = getProperties().fieldOffset; (void)tblgen_fieldOffset;
  if (!tblgen_fieldOffset) return emitOpError("requires attribute 'fieldOffset'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_JLCS1(*this, tblgen_fieldOffset, "fieldOffset")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult GetFieldOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult GetFieldOp::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;
  {

    auto &propStorage = prop.fieldOffset;
    auto fieldOffsetAttrName = ::mlir::StringAttr::get(ctx, "fieldOffset");
    auto attr = dict.get(fieldOffsetAttrName);
    usedKeys.insert(fieldOffsetAttrName);
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for fieldOffset in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `fieldOffset` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}

} // namespace jlcs
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::jlcs::GetFieldOp)

namespace mlir {
namespace jlcs {

//===----------------------------------------------------------------------===//
// ::mlir::jlcs::FFECallOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FFECallOpGenericAdaptorBase::FFECallOpGenericAdaptorBase(FFECallOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> FFECallOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

} // namespace detail
FFECallOpAdaptor::FFECallOpAdaptor(FFECallOp op) : FFECallOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult FFECallOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FFECallOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange FFECallOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FFECallOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::llvm::LogicalResult FFECallOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute FFECallOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code FFECallOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> FFECallOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void FFECallOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void FFECallOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult FFECallOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void FFECallOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

FFECallOp FFECallOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, resultTypes, operands, attributes);
  auto __res__ = ::llvm::dyn_cast<FFECallOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

FFECallOp FFECallOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), resultTypes, operands, attributes);
}

void FFECallOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  odsState.addTypes(resultTypes);
}

FFECallOp FFECallOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, resultTypes, operands, properties, discardableAttributes);
  auto __res__ = ::llvm::dyn_cast<FFECallOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

FFECallOp FFECallOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), resultTypes, operands, properties, discardableAttributes);
}

::llvm::LogicalResult FFECallOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult FFECallOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult FFECallOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> argsTypes;
  ::llvm::SmallVector<::mlir::Type, 1> resultsTypes;

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(argsTypes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(resultsTypes))
    return ::mlir::failure();
  result.addTypes(resultsTypes);
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FFECallOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArgs();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getArgs().getTypes();
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getResults().getTypes();
}

void FFECallOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace jlcs
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::jlcs::FFECallOp)

namespace mlir {
namespace jlcs {

//===----------------------------------------------------------------------===//
// ::mlir::jlcs::LoadArrayElementOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
LoadArrayElementOpGenericAdaptorBase::LoadArrayElementOpGenericAdaptorBase(LoadArrayElementOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> LoadArrayElementOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

} // namespace detail
LoadArrayElementOpAdaptor::LoadArrayElementOpAdaptor(LoadArrayElementOp op) : LoadArrayElementOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult LoadArrayElementOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LoadArrayElementOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange LoadArrayElementOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::LogicalResult LoadArrayElementOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute LoadArrayElementOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code LoadArrayElementOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> LoadArrayElementOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void LoadArrayElementOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void LoadArrayElementOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult LoadArrayElementOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

LoadArrayElementOp LoadArrayElementOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, Value view, ValueRange indices, Type resultType) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, view, indices, resultType);
  auto __res__ = ::llvm::dyn_cast<LoadArrayElementOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

LoadArrayElementOp LoadArrayElementOp::create(::mlir::ImplicitLocOpBuilder &builder, Value view, ValueRange indices, Type resultType) {
  return create(builder, builder.getLoc(), view, indices, resultType);
}

::llvm::LogicalResult LoadArrayElementOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult LoadArrayElementOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult LoadArrayElementOp::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}

} // namespace jlcs
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::jlcs::LoadArrayElementOp)

namespace mlir {
namespace jlcs {

//===----------------------------------------------------------------------===//
// ::mlir::jlcs::SetFieldOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SetFieldOpGenericAdaptorBase::SetFieldOpGenericAdaptorBase(SetFieldOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

uint64_t SetFieldOpGenericAdaptorBase::getFieldOffset() {
  auto attr = getFieldOffsetAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
SetFieldOpAdaptor::SetFieldOpAdaptor(SetFieldOp op) : SetFieldOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult SetFieldOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_fieldOffset = getProperties().fieldOffset; (void)tblgen_fieldOffset;
  if (!tblgen_fieldOffset) return emitError(loc, "'jlcs.set_field' op ""requires attribute 'fieldOffset'");

  if (tblgen_fieldOffset && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_fieldOffset))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_fieldOffset).getType().isSignlessInteger(64)))))
    return emitError(loc, "'jlcs.set_field' op ""attribute 'fieldOffset' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

::llvm::LogicalResult SetFieldOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.fieldOffset;
       auto attr = dict.get("fieldOffset");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `fieldOffset` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute SetFieldOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.fieldOffset;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("fieldOffset",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code SetFieldOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.fieldOffset.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> SetFieldOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "fieldOffset")
      return prop.fieldOffset;
  return std::nullopt;
}

void SetFieldOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "fieldOffset") {
       prop.fieldOffset = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.fieldOffset)>>(value);
       return;
    }
}

void SetFieldOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.fieldOffset) attrs.append("fieldOffset", prop.fieldOffset);
}

::llvm::LogicalResult SetFieldOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getFieldOffsetAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_JLCS1(attr, "fieldOffset", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult SetFieldOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.fieldOffset)))
    return ::mlir::failure();
  return ::mlir::success();
}

void SetFieldOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.fieldOffset);
}

uint64_t SetFieldOp::getFieldOffset() {
  auto attr = getFieldOffsetAttr();
  return attr.getValue().getZExtValue();
}

void SetFieldOp::setFieldOffset(uint64_t attrValue) {
  getProperties().fieldOffset = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

SetFieldOp SetFieldOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, Value structValue, Value newValue, IntegerAttr fieldOffset) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, structValue, newValue, fieldOffset);
  auto __res__ = ::llvm::dyn_cast<SetFieldOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

SetFieldOp SetFieldOp::create(::mlir::ImplicitLocOpBuilder &builder, Value structValue, Value newValue, IntegerAttr fieldOffset) {
  return create(builder, builder.getLoc(), structValue, newValue, fieldOffset);
}

::llvm::LogicalResult SetFieldOp::verifyInvariantsImpl() {
  auto tblgen_fieldOffset = getProperties().fieldOffset; (void)tblgen_fieldOffset;
  if (!tblgen_fieldOffset) return emitOpError("requires attribute 'fieldOffset'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_JLCS1(*this, tblgen_fieldOffset, "fieldOffset")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult SetFieldOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult SetFieldOp::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;
  {

    auto &propStorage = prop.fieldOffset;
    auto fieldOffsetAttrName = ::mlir::StringAttr::get(ctx, "fieldOffset");
    auto attr = dict.get(fieldOffsetAttrName);
    usedKeys.insert(fieldOffsetAttrName);
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for fieldOffset in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `fieldOffset` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}

void SetFieldOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace jlcs
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::jlcs::SetFieldOp)

namespace mlir {
namespace jlcs {

//===----------------------------------------------------------------------===//
// ::mlir::jlcs::StoreArrayElementOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
StoreArrayElementOpGenericAdaptorBase::StoreArrayElementOpGenericAdaptorBase(StoreArrayElementOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> StoreArrayElementOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

} // namespace detail
StoreArrayElementOpAdaptor::StoreArrayElementOpAdaptor(StoreArrayElementOp op) : StoreArrayElementOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult StoreArrayElementOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> StoreArrayElementOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange StoreArrayElementOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::LogicalResult StoreArrayElementOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute StoreArrayElementOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code StoreArrayElementOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> StoreArrayElementOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void StoreArrayElementOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void StoreArrayElementOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult StoreArrayElementOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

StoreArrayElementOp StoreArrayElementOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, Value value, Value view, ValueRange indices) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, value, view, indices);
  auto __res__ = ::llvm::dyn_cast<StoreArrayElementOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

StoreArrayElementOp StoreArrayElementOp::create(::mlir::ImplicitLocOpBuilder &builder, Value value, Value view, ValueRange indices) {
  return create(builder, builder.getLoc(), value, view, indices);
}

::llvm::LogicalResult StoreArrayElementOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult StoreArrayElementOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult StoreArrayElementOp::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}

} // namespace jlcs
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::jlcs::StoreArrayElementOp)

namespace mlir {
namespace jlcs {

//===----------------------------------------------------------------------===//
// ::mlir::jlcs::TypeInfoOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TypeInfoOpGenericAdaptorBase::TypeInfoOpGenericAdaptorBase(TypeInfoOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::llvm::StringRef TypeInfoOpGenericAdaptorBase::getTypeName() {
  auto attr = getTypeNameAttr();
  return attr.getValue();
}

::mlir::Type TypeInfoOpGenericAdaptorBase::getStructType() {
  auto attr = getStructTypeAttr();
  return ::llvm::cast<::mlir::Type>(attr.getValue());
}

::mlir::StringAttr TypeInfoOpGenericAdaptorBase::getSuperTypeAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::StringAttr>(getProperties().superType);
  return attr;
}

::llvm::StringRef TypeInfoOpGenericAdaptorBase::getSuperType() {
  auto attr = getSuperTypeAttr();
  return attr.getValue();
}

} // namespace detail
TypeInfoOpAdaptor::TypeInfoOpAdaptor(TypeInfoOp op) : TypeInfoOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TypeInfoOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_structType = getProperties().structType; (void)tblgen_structType;
  if (!tblgen_structType) return emitError(loc, "'jlcs.type_info' op ""requires attribute 'structType'");
  auto tblgen_superType = getProperties().superType; (void)tblgen_superType;
  auto tblgen_typeName = getProperties().typeName; (void)tblgen_typeName;
  if (!tblgen_typeName) return emitError(loc, "'jlcs.type_info' op ""requires attribute 'typeName'");

  if (tblgen_typeName && !((::llvm::isa<::mlir::StringAttr>(tblgen_typeName))))
    return emitError(loc, "'jlcs.type_info' op ""attribute 'typeName' failed to satisfy constraint: string attribute");

  if (tblgen_structType && !(((::llvm::isa<::mlir::TypeAttr>(tblgen_structType))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(tblgen_structType).getValue()))) && ((true))))
    return emitError(loc, "'jlcs.type_info' op ""attribute 'structType' failed to satisfy constraint: any type attribute");

  if (tblgen_superType && !((::llvm::isa<::mlir::StringAttr>(tblgen_superType))))
    return emitError(loc, "'jlcs.type_info' op ""attribute 'superType' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

::llvm::LogicalResult TypeInfoOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.structType;
       auto attr = dict.get("structType");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `structType` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.superType;
       auto attr = dict.get("superType");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `superType` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.typeName;
       auto attr = dict.get("typeName");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `typeName` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute TypeInfoOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.structType;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("structType",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.superType;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("superType",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.typeName;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("typeName",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TypeInfoOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.structType.getAsOpaquePointer()), 
    llvm::hash_value(prop.superType.getAsOpaquePointer()), 
    llvm::hash_value(prop.typeName.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> TypeInfoOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "structType")
      return prop.structType;

    if (name == "superType")
      return prop.superType;

    if (name == "typeName")
      return prop.typeName;
  return std::nullopt;
}

void TypeInfoOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "structType") {
       prop.structType = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.structType)>>(value);
       return;
    }

    if (name == "superType") {
       prop.superType = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.superType)>>(value);
       return;
    }

    if (name == "typeName") {
       prop.typeName = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.typeName)>>(value);
       return;
    }
}

void TypeInfoOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.structType) attrs.append("structType", prop.structType);

    if (prop.superType) attrs.append("superType", prop.superType);

    if (prop.typeName) attrs.append("typeName", prop.typeName);
}

::llvm::LogicalResult TypeInfoOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getStructTypeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_JLCS3(attr, "structType", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getSuperTypeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_JLCS2(attr, "superType", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getTypeNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_JLCS2(attr, "typeName", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult TypeInfoOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.structType)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.superType)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.typeName)))
    return ::mlir::failure();
  return ::mlir::success();
}

void TypeInfoOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.structType);

  writer.writeOptionalAttribute(prop.superType);
  writer.writeAttribute(prop.typeName);
}

::llvm::StringRef TypeInfoOp::getTypeName() {
  auto attr = getTypeNameAttr();
  return attr.getValue();
}

::mlir::Type TypeInfoOp::getStructType() {
  auto attr = getStructTypeAttr();
  return ::llvm::cast<::mlir::Type>(attr.getValue());
}

::llvm::StringRef TypeInfoOp::getSuperType() {
  auto attr = getSuperTypeAttr();
  return attr.getValue();
}

void TypeInfoOp::setTypeName(::llvm::StringRef attrValue) {
  getProperties().typeName = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void TypeInfoOp::setStructType(::mlir::Type attrValue) {
  getProperties().structType = ::mlir::TypeAttr::get(attrValue);
}

void TypeInfoOp::setSuperType(::llvm::StringRef attrValue) {
  getProperties().superType = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

TypeInfoOp TypeInfoOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, StringAttr typeName, TypeAttr structType, StringAttr superType) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, typeName, structType, superType);
  auto __res__ = ::llvm::dyn_cast<TypeInfoOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TypeInfoOp TypeInfoOp::create(::mlir::ImplicitLocOpBuilder &builder, StringAttr typeName, TypeAttr structType, StringAttr superType) {
  return create(builder, builder.getLoc(), typeName, structType, superType);
}

void TypeInfoOp::populateDefaultProperties(::mlir::OperationName opName, Properties &properties) {
  ::mlir::Builder odsBuilder(opName.getContext());
  if (!properties.superType)
    properties.superType = odsBuilder.getStringAttr("""");
}

::llvm::LogicalResult TypeInfoOp::verifyInvariantsImpl() {
  auto tblgen_structType = getProperties().structType; (void)tblgen_structType;
  if (!tblgen_structType) return emitOpError("requires attribute 'structType'");
  auto tblgen_superType = getProperties().superType; (void)tblgen_superType;
  auto tblgen_typeName = getProperties().typeName; (void)tblgen_typeName;
  if (!tblgen_typeName) return emitOpError("requires attribute 'typeName'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_JLCS2(*this, tblgen_typeName, "typeName")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_JLCS3(*this, tblgen_structType, "structType")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_JLCS2(*this, tblgen_superType, "superType")))
    return ::mlir::failure();
  return ::mlir::success();
}

::llvm::LogicalResult TypeInfoOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult TypeInfoOp::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;
  {

    auto &propStorage = prop.typeName;
    auto typeNameAttrName = ::mlir::StringAttr::get(ctx, "typeName");
    auto attr = dict.get(typeNameAttrName);
    usedKeys.insert(typeNameAttrName);
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for typeName in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `typeName` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  {

    auto &propStorage = prop.structType;
    auto structTypeAttrName = ::mlir::StringAttr::get(ctx, "structType");
    auto attr = dict.get(structTypeAttrName);
    usedKeys.insert(structTypeAttrName);
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for structType in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `structType` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  {

    auto &propStorage = prop.superType;
    auto superTypeAttrName = ::mlir::StringAttr::get(ctx, "superType");
    auto attr = dict.get(superTypeAttrName);
    usedKeys.insert(superTypeAttrName);
    if (attr || /*isRequired=*/false) {
      if (!attr) {
        emitError() << "expected key entry for superType in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `superType` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}

void TypeInfoOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace jlcs
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::jlcs::TypeInfoOp)

namespace mlir {
namespace jlcs {

//===----------------------------------------------------------------------===//
// ::mlir::jlcs::VirtualCallOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
VirtualCallOpGenericAdaptorBase::VirtualCallOpGenericAdaptorBase(VirtualCallOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> VirtualCallOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::SymbolRefAttr VirtualCallOpGenericAdaptorBase::getClassName() {
  auto attr = getClassNameAttr();
  return attr;
}

uint64_t VirtualCallOpGenericAdaptorBase::getVtableOffset() {
  auto attr = getVtableOffsetAttr();
  return attr.getValue().getZExtValue();
}

uint64_t VirtualCallOpGenericAdaptorBase::getSlot() {
  auto attr = getSlotAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
VirtualCallOpAdaptor::VirtualCallOpAdaptor(VirtualCallOp op) : VirtualCallOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult VirtualCallOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_class_name = getProperties().class_name; (void)tblgen_class_name;
  if (!tblgen_class_name) return emitError(loc, "'jlcs.vcall' op ""requires attribute 'class_name'");
  auto tblgen_slot = getProperties().slot; (void)tblgen_slot;
  if (!tblgen_slot) return emitError(loc, "'jlcs.vcall' op ""requires attribute 'slot'");
  auto tblgen_vtable_offset = getProperties().vtable_offset; (void)tblgen_vtable_offset;
  if (!tblgen_vtable_offset) return emitError(loc, "'jlcs.vcall' op ""requires attribute 'vtable_offset'");

  if (tblgen_class_name && !((::llvm::isa<::mlir::SymbolRefAttr>(tblgen_class_name))))
    return emitError(loc, "'jlcs.vcall' op ""attribute 'class_name' failed to satisfy constraint: symbol reference attribute");

  if (tblgen_vtable_offset && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_vtable_offset))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_vtable_offset).getType().isSignlessInteger(64)))))
    return emitError(loc, "'jlcs.vcall' op ""attribute 'vtable_offset' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_slot && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_slot))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_slot).getType().isSignlessInteger(64)))))
    return emitError(loc, "'jlcs.vcall' op ""attribute 'slot' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VirtualCallOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange VirtualCallOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VirtualCallOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::llvm::LogicalResult VirtualCallOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.class_name;
       auto attr = dict.get("class_name");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `class_name` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.slot;
       auto attr = dict.get("slot");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `slot` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.vtable_offset;
       auto attr = dict.get("vtable_offset");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `vtable_offset` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute VirtualCallOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.class_name;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("class_name",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.slot;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("slot",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.vtable_offset;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("vtable_offset",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code VirtualCallOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.class_name.getAsOpaquePointer()), 
    llvm::hash_value(prop.slot.getAsOpaquePointer()), 
    llvm::hash_value(prop.vtable_offset.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> VirtualCallOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "class_name")
      return prop.class_name;

    if (name == "slot")
      return prop.slot;

    if (name == "vtable_offset")
      return prop.vtable_offset;
  return std::nullopt;
}

void VirtualCallOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "class_name") {
       prop.class_name = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.class_name)>>(value);
       return;
    }

    if (name == "slot") {
       prop.slot = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.slot)>>(value);
       return;
    }

    if (name == "vtable_offset") {
       prop.vtable_offset = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.vtable_offset)>>(value);
       return;
    }
}

void VirtualCallOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.class_name) attrs.append("class_name", prop.class_name);

    if (prop.slot) attrs.append("slot", prop.slot);

    if (prop.vtable_offset) attrs.append("vtable_offset", prop.vtable_offset);
}

::llvm::LogicalResult VirtualCallOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getClassNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_JLCS4(attr, "class_name", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getSlotAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_JLCS1(attr, "slot", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getVtableOffsetAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_JLCS1(attr, "vtable_offset", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult VirtualCallOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.class_name)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.slot)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.vtable_offset)))
    return ::mlir::failure();
  return ::mlir::success();
}

void VirtualCallOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.class_name);
  writer.writeAttribute(prop.slot);
  writer.writeAttribute(prop.vtable_offset);
}

::mlir::SymbolRefAttr VirtualCallOp::getClassName() {
  auto attr = getClassNameAttr();
  return attr;
}

uint64_t VirtualCallOp::getVtableOffset() {
  auto attr = getVtableOffsetAttr();
  return attr.getValue().getZExtValue();
}

uint64_t VirtualCallOp::getSlot() {
  auto attr = getSlotAttr();
  return attr.getValue().getZExtValue();
}

void VirtualCallOp::setVtableOffset(uint64_t attrValue) {
  getProperties().vtable_offset = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void VirtualCallOp::setSlot(uint64_t attrValue) {
  getProperties().slot = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

VirtualCallOp VirtualCallOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, SymbolRefAttr class_name, ValueRange args, IntegerAttr vtable_offset, IntegerAttr slot, Type resultType) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, class_name, args, vtable_offset, slot, resultType);
  auto __res__ = ::llvm::dyn_cast<VirtualCallOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

VirtualCallOp VirtualCallOp::create(::mlir::ImplicitLocOpBuilder &builder, SymbolRefAttr class_name, ValueRange args, IntegerAttr vtable_offset, IntegerAttr slot, Type resultType) {
  return create(builder, builder.getLoc(), class_name, args, vtable_offset, slot, resultType);
}

::llvm::LogicalResult VirtualCallOp::verifyInvariantsImpl() {
  auto tblgen_class_name = getProperties().class_name; (void)tblgen_class_name;
  if (!tblgen_class_name) return emitOpError("requires attribute 'class_name'");
  auto tblgen_slot = getProperties().slot; (void)tblgen_slot;
  if (!tblgen_slot) return emitOpError("requires attribute 'slot'");
  auto tblgen_vtable_offset = getProperties().vtable_offset; (void)tblgen_vtable_offset;
  if (!tblgen_vtable_offset) return emitOpError("requires attribute 'vtable_offset'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_JLCS4(*this, tblgen_class_name, "class_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_JLCS1(*this, tblgen_vtable_offset, "vtable_offset")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_JLCS1(*this, tblgen_slot, "slot")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("result group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_JLCS1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult VirtualCallOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult VirtualCallOp::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;
  {

    auto &propStorage = prop.class_name;
    auto class_nameAttrName = ::mlir::StringAttr::get(ctx, "class_name");
    auto attr = dict.get(class_nameAttrName);
    usedKeys.insert(class_nameAttrName);
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for class_name in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `class_name` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  {

    auto &propStorage = prop.vtable_offset;
    auto vtable_offsetAttrName = ::mlir::StringAttr::get(ctx, "vtable_offset");
    auto attr = dict.get(vtable_offsetAttrName);
    usedKeys.insert(vtable_offsetAttrName);
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for vtable_offset in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `vtable_offset` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  {

    auto &propStorage = prop.slot;
    auto slotAttrName = ::mlir::StringAttr::get(ctx, "slot");
    auto attr = dict.get(slotAttrName);
    usedKeys.insert(slotAttrName);
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for slot in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `slot` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}

} // namespace jlcs
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::jlcs::VirtualCallOp)


#endif  // GET_OP_CLASSES

