//===- JLCSAbiOracle.cpp - LLVM's ABI library as a SysV oracle ------------===//
//
// LLVM 23's ABI lowering library (llvm/ABI) classifies a function's arguments
// and return for x86-64 SysV from a frontend-neutral description of their
// types. That makes it an independent second opinion on RepliBuild's own
// classification (FunctionGen.sysv_record_abi, stated on each call and acted
// on by buildCallShape in JLCSPasses.cpp), and test/test_sysv_abi_oracle.jl
// diffs the two over a corpus of C records.
//
// Test only. Nothing on the wrapping or JIT path calls this. It is compiled
// only when CMake finds the library and its x86-64 classifier (LLVM 23+:
// 22 has the type system alone, 21 has none), so libJLCS still builds
// against LLVM/MLIR 21 and 22, and there the symbol below is absent, which
// is how the test knows to skip. The library has no Win64 classifier, so
// classifyWin64Struct has no oracle here.
//
// The request and the answer are JSON; see jlcs_abi_oracle_sysv.
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <string>

using namespace llvm;

namespace {

// A type description from the request, as an llvm::abi::Type. Every failure
// sets `error` and returns null; the first one wins.
struct TypeReader {
    abi::TypeBuilder& TB;
    std::string error;

    const abi::Type* fail(const Twine& why)
    {
        if (error.empty())
            error = why.str();
        return nullptr;
    }

    std::optional<Align> alignOf(const json::Object& O)
    {
        std::optional<int64_t> a = O.getInteger("align");
        if (!a || *a <= 0 || !isPowerOf2_64(static_cast<uint64_t>(*a)))
            return std::nullopt;
        return Align(static_cast<uint64_t>(*a));
    }

    const abi::Type* read(const json::Value* V)
    {
        const json::Object* O = V ? V->getAsObject() : nullptr;
        if (!O)
            return fail("a type is not an object");
        std::optional<StringRef> K = O->getString("k");
        if (!K)
            return fail("a type has no \"k\"");

        if (*K == "void")
            return TB.getVoidType();
        if (*K == "ptr")
            return TB.getPointerType(64, Align(8));
        if (*K == "int") {
            std::optional<int64_t> bits = O->getInteger("bits");
            std::optional<bool> sign = O->getBoolean("signed");
            std::optional<Align> al = alignOf(*O);
            if (!bits || *bits <= 0 || !sign || !al)
                return fail("an int needs bits, signed and a power-of-two align");
            return TB.getIntegerType(static_cast<uint64_t>(*bits), *al, *sign);
        }
        if (*K == "float") {
            std::optional<StringRef> sem = O->getString("sem");
            std::optional<Align> al = alignOf(*O);
            if (!sem || !al)
                return fail("a float needs sem and a power-of-two align");
            const fltSemantics* S = *sem == "half"   ? &APFloat::IEEEhalf()
                                  : *sem == "bfloat" ? &APFloat::BFloat()
                                  : *sem == "single" ? &APFloat::IEEEsingle()
                                  : *sem == "double" ? &APFloat::IEEEdouble()
                                  : *sem == "x87"    ? &APFloat::x87DoubleExtended()
                                  : *sem == "quad"   ? &APFloat::IEEEquad()
                                                     : nullptr;
            if (!S)
                return fail("unknown float semantics \"" + *sem + "\"");
            return TB.getFloatType(*S, *al);
        }
        if (*K == "complex") {
            std::optional<Align> al = alignOf(*O);
            if (!al)
                return fail("a complex needs a power-of-two align");
            const abi::Type* elem = read(O->get("elem"));
            if (!elem)
                return nullptr;
            return TB.getComplexType(elem, *al);
        }
        if (*K == "vector") {
            std::optional<int64_t> n = O->getInteger("n");
            std::optional<Align> al = alignOf(*O);
            if (!n || *n <= 0 || !al)
                return fail("a vector needs n and a power-of-two align");
            const abi::Type* elem = read(O->get("elem"));
            if (!elem)
                return nullptr;
            return TB.getVectorType(elem,
                ElementCount::getFixed(static_cast<unsigned>(*n)), *al);
        }
        if (*K == "array") {
            std::optional<int64_t> n = O->getInteger("n");
            std::optional<int64_t> bits = O->getInteger("size_bits");
            if (!n || *n < 0 || !bits || *bits < 0)
                return fail("an array needs n and size_bits");
            const abi::Type* elem = read(O->get("elem"));
            if (!elem)
                return nullptr;
            return TB.getArrayType(elem, static_cast<uint64_t>(*n),
                static_cast<uint64_t>(*bits));
        }
        if (*K == "record") {
            std::optional<int64_t> bits = O->getInteger("size_bits");
            std::optional<Align> al = alignOf(*O);
            const json::Array* fs = O->getArray("fields");
            if (!bits || *bits < 0 || !al || !fs)
                return fail("a record needs size_bits, a power-of-two align and fields");
            SmallVector<abi::FieldInfo, 8> fields;
            for (const json::Value& FV : *fs) {
                const json::Object* F = FV.getAsObject();
                std::optional<int64_t> off = F ? F->getInteger("offset_bits") : std::nullopt;
                if (!off || *off < 0)
                    return fail("a field needs offset_bits");
                const abi::Type* ft = read(F->get("type"));
                if (!ft)
                    return nullptr;
                std::optional<int64_t> width = F->getInteger("bit_width");
                fields.push_back(abi::FieldInfo(ft, static_cast<uint64_t>(*off),
                    /*IsBitField=*/width.has_value(),
                    /*BitFieldWidth=*/width ? static_cast<uint64_t>(*width) : 0));
            }
            // C records: trivially copyable, so the CXX-ABI indirect rule
            // (getRecordArgABI) never applies.
            abi::RecordFlags flags = abi::RecordFlags::CanPassInRegisters;
            TypeSize size = TypeSize::getFixed(static_cast<uint64_t>(*bits));
            if (O->getBoolean("union").value_or(false))
                return TB.getUnionType(fields, size, *al, abi::StructPacking::Default, flags);
            return TB.getRecordType(fields, size, *al, abi::StructPacking::Default,
                {}, {}, flags);
        }
        return fail("unknown type kind \"" + *K + "\"");
    }
};

void printType(raw_ostream& os, const abi::Type* T)
{
    if (!T) {
        os << "original";
    } else if (auto* IT = dyn_cast<abi::IntegerType>(T)) {
        os << 'i' << IT->getSizeInBits().getFixedValue();
    } else if (T->isPointer()) {
        os << "ptr";
    } else if (auto* FT = dyn_cast<abi::FloatType>(T)) {
        const fltSemantics* S = FT->getSemantics();
        os << (S == &APFloat::IEEEhalf()            ? "half"
              : S == &APFloat::BFloat()             ? "bfloat"
              : S == &APFloat::IEEEsingle()         ? "float"
              : S == &APFloat::IEEEdouble()         ? "double"
              : S == &APFloat::x87DoubleExtended()  ? "x86_fp80"
              : S == &APFloat::IEEEquad()           ? "fp128"
                                                    : "float?");
    } else if (auto* VT = dyn_cast<abi::VectorType>(T)) {
        os << '<' << VT->getNumElements().getKnownMinValue() << " x ";
        printType(os, VT->getElementType());
        os << '>';
    } else if (auto* AT = dyn_cast<abi::ArrayType>(T)) {
        os << '[' << AT->getNumElements() << " x ";
        printType(os, AT->getElementType());
        os << ']';
    } else if (auto* CT = dyn_cast<abi::ComplexType>(T)) {
        os << "complex ";
        printType(os, CT->getElementType());
    } else if (auto* RT = dyn_cast<abi::RecordType>(T)) {
        os << '{';
        bool first = true;
        for (const abi::FieldInfo& F : RT->getFields()) {
            os << (first ? "" : ",");
            printType(os, F.FieldType);
            first = false;
        }
        os << '}';
    } else if (T->isVoid()) {
        os << "void";
    } else {
        os << '?';
    }
}

// The register class of each eightbyte a directly passed type occupies:
// INTEGER and SSE take a register of their own, SSEUP rides in the SSE
// register before it, X87 is st0 (returns only).
void classesOf(const abi::Type* T, SmallVectorImpl<StringRef>& out)
{
    if (auto* IT = dyn_cast<abi::IntegerType>(T)) {
        for (uint64_t b = 0; b < IT->getSizeInBits().getFixedValue(); b += 64)
            out.push_back("INTEGER");
    } else if (T->isPointer()) {
        out.push_back("INTEGER");
    } else if (auto* FT = dyn_cast<abi::FloatType>(T)) {
        const fltSemantics* S = FT->getSemantics();
        if (S == &APFloat::x87DoubleExtended()) {
            out.push_back("X87");
        } else {
            out.push_back("SSE");
            if (S == &APFloat::IEEEquad())
                out.push_back("SSEUP");
        }
    } else if (auto* VT = dyn_cast<abi::VectorType>(T)) {
        out.push_back("SSE");
        for (uint64_t b = 64; b < VT->getSizeInBits().getKnownMinValue(); b += 64)
            out.push_back("SSEUP");
    } else if (auto* RT = dyn_cast<abi::RecordType>(T)) {
        // createPairType's {lo, hi}, or the {x87, x87} of a complex return.
        for (const abi::FieldInfo& F : RT->getFields())
            classesOf(F.FieldType, out);
    } else {
        out.push_back("?");
    }
}

json::Object describe(const abi::ArgInfo& AI, const abi::Type* original)
{
    json::Object o;
    if (AI.isIgnore()) {
        o["kind"] = "ignore";
    } else if (AI.isIndirect()) {
        o["kind"] = "indirect";
        o["align"] = static_cast<int64_t>(AI.getIndirectAlign().value());
        o["byval"] = AI.getIndirectByVal();
    } else {
        o["kind"] = AI.isExtend() ? "extend" : "direct";
        const abi::Type* C = AI.getCoerceToType();
        SmallVector<StringRef, 4> cls;
        // Lo is NO_CLASS (padding): only the high eightbyte is passed.
        if (AI.getDirectOffset() == 8)
            cls.push_back("NONE");
        classesOf(C ? C : original, cls);
        json::Array arr;
        for (StringRef c : cls)
            arr.push_back(c.str());
        o["classes"] = std::move(arr);
        std::string coerce;
        raw_string_ostream os(coerce);
        printType(os, C);
        o["coerce"] = os.str();
    }
    return o;
}

std::string toText(json::Object o)
{
    std::string text;
    raw_string_ostream os(text);
    os << json::Value(std::move(o));
    return os.str();
}

std::string classify(StringRef request)
{
    Expected<json::Value> req = json::parse(request);
    if (!req)
        return toText(json::Object{{"error", "request is not JSON: " +
                                                 llvm::toString(req.takeError())}});
    const json::Object* root = req->getAsObject();
    const json::Array* fns = root ? root->getArray("functions") : nullptr;
    if (!fns)
        return toText(json::Object{{"error", "request has no \"functions\" array"}});

    BumpPtrAllocator alloc;
    abi::TypeBuilder TB(alloc);
    std::unique_ptr<abi::TargetInfo> TI = abi::createX86_64TargetInfo(
        TB, abi::X86AVXABILevel::None, /*Has64BitPointers=*/true, abi::ABICompatInfo());

    json::Array out;
    for (const json::Value& FV : *fns) {
        const json::Object* F = FV.getAsObject();
        std::optional<StringRef> name = F ? F->getString("name") : std::nullopt;
        json::Object res;
        res["name"] = name ? name->str() : std::string("?");

        TypeReader R{TB, {}};
        const abi::Type* ret = F ? R.read(F->get("ret")) : nullptr;
        SmallVector<const abi::Type*, 8> params;
        const json::Array* ps = F ? F->getArray("params") : nullptr;
        if (!ps && R.error.empty())
            R.error = "function has no \"params\" array";
        if (ps)
            for (const json::Value& P : *ps)
                params.push_back(R.read(&P));
        if (!R.error.empty() || !ret) {
            res["error"] = R.error.empty() ? std::string("no return type") : R.error;
            out.push_back(std::move(res));
            continue;
        }

        std::unique_ptr<abi::FunctionInfo> FI =
            abi::FunctionInfo::create(CallingConv::C, ret, params);
        TI->computeInfo(*FI);

        res["ret"] = describe(FI->getReturnInfo(), ret);

        // Where each argument lands. computeInfo decided every argument with
        // the free-register counts it kept, but an ArgInfo does not say
        // whether a direct one got its registers: an aggregate that did not
        // fit after the integer registers ran out comes back Direct(iN) for
        // the stack. Replaying the same counts (6 INTEGER, 8 SSE, one INTEGER
        // fewer under an sret) recovers it.
        int freeInt = FI->getReturnInfo().isIndirect() ? 5 : 6;
        int freeSSE = 8;
        json::Array args;
        for (const abi::ArgEntry& A : FI->arguments()) {
            json::Object a = describe(A.Info, A.ABIType);
            if (A.Info.isDirect() || A.Info.isExtend()) {
                int needInt = 0, needSSE = 0;
                if (const json::Array* cls = a.getArray("classes"))
                    for (const json::Value& c : *cls) {
                        std::optional<StringRef> s = c.getAsString();
                        needInt += s && *s == "INTEGER";
                        needSSE += s && *s == "SSE";
                    }
                bool fits = needInt <= freeInt && needSSE <= freeSSE;
                if (fits) {
                    freeInt -= needInt;
                    freeSSE -= needSSE;
                }
                a["in_registers"] = fits;
            }
            args.push_back(std::move(a));
        }
        res["params"] = std::move(args);
        out.push_back(std::move(res));
    }

    json::Object answer;
    answer["functions"] = std::move(out);
    return toText(std::move(answer));
}

} // namespace

extern "C" {

// Classify every function in `request` with LLVM's x86-64 SysV classifier.
//
// Request: {"functions": [{"name": N, "ret": T, "params": [T, ...]}, ...]}
// where a type T is one of
//   {"k":"void"}   {"k":"ptr"}
//   {"k":"int", "bits":32, "align":4, "signed":true}
//   {"k":"float", "sem":"half|bfloat|single|double|x87|quad", "align":8}
//   {"k":"complex", "elem":T, "align":8}
//   {"k":"array", "n":N, "size_bits":S, "elem":T}
//   {"k":"vector", "n":N, "elem":T, "align":16}   (a SIMD vector of N lanes)
//   {"k":"record", "size_bits":S, "align":A, "union":false,
//    "fields":[{"offset_bits":O, "type":T, "bit_width":W (bitfields only)}]}
// Records are C records: trivially copyable, so never passed by reference.
//
// Answer: {"functions": [{"name": N, "ret": I, "params": [I, ...]}, ...]},
// or {"error": ...} for a request that is not usable at all; a function that
// cannot be read gets an "error" field instead of "ret"/"params". I is
//   {"kind":"direct"|"extend", "classes":[...], "coerce":"{i64,double}",
//    "in_registers":true (arguments only)}
//   {"kind":"indirect", "align":8, "byval":true}
//   {"kind":"ignore"}
// "classes" holds one entry per eightbyte (INTEGER, SSE, SSEUP, X87, or NONE
// for a low eightbyte that is only padding).
//
// The pointer is into a thread-local buffer the next call overwrites.
const char* jlcs_abi_oracle_sysv(const char* request)
{
    static thread_local std::string answer;
    answer = classify(request ? StringRef(request) : StringRef());
    return answer.c_str();
}

} // extern "C"
