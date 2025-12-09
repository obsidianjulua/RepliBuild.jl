// JLCSTypes.cpp (very small stub showing storage)
#include "JLCSTypes.h"

#define GET_TYPEDEF_CLASSES
#include "JLCSTypes.cpp.inc"

using namespace mlir::jlir::jlcs;

// forward declare storage class cons/equals/hash
struct JLCStructTypeStorage : public mlir::TypeStorage {
    using KeyTy = std::pair<uint64_t, uint64_t>; // example: size, alignment

    JLCStructTypeStorage(uint64_t size, uint64_t align)
        : size(size)
        , align(align)
    {
    }

    bool operator==(const KeyTy& key) const
    {
        return key.first == size && key.second == align;
    }

    static llvm::hash_code hashKey(const KeyTy& key)
    {
        return llvm::hash_combine(key.first, key.second);
    }

    static JLCStructTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy& key)
    {
        return new (allocator.allocate<JLCStructTypeStorage>()) JLCStructTypeStorage(key.first, key.second);
    }

    uint64_t size;
    uint64_t align;
};
