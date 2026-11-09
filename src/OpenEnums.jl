"""
    RepliBuild.OpenEnums

The type a generated wrapper gives every C enum that is not a bitmask.

A C enum is an integer type whose enumerators are named constants: any value
of the underlying type is a value of the enum, declared or not, and APIs use
that. libvterm names its function keys `VTERM_KEY_FUNCTION(n)`, which is
`VTERM_KEY_FUNCTION_0 + n` and declares only the two ends; imgui's key chords
are `ImGuiMod_Ctrl | ImGuiKey_S`. A Julia `@enum` accepts declared values only,
so `VTermKey(257)` was an `ArgumentError` and a terminal built on the wrapper
could not send F1 (LlamaChat's GUI, 2026-11-08).

An `OpenEnum{T}` is a `primitive type` the width of `T` whose constructor takes
any `T`. It subtypes `Base.Enum{T}`, so everything Base defines for an enum
(`Integer(x)`, `Cuint(x)`, passing one to a `ccall`, `isless`, `hash`,
`read`/`write`, `x isa Enum`) is unchanged. A declared value prints exactly as an
`@enum` member does, and an undeclared one as `VTermKey(257)`, never as
`<invalid #257>`. The wrapper supplies each enum's names through `enumerators`.
"""
module OpenEnums

import Core.Intrinsics.bitcast

"""
    OpenEnum{T<:Integer} <: Base.Enum{T}

Supertype of every C enum a RepliBuild wrapper emits (bitmask enums excepted,
which are integer aliases). Its subtypes are `primitive type`s of `T`'s width,
constructible from any value of `T`.
"""
abstract type OpenEnum{T<:Integer} <: Base.Enum{T} end

"""
    enumerators(::Type{E}) -> Tuple{Vararg{Pair{Symbol}}}

`E`'s declared enumerators, `name => value`, in declaration order. A wrapper
defines one method per enum it emits; names are as the wrapper binds them.
"""
enumerators(::Type{<:OpenEnum}) = ()

_basetype(::Type{<:OpenEnum{T}}) where {T<:Integer} = T

(::Type{E})(x::Integer) where {E<:OpenEnum} = bitcast(E, convert(_basetype(E), x))
(::Type{E})(x::E) where {E<:OpenEnum} = x
Base.convert(::Type{E}, x::Integer) where {E<:OpenEnum} = E(x)

# The first declared name of a value, or `nothing`.
function _name(x::E) where {E<:OpenEnum}
    v = Integer(x)
    for (n, val) in enumerators(E)
        val == v && return n
    end
    return nothing
end

# What Base's own Enum printing reads. Defined only while Base has it, so a
# Julia without it still loads RepliBuild.
@static if isdefined(Base.Enums, :namemap)
    function Base.Enums.namemap(::Type{E}) where {E<:OpenEnum}
        T = _basetype(E)
        d = Dict{T,Symbol}()
        for (n, v) in enumerators(E)
            get!(d, T(v), n)                    # the first declared name wins
        end
        return d
    end
end

_undeclared(io::IO, x::E) where {E<:OpenEnum} = (show(io, E); print(io, "("); show(io, Integer(x)); print(io, ")"))

Base.Symbol(x::E) where {E<:OpenEnum} =
    something(_name(x), Symbol(nameof(E), "(", Integer(x), ")"))
Base.print(io::IO, x::OpenEnum) = print(io, Symbol(x))
Base.show(io::IO, x::OpenEnum) =
    _name(x) === nothing ? _undeclared(io, x) : invoke(Base.show, Tuple{IO,Base.Enum}, io, x)

Base.instances(::Type{E}) where {E<:OpenEnum} = Tuple(unique(E(v) for (_, v) in enumerators(E)))
Base.typemin(::Type{E}) where {E<:OpenEnum} = E(minimum(last, enumerators(E)))
Base.typemax(::Type{E}) where {E<:OpenEnum} = E(maximum(last, enumerators(E)))

# A C enum's values combine as integers do: imgui's key chords, masks over an
# enum that is not a bitmask by `_is_bitflag_enum`'s measure.
Base.:|(a::E, b::E) where {E<:OpenEnum} = E(Integer(a) | Integer(b))
Base.:&(a::E, b::E) where {E<:OpenEnum} = E(Integer(a) & Integer(b))
Base.xor(a::E, b::E) where {E<:OpenEnum} = E(xor(Integer(a), Integer(b)))
Base.:~(a::E) where {E<:OpenEnum} = E(~Integer(a))

end # module OpenEnums
