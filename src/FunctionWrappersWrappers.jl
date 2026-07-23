module FunctionWrappersWrappers

using FunctionWrappers: FunctionWrappers
using SciMLPublic: @public
import TruncatedStacktraces

export FunctionWrappersWrapper, unwrap, wrapped_signatures, wrapped_return_types
export NoCache, SingleCache, DictCache
export Strict, AllowAll, AllowNonIsBits

# ============================================================================
# Cache modes: control how fallback FunctionWrappers are cached
# ============================================================================
abstract type AbstractCacheMode end

"""
    NoCache() -> NoCache

Disable fallback wrapper caching.

Fallback calls use dynamic dispatch through the original function each time.

# Returns
- `NoCache`: A cache mode for `FunctionWrappersWrapper`.
"""
struct NoCache <: AbstractCacheMode end

"""
    SingleCache() -> SingleCache

Cache one fallback `FunctionWrappers.FunctionWrapper` for the most recent argument tuple
type.

Repeated fallback calls with the same argument tuple type reuse the cached
wrapper. Calls with a different tuple type replace the cache entry.

# Returns
- `SingleCache`: A cache mode for `FunctionWrappersWrapper`.
"""
struct SingleCache <: AbstractCacheMode end

"""
    DictCache() -> DictCache

Cache fallback `FunctionWrapper`s in a dictionary keyed by argument tuple type.

Use this mode when fallback calls are expected for several different
non-isbits argument tuple types.

# Returns
- `DictCache`: A cache mode for `FunctionWrappersWrapper`.
"""
struct DictCache <: AbstractCacheMode end

# ============================================================================
# Fallback policies: control when fallback is allowed
# ============================================================================
abstract type AbstractFallbackPolicy end

"""
    Strict() -> Strict

Disable fallback calls.

If no wrapped signature matches a call, `FunctionWrappersWrapper` throws a
`NoFunctionWrapperFoundError`.

# Returns
- `Strict`: A fallback policy for `FunctionWrappersWrapper`.
"""
struct Strict <: AbstractFallbackPolicy end

"""
    AllowAll() -> AllowAll

Always call the original function when no wrapped signature matches.

# Returns
- `AllowAll`: A fallback policy for `FunctionWrappersWrapper`.
"""
struct AllowAll <: AbstractFallbackPolicy end

"""
    AllowNonIsBits() -> AllowNonIsBits

Call the original function only when a mismatched argument tuple contains
non-isbits element types.

This policy keeps isbits type mismatches strict, while still supporting
non-isbits values such as `BigFloat` and tracer types.

# Returns
- `AllowNonIsBits`: A fallback policy for `FunctionWrappersWrapper`.
"""
struct AllowNonIsBits <: AbstractFallbackPolicy end

# ============================================================================
# Cache storage types
# ============================================================================
"""
    NoCacheStorage() -> NoCacheStorage

Storage marker for the `NoCache()` fallback mode.

`NoCacheStorage` carries no state. Use this type when constructing or
inspecting a `FunctionWrappersWrapper` whose fallback path should dispatch
through the original function without caching generated
`FunctionWrappers.FunctionWrapper`s.

# Examples

```julia
using FunctionWrappersWrappers

wrapper = FunctionWrappersWrapper(sin, (Tuple{Float64},), (Float64,);
    cache = NoCache(), policy = AllowAll())
storage = FunctionWrappersWrappers.NoCacheStorage()
```

# Returns
- `NoCacheStorage`: Empty storage marker for uncached fallback calls.
"""
struct NoCacheStorage end

"""
    SingleCacheStorage() -> SingleCacheStorage

Storage for the `SingleCache()` fallback mode.

`SingleCacheStorage` stores one fallback `FunctionWrappers.FunctionWrapper`
for the most recent argument tuple type. Use this type when constructing or
inspecting a `FunctionWrappersWrapper` that should reuse a single cached
fallback wrapper before replacing it on a different fallback argument tuple
type.

# Examples

```julia
using FunctionWrappersWrappers

wrapper = FunctionWrappersWrapper(sin, (Tuple{Float64},), (Float64,);
    cache = SingleCache(), policy = AllowAll())
storage = FunctionWrappersWrappers.SingleCacheStorage()
```

# Fields
- `cached`: The cached fallback wrapper, or `nothing` before fallback is used.

# Returns
- `SingleCacheStorage`: Empty storage for one cached fallback wrapper.
"""
mutable struct SingleCacheStorage
    cached::Any  # Union{Nothing, FunctionWrapper}
    SingleCacheStorage() = new(nothing)
end

"""
    DictCacheStorage() -> DictCacheStorage

Storage for the `DictCache()` fallback mode.

`DictCacheStorage` stores fallback `FunctionWrappers.FunctionWrapper`s keyed by
argument tuple type. Use this type when constructing or inspecting a
`FunctionWrappersWrapper` whose fallback path should reuse wrappers for several
different fallback argument tuple types.

# Examples

```julia
using FunctionWrappersWrappers

wrapper = FunctionWrappersWrapper(sin, (Tuple{Float64},), (Float64,);
    cache = DictCache(), policy = AllowAll())
storage = FunctionWrappersWrappers.DictCacheStorage()
```

# Fields
- `cache`: Mapping from fallback argument tuple type to cached fallback wrapper.

# Returns
- `DictCacheStorage`: Empty dictionary-backed storage for fallback wrappers.
"""
struct DictCacheStorage
    cache::Dict{DataType, Any}
    DictCacheStorage() = new(Dict{DataType, Any}())
end

@public NoCacheStorage, SingleCacheStorage, DictCacheStorage

_make_cache_storage(::NoCache) = NoCacheStorage()
_make_cache_storage(::SingleCache) = SingleCacheStorage()
_make_cache_storage(::DictCache) = DictCacheStorage()

# ============================================================================
# Main type
# ============================================================================

"""
    FunctionWrappersWrapper{FW, P, CS}

A callable wrapper around one or more `FunctionWrappers.FunctionWrapper`s.

Calls dispatch to the wrapped signature matching the argument tuple type. If no
signature matches, fallback behavior is determined by policy `P` and cache
storage `CS`.

# Fields
- `fw::FW`: Tuple of `FunctionWrappers.FunctionWrapper`s, in matching order.
- `cache_storage::CS`: Storage used to cache fallback wrappers.

# Type Parameters
- `FW`: Tuple type of the wrapped `FunctionWrapper`s.
- `P`: Fallback policy type, such as `Strict`, `AllowAll`, or `AllowNonIsBits`.
- `CS`: Cache storage type selected from the cache mode.
"""
struct FunctionWrappersWrapper{FW, P, CS}
    fw::FW
    cache_storage::CS
    function FunctionWrappersWrapper{FW, P, CS}(
            fw::FW, cs::CS
        ) where {FW, P, CS}
        return new{FW, P, CS}(fw, cs)
    end
end

TruncatedStacktraces.@truncate_stacktrace FunctionWrappersWrapper

"""
    FunctionWrappersWrapper{FW, P, CS}(f) -> FunctionWrappersWrapper{FW, P, CS}

Create a `FunctionWrappersWrapper` with explicit type parameters.

# Arguments
- `f`: Function to wrap.

# Type Parameters
- `FW`: Tuple type of the wrapped `FunctionWrapper`s.
- `P`: Fallback policy type, such as `Strict`, `AllowAll`, or `AllowNonIsBits`.
- `CS`: Cache storage type selected from the cache mode.

# Returns
- `FunctionWrappersWrapper{FW, P, CS}`: A callable wrapper around `f`.
"""
function FunctionWrappersWrapper{FW, P, CS}(f) where {K, FW <: NTuple{K, Any}, P, CS}
    fw = ntuple(i -> FW.parameters[i](f), Val(K))
    cs = CS()
    return FunctionWrappersWrapper{FW, P, CS}(fw, cs)
end

"""
    FunctionWrappersWrapper(f, argtypes, rettypes;
        cache = SingleCache(), policy = AllowNonIsBits()) -> FunctionWrappersWrapper

Create a callable wrapper for `f` over the given argument and return types.

# Arguments
- `f`: Function to wrap.
- `argtypes`: Tuple of argument tuple types, such as `(Tuple{Float64, Float64},)`.
- `rettypes`: Tuple of return types corresponding to `argtypes`, such as `(Float64,)`.

# Keywords
- `cache::AbstractCacheMode = SingleCache()`: Fallback cache mode. `NoCache()` performs
  no caching, `SingleCache()` caches the most recent fallback signature, and `DictCache()`
  caches each fallback signature.
- `policy::AbstractFallbackPolicy = AllowNonIsBits()`: Fallback policy. `Strict()` throws
  for every unmatched signature, `AllowAll()` always calls `f`, and `AllowNonIsBits()` only
  calls `f` when an unmatched argument type is non-isbits.

# Returns
- `FunctionWrappersWrapper`: A callable wrapper around `f`.

# Examples
```jldoctest
julia> fw = FunctionWrappersWrapper(+, (Tuple{Float64, Float64},), (Float64,));

julia> fw(1.0, 2.0)
3.0
```
"""
function FunctionWrappersWrapper(
        f::F, argtypes::Tuple{Vararg{Any, K}}, rettypes::Tuple{Vararg{Type, K}};
        cache::AbstractCacheMode = SingleCache(),
        policy::AbstractFallbackPolicy = AllowNonIsBits()
    ) where {F, K}
    fwt = map(argtypes, rettypes) do A, R
        FunctionWrappers.FunctionWrapper{R, A}(f)
    end
    cs = _make_cache_storage(cache)
    return FunctionWrappersWrapper{typeof(fwt), typeof(policy), typeof(cs)}(fwt, cs)
end

"""
    FunctionWrappersWrapper(fw::Tuple{Vararg{FunctionWrappers.FunctionWrapper}}; cache=SingleCache(), policy=AllowNonIsBits())

Create a callable wrapper from already-constructed `FunctionWrapper`s.

Use this when the `FunctionWrapper`s must be built by the caller, for instance to keep
construction inferrable: the `argtypes`/`rettypes` method takes its types as values, so the
wrapper type is only inferred when those values constant-propagate. Building `fw` with the
signatures bound as type parameters and passing it here keeps `typeof(fww)` concrete.

# Arguments
- `fw`: Tuple of `FunctionWrappers.FunctionWrapper`s, tried in order on call.

# Keyword Arguments
- `cache`: Fallback cache mode. Defaults to `SingleCache()`.
- `policy`: Fallback policy. Defaults to `AllowNonIsBits()`.

# Returns
- `FunctionWrappersWrapper`: A callable wrapper around `fw`.

# Examples
```jldoctest
julia> using FunctionWrappers: FunctionWrapper

julia> fw = (FunctionWrapper{Float64, Tuple{Float64, Float64}}(+),);

julia> fww = FunctionWrappersWrapper(fw);

julia> fww(1.0, 2.0)
3.0
```
"""
function FunctionWrappersWrapper(
        fw::Tuple{Vararg{FunctionWrappers.FunctionWrapper}};
        cache::AbstractCacheMode = SingleCache(),
        policy::AbstractFallbackPolicy = AllowNonIsBits()
    )
    cs = _make_cache_storage(cache)
    return FunctionWrappersWrapper{typeof(fw), typeof(policy), typeof(cs)}(fw, cs)
end

Base.convert(::Type{T}, obj) where {T <: FunctionWrappersWrapper} = T(obj)
Base.convert(::Type{T}, obj::T) where {T <: FunctionWrappersWrapper} = obj

# ============================================================================
# Call dispatch — entry point
# ============================================================================

function (fww::FunctionWrappersWrapper{FW, P, CS})(
        args::Vararg{Any, K}
    ) where {FW, K, P, CS}
    return _call(fww.fw, args, fww)
end

# Match path: try each FunctionWrapper in order
function _call(
        fw::Tuple{FunctionWrappers.FunctionWrapper{R, A}, Vararg},
        arg::A, fww::FunctionWrappersWrapper
    ) where {R, A}
    return first(fw)(arg...)
end
function _call(
        fw::Tuple{FunctionWrappers.FunctionWrapper{R, A1}, Vararg},
        arg::A2, fww::FunctionWrappersWrapper
    ) where {R, A1, A2}
    return _call(Base.tail(fw), arg, fww)
end

# ============================================================================
# Fallback — Strict: always error
# ============================================================================

const NO_FUNCTIONWRAPPER_FOUND_MESSAGE = "No matching function wrapper was found!"

struct NoFunctionWrapperFoundError <: Exception end

function Base.showerror(io::IO, e::NoFunctionWrapperFoundError)
    return print(io, NO_FUNCTIONWRAPPER_FOUND_MESSAGE)
end

function _call(::Tuple{}, arg, fww::FunctionWrappersWrapper{<:Any, Strict})
    throw(NoFunctionWrapperFoundError())
end

# ============================================================================
# Fallback — AllowAll: always fall back
# ============================================================================

function _call(::Tuple{}, arg, fww::FunctionWrappersWrapper{<:Any, AllowAll})
    return _fallback(arg, fww)
end

# ============================================================================
# Fallback — AllowNonIsBits: fall back only for non-isbits arg types
# ============================================================================

function _call(
        ::Tuple{}, arg::A, fww::FunctionWrappersWrapper{<:Any, AllowNonIsBits}
    ) where {A}
    if _has_non_isbits_args(A)
        return _fallback(arg, fww)
    end
    throw(NoFunctionWrapperFoundError())
end

@generated function _has_non_isbits_args(::Type{T}) where {T <: Tuple}
    checks = []
    for P in T.parameters
        if P <: AbstractArray
            push!(checks, :(!(isbitstype($(eltype(P))))))
        else
            push!(checks, :(!(isbitstype($P))))
        end
    end
    isempty(checks) && return :(false)
    return Expr(:||, checks...)
end

# ============================================================================
# Fallback execution — dispatch on cache storage type
# ============================================================================

# --- NoCache: direct dynamic dispatch every time ---
function _fallback(arg, fww::FunctionWrappersWrapper{<:Any, <:Any, NoCacheStorage})
    return first(fww.fw).obj[](arg...)
end

# --- SingleCache: cache one FunctionWrapper for the last arg types ---
function _fallback(
        arg::A, fww::FunctionWrappersWrapper{<:Any, <:Any, SingleCacheStorage}
    ) where {A}
    cached = fww.cache_storage.cached
    if cached isa FunctionWrappers.FunctionWrapper{Any, A}
        return cached(arg...)
    end
    f = first(fww.fw).obj[]
    new_fw = FunctionWrappers.FunctionWrapper{Any, A}(f)
    fww.cache_storage.cached = new_fw
    return new_fw(arg...)
end

# --- DictCache: cache FunctionWrappers keyed by arg type ---
function _fallback(
        arg::A, fww::FunctionWrappersWrapper{<:Any, <:Any, DictCacheStorage}
    ) where {A}
    cached = get(fww.cache_storage.cache, A, nothing)
    if cached isa FunctionWrappers.FunctionWrapper{Any, A}
        return cached(arg...)
    end
    f = first(fww.fw).obj[]
    new_fw = FunctionWrappers.FunctionWrapper{Any, A}(f)
    fww.cache_storage.cache[A] = new_fw
    return new_fw(arg...)
end

# ============================================================================
# Introspection
# ============================================================================

"""
    unwrap(fww::FunctionWrappersWrapper) -> Function

Return the original function that was wrapped.

# Arguments
- `fww`: Wrapper to inspect.

# Returns
- The function captured by `fww`.
"""
unwrap(fww::FunctionWrappersWrapper) = first(fww.fw).obj[]

"""
    wrapped_signatures(fww::FunctionWrappersWrapper) -> Tuple

Return a tuple of the argument type signatures that the wrapper can dispatch on.

# Arguments
- `fww`: Wrapper to inspect.

# Returns
- A tuple of argument tuple types stored in `fww`.
"""
function wrapped_signatures(fww::FunctionWrappersWrapper)
    return map(fw -> typeof(fw).parameters[2], fww.fw)
end

"""
    wrapped_return_types(fww::FunctionWrappersWrapper) -> Tuple

Return a tuple of the return types for each wrapped function signature.

# Arguments
- `fww`: Wrapper to inspect.

# Returns
- A tuple of return types corresponding to `wrapped_signatures(fww)`.
"""
function wrapped_return_types(fww::FunctionWrappersWrapper)
    return map(fw -> typeof(fw).parameters[1], fww.fw)
end

# ============================================================================
# Precompilation
# ============================================================================

using PrecompileTools: @compile_workload, @setup_workload

@setup_workload begin
    @compile_workload begin
        fw_binary = FunctionWrappersWrapper(
            +,
            (Tuple{Float64, Float64}, Tuple{Int, Int}),
            (Float64, Int)
        )
        fw_binary(1.0, 2.0)
        fw_binary(1, 2)

        fw_unary = FunctionWrappersWrapper(
            abs,
            (Tuple{Float64}, Tuple{Int}),
            (Float64, Int)
        )
        fw_unary(1.0)
        fw_unary(1)

        unwrap(fw_unary)
        wrapped_signatures(fw_binary)
        wrapped_return_types(fw_binary)
    end
end

end
