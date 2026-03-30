module FunctionWrappersWrappers

using FunctionWrappers
import TruncatedStacktraces

export FunctionWrappersWrapper, unwrap, wrapped_signatures, wrapped_return_types
export NoCache, SingleCache, DictCache
export Strict, AllowAll, AllowNonIsBits

# ============================================================================
# Cache modes: control how fallback FunctionWrappers are cached
# ============================================================================
abstract type AbstractCacheMode end

"""
    NoCache()

No caching — every fallback call goes through dynamic dispatch (`obj[](arg...)`),
incurring 1 allocation per call.
"""
struct NoCache <: AbstractCacheMode end

"""
    SingleCache()

Cache a single `FunctionWrapper` for the last-seen argument types. After the first
fallback call, subsequent calls with the same types are zero-allocation. If called with
different types, the cache is replaced (1 alloc on miss). This is the recommended default.
"""
struct SingleCache <: AbstractCacheMode end

"""
    DictCache()

Cache `FunctionWrapper`s in a `Dict` keyed by argument type. Handles multiple
non-isbits types without thrashing. Slightly higher lookup overhead than `SingleCache`.
"""
struct DictCache <: AbstractCacheMode end

# ============================================================================
# Fallback policies: control when fallback is allowed
# ============================================================================
abstract type AbstractFallbackPolicy end

"""
    Strict()

Never fall back — throw `NoFunctionWrapperFoundError` if no wrapper matches.
"""
struct Strict <: AbstractFallbackPolicy end

"""
    AllowAll()

Always fall back to the original function when no wrapper matches.
"""
struct AllowAll <: AbstractFallbackPolicy end

"""
    AllowNonIsBits()

Fall back only when argument types contain non-isbits elements (e.g., `BigFloat`,
`SparseConnectivityTracer` types). Throws `NoFunctionWrapperFoundError` for isbits
type mismatches (e.g., `Float32` when `Float64` was expected), which catches bugs.
This is the recommended default.
"""
struct AllowNonIsBits <: AbstractFallbackPolicy end

# ============================================================================
# Cache storage types
# ============================================================================
struct NoCacheStorage end
mutable struct SingleCacheStorage
    cached::Any  # Union{Nothing, FunctionWrapper}
    SingleCacheStorage() = new(nothing)
end
struct DictCacheStorage
    cache::Dict{DataType, Any}
    DictCacheStorage() = new(Dict{DataType, Any}())
end

_make_cache_storage(::NoCache) = NoCacheStorage()
_make_cache_storage(::SingleCache) = SingleCacheStorage()
_make_cache_storage(::DictCache) = DictCacheStorage()

# ============================================================================
# Main type
# ============================================================================

"""
    FunctionWrappersWrapper{FW, P, CS}

A wrapper around a tuple of `FunctionWrapper`s that dispatches calls to the
matching wrapper based on argument types. When no wrapper matches, behavior is
controlled by the fallback policy `P` and cache mode `CS`.

# Type parameters
- `FW`: Tuple type of `FunctionWrapper`s
- `P`: Fallback policy (`Strict`, `AllowAll`, or `AllowNonIsBits`)
- `CS`: Cache storage type (`NoCacheStorage`, `SingleCacheStorage`, `DictCacheStorage`)
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
    FunctionWrappersWrapper(f, argtypes, rettypes; cache=SingleCache(), policy=AllowNonIsBits())

Create a `FunctionWrappersWrapper` with configurable fallback behavior.

# Arguments
- `f`: The function to wrap
- `argtypes`: Tuple of argument type signatures (e.g., `(Tuple{Float64, Float64},)`)
- `rettypes`: Tuple of return types (e.g., `(Float64,)`)

# Keywords
- `cache`: Cache mode for fallback path — `NoCache()`, `SingleCache()` (default), or `DictCache()`
- `policy`: Fallback policy — `Strict()`, `AllowAll()`, or `AllowNonIsBits()` (default)
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
    unwrap(fww::FunctionWrappersWrapper)

Return the original function that was wrapped.
"""
unwrap(fww::FunctionWrappersWrapper) = first(fww.fw).obj[]

"""
    wrapped_signatures(fww::FunctionWrappersWrapper)

Return a tuple of the argument type signatures that the wrapper can dispatch on.
"""
function wrapped_signatures(fww::FunctionWrappersWrapper)
    return map(fw -> typeof(fw).parameters[2], fww.fw)
end

"""
    wrapped_return_types(fww::FunctionWrappersWrapper)

Return a tuple of the return types for each wrapped function signature.
"""
function wrapped_return_types(fww::FunctionWrappersWrapper)
    return map(fw -> typeof(fw).parameters[1], fww.fw)
end

# ============================================================================
# Precompilation
# ============================================================================

using PrecompileTools

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
