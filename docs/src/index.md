```@meta
CurrentModule = FunctionWrappersWrappers
```

# FunctionWrappersWrappers

`FunctionWrappersWrappers` provides a callable, type-stable wrapper for a finite set of
function signatures. Calls outside that set can be rejected or delegated to the original
function with a configurable cache.

## Getting Started

Construct a wrapper by listing the argument tuple types and corresponding return types that
should use `FunctionWrappers.FunctionWrapper` dispatch:

```julia
using FunctionWrappersWrappers

wrapped_add = FunctionWrappersWrapper(
    +,
    (Tuple{Float64, Float64},),
    (Float64,);
    cache = SingleCache(),
    policy = AllowNonIsBits(),
)

wrapped_add(1.0, 2.0) # 3.0 through the wrapped signature
wrapped_add(big(1), big(2)) # 3 as a cached fallback call
```

Use `Strict()` when unmatched calls must fail, `AllowAll()` when every unmatched call should
delegate to the original function, or `AllowNonIsBits()` to allow only non-isbits fallback
argument types. Use `unwrap`, `wrapped_signatures`, and `wrapped_return_types` to inspect a
wrapper without depending on its fields.

## API Reference

### Wrapper and Introspection

```@docs
FunctionWrappersWrapper
unwrap
wrapped_signatures
wrapped_return_types
```

### Fallback Configuration

```@docs
NoCache
SingleCache
DictCache
Strict
AllowAll
AllowNonIsBits
```

### Cache Storage

```@docs
NoCacheStorage
SingleCacheStorage
DictCacheStorage
```

```@index
```
