module FunctionWrappersWrappers

using FunctionWrappers
import TruncatedStacktraces

export FunctionWrappersWrapper, unwrap, wrapped_signatures, wrapped_return_types

struct FunctionWrappersWrapper{FW, FB}
    fw::FW
end

TruncatedStacktraces.@truncate_stacktrace FunctionWrappersWrapper

function (fww::FunctionWrappersWrapper{FW, FB})(args::Vararg{Any, K}) where {FW, K, FB}
    _call(fww.fw, args, fww)
end

function _call(fw::Tuple{FunctionWrappers.FunctionWrapper{R, A}, Vararg},
        arg::A, fww::FunctionWrappersWrapper) where {R, A}
    first(fw)(arg...)
end
function _call(fw::Tuple{FunctionWrappers.FunctionWrapper{R, A1}, Vararg},
        arg::A2, fww::FunctionWrappersWrapper) where {R, A1, A2}
    _call(Base.tail(fw), arg, fww)
end

const NO_FUNCTIONWRAPPER_FOUND_MESSAGE = "No matching function wrapper was found!"

struct NoFunctionWrapperFoundError <: Exception end

function Base.showerror(io::IO, e::NoFunctionWrapperFoundError)
    print(io, NO_FUNCTIONWRAPPER_FOUND_MESSAGE)
end

function _call(::Tuple{}, arg, fww::FunctionWrappersWrapper{<:Any, false})
    throw(NoFunctionWrapperFoundError())
end
function _call(::Tuple{}, arg, fww::FunctionWrappersWrapper{<:Any, true})
    first(fww.fw).obj[](arg...)
end

function FunctionWrappersWrapper(
        f::F, argtypes::Tuple{Vararg{Any, K}}, rettypes::Tuple{Vararg{DataType, K}},
        fallback::Val{FB} = Val{false}()) where {F, K, FB}
    fwt = map(argtypes, rettypes) do A, R
        FunctionWrappers.FunctionWrapper{R, A}(f)
    end
    FunctionWrappersWrapper{typeof(fwt), FB}(fwt)
end

"""
    unwrap(fww::FunctionWrappersWrapper)

Return the original function that was wrapped. This is useful for debugging
wrapped functions - you can use the returned function with debugging tools
like Debugger.jl or Infiltrator.jl.

# Example

```julia
using FunctionWrappersWrappers

# Create a wrapped function
fww = FunctionWrappersWrapper(sin, (Tuple{Float64},), (Float64,))

# Get the original function for debugging
f = unwrap(fww)  # Returns sin

# Now you can debug with Debugger.jl:
# using Debugger
# @enter f(0.5)

# Or use Infiltrator.jl in your original function definition
```

See also: [`wrapped_signatures`](@ref), [`wrapped_return_types`](@ref)
"""
unwrap(fww::FunctionWrappersWrapper) = first(fww.fw).obj[]

"""
    wrapped_signatures(fww::FunctionWrappersWrapper)

Return a tuple of the argument type signatures that the `FunctionWrappersWrapper`
can dispatch on. Each element is a `Tuple` type representing the argument types.

# Example

```julia
using FunctionWrappersWrappers

fww = FunctionWrappersWrapper(+, (Tuple{Float64, Float64}, Tuple{Int, Int}), (Float64, Int))
wrapped_signatures(fww)  # Returns (Tuple{Float64, Float64}, Tuple{Int, Int})
```

See also: [`unwrap`](@ref), [`wrapped_return_types`](@ref)
"""
function wrapped_signatures(fww::FunctionWrappersWrapper)
    map(fw -> typeof(fw).parameters[2], fww.fw)
end

"""
    wrapped_return_types(fww::FunctionWrappersWrapper)

Return a tuple of the return types for each wrapped function signature.

# Example

```julia
using FunctionWrappersWrappers

fww = FunctionWrappersWrapper(+, (Tuple{Float64, Float64}, Tuple{Int, Int}), (Float64, Int))
wrapped_return_types(fww)  # Returns (Float64, Int64)
```

See also: [`unwrap`](@ref), [`wrapped_signatures`](@ref)
"""
function wrapped_return_types(fww::FunctionWrappersWrapper)
    map(fw -> typeof(fw).parameters[1], fww.fw)
end

end
