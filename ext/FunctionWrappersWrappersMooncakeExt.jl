module FunctionWrappersWrappersMooncakeExt

using FunctionWrappersWrappers
import Mooncake
using Mooncake: @is_primitive, MinimalCtx, CoDual, NoRData, zero_tangent, NoTangent

# Make calling a FunctionWrappersWrapper a Mooncake primitive.
# Instead of differentiating through the FunctionWrapper dispatch machinery
# (which fails because the tuple of differently-typed FunctionWrappers produces
# incompatible FunctionWrapperTangent types), unwrap to the original function
# and differentiate through that directly.

@is_primitive MinimalCtx Tuple{<:FunctionWrappersWrapper, Vararg}

function Mooncake.rrule!!(
        f::CoDual{<:FunctionWrappersWrapper}, args::Vararg{CoDual},
    )
    f_orig = unwrap(f.x)
    f_orig_codual = CoDual(f_orig, zero_tangent(f_orig))
    y, pb = Mooncake.rrule!!(f_orig_codual, args...)
    fww_pb(dy) = (NoRData(), Mooncake.Base.tail(pb(dy))...)
    return y, fww_pb
end

# FunctionWrappersWrapper is not differentiable data itself — the wrapped function
# is what carries the derivative information, and we handle that in the rrule above.
Mooncake.tangent_type(::Type{<:FunctionWrappersWrapper}) = NoTangent

end
