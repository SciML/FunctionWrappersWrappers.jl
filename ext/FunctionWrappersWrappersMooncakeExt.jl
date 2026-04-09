module FunctionWrappersWrappersMooncakeExt

using FunctionWrappersWrappers
import Mooncake
using Mooncake: @is_primitive, MinimalCtx, CoDual, NoRData, zero_tangent, NoTangent, fdata

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
    # Build a derived rule for calling the unwrapped function with these arg types.
    # We can't use rrule!! directly since the unwrapped function (e.g. SciMLBase.Void)
    # is generally not a Mooncake primitive — it needs a derived (compiled) rule.
    sig = Tuple{typeof(f_orig), map(Core.Typeof ∘ Mooncake.primal, args)...}
    rule = Mooncake.build_rrule(sig)
    # Use fdata to get the correct tangent component for the CoDual — zero_tangent
    # returns NoTangent for singleton callables but derived rules expect NoFData.
    f_orig_codual = CoDual(f_orig, fdata(zero_tangent(f_orig)))
    y, pb = rule(f_orig_codual, args...)
    fww_pb(dy) = (NoRData(), Base.tail(pb(dy))...)
    return y, fww_pb
end

# FunctionWrappersWrapper is not differentiable data itself — the wrapped function
# is what carries the derivative information, and we handle that in the rrule above.
Mooncake.tangent_type(::Type{<:FunctionWrappersWrapper}) = NoTangent

end
