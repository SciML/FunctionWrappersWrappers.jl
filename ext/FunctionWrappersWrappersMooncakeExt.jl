module FunctionWrappersWrappersMooncakeExt

using FunctionWrappersWrappers
import Mooncake
using Mooncake:
    @is_primitive,
    MinimalCtx,
    CoDual,
    Dual,
    NoRData,
    zero_tangent,
    NoTangent,
    fdata,
    primal,
    build_frule,
    get_interpreter,
    ForwardMode

# Make calling a FunctionWrappersWrapper a Mooncake primitive.
# Instead of differentiating through the FunctionWrapper dispatch machinery
# (which fails because the tuple of differently-typed FunctionWrappers produces
# incompatible FunctionWrapperTangent types), unwrap to the original function
# and differentiate through that directly.

@is_primitive MinimalCtx Tuple{<:FunctionWrappersWrapper, Vararg}

# `unwrap` reaches directly into a nested `FunctionWrapper`'s internal `.obj` field
# (bypassing the call/construct interface entirely), which the generic `lgetfield` rule
# cannot handle here: it's constrained to `StandardTangentType`/`StandardFDataType`, and
# `FunctionWrapper`'s custom `FunctionWrapperTangent` isn't one. This matters for
# forward-over-reverse (HVP): computing a derivative of `rrule!!`/`frule!!` above requires
# tracing their own bodies, including this `unwrap` call, in forward mode. Since we already
# always rebuild a fresh `zero_tangent` for the unwrapped function in the rules above
# (never actually reading any tangent carried by the wrapper itself, consistent with
# `FunctionWrappersWrapper`'s own tangent being `NoTangent`), treating `unwrap` as an
# opaque primitive with the same "fresh zero tangent" behaviour is exact, not an
# approximation.
@is_primitive MinimalCtx Tuple{typeof(unwrap), <:FunctionWrappersWrapper}

function Mooncake.rrule!!(::CoDual{typeof(unwrap)}, fww::CoDual{<:FunctionWrappersWrapper})
    f_orig = unwrap(fww.x)
    unwrap_pb(::NoRData) = (NoRData(), NoRData())
    return CoDual(f_orig, fdata(zero_tangent(f_orig))), unwrap_pb
end

function Mooncake.frule!!(::Dual{typeof(unwrap)}, fww::Dual{<:FunctionWrappersWrapper})
    f_orig = unwrap(primal(fww))
    return Dual(f_orig, zero_tangent(f_orig))
end

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

function Mooncake.frule!!(
        f::Dual{<:FunctionWrappersWrapper}, args::Vararg{Dual},
    )
    f_orig = unwrap(primal(f))
    # Mirrors the rrule!! above: build a derived forward-mode rule for the unwrapped
    # function with these arg types, rather than differentiating through the
    # FunctionWrappersWrapper dispatch machinery itself.
    sig = Tuple{typeof(f_orig), map(Core.Typeof ∘ primal, args)...}
    rule = build_frule(get_interpreter(ForwardMode), sig)
    f_orig_dual = Dual(f_orig, zero_tangent(f_orig))
    return rule(f_orig_dual, args...)
end

# FunctionWrappersWrapper is not differentiable data itself — the wrapped function
# is what carries the derivative information, and we handle that in the rrule above.
Mooncake.tangent_type(::Type{<:FunctionWrappersWrapper}) = NoTangent

# For the same reason, `prepare_pullback_cache`/`value_and_pullback!!`'s generic
# "no pointers or aliased mutable state reachable from the output" safety check
# (SciML/SciMLSensitivity.jl#1424) has nothing to protect against here either: its
# `.fw` field holds raw-`Ptr`-carrying `FunctionWrapper`s and its `.cache_storage`
# field is mutable, shared cache state that can legitimately be aliased across
# multiple places in a returned value (e.g. an `ODESolution`) — but neither is ever
# reached via generic field access during real differentiation, only via the
# dedicated `rrule!!`/`unwrap` path above. Stop the check's recursion here.
Mooncake.__exclude_unsupported_output_internal!(::FunctionWrappersWrapper, ::Set{UInt}) = nothing

end
