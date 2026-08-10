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

# Unwrap to the original function and differentiate through that, rather than the
# FunctionWrapper dispatch machinery, which fails on mismatched FunctionWrapperTangent
# types when the tuple holds differently-typed wrappers.

@is_primitive MinimalCtx Tuple{<:FunctionWrappersWrapper, Vararg}

# unwrap reaches into FunctionWrapper's internal .obj field directly, which the generic
# getfield rule can't handle since FunctionWrapperTangent isn't a StandardTangentType.
# Also needed for HVP: forward-mode has to trace through this call when differentiating
# the rrule!!/frule!! above.
@is_primitive MinimalCtx Tuple{typeof(unwrap), <:FunctionWrappersWrapper}

# Every rule below builds a fresh zero tangent/dual for the unwrapped function rather than
# threading through any tangent it might really carry. That's exact only if the unwrapped
# function is genuinely non-differentiable (e.g. a stateless closure); if it carries real
# state (e.g. a struct field holding a learnable Matrix), silently zeroing it would give a
# wrong, too-small gradient instead of an error. Fail loud instead.
function _check_wrapped_fn_has_no_tangent(f_orig)
    Mooncake.tangent_type(typeof(f_orig)) === NoTangent && return nothing
    return error(
        "Differentiating through a FunctionWrappersWrapper whose wrapped function " *
            "itself carries differentiable state (tangent_type = " *
            "$(Mooncake.tangent_type(typeof(f_orig)))) is not supported: only " *
            "gradients flowing through the call arguments are tracked here, not " *
            "through the wrapped function's own fields.",
    )
end

function Mooncake.rrule!!(::CoDual{typeof(unwrap)}, fww::CoDual{<:FunctionWrappersWrapper})
    f_orig = unwrap(fww.x)
    _check_wrapped_fn_has_no_tangent(f_orig)
    unwrap_pb(::NoRData) = (NoRData(), NoRData())
    return CoDual(f_orig, fdata(zero_tangent(f_orig))), unwrap_pb
end

function Mooncake.frule!!(::Dual{typeof(unwrap)}, fww::Dual{<:FunctionWrappersWrapper})
    f_orig = unwrap(primal(fww))
    _check_wrapped_fn_has_no_tangent(f_orig)
    return Dual(f_orig, zero_tangent(f_orig))
end

# Cache derived rules by signature, mirroring Mooncake's own DynamicRRule/DynamicFRule
# (src/interpreter/{reverse,forward}_mode.jl): f_orig gets called once per ODE timestep, so
# rebuilding (and re-locking Mooncake's internal rule cache) on every call would be wasted
# work once the signature has stabilised.
const _CALL_RRULE_CACHE = Dict{Any, Any}()
const _CALL_FRULE_CACHE = Dict{Any, Any}()

function Mooncake.rrule!!(
        f::CoDual{<:FunctionWrappersWrapper}, args::Vararg{CoDual},
    )
    f_orig = unwrap(f.x)
    _check_wrapped_fn_has_no_tangent(f_orig)
    # The unwrapped function usually isn't a Mooncake primitive, so build a derived rule.
    sig = Tuple{typeof(f_orig), map(Core.Typeof ∘ Mooncake.primal, args)...}
    rule = get!(() -> Mooncake.build_rrule(sig), _CALL_RRULE_CACHE, sig)
    # fdata turns zero_tangent's NoTangent into the NoFData a derived rule expects.
    f_orig_codual = CoDual(f_orig, fdata(zero_tangent(f_orig)))
    y, pb = rule(f_orig_codual, args...)
    fww_pb(dy) = (NoRData(), Base.tail(pb(dy))...)
    return y, fww_pb
end

function Mooncake.frule!!(
        f::Dual{<:FunctionWrappersWrapper}, args::Vararg{Dual},
    )
    f_orig = unwrap(primal(f))
    _check_wrapped_fn_has_no_tangent(f_orig)
    # Mirrors the rrule!! above, but builds a derived forward-mode rule instead.
    sig = Tuple{typeof(f_orig), map(Core.Typeof ∘ primal, args)...}
    rule = get!(() -> build_frule(get_interpreter(ForwardMode), sig), _CALL_FRULE_CACHE, sig)
    f_orig_dual = Dual(f_orig, zero_tangent(f_orig))
    return rule(f_orig_dual, args...)
end

# The wrapper itself carries no derivative info; the wrapped function does, handled above.
Mooncake.tangent_type(::Type{<:FunctionWrappersWrapper}) = NoTangent

end
