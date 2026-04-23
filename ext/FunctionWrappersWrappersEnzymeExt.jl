module FunctionWrappersWrappersEnzymeExt

using FunctionWrappersWrappers
using Enzyme
using EnzymeCore
using EnzymeCore.EnzymeRules

# =============================================================================
# Helper: build a Forward mode from FwdConfig flags
# =============================================================================
# The outer caller may invoke `Enzyme.autodiff(set_runtime_activity(Forward), …)`
# or `set_strong_zero(Forward)` or `ForwardWithPrimal`.  Those settings flow
# into the `EnzymeRules.FwdConfig{NeedsPrimal, NeedsShadow, Width,
# RuntimeActivity, StrongZero}` type parameters of the rule's first argument.
# Before this fix the rules hard-coded plain `Forward` in their inner
# `Enzyme.autodiff` delegation, which dropped both `RuntimeActivity` and
# `StrongZero` — breaking users who need `set_runtime_activity(Forward)` to
# avoid `EnzymeRuntimeActivityError` inside the wrapped function (the SciML
# `Rosenbrock23(autodiff = AutoEnzyme(set_runtime_activity(Forward)))` path
# on an in-place time-dependent RHS; see OrdinaryDiffEq.jl PR #3518).
#
# `_fwd_mode(needs_primal, RuntimeActivity, StrongZero)` returns the
# `ForwardMode` matching the outer config so the delegated call inherits
# those flags.
@inline function _fwd_mode(
    ::Val{NeedsPrimal}, ::Val{RuntimeActivity}, ::Val{StrongZero}
) where {NeedsPrimal, RuntimeActivity, StrongZero}
    mode = NeedsPrimal ? ForwardWithPrimal : Forward
    RuntimeActivity && (mode = Enzyme.set_runtime_activity(mode))
    StrongZero && (mode = Enzyme.set_strong_zero(mode))
    return mode
end

# =============================================================================
# Forward mode rules — generalized to arbitrary batch width W
# =============================================================================

# Shadow only (Forward mode, no primal)
function EnzymeRules.forward(
    ::EnzymeRules.FwdConfig{false, true, W, RuntimeActivity, StrongZero},
    func::EnzymeCore.Const{<:FunctionWrappersWrapper},
    RT::Type{<:EnzymeCore.Annotation{T}},
    args::Vararg{EnzymeCore.Annotation, N}
) where {T, W, N, RuntimeActivity, StrongZero}
    f_orig = unwrap(func.val)
    mode = _fwd_mode(Val(false), Val(RuntimeActivity), Val(StrongZero))
    if W == 1
        shadow_result = Enzyme.autodiff(mode, Const(f_orig), Duplicated{T}, args...)
        return shadow_result[1]::T
    else
        shadow_result = Enzyme.autodiff(mode, Const(f_orig), BatchDuplicated{T, W}, args...)
        return shadow_result[1]::NTuple{W, T}
    end
end

# Both primal and shadow (ForwardWithPrimal mode)
function EnzymeRules.forward(
    ::EnzymeRules.FwdConfig{true, true, W, RuntimeActivity, StrongZero},
    func::EnzymeCore.Const{<:FunctionWrappersWrapper},
    RT::Type{<:EnzymeCore.Annotation{T}},
    args::Vararg{EnzymeCore.Annotation, N}
) where {T, W, N, RuntimeActivity, StrongZero}
    f_orig = unwrap(func.val)
    pargs = ntuple(i -> args[i].val, Val(N))
    primal = f_orig(pargs...)::T
    # Use plain Forward (not ForwardWithPrimal) here — we already have the
    # primal above, and `Duplicated{T}` / `BatchDuplicated{T,W}` as the RT
    # annotation asks only for the shadow.
    mode = _fwd_mode(Val(false), Val(RuntimeActivity), Val(StrongZero))
    if W == 1
        shadow_result = Enzyme.autodiff(mode, Const(f_orig), Duplicated{T}, args...)
        shadow = shadow_result[1]::T
        return Duplicated(primal, shadow)
    else
        shadow_result = Enzyme.autodiff(mode, Const(f_orig), BatchDuplicated{T, W}, args...)
        shadows = shadow_result[1]::NTuple{W, T}
        return BatchDuplicated(primal, shadows)
    end
end

# Primal only (Const return type) — width-independent
function EnzymeRules.forward(
    ::EnzymeRules.FwdConfig{true, false, W, RuntimeActivity, StrongZero},
    func::EnzymeCore.Const{<:FunctionWrappersWrapper},
    RT::Type{<:EnzymeCore.Annotation},
    args::Vararg{EnzymeCore.Annotation, N}
) where {W, N, RuntimeActivity, StrongZero}
    f_orig = unwrap(func.val)
    pargs = ntuple(i -> args[i].val, Val(N))
    return f_orig(pargs...)
end

# Neither primal nor shadow requested in the RETURN.  Enzyme dispatches on
# this combo for IIP functions (Const return type) where the caller still
# needs primal and shadow propagation through `Duplicated` args — e.g. SciML
# solvers calling an IIP RHS via `AutoEnzyme(…, function_annotation = Const)`.
# The previous revision ran `f_orig(pargs...)` by hand; that mutated the
# primal IIP buffer but left `Duplicated` shadow buffers untouched, giving
# trivial Jacobians and blowing up Rodas4/5/Veldd4 error tolerances 4–9
# orders of magnitude in OrdinaryDiffEq.jl v7.  Delegate to `Enzyme.autodiff`
# on the unwrapped function with a Const return annotation so the Duplicated
# arg shadows are propagated correctly and no return is produced.
#
# IMPORTANT: forward the `RuntimeActivity` and `StrongZero` flags from the
# outer config into the delegated `Enzyme.autodiff` call.  Prior to this
# fix the rule hard-coded `Forward`, silently dropping
# `set_runtime_activity(Forward)` on the way down into `f_orig`. 
function EnzymeRules.forward(
    ::EnzymeRules.FwdConfig{false, false, W, RuntimeActivity, StrongZero},
    func::EnzymeCore.Const{<:FunctionWrappersWrapper},
    RT::Type{<:EnzymeCore.Annotation},
    args::Vararg{EnzymeCore.Annotation, N}
) where {W, N, RuntimeActivity, StrongZero}
    f_orig = unwrap(func.val)
    mode = _fwd_mode(Val(false), Val(RuntimeActivity), Val(StrongZero))
    Enzyme.autodiff(mode, Const(f_orig), Const, args...)
    return nothing
end

# =============================================================================
# Reverse mode rules
# =============================================================================

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    func::EnzymeCore.Const{<:FunctionWrappersWrapper},
    RT::Type{<:EnzymeCore.Active{T}},
    args::Vararg{EnzymeCore.Annotation, N}
) where {T, N}
    f_orig = unwrap(func.val)
    pargs = ntuple(i -> args[i].val, Val(N))
    result = f_orig(pargs...)::T

    if EnzymeRules.needs_primal(config)
        return EnzymeRules.AugmentedReturn(result, nothing, nothing)
    else
        return EnzymeRules.AugmentedReturn(nothing, nothing, nothing)
    end
end

# Const return (e.g. IIP functions returning Nothing, or any non-differentiated
# return). Just run the primal for its side effects; no tape is needed because
# the reverse pass has nothing to propagate back from the return.
function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    func::EnzymeCore.Const{<:FunctionWrappersWrapper},
    RT::Type{<:EnzymeCore.Const},
    args::Vararg{EnzymeCore.Annotation, N}
) where {N}
    f_orig = unwrap(func.val)
    pargs = ntuple(i -> args[i].val, Val(N))
    f_orig(pargs...)
    return EnzymeRules.AugmentedReturn(nothing, nothing, nothing)
end

# Duplicated / BatchDuplicated return: record the primal so that reverse has
# it available when propagating dret through the arguments.
function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    func::EnzymeCore.Const{<:FunctionWrappersWrapper},
    RT::Type{<:EnzymeCore.Duplicated{T}},
    args::Vararg{EnzymeCore.Annotation, N}
) where {T, N}
    f_orig = unwrap(func.val)
    pargs = ntuple(i -> args[i].val, Val(N))
    primal = f_orig(pargs...)::T
    if EnzymeRules.needs_primal(config)
        return EnzymeRules.AugmentedReturn(primal, zero(primal), nothing)
    else
        return EnzymeRules.AugmentedReturn(nothing, zero(primal), nothing)
    end
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    func::EnzymeCore.Const{<:FunctionWrappersWrapper},
    RT::Type{<:EnzymeCore.BatchDuplicated{T, W}},
    args::Vararg{EnzymeCore.Annotation, N}
) where {T, W, N}
    f_orig = unwrap(func.val)
    pargs = ntuple(i -> args[i].val, Val(N))
    primal = f_orig(pargs...)::T
    shadows = ntuple(_ -> zero(primal), Val(W))
    if EnzymeRules.needs_primal(config)
        return EnzymeRules.AugmentedReturn(primal, shadows, nothing)
    else
        return EnzymeRules.AugmentedReturn(nothing, shadows, nothing)
    end
end

# Helper: build a Forward mode reflecting a RevConfig's runtime_activity /
# strong_zero flags so the internal forward-mode delegation inside reverse
# rules inherits the user's outer config.
@inline function _fwd_mode_from_rev(config::EnzymeRules.RevConfig)
    mode = Forward
    EnzymeRules.runtime_activity(config) && (mode = Enzyme.set_runtime_activity(mode))
    EnzymeRules.strong_zero(config) && (mode = Enzyme.set_strong_zero(mode))
    return mode
end

# Varargs reverse: compute each partial via forward-mode AD on the unwrapped
# function, then scale by dret. This avoids type-inference issues that arise
# from calling autodiff(Reverse, Const{Any}(...), ...).
@generated function _fww_reverse_grads(
    mode, f_orig, dret_val::T, args::Vararg{EnzymeCore.Active, N}
) where {T, N}
    # Build forward-mode calls for each partial derivative
    exprs = []
    for i in 1:N
        seeds = [j == i ? :(one(eltype(typeof(args[$j])))) : :(zero(eltype(typeof(args[$j])))) for j in 1:N]
        dups = [:(Duplicated(args[$j].val, $(seeds[j]))) for j in 1:N]
        Ti = :(eltype(typeof(args[$i])))
        push!(exprs, quote
            fwd = Enzyme.autodiff(mode, Const(f_orig), Duplicated{$T}, $(dups...))
            $Ti(fwd[1] * dret_val)::$Ti
        end)
    end
    return Expr(:tuple, exprs...)
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    func::EnzymeCore.Const{<:FunctionWrappersWrapper},
    dret::EnzymeCore.Active{T},
    tape,
    args::Vararg{EnzymeCore.Active, N}
) where {T, N}
    f_orig = unwrap(func.val)
    return _fww_reverse_grads(_fwd_mode_from_rev(config), f_orig, dret.val, args...)
end

# Handle mixed Active/Const args: return nothing for Const, gradient for Active
function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    func::EnzymeCore.Const{<:FunctionWrappersWrapper},
    dret::EnzymeCore.Active,
    tape,
    args::Vararg{EnzymeCore.Annotation, N}
) where {N}
    f_orig = unwrap(func.val)
    dret_val = dret.val
    mode = _fwd_mode_from_rev(config)
    return ntuple(Val(N)) do i
        if args[i] isa EnzymeCore.Const
            nothing
        else
            # Use forward-mode to compute partial derivative
            dup_args = ntuple(Val(N)) do j
                if j == i
                    Duplicated(args[j].val, one(eltype(typeof(args[j]))))
                else
                    Duplicated(args[j].val, zero(eltype(typeof(args[j]))))
                end
            end
            fwd = Enzyme.autodiff(mode, Const(f_orig), Duplicated, dup_args...)
            fwd[1] * dret_val
        end
    end
end

# Const return — Enzyme passes the RT as a `Type{<:Const}` to `reverse`, not
# as an instance.  Delegate the reverse pass to
# `Enzyme.autodiff(Reverse, Const(f_orig), Const, args...)` so gradients
# accumulate into any `Duplicated` arg shadow buffers (the SciML IIP
# pattern).  Simply returning `nothing` left Duplicated shadows at zero.
#
# Per Enzyme's rule return-type protocol, `Active` args require a concrete
# scalar gradient (not `nothing`).  Under a `Const` return there is no
# gradient source, so Active arg gradients are zero.  `Duplicated` /
# `BatchDuplicated` args return `nothing` because their gradients are
# accumulated in-place by the `Enzyme.autodiff(Reverse, …)` call above.
function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    func::EnzymeCore.Const{<:FunctionWrappersWrapper},
    dret::Type{<:EnzymeCore.Const},
    tape,
    args::Vararg{EnzymeCore.Annotation, N}
) where {N}
    f_orig = unwrap(func.val)
    # Only worth invoking Enzyme.autodiff when at least one arg is
    # Duplicated/BatchDuplicated — otherwise there's nothing to accumulate.
    if any(a -> a isa EnzymeCore.Duplicated || a isa EnzymeCore.BatchDuplicated, args)
        Enzyme.autodiff(Reverse, Const(f_orig), Const, args...)
    end
    return ntuple(Val(N)) do i
        if args[i] isa EnzymeCore.Active
            zero(eltype(typeof(args[i])))
        else
            nothing
        end
    end
end

end
