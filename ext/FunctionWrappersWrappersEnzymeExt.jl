module FunctionWrappersWrappersEnzymeExt

using FunctionWrappersWrappers
using FunctionWrappersWrappers: SingleCacheStorage, DictCacheStorage, NoCacheStorage
using Enzyme
using EnzymeCore
using EnzymeCore.EnzymeRules

# =============================================================================
# Mark cache-storage types as inactive
# =============================================================================
# `SingleCacheStorage` and `DictCacheStorage` are mutable / contain a `Dict`,
# and their cache-miss branches write to that storage. Without these
# declarations Enzyme conservatively treats any closure that *might* touch a
# `FunctionWrappersWrapper` (e.g. via `remake(prob; p = …)` capturing the
# problem in scope) as potentially writing to the wrapper's cache, and
# refuses to prove the captured argument read-only. The cache values are
# `FunctionWrapper`s used purely for dispatch / dynamic call speedup; they
# never hold derivative data.
EnzymeCore.EnzymeRules.inactive_type(::Type{<:SingleCacheStorage}) = true
EnzymeCore.EnzymeRules.inactive_type(::Type{<:DictCacheStorage}) = true
EnzymeCore.EnzymeRules.inactive_type(::Type{NoCacheStorage}) = true

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
#
# `func` is `Annotation{<:FunctionWrappersWrapper}` rather than
# `Const{<:FunctionWrappersWrapper}` so that callers passing
# `Duplicated{<:FunctionWrappersWrapper}` also dispatch here.  Enzyme drives
# the rule that way when the outer `autodiff` call is differentiating through
# a closure that carries an FWW (e.g. NonlinearSolve + SciMLSensitivity, see
# SciML/FunctionWrappersWrappers.jl#48).  The FWW struct itself only carries
# `FunctionWrapper`s plus cache storage — none of those fields have a
# meaningful tangent — so the function shadow is ignored and the inner
# `Enzyme.autodiff` call uses `Const(f_orig)`.
function EnzymeRules.forward(
        ::EnzymeRules.FwdConfig{false, true, W, RuntimeActivity, StrongZero},
        func::EnzymeCore.Annotation{<:FunctionWrappersWrapper},
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
        # Enzyme returns the batch shadow as an `AnonymousStruct` — a
        # `NamedTuple{(:1, :2, …), NTuple{W, T}}` (see
        # `Enzyme.Compiler.AnonymousStruct` in `Enzyme/src/compiler/utils.jl`).
        # Convert to a plain tuple so the rule's return matches the
        # `BatchDuplicated` shadow contract Enzyme expects from a forward rule.
        return Tuple(shadow_result[1])::NTuple{W, T}
    end
end

# Both primal and shadow (ForwardWithPrimal mode)
function EnzymeRules.forward(
        ::EnzymeRules.FwdConfig{true, true, W, RuntimeActivity, StrongZero},
        func::EnzymeCore.Annotation{<:FunctionWrappersWrapper},
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
        # See the comment on the {false, true} rule — `shadow_result[1]` is a
        # NamedTuple, not an NTuple.
        shadows = Tuple(shadow_result[1])::NTuple{W, T}
        return BatchDuplicated(primal, shadows)
    end
end

# Primal only (Const return type) — width-independent
function EnzymeRules.forward(
        ::EnzymeRules.FwdConfig{true, false, W, RuntimeActivity, StrongZero},
        func::EnzymeCore.Annotation{<:FunctionWrappersWrapper},
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
        func::EnzymeCore.Annotation{<:FunctionWrappersWrapper},
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

# A `ReverseSplitNoPrimal` mode reflecting the rule's RevConfig runtime-activity
# / strong-zero flags, so the split thunk we delegate to inherits the caller's
# outer settings.
@inline function _rev_split_mode(config::EnzymeRules.RevConfig)
    mode = Enzyme.ReverseSplitNoPrimal
    EnzymeRules.runtime_activity(config) && (mode = Enzyme.set_runtime_activity(mode))
    EnzymeRules.strong_zero(config) && (mode = Enzyme.set_strong_zero(mode))
    return mode
end

# Build one slot of a reverse rule's return tuple: `nothing` for every
# non-Active arg (their gradients accumulate in-place), and the concrete
# gradient for each `Active` arg.  Dispatching on the annotation *type* keeps
# the resulting tuple exactly typed (e.g. `Tuple{Nothing, Nothing, Float64}`)
# even though the raw `autodiff`/thunk return is `Any`-typed inside the rule —
# Enzyme rejects a union-typed return.  `g` is the matching entry of that raw
# per-argument gradient tuple.
@inline _revslot(::EnzymeCore.Active{T}, g) where {T} = convert(T, g)::T
@inline _revslot(::EnzymeCore.Annotation, @nospecialize(g)) = nothing

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::EnzymeCore.Annotation{<:FunctionWrappersWrapper},
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
# return). Delegate to a split reverse-mode thunk on the *unwrapped* function so
# Enzyme differentiates it directly, exactly as it would if the function were
# never wrapped. We forward the rule's `overwritten` flags as the thunk's
# `ModifiedBetween`, so Enzyme's own tape caches any argument the caller mutates
# before the reverse pass (the ODE-integrator pattern: `wf(du, u, p, t)` then an
# in-place step on `u`). The forward thunk's tape and the reverse thunk are
# stashed for the reverse rule.
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::EnzymeCore.Annotation{<:FunctionWrappersWrapper},
        RT::Type{<:EnzymeCore.Const},
        args::Vararg{EnzymeCore.Annotation, N}
    ) where {N}
    f_orig = unwrap(func.val)
    mode = Enzyme.ReverseSplitModified(
        _rev_split_mode(config), Val(EnzymeRules.overwritten(config))
    )
    fwd_thunk, rev_thunk = Enzyme.autodiff_thunk(
        mode, EnzymeCore.Const{typeof(f_orig)}, EnzymeCore.Const, map(typeof, args)...
    )
    tape = fwd_thunk(EnzymeCore.Const(f_orig), args...)[1]
    return EnzymeRules.AugmentedReturn(nothing, nothing, (tape, rev_thunk))
end

# Duplicated / BatchDuplicated return: record the primal so that reverse has
# it available when propagating dret through the arguments.
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::EnzymeCore.Annotation{<:FunctionWrappersWrapper},
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
        func::EnzymeCore.Annotation{<:FunctionWrappersWrapper},
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
        push!(
            exprs, quote
                fwd = Enzyme.autodiff(mode, Const(f_orig), Duplicated{$T}, $(dups...))
                $Ti(fwd[1] * dret_val)::$Ti
            end
        )
    end
    return Expr(:tuple, exprs...)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::EnzymeCore.Annotation{<:FunctionWrappersWrapper},
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
        func::EnzymeCore.Annotation{<:FunctionWrappersWrapper},
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
# accumulate into any `Duplicated` arg shadow buffers (the SciML IIP pattern).
#
# The reverse thunk stashed by `augmented_primal` reads its cached tape (which
# already captured any `ModifiedBetween` args) and accumulates gradients into
# the `Duplicated` arg shadows in place; its `[1]` return is the per-argument
# gradient tuple.  We rebuild that through `map(_revslot, …)` so it is exactly
# typed (Enzyme rejects a union-typed `Tuple{Union{Nothing,Float64},…}`), which
# also makes `Active` args (e.g. `t` in a time-dependent IIP rhs) correct rather
# than zeroed.
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::EnzymeCore.Annotation{<:FunctionWrappersWrapper},
        dret::Type{<:EnzymeCore.Const},
        tape,
        args::Vararg{EnzymeCore.Annotation, N}
    ) where {N}
    f_orig = unwrap(func.val)
    tape_data, rev_thunk = tape
    raw = rev_thunk(EnzymeCore.Const(f_orig), args..., tape_data)[1]::NTuple{N, Any}
    # `map` over tuples specialises per element, so dispatching `_revslot` on
    # each arg's concrete annotation type yields an exactly-typed result.
    return map(_revslot, args, raw)
end

end
