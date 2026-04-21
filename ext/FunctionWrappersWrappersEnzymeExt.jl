module FunctionWrappersWrappersEnzymeExt

using FunctionWrappersWrappers
using Enzyme
using EnzymeCore
using EnzymeCore.EnzymeRules

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
    if W == 1
        shadow_result = Enzyme.autodiff(Forward, Const(f_orig), Duplicated{T}, args...)
        return shadow_result[1]::T
    else
        shadow_result = Enzyme.autodiff(Forward, Const(f_orig), BatchDuplicated{T, W}, args...)
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
    if W == 1
        shadow_result = Enzyme.autodiff(Forward, Const(f_orig), Duplicated{T}, args...)
        shadow = shadow_result[1]::T
        return Duplicated(primal, shadow)
    else
        shadow_result = Enzyme.autodiff(Forward, Const(f_orig), BatchDuplicated{T, W}, args...)
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

# Neither primal nor shadow requested — Enzyme asks for this combo with Const
# return-type annotations where the caller only needs the side effects of the
# primal invocation (e.g. mutating an IIP RHS in SciML's solver path).  No rule
# previously matched this case, so dispatch fell through to Enzyme's default
# path which tried to differentiate through the raw FunctionWrappersWrapper
# and failed with `MethodError: no method matching forward(…)` when the wrapper
# only held plain-Float64 signatures.  Just run the primal and return nothing.
function EnzymeRules.forward(
    ::EnzymeRules.FwdConfig{false, false, W, RuntimeActivity, StrongZero},
    func::EnzymeCore.Const{<:FunctionWrappersWrapper},
    RT::Type{<:EnzymeCore.Annotation},
    args::Vararg{EnzymeCore.Annotation, N}
) where {W, N, RuntimeActivity, StrongZero}
    f_orig = unwrap(func.val)
    pargs = ntuple(i -> args[i].val, Val(N))
    f_orig(pargs...)
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

# Varargs reverse: compute each partial via forward-mode AD on the unwrapped
# function, then scale by dret. This avoids type-inference issues that arise
# from calling autodiff(Reverse, Const{Any}(...), ...).
@generated function _fww_reverse_grads(
    f_orig, dret_val::T, args::Vararg{EnzymeCore.Active, N}
) where {T, N}
    # Build forward-mode calls for each partial derivative
    exprs = []
    for i in 1:N
        seeds = [j == i ? :(one(eltype(typeof(args[$j])))) : :(zero(eltype(typeof(args[$j])))) for j in 1:N]
        dups = [:(Duplicated(args[$j].val, $(seeds[j]))) for j in 1:N]
        Ti = :(eltype(typeof(args[$i])))
        push!(exprs, quote
            fwd = Enzyme.autodiff(Forward, Const(f_orig), Duplicated{$T}, $(dups...))
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
    return _fww_reverse_grads(f_orig, dret.val, args...)
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
            fwd = Enzyme.autodiff(Forward, Const(f_orig), Duplicated, dup_args...)
            fwd[1] * dret_val
        end
    end
end

# Const return (no derivative to propagate from the return) — uniform Active args.
function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    func::EnzymeCore.Const{<:FunctionWrappersWrapper},
    dret::EnzymeCore.Const,
    tape,
    args::Vararg{EnzymeCore.Active, N}
) where {N}
    return ntuple(_ -> nothing, Val(N))
end

# Const return — mixed Active/Const args.
function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    func::EnzymeCore.Const{<:FunctionWrappersWrapper},
    dret::EnzymeCore.Const,
    tape,
    args::Vararg{EnzymeCore.Annotation, N}
) where {N}
    return ntuple(_ -> nothing, Val(N))
end

end
