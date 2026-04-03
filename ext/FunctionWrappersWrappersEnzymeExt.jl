module FunctionWrappersWrappersEnzymeExt

using FunctionWrappersWrappers
using Enzyme
using EnzymeCore
using EnzymeCore.EnzymeRules

# =============================================================================
# Forward mode rules
# =============================================================================

# Shadow only (Forward mode, no primal)
function EnzymeRules.forward(
    ::EnzymeRules.FwdConfig{false, true, 1, RuntimeActivity, StrongZero},
    func::EnzymeCore.Const{<:FunctionWrappersWrapper},
    RT::Type{<:EnzymeCore.Annotation{T}},
    args::Vararg{EnzymeCore.Annotation, N}
) where {T, N, RuntimeActivity, StrongZero}
    f_orig = unwrap(func.val)
    shadow_result = Enzyme.autodiff(Forward, Const(f_orig), Duplicated{T}, args...)
    return shadow_result[1]::T
end

# Both primal and shadow (ForwardWithPrimal mode)
function EnzymeRules.forward(
    ::EnzymeRules.FwdConfig{true, true, 1, RuntimeActivity, StrongZero},
    func::EnzymeCore.Const{<:FunctionWrappersWrapper},
    RT::Type{<:EnzymeCore.Annotation{T}},
    args::Vararg{EnzymeCore.Annotation, N}
) where {T, N, RuntimeActivity, StrongZero}
    f_orig = unwrap(func.val)
    pargs = ntuple(i -> args[i].val, Val(N))
    primal = f_orig(pargs...)::T
    shadow_result = Enzyme.autodiff(Forward, Const(f_orig), Duplicated{T}, args...)
    shadow = shadow_result[1]::T
    return Duplicated(primal, shadow)
end

# Primal only (Const return type)
function EnzymeRules.forward(
    ::EnzymeRules.FwdConfig{true, false, 1, RuntimeActivity, StrongZero},
    func::EnzymeCore.Const{<:FunctionWrappersWrapper},
    RT::Type{<:EnzymeCore.Annotation},
    args::Vararg{EnzymeCore.Annotation, N}
) where {N, RuntimeActivity, StrongZero}
    f_orig = unwrap(func.val)
    pargs = ntuple(i -> args[i].val, Val(N))
    return f_orig(pargs...)
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

end
