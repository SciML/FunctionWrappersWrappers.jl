using FunctionWrappersWrappers
using Enzyme
using EnzymeCore
using Test

@testset "Enzyme forward mode" begin
    f(x) = x^2
    fww = FunctionWrappersWrapper(f, (Tuple{Float64},), (Float64,))

    # Forward mode (shadow only)
    result = Enzyme.autodiff(Forward, Const(fww), Duplicated, Duplicated(3.0, 1.0))
    @test result[1] ≈ 6.0

    # ForwardWithPrimal (both primal and shadow)
    result = Enzyme.autodiff(ForwardWithPrimal, Const(fww), Duplicated, Duplicated(3.0, 1.0))
    @test result[1] ≈ 6.0   # shadow
    @test result[2] ≈ 9.0   # primal
end

@testset "Enzyme reverse mode - single arg" begin
    f(x) = x^2
    fww = FunctionWrappersWrapper(f, (Tuple{Float64},), (Float64,))

    # Reverse mode
    result = Enzyme.autodiff(Reverse, Const(fww), Active, Active(3.0))
    @test result[1][1] ≈ 6.0

    # ReverseWithPrimal
    result = Enzyme.autodiff(ReverseWithPrimal, Const(fww), Active, Active(3.0))
    @test result[1][1] ≈ 6.0  # gradient
    @test result[2] ≈ 9.0      # primal
end

@testset "Enzyme reverse mode - multi arg" begin
    g(x, y) = x * y + x^2
    fww = FunctionWrappersWrapper(g, (Tuple{Float64, Float64},), (Float64,))

    # g(x,y) = x*y + x^2 → ∂g/∂x = y + 2x, ∂g/∂y = x
    result = Enzyme.autodiff(Reverse, Const(fww), Active, Active(3.0), Active(4.0))
    @test result[1][1] ≈ 10.0  # ∂g/∂x at (3,4) = 4 + 6
    @test result[1][2] ≈ 3.0   # ∂g/∂y at (3,4) = 3
end

@testset "Enzyme with trig functions" begin
    fww_sin = FunctionWrappersWrapper(sin, (Tuple{Float64},), (Float64,))

    # sin'(x) = cos(x)
    result = Enzyme.autodiff(Forward, Const(fww_sin), Duplicated, Duplicated(1.0, 1.0))
    @test result[1] ≈ cos(1.0)

    result = Enzyme.autodiff(Reverse, Const(fww_sin), Active, Active(1.0))
    @test result[1][1] ≈ cos(1.0)
end

@testset "Enzyme batch forward mode (width > 1)" begin
    f(x) = x^2
    fww = FunctionWrappersWrapper(f, (Tuple{Float64},), (Float64,))

    # Batch width = 2: compute derivatives for two tangent directions simultaneously.
    # f(x) = x^2 → f'(x) = 2x; at x=3.0 with tangents (1.0, 2.0) → shadows (6.0, 12.0)
    result = Enzyme.autodiff(
        Forward, Const(fww), BatchDuplicated,
        BatchDuplicated(3.0, (1.0, 2.0))
    )
    shadows = result[1]
    @test shadows[1] ≈ 6.0   # f'(3) * 1.0
    @test shadows[2] ≈ 12.0  # f'(3) * 2.0

    # ForwardWithPrimal, batch width = 2
    result_wp = Enzyme.autodiff(
        ForwardWithPrimal, Const(fww), BatchDuplicated,
        BatchDuplicated(3.0, (1.0, 2.0))
    )
    @test result_wp[1][1] ≈ 6.0   # shadow 1
    @test result_wp[1][2] ≈ 12.0  # shadow 2
    @test result_wp[2] ≈ 9.0      # primal f(3) = 9
end

@testset "Enzyme batch forward rule return type is NTuple, not NamedTuple" begin
    # Regression test for the typeassert bug: the inner
    # `Enzyme.autodiff(Forward, …, BatchDuplicated{T,W}, …)` returns the
    # batch shadow wrapped in `Enzyme.Compiler.AnonymousStruct` — a
    # `NamedTuple{(:1, :2, …), NTuple{W, T}}`.  The rule must convert it
    # to a plain `NTuple{W, T}` before returning, otherwise the
    # `::NTuple{W, T}` typeassert fires and surfaces as:
    #   TypeError: in typeassert, expected Tuple{Float64, Float64},
    #   got a value of type @NamedTuple{1::Float64, 2::Float64}
    #
    # The outer `Enzyme.autodiff` testset above doesn't catch this on its
    # own because the outer call ALSO wraps the result in
    # `AnonymousStruct`, and `shadow[1] / shadow[2]` indexing works on
    # both `NamedTuple` and `Tuple`.  Call `EnzymeRules.forward`
    # directly so we observe the rule's actual return value and can
    # assert its concrete type.
    f(x) = x^2
    fww = FunctionWrappersWrapper(f, (Tuple{Float64},), (Float64,))

    # {NeedsPrimal=false, NeedsShadow=true, W=2, RuntimeActivity=false,
    #  StrongZero=false} — the shadow-only batch branch.
    config_shadow = EnzymeCore.EnzymeRules.FwdConfig{false, true, 2, false, false}()
    shadow = EnzymeCore.EnzymeRules.forward(
        config_shadow, Const(fww), EnzymeCore.BatchDuplicated{Float64, 2},
        BatchDuplicated(3.0, (1.0, 2.0))
    )
    @test shadow isa NTuple{2, Float64}
    @test !(shadow isa NamedTuple)
    @test shadow == (6.0, 12.0)

    # {NeedsPrimal=true, NeedsShadow=true, W=2, …} — ForwardWithPrimal
    # batch branch.  Same conversion bug existed on this path.
    config_primal = EnzymeCore.EnzymeRules.FwdConfig{true, true, 2, false, false}()
    result = EnzymeCore.EnzymeRules.forward(
        config_primal, Const(fww), EnzymeCore.BatchDuplicated{Float64, 2},
        BatchDuplicated(3.0, (1.0, 2.0))
    )
    @test result isa BatchDuplicated
    @test result.val ≈ 9.0
    @test result.dval isa NTuple{2, Float64}
    @test !(result.dval isa NamedTuple)
    @test result.dval == (6.0, 12.0)

    # Confirm the conversion generalises to W > 2.
    config_w3 = EnzymeCore.EnzymeRules.FwdConfig{false, true, 3, false, false}()
    shadow3 = EnzymeCore.EnzymeRules.forward(
        config_w3, Const(fww), EnzymeCore.BatchDuplicated{Float64, 3},
        BatchDuplicated(3.0, (1.0, 2.0, 4.0))
    )
    @test shadow3 isa NTuple{3, Float64}
    @test shadow3 == (6.0, 12.0, 24.0)
end

@testset "Enzyme forward mode, Const return (IIP, no return-shadow)" begin
    # Covers EnzymeRules.FwdConfig{false, false, W, ...} — Enzyme dispatches on
    # this combo for IIP functions with a Const return type where the caller
    # needs primal + shadow propagation via Duplicated args only (no return
    # value to shadow).  Reproduces the SciML/OrdinaryDiffEq.jl v7 Downstream
    # regression where this call previously produced:
    #   - without any rule:      MethodError: no method matching forward(…)
    #   - with a primal-only rule: trivial (zero) arg shadows, wrong Jacobians
    #     (Rodas4/5/Veldd4 errors 4–9 orders of magnitude above tolerance).
    # The rule must delegate to `Enzyme.autodiff` on the unwrapped function
    # so Duplicated arg shadows propagate correctly.
    f!(du, u) = (du[1] = -u[1]^2; nothing)
    fww = FunctionWrappersWrapper(
        f!, (Tuple{Vector{Float64}, Vector{Float64}},), (Nothing,)
    )

    du = [0.0]
    u = [3.0]
    du_shadow = [0.0]
    u_shadow = [1.0]  # seed: ∂/∂u[1] = 1

    config = EnzymeCore.EnzymeRules.FwdConfig{false, false, 1, false, false}()
    ret = EnzymeCore.EnzymeRules.forward(
        config, Const(fww), EnzymeCore.Const{Nothing},
        Duplicated(du, du_shadow), Duplicated(u, u_shadow)
    )
    @test ret === nothing
    # Primal side-effect: du[1] = -u[1]^2 = -9
    @test du[1] ≈ -9.0
    # Shadow propagation: ∂du[1]/∂u[1] * u_shadow[1] = -2*u[1]*1 = -6
    @test du_shadow[1] ≈ -6.0
end

@testset "Enzyme reverse mode, Const return — augmented_primal runs primal" begin
    # Mirrors the forward {false, false} case on the reverse side. Augmented
    # primal runs the wrapped function for its side effects and tapes a snapshot
    # of the call-time argument values (so a later mutation by the caller can't
    # make the reverse pass differentiate about the wrong state).  Reverse
    # returns `nothing`/zero per arg since there is no return derivative to
    # propagate.
    counter = Ref(0)
    g(x, y) = (counter[] += 1; x + y)  # returns Float64 (ignored via Const RT)
    fww = FunctionWrappersWrapper(g, (Tuple{Float64, Float64},), (Float64,))

    # Construct a concrete RevConfig. Fields:
    # (NeedsPrimal, NeedsShadow, Width, Overwritten, RuntimeActivity, StrongZero)
    # Overwritten is indexed (func, args...) — here (func, x, y).  Mark only `x`
    # as overwritten so we can check the rule snapshots exactly that arg.
    rconfig = EnzymeRules.RevConfig{false, false, 1, (false, true, false), false, false}()

    counter[] = 0
    aug = EnzymeRules.augmented_primal(
        rconfig, Const(fww), EnzymeCore.Const{Float64},
        Active(3.0), Active(4.0)
    )
    @test counter[] == 1                       # primal ran exactly once
    @test aug.primal === nothing               # NeedsPrimal == false
    @test aug.shadow === nothing
    @test aug.tape == (3.0, nothing)           # only the overwritten arg (x) is snapshotted

    # Reverse step — dret is Const (passed as TYPE not instance in reverse
    # rules).  Enzyme's rule protocol requires concrete gradients for Active
    # args; under a Const return they're zero (no gradient source).
    grads = EnzymeRules.reverse(
        rconfig, Const(fww), EnzymeCore.Const{Float64},
        aug.tape, Active(3.0), Active(4.0)
    )
    @test grads == (0.0, 0.0)
end

@testset "Enzyme reverse mode, Duplicated return — augmented_primal initializes shadow" begin
    # Covers augmented_primal with RT <: Duplicated{T}.
    f(x) = x^2
    fww = FunctionWrappersWrapper(f, (Tuple{Float64},), (Float64,))
    rconfig = EnzymeRules.RevConfig{true, true, 1, (false,), false, false}()

    aug = EnzymeRules.augmented_primal(
        rconfig, Const(fww), EnzymeCore.Duplicated{Float64},
        Active(3.0)
    )
    @test aug.primal ≈ 9.0                    # f(3) = 9
    @test aug.shadow ≈ 0.0                    # zero-initialized shadow
    @test aug.tape === nothing
end

@testset "Enzyme reverse mode, BatchDuplicated return — augmented_primal initializes shadows" begin
    # Covers augmented_primal with RT <: BatchDuplicated{T, W}.
    f(x) = x^2
    fww = FunctionWrappersWrapper(f, (Tuple{Float64},), (Float64,))
    rconfig = EnzymeRules.RevConfig{true, true, 2, (false,), false, false}()

    aug = EnzymeRules.augmented_primal(
        rconfig, Const(fww), EnzymeCore.BatchDuplicated{Float64, 2},
        Active(3.0)
    )
    @test aug.primal ≈ 9.0
    @test aug.shadow isa NTuple{2, Float64}
    @test aug.shadow == (0.0, 0.0)
    @test aug.tape === nothing
end

# =============================================================================
# End-to-end reverse-mode derivative tests — exercise Enzyme.autodiff(Reverse,
# …) through the FWW and assert the resulting gradients are numerically correct.
# The prior reverse-mode testsets only checked dispatch / shape of
# AugmentedReturn; they did NOT verify the gradients are right.
# =============================================================================

@testset "Enzyme Reverse: Const return, Active args — no-flow gradients" begin
    # For a function whose return is annotated Const in Reverse mode, there is
    # no gradient source from the return, so Active arg gradients must be 0.
    # (Enzyme's rule-return protocol requires concrete gradients for Active
    # args — `nothing` is not allowed — so the rule returns zeros.)
    g(x, y) = x * y + x^2
    fww = FunctionWrappersWrapper(g, (Tuple{Float64, Float64},), (Float64,))

    # Const return (instead of Active) → no gradient flows back
    result = Enzyme.autodiff(Reverse, Const(fww), Const, Active(3.0), Active(4.0))
    @test result[1] === (0.0, 0.0)
end

@testset "Enzyme Reverse: IIP with Duplicated args, Const return" begin
    # SciML's standard pattern: IIP RHS `f!(du, u)` with Const return, both du
    # and u are Duplicated.  Reverse mode should accumulate
    #    u_shadow[i] += du_shadow[j] * ∂(du[j])/∂(u[i])
    # into u_shadow.  For f!(du, u) = (du[1] = u[1]^2; nothing) with
    #   du_shadow = [1.0] (incoming adjoint),
    #   u[1] = 3.0,
    #   ∂du[1]/∂u[1] = 2*u[1] = 6,
    # the expected result is u_shadow[1] = 6.0 after the call.
    f!(du, u) = (du[1] = u[1]^2; nothing)
    fww = FunctionWrappersWrapper(
        f!, (Tuple{Vector{Float64}, Vector{Float64}},), (Nothing,)
    )

    du = [0.0]
    u = [3.0]
    du_shadow = [1.0]
    u_shadow = [0.0]

    Enzyme.autodiff(
        Reverse, Const(fww), Const,
        Duplicated(du, du_shadow), Duplicated(u, u_shadow)
    )
    @test du[1] ≈ 9.0          # primal effect: du[1] = u[1]^2 = 9
    @test u_shadow[1] ≈ 6.0    # reverse accumulation: 2 * u[1] * du_shadow[1]
end

@testset "Enzyme Reverse: IIP multi-component IIP with Duplicated args" begin
    # Cross-coupled IIP RHS: each output depends on multiple inputs.
    #   du[1] = u[1] * u[2]
    #   du[2] = u[1]^2 + u[2]^3
    # Jacobian at u = (x, y):
    #   J = [  y     x  ;
    #         2x   3y^2 ]
    # In reverse mode with du_shadow = [a, b], transpose of J applied to
    # du_shadow gives the accumulation into u_shadow:
    #   u_shadow[1] += a*y + b*2x
    #   u_shadow[2] += a*x + b*3y^2
    f!(du, u) = (du[1] = u[1] * u[2]; du[2] = u[1]^2 + u[2]^3; nothing)
    fww = FunctionWrappersWrapper(
        f!, (Tuple{Vector{Float64}, Vector{Float64}},), (Nothing,)
    )

    x, y = 2.0, 5.0
    a, b = 1.0, 0.5
    du = zeros(2)
    u = [x, y]
    du_shadow = [a, b]
    u_shadow = zeros(2)

    Enzyme.autodiff(
        Reverse, Const(fww), Const,
        Duplicated(du, du_shadow), Duplicated(u, u_shadow)
    )
    @test du ≈ [x * y, x^2 + y^3]
    @test u_shadow[1] ≈ a * y + b * 2 * x        # 5 + 2 = 7
    @test u_shadow[2] ≈ a * x + b * 3 * y^2      # 2 + 37.5 = 39.5
end

@testset "Enzyme ReverseWithPrimal: IIP with Duplicated args" begin
    # Same IIP pattern but with ReverseWithPrimal so we also check the primal
    # is available when the rule is asked to include it.
    f!(du, u) = (du[1] = u[1]^3; nothing)
    fww = FunctionWrappersWrapper(
        f!, (Tuple{Vector{Float64}, Vector{Float64}},), (Nothing,)
    )

    du = [0.0]
    u = [2.0]
    du_shadow = [1.0]
    u_shadow = [0.0]

    # Capture the expected gradient BEFORE the call — Enzyme may zero
    # `du_shadow` after consuming it during the reverse pass.
    expected_u_grad = 3 * u[1]^2 * du_shadow[1]  # = 12.0

    Enzyme.autodiff(
        ReverseWithPrimal, Const(fww), Const,
        Duplicated(du, du_shadow), Duplicated(u, u_shadow)
    )
    @test du[1] ≈ 8.0
    @test u_shadow[1] ≈ expected_u_grad
end

# =============================================================================
# Regression for the wrong gradient when a wrapped IIP function's arguments are
# MUTATED AFTER the call — the ODE-integrator pattern that the whole-solve
# Enzyme adjoint exercises (and the root cause of the EnsembleProblem adjoint
# failure, SciMLSensitivity.jl#1424).
#
# The Const-return reverse rule re-runs `Enzyme.autodiff(Reverse, …)` on the
# unwrapped function during the reverse pass.  If it differentiates about the
# arguments' *current* state rather than their *call-time* state, then any
# caller that steps `u` after the RHS call gets a silently wrong gradient.
# Before the snapshot/restore tape fix these end-to-end gradients were wrong.
# =============================================================================

@testset "Enzyme Reverse: IIP wrapper, args mutated after call (single step)" begin
    f!(du, u, p, t) = (du[1] = p[1] * u[1]; du[2] = p[2] * u[2]^2; nothing)
    ARGT = Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}

    function loss(p)
        u = [1.5, 2.0]
        du = zero(u)
        wf = FunctionWrappersWrapper(f!, (ARGT,), (Nothing,))
        wf(du, u, p, 0.0)
        @inbounds for k in 1:2
            u[k] += 0.05 * du[k]          # mutate u AFTER the wrapped call
        end
        return du[1]^2 + du[2]^2          # loss depends on du only
    end

    p = [0.7, 0.4]
    # du = [p1*1.5, p2*4];  loss = (1.5 p1)^2 + (4 p2)^2
    # ∂loss/∂p = [2*1.5^2*p1, 2*4^2*p2] = [4.5 p1, 32 p2]; evaluated at CALL-TIME u
    g = collect(Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), loss, p)[1])
    @test g ≈ [4.5 * p[1], 32 * p[2]]
end

@testset "Enzyme Reverse: IIP wrapper in a multi-step integrator" begin
    f!(du, u, p, t) = (
        du[1] = -p[1] * u[1] + p[2] * u[2];
        du[2] = -p[3] * u[2] + p[4] * u[1]; nothing
    )
    ARGT = Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}

    function loss(p)
        u = [1.0, 2.0]
        du = zero(u)
        wf = FunctionWrappersWrapper(f!, (ARGT,), (Nothing,))
        for _ in 1:8
            wf(du, u, p, 0.0)
            @inbounds for k in 1:2
                u[k] += 0.05 * du[k]      # integrator step mutates u each call
            end
        end
        return sum(abs2, u)
    end

    p = [1.0, 0.5, 2.0, 0.3]
    g = collect(Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), loss, p)[1])

    # central finite-difference reference (no extra deps)
    fd = map(eachindex(p)) do i
        h = 1.0e-6
        pp = copy(p); pp[i] += h
        pm = copy(p); pm[i] -= h
        (loss(pp) - loss(pm)) / (2h)
    end
    @test g ≈ fd rtol = 1.0e-4
end

@testset "Enzyme Reverse: IIP wrapper with a mix of Duplicated and Active args" begin
    # A time-dependent in-place rhs, differentiated so the reverse rule sees
    # (Duplicated du, Duplicated u, Duplicated p, Active t).  The rule must
    # return the *real* gradient for the Active `t` with an exact-typed tuple
    # (Nothing per Duplicated arg, Float64 for the Active).  Before the fix this
    # returned a union-typed `(nothing, …, 0.0)` — Enzyme rejected it with a
    # `ReverseRuleReturnError`, and the `t`-gradient was zeroed rather than
    # computed.
    f!(du, u, p, t) = (du[1] = p[1] * u[1] + t * u[2]; du[2] = p[2] * u[2]; nothing)
    ARGT = Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}

    function loss(x)              # x = [p1, p2, t]
        u = [1.5, 2.0]
        du = zero(u)
        wf = FunctionWrappersWrapper(f!, (ARGT,), (Nothing,))
        wf(du, u, [x[1], x[2]], x[3])   # t = x[3] flows in as an Active scalar
        return du[1]^2 + du[2]^2
    end

    x = [0.7, 0.4, 0.9]
    g = collect(Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), loss, x)[1])
    # du = [p1*u1 + t*u2, p2*u2]; loss = du1^2 + du2^2
    du1 = x[1] * 1.5 + x[3] * 2.0
    du2 = x[2] * 2.0
    @test g ≈ [2 * du1 * 1.5, 2 * du2 * 2.0, 2 * du1 * 2.0]   # ∂/∂t = 2*du1*u2 ≠ 0
end

# =============================================================================
# Runtime-activity propagation through the FWW forward rules.
#
# Prior to this fix the rules hard-coded plain `Forward` when delegating to
# `Enzyme.autodiff`, silently dropping the caller's
# `set_runtime_activity(Forward)` flag.  Enzyme's static IR-level activity
# analysis can't see through `FunctionWrappersWrapper`'s opaque cfunction
# indirection, so the inner call raised `EnzymeRuntimeActivityError` inside
# `@.` broadcast's `broadcast_unalias` → `mightalias` — despite
# `set_runtime_activity` being set on the outer `autodiff` call.
#
# Upstream motivation: OrdinaryDiffEq.jl PR #3518 —
#   Rosenbrock23(autodiff = AutoEnzyme(set_runtime_activity(Enzyme.Forward)))
# on any time-dependent in-place RHS routed through DiffEqBase's
# `wrapfun_iip`.  Here we reproduce the shape (`f!(du, u, p, t) = @. du = …`)
# in a 4-arg `FunctionWrappersWrapper` matching DiffEqBase's
# `wrapfun_iip` output, and assert both that (a) the call completes without
# an `EnzymeRuntimeActivityError` and (b) the resulting tangent is
# numerically correct.
# =============================================================================

# =============================================================================
# Duplicated function annotation on the FWW itself.
#
# Reproduces SciML/FunctionWrappersWrappers.jl#48: when Enzyme differentiates
# through a closure that captures an FWW (e.g. NonlinearSolve +
# SciMLSensitivity), the rule is invoked with
# `Duplicated{<:FunctionWrappersWrapper}` for the function argument, not
# `Const{<:FunctionWrappersWrapper}`.  The FWW struct itself only carries
# `FunctionWrapper`s + cache storage, so its "shadow" is ignored — we route
# through `unwrap(func.val)` exactly as with `Const`.
# =============================================================================

@testset "Enzyme forward, Duplicated FWW annotation — IIP Const return" begin
    f!(residual, u, p) = (residual[1] = u[1]^2 - p[1]; nothing)
    fww = FunctionWrappersWrapper(
        f!,
        (Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}},),
        (Nothing,)
    )

    residual = [0.0]; dresidual = [0.0]
    u = [2.0];        du = [1.0]
    p = [1.0];        dp = [0.0]

    config = EnzymeCore.EnzymeRules.FwdConfig{false, false, 1, false, false}()
    ret = EnzymeCore.EnzymeRules.forward(
        config,
        Duplicated(fww, fww),                # <-- the failing dispatch in #48
        EnzymeCore.Const{Nothing},
        Duplicated(residual, dresidual),
        Duplicated(u, du),
        Duplicated(p, dp),
    )
    @test ret === nothing
    @test residual[1] ≈ 3.0      # u[1]^2 - p[1] = 4 - 1
    @test dresidual[1] ≈ 4.0     # 2*u[1]*du[1] - 1*dp[1] = 4
end

@testset "Enzyme forward, Duplicated FWW annotation — shadow-only return" begin
    # Drive the {false, true, W, …} rule (shadow only, no primal) with a
    # Duplicated FWW.
    f(x) = x^2
    fww = FunctionWrappersWrapper(f, (Tuple{Float64},), (Float64,))

    config = EnzymeCore.EnzymeRules.FwdConfig{false, true, 1, false, false}()
    shadow = EnzymeCore.EnzymeRules.forward(
        config,
        Duplicated(fww, fww),
        EnzymeCore.Duplicated{Float64},
        Duplicated(3.0, 1.0),
    )
    @test shadow ≈ 6.0           # f'(3) = 2*3 = 6
end

@testset "Enzyme forward, Duplicated FWW annotation — primal + shadow return" begin
    # Drive the {true, true, W, …} rule (ForwardWithPrimal) with a Duplicated
    # FWW.
    f(x) = x^2
    fww = FunctionWrappersWrapper(f, (Tuple{Float64},), (Float64,))

    config = EnzymeCore.EnzymeRules.FwdConfig{true, true, 1, false, false}()
    result = EnzymeCore.EnzymeRules.forward(
        config,
        Duplicated(fww, fww),
        EnzymeCore.Duplicated{Float64},
        Duplicated(3.0, 1.0),
    )
    @test result isa Duplicated
    @test result.val ≈ 9.0       # primal
    @test result.dval ≈ 6.0      # shadow
end

@testset "Enzyme reverse, Duplicated FWW annotation — Const return IIP" begin
    # Mirror the forward IIP case on the reverse side.  Duplicated FWW must
    # still drive the rule, gradients must accumulate into u_shadow.
    f!(du, u) = (du[1] = u[1]^2; nothing)
    fww = FunctionWrappersWrapper(
        f!, (Tuple{Vector{Float64}, Vector{Float64}},), (Nothing,)
    )

    du = [0.0];       du_shadow = [1.0]
    u = [3.0];       u_shadow = [0.0]

    # Overwritten indexed (func, du, u); none modified between fwd and rev here.
    rconfig = EnzymeRules.RevConfig{false, false, 1, (false, false, false), false, false}()
    aug = EnzymeRules.augmented_primal(
        rconfig,
        Duplicated(fww, fww),                # <-- Duplicated FWW
        EnzymeCore.Const{Nothing},
        Duplicated(du, du_shadow),
        Duplicated(u, u_shadow),
    )
    @test aug.primal === nothing
    @test aug.shadow === nothing

    EnzymeRules.reverse(
        rconfig,
        Duplicated(fww, fww),
        EnzymeCore.Const{Nothing},
        aug.tape,
        Duplicated(du, du_shadow),
        Duplicated(u, u_shadow),
    )
    @test du[1] ≈ 9.0            # primal effect from augmented_primal
    @test u_shadow[1] ≈ 6.0      # reverse accumulation: 2*u[1]*du_shadow[1]
end

@testset "Enzyme Forward: set_runtime_activity propagates through FWW (IIP, time-dependent)" begin
    # DiffEqBase's `wrapfun_iip(ff, (u, u, p, t))` shape.
    const_INPUTS = Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}

    # 1) Time-independent RHS — ∂du/∂t = 0.
    f!(du, u, p, t) = (@. du = p * u; nothing)
    fww = FunctionWrappersWrapper(f!, (const_INPUTS,), (Nothing,))

    u = [1.0, 2.0, 3.0]
    p = [0.5, 0.5, 0.5]
    t = 1.7
    du = zero(u); ddu = zero(u); dt = 1.0

    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Forward),
        Const(fww), Const,
        Duplicated(du, ddu),
        Const(u), Const(p),
        Duplicated(t, dt),
    )
    @test du ≈ p .* u
    @test all(iszero, ddu)

    # 2) Non-trivial time dependence: g!(du,u,p,t) = @. sin(t)*u.
    #    Expected ∂du/∂t = cos(t) .* u.
    g!(du, u, p, t) = (@. du = sin(t) * u; nothing)
    gww = FunctionWrappersWrapper(g!, (const_INPUTS,), (Nothing,))

    du2 = zero(u); ddu2 = zero(u)
    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Forward),
        Const(gww), Const,
        Duplicated(du2, ddu2),
        Const(u), Const(p),
        Duplicated(t, 1.0),
    )
    @test du2 ≈ sin(t) .* u
    @test ddu2 ≈ cos(t) .* u

    # 3) Confirm the rule also propagates `set_strong_zero(Forward)` (the
    #    other ForwardMode flag carried in FwdConfig) — another RHS that
    #    doesn't need runtime activity but exercises a distinct flag.
    h!(du, u, p, t) = (du[1] = u[1] * t; nothing)
    hww = FunctionWrappersWrapper(
        h!, (Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64},),
        (Nothing,)
    )
    du_h = [0.0]; ddu_h = [0.0]
    Enzyme.autodiff(
        Enzyme.set_strong_zero(Forward),
        Const(hww), Const,
        Duplicated(du_h, ddu_h),
        Const([2.0]), Const([0.0]),
        Duplicated(3.5, 1.0),
    )
    @test du_h[1] ≈ 2.0 * 3.5     # primal: u[1] * t = 7.0
    @test ddu_h[1] ≈ 2.0          # ∂(u[1]*t)/∂t = u[1]
end
