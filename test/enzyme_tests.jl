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
    # primal runs the wrapped function for its side effects and returns
    # AugmentedReturn(nothing, nothing, nothing).  Reverse returns `nothing`
    # per arg since there is no return derivative to propagate.
    counter = Ref(0)
    g(x, y) = (counter[] += 1; x + y)  # returns Float64 (ignored via Const RT)
    fww = FunctionWrappersWrapper(g, (Tuple{Float64, Float64},), (Float64,))

    # Construct a concrete RevConfig. Fields:
    # (NeedsPrimal, NeedsShadow, Width, Overwritten, RuntimeActivity, StrongZero)
    rconfig = EnzymeRules.RevConfig{false, false, 1, (false, false), false, false}()

    counter[] = 0
    aug = EnzymeRules.augmented_primal(
        rconfig, Const(fww), EnzymeCore.Const{Float64},
        Active(3.0), Active(4.0)
    )
    @test counter[] == 1                       # primal ran exactly once
    @test aug.primal === nothing               # NeedsPrimal == false
    @test aug.shadow === nothing
    @test aug.tape === nothing

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
    f!(du, u) = (du[1] = u[1]*u[2]; du[2] = u[1]^2 + u[2]^3; nothing)
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
    @test du ≈ [x*y, x^2 + y^3]
    @test u_shadow[1] ≈ a*y + b*2*x        # 5 + 2 = 7
    @test u_shadow[2] ≈ a*x + b*3*y^2      # 2 + 37.5 = 39.5
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

