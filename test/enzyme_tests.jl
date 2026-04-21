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

@testset "Enzyme forward mode, neither primal nor shadow requested" begin
    # Covers EnzymeRules.FwdConfig{false, false, W, ...}: caller wants only the
    # side-effects of the primal invocation, no return value and no derivative.
    # Reproduces the SciML/OrdinaryDiffEq.jl v7 Downstream regression where
    # Enzyme dispatched on this config combination with a FWW wrapping an IIP
    # RHS and found no matching rule, throwing
    #   MethodError: no method matching forward(
    #       ::FwdConfigWidth{1, false, false, false, false},
    #       ::Const{<:FunctionWrappersWrapper}, ::Type{Const{Nothing}}, …)
    f!(du, u) = (du[1] = -u[1]^2; nothing)
    fww = FunctionWrappersWrapper(
        f!, (Tuple{Vector{Float64}, Vector{Float64}},), (Nothing,)
    )

    du = [0.0]
    u = [3.0]
    du_shadow = [0.0]
    u_shadow = [1.0]

    # Call forward directly with {false, false}: Enzyme's public-facing
    # autodiff front-end doesn't normally expose this config, so invoke the
    # rule by hand.
    config = EnzymeCore.EnzymeRules.FwdConfig{false, false, 1, false, false}()
    ret = EnzymeCore.EnzymeRules.forward(
        config, Const(fww), EnzymeCore.Const{Nothing},
        Duplicated(du, du_shadow), Duplicated(u, u_shadow)
    )
    @test ret === nothing
    # primal side-effect did happen: f!(du, u) sets du[1] = -u[1]^2 = -9
    @test du[1] ≈ -9.0
    # shadow buffer was not touched by this no-diff path
    @test du_shadow[1] == 0.0
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

    # Reverse step — dret is Const, no grads to accumulate.
    grads = EnzymeRules.reverse(
        rconfig, Const(fww), EnzymeCore.Const{Float64}(0.0),
        aug.tape, Active(3.0), Active(4.0)
    )
    @test grads == (nothing, nothing)
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

