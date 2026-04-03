using FunctionWrappersWrappers
using Enzyme
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
