using FunctionWrappersWrappers
using Mooncake
using Test

@testset "Mooncake reverse mode - single arg" begin
    f(x) = x^2
    fww = FunctionWrappersWrapper(f, (Tuple{Float64},), (Float64,))

    rule = Mooncake.build_rrule(fww, 3.0)
    val, (dfww, dx) = Mooncake.value_and_gradient!!(rule, fww, 3.0)
    @test val ≈ 9.0
    @test dx ≈ 6.0
end

@testset "Mooncake reverse mode - multi arg" begin
    g(x, y) = x * y + x^2
    fww = FunctionWrappersWrapper(g, (Tuple{Float64, Float64},), (Float64,))

    rule = Mooncake.build_rrule(fww, 3.0, 4.0)
    val, (dfww, dx, dy) = Mooncake.value_and_gradient!!(rule, fww, 3.0, 4.0)
    @test val ≈ 21.0   # 3*4 + 9
    @test dx ≈ 10.0    # y + 2x = 4 + 6
    @test dy ≈ 3.0     # x
end

@testset "Mooncake with trig functions" begin
    fww_sin = FunctionWrappersWrapper(sin, (Tuple{Float64},), (Float64,))

    rule = Mooncake.build_rrule(fww_sin, 1.0)
    val, (dfww, dx) = Mooncake.value_and_gradient!!(rule, fww_sin, 1.0)
    @test val ≈ sin(1.0)
    @test dx ≈ cos(1.0)
end

@testset "Mooncake unwrap correctness" begin
    # Verify that the overlay correctly unwraps to the original function
    f(x) = exp(x) + x^3
    fww = FunctionWrappersWrapper(f, (Tuple{Float64},), (Float64,))

    rule = Mooncake.build_rrule(fww, 2.0)
    val, (dfww, dx) = Mooncake.value_and_gradient!!(rule, fww, 2.0)
    @test val ≈ exp(2.0) + 8.0
    @test dx ≈ exp(2.0) + 12.0  # exp(x) + 3x^2
end
