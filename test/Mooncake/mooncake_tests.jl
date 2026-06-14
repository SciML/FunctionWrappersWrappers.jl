using FunctionWrappersWrappers
using Mooncake
using Test

@testset "Mooncake reverse mode - single arg" begin
    f(x) = x^2
    fww = FunctionWrappersWrapper(f, (Tuple{Float64},), (Float64,))

    rule = Mooncake.build_rrule(fww, 3.0)
    val, (_, dx) = Mooncake.value_and_gradient!!(rule, fww, 3.0)
    @test val ≈ 9.0
    @test dx ≈ 6.0
end

@testset "Mooncake reverse mode - multi arg" begin
    g(x, y) = x * y + x^2
    fww = FunctionWrappersWrapper(g, (Tuple{Float64, Float64},), (Float64,))

    # g(x,y) = x*y + x^2 → ∂g/∂x = y + 2x, ∂g/∂y = x
    rule = Mooncake.build_rrule(fww, 3.0, 4.0)
    val, (_, dx, dy) = Mooncake.value_and_gradient!!(rule, fww, 3.0, 4.0)
    @test val ≈ 21.0
    @test dx ≈ 10.0  # ∂g/∂x at (3,4) = 4 + 6
    @test dy ≈ 3.0   # ∂g/∂y at (3,4) = 3
end

@testset "Mooncake with trig functions" begin
    fww_sin = FunctionWrappersWrapper(sin, (Tuple{Float64},), (Float64,))

    rule = Mooncake.build_rrule(fww_sin, 1.0)
    val, (_, dx) = Mooncake.value_and_gradient!!(rule, fww_sin, 1.0)
    @test val ≈ sin(1.0)
    @test dx ≈ cos(1.0)
end

@testset "Mooncake through loss function" begin
    # Test that Mooncake can differentiate a loss function that calls FunctionWrappersWrapper
    f(x) = x[1]^2 + x[2]^2
    fww = FunctionWrappersWrapper(f, (Tuple{Vector{Float64}},), (Float64,))

    loss(x) = fww(x)
    rule = Mooncake.build_rrule(loss, [3.0, 4.0])
    val, (_, dx) = Mooncake.value_and_gradient!!(rule, loss, [3.0, 4.0])
    @test val ≈ 25.0
    @test dx ≈ [6.0, 8.0]
end

@testset "Mooncake with wrapped callable struct" begin
    # SciML wraps functions in Void{F} or similar callable structs before
    # putting them in FunctionWrappersWrapper. The unwrapped function is
    # then a non-primitive callable struct, not a plain function.
    struct VoidWrapper{F}
        f::F
    end
    function (v::VoidWrapper)(args...)
        v.f(args...)
        return nothing
    end

    function f!(du, u, p)
        du[1] = p[1] * u[1]
        du[2] = p[2] * u[2]
        return nothing
    end

    wrapped = VoidWrapper(f!)
    fww = FunctionWrappersWrapper(
        wrapped,
        (Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}},),
        (Nothing,),
    )

    function loss(p)
        u = [3.0, 4.0]
        du = similar(u)
        fww(du, u, p)
        return sum(abs2, du)
    end

    rule = Mooncake.build_rrule(loss, [2.0, 3.0])
    val, (_, dp) = Mooncake.value_and_gradient!!(rule, loss, [2.0, 3.0])
    # du = [2*3, 3*4] = [6, 12], loss = 36 + 144 = 180
    @test val ≈ 180.0
    # ∂loss/∂p1 = 2*du[1]*u[1] = 2*6*3 = 36
    # ∂loss/∂p2 = 2*du[2]*u[2] = 2*12*4 = 96
    @test dp ≈ [36.0, 96.0]
end

@testset "Mooncake in-place function" begin
    # In-place functions are common in SciML (f!(du, u, p, t))
    function f!(du, u, p)
        du[1] = p[1] * u[1] + p[2] * u[2]
        du[2] = p[3] * u[1] - u[2]
        return nothing
    end
    fww = FunctionWrappersWrapper(
        f!,
        (Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}},),
        (Nothing,),
    )

    function loss(p)
        u = [1.0, 2.0]
        du = similar(u)
        fww(du, u, p)
        return sum(abs2, du)
    end

    rule = Mooncake.build_rrule(loss, [1.0, 2.0, 3.0])
    val, (_, dp) = Mooncake.value_and_gradient!!(rule, loss, [1.0, 2.0, 3.0])
    # f!(du, [1,2], [1,2,3]) → du = [1*1+2*2, 3*1-2] = [5, 1]
    # loss = 25 + 1 = 26
    @test val ≈ 26.0
    # ∂loss/∂p1 = 2*du[1]*u[1] = 2*5*1 = 10
    # ∂loss/∂p2 = 2*du[1]*u[2] = 2*5*2 = 20
    # ∂loss/∂p3 = 2*du[2]*u[1] = 2*1*1 = 2
    @test dp ≈ [10.0, 20.0, 2.0]
end
