using FunctionWrappersWrappers
using Test

@testset "FunctionWrappersWrappers.jl" begin
    fwplus = FunctionWrappersWrapper(
        +, (Tuple{Float64, Float64}, Tuple{Int, Int}), (
            Float64, Int,
        )
    )
    @test fwplus(4.0, 8.0) === 12.0
    @test fwplus(4, 8) === 12

    fwexp2 = FunctionWrappersWrapper(
        exp2, (Tuple{Float64}, Tuple{Float32}, Tuple{Int}), (Float64, Float32, Float64)
    )
    @test fwexp2(4.0) === 16.0
    @test fwexp2(4.0f0) === 16.0f0
    @test fwexp2(4) === 16.0
end

@testset "Type inference" begin
    fwplus = FunctionWrappersWrapper(
        +, (Tuple{Float64, Float64}, Tuple{Int, Int}), (
            Float64, Int,
        )
    )
    @test @inferred(fwplus(4.0, 8.0)) === 12.0
    @test @inferred(fwplus(4, 8)) === 12

    fwexp2 = FunctionWrappersWrapper(
        exp2, (Tuple{Float64}, Tuple{Float32}, Tuple{Int}), (Float64, Float32, Float64)
    )
    @test @inferred(fwexp2(4.0)) === 16.0
    @test @inferred(fwexp2(4.0f0)) === 16.0f0
    @test @inferred(fwexp2(4)) === 16.0
end

@testset "Introspection functions" begin
    fwsin = FunctionWrappersWrapper(sin, (Tuple{Float64},), (Float64,))

    @testset "unwrap" begin
        f = unwrap(fwsin)
        @test f === sin
        @test f(0.5) == sin(0.5)
    end

    @testset "wrapped_signatures" begin
        sigs = wrapped_signatures(fwsin)
        @test sigs == (Tuple{Float64},)
    end

    @testset "wrapped_return_types" begin
        rets = wrapped_return_types(fwsin)
        @test rets == (Float64,)
    end

    fwplus = FunctionWrappersWrapper(
        +, (Tuple{Float64, Float64}, Tuple{Int, Int}), (
            Float64, Int,
        )
    )

    @testset "unwrap with multiple signatures" begin
        f = unwrap(fwplus)
        @test f === +
        @test f(1, 2) == 3
    end

    @testset "wrapped_signatures with multiple signatures" begin
        sigs = wrapped_signatures(fwplus)
        @test sigs == (Tuple{Float64, Float64}, Tuple{Int, Int})
    end

    @testset "wrapped_return_types with multiple signatures" begin
        rets = wrapped_return_types(fwplus)
        @test rets == (Float64, Int)
    end

    my_func(x) = x^2
    fwcustom = FunctionWrappersWrapper(
        my_func, (Tuple{Float64}, Tuple{Int}), (
            Float64, Int,
        )
    )

    @testset "unwrap with custom function" begin
        f = unwrap(fwcustom)
        @test f === my_func
        @test f(3) == 9
        @test f(2.5) == 6.25
    end
end

@testset "Legacy API (Val{true}/Val{false})" begin
    fwplus = FunctionWrappersWrapper(
        +, (Tuple{Float64, Float64}, Tuple{Int, Int}), (Float64, Int), Val{false}()
    )
    @test fwplus(4.0, 8.0) === 12.0
    @test fwplus(4, 8) === 12
    @test_throws FunctionWrappersWrappers.NoFunctionWrapperFoundError fwplus(4.0f0, 8.0f0)

    fwplus_fb = FunctionWrappersWrapper(
        +, (Tuple{Float64, Float64}, Tuple{Int, Int}), (Float64, Int), Val{true}()
    )
    @test fwplus_fb(4.0, 8.0) === 12.0
    @test fwplus_fb(4, 8) === 12
    @test fwplus_fb(4.0f0, 8.0f0) == 12.0f0  # fallback to original function
end

@testset "Legacy FW{FW,Bool} constructor" begin
    using FunctionWrappers
    fw1 = FunctionWrappers.FunctionWrapper{Float64, Tuple{Float64, Float64}}(+)
    fw2 = FunctionWrappers.FunctionWrapper{Int, Tuple{Int, Int}}(+)
    fwt = (fw1, fw2)

    fww_strict = FunctionWrappersWrapper{typeof(fwt), false}(fwt)
    @test fww_strict(4.0, 8.0) === 12.0
    @test fww_strict(4, 8) === 12
    @test_throws FunctionWrappersWrappers.NoFunctionWrapperFoundError fww_strict(4.0f0, 8.0f0)

    fww_fb = FunctionWrappersWrapper{typeof(fwt), true}(fwt)
    @test fww_fb(4.0, 8.0) === 12.0
    @test fww_fb(4, 8) === 12
    @test fww_fb(4.0f0, 8.0f0) == 12.0f0
end

@testset "Fallback policies" begin
    @testset "Strict" begin
        fww = FunctionWrappersWrapper(
            +, (Tuple{Float64, Float64},), (Float64,);
            cache = NoCache(), policy = Strict()
        )
        @test fww(4.0, 8.0) === 12.0
        @test_throws FunctionWrappersWrappers.NoFunctionWrapperFoundError fww(4, 8)
        @test_throws FunctionWrappersWrappers.NoFunctionWrapperFoundError fww(
            BigFloat(4), BigFloat(8)
        )
    end

    @testset "AllowAll" begin
        fww = FunctionWrappersWrapper(
            +, (Tuple{Float64, Float64},), (Float64,);
            cache = NoCache(), policy = AllowAll()
        )
        @test fww(4.0, 8.0) === 12.0
        @test fww(4, 8) === 12
        @test fww(4.0f0, 8.0f0) == 12.0f0
        @test fww(BigFloat(4), BigFloat(8)) == BigFloat(12)
    end

    @testset "AllowNonIsBits" begin
        fww = FunctionWrappersWrapper(
            +, (Tuple{Float64, Float64},), (Float64,);
            cache = NoCache(), policy = AllowNonIsBits()
        )
        @test fww(4.0, 8.0) === 12.0
        # Float32 is isbits but doesn't match Float64 wrapper → error
        @test_throws FunctionWrappersWrappers.NoFunctionWrapperFoundError fww(4.0f0, 8.0f0)
        # Int is isbits but doesn't match Float64 wrapper → error
        @test_throws FunctionWrappersWrappers.NoFunctionWrapperFoundError fww(4, 8)
        # BigFloat is non-isbits → allowed
        @test fww(BigFloat(4), BigFloat(8)) == BigFloat(12)
    end

    @testset "AllowNonIsBits with arrays" begin
        f!(du, u) = (du[1] = u[1]^2; nothing)
        fww = FunctionWrappersWrapper(
            f!, (Tuple{Vector{Float64}, Vector{Float64}},), (Nothing,);
            cache = NoCache(), policy = AllowNonIsBits()
        )
        du_f = [0.0]; u_f = [3.0]
        fww(du_f, u_f)
        @test du_f[1] === 9.0

        # Float32 arrays: eltype is isbits but doesn't match → error
        @test_throws FunctionWrappersWrappers.NoFunctionWrapperFoundError fww(
            Float32[0.0], Float32[3.0]
        )

        # BigFloat arrays: eltype is non-isbits → allowed
        du_bf = BigFloat[0]; u_bf = BigFloat[3]
        fww(du_bf, u_bf)
        @test du_bf[1] == BigFloat(9)
    end
end

@testset "Cache modes" begin
    f!(du, u, p, t) = (du[1] = p[1] * u[1]; nothing)

    @testset "NoCache" begin
        fww = FunctionWrappersWrapper(
            f!,
            (Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64},),
            (Nothing,);
            cache = NoCache(), policy = AllowAll()
        )
        # Float64 match
        du = [0.0]; u = [2.0]; p = [3.0]
        fww(du, u, p, 0.0)
        @test du[1] === 6.0

        # BigFloat fallback (NoCache: 1 alloc per call)
        du_bf = BigFloat[0]; u_bf = BigFloat[2]; p_bf = BigFloat[3]; t_bf = BigFloat(0)
        fww(du_bf, u_bf, p_bf, t_bf)
        @test du_bf[1] == BigFloat(6)
    end

    @testset "SingleCache" begin
        fww = FunctionWrappersWrapper(
            f!,
            (Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64},),
            (Nothing,);
            cache = SingleCache(), policy = AllowAll()
        )
        du_bf = BigFloat[0]; u_bf = BigFloat[2]; p_bf = BigFloat[3]; t_bf = BigFloat(0)
        # First call caches
        fww(du_bf, u_bf, p_bf, t_bf)
        @test du_bf[1] == BigFloat(6)
        # Second call uses cache (0 alloc)
        du_bf[1] = BigFloat(0)
        fww(du_bf, u_bf, p_bf, t_bf)
        @test du_bf[1] == BigFloat(6)
    end

    @testset "DictCache" begin
        fww = FunctionWrappersWrapper(
            f!,
            (Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64},),
            (Nothing,);
            cache = DictCache(), policy = AllowAll()
        )
        du_bf = BigFloat[0]; u_bf = BigFloat[2]; p_bf = BigFloat[3]; t_bf = BigFloat(0)
        fww(du_bf, u_bf, p_bf, t_bf)
        @test du_bf[1] == BigFloat(6)

        # Different type also works and caches separately
        du_f32 = Float32[0]; u_f32 = Float32[2]; p_f32 = Float32[3]; t_f32 = Float32(0)
        fww(du_f32, u_f32, p_f32, t_f32)
        @test du_f32[1] === Float32(6)

        # BigFloat still cached
        du_bf[1] = BigFloat(0)
        fww(du_bf, u_bf, p_bf, t_bf)
        @test du_bf[1] == BigFloat(6)
    end

    @testset "SingleCache thrashing recovers" begin
        fww = FunctionWrappersWrapper(
            f!,
            (Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64},),
            (Nothing,);
            cache = SingleCache(), policy = AllowAll()
        )
        du_bf = BigFloat[0]; u_bf = BigFloat[2]; p_bf = BigFloat[3]; t_bf = BigFloat(0)
        du_f32 = Float32[0]; u_f32 = Float32[2]; p_f32 = Float32[3]; t_f32 = Float32(0)

        # Alternate types — each call replaces the cache but still works
        fww(du_bf, u_bf, p_bf, t_bf)
        @test du_bf[1] == BigFloat(6)
        fww(du_f32, u_f32, p_f32, t_f32)
        @test du_f32[1] === Float32(6)
        du_bf[1] = BigFloat(0)
        fww(du_bf, u_bf, p_bf, t_bf)
        @test du_bf[1] == BigFloat(6)
    end
end

@testset "Default constructor uses SingleCache + AllowNonIsBits" begin
    fww = FunctionWrappersWrapper(
        +, (Tuple{Float64, Float64},), (Float64,)
    )
    # Float64 matches wrapper
    @test fww(4.0, 8.0) === 12.0
    # BigFloat is non-isbits → falls back
    @test fww(BigFloat(4), BigFloat(8)) == BigFloat(12)
    # Float32 is isbits mismatch → errors
    @test_throws FunctionWrappersWrappers.NoFunctionWrapperFoundError fww(4.0f0, 8.0f0)
end
