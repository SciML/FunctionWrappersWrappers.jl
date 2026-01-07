using Test, Pkg

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "All" || GROUP == "Core"
    @testset "FunctionWrappersWrappers.jl" begin
        include("basictests.jl")
    end
end

if GROUP == "nopre"
    Pkg.activate("nopre")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
    include("nopre/jet_tests.jl")
end

@testset "BigFloat support" begin
    fwplus_big = FunctionWrappersWrapper(
        +,
        (Tuple{BigFloat, BigFloat}, Tuple{Float64, Float64}),
        (BigFloat, Float64)
    )
    a = BigFloat("3.14159265358979323846264338327950288")
    b = BigFloat("2.71828182845904523536028747135266250")
    @test fwplus_big(a, b) isa BigFloat
    @test fwplus_big(a, b) == a + b
    @test fwplus_big(1.0, 2.0) === 3.0

    fwsin_big = FunctionWrappersWrapper(
        sin,
        (Tuple{BigFloat}, Tuple{Float64}),
        (BigFloat, Float64)
    )
    @test fwsin_big(BigFloat("1.0")) isa BigFloat
    @test fwsin_big(1.0) === sin(1.0)
end

@testset "UnionAll return types" begin
    # Test that UnionAll types (like AbstractArray{Float64}) work as return types
    function double_array(x::AbstractArray{Float64})
        return x .* 2
    end

    fwdouble = FunctionWrappersWrapper(
        double_array,
        (Tuple{AbstractArray{Float64}},),
        (AbstractArray{Float64},)
    )

    v = [1.0, 2.0, 3.0]
    result = fwdouble(v)
    @test result isa Vector{Float64}
    @test result == [2.0, 4.0, 6.0]
end
