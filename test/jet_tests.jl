using FunctionWrappersWrappers
using JET: JET, @test_opt
using Test

@testset "JET static analysis" begin
    # Test that the main call path is type-stable (no fallback)
    fwplus = FunctionWrappersWrapper(+, (Tuple{Float64, Float64}, Tuple{Int, Int}), (
        Float64, Int))

    # Core functionality should be type-stable
    @test_opt target_modules = (FunctionWrappersWrappers,) fwplus(4.0, 8.0)
    @test_opt target_modules = (FunctionWrappersWrappers,) fwplus(4, 8)

    # Test single-argument wrapper
    fwexp2 = FunctionWrappersWrapper(
        exp2, (Tuple{Float64}, Tuple{Float32}, Tuple{Int}), (Float64, Float32, Float64))
    @test_opt target_modules = (FunctionWrappersWrappers,) fwexp2(4.0)
    @test_opt target_modules = (FunctionWrappersWrappers,) fwexp2(4.0f0)
    @test_opt target_modules = (FunctionWrappersWrappers,) fwexp2(4)

    # Verify no errors detected by JET.report_call for core paths
    rep = JET.report_call(fwplus, (Float64, Float64))
    @test length(JET.get_reports(rep)) == 0
    rep = JET.report_call(fwplus, (Int, Int))
    @test length(JET.get_reports(rep)) == 0
end
