using SafeTestsets
using SciMLTesting

run_tests(;
    core = function ()
        @safetestset "FunctionWrappersWrappers.jl" begin
            include(joinpath(@__DIR__, "basictests.jl"))
        end
        return @safetestset "BigFloat + UnionAll" begin
            include(joinpath(@__DIR__, "shared", "bigfloat_unionall_tests.jl"))
        end
    end,
    groups = Dict(
        "nopre" => (;
            env = joinpath(@__DIR__, "nopre"),
            body = function ()
                @safetestset "JET" begin
                    include(joinpath(@__DIR__, "nopre", "jet_tests.jl"))
                end
                return @safetestset "BigFloat + UnionAll" begin
                    include(joinpath(@__DIR__, "shared", "bigfloat_unionall_tests.jl"))
                end
            end,
        ),
        "Enzyme" => joinpath(@__DIR__, "Enzyme", "enzyme_tests.jl"),
        "Mooncake" => joinpath(@__DIR__, "Mooncake", "mooncake_tests.jl"),
    ),
    all = ["Core", "Enzyme", "Mooncake"],
)
