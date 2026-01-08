using Test

const GROUP = get(ENV, "GROUP", "All")

@testset "FunctionWrappersWrappers.jl" begin
    if GROUP == "All" || GROUP == "Core"
        include("basictests.jl")
    end

    if GROUP == "All" || GROUP == "nopre"
        include("jet_tests.jl")
    end
end