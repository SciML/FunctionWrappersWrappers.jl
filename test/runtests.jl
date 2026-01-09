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
