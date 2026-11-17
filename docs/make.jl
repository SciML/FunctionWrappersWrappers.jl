using FunctionWrappersWrappers
using Documenter

DocMeta.setdocmeta!(
    FunctionWrappersWrappers, :DocTestSetup,
    :(using FunctionWrappersWrappers); recursive = true
)

makedocs(;
    modules = [FunctionWrappersWrappers],
    authors = "Chris Elrod <elrodc@gmail.com> and contributors",
    sitename = "FunctionWrappersWrappers.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://docs.sciml.ai/FunctionWrappersWrappers/stable/",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md",
    ]
)

deploydocs(;
    repo = "github.com/SciML/FunctionWrappersWrappers.jl",
    devbranch = "main"
)
