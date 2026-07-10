using SciMLTesting, FunctionWrappersWrappers, Test
using JET

run_qa(
    FunctionWrappersWrappers;
    explicit_imports = true,
    ei_kwargs = (;
        # Qualified accesses to names that are not (yet) declared `public` in their
        # owning module: `Base.tail`, `FunctionWrappers.FunctionWrapper`, and
        # `TruncatedStacktraces.@truncate_stacktrace`. These are core, long-standing
        # API of Base / FunctionWrappers / TruncatedStacktraces that predate the
        # `public` keyword; ignore until those packages mark them public.
        all_qualified_accesses_are_public = (;
            ignore = (:tail, :FunctionWrapper, Symbol("@truncate_stacktrace")),
        ),
    ),
)

function public_api_names(mod::Module)
    public_names = Set(Symbol.(names(mod; all = false, imported = false)))
    if isdefined(Base, :ispublic)
        for name in names(mod; all = true, imported = false)
            Base.ispublic(mod, name) && push!(public_names, Symbol(name))
        end
    end
    delete!(public_names, nameof(mod))
    return sort!(collect(public_names))
end

function has_source_docstring(mod::Module, name::Symbol)
    if isdefined(Docs, :hasdoc)
        return Docs.hasdoc(mod, name)
    end

    obj = getproperty(mod, name)
    (obj isa Function || obj isa Type) || return false
    return !startswith(strip(string(Docs.doc(obj))), "No documentation found.")
end

function docs_include_public_api(mod::Module, public_names)
    package_root = pkgdir(mod)
    isnothing(package_root) && return false

    docs_src = joinpath(package_root, "docs", "src")
    isdir(docs_src) || return false

    markdown = join(
        read(path, String)
            for path in readdir(docs_src; join = true) if endswith(path, ".md")
    )
    module_name = string(nameof(mod))
    autodocs_pattern = Regex(
        "```@autodocs[\\s\\S]*?Modules\\s*=\\s*\\[[^\\]]*\\b" *
            module_name *
            "\\b[^\\]]*\\]"
    )
    occursin(autodocs_pattern, markdown) && return true

    docs_blocks = [match.match for match in eachmatch(r"```@docs[\s\S]*?```", markdown)]
    return all(public_names) do name
        any(block -> occursin(string(name), block), docs_blocks)
    end
end

@testset "Public API documentation" begin
    public_names = public_api_names(FunctionWrappersWrappers)
    undocumented_names = filter(
        name -> !has_source_docstring(FunctionWrappersWrappers, name),
        public_names
    )

    @test isempty(undocumented_names)
    @test docs_include_public_api(FunctionWrappersWrappers, public_names)
end
