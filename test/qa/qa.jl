using SciMLTesting, FunctionWrappersWrappers

run_qa(
    FunctionWrappersWrappers;
    ei_kwargs = (;
        # `FunctionWrappers.FunctionWrapper` is not declared `public` in its owning
        # module. It is the type this package exists to wrap: it appears in the
        # signature of every public constructor, and callers must build
        # `FunctionWrapper` tuples themselves to reach the inference-stable
        # construction path, so it cannot be hidden behind our own API. Drop this
        # entry once JuliaLang/FunctionWrappers.jl#41 is released and the compat
        # floor is raised.
        all_qualified_accesses_are_public = (; ignore = (:FunctionWrapper,)),
    ),
)
