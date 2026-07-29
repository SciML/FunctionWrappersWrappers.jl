using SciMLTesting, FunctionWrappersWrappers

run_qa(
    FunctionWrappersWrappers;
    ei_kwargs = (;
        # Qualified accesses to names their owning module has not declared `public`.
        # Both are irreducible — neither can be promoted at its source:
        #   FunctionWrappers.FunctionWrapper is the type this package exists to wrap. It
        #     appears in the signature of every public constructor, and callers must build
        #     `FunctionWrapper` tuples themselves to reach the inference-stable path, so it
        #     cannot be hidden. FunctionWrappers.jl is JuliaLang-owned.
        #   TruncatedStacktraces.@truncate_stacktrace is that package's only user-facing
        #     macro and is never exported. SciML/TruncatedStacktraces.jl is archived and
        #     read-only, so it cannot be made public there.
        all_qualified_accesses_are_public = (;
            ignore = (:FunctionWrapper, Symbol("@truncate_stacktrace")),
        ),
    ),
)
