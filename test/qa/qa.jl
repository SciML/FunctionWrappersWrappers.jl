using SciMLTesting, FunctionWrappersWrappers
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
    api_docs_kwargs = (; rendered = true),
)
