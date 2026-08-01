using SciMLTesting, FunctionWrappersWrappers

# ExplicitImports only sees an extension module once its triggers are loaded
# (`Base.get_extension` returns `nothing` otherwise), so loading the weakdeps here
# is what puts `FunctionWrappersWrappersEnzymeExt` and
# `FunctionWrappersWrappersMooncakeExt` under QA at all.
using Enzyme, EnzymeCore, Mooncake

run_qa(
    FunctionWrappersWrappers;
    ei_kwargs = (;
        # `FunctionWrappers.FunctionWrapper` is not declared `public` in its owning
        # module. It is the type this package exists to wrap: it appears in the
        # signature of every public constructor, and callers must build
        # `FunctionWrapper` tuples themselves to reach the inference-stable
        # construction path, so it cannot be hidden behind our own API. Drop this
        # entry once JuliaLang/FunctionWrappers.jl#35 is released and the compat
        # floor is raised.
        #
        # `Mooncake` declares only its top-level differentiation entry points public
        # (`value_and_gradient!!`, `prepare_gradient_cache`, `Config`, `Dual`, ...).
        # Its whole rule-authoring interface — the thing an extension that teaches
        # Mooncake about a new callable has to use — is documented but not `public`:
        # `build_rrule`, `rrule!!`, `tangent_type` and `primal` below, plus the
        # explicitly-imported names in the matching list. There is no public spelling
        # for writing a Mooncake rule, so these stay ignored until Mooncake marks its
        # rule API `public`.
        #
        # `Core.Typeof` is the standard idiom for building a call-signature tuple type
        # (`typeof` is wrong for arguments that are themselves types); `Core` does not
        # declare it public and there is no `Base` equivalent.
        #
        # `EnzymeCore.EnzymeRules` exports the config accessors a rule reads
        # (`needs_primal`, `overwritten`, `runtime_activity`, ...) but not the generic
        # functions a rule author adds methods to. `forward`, `augmented_primal`,
        # `reverse`, `inactive_type` and `strong_zero` are exactly the documented
        # extension points of the EnzymeRules interface, and there is no public
        # spelling for them.
        all_qualified_accesses_are_public = (;
            ignore = (
                :FunctionWrapper,
                :augmented_primal, :forward, :inactive_type, :reverse, :strong_zero,
                :build_rrule, :primal, Symbol("rrule!!"), :tangent_type,
                :Typeof,
            ),
        ),
        all_explicit_imports_are_public = (;
            ignore = (
                Symbol("@is_primitive"), :CoDual, :MinimalCtx, :NoRData, :NoTangent,
                :fdata, :zero_tangent,
            ),
        ),
    ),
)
