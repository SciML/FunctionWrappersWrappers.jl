using FunctionWrappersWrappers, BenchmarkTools

const SUITE = BenchmarkGroup()

# FunctionWrappersWrapper wraps one function across multiple argument signatures
argtypes = (Tuple{Float64, Float64}, Tuple{Int, Int})
rettypes = (Float64, Int)

# =============================================================================
# Construction
# =============================================================================

SUITE["construct"] = BenchmarkGroup()

SUITE["construct"]["default"] = @benchmarkable FunctionWrappersWrapper(
    +, $argtypes, $rettypes
)
SUITE["construct"]["dictcache"] = @benchmarkable FunctionWrappersWrapper(
    +, $argtypes, $rettypes; cache = DictCache()
)
SUITE["construct"]["no_policy"] = @benchmarkable FunctionWrappersWrapper(
    +, $argtypes, $rettypes; policy = AllowAll()
)

# =============================================================================
# Calls and metadata
# =============================================================================

SUITE["call"] = BenchmarkGroup()

fww = FunctionWrappersWrapper(+, argtypes, rettypes)
fww_dict = FunctionWrappersWrapper(+, argtypes, rettypes; cache = DictCache())

SUITE["call"]["float_args"] = @benchmarkable $fww(1.0, 2.0)
SUITE["call"]["int_args"] = @benchmarkable $fww(1, 2)
SUITE["call"]["dictcache"] = @benchmarkable $fww_dict(1.0, 2.0)

SUITE["call"]["unwrap"] = @benchmarkable unwrap($fww)
SUITE["call"]["wrapped_signatures"] = @benchmarkable wrapped_signatures($fww)
SUITE["call"]["wrapped_return_types"] = @benchmarkable wrapped_return_types($fww)

# =============================================================================
# Vector of wrappers
# =============================================================================

SUITE["vector"] = BenchmarkGroup()

fww1 = FunctionWrappersWrapper(sin, (Tuple{Float64},), (Float64,))
FWW1 = typeof(fww1)
fww_vec = FWW1[sin, tan, log]

SUITE["vector"]["getindex_call"] = @benchmarkable $fww_vec[1](0.5)
SUITE["vector"]["construct"] = @benchmarkable $FWW1[sin, tan, log]
