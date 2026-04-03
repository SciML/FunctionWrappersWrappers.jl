module FunctionWrappersWrappersMooncakeExt

using FunctionWrappersWrappers
using FunctionWrappers: FunctionWrapper
using Mooncake: @is_primitive, MinimalCtx, CoDual, NoRData, primal
import Mooncake: rrule!!

# Make FunctionWrappersWrapper calls a primitive so Mooncake bypasses the
# internal _call dispatch and directly delegates to the FunctionWrapper's rrule
# (from MooncakeFunctionWrappersExt), which auto-unwraps to the original function.
@is_primitive MinimalCtx Tuple{<:FunctionWrappersWrapper, Vararg}

function rrule!!(
    fww_dual::CoDual{<:FunctionWrappersWrapper}, args::Vararg{CoDual, N}
) where {N}
    fww = primal(fww_dual)
    fww_fdata = fww_dual.dx  # FData{NamedTuple{(:fw, :cache_storage), ...}}

    # Extract first FunctionWrapper and its tangent
    fw = first(fww.fw)
    fw_tang = first(fww_fdata.data.fw)
    fw_dual = CoDual(fw, fw_tang)

    # Delegate to FunctionWrapper's rrule (from MooncakeFunctionWrappersExt)
    y, fw_pb = rrule!!(fw_dual, args...)

    function fww_pullback(dy)
        result = fw_pb(dy)
        # result = (NoRData(), dx...) from FunctionWrapper's pullback
        return (NoRData(), Base.tail(result)...)
    end

    return y, fww_pullback
end

end
