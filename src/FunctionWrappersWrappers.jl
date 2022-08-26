module FunctionWrappersWrappers

using FunctionWrappers

export FunctionWrappersWrapper

struct FunctionWrappersWrapper{FW,FB}
  fw::FW
end
(fww::FunctionWrappersWrapper{FW,FB})(args::Vararg{Any,K}) where {FW,K,FB} = _call(fww.fw, args, Val{FB})

_call(fw::Tuple{FunctionWrappers.FunctionWrapper{R,A},Vararg}, arg::A, fww::FunctionWrappersWrapper{FW,FB}) where {R,A,FW,FB} = first(fw)(arg...)
_call(fw::Tuple{FunctionWrappers.FunctionWrapper{R,A1},Vararg}, arg::A2, fww::FunctionWrappersWrapper{FW,FB}) where {R,A1,A2,FW,FB} = _call(Base.tail(fw), arg, fww)
_call(::Tuple{}, arg, fww::FunctionWrappersWrapper{FW,false}) = throw("No matching function wrapper was found!")
_call(::Tuple{}, arg, fww::FunctionWrappersWrapper{FW,true}) = first(fww.fw).obj[](arg...)

function FunctionWrappersWrapper(f::F, argtypes::Tuple{Vararg{Any,K}}, rettypes::Tuple{Vararg{DataType,K}}, fallback::Val{FB}=Val{false}) where {F,K,FB}
  fwt = map(argtypes, rettypes) do A, R
    FunctionWrappers.FunctionWrapper{R,A}(f)
  end
  FunctionWrappersWrapper{typeof(fwt),FB}(fwt)
end

end
