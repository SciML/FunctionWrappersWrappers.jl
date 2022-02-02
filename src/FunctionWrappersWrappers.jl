module FunctionWrappersWrappers

using FunctionWrappers

export FunctionWrappersWrapper

struct FunctionWrappersWrapper{FW}
  fw::FW
end
(fww::FunctionWrappersWrapper{FW})(args::Vararg{Any,K}) where {FW,K} = _call(fww.fw, args)

_call(fw::Tuple{FunctionWrappers.FunctionWrapper{R,A},Vararg}, arg::A) where {R,A} = first(fw)(arg...)
_call(fw::Tuple{FunctionWrappers.FunctionWrapper{R,A1},Vararg}, arg::A2) where {R,A1,A2} = _call(Base.tail(fw), arg)
_call(::Tuple{}, arg) = throw("No matching function wrapper was found!")

function FunctionWrappersWrapper(f::F, argtypes::Tuple{Vararg{Any,K}}, rettypes::Tuple{Vararg{DataType,K}}) where {F,K}
  fwt = map(argtypes, rettypes) do A, R
    FunctionWrappers.FunctionWrapper{R,A}(f)
  end
  FunctionWrappersWrapper(fwt)
end

end
