# FunctionWrappersWrappers.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/FunctionWrappersWrappers/stable/)

[![codecov](https://codecov.io/gh/SciML/FunctionWrappersWrappers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/SciML/FunctionWrappersWrappers.jl)
[![Build Status](https://github.com/SciML/FunctionWrappersWrappers.jl/actions/workflows/Tests.yml/badge.svg?branch=main)](https://github.com/SciML/FunctionWrappersWrappers.jl/actions/workflows/Tests.yml?query=branch%3Amain)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

If wrapping your functions once makes them better, why not try wrapping them twice?

FunctionWrappersWrappers.jl provides a callable, type-stable wrapper for a finite set of
function signatures. Calls outside that set can be rejected or delegated to the original
function with a configurable cache.

For examples and the API, see the [documentation](https://docs.sciml.ai/FunctionWrappersWrappers/stable/).
