#=
make:
- Julia version: 1.1.0
- Author: ta_nyan
- Date: 2019-06-30
=#

include("../src/kernels.jl")
include("../src/basis.jl")
include("../src/gauss_error.jl")
include("../src/vector.jl")
include("../src/config.jl")

using Documenter

makedocs(
    sitename = "StatReg.jl",
    pages = Any[
        "Home" => "introduction.md",
        "Getting started" => "getting_started.md",
        "User's Guide" => "users_guide.md",
        "Examples" => "examples.md",
    ]
)
