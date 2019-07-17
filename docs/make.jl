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
# include("../src/StatReg.jl")
# using StatReg

using Documenter

makedocs(
    sitename = "StatReg.jl",
    format = Documenter.HTML(
        prettyurls = !("local" in ARGS),
        canonical = "https://mipt-npm.github.io/StatReg.jl/stable"
    ),
    pages = Any[
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "User's Guide" => "users_guide.md",
        "Examples" => "examples.md",
    ]
)
