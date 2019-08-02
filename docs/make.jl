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
include("../src/mcmc.jl")
include("../src/check.jl")
# include("../src/StatReg.jl")
# using ../srcStatReg

using Documenter

makedocs(
    sitename = "StatReg.jl",
    format = Documenter.HTML(prettyurls = true),
    pages = Any[
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "User's Guide" => "users_guide.md",
        "Examples" => "examples.md",
    ]
)

deploydocs(
    repo = "github.com/mipt-npm/StatReg.jl.git",
    forcepush = true
)
