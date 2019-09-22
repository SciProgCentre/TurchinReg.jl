include("../src/StatReg.jl")
using .StatReg
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
