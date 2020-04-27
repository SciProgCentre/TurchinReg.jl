include("../src/TurchinReg.jl")
using .TurchinReg
using Documenter

makedocs(
    sitename = "TurchinReg.jl",
    format = Documenter.HTML(prettyurls = true),
    pages = Any[
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "User's Guide" => "users_guide.md",
        "Examples" => "examples.md",
    ]
)

deploydocs(
    repo = "github.com/mipt-npm/TurchinReg.jl.git",
    forcepush = true
)
