#=
make:
- Julia version: 1.1.0
- Author: ta_nyan
- Date: 2019-06-30
=#

using Documenter, OhMyREPL

makedocs(
    sitename = "Statreg",
    pages = Any[
        "Home" => "myfirstdoc.md",
        "Hello" => "myseconddoc.md",
    ]
)

deploydocs(
    repo = "github.com/KristofferC/OhMyREPL.jl.git",
)