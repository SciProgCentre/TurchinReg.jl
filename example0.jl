#=
example0:
- Julia version: 1.1.0
- Author: ta_nyan
- Date: 2019-03-25
=#
# using Pkg
# Pkg.add("PyCall")
using PyCall
include("src/gauss_error.jl")

a = 0
b = 5

function phi(x)
    mu1 = 2.
    mu2 = 4.
    n1 = 4.
    n2 = 2.
    sig1 = 0.4
    sig2 = 0.5

    norm(n, mu, sig, x) = n / sqrt(2 * pi*sig^2) * exp(-(x - mu)^2 / (2 * sig^2))
    return norm(n1, mu1, sig1, x) + norm(n2, mu2, sig2, x)
end

x = range(a, stop=b, length=100)

using PyPlot


myplot = plot(x, phi.(x))
savefig(myplot, "example0.png")