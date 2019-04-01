include("gauss_error.jl")
include("kernels.jl")


a = 0.
b = 4.

function phi(x::Float64)
    mu1 = 2.
    mu2 = 4.
    n1 = 4.
    n2 = 2.
    sig1 = 0.4
    sig2 = 0.5

    norm(n, mu, sig, x) = n / sqrt(2 * pi*sig^2) * exp(-(x - mu)^2 / (2 * sig^2))
    return norm(n1, mu1, sig1, x)# + norm(n2, mu2, sig2, x)
end

x = range(a, stop=b, length=100)

function kernel(x::Float64, y::Float64)
    return getOpticsKernels("triangular")(x, y)
end

convolution = y -> quadgk(x -> kernel(x,y) * phi(x), a, b, rtol=10^-5, maxevals=10^2)[1]

y = Float64[]
for i = 0:50
    push!(y, a + (b - a) * i/50)
end

ftrue = convolution.(y)
sig = 0.01*ftrue + [0.01 for i = 1:Base.length(ftrue)]
using Compat, Random, Distributions
noise = []
for sigma in sig
    n = rand(Normal(0., sigma), 1)[1]
    push!(noise, n)
end
f = ftrue + noise


basis = FourierBasis(a, b, 7)
model = GaussErrorUnfolder(basis, omega(basis, 2))

a = 0.
c = 4.
x = Float64[]
for i = 0:100
    push!(x, a + (c - a) * i/100)
end
# for f in basis.basis_functions
#     plot(x, f.f.(x))
# end

using Profile
phi_reconstruct1 = solve(model, kernel, f, sig, y)
Profile.clear()
@profile phi_reconstruct = solve(model, kernel, f, sig, y)
Juno.profiletree()
Juno.profiler()

# function g()
#     a = []
#     for i = 0:5
#         push!(a, 10.0)
#     end
# end
#
# using Profile
#
# Profile.clear()
# @profile g()
# Juno.profiletree()
# Juno.profiler()
