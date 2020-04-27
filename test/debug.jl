include("../src/TurchinReg.jl")
using .TurchinReg

include("../src/utils/config.jl")
include("../src/bases/b_spline_implementation.jl")
include("../src/utils/check.jl")

using Test, QuadGK, Random, Distributions, Polynomials, PiecewisePolynomials

a = 0
b = 6.

function phi(x::Real)
    mu1 = 2.
    mu2 = 4.
    n1 = 4.
    n2 = 2.
    sig1 = 0.4
    sig2 = 0.5
    norm(n, mu, sig, x) = n / sqrt(2 * pi*sig^2) * exp(-(x - mu)^2 / (2 * sig^2))
    return norm(n1, mu1, sig1, x) + norm(n2, mu2, sig2, x)
end


my_kernel = getOpticsKernels("gaussian")
convolution = y -> quadgk(x -> my_kernel(x,y) * phi(x), a, b, rtol=10^-5, maxevals=10^7)[1]
y = collect(range(a, stop=b, length=30))
ftrue = convolution.(y)
sig = 0.05*ftrue + [0.01 for i = 1:Base.length(ftrue)]
noise = []
for sigma in sig
    n = rand(Normal(0., sigma), 1)[1]
    push!(noise, n)
end
f = ftrue + noise

println("Bases")
basis_fourier1 = FourierBasis(-5, 3, 40)
omega_fourier1 = omega(basis_fourier1, 1)
isapprox(omega_fourier1[3, 5], 0.0)
isapprox(discretize_kernel(basis_fourier1, my_kernel1, y)[2, 1], -0.48420495984870765)
basis_fourier2 = FourierBasis(0., 7, 10)
omega_fourier2 = omega(basis_fourier2, 2)
isapprox(omega_fourier2[2, 6], 0.0)
isapprox(discretize_kernel(basis_fourier2, my_kernel2, y)[8, 7], -8.381083487903461e-5)
basis_fourier3 = FourierBasis(-5, 150., 30)
omega_fourier3 = omega(basis_fourier3, 3)
isapprox(discretize_kernel(basis_fourier3, my_kernel3, y)[5, 5], 1.2509052207841052e-13)
# @test @returntrue basis_fourier4 = FourierBasis(-5., 3., 20) #Nan during the integration
basis_fourier4 = FourierBasis(-5., 20., 20)
omega_fourier4 = omega(basis_fourier4, 2)
isapprox(discretize_kernel(basis_fourier4, my_kernel4, y)[1, 1], -0.08506778038445181)
basis_fourier = FourierBasis(a, b, 10)
omega_fourier = omega(basis_fourier, 2)
isapprox(discretize_kernel(basis_fourier, my_kernel, y)[4, 6], 0.20558002069279474)
