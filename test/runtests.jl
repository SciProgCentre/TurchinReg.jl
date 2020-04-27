include("../src/TurchinReg.jl")
using .TurchinReg

include("../src/utils/config.jl")
include("../src/bases/b_spline_implementation.jl")
include("../src/utils/check.jl")


using Test, QuadGK, Random, Distributions, Polynomials, PiecewisePolynomials

Random.seed!(1234)

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


@testset "Config" begin
    @test config.MAXEVALS_QUADGK >= 300 && config.MAXEVALS_QUADGK > 0
    @test config.X_TOL_OPTIM <= 1e-3 && config.X_TOL_OPTIM > 0
    @test config.RTOL_QUADGK <= 1e-3 && config.RTOL_QUADGK > 0
end

b_spline = BSpline(1, 2, [1, 2, 3, 4, 5])
b_spline = BSpline(1, 2, [1., 2, 3., 4, 5])

@testset "B-splines implementation" begin
    @test isapprox(b_spline.func(3), 0.5)
    @test isapprox(b_spline.func(4.), 0.5)
    @test isapprox(PiecewisePolynomials.derivative(b_spline.func)(3), 1)
    @test isapprox(b_spline(3), 0.5)
    @test isapprox(b_spline(1.98), 0)
end


my_kernel1 = getOpticsKernels("diffraction", -1.)
my_kernel2 = getOpticsKernels("gaussian", 69.)
my_kernel3 = getOpticsKernels("triangular", 123456234)
my_kernel4 = getOpticsKernels("diffraction", -12.)
my_kernel5 = getOpticsKernels("exponential", 1.)
my_kernel6 = getOpticsKernels("heaviside", -13.)
my_kernel7 = getOpticsKernels("gaussian")

@testset "Kernels" begin
    @test isapprox(my_kernel1(-0, 4), -0.0070116734630404345)
    @test isapprox(my_kernel2(1000000, -2), 0.0)
    @test isapprox(my_kernel3(304, 64), 8.100020740882498e-9)
    @test isapprox(my_kernel4(12, 15.), -0.06265939172909665)
    @test isapprox(my_kernel5(123, -15.), 5.708824047855524e-84)
    @test isapprox(my_kernel6(0., -0), 0.5)
    @test isapprox(my_kernel7(0., -0), 0.9394372786996513)
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

basis_fourier1 = FourierBasis(-5, 3, 40)
omega_fourier1 = omega(basis_fourier1, 1)
basis_fourier2 = FourierBasis(0., 7, 10)
omega_fourier2 = omega(basis_fourier2, 2)
basis_fourier3 = FourierBasis(-5, 150., 30)
omega_fourier3 = omega(basis_fourier3, 3)
basis_fourier4 = FourierBasis(-5., 20., 20)
omega_fourier4 = omega(basis_fourier4, 2)
basis_fourier = FourierBasis(a, b, 10)
omega_fourier = omega(basis_fourier, 2)

@testset "Fourier basis" begin
    @test isapprox(omega_fourier1[3, 5], 0.0)
    @test isapprox(discretize_kernel(basis_fourier1, my_kernel1, y)[2, 1], -0.48420495984870765)
    @test isapprox(omega_fourier2[2, 6], 0.0)
    @test isapprox(discretize_kernel(basis_fourier2, my_kernel2, y)[8, 7], -8.381083487903461e-5)
    @test isapprox(discretize_kernel(basis_fourier3, my_kernel3, y)[5, 5], 1.2509052207841052e-13)
    @test isapprox(discretize_kernel(basis_fourier4, my_kernel4, y)[1, 1], -0.39201103669339094)
    @test isapprox(discretize_kernel(basis_fourier, my_kernel, y)[4, 6], 0.20558002069279474)
end

basis_cubic_spline1 = CubicSplineBasis([1, 2, 3, 4, 5, 6, 7, 8], ("dirichlet", nothing))
omega_cubic_spline1 = omega(basis_cubic_spline1, 2)
basis_cubic_spline2 = CubicSplineBasis([2, 3, 3.5, 4, 5, 6, 7, 8, 12, 16], ("dirichlet", "dirichlet"))
omega_cubic_spline2 = omega(basis_cubic_spline2, 1)
basis_cubic_spline3 = CubicSplineBasis([8, 8.1, 8.3, 9, 10, 11, 12, 100, 112, 112.5, 120])
omega_cubic_spline3 = omega(basis_cubic_spline3, 2)
basis_cubic_spline = CubicSplineBasis(y, "dirichlet")
omega_cubic_spline = omega(basis_cubic_spline, 2)

@testset "Cubic spline basis" begin
    @test isapprox(omega_cubic_spline1[2, 4], 0.08680555555555003)
    @test isapprox(omega_cubic_spline2[5, 8], -0.0006172839507030403)
    @test isapprox(omega_cubic_spline3[3, 5], -0.6862745098042069)
    @test isapprox(omega_cubic_spline[1, 7], 0)
end


basis_legendre1 = LegendreBasis(-5, 10., 3)
omega_legendre1 = omega(basis_legendre1, 3)
basis_legendre2 = LegendreBasis(2, 2.5, 9)
omega_legendre2 = omega(basis_legendre2, 1)
basis_legendre3 = LegendreBasis(1, 10, 20)
omega_legendre3 = omega(basis_legendre3, 2)
basis_legendre4 = LegendreBasis(0, 12, 50)
omega_legendre4 = omega(basis_legendre4, 4)
basis_legendre = LegendreBasis(a, b, 10)
omega_legendre = omega(basis_legendre, 2)

@testset "Legendre basis" begin
    @test isapprox(omega_legendre1[2, 3], 0)
    @test isapprox(omega_legendre2[5, 8], 1.5543122344752192e-15)
    @test isapprox(omega_legendre3[3, 5], 0.6584362139917693)
    @test isapprox(omega_legendre4[1, 7], 0)
    @test isapprox(omega_legendre[6, 3], -2.498001805406602e-16)
end


basis_bernstein1 = BernsteinBasis(1, 10, 10, ("dirichlet", "dirichlet"))
omega_bernstein1 = omega(basis_bernstein1, 0)
basis_bernstein2 = BernsteinBasis(20, 25., 20, ("dirichlet", nothing))
omega_bernstein2 = omega(basis_bernstein2, 1)
basis_bernstein3 = BernsteinBasis(-4.5, 8, 5, "dirichlet")
omega_bernstein3 = omega(basis_bernstein3, 2)
basis_bernstein4 = BernsteinBasis(6., 7., 10)
omega_bernstein4 = omega(basis_bernstein4, 1)
basis_bernstein = BernsteinBasis(a, b, 10)
omega_bernstein = omega(basis_bernstein, 2)

@testset "Bernstein basis" begin
    @test isapprox(omega_bernstein1[2, 4], 0.09518681614522237)
    @test isapprox(omega_bernstein2[5, 8], 1.3317242238265603)
    @test isapprox(omega_bernstein3[3, 4], 0.3174400000000001)
    @test isapprox(omega_bernstein4[1, 7], 0.10121457489878544)
    @test isapprox(omega_bernstein[6, 3], 5.056561085972847)
end

res0 = solve(
    basis_cubic_spline,
    f, sig, my_kernel, y,
    Analytically(),
    ArgmaxBAT(),
    [omega_cubic_spline],
    )

res1 = solve(
    basis_cubic_spline,
    f, sig, my_kernel, y,
    Analytically(),
    User([0.01]),
    [omega_cubic_spline],
    )

res2 = solve(
    basis_cubic_spline,
    f, sig, my_kernel, y,
    Analytically(),
    ArgmaxOptim(),
    [omega_cubic_spline],
    )

res3 = solve(
    basis_cubic_spline,
    f, sig, my_kernel, y,
    BATSampling(),
    User([0.09]),
    [omega_cubic_spline],
    PhiBounds()
    )

res4 = solve(
    basis_cubic_spline,
    f, sig, my_kernel, y,
    AHMCSampling(),
    User([0.09]),
    [omega_cubic_spline],
    PhiBounds()
    )

res5 = solve(
    basis_cubic_spline,
    f, sig, my_kernel, y,
    DHMCSampling(),
    User([0.09]),
    [omega_cubic_spline],
    PhiBounds()
    )

@testset "Solve" begin
    @test abs(res0.solution_function(1.5) - phi(1.5)) < 2
    @test abs(res1.solution_function(3) - phi(3)) < 0.1
    @test abs(res2.solution_function(4) - phi(4)) < 1
    @test abs(res3.solution_function(2) - phi(2)) < 2
    @test abs(res4.solution_function(5.6) - phi(5.6)) < 0.1
    @test abs(res5.solution_function(1.3) - phi(1.3)) < 1
end
