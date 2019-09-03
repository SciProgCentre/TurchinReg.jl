include("../src/gauss_error.jl")
include("../src/kernels.jl")
include("../src/mcmc.jl")
include("../src/utils.jl")

using Test, QuadGK, Random, Distributions

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

@testset "B-splines implementation" begin
    @test @returntrue b_spline = BSpline(1, 2, [1, 2, 3, 4, 5])
    @test @returntrue derivative(b_spline, 3.5, 3)
    @test @returntrue b_spline(3)
    # можно написать больше тестов с разными параметрами и разными типами параметров (см. типы в коде)
end

@testset "Kernels" begin
    @test @returntrue getOpticsKernels("rectangular", 2)
    @test @returntrue my_kernel = getOpticsKernels("diffraction", 2.)
    @test @returntrue my_kernel(0, 1)
    @test @returntrue BaseFunction("fun", 2, 3)
    @test @returntrue BaseFunction(x -> x, 2., 3)
    @test @returntrue BaseFunction(2, 2., 5.)
    #дописать подобные тесты для всех ядер в файлеkernels.jl,
    #поменять alpha и точку вычисления функции как-нибудь рандомно
end

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

@testset "Fourier basis" begin
    @test @returntrue basis_fourier = FourierBasis(a, b, 10)
    @test @returntrue omega_fourier = omega(basis_fourier, 2)
    # @test @returntrue discretize_kernel(basis_fourier, my_kernel, )
    #написать еще тесты для проверки параметров
end

@testset "Cubic spline basis" begin
    @test @returntrue basis_cubic_spline = CubicSplineBasis(y, "dirichlet")
    @test @returntrue omega_cubic_spline = omega(basis_cubic_spline, 2)
    #написать еще тесты для проверки параметров
end

@testset "Legendre basis" begin
    @test @returntrue basis_legendre = LegendreBasis(a, b, 10)
    @test @returntrue omega_legendre = omega(basis_legendre, 2)
    #написать еще тесты для проверки параметров
end

@testset "Bernstein basis" begin
    @test @returntrue basis_bernstein = BernsteinBasis(a, b, 10)
    @test @returntrue omega_bernstein = omega(basis_bernstein, 2)
    #написать еще тесты для проверки параметров
end


@testset "Gaussian errors" begin
    @test @returntrue model_fourier = GaussErrorUnfolder(basis_fourier, [omega_fourier], "EmpiricalBayes", nothing, [1e-8], [10.], [0.3])
    @test @returntrue model_cubic_spline = GaussErrorUnfolder(basis_cubic_spline, [omega_cubic_spline], "EmpiricalBayes", nothing, [1e-8], [10.], [0.3])
    @test @returntrue model_legendre = GaussErrorUnfolder(basis_legendre, [omega_legendre], "EmpiricalBayes", nothing, [1e-8], [10.], [0.3])
    @test @returntrue model_bernstein = GaussErrorUnfolder(basis_bernstein, [omega_bernstein], "EmpiricalBayes", nothing, [1e-8], [10.], [0.3])

    @test @returntrue phi_reconstruct_fourier = solve(model_fourier, my_kernel, f, sig, y)
    @test @returntrue phi_reconstruct_cubic_spline = solve(model_cubic_spline, my_kernel, f, sig, y)
    @test @returntrue phi_reconstruct_legendre = solve(model_legendre, my_kernel, f, sig, y)
    @test @returntrue phi_reconstruct_bernstein = solve(model_bernstein, my_kernel, f, sig, y)

    #написать аналогично для других базисов и для GaussErrorMatrixUnfolder, метод "User"
end

@testset "MCMC" begin
    @test @returntrue model_mcmc__cubic_spline = MCMCUnfolder(basis_cubic_spline, [omega_cubic_spline], "EmpiricalBayes", nothing, [1e-5], [10.], [0.5])
    #дописать недостающие тесты (тут только для сплайнов, для других базисов не надо)
end

x = collect(range(a, stop=b, length=500))

@testset "Results representation" begin
    @test @returntrue phivec_fourier = PhiVec(phi_reconstruct_fourier["coeff"], basis_fourier, phi_reconstruct_fourier["errors"])
    @test @returntrue phivec_cubic_spline = PhiVec(phi_reconstruct_cubic_spline["coeff"], basis_cubic_spline, phi_reconstruct_cubic_spline["errors"])
    @test @returntrue phivec_legendre = PhiVec(phi_reconstruct_legendre["coeff"], basis_legendre, phi_reconstruct_legendre["errors"])
    @test @returntrue phivec_bernstein = PhiVec(phi_reconstruct_bernstein["coeff"], basis_bernstein, phi_reconstruct_bernstein["errors"])

    @test @returntrue phi_reconstructed_fourier = phivec_fourier.phi_function.(x)
    @test @returntrue phi_reconstructed_errors_fourier = phivec_fourier.error_function.(x)

    @test @returntrue phi_reconstructed_cubic_spline = phivec_cubic_spline.phi_function.(x)
    @test @returntrue phi_reconstructed_errors_cubic_spline = phivec_cubic_spline.error_function.(x)

    @test @returntrue phi_reconstructed_legendre = phivec_legendre.phi_function.(x)
    @test @returntrue phi_reconstructed_errors_legendre = phivec_legendre.error_function.(x)

    @test @returntrue phi_reconstructed_bernstein= phivec_bernstein.phi_function.(x)
    @test @returntrue phi_reconstructed_errors_bernstein = phivec_bernstein.error_function.(x)
end
