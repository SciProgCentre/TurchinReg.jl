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
    @test @returntrue b_spline = BSpline(1, 2, [1., 2, 3., 4, 5])
    @test @returntrue b_spline.func(3)
    @test @returntrue b_spline.func(4.)
    @test @returntrue der(b_spline.func)(3)
    @test @returntrue der(der(b_spline.func)) == der_order(b_spline.func, 2)
    @test @returntrue b_spline(3)
    @test @returntrue b_spline(2.)
end

@testset "Kernels" begin

    @test @returntrue getOpticsKernels("rectangular", 10000)

    #diffraction
    @test @returntrue my_kernel1 = getOpticsKernels("diffraction", -1.)
    @test @returntrue my_kernel1(-00, 4)

    #gaussian
    @test @returntrue my_kernel2 = getOpticsKernels("gaussian", 69.)
    @test @returntrue my_kernel2(1000000, -2)

    #triangular
    @test @returntrue my_kernel3 = getOpticsKernels("triangular", 123456234)
    @test @returntrue my_kernel3(304, 64)

    #dispersive
    @test @returntrue my_kernel4 = getOpticsKernels("diffraction", -12345678.)
    @test @returntrue my_kernel4(12, 12.)

    #exponential
    @test @returntrue my_kernel5 = getOpticsKernels("exponential", -0.)
    @test @returntrue my_kernel5(123, -1234567.)

    #heaviside
    @test @returntrue my_kernel6 = getOpticsKernels("heaviside", -1337.)
    @test @returntrue my_kernel6(0., -0)

    @test @returntrue my_kernel = getOpticsKernels("gaussian")

    @test @returntrue BaseFunction("fun", 2, 3)
    @test @returntrue BaseFunction(x -> x, 2., 3)
    @test @returntrue BaseFunction(2, 2., 5.)
    @test @returntrue BaseFunction(x -> x * x, 2222222., -10101.00001)
    @test @returntrue BaseFunction(-0.000001, 2, 5.)
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

    @test @returntrue basis_fourier1 = FourierBasis(-5, 3, 40)
    @test @returntrue omega_fourier1 = omega(basis_fourier1, 1)
    @test @returntrue discretize_kernel(basis_fourier1, my_kernel1, y)
    @test @returntrue basis_fourier2 = FourierBasis(0., 7, 10)
    @test @returntrue omega_fourier2 = omega(basis_fourier2, 2)
    @test @returntrue discretize_kernel(basis_fourier2, my_kernel2, y)
    @test @returntrue basis_fourier3 = FourierBasis(-5, 150., 30)
    @test @returntrue omega_fourier3 = omega(basis_fourier3, 3)
    @test @returntrue discretize_kernel(basis_fourier3, my_kernel3, y)
    # @test @returntrue basis_fourier4 = FourierBasis(-5., 3., 20) #Nan during the integration
    @test @returntrue basis_fourier4 = FourierBasis(-5., 20., 20)
    @test @returntrue omega_fourier4 = omega(basis_fourier4, 2)
    @test @returntrue discretize_kernel(basis_fourier4, my_kernel4, y)

    @test @returntrue basis_fourier = FourierBasis(a, b, 10)
    @test @returntrue omega_fourier = omega(basis_fourier, 2)
    @test @returntrue discretize_kernel(basis_fourier, my_kernel, y)
end

@testset "Cubic spline basis" begin

    @test @returntrue basis_cubic_spline1 = CubicSplineBasis([1, 2, 3, 4, 5, 6, 7, 8], ("dirichlet", nothing))
    @test @returntrue omega_cubic_spline1 = omega(basis_cubic_spline1, 2)

    @test @returntrue basis_cubic_spline2 = CubicSplineBasis([2, 3, 3.5, 4, 5, 6, 7, 8, 12, 16], ("dirichlet", "dirichlet"))
    @test @returntrue omega_cubic_spline2 = omega(basis_cubic_spline2, 1)

    @test @returntrue basis_cubic_spline3 = CubicSplineBasis([8, 8.1, 8.3, 9, 10, 11, 12, 100, 112, 112.5, 120])
    @test @returntrue omega_cubic_spline3 = omega(basis_cubic_spline3, 2)

    @test @returntrue basis_cubic_spline = CubicSplineBasis(y, "dirichlet")
    @test @returntrue omega_cubic_spline = omega(basis_cubic_spline, 2)
end

@testset "Legendre basis" begin

    @test @returntrue basis_legendre = LegendreBasis(-5, 10., 3)
    @test @returntrue omega_legendre = omega(basis_legendre, 3)
    @test @returntrue basis_legendre = LegendreBasis(2, 2.5, 9)
    @test @returntrue omega_legendre = omega(basis_legendre, 2)
    @test @returntrue basis_legendre = LegendreBasis(1, 10, 20)
    @test @returntrue omega_legendre = omega(basis_legendre, 2)
    @test @returntrue basis_legendre = LegendreBasis(0, 12, 50)
    @test @returntrue omega_legendre = omega(basis_legendre, 2)

    @test @returntrue basis_legendre = LegendreBasis(a, b, 10)
    @test @returntrue omega_legendre = omega(basis_legendre, 2)
end

@testset "Bernstein basis" begin

    @test @returntrue basis_bernstein = BernsteinBasis(1, 10, 10, ("dirichlet", "dirichlet"))
    @test @returntrue omega_bernstein = omega(basis_bernstein, 0)
    @test @returntrue basis_bernstein = BernsteinBasis(20, 25., 20, ("dirichlet", nothing))
    @test @returntrue omega_bernstein = omega(basis_bernstein, 1)
    @test @returntrue basis_bernstein = BernsteinBasis(-4.5, 8, 5, "dirichlet")
    @test @returntrue omega_bernstein = omega(basis_bernstein, 2)
    @test @returntrue basis_bernstein = BernsteinBasis(6., 7., 10)
    @test @returntrue omega_bernstein = omega(basis_bernstein, 1)

    @test @returntrue basis_bernstein = BernsteinBasis(a, b, 10)
    @test @returntrue omega_bernstein = omega(basis_bernstein, 2)
end


@testset "Gaussian errors" begin
    @test @returntrue model_fourier = GaussErrorUnfolder(basis_fourier, [omega_fourier], "EmpiricalBayes", nothing, [1e-8], [10.], [0.3])
    @test @returntrue model_cubic_spline = GaussErrorUnfolder(basis_cubic_spline, [omega_cubic_spline], "User", [0.3])
    @test @returntrue model_legendre = GaussErrorUnfolder(basis_legendre, [omega_legendre], "EmpiricalBayes", [1.], [1e-8], [10.], [0.3])
    @test @returntrue model_bernstein = GaussErrorUnfolder(basis_bernstein, [omega_bernstein], "EmpiricalBayes", nothing, [1e-8], [10.], [0.3])

    @test @returntrue phi_reconstruct_fourier = solve(model_fourier, my_kernel, f, sig, y)
    @test @returntrue phi_reconstruct_cubic_spline = solve(model_cubic_spline, getOpticsKernels("heaviside"), f, sig, y)
    @test @returntrue phi_reconstruct_legendre = solve(model_legendre, getOpticsKernels("rectangular"), f, sig, y)
    @test @returntrue phi_reconstruct_bernstein = solve(model_bernstein, getOpticsKernels("triangular"), f, sig, y)

    @test @returntrue model_fourier_matrix = GaussErrorMatrixUnfolder([omega_fourier], "EmpiricalBayes", nothing, [1e-8], [10.], [0.3])
    @test @returntrue model_cubic_spline_matrix = GaussErrorMatrixUnfolder([omega_cubic_spline], "User", [0.3])
    @test @returntrue model_legendre_matrix = GaussErrorMatrixUnfolder([omega_legendre], "EmpiricalBayes", [1.], [1e-8], [10.], [0.3])
    @test @returntrue model_bernstein_matrix = GaussErrorMatrixUnfolder([omega_bernstein], "EmpiricalBayes", nothing, [1e-8], [10.], [0.3])

    @test @returntrue phi_reconstruct_fourier_matrix = solve(model_fourier_matrix, discretize_kernel(basis_fourier, my_kernel, y), f, sig)
    @test @returntrue phi_reconstruct_cubic_spline_matrix = solve(model_cubic_spline_matrix, discretize_kernel(basis_cubic_spline, getOpticsKernels("heaviside"), y), f, sig)
    @test @returntrue phi_reconstruct_legendre_matrix = solve(model_legendre_matrix, discretize_kernel(basis_legendre, getOpticsKernels("rectangular"), y), f, sig)
    @test @returntrue phi_reconstruct_bernstein_matrix = solve(model_bernstein_matrix, discretize_kernel(basis_bernstein, getOpticsKernels("triangular"), y), f, sig)
end

@testset "MCMC" begin

    @test @returntrue model_cubic_spline1 = MCMCUnfolder(basis_cubic_spline, [omega_cubic_spline], "EmpiricalBayes", nothing, [1e-8], [10.], [0.3])
    @test @returntrue model_cubic_spline2 = MCMCUnfolder(basis_cubic_spline, [omega_cubic_spline], "User", [0.3])
    @test @returntrue model_cubic_spline3 = MCMCUnfolder(basis_cubic_spline, [omega_cubic_spline], "EmpiricalBayes", [1.], [1e-8], [10.], [0.3])
    @test @returntrue model_cubic_spline4 = MCMCUnfolder(basis_cubic_spline, [omega_cubic_spline], "EmpiricalBayes", nothing, [1e-8], [10.], [0.3])

    @test @returntrue model_1, line1, inits1, samples1, burnin1, thin1, chains1 = solve(model_cubic_spline1, my_kernel, f, sig, y)
    @test @returntrue model_2, line2, inits2, samples2, burnin2, thin2, chains2 = solve(model_cubic_spline2, getOpticsKernels("heaviside"), f, sig, y)
    @test @returntrue model_3, line3, inits3, samples3, burnin3, thin3, chains3 = solve(model_cubic_spline3, getOpticsKernels("rectangular"), f, sig, y)
    @test @returntrue model_4, line4, inits4, samples4, burnin4, thin4, chains4 = solve(model_cubic_spline4, getOpticsKernels("triangular"), f, sig, y)

    @test @returntrue sim1 = mcmc(model_1, line1, inits1, 100, burnin=burnin1, thin=thin1, chains=chains1)
    @test @returntrue res1 = get_values(sim1, chains1, length(basis_cubic_spline))
    @test @returntrue sim2 = mcmc(model_2, line2, inits2, 100, burnin=burnin2, thin=thin2, chains=chains2)
    @test @returntrue res2 = get_values(sim2, chains2, length(basis_cubic_spline))
    @test @returntrue sim3 = mcmc(model_3, line3, inits3, 100, burnin=burnin3, thin=thin3, chains=chains3)
    @test @returntrue res3 = get_values(sim3, chains3, length(basis_cubic_spline))
    @test @returntrue sim4 = mcmc(model_4, line4, inits4, 100, burnin=burnin4, thin=thin4, chains=chains4)
    @test @returntrue res4 = get_values(sim4, chains4, length(basis_cubic_spline))

    @test @returntrue model_cubic_spline1_matrix = MCMCMatrixUnfolder([omega_cubic_spline], "EmpiricalBayes", nothing, [1e-8], [10.], [0.3])
    @test @returntrue model_cubic_spline2_matrix = MCMCMatrixUnfolder([omega_cubic_spline], "User", [0.3])
    @test @returntrue model_cubic_spline3_matrix = MCMCMatrixUnfolder([omega_cubic_spline], "EmpiricalBayes", [1.], [1e-8], [10.], [0.3])
    @test @returntrue model_cubic_spline4_matrix = MCMCMatrixUnfolder([omega_cubic_spline], "EmpiricalBayes", nothing, [1e-8], [10.], [0.3])

    @test @returntrue model_1, line1, inits1, samples1, burnin1, thin1, chains1 = solve(model_cubic_spline1_matrix, discretize_kernel(basis_cubic_spline, my_kernel, y), f, sig)
    @test @returntrue model_2, line2, inits2, samples2, burnin2, thin2, chains2 = solve(model_cubic_spline2_matrix, discretize_kernel(basis_cubic_spline, getOpticsKernels("heaviside"), y), f, sig)
    @test @returntrue model_3, line3, inits3, samples3, burnin3, thin3, chains3 = solve(model_cubic_spline3_matrix, discretize_kernel(basis_cubic_spline, getOpticsKernels("rectangular"), y), f, sig)
    @test @returntrue model_4, line4, inits4, samples4, burnin4, thin4, chains4 = solve(model_cubic_spline4_matrix, discretize_kernel(basis_cubic_spline, getOpticsKernels("triangular"), y), f, sig)

    @test @returntrue sim1 = mcmc(model_1, line1, inits1, 100, burnin=burnin1, thin=thin1, chains=chains1)
    @test @returntrue res1 = get_values(sim1, chains1, length(basis_cubic_spline))
    @test @returntrue sim2 = mcmc(model_2, line2, inits2, 100, burnin=burnin2, thin=thin2, chains=chains2)
    @test @returntrue res2 = get_values(sim2, chains2, length(basis_cubic_spline))
    @test @returntrue sim3 = mcmc(model_3, line3, inits3, 100, burnin=burnin3, thin=thin3, chains=chains3)
    @test @returntrue res3 = get_values(sim3, chains3, length(basis_cubic_spline))
    @test @returntrue sim4 = mcmc(model_4, line4, inits4, 100, burnin=burnin4, thin=thin4, chains=chains4)
    @test @returntrue res4 = get_values(sim4, chains4, length(basis_cubic_spline))
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
