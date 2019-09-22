var documenterSearchIndex = {"docs":
[{"location":"#Statreg.jl-1","page":"Home","title":"Statreg.jl","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"This is documentation for Statreg.jl – a Julia package that allows to apply Turchin's method of statistical regularisation to solve the Fredholm equation of the first kind.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Let's consider equation","category":"page"},{"location":"#","page":"Home","title":"Home","text":"f(y) = intlimits_a^b K(x y) varphi(x) dx","category":"page"},{"location":"#","page":"Home","title":"Home","text":"The problem is, given kernel function K(x y) and observed function f(y), to find the function varphi(x). f(y) contains a random noise factor both from initial statistical uncertainty of varphi(x) and additional noise from measurement procedure. The equation is ill-posed: a small measurement error of f(y) leads to big instability of varphi(x). Solving such ill-posed problems requires operation called regularisation. It means we need to introduce additional information to make the problem well-posed one.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"The idea of statistical regularisation is related to the Bayesian statistics approach: unknown statistical value varphi(x) could be reconstructed using measured value f(y), kernel K(x y) and some prior information about varphi(x) behaviour: smoothness, constraints on boundary conditions, non-negativity, etc. Also, it is important to note that statistical regularisation allows to estimate errors of the obtained solution. More information about the theory of statistical regularisation you can find here, but the main concepts will be explained further in this documentation.","category":"page"},{"location":"#Description-of-statistical-regularisation-method-1","page":"Home","title":"Description of statistical regularisation method","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Firstly, it is necessary to make a parameterised discrete representation of the continuous functional space. We should introduce basis  psi_k _k=1^N, in which the required function will be calculated. Thus, the Fredholm equation will be сonverted to the matrix equation:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"f_m = K_mn varphi_n","category":"page"},{"location":"#","page":"Home","title":"Home","text":"where f_m = f(y_m),  varphi_n :  varphi(x) = sumlimits_k=1^N varphi_k psi_k(x),  K_mn = intlimits_a^b K(x y_m) psi_n(x) dx.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Let's introduce function overrightarrowS that will evaluate overrightarrowvarphi based on the function overrightarrowf and loss function L(overrightarrowwidehatvarphi overrightarrowS) = sumlimits_n=1^N mu_n (widehatvarphi_n - S_n)^2, where overrightarrowwidehatvarphi=overrightarrowwidehatS(overrightarrowf) – the best solution.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"For this loss function the best strategy is","category":"page"},{"location":"#","page":"Home","title":"Home","text":"overrightarrowwidehatSf=Eoverrightarrowvarphioverrightarrowf=int overrightarrowvarphi P(overrightarrowvarphioverrightarrowf) doverrightarrowvarphi","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Errors of the solution:","category":"page"},{"location":"#","page":"Home","title":"Home","text":" sigma_n^2  = int (varphi_n - widehatS_n)^2 P(overrightarrowvarphioverrightarrowf)doverrightarrowvarphi","category":"page"},{"location":"#","page":"Home","title":"Home","text":"P(overrightarrowvarphioverrightarrowf) = fracP(overrightarrowvarphi)P(overrightarrowfoverrightarrowvarphi)int doverrightarrowvarphiP(overrightarrowvarphi)P(overrightarrowfoverrightarrowvarphi)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Thus, P(overrightarrowvarphi) and P(overrightarrowfoverrightarrowvarphi) are required to find the solution. P(overrightarrowvarphi) can be chosen using prior information about overrightarrowvarphi). P(overrightarrowfoverrightarrowvarphi) depends on overrightarrowf distribution. Let's consider different distributions of overrightarrowvarphi and overrightarrowf.","category":"page"},{"location":"#Smoothness-as-a-prior-information-1","page":"Home","title":"Smoothness as a prior information","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"We expect varphi(x) to be relatively smooth and can choose this information as a prior. The matrix of the mean value of derivatives of order p can be used as a prior information about the solution.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Omega_mn = intlimits_a^b left( fracd^p psi_m(x)dx^p right) left( fracd^p psi_n(x)dx^p right) dx","category":"page"},{"location":"#","page":"Home","title":"Home","text":"We require a certain value of the smoothness functional to be achieved:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"int (overrightarrowvarphi Omega overrightarrowvarphi) P(overrightarrowvarphi)doverrightarrowvarphi=omega","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Thus, the overrightarrowvarphi probability distribution depends on the parameter:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"P_alpha(overrightarrowvarphi)=fracalpha^Rg(Omega)2 sqrttextdet(Omega)(2pi)^N2textexpleft( -frac12 (overrightarrowvarphi Omega overrightarrowvarphi) right)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"where alpha=frac1omega.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"The value of the parameter alpha is unknown and can be obtained in following ways:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"directly from some external data or manually selected\nas a maximum of a posterior information P(alphaoverrightarrowf)\nas the mean of all possible alpha, defining the prior probability density as P(overrightarrowvarphi)=int dalpha P(alpha) P(overrightarrowvarphialpha) (all alphas are equally probable).","category":"page"},{"location":"#","page":"Home","title":"Home","text":"StatReg.jl allows to apply all of these options.","category":"page"},{"location":"#Gaussian-random-process-1","page":"Home","title":"Gaussian random process","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Experimental data usually follows a normal distribution. At that rate the regularisation has an analytical solution. Let the measurement vector f have errors described by a multidimensional Gaussian distribution with a covariance matrix Sigma:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"P(overrightarrowfoverrightarrowvarphi)=frac1(2pi)^N2Sigma^12expleft( -frac12 (overrightarrowf - Koverrightarrowvarphi)^T Sigma^-1 (overrightarrowf - Koverrightarrowvarphi) right)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Using the most probable alpha, one can get the best solution:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"overrightarrowwidehatS = (K^T Sigma^-1 K + alpha^* Omega)^-1 K^T Sigma^-1 T overrightarrowf","category":"page"},{"location":"#","page":"Home","title":"Home","text":"cov(varphi_m varphi_n) = (K^T Sigma^-1 K + alpha^* Omega)^-1_mn","category":"page"},{"location":"#","page":"Home","title":"Home","text":"This package allows to apply statistical regularisation in different bases using such prior information as smoothness or zero boundary conditions, or another information provided by user in a matrix form. Omega can be set manually or calculated for every derivative  of degree p. alpha can be calculated as a maximum of a posterior information or can be set manually.","category":"page"},{"location":"#Non-Gaussian-random-process-1","page":"Home","title":"Non-Gaussian random process","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"If the f function errors do not follow Gaussian distribution, the strategy overrightarrowwidehatS can not be calculated analytically in general case.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"overrightarrowwidehatSf=Eoverrightarrowvarphioverrightarrowf=int overrightarrowvarphi P(overrightarrowvarphioverrightarrowf) doverrightarrowvarphi","category":"page"},{"location":"#","page":"Home","title":"Home","text":"The posterior probability P(overrightarrowvarphioverrightarrowf) should be obtained from MCMC sampling. It is applied in the StatReg.jl using Mamba.jl package.","category":"page"},{"location":"getting_started/#Getting-started-1","page":"Getting started","title":"Getting started","text":"","category":"section"},{"location":"getting_started/#Installation-1","page":"Getting started","title":"Installation","text":"","category":"section"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"To install StatReg.jl, start up Julia and type the following code-snipped into the REPL.","category":"page"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"import Pkg\nPkg.clone(\"https://github.com/mipt-npm/StatReg.jl.git\")","category":"page"},{"location":"getting_started/#Usage-1","page":"Getting started","title":"Usage","text":"","category":"section"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"Let's consider the simplest case of deconvolution. The function to be reconstructed varphi(x) is the sum of two Gaussian distributions.","category":"page"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"using PyCall\n\na = 0\nb = 6.\n\nfunction phi(x::Real)\n    mu1 = 2.\n    mu2 = 4.\n    n1 = 4.\n    n2 = 2.\n    sig1 = 0.4\n    sig2 = 0.5\n    norm(n, mu, sig, x) = n / sqrt(2 * pi*sig^2) * exp(-(x - mu)^2 / (2 * sig^2))\n    return norm(n1, mu1, sig1, x) + norm(n2, mu2, sig2, x)\nend\n\nx = collect(range(a, stop=b, length=300))\n\nmyplot = plot(x, phi.(x))","category":"page"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"(Image: real_phi)","category":"page"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"After integration we get data and errors. kernel - kernel function, y - measurement points, f - data points, sig - data errors.","category":"page"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"kernel(x::Real, y::Real) = getOpticsKernels(\"gaussian\")(x, y)\n\nconvolution = y -> quadgk(x -> kernel(x,y) * phi(x), a, b, rtol=10^-5, maxevals=10^7)[1]\ny = collect(range(a, stop=b, length=30))\nftrue = convolution.(y)\n\nsig = 0.05*ftrue + [0.01 for i = 1:Base.length(ftrue)]\n\nusing Compat, Random, Distributions\nnoise = []\nRandom.seed!(1234)\nfor sigma in sig\n    n = rand(Normal(0., sigma), 1)[1]\n    push!(noise, n)\nend\n\nf = ftrue + noise\nplot(y, f, title=\"Integrated function\",label=[\"f(y)\"])","category":"page"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"(Image: integrated)","category":"page"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"Let's proceed to the reconstruction.","category":"page"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"To reconstruct function you need to load data f(y) and data errors delta f(y) and define kernel K(x y). There are two possibilities: use vector & matrix form or continuous form. In the first case K(x y) is matrix n times m, f(y) and delta f(y) - n-dimensional vectors. In the second case K(x y) is a function, f(y) and delta f(y) can be either functions or vectors. If they are functions, knot vector y should be specified (points where the measurement is taken).","category":"page"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"We have already defined all needed data (y is a list of measurement points, f is a list of function values at these points, sig is a list of error in these points)\nBasis:","category":"page"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"We will use Cubic Spline Basis with knots in data points and zero boundary conditions on both sides.","category":"page"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"basis = CubicSplineBasis(y, \"dirichlet\")\nfor func in basis.basis_functions\n    plot(x, func.f.(x))\nend","category":"page"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"(Image: cubic_spline_basis)","category":"page"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"Model:","category":"page"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"To reconstruct the function, we use matrix of the second derivatives as a prior information. Then we choose a solution model. It requires basis and a set of matrices that contain prior information, in our case it is smoothness. The method we use is called \"EmpiricalBayes\", it means that alpha is chosen as a maximum of posterior probability P(alpha  f). Also, it is important to set higher and lower bounds of alpha and initial value for optimisation.","category":"page"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"Omega = omega(basis, 2)\nmodel = GaussErrorUnfolder(basis, [Omega], \"EmpiricalBayes\", nothing, [1e-8], [10.], [0.3])","category":"page"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"Reconstruction:","category":"page"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"To reconstruct the function we use solve() that returns dictionary containing coefficients of basis function in the sum varphi(x) = sum_k=1^N coeff_n psi_n(x), their errors errors_n (delta varphi =  sum_k=1^N errors_n psi_n(x)) and optimal parameter of smoothness alpha.","category":"page"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"result = solve(model, kernel, f, sig, y)","category":"page"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"Results","category":"page"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"Representation of results in a convenient way is possible with PhiVec:","category":"page"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"phivec = PhiVec(result, basis)\n\nphi_reconstructed = phivec.phi_function.(x)\nphi_reconstructed_errors = phivec.error_function.(x)\n\nplot(x, phi_reconstructed, ribbon=phi_reconstructed_errors, fillalpha=0.3, label=\"Reconstructed function with errors\")\nplot!(x, phi.(x), label=\"Real function\")","category":"page"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"(Image: reconstructed)","category":"page"},{"location":"getting_started/#","page":"Getting started","title":"Getting started","text":"Full notebook you can find in examples/getting_started.ipynb","category":"page"},{"location":"users_guide/#User's-Guide-1","page":"User's Guide","title":"User's Guide","text":"","category":"section"},{"location":"users_guide/#Kernel-1","page":"User's Guide","title":"Kernel","text":"","category":"section"},{"location":"users_guide/#","page":"User's Guide","title":"User's Guide","text":"Kernel can be specified as a matrix or as a function. It is possible to set arbitrary function of 2 variables or use one of predefined kernels.","category":"page"},{"location":"users_guide/#","page":"User's Guide","title":"User's Guide","text":"getOpticsKernels(name::String,)","category":"page"},{"location":"users_guide/#Main.StatReg.getOpticsKernels-Tuple{String}","page":"User's Guide","title":"Main.StatReg.getOpticsKernels","text":"getOpticsKernels(name::String, alpha::Real = 1.)\n\nArguments\n\nname - name of a kernel\nalpha - kernel function parameter\n\nReturns: kernel, function of 2 variables.\n\nAvailable kernels:\n\nrectangular:\n\nK(x y) =\nbegincases\n1 textif  fracx-yalpha  1\n\ntext  0 text otherwise\nendcases\n\ndiffraction:\n\nK(x y) = left(fracsin(fracpi (x-y)s_0)fracpi (x-y)s_0right)^2\n\ns_0 = fracalpha0886\n\ngaussian:\n\nK(x y) = frac2alphasqrtfracln2pie^4ln2left(fracx-yalpharight)^2\n\ntriangular:\n\nK(x y) =\nbegincases\nfrac1 - fracx-yalphaalpha textif  fracx-yalpha  1\n\ntext  0 text otherwise\nendcases\n\ndispersive:\n\nK(x y) = fracalpha2 pileft((x-y)^2 + left(fracalpha2right)^2right)\n\nexponential:\n\nK(x y) = fracln2alphae^2ln2fracx-yalpha\n\nheaviside:\n\nK(x y) =\nbegincases\n1 textif  x0\n\ntext   0 text otherwise\nendcases\n\n\n\n\n\n","category":"method"},{"location":"users_guide/#","page":"User's Guide","title":"User's Guide","text":"discretize_kernel(basis::Basis, kernel::Function, data_points::AbstractVector{<:Real})","category":"page"},{"location":"users_guide/#Main.StatReg.discretize_kernel-Tuple{Basis,Function,AbstractArray{#s1020,1} where #s1020<:Real}","page":"User's Guide","title":"Main.StatReg.discretize_kernel","text":"discretize_kernel(basis::Basis, kernel::Function, data_points::AbstractVector{<:Real})\n\nArguments\n\nbasis – basis\nkernel – kernel function\ndata_points – array of data points\n\nReturns: discretized kernel K::Array{Real, 2}, K_mn = intlimits_a^b K(x y_n) psi_m(x) dx - matrix of size ntimesm, where m - number of basis functions, n - number of data points.\n\n\n\n\n\n","category":"method"},{"location":"users_guide/#Basis-1","page":"User's Guide","title":"Basis","text":"","category":"section"},{"location":"users_guide/#","page":"User's Guide","title":"User's Guide","text":"Basis","category":"page"},{"location":"users_guide/#Main.StatReg.Basis","page":"User's Guide","title":"Main.StatReg.Basis","text":"Abstract type for all bases.\n\n\n\n\n\n","category":"type"},{"location":"users_guide/#","page":"User's Guide","title":"User's Guide","text":"BaseFunction","category":"page"},{"location":"users_guide/#Main.StatReg.BaseFunction","page":"User's Guide","title":"Main.StatReg.BaseFunction","text":"Type for function with its support.\n\nBaseFunction(f, support::Tuple{<:Real,<:Real})\nBaseFunction(f, a::Real, b::Real)\n\nFields\n\nf – function (type depends on the basis)\nsupport::Tuple{<:Real,<:Real} – support of the function\n\n\n\n\n\n","category":"type"},{"location":"users_guide/#","page":"User's Guide","title":"User's Guide","text":"omega(basis::Basis, order::Int)","category":"page"},{"location":"users_guide/#Main.StatReg.omega-Tuple{Basis,Int64}","page":"User's Guide","title":"Main.StatReg.omega","text":"omega(basis::Basis, ord::Int)\n\nArguments\n\nbasis - basis\nord - order of derivatives\n\nReturns: Omega::Array{Real, 2}, Omega_mn = intlimits_a^b fracd^ord psi_mdx^ord fracd^ord psi_ndx^ord - matrix of size ntimesn of the mean values of derivatives of order ord, where n - number of functions in basis.\n\n\n\n\n\n","category":"method"},{"location":"users_guide/#Fourier-basis-1","page":"User's Guide","title":"Fourier basis","text":"","category":"section"},{"location":"users_guide/#","page":"User's Guide","title":"User's Guide","text":"FourierBasis","category":"page"},{"location":"users_guide/#Main.StatReg.FourierBasis","page":"User's Guide","title":"Main.StatReg.FourierBasis","text":"Fourier basis with length 2n+1: {05, sin(fracpi (x - fraca+b2)b-a), cos(fracpi (x - fraca+b2)b-a), ..., sin(fracpi n (x - fraca+b2)b-a), cos(fracpi n (x - fraca+b2)b-a)}.\n\nFourierBasis(a::Real, b::Real, n::Int)\n\na, b – the beginning and the end of the segment n – number of basis functions\n\nFields\n\na::Real – beginning of the support\nb::Real – end of the support\nn::Int – number of basis functions\nbasis_functions::AbstractVector{BaseFunction} – array of basis functions\n\n\n\n\n\n","category":"type"},{"location":"users_guide/#Cubic-Spline-basis-1","page":"User's Guide","title":"Cubic Spline basis","text":"","category":"section"},{"location":"users_guide/#","page":"User's Guide","title":"User's Guide","text":"CubicSplineBasis","category":"page"},{"location":"users_guide/#Main.StatReg.CubicSplineBasis","page":"User's Guide","title":"Main.StatReg.CubicSplineBasis","text":"Cubic spline basis on given knots with length n, where n – length of knots array.\n\nCubicSplineBasis(\n    knots::AbstractVector{<:Real},\n    boundary_condition::Union{Tuple{Union{String, Nothing}, Union{String, Nothing}}, Nothing, String}=nothing\n    )\n\nCubicSplineBasis(\n    a::Real, b::Real, n::Int,\n    boundary_condition::Union{Tuple{Union{String, Nothing}, Union{String, Nothing}}, Nothing, String}=nothing\n    )\n\nknots – knots of spline boundary_condition – boundary conditions of basis functions. If tuple, the first element affects left bound, the second element affects right bound. If string, both sides are affected. Possible options: \"dirichlet\", nothing\n\nFields\n\na::Real – beginning of the support, matches the first element of the array knots\nb::Real – end of the support, matches the last element of the array knots\nknots::AbstractVector{<:Real} – array of points on which the spline is built\nbasis_functions::AbstractVector{BaseFunction} – array of basis functions\n\n\n\n\n\n","category":"type"},{"location":"users_guide/#Legendre-polynomials-basis-1","page":"User's Guide","title":"Legendre polynomials basis","text":"","category":"section"},{"location":"users_guide/#","page":"User's Guide","title":"User's Guide","text":"LegendreBasis","category":"page"},{"location":"users_guide/#Main.StatReg.LegendreBasis","page":"User's Guide","title":"Main.StatReg.LegendreBasis","text":"Legendre polynomials basis with length n.\n\nLegendreBasis(a::Real, b::Real, n::Int)\n\na, b – the beginning and the end of the support n – number of basis functions\n\nFields\n\na::Real – beginning of the support\nb::Real – end of the support\nbasis_functions::AbstractVector{BaseFunction} – array of basis functions\n\n\n\n\n\n","category":"type"},{"location":"users_guide/#Bernstein-polynomials-basis-1","page":"User's Guide","title":"Bernstein polynomials basis","text":"","category":"section"},{"location":"users_guide/#","page":"User's Guide","title":"User's Guide","text":"BernsteinBasis","category":"page"},{"location":"users_guide/#Main.StatReg.BernsteinBasis","page":"User's Guide","title":"Main.StatReg.BernsteinBasis","text":"Bernstein polynomials basis.\n\nBernsteinBasis(\n    a::Real, b::Real, n::Int,\n    boundary_condition::Union{Tuple{Union{String, Nothing}, Union{String, Nothing}}, Nothing, String}=nothing\n    )\n\na, b – the beginning and the end of the segment n – number of basis functions boundary_condition – boundary conditions of basis functions. If tuple, the first element affects left bound, the second element affects right bound. If string, both sides are affected. Possible options: \"dirichlet\", nothing.\n\nFields\n\na::Real – beginning of the support\nb::Real – end of the support\nbasis_functions::AbstractVector{BaseFunction} – array of basis functions\nboundary_condition::Tuple{Union{String, Nothing}, Union{String, Nothing}} – boundary conditions of basis functions. If tuple, the first element affects left bound, the second element affects right bound. If string, both sides are affected. Possible options: \"dirichlet\", nothing.\n\n\n\n\n\n","category":"type"},{"location":"users_guide/#Gaussian-noise-distribution-1","page":"User's Guide","title":"Gaussian noise distribution","text":"","category":"section"},{"location":"users_guide/#Model-1","page":"User's Guide","title":"Model","text":"","category":"section"},{"location":"users_guide/#","page":"User's Guide","title":"User's Guide","text":"GaussErrorMatrixUnfolder","category":"page"},{"location":"users_guide/#Main.StatReg.GaussErrorMatrixUnfolder","page":"User's Guide","title":"Main.StatReg.GaussErrorMatrixUnfolder","text":"Model for discrete data and kernel.\n\nGaussErrorMatrixUnfolder(\n    omegas::Array{Array{T, 2}, 1} where T<:Real,\n    method::String=\"EmpiricalBayes\";\n    alphas::Union{AbstractVector{<:Real}, Nothing}=nothing,\n    lower::Union{AbstractVector{<:Real}, Nothing}=nothing,\n    higher::Union{AbstractVector{<:Real}, Nothing}=nothing,\n    initial::Union{AbstractVector{<:Real}, Nothing}=nothing\n    )\n\nomegas – array of matrices that provide information about basis functions\n\nmethod – constant selection method, possible options: \"EmpiricalBayes\" and \"User\"\n\nalphas – array of constants, in case method=\"User\" should be provided by user\n\nlower – lowerer limits for alphas\n\nhigher – higherer limits for alphas\n\ninitial – unitial values for alphas\n\nFields\n\nomegas::Array{Array{T, 2}, 1} where T<:Real\nn::Int – size of square omega matrix\nmethod::String\nalphas::Union{AbstractVector{<:Real}, Nothing}\nlower::Union{AbstractVector{<:Real}, Nothing}\nhigher::Union{AbstractVector{<:Real}, Nothing}\ninitial::Union{AbstractVector{<:Real}, Nothing}\n\n\n\n\n\n","category":"type"},{"location":"users_guide/#","page":"User's Guide","title":"User's Guide","text":"GaussErrorUnfolder","category":"page"},{"location":"users_guide/#Main.StatReg.GaussErrorUnfolder","page":"User's Guide","title":"Main.StatReg.GaussErrorUnfolder","text":"Model for continuous kernel. Data can be either discrete or continuous.\n\nGaussErrorUnfolder(\n    basis::Basis,\n    omegas::Array{Array{T, 2}, 1} where T<:Real,\n    method::String=\"EmpiricalBayes\";\n    alphas::Union{AbstractVector{<:Real}, Nothing} =nothing,\n    lower::Union{AbstractVector{<:Real}, Nothing} =nothing,\n    higher::Union{AbstractVector{<:Real}, Nothing} =nothing,\n    initial::Union{AbstractVector{<:Real}, Nothing} =nothing\n    )\n\nbasis – basis for reconstruction\n\nomegas – array of matrices that provide information about basis functions\n\nmethod – constant selection method, possible options: \"EmpiricalBayes\" and \"User\"\n\nalphas – array of constants, in case method=\"User\" should be provided by user\n\nlower – lowerer limits for alphas\n\nhigher – higherer limits for alphas\n\ninitial – unitial values for alphas\n\nFields\n\nbasis::Basis\nsolver::GaussErrorMatrixUnfolder\n\n\n\n\n\n","category":"type"},{"location":"users_guide/#Reconstruction-1","page":"User's Guide","title":"Reconstruction","text":"","category":"section"},{"location":"users_guide/#","page":"User's Guide","title":"User's Guide","text":"solve(\n    unfolder::GaussErrorMatrixUnfolder,\n    kernel::AbstractMatrix{<:Real},\n    data::AbstractVector{<:Real},\n    data_errors::AbstractVecOrMat{<:Real}\n    )","category":"page"},{"location":"users_guide/#Main.StatReg.solve-Tuple{GaussErrorMatrixUnfolder,AbstractArray{#s1020,2} where #s1020<:Real,AbstractArray{#s1019,1} where #s1019<:Real,Union{AbstractArray{#s1018,1}, AbstractArray{#s1018,2}} where #s1018<:Real}","page":"User's Guide","title":"Main.StatReg.solve","text":"solve(\n    unfolder::GaussErrorMatrixUnfolder,\n    kernel::AbstractMatrix{<:Real},\n    data::AbstractVector{<:Real},\n    data_errors::AbstractVecOrMat{<:Real}\n    )\n\nArguments\n\nunfolder – model\nkernel – discrete kernel\ndata – function values\ndata_errors – function errors\n\nReturns: Dict{String, AbstractVector{Real}} with coefficients (\"coeff\"), errors (\"errors\") and optimal constants (\"alphas\").\n\n\n\n\n\n","category":"method"},{"location":"users_guide/#","page":"User's Guide","title":"User's Guide","text":"solve(\n    unfolder::GaussErrorUnfolder,\n    kernel::Union{Function, AbstractMatrix{<:Real}},\n    data::Union{Function, AbstractVector{<:Real}},\n    data_errors::Union{Function, AbstractVector{<:Real}},\n    y::Union{AbstractVector{<:Real}, Nothing}=nothing,\n    )","category":"page"},{"location":"users_guide/#Main.StatReg.solve","page":"User's Guide","title":"Main.StatReg.solve","text":"solve(\n    unfolder::GaussErrorUnfolder,\n    kernel::Union{Function, AbstractMatrix{<:Real}},\n    data::Union{Function, AbstractVector{<:Real}},\n    data_errors::Union{Function, AbstractVector{<:Real}},\n    y::Union{AbstractVector{<:Real}, Nothing}=nothing,\n    )\n\nArguments\n\nunfolder – model\nkernel – discrete or continuous kernel\ndata – function values\ndata_errors – function errors\ny – points to calculate function values and its errors (when data is given as a function)\n\nReturns: `Dict{String, AbstractVector{Real} with coefficients (\"coeff\"), errors (\"errors\") and optimal constants (\"alphas\").\n\n\n\n\n\n","category":"function"},{"location":"users_guide/#Non-Gaussian-noise-distribution-1","page":"User's Guide","title":"Non-Gaussian noise distribution","text":"","category":"section"},{"location":"users_guide/#Model-2","page":"User's Guide","title":"Model","text":"","category":"section"},{"location":"users_guide/#","page":"User's Guide","title":"User's Guide","text":"MCMCMatrixUnfolder","category":"page"},{"location":"users_guide/#Main.StatReg.MCMCMatrixUnfolder","page":"User's Guide","title":"Main.StatReg.MCMCMatrixUnfolder","text":"MCMC model for discrete data and kernel.\n\nMCMCMatrixUnfolder(\n    omegas::Array{Array{T, 2}, 1} where T<:Real,\n    method::String=\"EmpiricalBayes\";\n    alphas::Union{AbstractVector{<:Real}, Nothing}=nothing,\n    lower::Union{AbstractVector{<:Real}, Nothing}=nothing,\n    higher::Union{AbstractVector{<:Real}, Nothing}=nothing,\n    initial::Union{AbstractVector{<:Real}, Nothing}=nothing\n    )\n\nomegas – array of matrices that provide information about basis functions\n\nmethod – constant selection method, possible options: \"EmpiricalBayes\" and \"User\"\n\nalphas – array of constants, in case method=\"User\" should be provided by user\n\nlower – lowerer limits for alphas\n\nhigher – higherer limits for alphas\n\ninitial – unitial values for alphas\n\nFields\n\nomegas::Array{Array{T, 2}, 1} where T<:Real\nn::Int – size of square omega matrix\nmethod::String\nalphas::Union{AbstractVector{<:Real}, Nothing}\nlower::Union{AbstractVector{<:Real}, Nothing}\nhigher::Union{AbstractVector{<:Real}, Nothing}\ninitial::Union{AbstractVector{<:Real}, Nothing}\n\n\n\n\n\n","category":"type"},{"location":"users_guide/#","page":"User's Guide","title":"User's Guide","text":"MCMCUnfolder","category":"page"},{"location":"users_guide/#Main.StatReg.MCMCUnfolder","page":"User's Guide","title":"Main.StatReg.MCMCUnfolder","text":"MCMC model for continuous kernel. Data can be either discrete or continuous.\n\nMCMCUnfolder(\n    basis::Basis,\n    omegas::Array{Array{T, 2}, 1} where T<:Real,\n    method::String=\"EmpiricalBayes\";\n    alphas::Union{AbstractVector{<:Real}, Nothing}=nothing,\n    lower::Union{AbstractVector{<:Real}, Nothing}=nothing,\n    higher::Union{AbstractVector{<:Real}, Nothing}=nothing,\n    initial::Union{AbstractVector{<:Real}, Nothing}=nothing,\n    )\n\nbasis – basis for reconstruction\n\nomegas – array of matrices that provide information about basis functions\n\nmethod – constant selection method, possible options: \"EmpiricalBayes\" and \"User\"\n\nalphas – array of constants, in case method=\"User\" should be provided by user\n\nFields\n\nbasis::Basis\nsolver::MCMCMatrixUnfolder\n\n\n\n\n\n","category":"type"},{"location":"users_guide/#Reconstruction-2","page":"User's Guide","title":"Reconstruction","text":"","category":"section"},{"location":"users_guide/#","page":"User's Guide","title":"User's Guide","text":"solve(\n    mcmcunfolder::MCMCUnfolder,\n    kernel::Union{Function, AbstractMatrix{<:Real}},\n    data::Union{Function, AbstractVector{<:Real}},\n    data_errors::Union{Function, AbstractVector{<:Real}},\n    y::Union{AbstractVector{<:Real}, Nothing}=nothing;\n    model::Union{Model, String} = \"Gaussian\",\n    samples::Int = 10 * 1000,\n    burnin::Int = 0,\n    thin::Int = 1,\n    chains::Int = 1,\n    verbose::Bool = false\n    )","category":"page"},{"location":"users_guide/#Main.StatReg.solve","page":"User's Guide","title":"Main.StatReg.solve","text":"solve(\n    mcmcunfolder::MCMCUnfolder,\n    kernel::Union{Function, AbstractMatrix{<:Real}},\n    data::Union{Function, AbstractVector{<:Real}},\n    data_errors::Union{Function, AbstractVector{<:Real}},\n    y::Union{AbstractVector{<:Real}, Nothing}=nothing;\n    model::Union{Model, String} = \"Gaussian\",\n    samples::Int = 10 * 1000,\n    burnin::Int = 0,\n    thin::Int = 1,\n    chains::Int = 1,\n    verbose::Bool = false\n    )\n\nArguments\n\nunfolder – model\nkernel – discrete or continuous kernel\ndata – function values\ndata_errors – function errors\ny – points to calculate function values and its errors (when data is given as a function)\nmodel – errors model, \"Gaussian\" or predefined Mamba.jl model\nburnin– numer of initial draws to discard as a burn-in sequence to allow for convergence\nthin – step-size between draws to output\nchains– number of simulation runs to perform\nverbose – whether to print sampler progress at the console\n\nReturns: parameters for mcmc() function.\n\n\n\n\n\n","category":"function"},{"location":"users_guide/#","page":"User's Guide","title":"User's Guide","text":"solve(\n    unfolder::MCMCMatrixUnfolder,\n    kernel::AbstractMatrix{<:Real},\n    data::AbstractVector{<:Real},\n    data_errors::AbstractVecOrMat{<:Real};\n    model::Union{Model, String} = \"Gaussian\",\n    samples::Int = 10 * 1000,\n    burnin::Int = 0,\n    thin::Int = 1,\n    chains::Int = 1,\n    verbose::Bool = false\n    )","category":"page"},{"location":"users_guide/#Main.StatReg.solve-Tuple{MCMCMatrixUnfolder,AbstractArray{#s1020,2} where #s1020<:Real,AbstractArray{#s1019,1} where #s1019<:Real,Union{AbstractArray{#s1018,1}, AbstractArray{#s1018,2}} where #s1018<:Real}","page":"User's Guide","title":"Main.StatReg.solve","text":"MCMC solver for discrete data and kernel.\n\nsolve(\n    unfolder::MCMCMatrixUnfolder,\n    kernel::AbstractMatrix{<:Real},\n    data::AbstractVector{<:Real},\n    data_errors::AbstractVecOrMat{<:Real};\n    model::Union{Model, String} = \"Gaussian\",\n    samples::Int = 10 * 1000,\n    burnin::Int = 0,\n    thin::Int = 1,\n    chains::Int = 1,\n    verbose::Bool = false\n    )\n\nArguments\n\nunfolder – model\nkernel – discrete kernel\ndata – function valuess\ndata_errors – function errors\nmodel – errors model, \"Gaussian\" or predefined Mamba.jl model\nburnin– numer of initial draws to discard as a burn-in sequence to allow for convergence\nthin – step-size between draws to output\nchains– number of simulation runs to perform\nverbose – whether to print sampler progress at the console\n\nReturns: parameters for mcmc() function.\n\n\n\n\n\n","category":"method"},{"location":"users_guide/#Result-1","page":"User's Guide","title":"Result","text":"","category":"section"},{"location":"users_guide/#","page":"User's Guide","title":"User's Guide","text":"get_values","category":"page"},{"location":"users_guide/#Main.StatReg.get_values","page":"User's Guide","title":"Main.StatReg.get_values","text":"Allowers to get coefficients and errors from generated data set.\n\nget_values(sim::ModelChains)\n\nArguments\n\nsim – data generated by mcmc()\n\nReturns: Dict{String, AbstractVector{Real}} with coefficients (\"coeff\") and errors (\"errors\").\n\n\n\n\n\n","category":"function"},{"location":"users_guide/#","page":"User's Guide","title":"User's Guide","text":"PhiVec","category":"page"},{"location":"users_guide/#Main.StatReg.PhiVec","page":"User's Guide","title":"Main.StatReg.PhiVec","text":"Constructs solution function by coefficients, basis and errors.\n\nPhiVec(coeff::Array{<:Real}, basis::Basis, errors::Array{<:Real})\nPhiVec(coeff::Array{<:Real}, basis::Basis)\nPhiVec(result::Dict{String, Array{<:Real}}, basis::Basis)\n\nFields\n\ncoeff::Array{<:Real} – coefficients of decomposition of a function in basis\nbasis::Basis – basis\nerrors::Union{Array{<:Real}, Nothing} – coefficients of decomposition of a function errors in basis\nphi_function(x::Real)::Function – returns constructed function's value at given point\nerror_function(x::Real)::Union{Function, Nothing} – returns constructed function's error at given point, if errors are specified, otherwise is nothing\n\n\n\n\n\n","category":"type"},{"location":"examples/#Examples-1","page":"Examples","title":"Examples","text":"","category":"section"},{"location":"examples/#","page":"Examples","title":"Examples","text":"You can find more examples here.","category":"page"}]
}
