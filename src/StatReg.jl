module StatReg

    using QuadGK, LinearAlgebra, Dierckx, Memoize, ApproxFun
    using Optim
    using Mamba
    using Logging
    using Polynomials
    using PiecewisePolynomials

    include("config.jl")
    include("kernels.jl")
    include("basis.jl")
    include("vector.jl")
    include("check.jl")
    include("gauss_error.jl")
    include("mcmc.jl")

    export BaseFunction, Basis, omega, FourierBasis, CubicSplineBasis, LegendreBasis, BernsteinBasis, discretize_kernel
    export GaussErrorMatrixUnfolder, solve, GaussErrorUnfolder
    export PhiVec, call, errors
    export config, Config
    export MCMCMatrixUnfolder, MCMCUnfolder, get_values
    export getOpticsKernels

end
