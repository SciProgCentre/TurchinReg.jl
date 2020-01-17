module StatReg

    using QuadGK, LinearAlgebra, Dierckx, Memoize, ApproxFun
    using Optim
    # using Mamba
    using Logging
    using Polynomials
    using PiecewisePolynomials
    using BAT, ValueShapes
    using AdvancedHMC, Distributions, ForwardDiff

    include("./utils/utils.jl")
    include("./bases/bases.jl")
    include("./solution_utils/solution_utils.jl")
    include("./solvers/solvers.jl")

    export BaseFunction, Basis, omega, FourierBasis, CubicSplineBasis, LegendreBasis, BernsteinBasis, discretize_kernel
    export GaussErrorMatrixUnfolder, solve, GaussErrorUnfolder, simple_solver
    export PhiVec, call, errors
    export config, Config
    export MCMCMatrixUnfolder, MCMCUnfolder, get_values
    export getOpticsKernels

end
