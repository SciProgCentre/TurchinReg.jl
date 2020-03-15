module StatReg

    using QuadGK, LinearAlgebra, Dierckx, Memoize, ApproxFun
    using Optim
    using Logging
    using Polynomials
    using PiecewisePolynomials
    using BAT, ValueShapes
    using AdvancedHMC, Distributions, ForwardDiff
    using TransformVariables, LogDensityProblems, DynamicHMC, Parameters, Random

    include("./bases/bases.jl")
    include("./utils/utils.jl")

    include("./solution_utils/solution_utils.jl")
    include("./solvers/solvers.jl")

    export BaseFunction, Basis, omega, FourierBasis, CubicSplineBasis, LegendreBasis, BernsteinBasis, discretize_kernel
    export AlphasType, ArgmaxBAT, ArgmaxOptim, Marginalize, User, PhiBounds, AlgoType, Analytically, BATSampling, AHMCSampling, DHMCSampling, make_bounds, solve
    export PhiVec
    export config, Config
    export getOpticsKernels

end
