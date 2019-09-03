module StatReg

    include("basis.jl")
    include("gauss_error.jl")
    include("vector.jl")
    include("config.jl")
    include("mcmc.jl")

    export BaseFunction, Basis, omega, FourierBasis, CubicSplineBasis, LegendreBasis, BernsteinBasis
    export GaussErrorMatrixUnfolder, solve, GaussErrorUnfolder
    export PhiVec, call, errors
    export config, Config
    export MCMCMatrixUnfolder, MCMCUnfolder, get_values
end
