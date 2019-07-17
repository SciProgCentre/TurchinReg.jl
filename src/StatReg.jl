#=
main:
- Julia version: 1.1.0
- Author: ta_nyan
- Date: 2019-07-03
=#
module StatReg

    include("basis.jl")
    include("gauss_error.jl")
    include("vector.jl")
    include("config.jl")

    export BaseFunction, Basis, omega, FourierBasis, CubicSplineBasis, LegendreBasis, BernsteinBasis
    export GaussErrorMatrixUnfolder, solve, GaussErrorUnfolder
    export PhiVec, call, errors
    export RTOL_QUADGK, MAXEVALS_QUADGK, X_TOL_OPTIM, ORDER_QUADGK
end
