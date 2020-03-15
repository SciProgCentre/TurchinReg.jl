mutable struct Config
    RTOL_QUADGK::Real
    MAXEVALS_QUADGK::Real
    X_TOL_OPTIM::Real
    ORDER_QUADGK::Int
    ALPHAS_LOWER::Real
    ALPHAS_HIGHER::Real
    ALPHAS_INITIAL::Real
    PHI_LOWER::Real
    PHI_HIGHER::Real
    PHI_INITIAL::Real
    PHI_MEASURE::Real
    ALPHAS_NOT_CONVERGED::Real

end

config = Config(1e-5, 1e5, 1e-5, 500, 1e-4, 10., 0.1, -1e6, 1e6, 0., 1., 0.05)
