using Logging

mutable struct Config
    RTOL_QUADGK::Real# = 1e-8
    MAXEVALS_QUADGK::Real# = 1e5
    X_TOL_OPTIM::Real# = 1e-8
    ORDER_QUADGK::Int# = 500

    Config() = new(1e-8, 1e5, 1e-8, 500)
end

config = Config()
