using Logging


mutable struct Config
    RTOL_QUADGK::Float64# = 1e-8
    MAXEVALS_QUADGK::Float64# = 1e5
    X_TOL_OPTIM::Float64# = 1e-8
    ORDER_QUADGK::Int64# = 500

    Config() = new(1e-8, 1e5, 1e-8, 500)
end

config = Config()

# RTOL_QUADGK = 1e-8
# MAXEVALS_QUADGK = 1e5
# X_TOL_OPTIM = 1e-8
# ORDER_QUADGK = 500

# global_logger()
