include("gauss_error.jl")
include("kernels.jl")
include("basis.jl")
include("vector.jl")
include("config.jl")

using Test

@testset "Config" begin
    @test config.MAXEVALS_QUADGK >= 300 && config.MAXEVALS_QUADGK > 0
    @test config.X_TOL_OPTIM <= 1e-3 && config.X_TOL_OPTIM > 0
    @test config.RTOL_QUADGK <= 1e-3 && config.RTOL_QUADGK > 0
    end
