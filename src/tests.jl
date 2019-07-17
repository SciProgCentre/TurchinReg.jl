include("gauss_error.jl")
include("kernels.jl")
include("basis.jl")
include("vector.jl")
include("config.jl")

using Test

@testset "Config" begin
    @test MAXEVALS_QUADGK >= 300 && MAXEVALS_QUADGK > 0
    @test X_TOL_OPTIM <= 1e-3 && X_TOL_OPTIM > 0
    @test RTOL_QUADGK <= 1e-3 && RTOL_QUADGK > 0
    end
