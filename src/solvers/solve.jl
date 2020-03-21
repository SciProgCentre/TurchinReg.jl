"""
Solve the problem

```julia
solve(
    basis::Basis,
    data::Union{AbstractVector{<:Real}, Function},
    data_errors::Union{AbstractVecOrMat{<:Real}, Function},
    kernel::Union{Function, AbstractMatrix{<:Real}},
    measurement_points::Union{AbstractVector{<:Real}, Nothing}=nothing,
    algo::AlgoType=Analytically(),
    alphas::AlphasType=ArgmaxBAT(),
    omegas::Union{Array{Array{T, 2}, 1}, Nothing} where T<:Real = nothing,
    phi_bounds::PhiBounds=PhiBounds(),
    )
```
"""
function solve(
    basis::Basis,
    data::Union{AbstractVector{<:Real}, Function},
    data_errors::Union{AbstractVecOrMat{<:Real}, Function},
    kernel::Union{Function, AbstractMatrix{<:Real}},
    measurement_points::Union{AbstractVector{<:Real}, Nothing}=nothing,
    algo::AlgoType=Analytically(),
    alphas::AlphasType=ArgmaxBAT(),
    omegas::Union{Array{Array{T, 2}, 1}, Nothing} where T<:Real = nothing,
    phi_bounds::Union{PhiBounds, Nothing}=nothing,
    )
    @info "Checking data"
    data, data_errors, kernel = check_data(data, data_errors, kernel, basis, measurement_points)
    alphas, omegas = check_alphas_omegas(alphas, omegas, basis)
    data_errors_inv = sym_inv(data_errors)
    B = make_sym(transpose(kernel) * data_errors_inv * kernel)
    b = transpose(kernel) * transpose(data_errors_inv) * data
    @info "Finding optimal alpha"
    alphas = find_optimal_alpha(alphas, omegas, B, b)
    @info "Optimal alpha found"
    @assert !((typeof(algo) == Analytically) && ((phi_bounds != nothing) && (phi_bounds != PhiBounds()))) "Can not apply bounds for analytical solution"
    @assert !((typeof(alphas) == Marginalize) && (typeof(algo) == Analytically)) "Marginalizing is not posiible for analytical solution"
    phi_bounds = check_phi_bounds(phi_bounds, basis)
    if typeof(algo) != Analytically
        if algo.log_data_distribution == nothing
            log_likelihood = let kernel=kernel,  data=data, data_errors_inv=data_errors_inv
                phi -> begin
                    mu = kernel * phi
                    likelihood_res = -1/2 * transpose(data - mu) * data_errors_inv * (data - mu)
                    return likelihood_res
                end
            end
            algo.log_data_distribution = log_likelihood
        end
        @assert isfinite(algo.log_data_distribution([1. for _ in 1:length(basis)])) "Incorrect log_data_distribution"
    end
    @info "Starting solution"
    res = _solve(algo, alphas, omegas, B, b, phi_bounds, basis)
    return res
end
