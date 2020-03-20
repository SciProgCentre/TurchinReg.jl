struct TurchinProblem
    algo::DHMCSampling
    sig::Matrix
    phi_bounds::PhiBounds
end

struct TurchinProblemMarginalize
    algo::DHMCSampling
    omegas::Array{Array{T, 2}, 1} where T<:Real
    phi_bounds::PhiBounds
end

function (problem::TurchinProblem)(θ)
    @unpack phi = θ
    @unpack algo, sig, phi_bounds = problem
    mu = phi_bounds.initial
    prior_log = -1/2 * transpose(phi - mu) * sig * (phi - mu)
    likelihood_log = algo.log_data_distribution(phi)
    bounds_correction = log_bounds_correction(phi, phi_bounds)
    return prior_log + likelihood_log + bounds_correction
end

function (problem::TurchinProblemMarginalize)(θ)
    @unpack phi, alpha = θ
    @unpack algo, omegas, phi_bounds = problem
    # if any(x -> x<=0, alpha)
    #     return -1e10
    # end
    mu = zeros(length(phi_bounds.lower))
    sig = transpose(alpha) * omegas
    prior_log = -1/2 * transpose(phi - mu) * sig * (phi - mu)
    likelihood_log = algo.log_data_distribution(phi)
    bounds_correction = log_bounds_correction(phi, phi_bounds)
    return prior_log + likelihood_log + bounds_correction
end

function _solve(algo::DHMCSampling, alphas::AlphasType, omegas::Union{Array{Array{T, 2}, 1}, Nothing} where T<:Real, B, b, phi_bounds::PhiBounds, basis::Basis)
    if typeof(alphas) == Marginalize
        t = TurchinProblemMarginalize(algo, omegas, phi_bounds)
        transT = as((phi = as(Array, asℝ, length(basis)), alpha = as(Array, asℝ, length(omegas))))
    else
        sig = transpose(alphas.alphas) * omegas
        sig_inv = sym_inv(sig)
        t = TurchinProblem(algo, sig, phi_bounds)
        transT = as((phi = as(Array, asℝ, length(basis)),))
    end
    T = TransformedLogDensity(transT, t)
    ∇T = ADgradient(:ForwardDiff, T)
    @info "Starting sampling"
    results = mcmc_with_warmup(Random.GLOBAL_RNG, ∇T, algo.nsamples; reporter = NoProgressReport())
    posterior = TransformVariables.transform.(transT, results.chain)
    coeff = [mean([posterior[j].phi[i] for j in range(1, stop=algo.nsamples)]) for i in range(1, stop=length(basis))]
    errors = [cov([posterior[j].phi[i] for j in range(1, stop=algo.nsamples)]) for i in range(1, stop=length(basis))]
    @info "Solved with DHMC.jl"
    return PhiVec(coeff, basis, errors, alphas.alphas)
end
