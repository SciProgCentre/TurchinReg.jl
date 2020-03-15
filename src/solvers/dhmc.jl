struct TurchinProblem
    algo::DHMCSampling
    sig::Matrix
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

function _solve(algo::DHMCSampling, alphas::AlphasType, omegas::Union{Array{Array{T, 2}, 1}, Nothing} where T<:Real, B, b, phi_bounds::PhiBounds, basis::Basis)
    sig = transpose(alphas.alphas) * omegas
    sig_inv = sym_inv(sig)

    t = TurchinProblem(algo, sig, phi_bounds)

    transT = as((phi = as(Array, asℝ, length(basis)),))
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
