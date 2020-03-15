function _solve(algo::AHMCSampling, alphas::AlphasType, omegas::Union{Array{Array{T, 2}, 1}, Nothing} where T<:Real, B, b, phi_bounds::PhiBounds, basis::Basis)
    sig = transpose(alphas.alphas) * omegas
    posterior = let algo=algo, sig=sig, phi_bounds=phi_bounds
        phi -> begin
            prior_log = -1/2 * transpose(phi) * sig * (phi)
            bounds_correction = log_bounds_correction(phi, phi_bounds)
            likelihood_log = algo.log_data_distribution(phi)
            return prior_log + likelihood_log + bounds_correction
        end
    end
    metric = DiagEuclideanMetric(length(phi_bounds.initial))
    hamiltonian = Hamiltonian(metric, posterior, ForwardDiff)
    initial_ϵ = find_good_eps(hamiltonian, phi_bounds.initial)
    integrator = Leapfrog(initial_ϵ)
    proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(Preconditioner(metric), NesterovDualAveraging(0.8, integrator))
    n_adapts = 100
    @info "Starting sampling"
    res = AdvancedHMC.sample(hamiltonian, proposal, phi_bounds.initial, algo.nsamples, adaptor, n_adapts; progress=false)
    samples = res[1]
    coeff = [mean(getindex.(samples, x)) for x in eachindex(samples[1])]
    errors = [cov(getindex.(samples, x)) for x in eachindex(samples[1])]
    @info "Solved with AHMC.jl"
    return PhiVec(coeff, basis, errors, alphas.alphas)
end
