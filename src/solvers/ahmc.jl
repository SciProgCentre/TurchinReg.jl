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
    @info "Starting sampling"
    res = AdvancedHMC.sample(
        hamiltonian, proposal, phi_bounds.initial,
        algo.nsamples, adaptor, algo.n_adapts; progress=false
        )
    samples = res[1]
    coeff = [mean(getindex.(samples, x)) for x in eachindex(samples[1])]
    errors = [cov(getindex.(samples, x)) for x in eachindex(samples[1])]
    @info "Solved with AHMC.jl"
    return PhiVec(coeff, basis, errors, alphas.alphas)
end

function _solve(algo::AHMCSampling, alphas::Marginalize, omegas::Union{Array{Array{T, 2}, 1}, Nothing} where T<:Real, B, b, phi_bounds::PhiBounds, basis::Basis)
    alphas_length = length(omegas)
    posterior = let algo=algo, phi_bounds=phi_bounds, alphas_length = alphas_length, omegas=omegas
        phialphas -> begin
            alpha = phialphas[1:alphas_length]
            phi = phialphas[alphas_length+1:end]
            if any(x -> x<=0, alpha)
                return -1e10
            end
            sig = transpose(alpha) * omegas
            prior_log = -1/2 * transpose(phi) * sig * (phi)
            bounds_correction = log_bounds_correction(phi, phi_bounds)
            likelihood_log = algo.log_data_distribution(phi)
            return prior_log + likelihood_log + bounds_correction
        end
    end
    phialphas0 = push!(ones(alphas_length), phi_bounds.initial...)
    metric = DiagEuclideanMetric(length(phialphas0))
    hamiltonian = Hamiltonian(metric, posterior, ForwardDiff)
    initial_ϵ = find_good_eps(hamiltonian, phialphas0)
    integrator = Leapfrog(initial_ϵ)
    proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(Preconditioner(metric), NesterovDualAveraging(0.8, integrator))
    @info "Starting sampling"
    res = AdvancedHMC.sample(
        hamiltonian, proposal, phialphas0,
        algo.nsamples, adaptor, algo.n_adapts; progress=false
        )
    samples = res[1]
    return samples
    coeff = [mean(getindex.(samples, x)) for x in eachindex(samples[1])][2:end]
    errors = [cov(getindex.(samples, x)) for x in eachindex(samples[1])][2:end]
    @info "Solved with AHMC.jl"
    return PhiVec(coeff, basis, errors, alphas.alphas)
end
