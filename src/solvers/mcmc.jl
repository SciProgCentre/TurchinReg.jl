"""
MCMC model for discrete data and kernel.



```julia
MCMCMatrixUnfolder(
    omegas::Array{Array{T, 2}, 1} where T<:Real,
    method::String="EmpiricalBayes";
    alphas::Union{AbstractVector{<:Real}, Nothing}=nothing,
    lower::Union{AbstractVector{<:Real}, Nothing}=nothing,
    higher::Union{AbstractVector{<:Real}, Nothing}=nothing,
    initial::Union{AbstractVector{<:Real}, Nothing}=nothing
    )
```
`omegas` -- array of matrices that provide information about basis functions

`method` -- constant selection method, possible options: "EmpiricalBayes" and "User"

`alphas` -- array of constants, in case method="User" should be provided by user

`lower` -- lowerer limits for alphas

`higher` -- higherer limits for alphas

`initial` -- unitial values for alphas

**Fields**

* `omegas::Array{Array{T, 2}, 1} where T<:Real`
* `n::Int` -- size of square omega matrix
* `method::String`
* `alphas::Union{AbstractVector{<:Real}, Nothing}`
* `lower::Union{AbstractVector{<:Real}, Nothing}`
* `higher::Union{AbstractVector{<:Real}, Nothing}`
* `initial::Union{AbstractVector{<:Real}, Nothing}`
"""
mutable struct MCMCMatrixUnfolder
    omegas::Array{Array{T, 2}, 1} where T<:Real
    n::Int
    method::String
    alphas::Union{AbstractVector{<:Real}, Nothing}
    lower::Union{AbstractVector{<:Real}, Nothing}
    higher::Union{AbstractVector{<:Real}, Nothing}
    initial::Union{AbstractVector{<:Real}, Nothing}

    function MCMCMatrixUnfolder(
        omegas::Array{Array{T, 2}, 1} where T<:Real,
        method::String="EmpiricalBayes";
        alphas::Union{AbstractVector{<:Real}, Nothing}=nothing,
        lower::Union{AbstractVector{<:Real}, Nothing}=nothing,
        higher::Union{AbstractVector{<:Real}, Nothing}=nothing,
        initial::Union{AbstractVector{<:Real}, Nothing}=nothing
        )
        n = check_args(omegas, method, alphas, lower, higher, initial)
        @info "MCMCMatrixUnfolder is created."
        return new(omegas, n, method, alphas, lower, higher, initial)
    end
end


"""
MCMC solver for discrete data and kernel.

```julia
solve(
    unfolder::MCMCMatrixUnfolder,
    kernel::AbstractMatrix{<:Real},
    data::AbstractVector{<:Real},
    data_errors::AbstractVecOrMat{<:Real};
    model::Union{Function, String} = "Gaussian",
    nsamples::Int = 10 * 1000,
    nchains::Int = 1,
    algorithm::BAT.AbstractSamplingAlgorithm = MetropolisHastings()
    )
```

**Arguments**
* `unfolder` -- model
* `kernel` -- discrete kernel
* `data` -- function valuess
* `data_errors` -- function errors
* `model` -- errors model, "Gaussian" or predefined likelihood function
* `nsamples` -- number of nsamples
* `nchains`-- number of simulation runs to perform

**Returns:** dictionary with coefficients of basis functions and errors.
"""
function solve(
    unfolder::MCMCMatrixUnfolder,
    kernel::AbstractMatrix{<:Real},
    data::AbstractVector{<:Real},
    data_errors::AbstractVecOrMat{<:Real};
    model::Union{Function, String} = "Gaussian",
    nsamples::Int = 10 * 1000,
    nchains::Int = 1,
    sampler = "BAT",
    prior_phi::AbstractVector{<:Real},
    lower_phi::AbstractVector{<:Real},
    higher_phi::AbstractVector{<:Real},
    measure_phi::AbstractVector{<:Real},
    )

    @info "Starting solve..."
    data_errors = check_args(unfolder, kernel, data, data_errors)
    data_errors_inv = sym_inv(data_errors)
    B = make_sym(transpose(kernel) * data_errors_inv * kernel)
    b = transpose(kernel) * transpose(data_errors_inv) * data
    initial = unfolder.initial
    if unfolder.method == "EmpiricalBayes"
        unfolder.alphas = find_optimal_alpha(
            unfolder.omegas, B, b,
            unfolder.initial, unfolder.lower, unfolder.higher
            )
    elseif unfolder.method != "User"
        @error "Unknown method: " + unfolder.method
        Base.eror("Unknown method: " + unfolder.method)
    end
    @info "Ending solve..."
    if sampler == "BAT"
        return solve_BAT(
            unfolder, kernel, data, data_errors,
            model, nsamples, nchains, prior_phi, lower_phi, higher_phi, measure_phi
            )
    elseif sampler == "AdvancedHMC"
        return solve_AHMC(
            unfolder, kernel, data, data_errors,
            model, nsamples, nchains, prior_phi, lower_phi, higher_phi, measure_phi
            )
    elseif sampler == "DynamicHMC"
        return solve_DHMC(
            unfolder, kernel, data, data_errors,
            model, nsamples, nchains, prior_phi, lower_phi, higher_phi, measure_phi
            )
    else
        @error "Unknown sampler"
        Base.error("Unknown sampler")
    end
end


function solve_BAT(
    unfolder::MCMCMatrixUnfolder,
    kernel::AbstractMatrix{<:Real},
    data::AbstractVector{<:Real},
    data_errors::AbstractMatrix{<:Real},
    model::Union{Function, String},
    nsamples::Int,
    nchains::Int,
    prior_phi::AbstractVector{<:Real},
    lower_phi::AbstractVector{<:Real},
    higher_phi::AbstractVector{<:Real},
    measure_phi::AbstractVector{<:Real},
    )
    @info "Starting solve_BAT..."

    sig = transpose(unfolder.alphas) * unfolder.omegas
    sig_inv = sym_inv(sig)
    data_errors_inv = sym_inv(data_errors)

    likelihood = let kernel=kernel, prior_phi=prior_phi, lower_phi=lower_phi,
        higher_phi=higher_phi, measure_phi=measure_phi, data=data, data_errors_inv=data_errors_inv
        phi -> begin
            mu = kernel * phi.phi
            bounds_correction = 0
            for i in 1:length(phi.phi)
                if phi.phi[i] > higher_phi[i]
                    bounds_correction -= (phi.phi[i] - higher_phi[i])/measure_phi[i]
                end
                if phi.phi[i] < lower_phi[i]
                    bounds_correction -= (lower_phi[i] - phi.phi[i])/measure_phi[i]
                end
            end
            likelihood_res = -1/2 * transpose(data - mu) * data_errors_inv * (data - mu)
            return likelihood_res + bounds_correction
        end
    end

    if typeof(model) == String
        if model != "Gaussian"
            @error "Unknown model name."
            Base.error("Unknown model name.")
        end
        model = phi -> LogDVal(likelihood(phi))
    end

    if isapprox(det(sig)+1, 1)
        @error "Sigma matrix is singular."
        Base.error("Sigma matrix is singular.")
    end

    prior = NamedTupleDist(phi = MvNormal(prior_phi, sig_inv))
    posterior = BAT.PosteriorDensity(model, prior)
    samples = BAT.bat_sample(posterior, (nsamples, nchains), MetropolisHastings()).result;
    samples_mode = BAT.mode(samples).phi
    samples_cov = BAT.cov(unshaped.(samples))
    return Dict("coeff" => samples_mode, "alphas" => unfolder.alphas, "errors" => samples_cov)
end

struct TurchinProblem
    f::Vector
    data_errors_inv::Matrix
    K::Matrix
    D::Int
    sig::Matrix
    prior_phi::AbstractVector{<:Real}
    lower_phi::AbstractVector{<:Real}
    higher_phi::AbstractVector{<:Real}
    measure_phi::AbstractVector{<:Real}
end

function (problem::TurchinProblem)(θ)
    @unpack phi = θ
    @unpack f, data_errors_inv, K, D, sig, prior_phi, lower_phi, higher_phi, measure_phi = problem
    mu = prior_phi
    prior_log = -1/2 * transpose(phi - mu) * sig * (phi - mu)
    mu_ = K * phi
    likelihood_log = -1/2 * transpose(f - mu_) * data_errors_inv * (f - mu_)
    bounds_correction = 0
    for i in 1:length(phi)
        if phi[i] > higher_phi[i]
            bounds_correction -= (phi[i] - higher_phi[i])/measure_phi[i]
        end
        if phi[i] < lower_phi[i]
            bounds_correction -= (lower_phi[i] - phi[i])/measure_phi[i]
        end
    end
    return prior_log + likelihood_log + bounds_correction
end

function solve_DHMC(
    unfolder::MCMCMatrixUnfolder,
    kernel::AbstractMatrix{<:Real},
    data::AbstractVector{<:Real},
    data_errors::AbstractMatrix{<:Real},
    model::Union{Function, String},
    nsamples::Int,
    nchains::Int,
    prior_phi::AbstractVector{<:Real},
    lower_phi::AbstractVector{<:Real},
    higher_phi::AbstractVector{<:Real},
    measure_phi::AbstractVector{<:Real},
    )
    sig = transpose(unfolder.alphas) * unfolder.omegas
    sig_inv = sym_inv(sig)
    data_errors_inv = sym_inv(data_errors)
    n = unfolder.n

    t = TurchinProblem(data, data_errors_inv, kernel, n, sig,
        prior_phi, lower_phi, higher_phi, measure_phi)

    transT = as((phi = as(Array, asℝ, n),))
    T = TransformedLogDensity(transT, t)
    ∇T = ADgradient(:ForwardDiff, T)

    results = mcmc_with_warmup(Random.GLOBAL_RNG, ∇T, nsamples; reporter = NoProgressReport())
    posterior = TransformVariables.transform.(transT, results.chain)

    coeff = [mean([posterior[j].phi[i] for j in range(1, stop=nsamples)]) for i in range(1, stop=n)]
    errors = [cov([posterior[j].phi[i] for j in range(1, stop=nsamples)]) for i in range(1, stop=n)]

    return Dict("coeff" => coeff, "alphas" => unfolder.alphas, "errors" => errors)

end


function solve_AHMC(
    unfolder::MCMCMatrixUnfolder,
    kernel::AbstractMatrix{<:Real},
    data::AbstractVector{<:Real},
    data_errors::AbstractMatrix{<:Real},
    model::Union{Function, String},
    nsamples::Int,
    nchains::Int,
    prior_phi::AbstractVector{<:Real},
    lower_phi::AbstractVector{<:Real},
    higher_phi::AbstractVector{<:Real},
    measure_phi::AbstractVector{<:Real},
    )

    sig = transpose(unfolder.alphas) * unfolder.omegas
    sig_inv = sym_inv(sig)
    data_errors_inv = sym_inv(data_errors)
    n = unfolder.n

    posterior = let n=n, kernel=kernel, data=data, data_errors=data_errors, sig=sig, prior_phi=prior_phi
        phi -> begin
            mu = prior_phi#zeros(n)
            prior_log = -1/2 * transpose(phi - mu) * sig * (phi - mu)
            bounds_correction = 0
            for i in 1:length(phi)
                if phi[i] > higher_phi[i]
                    bounds_correction -= (phi[i] - higher_phi[i])/measure_phi[i]
                end
                if phi[i] < lower_phi[i]
                    bounds_correction -= (lower_phi[i] - phi[i])/measure_phi[i]
                end
            end
            mu_ = kernel * phi
            likelihood_log = -1/2 * transpose(data - mu_) * data_errors_inv * (data - mu_)
            return prior_log + likelihood_log + bounds_correction
        end
    end

    metric = DiagEuclideanMetric(n)
    hamiltonian = Hamiltonian(metric, posterior, ForwardDiff)
    initial_ϵ = find_good_eps(hamiltonian, prior_phi)
    integrator = Leapfrog(initial_ϵ)
    proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(Preconditioner(metric), NesterovDualAveraging(0.8, integrator))
    n_adapts = 100
    res = AdvancedHMC.sample(hamiltonian, proposal, prior_phi, nsamples, adaptor, n_adapts; progress=true)
    @info "Finished sampling"
    samples = res[1]
    coeff = [mean(getindex.(samples, x)) for x in eachindex(samples[1])]
    errors = [cov(getindex.(samples, x)) for x in eachindex(samples[1])]
    return Dict("coeff" => coeff, "alphas" => unfolder.alphas, "errors" => errors)
end


"""
MCMC model for continuous kernel. Data can be either discrete or continuous.


```julia
MCMCUnfolder(
    basis::Basis,
    omegas::Array{Array{T, 2}, 1} where T<:Real,
    method::String="EmpiricalBayes";
    alphas::Union{AbstractVector{<:Real}, Nothing}=nothing,
    lower::Union{AbstractVector{<:Real}, Nothing}=nothing,
    higher::Union{AbstractVector{<:Real}, Nothing}=nothing,
    initial::Union{AbstractVector{<:Real}, Nothing}=nothing,
    )
```

`basis` -- basis for reconstruction

`omegas` -- array of matrices that provide information about basis functions

`method` -- constant selection method, possible options: "EmpiricalBayes" and "User"

`alphas` -- array of constants, in case method="User" should be provided by user


**Fields**
* `basis::Basis`
* `solver::MCMCMatrixUnfolder`
"""
mutable struct MCMCUnfolder
    basis::Basis
    solver::MCMCMatrixUnfolder

    function MCMCUnfolder(
        basis::Basis,
        omegas::Array{Array{T, 2}, 1} where T<:Real,
        method::String="EmpiricalBayes";
        alphas::Union{AbstractVector{<:Real}, Nothing}=nothing,
        lower::Union{AbstractVector{<:Real}, Nothing}=nothing,
        higher::Union{AbstractVector{<:Real}, Nothing}=nothing,
        initial::Union{AbstractVector{<:Real}, Nothing}=nothing,
        )
        solver = MCMCMatrixUnfolder(
            omegas, method,
            alphas=alphas, lower=lower, higher=higher, initial=initial
        )
        @info "MCMCUnfolder is created."
        return new(basis, solver)
    end
end


"""
```julia
solve(
    mcmcunfolder::MCMCUnfolder,
    kernel::Union{Function, AbstractMatrix{<:Real}},
    data::Union{Function, AbstractVector{<:Real}},
    data_errors::Union{Function, AbstractVector{<:Real}},
    y::Union{AbstractVector{<:Real}, Nothing}=nothing;
    model::Union{Function, String} = "Gaussian",
    nsamples::Int = 10 * 1000,
    nchains::Int = 1,
    algorithm::BAT.AbstractSamplingAlgorithm = MetropolisHastings()
    )
```

**Arguments**
* `unfolder` -- model
* `kernel` -- discrete or continuous kernel
* `data` -- function values
* `data_errors` -- function errors
* `y` -- points to calculate function values and its errors (when data is given as a function)
* `model` -- errors model, "Gaussian" or predefined likelihood function
* `nsamples` -- number of nsamples
* `nchains`-- number of simulation runs to perform
* `algorithm` -- BAT.jl algorithm for sampling

**Returns:** dictionary with coefficients of basis functions and errors.
"""
function solve(
    mcmcunfolder::MCMCUnfolder,
    kernel::Union{Function, AbstractMatrix{<:Real}},
    data::Union{Function, AbstractVector{<:Real}},
    data_errors::Union{Function, AbstractVector{<:Real}},
    y::Union{AbstractVector{<:Real}, Nothing}=nothing;
    model::Union{Function, String} = "Gaussian",
    nsamples::Int = 10 * 1000,
    nchains::Int = 1,
    sampler = "BAT",
    prior_phi::AbstractVector{<:Real},
    lower_phi::AbstractVector{<:Real},
    higher_phi::AbstractVector{<:Real},
    measure_phi::AbstractVector{<:Real},
    )
    @info "Starting solve..."
    kernel_array, data_array, data_errors_array = check_args(
        mcmcunfolder, kernel, data, data_errors, y
        )
    result = solve(
        mcmcunfolder.solver,
        kernel_array, data_array, data_errors_array, model=model,
        nsamples=nsamples, nchains=nchains, sampler=sampler,
        prior_phi=prior_phi, lower_phi=lower_phi, higher_phi=higher_phi, measure_phi=measure_phi
        )
    @info "Ending solve..."
    return result
end
