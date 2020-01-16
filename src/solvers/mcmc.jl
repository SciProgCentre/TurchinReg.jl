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
        m = check_args(omegas, method, alphas, lower, higher, initial)
        @info "MCMCMatrixUnfolder is created."
        return new(omegas, m, method, alphas, lower, higher, initial)
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
    model::Union{Model, String} = "Gaussian",
    samples::Int = 10 * 1000,
    burnin::Int = 0,
    thin::Int = 1,
    chains::Int = 1,
    verbose::Bool = false
    )
```

**Arguments**
* `unfolder` -- model
* `kernel` -- discrete kernel
* `data` -- function valuess
* `data_errors` -- function errors
* `model` -- errors model, "Gaussian" or predefined Mamba.jl model
* `burnin`-- numer of initial draws to discard as a burn-in sequence to allow for convergence
* `thin` -- step-size between draws to output
* `chains`-- number of simulation runs to perform
* `verbose` -- whether to print sampler progress at the console

**Returns:** parameters for mcmc() function.
"""
function solve(
    unfolder::MCMCMatrixUnfolder,
    kernel::AbstractMatrix{<:Real},
    data::AbstractVector{<:Real},
    data_errors::AbstractVecOrMat{<:Real};
    model::Union{Model, String} = "Gaussian",
    samples::Int = 10 * 1000,
    burnin::Int = 0,
    thin::Int = 1,
    chains::Int = 1,
    verbose::Bool = false
    )

    @info "Starting solve..."
    data_errors = check_args(unfolder, kernel, data, data_errors)
    data_errors = check_args(unfolder, kernel, data, data_errors)
    data_errorsInv = make_sym(pinv(data_errors))
    B = make_sym(transpose(kernel) * data_errorsInv * kernel)
    b = transpose(kernel) * transpose(data_errorsInv) * data
    initial = unfolder.initial
    if unfolder.method == "EmpiricalBayes"
        unfolder.alphas = find_optimal_alpha(
            unfolder.omegas, B, b,
            unfolder.initial, unfolder.lower, unfolder.higher
            )
    elseif unfolder.method != "User"
        @error "Unknown method" + unfolder.method
        Base.eror("Unknown method" + unfolder.method)
    end
    @info "Ending solve..."
    return solve_MCMC(
        unfolder, kernel, data, data_errors,
        model, samples, burnin, thin, chains, verbose
        )
end

function solve_MCMC(
    unfolder::MCMCMatrixUnfolder,
    kernel::AbstractMatrix{<:Real},
    data::AbstractVector{<:Real},
    data_errors::AbstractMatrix{<:Real},
    model::Union{Model, String} = "Gaussian",
    samples::Int = 10 * 1000,
    burnin::Int = 0,
    thin::Int = 1,
    chains::Int = 1,
    verbose::Bool = false
    )
    @info "Starting solve_MCMC..."
    if typeof(model) == String
        if model != "Gaussian"
            @error "Unknown model name."
            Base.error("Unknown model name.")
        end
        model = Model(
            phi = Stochastic(1, (n, sigma) ->  MvNormal(zeros(n), sigma)),
            mu = Logical(1, (kernel, phi) -> kernel * phi, false),
            f = Stochastic(1, (mu, data_errors) ->  MvNormal(mu, data_errors), false),
            )
    end

    scheme = [AMM([:phi], 1 * Matrix{Float64}(I, unfolder.n, unfolder.n))]
    val = det(transpose(unfolder.alphas) * unfolder.omegas)+1
    if isapprox(val, 1)
        @error "Sigma matrix is singular."
        Base.error("Sigma matrix is singular.")
    end

    line = Dict{Symbol, Any}(
        :f => data,
        :kernel => kernel,
        :data_errors => data_errors,
        :n => unfolder.n,
        :sigma => make_sym(pinv(transpose(unfolder.alphas) * unfolder.omegas)),
        )

    inits = [Dict{Symbol, Any}(
            :phi => rand(Normal(0, 1), unfolder.n),
            :f => data
            ) for i in 1:chains]

    setsamplers!(model, scheme)
    @info "Ending solve_MCMC..."
    return (model, line, inits, samples), (
        :burnin => burnin,
        :thin => thin,
        :chains => chains,
        :verbose => verbose)
end


"""
Allowers to get coefficients and errors from generated data set.

```julia
get_values(sim::ModelChains)
```

**Arguments**
* `sim` -- data generated by `mcmc()`

**Returns:** `Dict{String, AbstractVector{Real}}` with coefficients ("coeff") and errors ("errors").

"""
function get_values(sim::ModelChains)
    shape = size(sim.value)
    chains = shape[3]
    n = shape[2]
    values = [sim.value[:, :, j] for j in range(1, stop=chains)]
    res =  mean(values)
    coeff = []
    cov_ = cov(res)
    for i in range(1, stop=n)
        append!(coeff, mean(res[:, i]))
    end
    return Dict(
        "coeff" => convert(AbstractVector{Real}, coeff),
        "errors" => cov(res),
        )
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
    model::Union{Model, String} = "Gaussian",
    samples::Int = 10 * 1000,
    burnin::Int = 0,
    thin::Int = 1,
    chains::Int = 1,
    verbose::Bool = false
    )
```

**Arguments**
* `unfolder` -- model
* `kernel` -- discrete or continuous kernel
* `data` -- function values
* `data_errors` -- function errors
* `y` -- points to calculate function values and its errors (when data is given as a function)
* `model` -- errors model, "Gaussian" or predefined Mamba.jl model
* `burnin`-- numer of initial draws to discard as a burn-in sequence to allow for convergence
* `thin` -- step-size between draws to output
* `chains`-- number of simulation runs to perform
* `verbose` -- whether to print sampler progress at the console

**Returns:** parameters for mcmc() function.
"""
function solve(
    mcmcunfolder::MCMCUnfolder,
    kernel::Union{Function, AbstractMatrix{<:Real}},
    data::Union{Function, AbstractVector{<:Real}},
    data_errors::Union{Function, AbstractVector{<:Real}},
    y::Union{AbstractVector{<:Real}, Nothing}=nothing;
    model::Union{Model, String} = "Gaussian",
    samples::Int = 10 * 1000,
    burnin::Int = 0,
    thin::Int = 1,
    chains::Int = 1,
    verbose::Bool = false
    )
    @info "Starting solve..."
    kernel_array, data_array, data_errors_array = check_args(
        mcmcunfolder, kernel, data, data_errors, y
        )
    result = solve(
        mcmcunfolder.solver,
        kernel_array, data_array, data_errors_array,
        model=model, samples=samples,
        burnin=burnin, thin=thin, chains=chains, verbose=verbose
        )
    @info "Ending solve..."
    return result
end
