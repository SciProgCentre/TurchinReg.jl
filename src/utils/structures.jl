make_sym(A::AbstractMatrix{<:Real}) = (transpose(A) + A) / 2
sym_inv(A::AbstractMatrix{<:Real}) = make_sym(pinv(A))


"""
Abstract type for all algorithms for choosing regularization parameters
"""
abstract type AlphasType end


"""
``alphas = argmax(P(alphas|f))``
Mode of the distribution is found by sampling using BAT.jl

```julia
ArgmaxBAT(alphas::Union{AbstractVector{<:Real}, Nothing}=nothing,
lower::Union{AbstractVector{<:Real}, Nothing}=nothing,
higher::Union{AbstractVector{<:Real}, Nothing}=nothing,
initial::Union{AbstractVector{<:Real}, Nothing}=nothing,
algo::MCMCAlgorithm=MetropolisHastings(),
nchains::Int=2,
nsamples::Int=1e4,
)
```
**Fields**
* `alphas::Union{AbstractVector{<:Real}, Nothing}` -- values of regularization parameters
* `lower::Union{AbstractVector{<:Real}, Nothing}` -- lower possible values of regularization parameters
* `higher::Union{AbstractVector{<:Real}, Nothing}` -- higher possible values of regularization parameters
* `initial::Union{AbstractVector{<:Real}, Nothing}` -- initial values of regularization parameters
* `algo::MCMCAlgorithm` -- algorithm of sampling (for more info see BAT.jl documentation)
* `nchains::Int` -- number of chains to sample
* `nsamples::Int` -- number of samples
"""
mutable struct ArgmaxBAT<:AlphasType
    alphas::Union{AbstractVector{<:Real}, Nothing}
    lower::Union{AbstractVector{<:Real}, Nothing}
    higher::Union{AbstractVector{<:Real}, Nothing}
    initial::Union{AbstractVector{<:Real}, Nothing}
    algo::MCMCAlgorithm
    nchains::Int
    nsamples::Int

    ArgmaxBAT(alphas::Union{AbstractVector{<:Real}, Nothing}=nothing,
    lower::Union{AbstractVector{<:Real}, Nothing}=nothing,
    higher::Union{AbstractVector{<:Real}, Nothing}=nothing,
    initial::Union{AbstractVector{<:Real}, Nothing}=nothing,
    algo::MCMCAlgorithm=MetropolisHastings(),
    nchains::Int=2,
    nsamples::Int=10000,
    ) = new(alphas, lower, higher, initial, algo, nchains, nsamples)
end


"""
``alphas = argmax(P(alphas|f))``
Mode of the distribution is found by sampling using Optim.jl

```julia
ArgmaxOptim(alphas::Union{AbstractVector{<:Real}, Nothing}=nothing,
lower::Union{AbstractVector{<:Real}, Nothing}=nothing,
higher::Union{AbstractVector{<:Real}, Nothing}=nothing,
initial::Union{AbstractVector{<:Real}, Nothing}=nothing,
algo=BFGS()
)
```


**Fields**
* `alphas::Union{AbstractVector{<:Real}, Nothing}` -- values of regularization parameters
* `lower::Union{AbstractVector{<:Real}, Nothing}` -- lower possible values of regularization parameters
* `higher::Union{AbstractVector{<:Real}, Nothing}` -- higher possible values of regularization parameters
* `initial::Union{AbstractVector{<:Real}, Nothing}` -- initial values of regularization parameters
* `algo` -- algorithm of optimization (for more info see Optim.jl documentation)
"""
mutable struct ArgmaxOptim<:AlphasType
    alphas::Union{AbstractVector{<:Real}, Nothing}
    lower::Union{AbstractVector{<:Real}, Nothing}
    higher::Union{AbstractVector{<:Real}, Nothing}
    initial::Union{AbstractVector{<:Real}, Nothing}
    algo

    ArgmaxOptim(alphas::Union{AbstractVector{<:Real}, Nothing}=nothing,
    lower::Union{AbstractVector{<:Real}, Nothing}=nothing,
    higher::Union{AbstractVector{<:Real}, Nothing}=nothing,
    initial::Union{AbstractVector{<:Real}, Nothing}=nothing,
    algo=BFGS()
    ) = new(alphas, lower, higher, initial, algo)
end


"""
Allows to compute ``\\varphi`` averaging over all parameters

```julia
Marginalize()
```

**Fields**
* `alphas::Nothing` -- no fixed regularization parameters
"""
mutable struct Marginalize<:AlphasType
    alphas::Nothing

    Marginalize() = new(nothing)
end

"""
User-defined parameters

```julia
User(alphas::AbstractVector{<:Real})
```

**Fields**
* `alphas::AbstractVector{<:Real}` -- fixed regularization parameters
"""
mutable struct User<:AlphasType
    alphas::AbstractVector{<:Real}

    User(alphas::AbstractVector{<:Real}) = new(alphas)
end


"""
Bounds, initial values and measure of of influence of the bounds on the solution

```julia
PhiBounds(lower::Union{AbstractVector{<:Real}, Nothing}=nothing,
higher::Union{AbstractVector{<:Real}, Nothing}=nothing,
initial::Union{AbstractVector{<:Real}, Nothing}=nothing,
measure::Union{AbstractVector{<:Real}, Nothing}=nothing,
)
```

**Fields**
* `lower::Union{AbstractVector{<:Real}, Nothing}` -- lover bounds of solution vector components
* `higher::Union{AbstractVector{<:Real}, Nothing}` -- higher bounds of solution vector components
* `initial::Union{AbstractVector{<:Real}, Nothing}` -- initial values of solution vector components
* `measure::Union{AbstractVector{<:Real}, Nothing}`  -- measure of solution vector components
"""
mutable struct PhiBounds
    lower::Union{AbstractVector{<:Real}, Nothing}
    higher::Union{AbstractVector{<:Real}, Nothing}
    initial::Union{AbstractVector{<:Real}, Nothing}
    measure::Union{AbstractVector{<:Real}, Nothing}

    PhiBounds(lower::Union{AbstractVector{<:Real}, Nothing}=nothing,
    higher::Union{AbstractVector{<:Real}, Nothing}=nothing,
    initial::Union{AbstractVector{<:Real}, Nothing}=nothing,
    measure::Union{AbstractVector{<:Real}, Nothing}=nothing,
    ) = new(lower, higher, initial, measure)
end


"""
Abstract type for all solution algorithms
"""
abstract type AlgoType end

"""
Analytical solution

```julia
Analytically()
```
**Fields**
no fields
"""
mutable struct Analytically<:AlgoType end


"""
Solve with BATSampling

```julia
BATSampling(log_data_distribution::Union{Function, Nothing}=nothing,
algo::MCMCAlgorithm=MetropolisHastings(),
nchains::Int=2,
nsamples::Int=1e4
)
```

**Fields**
* `log_data_distribution::Union{Function, Nothing}` -- logarithm of data distribution
* `algo::MCMCAlgorithm` -- algorithm foe sampling (foe more info see BAT.jlj documentation)
* `nchains::Int` -- number of chains to sample
* `nsamples::Int` -- number of samples
"""
mutable struct BATSampling<:AlgoType
    log_data_distribution::Union{Function, Nothing}
    algo::MCMCAlgorithm
    nchains::Int
    nsamples::Int

    BATSampling(log_data_distribution::Union{Function, Nothing}=nothing,
    algo::MCMCAlgorithm=MetropolisHastings(),
    nchains::Int=2,
    nsamples::Int=10000
    ) = new(log_data_distribution, algo, nchains, nsamples)
end


"""
Solve with AHMCSampling

```julia
AHMCSampling(log_data_distribution::Union{Function, Nothing}=nothing,
nchains::Int=1,
nsamples::Int=1e4
)
```

**Fields**
* `log_data_distribution::Union{Function, Nothing}` -- logarithm of data distribution
* `nchains::Int` -- number of chains to sample
* `nsamples::Int` -- number of samples
"""
mutable struct AHMCSampling<:AlgoType
    log_data_distribution::Union{Function, Nothing}
    nchains::Int
    nsamples::Int

    AHMCSampling(log_data_distribution::Union{Function, Nothing}=nothing,
    nchains::Int=1,
    nsamples::Int=10000
    ) = new(log_data_distribution, nchains, nsamples)
end


"""
Solve with DHMCSampling

```julia
DHMCSampling(log_data_distribution::Union{Function, Nothing}=nothing,
nchains::Int=1,
nsamples::Int=1e4
)
```

**Fields**
* `log_data_distribution::Union{Function, Nothing}` -- logarithm of data distribution
* `nchains::Int` -- number of chains to sample
* `nsamples::Int` -- number of samples
"""
mutable struct DHMCSampling<:AlgoType
    log_data_distribution::Union{Function, Nothing}
    nchains::Int
    nsamples::Int

    DHMCSampling(log_data_distribution::Union{Function, Nothing}=nothing,
    nchains::Int=1,
    nsamples::Int=10000
    ) = new(log_data_distribution, nchains, nsamples)
end

function make_bounds(alphas::AlphasType, omegas::Union{Array{Array{T, 2}, 1}, Nothing} where T<:Real)
    if typeof(alphas) != User
        if alphas.lower == nothing
            alphas.lower = [config.ALPHAS_LOWER for _ in 1:length(omegas)]
        end
        if alphas.higher == nothing
            alphas.higher = [config.ALPHAS_HIGHER for _ in 1:length(omegas)]
        end
        if alphas.initial == nothing
            alphas.initial = [config.ALPHAS_INITIAL for _ in 1:length(omegas)]
        end
        str = "lower, higher, initial alphas and omegas must have same lengths"
        @assert length(alphas.lower) == length(alphas.initial) str
        @assert length(alphas.lower) == length(alphas.higher) str
        @assert length(alphas.lower) == length(omegas) str
    end
    return alphas
end

function make_bounds(phi_bounds::Union{PhiBounds, Nothing}, basis::Basis)
    if phi_bounds == nothing
        phi_bounds = PhiBounds()
    end
    if phi_bounds.lower == nothing
        phi_bounds.lower = [config.PHI_LOWER for _ in 1:length(basis.basis_functions)]
    end
    if phi_bounds.higher == nothing
        phi_bounds.higher = [config.PHI_HIGHER for _ in 1:length(basis.basis_functions)]
    end
    if phi_bounds.initial == nothing
        phi_bounds.initial = [config.PHI_INITIAL for _ in 1:length(basis.basis_functions)]
    end
    if phi_bounds.measure == nothing
        phi_bounds.measure = [config.PHI_MEASURE for _ in 1:length(basis.basis_functions)]
    end
    str = "lower, higher, initial, omegas must have same lengths"
    @assert length(phi_bounds.lower) == length(phi_bounds.initial) str
    @assert length(phi_bounds.lower) == length(phi_bounds.higher) str
    @assert length(phi_bounds.lower) == length(phi_bounds.measure) str
    return phi_bounds
end

function log_bounds_correction(phi::AbstractVector{<:Real}, phi_bounds::PhiBounds)
    bounds_correction = 0
    for i in 1:length(phi)
        if phi[i] > phi_bounds.higher[i]
            bounds_correction -= (phi[i] - phi_bounds.higher[i])/phi_bounds.measure[i]
        end
        if phi[i] < phi_bounds.lower[i]
            bounds_correction -= (phi_bounds.lower[i] - phi[i])/phi_bounds.measure[i]
        end
    end
    return bounds_correction
end
