# User's Guide

## Kernel
Kernel can be specified as a matrix or as a function.
It is possible to set arbitrary function of 2 variables or use one of predefined kernels.


```@docs
getOpticsKernels(name::String,)
```

```@docs
discretize_kernel(basis::Basis, kernel::Function, data_points::AbstractVector{<:Real})
```

## Basis

```@docs
Basis
```

```@docs
BaseFunction
```

```@docs
omega(basis::Basis, order::Int)
```

### Fourier basis

```@docs
FourierBasis
```

### Cubic Spline basis

```@docs
CubicSplineBasis
```

### Legendre polynomials basis

```@docs
LegendreBasis
```

### Bernstein polynomials basis

```@docs
BernsteinBasis
```

## Gaussian noise distribution

### Model

```@docs
GaussErrorMatrixUnfolder
```

```@docs
GaussErrorUnfolder
```

### Reconstruction

```@docs
solve(
    unfolder::GaussErrorMatrixUnfolder,
    kernel::AbstractMatrix{<:Real},
    data::AbstractVector{<:Real},
    data_errors::AbstractVecOrMat{<:Real}
    )
```

```@docs
solve(
    unfolder::GaussErrorUnfolder,
    kernel::Union{Function, AbstractMatrix{<:Real}},
    data::Union{Function, AbstractVector{<:Real}},
    data_errors::Union{Function, AbstractVector{<:Real}},
    y::Union{AbstractVector{<:Real}, Nothing}=nothing,
    )
```

## Non-Gaussian noise distribution

### Model

```@docs
MCMCMatrixUnfolder
```

```@docs
MCMCUnfolder
```

### Reconstruction

```@docs
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

```@docs
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

## Result

```@docs
PhiVec
```
