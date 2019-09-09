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
    gausserrorunfolder::GaussErrorUnfolder,
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
    kernel::Union{Function, Array{<:Real, 2}},
    data::Union{Function, Array{<:Real, 1}},
    data_errors::Union{Function, Array{<:Real, 1}},
    y::Union{Array{<:Real, 1}, Nothing}=nothing,
    chains::Int = 1,
    samples::Int = 10 * 1000
    )
```

```@docs
solve(
    unfolder::MCMCMatrixUnfolder,
    kernel::Array{<:Real, 2},
    data::Array{<:Real, 1},
    data_errors::Union{Array{<:Real, 1}, Array{<:Real, 2}},
    chains::Int = 1,
    samples::Int = 10 * 1000
    )
```

## Result

```@docs
get_values(sim::ModelChains, chains::Int, n::Int)
```

```@docs
PhiVec
```
