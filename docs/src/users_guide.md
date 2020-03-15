# User's Guide

## Kernel
Kernel can be specified as a matrix or as a function.
It is possible to set arbitrary function of 2 variables or use one of predefined kernels.


```@docs
getOpticsKernels(name::String,)
```

```@docs
discretize_kernel(basis::Basis, kernel::Function, measurement_points::AbstractVector{<:Real})
```

## Bases

```@docs
Basis
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

## Parameters of reconstruction algorithm

### Regularization parameters

```@docs
AlphasType
```

```@docs
ArgmaxBAT
```

```@docs
ArgmaxOptim
```

```@docs
Marginalize
```

```@docs
User
```

### Reconstruction algorithms

```@docs
AlgoType
```

```@docs
Analytically
```

```@docs
BATSampling
```

```@docs
AHMCSampling
```

```@docs
DHMCSampling
```


## Reconstruction

```@docs
solve
```

## Result

```@docs
PhiVec
```
