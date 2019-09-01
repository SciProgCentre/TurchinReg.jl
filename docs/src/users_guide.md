# User's Guide

## Kernel
Kernel can be specified as a matrix or as a function.
Initialize a kernel as a function:


```@docs
getOpticsKernels(name::String,)
```

Available kernels:
* `rectangular`:
```math
K(x, y) =
\begin{cases}
1, \text{if } \frac{|x-y|}{\alpha} < 1
\\
0, \text{otherwise}
\end{cases}
```

* `diffraction`:
```math
K(x, y) = \left(\frac{sin(\frac{\pi (x-y)}{s_0})}{\frac{\pi (x-y)}{s_0}}\right)^2
```
```math
s_0 = \frac{\alpha}{0.886}
```

* `gaussian`:
```math
K(x, y) = \frac{2}{\alpha}\sqrt{\frac{\ln2}{\pi}}e^{4\ln2\left(\frac{x-y}{\alpha}\right)^2}
```

* `triangular`:
```math
K(x, y) =
\begin{cases}
\frac{1 - \frac{|x-y|}{\alpha}}{\alpha}, \text{if } \frac{|x-y|}{\alpha} < 1
\\
0, \text{otherwise}
\end{cases}
```

* `dispersive`:
```math
K(x, y) = \frac{\alpha}{2 \pi}\left((x-y)^2 + \left(\frac{\alpha}{2}\right)^2\right)
```

* `exponential`:
```math
K(x, y) = \frac{\ln2}{\alpha}e^{2\ln2\frac{|x-y|}{\alpha}}
```

* `heaviside`:
```math
K(x, y) =
\begin{cases}
1, \text{if } x>0
\\
0, \text{otherwise}
\end{cases}
```

```@docs
discretize_kernel(basis::Basis, kernel::Function, xs::Array{<:Real, 1})
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

## Gaussian noise distribution with alpha as argmax of posterior probability

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
    kernel::Array{<:Real, 2},
    data::Array{<:Real, 1},
    data_errors::Union{Array{<:Real, 1}, Array{<:Real, 2}},
    )
```

```@docs
solve(
    gausserrorunfolder::GaussErrorUnfolder,
    kernel::Union{Function, Array{<:Real, 2}},
    data::Union{Function, Array{<:Real, 1}},
    data_errors::Union{Function, Array{<:Real, 1}},
    y::Union{Array{<:Real, 1}, Nothing},
    )
```

## Any othes noise distribution with alpha as argmax of posterior probability

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
