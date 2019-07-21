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
discretize_kernel(basis::Basis, kernel::Function, xs::Array{Float64, 1})
```

## Basis

```@docs
Basis
```

```@docs
BaseFunction
```

```@docs
omega(basis::Basis, order::Int64)
```

### Fourier basis

```@docs
FourierBasis
```

### Cubic Spline basis

```@docs
CubicSplineBasis
```

### Legendre polynomial basis

```@docs
LegendreBasis
```

### Bernstein polynomial basis

```@docs
BernsteinBasis
```

## Model

```@docs
GaussErrorMatrixUnfolder
```

```@docs
GaussErrorUnfolder
```


## Reconstruction

```@docs
solve(
    unfolder::GaussErrorMatrixUnfolder,
    kernel::Array{Float64, 2},
    data::Array{Float64, 1},
    data_errors::Union{Array{Float64, 1}, Array{Float64, 2}},
    )
```

```@docs
solve(
    gausserrorunfolder::GaussErrorUnfolder,
    kernel::Union{Function, Array{Float64, 2}},
    data::Union{Function, Array{Float64, 1}},
    data_errors::Union{Function, Array{Float64, 1}},
    y::Union{Array{Float64, 1}, Nothing},
    )
```
## Result

```@docs
PhiVec
```
