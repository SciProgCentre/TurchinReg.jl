#=
kernels:
- Julia version: 1.1.0
- Author: ta_nyan
- Date: 2019-03-30
=#
include("config.jl")

"""
```julia
getOpticsKernels(name::String, alpha::Real = 1.)
```
**Arguments**

* `name` - name of a kernel

* `alpha` - kernel function parameter

**Returns:** kernel, function of 2 variables.

Available kernels:
* `rectangular`:
```math
K(x, y) =
\\begin{cases}
1, \\text{if } \\frac{|x-y|}{\\alpha} < 1
\\
\\text{; } 0 \\text{ otherwise}
\\end{cases}
```

* `diffraction`:
```math
K(x, y) = \\left(\\frac{sin(\\frac{\\pi (x-y)}{s_0})}{\\frac{\\pi (x-y)}{s_0}}\\right)^2
```
```math
s_0 = \\frac{\\alpha}{0.886}
```

* `gaussian`:
```math
K(x, y) = \\frac{2}{\\alpha}\\sqrt{\\frac{\\ln2}{\\pi}}e^{4\\ln2\\left(\\frac{x-y}{\\alpha}\\right)^2}
```

* `triangular`:
```math
K(x, y) =
\\begin{cases}
\\frac{1 - \\frac{|x-y|}{\\alpha}}{\\alpha}, \\text{if } \\frac{|x-y|}{\\alpha} < 1
\\
\\text{; } 0 \\text{ otherwise}
\\end{cases}
```

* `dispersive`:
```math
K(x, y) = \\frac{\\alpha}{2 \\pi}\\left((x-y)^2 + \\left(\\frac{\\alpha}{2}\\right)^2\\right)
```

* `exponential`:
```math
K(x, y) = \\frac{\\ln2}{\\alpha}e^{2\\ln2\\frac{|x-y|}{\\alpha}}
```

* `heaviside`:
```math
K(x, y) =
\\begin{cases}
1, \\text{if } x>0
\\
\\text{; }  0 \\text{ otherwise}
\\end{cases}
```
"""
function getOpticsKernels(name::String, alpha::Real = 1.)
    if name == "rectangular"
        return (x, y) -> rectangular(x-y, alpha)
    elseif name == "diffraction"
        return (x, y) -> diffraction(x-y, alpha)
    elseif name == "gaussian"
        return (x, y) -> gaussian(x-y, alpha)
    elseif name == "triangular"
        return (x, y) -> triangular(x-y, alpha)
    elseif name == "dispersive"
        return (x, y) -> dispersive(x-y, alpha)
    elseif name == "exponential"
        return (x, y) -> exponential(x-y, alpha)
    elseif name == "heaviside"
        return (x, y) -> heaviside(x-y, alpha)
    else
        return Base.error("Unknown name of a kernel")
    end
end

function rectangular(x::Real, alpha::Real)
    if abs(x)/alpha < 1
        return 1. / alpha
    end
    return 0.
end

function diffraction(x::Real, alpha::Real)
    s0 = alpha/0.886
    return (sin(pi * x / s0) / (pi * x / s0))^2 /s0
end

gaussian(x::Real, alpha::Real) = (2. / alpha) * sqrt(log(2.) / pi) * exp(-4. * log(2.) * (x / alpha)^2)

function triangular(x::Real, alpha::Real)
    if abs(x) / alpha <= 1
        return (1. -  abs(x) / alpha) / alpha
    end
    return 0.
end

dispersive(x::Real, alpha::Real) = (alpha / (2. * pi)) / (x^2 + (alpha / 2.)^2)

exponential(x::Real, alpha::Real) = (log(2.) / alpha) * exp(-2. * log(2.) * abs(x) / alpha)

function heaviside(x::Real, alpha::Real)
    if x > 0
        return 1.
    elseif x < 0
        return 0.
    else
        return 1.0/2
    end
end
