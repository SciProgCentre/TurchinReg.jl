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
