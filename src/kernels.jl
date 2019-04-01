#=
kernels:
- Julia version: 1.1.0
- Author: ta_nyan
- Date: 2019-03-30
=#

function getOpticsKernels(name::String, alpha::Float64 = 1.)
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
        return Base.error("Unknown name of kernel")
    end
end

function rectangular(x::Float64, alpha::Float64)
    if abs(x)/alpha < 1
        return 1. / alpha
    end
    return 0.
end

function diffraction(x::Float64, alpha::Float64)
    s0 = alpha/0.886
    return (sin(pi * x / s0) / (pi * x / s0))^2 /s0
end

gaussian(x::Float64, alpha::Float64) = (2. / alpha) * sqrt(log(2.) / pi) * exp(-4. * log(2.) * (x / alpha)^2)

function triangular(x::Float64, alpha::Float64)
    if abs(x) / alpha <= 1
        return (1. -  abs(x) / alpha) / alpha
    end
    return 0.
end

dispersive(x::Float64, alpha::Float64) = (alpha / (2. * pi)) / (x^2 + (alpha / 2.)^2)

exponential(x::Float64, alpha::Float64) = (log(2.) / alpha) * exp(-2. * log(2.) * abs(x) / alpha)

function heaviside(x::Float64, alpha::Float64)
    if x > 0
        return 1.
    elseif x < 0
        return 0.
    else
        return 1.0/2
    end
end