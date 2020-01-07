"""
Type for function with its support.

```julia
BaseFunction(f, support::Tuple{<:Real,<:Real})
BaseFunction(f, a::Real, b::Real)
```

**Fields**

* `f` -- function (type depends on the basis)
* `support::Tuple{<:Real,<:Real}` -- support of the function
"""
struct BaseFunction
    f
    support::Tuple{<:Real, <:Real}
    BaseFunction(f, support::Tuple{<:Real, <:Real}) = new(f, support)
    BaseFunction(f, a::Real, b::Real) = new(f, (a, b))
end

"""
Abstract type for all bases.
"""
abstract type Basis end


Base.length(basis::Basis) = Base.length(basis.basis_functions)


"""
```julia
discretize_kernel(basis::Basis, kernel::Function, data_points::AbstractVector{<:Real})
```
**Arguments**
* `basis` -- basis
* `kernel` -- kernel function
* `data_points` -- array of data points
**Returns:** discretized kernel `K::Array{Real, 2}`, ``K_{mn} = \\int\\limits_a^b K(x, y_n) \\psi_m(x) dx`` - matrix of size n``\\times``m, where ``m`` - number of basis functions, ``n`` - number of data points.
"""
function discretize_kernel(
    basis::Basis, kernel::Function, data_points::AbstractVector{<:Real}
    )

    @info "Starting discretize kernel..."
    Kmn = zeros(Float64, Base.length(data_points), Base.length(basis))
    for (m, x) in enumerate(data_points)
        for (n, func) in enumerate(basis.basis_functions)
            a, b = func.support
            res = quadgk(y -> kernel(y, x) * func.f(y),
                a, b, rtol=config.RTOL_QUADGK, maxevals=config.MAXEVALS_QUADGK, order=config.ORDER_QUADGK
                )
            Kmn[m, n] = res[1]
        end
    end
    @info "Kernel was discretized successfully."
    return Kmn
end


"""
```julia
omega(basis::Basis, ord::Int)
```
**Arguments**
* `basis` - basis
* `ord` - order of derivatives
**Returns:** `Omega::Array{Real, 2}`, ``\\Omega_{mn} = \\int\\limits_a^b \\frac{d^{ord} \\psi_m}{dx^{ord}} \\frac{d^{ord} \\psi_n}{dx^{ord}}`` - matrix of size n``\\times``n of the mean values of derivatives of order `ord`, where n - number of functions in basis.
"""
omega(basis::Basis, ord::Int) = ()
