"""
Abstract type for all bases.
"""
abstract type Basis end
Base.length(basis::Basis) = Base.length(basis.basis_functions)

"""
```julia
discretize_kernel(basis::Basis, kernel::Function, measurement_points::AbstractVector{<:Real})
```
**Arguments**
* `basis` -- basis
* `kernel` -- kernel function
* `measurement_points` -- array of data points
**Returns:** discretized kernel `K::Array{Real, 2}`, ``K_{mn} = \\int\\limits_a^b K(x, y_n) \\psi_m(x) dx`` - matrix of size n``\\times``m, where ``m`` - number of basis functions, ``n`` - number of data points.
"""
function discretize_kernel(
    basis::Basis, kernel::Function, measurement_points::AbstractVector{<:Real}
    )
    @info "Starting discretize kernel..."
    Kmn = zeros(Float64, length(measurement_points), length(basis))
    a, b = basis.a, basis.b
    for (m, x) in enumerate(measurement_points)
        for (n, func) in enumerate(basis.basis_functions)
            res = quadgk(y -> kernel(y, x) * func(y),
                a, b, rtol=config.RTOL_QUADGK, maxevals=config.MAXEVALS_QUADGK,
                order=config.ORDER_QUADGK
                )
            Kmn[m, n] = res[1]
        end
    end
    @info "Kernel was discretized successfully."
    return Kmn
end


"""
```julia
omega(basis::Basis, order::Int)
```
**Arguments**
* `basis` - basis
* `order` - order of derivatives
**Returns:** `Omega::Array{Real, 2}`, ``\\Omega_{mn} = \\int\\limits_a^b \\frac{d^{order} \\psi_m}{dx^{order}} \\frac{d^{order} \\psi_n}{dx^{order}}`` - matrix of size n``\\times``n of the mean values of derivatives of order `order`, where n - number of functions in basis.
"""
omega(basis::Basis, order::Int) = ()
