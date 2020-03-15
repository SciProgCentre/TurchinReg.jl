"""
Legendre polynomials basis with length ``n``.

```julia
LegendreBasis(a::Real, b::Real, n::Int)
```

`a`, `b` -- the beginning and the end of the support
`n` -- number of basis functions

**Fields**

* `a::Real` -- beginning of the support
* `b::Real` -- end of the support
* `basis_functions::AbstractVector{BaseFunction}` -- array of basis functions
"""
struct LegendreBasis <: Basis
    a::Real
    b::Real
    basis_functions::AbstractVector

    function basis_functions_legendre(a::Real, b::Real, n::Int)
        basis_functions = []
        for i = 1:n
            func_ = Fun(Legendre(), [zeros(i-1);1])
            func = x::Float64 -> func_(2 * (x - a) / (b - a) - 1)
            push!(basis_functions, func)
        end
        return basis_functions
    end

    function LegendreBasis(a::Real, b::Real, n::Int)
        @assert a < b "Incorrect specification of a segment: `a` should be less than `b`"
        @assert n > 0 "Number of basis functions should be positive"
        basis_functions = basis_functions_legendre(a, b, n)
        return new(a, b, basis_functions)
    end
end


@memoize function omega(basis::LegendreBasis, order::Int)
    @info "Calculating omega matrix for Legendre polynomials basis derivatives of order $order..."
    @assert order >= 0 "Order of derivative should be positive"
    n = length(basis)
    a = basis.a
    b = basis.b
    omega = zeros(Float64, n, n)
    D = Derivative()
    for i = 1:n
        for j = i:n
            func_i = Fun(Legendre(), [zeros(i-1);1])
            func_j = Fun(Legendre(), [zeros(j-1);1])
            der_func_i = D^order * func_i
            der_func_j = D^order * func_j
            omega[i, j] = quadgk(
            x::Float64 -> (2 / (b - a))^(2 * order) *
            der_func_i(2 * (x - a) / (b - a) - 1) *
            der_func_j(2 * (x - a) / (b - a) - 1),
            a, b, rtol=config.RTOL_QUADGK,
            maxevals=config.MAXEVALS_QUADGK, order=config.ORDER_QUADGK
            )[1]
            omega[j, i] = omega[i, j]
        end
    end
    @info "Omega caclulated successfully."
    return omega
end
