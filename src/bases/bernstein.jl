"""
Bernstein polynomials basis.

```julia
BernsteinBasis(
    a::Real, b::Real, n::Int,
    boundary_condition::Union{Tuple{Union{String, Nothing}, Union{String, Nothing}}, Nothing, String}=nothing
    )
```

`a`, `b` -- the beginning and the end of the segment
`n` -- number of basis functions
`boundary_condition` -- boundary conditions of basis functions. If tuple, the first element affects left bound, the second element affects right bound. If string, both sides are affected. Possible options: `"dirichlet"`, `nothing`.

**Fields**

* `a::Real` -- beginning of the support
* `b::Real` -- end of the support
* `basis_functions::AbstractVector{Function}` -- array of basis functions
* `boundary_condition::Tuple{Union{String, Nothing}, Union{String, Nothing}}` -- boundary conditions of basis functions. Possible options: `"dirichlet"`, `nothing`.
"""
struct BernsteinBasis <: Basis
    a::Real
    b::Real
    basis_functions::AbstractVector
    boundary_condition::Tuple{Union{String, Nothing}, Union{String, Nothing}}

    @memoize function basis_functions_bernstein(
        a::Real, b::Real, n::Int,
        boundary_condition::Tuple{Union{String, Nothing}, Union{String, Nothing}}
        )
        basis_functions = []
        n -= 1
        left_condition, right_condition = boundary_condition
        beg_function = 1
        end_function = n + 1

        if left_condition == "dirichlet"
            n += 1
            beg_function = 2
        elseif left_condition != nothing
            @error "BernsteinPolynomial: Unknown boundary condition: " + condition
            Base.error(
            "BernsteinPolynomial: Unknown boundary condition: " + condition)
        end

        if right_condition == "dirichlet"
            n += 1
            end_function = n-1
        elseif right_condition != nothing
            @error "BernsteinPolynomial: Unknown boundary condition: " + condition
            Base.error(
            "BernsteinPolynomial: Unknown boundary condition: " + condition)
        end

        for k = 0:n
            coeff = convert(Float64, binomial(BigInt(n),BigInt(k)))
            func = x::Float64 -> coeff  *
                ((x - a) / (b - a))^k *
                (1 - ((x - a) / (b - a)))^(n - k)
            push!(basis_functions, func)
        end

        return basis_functions[beg_function:end_function]
    end

    function BernsteinBasis(
        a::Real, b::Real, n::Int,
        boundary_condition::Union{
            Tuple{Union{String, Nothing}, Union{String, Nothing}},
            Nothing,
            String}=nothing
            )
        @assert a < b "Incorrect specification of a segment: `a` should be less than `b`"
        @assert n > 0 "Number of basis functions should be positive"
        if typeof(boundary_condition) == String || typeof(boundary_condition) == Nothing
            basis_functions = basis_functions_bernstein(
                a, b, n, (boundary_condition, boundary_condition)
                )
            return new(
                a, b, basis_functions,
                (boundary_condition, boundary_condition)
                )
        else
            basis_functions = basis_functions_bernstein(
                a, b, n, boundary_condition
                )
            return new(a, b, basis_functions, boundary_condition)
        end
    end
end


@memoize function omega(basis::BernsteinBasis, order::Int)
    @info "Calculating omega matrix for Bernstein polynomials basis derivatives of order $order..."
    @assert order >= 0 "Order of derivative should be non-negative"
    left_condition, right_condition = basis.boundary_condition
    n = length(basis)
    a = basis.a
    b = basis.b

    @memoize function basis_function_bernstein(k::Int, n::Int, x::Real)
        if k < 0 || n < 0 || k > n
            return 0.
        end
        coeff = convert(Float64, binomial(BigInt(n),BigInt(k)))
        return coeff *
            ((x - a) / (b - a))^k *
            (1 - ((x - a) / (b - a)))^(n - k)
    end

    @memoize function derivative(k::Int, n::Int, l::Int, x::Real)
        coeff = [convert(Float64, binomial(BigInt(l),BigInt(j))) for j = 0:l]
        basis_functions = [basis_function_bernstein(k-j, n-l, x) for j = 0:l]
        return convert(
            Float64, binomial(BigInt(n),BigInt(l)) * factorial(BigInt(l))
            ) / (b - a)^l * sum(coeff .* basis_functions)
    end

    beg_omega = 1
    end_omega = n

    if left_condition == "dirichlet"
        n += 1
        beg_omega = 2
    end

    if right_condition == "dirichlet"
        n += 1
        end_omega = n-1
    end

    omega = zeros(Float64, n, n)

    for i = 1:n
        for j = 1:n
            omega[i, j] = quadgk(
            x::Float64 ->
            derivative(i-1, n, order, x) *
            derivative(j-1, n, order, x),
            a, b, rtol=config.RTOL_QUADGK,
            maxevals=config.MAXEVALS_QUADGK, order=config.ORDER_QUADGK)[1]
            omega[j, i] = omega[i, j]
        end
    end
    @info "Omega caclulated successfully."
    return omega[beg_omega:end_omega, beg_omega:end_omega]
end
