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
* `basis_functions::AbstractVector{BaseFunction}` -- array of basis functions
* `boundary_condition::Tuple{Union{String, Nothing}, Union{String, Nothing}}` -- boundary conditions of basis functions. If tuple, the first element affects left bound, the second element affects right bound. If string, both sides are affected. Possible options: `"dirichlet"`, `nothing`.
"""
struct BernsteinBasis <: Basis
    a::Real
    b::Real
    basis_functions::AbstractVector{BaseFunction}
    boundary_condition::Tuple{Union{String, Nothing}, Union{String, Nothing}}

    @memoize function basis_functions_bernstein(
        a::Real, b::Real, n::Int,
        boundary_condition::Tuple{Union{String, Nothing}, Union{String, Nothing}}
        )

        basis_functions = []
        for k = 0:n
            coeff = convert(Float64, binomial(BigInt(n),BigInt(k)))
            func = x::Float64 -> coeff  *
                ((x - a) / (b - a))^k *
                (1 - ((x - a) / (b - a)))^(n - k)
            support = (a, b)
            push!(basis_functions, BaseFunction(func, support))
        end

        function apply_condition(
            condition::Union{String, Nothing},
            side::String,
            basis_functions::AbstractVector
            )

            if condition == nothing
                return basis_functions
            end

            if condition == "dirichlet"
                if side == "left"
                    return basis_functions[2:end]
                else
                    return basis_functions[1:end-1]
                end
            end
            @error "BernsteinPolynomial: Unknown boundary condition: " + condition
            Base.error(
            "BernsteinPolynomial: Unknown boundary condition: " + condition)
        end

        left_condition, right_condition = boundary_condition
        basis_functions = apply_condition(
            left_condition, "left", basis_functions
            )
        basis_functions = apply_condition(
            right_condition, "right", basis_functions
            )

        return basis_functions
    end

    function BernsteinBasis(
        a::Real, b::Real, n::Int,
        boundary_condition::Union{
            Tuple{Union{String, Nothing}, Union{String, Nothing}},
            Nothing,
            String}=nothing
            )

        if a >= b
            @error "Incorrect specification of a segment: `a` should be less than `b`."
            Base.error("Incorrect specification of a segment: `a` should be less than `b`.")
        end
        if n <= 0
            @error "Number of basis functions should be positive."
            Base.error("Number of basis functions should be positive.")
        end

        if typeof(boundary_condition) == String || typeof(boundary_condition) == Nothing
            basis_functions = basis_functions_bernstein(
                a, b, n, (boundary_condition, boundary_condition)
                )
            @info "Bernstein polynomials basis is created."
            return new(
                a, b, basis_functions,
                (boundary_condition, boundary_condition)
                )
        else
            basis_functions = basis_functions_bernstein(
                a, b, n, boundary_condition
                )
            @info "Bernstein polynomials basis is created."
            return new(a, b, basis_functions, boundary_condition)
        end
    end
end


@memoize function omega(basis::BernsteinBasis, order::Int)
    @info "Calculating omega matrix for Bernstein polynomials basis derivatives of order $order..."
    if order < 0
        @error "Order of derivative should be positive."
        Base.error("Order of derivative should be positive.")
    end
    left_condition, right_condition = basis.boundary_condition
    n = Base.length(basis) - 1
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

    begin_function_number = 0
    end_function_number = n
    n_true_value = n

    if left_condition == "dirichlet"
        n_true_value += 1
    end

    if right_condition == "dirichlet"
        n_true_value += 1
    end

    omega = zeros(Float64, n_true_value+1, n_true_value+1)

    for i = 0:n_true_value
        for j = i:n_true_value
            omega[i+1, j+1] = quadgk(
            x::Float64 ->
            derivative(i, n_true_value, order, x) *
            derivative(j, n_true_value, order, x),
            a, b, rtol=config.RTOL_QUADGK,
            maxevals=config.MAXEVALS_QUADGK, order=config.ORDER_QUADGK)[1]
            omega[j+1, i+1] = omega[i+1, j+1]
        end
    end

    if left_condition == "dirichlet"
        omega = omega[2:end, 2:end]
    end

    if right_condition == "dirichlet"
        omega = omega[1:end-1, 1:end-1]
    end
    @info "Omega caclulated successfully."
    return omega
end
