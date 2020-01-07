"""
Cubic spline basis on given knots with length ``n``, where n -- length of knots array.

```julia
CubicSplineBasis(
    knots::AbstractVector{<:Real},
    boundary_condition::Union{Tuple{Union{String, Nothing}, Union{String, Nothing}}, Nothing, String}=nothing
    )
```

```julia
CubicSplineBasis(
    a::Real, b::Real, n::Int,
    boundary_condition::Union{Tuple{Union{String, Nothing}, Union{String, Nothing}}, Nothing, String}=nothing
    )
```

`knots` -- knots of spline
`boundary_condition` -- boundary conditions of basis functions. If tuple, the first element affects left bound, the second element affects right bound. If string, both sides are affected. Possible options: `"dirichlet"`, `nothing`

**Fields**

* `a::Real` -- beginning of the support, matches the first element of the array `knots`
* `b::Real` -- end of the support, matches the last element of the array `knots`
* `knots::AbstractVector{<:Real}` -- array of points on which the spline is built
* `basis_functions::AbstractVector{BaseFunction}` -- array of basis functions
"""
struct CubicSplineBasis <: Basis
    a::Real
    b::Real
    knots::AbstractVector{<:Real}
    basis_functions::AbstractVector{BaseFunction}

    function basis_functions_cubic_splines(
        segment_a::Real, segment_b::Real, knots::AbstractVector{<:Real},
        boundary_condition::Tuple{Union{String, Nothing}, Union{String, Nothing}}
        )

        k = 3
        basis_functions = []
        for i = 0:(length(knots)-5)
            func = BSpline(i, k, knots)
            support = (segment_a, segment_b)
            push!(basis_functions, BaseFunction(func, support))
        end

        function apply_condition(
            condition::Union{String, Nothing}, side::String,
            basis_functions::Array
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
            @error "CubicSplineBasis: Unknown boundary condition: " + condition
            Base.error(
            "CubicSplineBasis: Unknown boundary condition: " + condition)
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

    function CubicSplineBasis(knots::AbstractVector{<:Real},
        boundary_condition::Union{Tuple{Union{String, Nothing}, Union{String, Nothing}}, Nothing, String}=nothing)

        if sort(knots) != knots
            @error "Array of knots should be sorted in ascending order"
            Base.error("Array of knots should be sorted in ascending order")
        end
        if length(Set(knots)) != length(knots)
            @error "There should not be two equal elements in the list of knots"
            Base.error("There should not be two equal elements in the list of knots")
        end

        knots = [
                [knots[1] for i = 1:4];
                knots[3:end-2];
                [knots[end] for i = 1:4]
                ]
        segment_a, segment_b = knots[1], knots[end]
        if typeof(boundary_condition) == String || typeof(boundary_condition) == Nothing
            basis_functions = basis_functions_cubic_splines(
                segment_a, segment_b, knots, (boundary_condition, boundary_condition)
                )
            @info "Cubic spline basis is created."
            return new(knots[1], knots[end], knots, basis_functions)
        else
            basis_functions = basis_functions_cubic_splines(
                segment_a, segment_b, knots, boundary_condition
                )
            @info "Cubic spline basis is created."
            return new(knots[1], knots[end], knots, basis_functions)
        end
    end

    function CubicSplineBasis(
        a::Real, b::Real, n::Int,
        boundary_condition::Union{Tuple{Union{String, Nothing}, Union{String, Nothing}}, Nothing, String}=nothing
        )
        return CubicSplineBasis(collect(range(a, stop=b, length=n)), boundary_condition)
    end

end

@memoize function omega(basis::CubicSplineBasis, order::Int=2)
    @info "Calculating omega matrix for Cubis spline basis derivatives of order $order..."
    if order < 0
        @error "Order of derivative should be positive."
        Base.error("Order of derivative should be positive.")
    end
    n = length(basis)
    omega = zeros(Float64, n, n)
    for i = 1:n
        for j = i:n
            segment_a = max(basis.basis_functions[i].support[1], basis.basis_functions[j].support[1])
            segment_b = min(basis.basis_functions[i].support[2], basis.basis_functions[j].support[2])
            omega[i, j] = integral(
                PiecewisePolynomials.derivative(basis.basis_functions[i].f.func, order=order) * PiecewisePolynomials.derivative(basis.basis_functions[j].f.func, order=order),
                segment_a, segment_b
                )
            omega[j, i] = omega[i, j]
        end
    end
    @info "Omega caclulated successfully."
    return omega
end
