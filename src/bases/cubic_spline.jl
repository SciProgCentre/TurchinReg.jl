"""
Cubic spline basis on given knots with length ``n``.

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
* `basis_functions::AbstractVector` -- array of basis functions
"""
struct CubicSplineBasis <: Basis
    a::Real
    b::Real
    knots::AbstractVector{<:Real}
    basis_functions::AbstractVector

    function basis_functions_cubic_splines(
        segment_a::Real, segment_b::Real, knots::AbstractVector{<:Real},
        boundary_condition::Tuple{Union{String, Nothing}, Union{String, Nothing}}
        )

        k = 3
        basis_functions = []
        for i = 0:(length(knots)-5)
            func = BSpline(i, k, knots)
            push!(basis_functions, func)
        end
        left_condition, right_condition = boundary_condition
        beg_function = 1
        end_function = length(basis_functions)

        if left_condition == "dirichlet"
            beg_function = 2
        elseif left_condition != nothing
            @error "CubicSplineBasis: Unknown boundary condition: " + condition
            Base.error(
            "CubicSplineBasis: Unknown boundary condition: " + condition)
        end

        if right_condition == "dirichlet"
            end_function -= 1
        elseif right_condition != nothing
            @error "CubicSplineBasis: Unknown boundary condition: " + condition
            Base.error(
            "CubicSplineBasis: Unknown boundary condition: " + condition)
        end

        return basis_functions[beg_function:end_function]
    end

    function CubicSplineBasis(knots::AbstractVector{<:Real},
        boundary_condition::Union{Tuple{Union{String, Nothing}, Union{String, Nothing}}, Nothing, String}=nothing)

        @assert sort(knots) == knots "Array of knots should be sorted in ascending order"
        @assert length(Set(knots)) == length(knots) "There should not be two equal elements in the list of knots"

        knots = [
                [knots[1] for i = 1:4];
                knots[3:end-2];
                [knots[end] for i = 1:4]
                ]
        if typeof(boundary_condition) == String || typeof(boundary_condition) == Nothing
            basis_functions = basis_functions_cubic_splines(
                knots[1], knots[end], knots, (boundary_condition, boundary_condition)
                )
            return new(knots[1], knots[end], knots, basis_functions)
        else
            basis_functions = basis_functions_cubic_splines(
                knots[1], knots[end], knots, boundary_condition
                )
            return new(knots[1], knots[end], knots, basis_functions)
        end
    end

    function CubicSplineBasis(
        a::Real, b::Real, n::Int,
        boundary_condition::Union{Tuple{Union{String, Nothing}, Union{String, Nothing}}, Nothing, String}=nothing
        )
        n += count(i->(i == "dirichlet"), boundary_condition)
        return CubicSplineBasis(collect(range(a, stop=b, length=n)), boundary_condition)
    end

end

@memoize function omega(basis::CubicSplineBasis, order::Int=2)
    @info "Calculating omega matrix for Cubis spline basis derivatives of order $order..."
    @assert order >= 0 "Order of derivative should be positive"
    n = length(basis)
    a, b = basis.a, basis.b
    omega = zeros(Float64, n, n)
    for i = 1:n
        for j = i:n
            omega[i, j] = integral(
                PiecewisePolynomials.derivative(basis.basis_functions[i].func, order=order) * PiecewisePolynomials.derivative(basis.basis_functions[j].func, order=order),
                a, b
                )
            omega[j, i] = omega[i, j]
        end
    end
    @info "Omega caclulated successfully."
    return omega
end
