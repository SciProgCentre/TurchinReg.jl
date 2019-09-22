include("b_spline_implementation.jl")

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


"""
Fourier basis with length ``2n+1``: {``0.5``, ``\\sin(\\frac{\\pi (x - \\frac{a+b}{2})}{b-a})``, ``\\cos(\\frac{\\pi (x - \\frac{a+b}{2})}{b-a})``, ..., ``\\sin(\\frac{\\pi n (x - \\frac{a+b}{2})}{b-a})``, ``\\cos(\\frac{\\pi n (x - \\frac{a+b}{2})}{b-a})``}.

```julia
FourierBasis(a::Real, b::Real, n::Int)
```
`a`, `b` -- the beginning and the end of the segment
`n` -- number of basis functions

**Fields**

* `a::Real` -- beginning of the support
* `b::Real` -- end of the support
* `n::Int` -- number of basis functions
* `basis_functions::AbstractVector{BaseFunction}` -- array of basis functions
"""
struct FourierBasis <: Basis
    a::Real
    b::Real
    n::Int
    basis_functions::AbstractVector{BaseFunction}

    function basis_functions_fourier(n::Int, a::Real, b::Real)
        l = (b - a) / 2.
        mid = (a + b) / 2.
        func = [BaseFunction(x::Float64 -> 0.5, a, b)]
        for i = 1:n
            push!(func, BaseFunction(x::Float64 ->
            cos(i * pi * (x - mid) / l), a, b))
            push!(func, BaseFunction(x::Float64 ->
            sin(i * pi * (x - mid) / l), a, b))
        end
        return func
    end

    function FourierBasis(a::Real, b::Real, n::Int)
        if a >= b
            @error "Incorrect specification of a segment: `a` should be less than `b`."
            Base.error("Incorrect specification of a segment: `a` should be less than `b`.")
        end
        if n <= 0
            @error "Number of basis functions should be positive."
            Base.error("Number of basis functions should be positive.")
        end
        basis_functions = basis_functions_fourier(n, a, b)
        @info "Fourier basis is created."
        return new(a, b, n, basis_functions)
    end
end


@memoize function omega(basis::FourierBasis, order::Int)
    @info "Calculating omega matrix for Fourier basis derivatives of order $order..."
    if order < 0
        @error "Order of derivative should be positive."
        Base.error("Order of derivative should be positive.")
    end
    a, b = basis.a, basis.b
    delta = (b - a) / 2
    temp = zeros(Float64, 2 * basis.n + 1)
    if order == 0
        temp[1] = delta
    end
    for i = 1:basis.n
        val = ((i * pi) / delta) ^ (2 * order) * delta / 2
        temp[2 * i] = val
        temp[2 * i + 1] = val
    end
    @info "Omega caclulated successfully."
    return cat(temp...; dims=(1,2))
end


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
            omega[i, j] = int(
                der_order(basis.basis_functions[i].f.func, order) * der_order(basis.basis_functions[j].f.func, order),
                segment_a, segment_b
                )
            omega[j, i] = omega[i, j]
        end
    end
    @info "Omega caclulated successfully."
    return omega
end


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
    basis_functions::AbstractVector{BaseFunction}

    function basis_functions_legendre(a::Real, b::Real, n::Int)
        basis_functions = []
        for i = 1:n
            func_ = Fun(Legendre(), [zeros(i-1);1])
            func = x::Float64 -> func_(2 * (x - a) / (b - a) - 1)
            support = (a, b)
            push!(basis_functions, BaseFunction(func, support))
        end
        return basis_functions
    end

    function LegendreBasis(a::Real, b::Real, n::Int)
        if a >= b
            @error "Incorrect specification of a segment: `a` should be less than `b`."
            Base.error("Incorrect specification of a segment: `a` should be less than `b`.")
        end
        if n <= 0
            @error "Number of basis functions should be positive."
            Base.error("Number of basis functions should be positive.")
        end
        basis_functions = basis_functions_legendre(a, b, n)
        @info "Legendre polynomials basis is created."
        return new(a, b, basis_functions)
    end
end


@memoize function omega(basis::LegendreBasis, order::Int)
    @info "Calculating omega matrix for Legendre polynomials basis derivatives of order $order..."
    if order < 0
        @error "Order of derivative should be positive."
        Base.error("Order of derivative should be positive.")
    end
    n = Base.length(basis)
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
