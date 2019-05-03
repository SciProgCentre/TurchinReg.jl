include("config.jl")
include("b_spline_implementation.jl")
using QuadGK, LinearAlgebra, Dierckx, Memoize, ApproxFun


struct BaseFunction
    f
    support::Tuple{Float64, Float64}
    BaseFunction(f, support::Tuple{Float64,Float64}) = new(f, support)
    BaseFunction(f, a::Float64, b::Float64) = new(f, (a, b))
end


abstract type Basis end


Base.length(basis::Basis) = Base.length(basis.basis_functions)


function discretize_kernel(basis::Basis, kernel::Function, xs::Array{Float64, 1})
    Kmn = zeros(Float64, Base.length(xs), Base.length(basis))
    for (m, x) in enumerate(xs)
        for (n, func) in enumerate(basis.basis_functions)
            a, b = func.support
            res = quadgk(y -> kernel(y, x) * func.f(y),
                a, b, rtol=RTOL_QUADGK, maxevals=MAXEVALS_QUADGK
                )
            Kmn[m, n] = res[1]
        end
    end
    return Kmn
end


struct FourierBasis <: Basis
    a::Float64
    b::Float64
    n::Int64
    basis_functions::Array{BaseFunction}

    function basis_functions_fourier(n::Int64, a::Float64, b::Float64)
        l = (b - a) / 2.
        mid = (a + b) / 2.
        func = [BaseFunction(x::Float64 -> 0.5, a, b)]
        for i = 1:n
            push!(func, BaseFunction(x::Float64 -> cos(i * pi * (x - mid) / l), a, b))
            push!(func, BaseFunction(x::Float64 -> sin(i * pi * (x - mid) / l), a, b))
        end
        return func
    end

    FourierBasis(a::Float64, b::Float64, n::Int64) = new(
        a, b, n, basis_functions_fourier(n, a, b)
        )
end


@memoize function omega(basis::FourierBasis, deg::Int64)
    a, b = basis.a, basis.b
    delta = (b - a) / 2
    temp = zeros(Float64, 2 * basis.n + 1)
    if deg == 0
        temp[1] = delta
    end
    for i = 1:basis.n
        val = ((i * pi) / delta) ^ (2 * deg) * delta / 2
        temp[2 * i] = val
        temp[2 * i + 1] = val
    end
    return [cat(temp...; dims=(1,2))]
end


#TODO: find suitable basis functions
struct CubicSplineBasis <: Basis
    a::Float64
    b::Float64
    knots::Array{Float64}
    basis_functions::Array

    function basis_functions_cubic_splines(
        a::Float64, b::Float64, knots::Array{Float64},
        boundary_condition::Tuple{Union{String, Nothing}, Union{String, Nothing}})
        k = 3
        basis_functions = []
        for i = 0:(length(knots)-5)
            func = BSpline(i, k, knots)
            support = (a, b)
            push!(basis_functions, BaseFunction(func, support))
        end

        function apply_condition(
            condition::Union{String, Nothing}, side::String, basis_functions::Array
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
            Base.error(
            "BernsteinPolynomial: Unknown boundary condition: " + condition)
        end

        left_condition, right_condition = boundary_condition
        basis_functions = apply_condition(left_condition, "left", basis_functions)
        basis_functions = apply_condition(right_condition, "right", basis_functions)

        return basis_functions
    end

    function CubicSplineBasis(knots::Array{Float64},
        boundary_condition::Union{Tuple{Union{String, Nothing}, Union{String, Nothing}}, Nothing, String}=nothing)

        knots = append!(append!([knots[1], knots[1], knots[1]], knots), [knots[end], knots[end], knots[end]])

        if typeof(boundary_condition) == String || typeof(boundary_condition) == Nothing
            return new(knots[1], knots[length(knots)], knots,
                basis_functions_cubic_splines(a, b, knots, (boundary_condition, boundary_condition)))
        else
            return new(knots[1], knots[length(knots)], knots,
                basis_functions_cubic_splines(a, b, knots, boundary_condition))
        end
    end

end

@memoize function omega(basis::CubicSplineBasis, deg::Int64)
    n = Base.length(basis)
    omega = zeros(Float64, n, n)
    for i = 1:n
        for j = i:n
            a_i, b_i = basis.basis_functions[i].support
            a_j, b_j = basis.basis_functions[j].support
            # a, b = max(a_i, a_j), min(b_i, b_j)
            a, b = basis.a, basis.b
            # if a < b
                omega[i, j] = quadgk(
                    x::Float64 -> derivative(basis.basis_functions[i].f, x, deg) * derivative(basis.basis_functions[j].f, x, deg),
                        a, b, rtol=RTOL_QUADGK, maxevals=MAXEVALS_QUADGK)[1]
                if omega[i, j] == 0 && i == j
                    println("omega[i, j] = 0 !!!")
                    q = collect(range(a, b, length=500))
                    plot(q, [derivative(basis.basis_functions[i].f, x, deg) * derivative(basis.basis_functions[j].f, x, deg) for x in q])
                end
                omega[j, i] = omega[i, j]
            # end
        end
    end
    return [omega]
end


struct LegendreBasis <: Basis
    a::Float64
    b::Float64
    basis_functions::Array

    function basis_functions_legendre(a::Float64, b::Float64, n::Int64)
        basis_functions = []
        for i = 1:n
            func_ = Fun(Legendre(), [zeros(i-1);1])
            func = x::Float64 -> func_(2 * (x - a) / (b - a) - 1)
            support = (a, b)
            push!(basis_functions, BaseFunction(func, support))
        end
        return basis_functions
    end

    LegendreBasis(a::Float64, b::Float64, n::Int64) = new(
        a, b, basis_functions_legendre(a, b, n)
        )
end


@memoize function omega(basis::LegendreBasis, deg::Int64)
    n = Base.length(basis)
    a = basis.a
    b = basis.b
    omega = zeros(Float64, n, n)
    D = Derivative()
    for i = 1:n
        for j = i:n
            func_i = Fun(Legendre(), [zeros(i-1);1])
            func_j = Fun(Legendre(), [zeros(j-1);1])
            der_func_i = D^deg * func_i
            der_func_j = D^deg * func_j
            omega[i, j] = quadgk(
            x::Float64 -> (2 / (b - a))^(2 * deg) *
            der_func_i(2 * (x - a) / (b - a) - 1) *
            der_func_j(2 * (x - a) / (b - a) - 1),
            a, b, rtol=RTOL_QUADGK, maxevals=MAXEVALS_QUADGK)[1]
            omega[j, i] = omega[i, j]
        end
    end
    return [omega]
end


#TODO: не работает из-за вырождения омеги
struct BernsteinBasis <: Basis
    a::Float64
    b::Float64
    basis_functions::Array
    boundary_condition::Tuple{Union{String, Nothing}, Union{String, Nothing}}

    @memoize function basis_functions_bernstein(
        a::Float64, b::Float64, n::Int64,
        boundary_condition::Tuple{Union{String, Nothing}, Union{String, Nothing}}
        )

        basis_functions = []
        for k = 0:n
            coeff = convert(Float64, binomial(BigInt(n),BigInt(k)))
            println(coeff)
            func = x::Float64 -> coeff  *
                ((x - a) / (b - a))^k *
                (1 - ((x - a) / (b - a)))^(n - k)
            support = (a, b)
            push!(basis_functions, BaseFunction(func, support))
        end

        function apply_condition(
            condition::Union{String, Nothing}, side::String, basis_functions::Array
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
            Base.error(
            "BernsteinPolynomial: Unknown boundary condition: " + condition)
        end

        left_condition, right_condition = boundary_condition
        basis_functions = apply_condition(left_condition, "left", basis_functions)
        basis_functions = apply_condition(right_condition, "right", basis_functions)

        return basis_functions
    end

    function BernsteinBasis(a::Float64, b::Float64, n::Int64,
    boundary_condition::Union{Tuple{Union{String, Nothing}, Union{String, Nothing}}, Nothing, String}=nothing)
        if typeof(boundary_condition) == String || typeof(boundary_condition) == Nothing
            return new(a, b, basis_functions_bernstein(a, b, n, (boundary_condition, boundary_condition)), (boundary_condition, boundary_condition))
        else
            return return new(a, b, basis_functions_bernstein(a, b, n, boundary_condition), boundary_condition)
        end
    end
end

@memoize function omega(basis::BernsteinBasis, deg::Int64)
    left_condition, right_condition = basis.boundary_condition
    n = Base.length(basis) - 1
    a = basis.a
    b = basis.b

    @memoize function basis_function_bernstein(k::Int64, n::Int64, x::Float64)
        if k < 0 || n < 0 || k > n
            return 0.
        end
        coeff = convert(Float64, binomial(BigInt(n),BigInt(k)))
        return coeff *
            ((x - a) / (b - a))^k *
            (1 - ((x - a) / (b - a)))^(n - k)
    end

    @memoize function derivative(k::Int64, n::Int64, l::Int64, x::Float64)
        coeff = [convert(Float64, binomial(BigInt(l),BigInt(j))) for j = 0:l]
        basis_functions = [basis_function_bernstein(k-j, n-l, x) for j = 0:l]
        return convert(Float64, binomial(BigInt(n),BigInt(l)) * factorial(BigInt(l))) /
            (b - a)^l * sum(coeff .* basis_functions)
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
            x::Float64 -> derivative(i, n_true_value, deg, x) * derivative(j, n_true_value, deg, x),
            a, b, rtol=RTOL_QUADGK, maxevals=MAXEVALS_QUADGK)[1]
            omega[j+1, i+1] = omega[i+1, j+1]
        end
    end

    if left_condition == "dirichlet"
        omega = omega[2:end, 2:end]
    end

    if right_condition == "dirichlet"
        omega = omega[1:end-1, 1:end-1]
    end

    return [omega]
end
