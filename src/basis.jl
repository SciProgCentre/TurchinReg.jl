include("config.jl")
using QuadGK, LinearAlgebra, Dierckx, Memoize, ApproxFun


struct BaseFunction
    f
    support::Tuple{Float64, Float64}
    BaseFunction(f, support::Tuple{Float64,Float64}) = new(f, support)
    BaseFunction(f, a::Float64, b::Float64) = new(f, (a, b))
end


abstract type Basis end


Base.length(basis::Basis) = Base.length(basis.basis_functions)


Base.get(basis::Basis, i) = basis.basis_functions[i]


function discretize_kernel(basis::Basis, kernel::Function, xs::Array{Float64, 1})
    Kmn = zeros(Float64, Base.length(xs), Base.length(basis))
    for (m, x) in enumerate(xs)
        for (n, func) in enumerate(basis.basis_functions)
            a, b = func.support
            res = quadgk(y -> kernel(y, x) * func.f(y), a, b, rtol=RTOL_QUADGK, maxevals=MAXEVALS_QUADGK)
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
        func = [BaseFunction(x -> 0.5, a, b)]
        for i = 1:n
            push!(func, BaseFunction(x::Float64 -> cos(i * pi * (x - mid) / l), a, b))
            push!(func, BaseFunction(x::Float64 -> sin(i * pi * (x - mid) / l), a, b))
        end
        return func
    end

    FourierBasis(a::Float64, b::Float64, n::Int64) = new(a, b, n, basis_functions_fourier(n, a, b))
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


#TODO: нормально написать
struct CubicSplineBasis <: Basis
    a::Float64
    b::Float64
    knots::Array{Float64}
    basis_functions::Array

    function basis_functions_cubic_splines(a::Float64, b::Float64, knots::Array{Float64}, boundary::Union{Tuple, Nothing}=nothing)
        basis_functions = []
        for i = 1:length(knots)
            ys = zeros(Float64, Base.length(knots))
            ys[i] = 1.
            func = Spline1D(knots, ys)
            support = (a, b)
            push!(basis_functions, BaseFunction(func, support))
        end

        function apply_cnd(cnd::Union{String, Nothing}, side::String, basis_functions::Array)
            if cnd == nothing
                return basis_functions
            end

            if cnd == "dirichlet"
                if side == "left"
                    return basis_functions[2:end]
                elseif side == "right"
                    return basis_functions[1:end-1]
                else
                    Base.error("CubicSpline: Unknown boundary side: " + side)
                end
            end
            Base.error("CubicSpline: Unknown boundary condition: " + cnd)
        end

        if boundary != nothing
            if length(boundary) == 2
                l,r = boundary
                apply_cnd(l, "left", basis_functions)
                apply_cnd(r, "right", basis_functions)
            elseif length(boundary) == 1
                apply_cnd(boundary, "left", basis_functions)
                apply_cnd(boundary, "right", basis_functions)
            else
                Base.error("Boundary conditions should be Tuple or Nothing")
            end
        end
        return basis_functions
    end

    CubicSplineBasis(
        a::Float64, b::Float64, knots::Array{Float64},
        boundary::Union{Tuple, Nothing}=nothing) = new(knots[1], knots[length(knots)], knots,
        basis_functions_cubic_splines(a, b, knots, boundary))

end

@memoize function omega(basis::CubicSplineBasis, deg::Int64)
    n = Base.length(basis)
    omega = zeros(Float64, n, n)
    for i = 1:n
        for j = i:n
            a_i, b_i = basis.basis_functions[i].support
            a_j, b_j = basis.basis_functions[j].support
            a, b = max(a_i, a_j), min(b_i, b_j)
            if a<b
                omega[i, j] = quadgk(
                    x -> derivative(basis.basis_functions[i].f, x, deg)*derivative(basis.basis_functions[j].f, x, deg),
                    a, b, rtol=RTOL_QUADGK, maxevals=MAXEVALS_QUADGK)[1]
                omega[j, i] = omega[i, j]
            end
        end
    end
    return [omega]
end


#TODO: использовать библиотеку, чтобы это быстро считалось
function legendre_polynomial(n::Int64, x::Float64)
    if n == 0
        return 1.
    elseif n == 1
        return x
    else
        return (2 * n -1) / n * x * legendre_polynomial(n-1, x) - (n - 1) / n * legendre_polynomial(n-2, x)
    end
end

struct LegendreBasis <: Basis
    a::Float64
    b::Float64
    basis_functions::Array

    function basis_functions_legendre(a::Float64, b::Float64, n::Int64)
        basis_functions = []
        for i = 1:n
            func_ = Fun(Legendre(), [zeros(i-1);1])
            func = x -> func_(2 * (x - a) / (b - a) - 1)
            support = (a, b)
            push!(basis_functions, BaseFunction(func, support))
        end
        return basis_functions
    end

    LegendreBasis(a::Float64, b::Float64, n::Int64) = new(a, b, basis_functions_legendre(a, b, n))
end


@memoize function omega(basis::LegendreBasis, deg::Int64)
    n = Base.length(basis)
    omega = zeros(Float64, n, n)
    D = Derivative()
    for i = 1:n
        for j = i:n
            func_i = Fun(Legendre(), [zeros(i-1);1])
            func_j = Fun(Legendre(), [zeros(j-1);1])
            der_func_i = D^deg * func_i
            der_func_j = D^deg * func_j
            omega[i, j] = quadgk(
            x -> (2 / (b - a))^(2 * deg) * der_func_i(2 * (x - a) / (b - a) - 1) * der_func_j(2 * (x - a) / (b - a) - 1),
            a, b, rtol=RTOL_QUADGK, maxevals=MAXEVALS_QUADGK)[1]
            omega[j, i] = omega[i, j]
        end
    end
    return [omega]
end