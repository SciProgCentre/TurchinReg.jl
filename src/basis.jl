using QuadGK, LinearAlgebra, Dierckx, Memoize


struct BaseFunction
    f::Function
    support::Tuple{Float64, Float64}
    BaseFunction(f::Function, a::Float64, b::Float64) = new(f, (a, b))
end


abstract type Basis end


Base.length(basis::Basis) = Base.length(basis.basis_functions)


Base.get(basis::Basis, i) = basis.basis_functions[i]


function discretize_kernel(basis::Basis, kernel::Function, xs::Array{Float64, 1})
    Kmn = zeros(Float64, Base.length(xs), Base.length(basis))
    for (m, x) in enumerate(xs)
        for (n, func) in enumerate(basis.basis_functions)
            a, b = func.support
            res = quadgk(y -> kernel(y, x) * func.f(y), a, b, rtol=10^-5, maxevals=5*10^2)
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
        temp[2 * i - 1] = val
        temp[2 * i] = val
    end
    return [cat(temp...; dims=(1,2))]
end


#TODO: нормально написать
# struct CubicSplineBasis <: Basis
#     a::Float64
#     b::Float64
#     knots::Array{Float64}
#     basis_functions::Array
#
#     function basis_functionscCubicSplines(a::Float64, b::Float64, knots::Array{Float64}, boundary::Union{Tuple, Nothing})
#         basis_functions = []
#         for i = 1:lenght(knots)
#             ys = zeros(Float64, length(knots))
#             ys[i] = 1.
#             func = Spline1D(knots, ys)
#             support = (a, b)
#             push!(basis_functions, BaseFunction(func, support))
#             #TODO: сделать нормальный сплайн. с таким не будут работать краевые условия
#         end
#
#         function apply_cnd!(cnd::Union{String, Nothing}, side::String, basis_functions::Array)
#             if cnd == nothing
#                 return basis_functions
#             end
#
#             if cnd == "dirichlet"
#                 if side == "l"
#                     return basis_functions[2:lenght(basis_functions)]
#                 elseif side == "r"
#                     return basis_functions[1:lenght(basis_functions)-1]
#                 else
#                     Base.error("CubicSpline: Unknown boundary side: " + side)
#                 end
#             end
#             Base.error("CubicSpline: Unknown boundary condition: " + cnd)
#         end
#
#         if boundary != nothing
#             if length(boundary) == 2
#                 l,r = boundary
#                 apply_cnd(l, "l", basis_functions)
#                 apply_cnd(r, "r", basis_functions)
#             elseif length(boundary) == 1
#                 apply_cnd(boundary, "l", basis_functions)
#                 apply_cnd(boundary, "r", basis_functions)
#             else
#                 Base.error("Boundary conditions should be tuple")
#             end
#         end
#         return basis_functions
#     end
#
#     CubicSplineBasis(a::Float64, b::Float64, knots::Array{Float64}, boundary::Union{Tuple, Nothing}=nothing) = new(knots[1], knots[lenght(knots)], knots, basis_functionscCubicSplines(a, b, knots, boundary))
#
# end
#
# @memoize function omega(basis::CubicSplineBasis, deg::Int32, equalize::Bool=False)
#     #=
#         Calculate matrix of second derivatives for regularization matrix.
#
#         Parameters
#         ----------
#         deg       : int
#             Number of differentiations in omega operator
#         equalize : (optional) bool
#             If set to true integral will be weighted to ensure that
#             they contribute equally
#         =#
#     function spline_derivative(deg::Int32, spl, x::Array)
#         return derivative(spl, x, deg)
#     end
#
#     pp = [spline_derivative(deg, basis_functions, x) for basis_functions in basis.basis_functions]
#     pdeg  = 2 * (3 - deg) + 1
#     n = length(basis)
#     omega = zeros(Float64, n, n)
#     for i = 1:n
#         for j = 1:(i+1)
#             c1 = pp[i].c
#             c2 = pp[j].c
#             c = zeros(Int32, pdeg, size(c1)[2] )
#             for k1 = 1:(3 - deg +1)
#                 for k2 = 1:(3 - deg +1)
#                     c[k1+k2] = c[k1+k2] + c1[k1] * c2[k2]
#                 end
#             end
#             p = 1
#             #TODO: разобраться с омегой
#         end
#     end
# end
