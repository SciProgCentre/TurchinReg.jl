#=
This module contain definitions for basis of functions which could be
used to describe function being unfolded.

There are several standard interfaces which are used in this module:

HaveSupport: x.support should return interval where function is not
     zero as 2-tuple. None should be used to designate infinities
=#

using QuadGK, LinearAlgebra, Dierckx, Memoize

struct my_func
    f::Function
    support::Tuple{Float64, Float64}
    my_func(f, a::Float64, b::Float64) = new(f, (a, b))
end

    #TODO: inf support

abstract type  Basis
    #=
    Parent class for basis in functional space
    Basis of functions for describing true distribution.

    Should implement
    ----------------
    support: 2-tuple
        see HaveSupport for details

    basisFun: [float -> float | None]
        list of basis function-like objects, each should implement
        HaveSupport. If basis distribution is not representable as
        function None should be present in the list.

    omega(self, deg) :
        function to calculate matrix for integral of n-th
        derivatives. deg is degree of derivative. It's allowed to add
        more optional parameters

    aristotelianA(self) [optional]:
        Matrix for implementation of Aristotelian boundary condition
        on left border of basis

    aristotelianB(self) [optional]:
        Matrix for implementation of Aristotelian boundary condition
        on right border of basis

    aristotelianAB(self) [optional,default]:
        Matrix for implementation of Aristotelian boundary condition
        on both borders where both are assumed to be same.
    =#
end

function call(basis::Basis, coeff::Array{Float64}, x::Float64)
    #=
        Evaluate function at the point. All basis functions which
        aren't representable as functions are discarded.

        Parameters
        ----------
        coef : vector of float
            coefficients for basis functions
        x    : float
            point to evaluate function
    =#
    res::Float64 = 0

    for i in range(1:Base.length(coeff))
        fun = Base.get(basis, i)
        res += coef[i] * fun.f(x)
    end
    return res
end


Base.length(basis::Basis) = Base.length(basis.basisFun)

Base.get(basis::Basis, i) = basis.basisFun[i]


function discretizeKernel(basis::Basis, K::Function, xs::Array{Float64, 1})
    #=
        Discretize convolution kernel:

        -- math: f(x) = \\int K(y,x) \\phi(y) dy

        Parameters
        ----------
        K  : 2-arg function
            2-parameter convolution kernel
        ys : 1darray
            points for observed data

        Returns
        -------
        K : matrix
    =#
    Kmn = zeros(Float64, Base.length(xs), Base.length(basis))
    for (m, x) in enumerate(xs)
        for (n, f) in enumerate(basis.basisFun)
            a, b = f.support
            r = quadgk(y -> K(y,x) * f.f(y), a, b, maxevals=10^7)[1]
            Kmn[m, n] = r
        end
    end
    return Kmn
end


struct FourierBasis <: Basis
    a::Float64
    b::Float64
    n::Int32
    basisFun::Array{my_func}

    function basisFunFourier(n::Int64, a::Float64, b::Float64)
        l::Float64 = (b - a) / 2.
        mid::Float64 = (a + b) / 2.
        func = [my_func(x -> 0.5, a, b)]
        for i = 1:n
            push!(func, my_func(x::Float64 -> cos(i * pi * (x - mid) / l), a, b))
            push!(func, my_func(x::Float64 -> sin(i * pi * (x - mid) / l), a, b))
        end
        return func
    end
    FourierBasis(a::Float64, b::Float64, n::Int64) = new(a, b, n, basisFunFourier(n, a, b))
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
    #TODO: переписать
    result = Array{Array{Float64, 2}}(undef, 1)
    result[1] = cat(temp...; dims=(1,2))
    return result
end


#TODO: нормально написать
# struct CubicSplineBasis <: Basis
#     a::Float64
#     b::Float64
#     knots::Array{Float64}
#     basisFun::Array
#
#     function basisFuncCubicSplines(a::Float64, b::Float64, knots::Array{Float64}, boundary::Union{Tuple, Nothing})
#         basisFun = []
#         for i = 1:lenght(knots)
#             ys = zeros(Float64, length(knots))
#             ys[i] = 1.
#             func = Spline1D(knots, ys)
#             support = (a, b)
#             push!(basisFun, my_func(func, support))
#             #TODO: сделать нормальный сплайн. с таким не будут работать краевые условия
#         end
#
#         function apply_cnd!(cnd::Union{String, Nothing}, side::String, basisFun::Array)
#             if cnd == nothing
#                 return basisFun
#             end
#
#             if cnd == "dirichlet"
#                 if side == "l"
#                     return basisFun[2:lenght(basisFun)]
#                 elseif side == "r"
#                     return basisFun[1:lenght(basisFun)-1]
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
#                 apply_cnd(l, "l", basisFun)
#                 apply_cnd(r, "r", basisFun)
#             elseif length(boundary) == 1
#                 apply_cnd(boundary, "l", basisFun)
#                 apply_cnd(boundary, "r", basisFun)
#             else
#                 Base.error("Boundary conditions should be tuple")
#             end
#         end
#         return basisFun
#     end
#
#     CubicSplineBasis(a::Float64, b::Float64, knots::Array{Float64}, boundary::Union{Tuple, Nothing}=nothing) = new(knots[1], knots[lenght(knots)], knots, basisFuncCubicSplines(a, b, knots, boundary))
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
#     pp = [spline_derivative(deg, basisFun, x) for basisFun in basis.basisFun]
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