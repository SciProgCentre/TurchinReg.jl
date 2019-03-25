#=
vector:
- Julia version: 1.1.0
- Author: ta_nyan
- Date: 2019-03-17
=#

struct PhiVec
    coeff::Array{Float64}
    basis::Basis
    sig::Union{Array{Float64}, Nothing}
    #=
    PhiVec(self, coef, basis, sig=None)

    Generalized Vector for discretization of true function class.
    Parameters
    ----------
    coef : ndarray
        Coeficient (phi vector) for basis function.
    basis : Basis
        basis in which function is described
    sig : 2darray, optional
        Covariance matrix of phi vector
    Returns
    -------
    PhiVec : callable
        Phi vector of function.
    =#
    function PhiVec(coeff::Array{Float64}, basis::Basis)
        if Base.length(coeff) != Base.length(Basis)
            error("Phi and basis should have equal dimentions")
        end
        return new(coeff, basis, nothing)
    end

    function PhiVec(coeff::Array{Float64}, basis::Basis, sig::Array{Float64})
        if Base.length(coeff) != Base.length(Basis)
            error("Phi and basis should have equal dimentions")
        end
        if Base.length(size(sig)) != 2
            error("Sigma matrix should be 2-dimentional")
        end
        n, m = size(sig)
        if n != m
            error("Sigma matrix should be square")
        end
        if n != Base.length(coeff)
            error("If Phi in N-dimentional vector, sigma should be matrix NxN")
        end
        return new(coeff, basis, sig)
    end
end


function call(phi::PhiVec, x::Float64)
    res = 0.
    for i = 1:(length(phi.basis))
        res += phi.coeff[i] * phi.basis[i]
    end
    return res
end


function error(phi::PhiVec, x::Array{Float64, 1})
    if phi.sig == nothing
        error("Unable to calculate error without sigma matrix")
    end
    #=Calculate error at given point(s)=#
    bfValue = [f.f(x) for f in phi.basis.basisFun]
    res = zeros(Float64, Base.length(x))
    for (index, value) in enumerate(bfValue)
        res[index] = ((transpose(val)*phi.sig)*val)^0.5
    end
    return res
end
