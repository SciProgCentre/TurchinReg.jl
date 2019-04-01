include("basis.jl")

struct PhiVec
    coeff::Array{Float64}
    basis::Basis
    sig::Union{Array{Float64}, Nothing}

    function PhiVec(coeff::Array{Float64}, basis::Basis)
        if Base.length(coeff) != Base.length(basis)
            Base.error("Phi and basis should have equal dimentions")
        end
        return new(coeff, basis, nothing)
    end

    function PhiVec(coeff::Array{Float64}, basis::Basis, sig::Array{Float64})
        if Base.length(coeff) != Base.length(basis)
            Base.error("Phi and basis should have equal dimentions")
        end
        if Base.length(size(sig)) != 2
            Base.error("Sigma matrix should be 2-dimentional")
        end
        n, m = size(sig)
        if n != m
            Base.error("Sigma matrix should be square")
        end
        if n != Base.length(coeff)
            Base.error(
                "If Phi is N-dimentional vector, sigma should be matrix NxN")
        end
        return new(coeff, basis, sig)
    end
end


Base.length(phivec::PhiVec) = Base.length(phivec.basis)


function call(phivec::PhiVec, x::Float64)
    res = sum(z -> z[1] * z[2].f(x),
        zip(phivec.coeff, phivec.basis.basis_functions))
    return res
end


function call(phivec::PhiVec, xs::Array{Float64, 1})
    res = collect(map(x -> call(phivec, x), xs))
    return res
end


function errors(phi::PhiVec, x::Float64)
    if phi.sig == nothing
        Base.error("Unable to calculate errors without sigma matrix")
    end
    bfValue = [func.f(x) for func in phi.basis.basis_functions]
    return (transpose(bfValue) * phi.sig * bfValue)^0.5
end


function errors(phi::PhiVec, xs::Array{Float64})
    return collect(map(x -> errors(phi, x), xs))
end
