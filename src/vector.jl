include("basis.jl")


"""
Constructs solution function by coefficients, basis and errors.

**Constructor**

```julia
PhiVec(coeff::Array{Float64}, basis::Basis, errors::Array{Float64})
PhiVec(coeff::Array{Float64}, basis::Basis)
PhiVec(result::Dict{String, Array{Float64}}, basis::Basis)
```

**Fields**

* `coeff::Array{Float64}` -- coefficients of decomposition of a function in basis
* `basis::Basis` -- basis
* `errors::Union{Array{Float64}, Nothing}` -- coefficients of decomposition of a function errors in basis
"""
struct PhiVec
    coeff::Array{Float64}
    basis::Basis
    errors::Union{Array{Float64}, Nothing}

    function PhiVec(coeff::Array{Float64}, basis::Basis)
        if Base.length(coeff) != Base.length(basis)
            Base.error("Phi and basis should have equal dimentions")
        end
        return new(coeff, basis, nothing)
    end

    function PhiVec(coeff::Array{Float64}, basis::Basis, errors::Array{Float64})
        if Base.length(coeff) != Base.length(basis)
            Base.error("Phi and basis should have equal dimentions")
        end
        if Base.length(size(errors)) != 2
            Base.error("Sigma matrix should be 2-dimentional")
        end
        n, m = size(errors)
        if n != m
            Base.error("Sigma matrix should be square")
        end
        if n != Base.length(coeff)
            Base.error(
                "If Phi is N-dimentional vector, sigma should be matrix NxN")
        end
        return new(coeff, basis, errors)
    end

    function PhiVec(result::Dict{String, Array{Float64}}, basis::Basis)
        if !haskey(result, "coeff")
            Base.error("No 'coeff' in dictionary")
        end
        if !haskey(result, "errors")
            return PhiVec(get(result, "coeff"), basis, nothing)
        end
        return PhiVec(get(result, "coeff"), basis, get(result, "errors"))
    end
end


Base.length(phivec::PhiVec) = Base.length(phivec.basis)


"""
```julia
call(phivec::PhiVec, x::Float64)
```
**Arguments**

* `phivec::PhiVec` -- unfolded function
* `x::Float64` -- point to calculate the value of the function

**Returns:** solution function value in given point.
"""
function call(phivec::PhiVec, x::Float64)
    res = sum(z -> z[1] * z[2].f(x),
        zip(phivec.coeff, phivec.basis.basis_functions))
    return res
end


"""
```julia
call(phivec::PhiVec, xs::Array{Float64, 1})
```
**Arguments**

* `phivec::PhiVec` -- unfolded function
* `xs::Array{Float64, 1}` -- points to calculate the value of the function

**Returns:** solution function value in given points.
"""
function call(phivec::PhiVec, xs::Array{Float64, 1})
    res = collect(map(x -> call(phivec, x), xs))
    return res
end


"""
```julia
errors(phi::PhiVec, x::Float64)
```

**Arguments**

* `phi::PhiVec` -- unfolded function
* `x::Float64` -- point to calculate the error of the function

**Returns:** solution function error in given point `x`.
"""
function errors(phi::PhiVec, x::Float64)
    if phi.errors == nothing
        Base.error("Unable to calculate errors without sigma matrix")
    end
    bfValue = [func.f(x) for func in phi.basis.basis_functions]
    return (transpose(bfValue) * phi.errors * bfValue)^0.5
end


"""
```julia
errors(phi::PhiVec, xs::Array{Float64})
```

**Arguments**

* `phi::PhiVec` -- unfolded function
* `xs::Array{Float64}` -- points to calculate the error of the function

**Returns:** solution function error in given point `x`.
"""
function errors(phi::PhiVec, xs::Array{Float64})
    return collect(map(x -> errors(phi, x), xs))
end
