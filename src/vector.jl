include("basis.jl")
include("config.jl")


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
* `phi_function(x::Float64)`::Function -- returns constructed function's value at given point
* `error_function(x::Float64)`::Union{Function, Nothing} -- returns constructed function's error at given point,
 if errors are specified, otherwise is `nothing`
"""
struct PhiVec
    coeff::Array{Float64, 1}
    basis::Basis
    errors::Union{Array{Float64, 2}, Nothing}
    phi_function::Function
    error_function::Union{Function, Nothing}

    function phi_function_(coeff::Array{Float64, 1}, basis::Basis)
        return x::Float64 -> sum(z -> z[1] * z[2].f(x),
        zip(coeff, basis.basis_functions))
    end

    function error_function_(errors::Array{Float64, 2}, basis::Basis)
        function errors_(x::Float64)
            bfValue = [func.f(x) for func in basis.basis_functions]
            return (abs.(transpose(bfValue) * errors * bfValue))^0.5
        end
        return errors_
    end

    function PhiVec(coeff::Array{Float64, 1}, basis::Basis)
        if Base.length(coeff) != Base.length(basis)
            @error "Phi and basis should have equal dimentions"
            Base.error("Phi and basis should have equal dimentions")
        end
        @info "PhiVec is created."
        return new(coeff, basis, nothing, phi_function_(coeff, basis), nothing)
    end

    function PhiVec(coeff::Array{Float64, 1}, basis::Basis, errors::Array{Float64, 2})
        if Base.length(coeff) != Base.length(basis)
            @error "Phi and basis should have equal dimentions"
            Base.error("Phi and basis should have equal dimentions")
        end
        n, m = size(errors)
        if n != m
            @error "Sigma matrix should be square"
            Base.error("Sigma matrix should be square")
        end
        if n != Base.length(coeff)
            @error "If Phi is N-dimentional vector, sigma should be matrix NxN"
            Base.error(
                "If Phi is N-dimentional vector, sigma should be matrix NxN")
        end
        @info "PhiVec is created."
        return new(coeff, basis, errors, phi_function_(coeff, basis), error_function_(errors, basis))
    end

    function PhiVec(result::Dict, basis::Basis)
        if !haskey(result, "coeff")
            @error "No 'coeff' in dictionary"
            Base.error("No 'coeff' in dictionary")
        end
        if !haskey(result, "errors")
            return PhiVec(result["coeff"], basis)
        end
        return PhiVec(result["coeff"], basis, result["errors"])
    end

end
