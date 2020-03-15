"""
Constructs solution function by coefficients, basis and errors.



```julia
PhiVec(coeff::Array{<:Real}, basis::Basis, errors::Array{<:Real})
```

```julia
PhiVec(coeff::Array{<:Real}, basis::Basis)
```

```julia
PhiVec(result::Dict{String, Array{<:Real}}, basis::Basis)
```

**Fields**

* `coeff::Array{<:Real}` -- coefficients of decomposition of a function in basis
* `basis::Basis` -- basis
* `errors::Union{Array{<:Real}, Nothing}` -- coefficients of decomposition of a function errors in basis
* `solution_function(x::Real)`::Function -- returns constructed function's value at given point
* `error_function(x::Real)`::Union{Function, Nothing} -- returns constructed function's error at given point, if errors are specified, otherwise is `nothing`
* `alphas::Union{AbstractVector{<:Real}, Nothing}` -- list of regularization parameters
"""
struct PhiVec
    coeff::AbstractVector{<:Real}
    basis::Basis
    errors::Union{AbstractMatrix{<:Real}, Nothing}
    solution_function::Function
    error_function::Union{Function, Nothing}
    alphas::Union{AbstractVector{<:Real}, Nothing}

    function solution_function_(coeff::AbstractVector{<:Real}, basis::Basis)
        return x::Real -> sum(z -> z[1] * z[2](x),
        zip(coeff, basis.basis_functions))
    end

    function error_function_(errors::AbstractMatrix{<:Real}, basis::Basis)
        function errors_(x::Real)
            bfValue = [func(x) for func in basis.basis_functions]
            return (abs.(transpose(bfValue) * errors * bfValue))^0.5
        end
        return errors_
    end

    function PhiVec(coeff::AbstractVector{<:Real}, basis::Basis, alphas::Union{AbstractVector{<:Real}, Nothing}=nothing)
        @assert length(coeff) == length(basis) "Phi and basis should have equal dimentions"
        return new(coeff, basis, nothing, solution_function_(coeff, basis), nothing, alphas)
    end

    function PhiVec(coeff::AbstractVector{<:Real}, basis::Basis, errors::AbstractMatrix{<:Real}, alphas::Union{AbstractVector{<:Real}, Nothing}=nothing)
        @assert length(coeff) == Base.length(basis) "Phi and basis should have equal dimentions"
        n, m = size(errors)
        @assert n == m "Covariational matrix should be square"
        @assert n == length(coeff) "If Phi is N-dimentional vector, covariational matrix should be matrix NxN"
        return new(coeff, basis, errors, solution_function_(coeff, basis), error_function_(errors, basis), alphas)
    end

    function PhiVec(coeff::AbstractVector{<:Real}, basis::Basis, errors::AbstractVector{<:Real}, alphas::Union{AbstractVector{<:Real}, Nothing}=nothing)
        return PhiVec(coeff, basis, cat(errors...; dims=(1, 2)), alphas)
    end

    function PhiVec(result::Dict, basis::Basis, alphas::Union{AbstractVector{<:Real}, Nothing}=nothing)
        @assert haskey(result, "coeff") "No 'coeff' in dictionary"
        if !haskey(result, "errors")
            return PhiVec(result["coeff"], basis)
        end
        return PhiVec(result["coeff"], basis, result["errors"], alphas)
    end

end
