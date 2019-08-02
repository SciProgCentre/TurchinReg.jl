include("basis.jl")
include("vector.jl")
include("config.jl")
include("check.jl")

using Optim

make_sym(A::Array{Float64, 2}) = (transpose(A) + A) / 2


"""
Model for dicsrete data and kernel.

**Constructor**

```julia
GaussErrorMatrixUnfolder(
    omegas::Array{Array{Float64, 2} ,1},
    method::String="EmpiricalBayes",
    alphas::Union{Array{Float64, 1}, Nothing}=nothing,
    )
```
`omegas` -- array of matrices that provide information about basis functions

`method` -- constant selection method, possible options: "EmpiricalBayes" and "User"

`alphas` -- array of constants, in case method="User" should be provided by user

`low` -- lower limits for alphas

`high` -- higher limits for alphas

`alpha0` -- unitial values for alphas

**Fields**

* `omegas::Array{Array{Float64, 2} ,1}`
* `n::Int64` -- size of square omega matrix
* `method::String`
* `alphas::Union{Array{Float64}, Nothing}`
* `low::Union{Array{Float64, 1}, Nothing}`
* `high::Union{Array{Float64, 1}, Nothing}`
* `alpha0::Union{Array{Float64, 1}, Nothing}`
"""
mutable struct GaussErrorMatrixUnfolder
    omegas::Array{Array{Float64, 2} ,1}
    n::Int64
    method::String
    alphas::Union{Array{Float64}, Nothing}
    low::Union{Array{Float64, 1}, Nothing}
    high::Union{Array{Float64, 1}, Nothing}
    alpha0::Union{Array{Float64, 1}, Nothing}

    function GaussErrorMatrixUnfolder(
        omegas::Array{Array{Float64, 2} ,1},
        method::String="EmpiricalBayes",
        alphas::Union{Array{Float64, 1}, Nothing}=nothing,
        low::Union{Array{Float64, 1}, Nothing}=nothing,
        high::Union{Array{Float64, 1}, Nothing}=nothing,
        alpha0::Union{Array{Float64, 1}, Nothing}=nothing
        )
        m = check_args(omegas, method, alphas, low, high, alpha0)
        @info "GaussErrorMatrixUnfolder is created"
        return new(omegas, m, method, alphas, low, high, alpha0)
    end
end


"""
```julia
solve(
    unfolder::GaussErrorMatrixUnfolder,
    kernel::Array{Float64, 2},
    data::Array{Float64, 1},
    data_errors::Union{Array{Float64, 1}, Array{Float64, 2}},
    )
```

**Arguments**
* `unfolder::GaussErrorMatrixUnfolder` -- model
* `kernel::Array{Float64, 2}` -- discrete kernel
* `data::Array{Float64, 1}` -- function values
* `data_errors::Union{Array{Float64, 1}, Array{Float64, 2}}` -- function errors

**Returns:** `Dict{String, Array{Float64, 1}}` with coefficients ("coeff"), errors ("errors") and optimal constants ("alphas").
"""
function solve(
    unfolder::GaussErrorMatrixUnfolder,
    kernel::Array{Float64, 2},
    data::Array{Float64, 1},
    data_errors::Union{Array{Float64, 1}, Array{Float64, 2}}
    )
    @info "Starting solve..."
    data_errors = check_args(unfolder, kernel, data, data_errors)
    data_errorsInv = make_sym(pinv(data_errors))
    B = make_sym(transpose(kernel) * data_errorsInv * kernel)
    b = transpose(kernel) * transpose(data_errorsInv) * data
    alpha0 = unfolder.alpha0
    if unfolder.method == "EmpiricalBayes"
        unfolder.alphas = find_optimal_alpha(
            unfolder.omegas, B, b,
            unfolder.alpha0, unfolder.low, unfolder.high
            )
    elseif unfolder.method != "User"
        @error "Unknown method" + unfolder.method
        Base.eror("Unknown method" + unfolder.method)
    end
    Ba0 = B + transpose(unfolder.alphas) * unfolder.omegas
    iBa0 = make_sym(pinv(Ba0))
    r = iBa0 * b
    @info "Ending solve..."
    return Dict("coeff" => r, "errors" => iBa0, "alphas" => unfolder.alphas)
end


"""
Model for continuous kernel. Data can be either discrete or continuous.

**Constructor**

```julia
GaussErrorUnfolder(
    basis::Basis,
    omegas::Array{Array{Float64, 2}, 1},
    method::String="EmpiricalBayes",
    alphas::Union{Array{Float64, 1}, Nothing}=nothing,
    low::Union{Array{Float64, 1}, Nothing}=nothing,
    high::Union{Array{Float64, 1}, Nothing}=nothing,
    alpha0::Union{Array{Float64, 1}, Nothing}=nothing
    )
```

`basis` -- basis for reconstruction

`omegas` -- array of matrices that provide information about basis functions

`method` -- constant selection method, possible options: "EmpiricalBayes" and "User"

`alphas` -- array of constants, in case method="User" should be provided by user

`low` -- lower limits for alphas

`high` -- higher limits for alphas

`alpha0` -- unitial values for alphas

**Fields**
* `basis::Basis`
* `solver::GaussErrorMatrixUnfolder`
"""
mutable struct GaussErrorUnfolder
    basis::Basis
    solver::GaussErrorMatrixUnfolder

    function GaussErrorUnfolder(
        basis::Basis,
        omegas::Array{Array{Float64, 2}, 1},
        method::String="EmpiricalBayes",
        alphas::Union{Array{Float64, 1}, Nothing}=nothing,
        low::Union{Array{Float64, 1}, Nothing}=nothing,
        high::Union{Array{Float64, 1}, Nothing}=nothing,
        alpha0::Union{Array{Float64, 1}, Nothing}=nothing
        )
        solver = GaussErrorMatrixUnfolder(
            omegas, method, alphas, low, high, alpha0
        )
        @info "GaussErrorUnfolder is created."
        return new(basis, solver)
    end
end


"""
```julia
solve(
    gausserrorunfolder::GaussErrorUnfolder,
    kernel::Union{Function, Array{Float64, 2}},
    data::Union{Function, Array{Float64, 1}},
    data_errors::Union{Function, Array{Float64, 1}},
    y::Union{Array{Float64, 1}, Nothing}=nothing,
    )
```

**Arguments**
* `gausserrorunfolder::GaussErrorUnfolder` -- model
* `kernel::Union{Function, Array{Float64, 2}}` -- discrete or continuous kernel
* `data::Union{Function, Array{Float64, 1}}` -- function values
* `data_errors::Union{Function, Array{Float64, 1}}` -- function errors
* `y::Union{Array{Float64, 1}, Nothing}` -- points to calculate function values and its errors (when data is given as a function)

**Returns:** `Dict{String, Array{Float64, 1}}` with coefficients ("coeff"), errors ("errors") and optimal constants ("alphas").
"""
function solve(
    gausserrorunfolder::GaussErrorUnfolder,
    kernel::Union{Function, Array{Float64, 2}},
    data::Union{Function, Array{Float64, 1}},
    data_errors::Union{Function, Array{Float64, 1}},
    y::Union{Array{Float64, 1}, Nothing}=nothing,
    )
    @info "Starting solve..."
    kernel_array, data_array, data_errors_array = check_args(
        gausserrorunfolder, kernel, data, data_errors, y
        )
    result = solve(
        gausserrorunfolder.solver,
        kernel_array, data_array, data_errors_array
        )
    @info "Ending solve..."
    return result
end
