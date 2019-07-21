include("basis.jl")
include("vector.jl")
include("config.jl")

using Optim

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

**Fields**

* `omegas::Array{Array{Float64, 2} ,1}`
* `n::Int64` -- size of square omega matrix
* `method::String`
* `alphas::Union{Array{Float64}, Nothing}`
"""
mutable struct GaussErrorMatrixUnfolder
    omegas::Array{Array{Float64, 2} ,1}
    n::Int64
    method::String
    alphas::Union{Array{Float64}, Nothing}

    function GaussErrorMatrixUnfolder(
        omegas::Array{Array{Float64, 2} ,1},
        method::String="EmpiricalBayes",
        alphas::Union{Array{Float64, 1}, Nothing}=nothing,
        )

        if Base.length(omegas) == 0
            @error "Regularization matrix Omega is absent"
            Base.error("Regularization matrix Omega is absent")
        end

        if Base.length(size(omegas[1])) != 2
            @error "Matrix Omega must have two dimensions"
            Base.error("Matrix Omega must have two dimensions")
        end

        n, m = size(omegas[1])
        if n != m
            @error "Matrix Omega must be square"
            Base.error("Matrix Omega must be square")
        end

        for omega in omegas
            if length(size(omega)) != 2
                @error "Matrix Omega must have two dimensions"
                Base.error("Matrix Omega must have two dimensions")
            end
            n1, m1 = size(omega)
            if m1 != m
                @error "All omega matrices must have equal dimensions"
                Base.error("All omega matrices must have equal dimensions")
            end
            if m1 != n1
                @error "Omega must be square"
                Base.error("Omega must be square")
            end
        end

        if method == "User"
            if alphas == nothing
                @error "Alphas must be defined for method='User'"
                Base.error("Alphas must be defined for method='User'")
            end
            if Base.length(alphas) != Base.length(omegas)
                @error "Omegas and alphas must have equal lengths"
                Base.error("Omegas and alphas must have equal lengths")
            end
        end
        @info "GaussErrorMatrixUnfolder is created"
        return new(omegas, m, method, alphas)
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
    data_errors::Union{Array{Float64, 1}, Array{Float64, 2}},
    )

    @info "Starting solve..."
    m, n = size(kernel)
    if n != unfolder.n
        @error "Kernel and unfolder must have equal dimentions."
        Base.error("Kernel and unfolder must have equal dimentions.")
    end

    if length(data) != m
        @error "K and f must be (m,n) and (m,) dimensional."
        Base.error("K and f must be (m,n) and (m,) dimensional.")
    end

    if length(size(data_errors)) == 1
        data_errors = cat(data_errors...; dims=(1,2))
    elseif length(size(data_errors)) != 2
        @error "Sigma matrix must be two-dimensional."
        Base.error("Sigma matrix must be two-dimensional.")
    end

    if size(data_errors)[1] != size(data_errors)[2]
        @error "Sigma matrix must be square."
        Base.error("Sigma matrix must be square.")
    end

    if length(data) != size(data_errors)[1]
        @error "Sigma matrix and f must have equal dimensions."
        Base.error("Sigma matrix and f must have equal dimensions.")
    end
    @info "Ending solve..."
    return solve_correct(unfolder, kernel, data, data_errors)
end

function solve_correct(
    unfolder::GaussErrorMatrixUnfolder,
    kernel::Array{Float64, 2},
    data::Array{Float64, 1},
    data_errors::Array{Float64, 2},
    )

    @info "Starting solve_correct..."
    K = kernel
    Kt = transpose(kernel)
    data_errorsInv = pinv(data_errors)
    B = Kt * data_errorsInv * K
    b = Kt * transpose(data_errorsInv) * data

    function find_optimal_alpha()
        @info "Starting find_optimal_alpha..."

        function alpha_prob(a::Array{Float64, 1})
            aO = transpose(a)*unfolder.omegas
            BaO = B + aO
            if det(BaO) == 0
                @warn "det(BaO) = 0" maxlog=1
            end
            iBaO = pinv(BaO)
            dotp = transpose(b) * iBaO * b
            if det(aO) != 0
                detaO = log(abs(det(aO)))
            else
                eigvals_aO = sort(eigvals(aO))
                rank_deficiency = size(aO)[1] - rank(aO)
                detaO = sum(log.(abs.(eigvals_aO[(rank_deficiency+1):end])))
            end
            detBaO = log(abs(det(BaO)))
            return detaO - detBaO + dotp
        end

        a0 = zeros(Float64, Base.length(unfolder.omegas))
        @info "Starting optimization..."

        res = optimize(
            a -> -alpha_prob(exp.(a)), a0,  BFGS(),
            Optim.Options(x_tol=X_TOL_OPTIM, show_trace=true,
            store_trace=true, allow_f_increases=true))

        if !Optim.converged(res)
            @warn "Minimization did not succeed, return alpha = 0.05."
            return [0.05]
        end
        alpha = exp.(Optim.minimizer(res))
        if alpha[1] < 1e-6 || alpha[1] > 1e3
            @warn "Incorrect alpha: too small or too big, return alpha = 0.05."
            return [0.05]
        end
        @info "Optimized successfully."
        return alpha
    end

    if unfolder.method == "EmpiricalBayes"
        unfolder.alphas = find_optimal_alpha()
    elseif unfolder.method != "User"
        @error "Unknown method" + unfolder.method
        Base.eror("Unknown method" + unfolder.method)
    end


    BaO = B + transpose(unfolder.alphas)*unfolder.omegas
    iBaO = pinv(BaO)
    r = iBaO * b
    @info "Ending solve_correct..."
    return Dict("coeff" => r, "errors" => iBaO, "alphas" => unfolder.alphas)
end


"""
Model for continuous kernel. Data can be either discrete or continuous.

**Constructor**

```julia
GaussErrorUnfolder(
    basis::Basis,
    omegas::Array,
    method::String="EmpiricalBayes",
    alphas::Union{Array{Float64, 1}, Nothing}=nothing,
    )
```

`basis` -- basis for reconstruction

`omegas` -- array of matrices that provide information about basis functions

`method` -- constant selection method, possible options: "EmpiricalBayes" and "User"

`alphas` -- array of constants, in case method="User" should be provided by user


**Fields**
* `basis::Basis`
* `solver::GaussErrorMatrixUnfolder`
"""
mutable struct GaussErrorUnfolder
    basis::Basis
    solver::GaussErrorMatrixUnfolder

    function GaussErrorUnfolder(
        basis::Basis,
        omegas::Array,
        method::String="EmpiricalBayes",
        alphas::Union{Array{Float64, 1}, Nothing}=nothing,
        )

        solver = GaussErrorMatrixUnfolder(omegas, method, alphas)
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
    y::Union{Array{Float64, 1}, Nothing},
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
    y::Union{Array{Float64, 1}, Nothing},
    )

    @info "Starting solve..."
    function check_y()
        if y == nothing
            @error "For callable arguments `y` must be defined"
            Base.error("For callable arguments `y` must be defined")
        end
    end

    if !(typeof(kernel) == Array{Float64, 2})
        check_y()
        kernel_array = discretize_kernel(gausserrorunfolder.basis, kernel, y)
    else
        kernel_array = kernel
    end

    if !(typeof(data) == Array{Float64, 1})
        check_y()
        data_array = data.(y)
    else
        data_array = data
    end

    if !(typeof(data_errors) == Array{Float64, 1})
        check_y()
        data_errors_array = data_errors.(y)
    else
        data_errors_array = data_errors
    end
    @info "Ending solve..."
    result = solve(
        gausserrorunfolder.solver,
        kernel_array, data_array, data_errors_array
    )
    return result
end
