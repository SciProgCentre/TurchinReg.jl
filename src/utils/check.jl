function check_args(
    omegas::Array{Array{T, 2}, 1} where T<:Real,
    method::String,
    alphas::Union{AbstractVector{<:Real}, Nothing},
    low::Union{AbstractVector{<:Real}, Nothing},
    high::Union{AbstractVector{<:Real}, Nothing},
    alpha0::Union{AbstractVector{<:Real}, Nothing}
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
    if method == "EmpiricalBayes"
        if low == nothing
            @error "Lower limits of alphas should be provided for method=EmpiricalBayes"
            Base.error("Lower limits of alphas should be provided for method=EmpiricalBayes")
        end
        if high == nothing
            @error "Higher limits of alphas should be provided for method=EmpiricalBayes"
            Base.error("Higher limits of alphas should be provided for method=EmpiricalBayes")
        end
        if alpha0 == nothing
            @error "Initial values of alphas should be provided for method=EmpiricalBayes"
            Base.error("Initial values of alphas should be provided for method=EmpiricalBayes")
        end
        if (length(low) != length(high)) || (length(low) != length(alpha0)) || (length(alpha0) != length(omegas))
            @error "Low, high and alpha0 should have equal lengths"
            Base.error("Low, high and alpha0 should have equal lengths")
        end
        if any(i -> (i <= 0), alpha0)
            @error "All elements in alpha0 should be positive"
            Base.error("All elements in alpha0 should be positive")
        end
    end
    return m
end


function check_args(
    unfolder,
    kernel::AbstractMatrix{<:Real},
    data::AbstractVector{<:Real},
    data_errors::AbstractVecOrMat{<:Real},
    )
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
    return data_errors
end


function check_args(
    unfolder,
    kernel::Union{Function, AbstractMatrix{<:Real}},
    data::Union{Function, AbstractVector{<:Real}},
    data_errors::Union{Function, AbstractVector{<:Real}},
    y::Union{AbstractVector{<:Real}, Nothing},
    )

    function check_y()
        if y == nothing
            @error "For callable arguments `y` must be defined"
            Base.error("For callable arguments `y` must be defined")
        end
    end

    if !(typeof(kernel)<:AbstractMatrix{<:Real})
        check_y()
        kernel_array = discretize_kernel(unfolder.basis, kernel, y)
    else
        kernel_array = kernel
    end

    if !(typeof(data)<:AbstractVector{<:Real})
        check_y()
        data_array = data.(y)
    else
        data_array = data
    end

    if !(typeof(data_errors)<:AbstractVector{<:Real})
        check_y()
        data_errors_array = data_errors.(y)
    else
        data_errors_array = data_errors
    end
    return kernel_array, data_array, data_errors_array
end
