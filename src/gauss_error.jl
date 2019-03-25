#=
gauss_error:
- Julia version: 1.1.0
- Author: ta_nyan
- Date: 2019-03-17
=#

struct GaussErrorMatrixUnfolder
    omegas::Array{Float64, 1}
    method::String
    alphas::Union{Array{Float64}, Nothing}
    #=Implementation of statreg algorithm for case of Gauss errors using empirical Bayes.

    Solve  the next matrix equation:
    .. math:: f_m = K_{mn} \phi_n
    using Turchin's method of statistical regularization for ill-possed problem for case of Gauss errors in measurable
    vector :math:`f_m`. Use empirical Bayes for computation of regularization parameters.

    Parameters
    ----------
    *omegas: sequence of matrices
        list of regularizing matrices. Normally they're derived from
        basis parameter.
    method : str, optional
        Type of method for choise regularization parameter. Should be one of:

        - 'User'
        - 'EmpiricalBayes'

    alphas : ndarray or float
        Only for `method='User'` - solver will use users value of regularization parameter

    Methods
    -------
    solve(kernel, data, dataError)
        Solve given matrix equation

    =#
    function GaussErrorMatrixUnfolder(omegas::Array{Float64, 1}, method::String="EmpiricalBayes", alphas::Union{Array{Float64}, Nothing}=nothing)

        if Base.length(omegas) == 0:
            error("Regularization matrix Omega is absent")
        end

        if Base.length(size(omegas[1])) != 2
            error("Matrix Omega must have two dimensions")
        end

        n, m = size(omegas[1])
        if n != m
            error("Matrix Omega must be square")
        end

        for omega in omegas
            if length(size(omega)) != 2
                error("Matrix Omega must have two dimensions")
            end
            n1, m1 = size(omega)
            if m1 != m
                error("All omega matrixes must have equal dimensions")
            end
            if m1 != n1!
                error("Matrix Omega must be square")
            end
        end

        if method == "User"
            if alphas == nothing
                error("alphas must be defined for method='User'")
            end
            if Base.length(alphas) != Base.length(omegas)
                error("Omegas and alphas must have equal size")
            end
        end

        return new(omegas, method, alphas)
    end
end


function solve(unfolder::GaussErrorMatrixUnfolder, kernel::Array{Float64, 2}, data::Array{Float64, 1}, data_errors::Union{Array{Float64, 1}, Array{Float64, 2}})
            #=Solve given matrix equation.

         Note
         ----
         Solve  the vector  equation:
     .. math:: f_m = K_{mn} \phi_n

         Parameters
         ----------
         kernel: ndarray
             Kernel matrix :math:`K_{mn}`
         data: ndarray
             Vector of measured value :math:`f_m`
         dataError: ndarray
             Error (parameter :math:`\sigma^2` of Gauss distribution) of measured value, can be vector or covariance matrix
         Returns
         -------
         UnfoldingResult
             Result of unfolding, like as ``OptimizeResult``.  Important attributes are: ``phi`` - solution, ``covariance``- covariance matrix of solition,``success`` a Boolean flag  indicating if the unfolder exited successfully, ``alphas`` the list of regularization parameters for empirical Bayes.

         =#

    if length(size(kernel)) != 2
        error("Kernel matrix must be two-dimensional")
    end
    n, m = size(kernel)
    if n != unfolder.n
        error("Kernel and ")
    end
    if length(size(data)) != 1
        error("f vector must be one-dimensional")
    end
    if size(data) != m
        error("K and f must be (m,n) and (m,) dimensional")
    end
    if length(size(data_errors)) == 1
        data_errors_matrix = 1
    elseif length(size(data_errors)) != 2
        error("Sigma matrix must be two-dimensional")
    end
    if size(data_errors)[1] != size(data_errors)[2]
        error("Sigma matrix must be square")
    end
    if size(data) != size(data_errors)[1]
        error("Sigma matrix and f must have equal dimensions")
    end
    return solve_correct(unfolder, kernel, data, data_errors)
end


function regularization_matrix(unfolder::GaussErrorMatrixUnfolder, alphas::Array{Float64})
    return sum()
end


function solve_correct(unfolder::GaussErrorMatrixUnfolder, kernel::Array{Float64}, data::Array{Float64}, data_errors::Array{Float64})

end