#=
gauss_error:
- Julia version: 1.1.0
- Author: ta_nyan
- Date: 2019-03-17
=#
include("basis.jl")
include("vector.jl")

using Optim

mutable struct GaussErrorMatrixUnfolder
    omegas::Array{Array{Float64, 2} ,1}
    n::Int64
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
    function GaussErrorMatrixUnfolder(omegas::Array{Array{Float64, 2} ,1}, method::String="EmpiricalBayes", alphas::Union{Array{Float64, 1}, Nothing}=nothing)

        if Base.length(omegas) == 0
            Base.error("Regularization matrix Omega is absent")
        end

        if Base.length(size(omegas[1])) != 2
            Base.error("Matrix Omega must have two dimensions")
        end

        n, m = size(omegas[1])
        if n != m
            Base.error("Matrix Omega must be square")
        end

        for omega in omegas
            if length(size(omega)) != 2
                Base.error("Matrix Omega must have two dimensions")
            end
            n1, m1 = size(omega)
            if m1 != m
                Base.error("All omega matrixes must have equal dimensions")
            end
            if m1 != n1
                Base.error("Matrix Omega must be square")
            end
        end

        if method == "User"
            if alphas == nothing
                Base.error("alphas must be defined for method='User'")
            end
            if Base.length(alphas) != Base.length(omegas)
                Base.error("Omegas and alphas must have equal size")
            end
        end

        return new(omegas, m, method, alphas)
    end
end


function solve_matrix(unfolder::GaussErrorMatrixUnfolder, kernel::Array{Float64, 2}, data::Array{Float64, 1}, data_errors::Union{Array{Float64, 1}, Array{Float64, 2}})
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
    println("starting solve_matrix")
    m, n = size(kernel)
    if n != unfolder.n
        Base.error("Kernel and unfolder must have equal dimentions.")# Got " + String{(m, n)} + "and " + String{n})
    end

    if size(data)[1] != m
        Base.error("K and f must be (m,n) and (m,) dimensional.")# Got " + String{(m, n)} + "and " + String{size(data)})
    end

    if length(size(data_errors)) == 1
        data_errors = cat(data_errors...; dims=(1,2))
    elseif length(size(data_errors)) != 2
        Base.error("Sigma matrix must be two-dimensional.")# Got " + String{length(size(data_errors))})
    end

    if size(data_errors)[1] != size(data_errors)[2]
        Base.error("Sigma matrix must be square.")# Got " + String{size(data_errors)})
    end

    if size(data)[1] != size(data_errors)[1]
        Base.error("Sigma matrix and f must have equal dimensions.")# Got " + String{size(data)} + "end " + String{size(data_errors)[1]})
    end
    println("ending solve_matrix")
    return solve_correct(unfolder, kernel, data, data_errors)
end

function solve_correct(unfolder::GaussErrorMatrixUnfolder, kernel::Array{Float64, 2}, data::Array{Float64, 1}, data_errors::Array{Float64, 2})
    println("starting solve_correct")
    K = kernel
    Kt = transpose(kernel)
    dataErrorInv = inv(data_errors)
    B = Kt * dataErrorInv * K
    b = Kt * dataErrorInv * data

    function optimal_alpha()
        println("starting optimal_alpha")
        function alpha_prob(a::Array{Float64, 1})
            aO = transpose(a)*unfolder.omegas
            BaO = B + aO
            iBaO = inv(BaO)
            dotp = transpose(b) * iBaO * b
            if det(aO) != 0
                detaO = log(abs(det(aO)))
            else
                eigvals_aO = sort(eigvals(aO))
                rank_deficiency = size(aO)[1] - rank(aO)
                detaO = sum(log.(eigvals_aO[(rank_deficiency+1):end]))
            end
            detBaO = log(abs(det(BaO)))
            return (detaO - detBaO) / 2.0 + dotp / 2.0
        end

        a0 = zeros(Float64, Base.length(unfolder.omegas))
        println("starting optimize")
#         res = optimize(a -> -alpha_prob(exp.(a)), a0,  BFGS())
#         if !Optim.converged(res)
#             Base.error("Minimization did not succeed")
#         end
        return [0.5] #exp.(Optim.minimizer(res))
    end

    if unfolder.method == "EmpiricalBayes"
        unfolder.alphas = optimal_alpha()
    end

    BaO = B + transpose(unfolder.alphas)*unfolder.omegas
    iBaO = inv(BaO)
    r = iBaO * b
    println("ending solve_correct")
    return Dict("phi" => r, "covariance" => iBaO, "alphas" => unfolder.alphas)
end

mutable struct GaussErrorUnfolder
    basis::Basis
    solver::GaussErrorMatrixUnfolder
    #=Implementation of statreg algorithm for case of Gauss errors using empirical Bayes.

    Solve  the Fredholm integral equation:
    .. math:: f(y) = \int K(y,x) \phi(x) dx
    using Turchin's method of statistical regularization for ill-possed problem for case of Gauss errors in measurable function :math:`f(y)`. Use empirical Bayes for computation of regularization parameters.

    Parameters
    ----------
    basis: instance of Basis class
        Basis in functional space. Reconstructed function will be
        represented as sum of elements in the basis
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
    solve(kernel, data, dataError, y = None)
        Solve given Fredholm integral equation

    =#

    function GaussErrorUnfolder(basis::Basis, omegas::Array, method::String="EmpiricalBayes", alphas::Union{Array{Float64, 1}, Nothing}=nothing)
        solver = GaussErrorMatrixUnfolder(omegas, method, alphas)
        return new(basis, solver)
    end
end

function solve(gausserrorunfolder::GaussErrorUnfolder, kernel::Union{Function, Array{Float64, 2}}, data::Union{Function, Array{Float64, 1}}, dataError::Union{Function, Array{Float64, 1}}, y::Union{Array{Float64, 1}, Nothing})
     #=Solve given Fredholm integral equation.

        Note
        ----
        Solve  the Fredholm integral equation:
    .. math:: f(y) = \int K(y,x) \phi(x) dx
        or its vector representation:
    .. math:: f_m = K_{mn} \phi_n

        Parameters
        ----------
        kernel: Callable or ndarray
            Kernel function :math:`K(x,y` or kernel matrix :math:`K_{mn}`
        data: Callable or ndarray
            Measurable function :math:`f(y)` or vector of measured value :math:`f_m`
        dataError: Callable or ndarray
            Error (parameter :math:`\sigma^2` of Gauss distribution) of measured value, can be function, vector or covariance matrix
        y: ndarray or None
        Points where measurements were made.  Necessarily if method get Callable argument
        Returns
        -------
        FunctionalUnfoldingResult
            Result of unfolding, Callable - can be compute :math:`\phi(x)`, method ``error(x)`` give error of result.
            Another field contains information about regularization like as ``OptimizeResult``.  Important attributes are: ``phi`` - vector in functional space, ``covariance``- covariance matrix  of ``phi``-vector, ``success`` a Boolean flag  indicating if the unfolder exited successfully, ``alphas`` the list of regularization parameters for empirical Bayes.

        =#
    println("starting solve")
    function check_y()
        if y == nothing
            Base.error("For callable arguments `y` must be defined")
        end
    end

    if !(typeof(kernel) == Array{Float64, 2})
        check_y()
        kernel_array = discretizeKernel(gausserrorunfolder.basis, kernel, y)
    else
        kernel_array = kernel
    end

    if !(typeof(data) == Array{Float64, 1})
        check_y()
        data_array = data.(y)
    else
        data_array = data
    end

    if !(typeof(dataError) == Array{Float64, 1})
        check_y()
        dataError_array = dataError.(y)
    else
        dataError_array = dataError
    end
    println("ending solve")
    result = solve_matrix(gausserrorunfolder.solver, kernel_array, data_array, dataError_array)
    return result
end