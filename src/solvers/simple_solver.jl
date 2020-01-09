function simple_solver(basis::Basis, kernel::Function,
    f::AbstractVector{<:Real}, sig::AbstractVector{<:Real}, y::AbstractVector{<:Real}
    )
        Omega = omega(basis, 2)
        model = GaussErrorUnfolder(
            basis, [Omega], "EmpiricalBayes";
            alphas=nothing, lower=[1e-8], higher=[10.], initial=[0.3]
            )
        result = solve(model, kernel, f, sig, y)
    return PhiVec(result, basis)
end
