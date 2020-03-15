function check_data(data, data_errors, kernel, basis, measurement_points)
    if !(typeof(data)<:AbstractVector{<:Real})
        @assert measurement_points!=nothing "For data as function measurement_points are required"
        data = data.(measurement_points)
    end
    if !(typeof(data_errors)<:AbstractVecOrMat{<:Real})
        @assert measurement_points!=nothing "For data_errors as function measurement_points are required"
        data_errors = data_errors.(measurement_points)
    end
    if length(size(data_errors)) == 1
        data_errors = cat(data_errors...; dims=(1,2))
    end
    if !(typeof(kernel)<:AbstractMatrix{<:Real})
        @assert isfinite(kernel(1, 1)) "Kernel should be function of 2 variables"
        @assert measurement_points!=nothing "For kernel as function measurement_points are required"
        kernel = discretize_kernel(basis, kernel, measurement_points)
    end
    m, n = size(kernel)
    @assert length(basis) == n "Kernel and basis must be (m,n) and (n,) dimensional"
    @assert length(data) == m "Kernel and data must be (m,n) and (m,) dimensional"
    @assert length(size(data_errors)) == 2 "data_errors must be two-dimensional"
    @assert size(data_errors)[1] == size(data_errors)[2] "data_errors must be square matrix"
    @assert length(data) == size(data_errors)[1] "data and data_errors must have equal dimensions."
    return data, data_errors, kernel
end

function check_alphas_omegas(alphas, omegas, basis)
    if omegas == nothing
        omegas = [omega(basis, 2)]
    end

    alphas = make_bounds(alphas, omegas)
    @assert length(omegas) != 0 "Regularization matrix Omega is absent"
    for Omega in omegas
        m, n = size(Omega)
        @assert m == n "omegas must be square matrices"
        @assert length(basis) == n "omega and basis must be (n,n) and (n,) dimensional"
    end
    if typeof(alphas) == User
        @assert length(alphas.alphas) == length(omegas) "alphas and omegas must have equal lengths"
    end
    return alphas, omegas
end

function check_phi_bounds(phi_bounds, basis)
    phi_bounds = make_bounds(phi_bounds, basis)
    @assert all(x -> x > 0, phi_bounds.higher-phi_bounds.lower) "Higher bound should be greater than lower one"
    @assert all(x -> x > 0, phi_bounds.higher-phi_bounds.initial) "Higher bound should be greater than initial"
    @assert all(x -> x > 0, phi_bounds.initial-phi_bounds.lower) "Initial should be greater than lower bound"
    return phi_bounds
end
