function make_bounds(alphas::AlphasType, omegas::Union{Array{Array{T, 2}, 1}, Nothing} where T<:Real)
    if typeof(alphas) != User
        if alphas.lower == nothing
            alphas.lower = [config.ALPHAS_LOWER for _ in 1:length(omegas)]
        end
        if alphas.higher == nothing
            alphas.higher = [config.ALPHAS_HIGHER for _ in 1:length(omegas)]
        end
        if alphas.initial == nothing
            alphas.initial = [config.ALPHAS_INITIAL for _ in 1:length(omegas)]
        end
        str = "lower, higher, initial alphas and omegas must have same lengths"
        @assert length(alphas.lower) == length(alphas.initial) str
        @assert length(alphas.lower) == length(alphas.higher) str
        @assert length(alphas.lower) == length(omegas) str
    end
    return alphas
end

function make_bounds(alphas::User, omegas::Union{Array{Array{T, 2}, 1}, Nothing} where T<:Real)
    str = "alphas and omegas must have same lengths"
    @assert length(alphas.alphas) == length(omegas) str
    return alphas
end
function make_bounds(alphas::Marginalize, omegas::Union{Array{Array{T, 2}, 1}, Nothing} where T<:Real)
    if alphas.lower == nothing
        alphas.lower = [config.ALPHAS_LOWER for _ in 1:length(omegas)]
    end
    if alphas.higher == nothing
        alphas.higher = [config.ALPHAS_HIGHER for _ in 1:length(omegas)]
    end
    str = "lower and higher alphas and omegas must have same lengths"
    @assert length(alphas.lower) == length(alphas.higher) str
    @assert length(alphas.lower) == length(omegas) str
    return alphas
end

function make_bounds(phi_bounds::Union{PhiBounds, Nothing}, basis::Basis)
    if phi_bounds == nothing
        phi_bounds = PhiBounds()
    end
    if phi_bounds.lower == nothing
        phi_bounds.lower = [config.PHI_LOWER for _ in 1:length(basis.basis_functions)]
    end
    if phi_bounds.higher == nothing
        phi_bounds.higher = [config.PHI_HIGHER for _ in 1:length(basis.basis_functions)]
    end
    if phi_bounds.initial == nothing
        phi_bounds.initial = [config.PHI_INITIAL for _ in 1:length(basis.basis_functions)]
    end
    if phi_bounds.measure == nothing
        phi_bounds.measure = [config.PHI_MEASURE for _ in 1:length(basis.basis_functions)]
    end
    str = "lower, higher, initial, omegas must have same lengths"
    @assert length(phi_bounds.lower) == length(phi_bounds.initial) str
    @assert length(phi_bounds.lower) == length(phi_bounds.higher) str
    @assert length(phi_bounds.lower) == length(phi_bounds.measure) str
    return phi_bounds
end

function log_bounds_correction(phi::AbstractVector{<:Real}, phi_bounds::PhiBounds)
    bounds_correction = 0
    for i in 1:length(phi)
        if phi[i] > phi_bounds.higher[i]
            bounds_correction -= (phi[i] - phi_bounds.higher[i])/phi_bounds.measure[i]
        end
        if phi[i] < phi_bounds.lower[i]
            bounds_correction -= (phi_bounds.lower[i] - phi[i])/phi_bounds.measure[i]
        end
    end
    return bounds_correction
end
