make_sym(A::AbstractMatrix{<:Real}) = (transpose(A) + A) / 2
sym_inv(A::AbstractMatrix{<:Real}) = make_sym(pinv(A))

function find_optimal_alpha(
    alphas::ArgmaxOptim,
    omegas::Array{Array{T, 2}, 1} where T<:Real,
    B::AbstractMatrix{<:Real},
    b::AbstractVector{<:Real},
    )

    function alpha_prob(a::AbstractVector{<:Real}, alphas)
        for i in eachindex(a)
            if (a[i] <= alphas.lower[i]) || (a[i] >= alphas.higher[i])
                return -1e4
            end
        end
        a0 = transpose(a) * omegas
        Ba0 = B + a0
        iBa0 = sym_inv(Ba0)
        dotp = transpose(b) * iBa0 * b

        function det_(A::Array{<:Real, 2})
            if det(A) != 0
                return logabsdet(A)[1]
            else
                eigvals_A = sort(eigvals(A))
                rank_deficiency = size(A)[1] - rank(A)
                return sum(log.(abs.(eigvals_A[(rank_deficiency+1):end])))
            end
        end

        return det_(a0) - det_(Ba0) + dotp
    end

    @assert all(map(x -> x > 0, alphas.initial)) "All aplha0 should be positive"
    a0 = log.(alphas.initial)
    res = optimize(
        a -> -alpha_prob(exp.(a), alphas), a0,  BFGS(),
        Optim.Options(x_tol=config.X_TOL_OPTIM, show_trace=false,
        store_trace=true, allow_f_increases=true))
    if !Optim.converged(res)
        @error "Minimization did not succeed, alphas = $(exp.(Optim.minimizer(res))), return alphas = [config.ALPHA_NOT_CONVERGED ...]."
        alphas.alphas = [config.ALPHAS_NOT_CONVERGED for _ in length(omegas)]
    end
    alphas.alphas = exp.(Optim.minimizer(res))
    @info "Optimized successfully, alphas = $alphas."
    return alphas
end


function find_optimal_alpha(
    alphas::ArgmaxBAT,
    omegas::Array{Array{T, 2}, 1} where T<:Real,
    B::AbstractMatrix{<:Real},
    b::AbstractVector{<:Real},
    )
    likelihood = alpha -> begin
            a0 = transpose(alpha.alpha) * omegas
            Ba0 = B + a0
            iBa0 = sym_inv(Ba0)
            iB = sym_inv(B)
            dotp = transpose(b) * iBa0 * b
            dotp2 = transpose(b) * iB * b

            return LinDVal(exp((dotp - dotp2) / 2) * sqrt(abs(det(iBa0) * det(a0))))
        end
    prior = NamedTupleDist(alpha = [alphas.lower[i]..alphas.higher[i] for i in eachindex(alphas.lower)])
    posterior = BAT.PosteriorDensity(likelihood, prior)
    samples = BAT.bat_sample(posterior, (alphas.nsamples, alphas.nchains), alphas.algo).result
    alphas.alphas = BAT.mode(samples).alpha
    return alphas

    #TODO:: find appropriate algo
    # algorithm = BAT.ModeAsDefined()
    # findmode_result = bat_findmode(
    #     posterior, algorithm,
    #     initial_mode = samples_mode
    #     )
    # alphas = findmode_result.result[].alpha
    # return alphas
end

find_optimal_alpha(
    alphas::Marginalize,
    omegas::Array{Array{T, 2}, 1} where T<:Real,
    B::AbstractMatrix{<:Real},
    b::AbstractVector{<:Real}
    ) = alphas
find_optimal_alpha(
    alphas::User,
    omegas::Array{Array{T, 2}, 1} where T<:Real,
    B::AbstractMatrix{<:Real},
    b::AbstractVector{<:Real}
    ) = alphas
