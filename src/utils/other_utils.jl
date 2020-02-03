using Plots
plotly()
gr(size=(500,500), html_output_format=:png)

make_sym(A::AbstractMatrix{<:Real}) = (transpose(A) + A) / 2
sym_inv(A::AbstractMatrix{<:Real}) = make_sym(pinv(A))

function find_optimal_alpha(
    omegas::Array{Array{T, 2}, 1} where T<:Real,
    B::AbstractMatrix{<:Real},
    b::AbstractVector{<:Real},
    alpha0::AbstractVector{<:Real},
    low::AbstractVector{<:Real},
    high::AbstractVector{<:Real}
    )

    @info "Starting find_optimal_alpha..."
    function alpha_prob(a::AbstractVector{<:Real})
        for i in eachindex(a)
            if (a[i] <= low[i]) || (a[i] >= high[i])
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

    @info "Starting optimization..."
    if any(map(x -> x <= 0, alpha0))
        @error "All aplha0 should be positive"
        Base.error("All aplha0 should be positive")
    end
    a0 = log.(alpha0)
    res = optimize(
        a -> -alpha_prob(exp.(a)), a0,  BFGS(),
        Optim.Options(x_tol=config.X_TOL_OPTIM, show_trace=false,
        store_trace=true, allow_f_increases=true))
    if !Optim.converged(res)
        @error "Minimization did not succeed, alpha = $(exp.(Optim.minimizer(res))), return alpha = 0.05."
        return [0.05]
    end
    alphas = exp.(Optim.minimizer(res))
    plot(a -> alpha_prob([a]))
    display()
    savefig("plot_prob.png")
    @info "Optimized successfully, alphas = $alphas."
    return alphas
end


function find_optimal_alpha_BAT(
    omegas::Array{Array{T, 2}, 1} where T<:Real,
    B::AbstractMatrix{<:Real},
    b::AbstractVector{<:Real},
    alpha0::AbstractVector{<:Real},
    low::AbstractVector{<:Real},
    high::AbstractVector{<:Real}
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

    prior = NamedTupleDist(alpha = [low[i]..high[i] for i in eachindex(low)])
    posterior = BAT.PosteriorDensity(likelihood, prior)
    nsamples = 10^4
    nchains = 4
    nsamples = BAT.bat_sample(posterior, (nsamples, nchains), MetropolisHastings()).result;
    samples_mode = BAT.mode(nsamples).alpha
    return samples_mode

    #TODO:: find appropriate algo
    algorithm = BAT.ModeAsDefined()
    findmode_result = bat_findmode(
        posterior, algorithm,
        initial_mode = samples_mode
        )
    return findmode_result.result[].alpha
end
