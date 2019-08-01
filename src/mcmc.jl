using Mamba

include("basis.jl")
include("vector.jl")
include("config.jl")


mutable struct MCMC_matrix
    omegas::Array{Array{Float64, 2} ,1}
    n::Int64
    method::String
    alphas::Union{Array{Float64}, Nothing}
    low::Union{Array{Float64, 1}, Nothing}
    high::Union{Array{Float64, 1}, Nothing}
    alpha0::Union{Array{Float64, 1}, Nothing}

    function MCMC_matrix(
        omegas::Array{Array{Float64, 2} ,1},
        method::String="EmpiricalBayes",
        alphas::Union{Array{Float64, 1}, Nothing}=nothing,
        low::Union{Array{Float64, 1}, Nothing}=nothing,
        high::Union{Array{Float64, 1}, Nothing}=nothing,
        alpha0::Union{Array{Float64, 1}, Nothing}=nothing
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
        @info "GaussErrorMatrixUnfolder is created."
        return new(omegas, m, method, alphas, low, high, alpha0)
    end
end


function solve(
    unfolder::MCMC_matrix,
    kernel::Array{Float64, 2},
    data::Array{Float64, 1},
    data_errors::Union{Array{Float64, 1}, Array{Float64, 2}},
    chains::Int64 = 1,
    samples::Int64 = 10 * 1000
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
    return solve_correct(unfolder, kernel, data, data_errors, chains, samples)
end


function solve_correct(
    unfolder::MCMC_matrix,
    kernel::Array{Float64, 2},
    data::Array{Float64, 1},
    data_errors::Array{Float64, 2},
    chains::Int64 = 1,
    samples::Int64 = 10 * 1000
    )

    @info "Starting solve_correct..."
    K = kernel
    Kt = transpose(kernel)
    data_errorsInv = pinv(data_errors)
    data_errorsInv = (transpose(data_errorsInv) + data_errorsInv) / 2.
    B = Kt * data_errorsInv * K
    B = (transpose(B) + B) / 2.
    b = Kt * transpose(data_errorsInv) * data
    low = unfolder.low
    high = unfolder.high
    alpha0 = unfolder.alpha0

    function find_optimal_alpha()
        @info "Starting find_optimal_alpha..."

        function alpha_prob(a::Array{Float64, 1})

            for i in range(1, stop=length(a))
                if (a[i] <= low[i]) || (a[i] >= high[i])
                    return -1e4
                end
            end

            aO = transpose(a)*unfolder.omegas
            BaO = B + aO
            if abs(det(BaO)) < 1e-6
                @warn "a = $(a[1]), abs(det(BaO)) < 1e-6" maxlog=10
            end
            iBaO = pinv(BaO)
            iBaO = (transpose(iBaO) + iBaO) / 2.
            dotp = transpose(b) * iBaO * b
            if det(aO) != 0
                detaO = logabsdet(aO)[2]
            else
                eigvals_aO = sort(eigvals(aO))
                rank_deficiency = size(aO)[1] - rank(aO)
                detaO = sum(log.(abs.(eigvals_aO[(rank_deficiency+1):end])))
            end
            if det(BaO) != 0
                detBaO = logabsdet(BaO)[2]
            else
                eigvals_BaO = sort(eigvals(BaO))
                rank_deficiency = size(BaO)[1] - rank(BaO)
                detBaO = sum(log.(abs.(eigvals_BaO[(rank_deficiency+1):end])))
            end
            return detaO - detBaO + dotp
        end

        @info "Starting optimization..."

        function my_alpha_prob(a::Array{Float64, 1})
            delta = alpha_prob(a + [1e-5 for i in range(1, length(a))]) - alpha_prob(a)
            if abs(delta/1e-5) > 1e3
                return 0.
            end
            return alpha_prob(a)
        end


        # s = collect(range(low[1], stop=high[1], length=100*1000))
        # res1 = [-alpha_prob([s_]) for s_ in s ]
        # # s_log = [log(s_) for s_ in s]
        # plot(s[2:end-1], res1[2:end-1])#, "o", markersize=1)

        a0 = log.(alpha0)
        res = optimize(
            a -> -alpha_prob(exp.(a)), a0,  BFGS(),
            Optim.Options(x_tol=X_TOL_OPTIM, show_trace=true,
            store_trace=true, allow_f_increases=true))

        if !Optim.converged(res)
            @warn "Minimization did not succeed, alpha = $(exp.(Optim.minimizer(res))), return alpha = 0.05."
            return [0.05]
        end
        alpha = exp.(Optim.minimizer(res))
        if alpha[1] > 1e3
            @warn "Incorrect alpha: too small or too big, alpha = $(exp.(Optim.minimizer(res))), return alpha = 0.05."
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

    return solve_MCMC(unfolder, kernel, data, data_errors, chains, samples)
end

function solve_MCMC(
    unfolder::MCMC_matrix,
    kernel::Array{Float64, 2},
    data::Array{Float64, 1},
    data_errors::Array{Float64, 2},
    chains::Int64 = 1,
    samples::Int64 = 10 * 1000
    )

    sigma = pinv(unfolder.omegas[1])
    sigma = (transpose(sigma) + sigma) / 2.
    n = unfolder.n
    m = length(data)
    alpha = unfolder.alphas[1]

    model = Model(
        phi = Stochastic(1, (n, sigma, alpha) ->  MvNormal(zeros(n), sigma/sqrt(alpha))),
        mu = Logical(1, (kernel, phi) -> kernel * phi, false),
        f = Stochastic(1, (mu, data_errors) ->  MvNormal(mu, data_errors), false),
        )

    scheme = [NUTS([:phi])]

    line = Dict{Symbol, Any}(
        :f => data,
        :kernel => kernel,
        :data_errors => data_errors,
        :n => n,
        :sigma => sigma,
        :alpha => alpha
        )

    inits = [
            Dict{Symbol, Any}(
            :phi => rand(Normal(0, 1), n),
            :f => data
            ) for i in 1:chains
            ]

    setsamplers!(model, scheme)
    return model, line, inits, samples, 250, 2, chains
end

function get_values(sim::ModelChains, chains::Int64, n::Int64)
    res =  mean([sim.value[:, :, j] for j in range(1, stop=chains)])
    ans = []
    for i in range(1, stop=n)
        append!(ans, mean(res[:, i]))
    end
    return convert(Array{Float64}, ans)
end

    # function P_f_phi(
    #     f::Array{Float64, 1},
    #     phi::Array{Float64, 1},
    #     kernel::Array{Float64, 2},
    #     sigma::Array{Float64, 2}
    #     )
    #
    #     m = length(f)
    #     norm = 1 / ((2 * pi)^(m / 2) * (det(sigma))^(1 / 2))
    #     P = exp(- 1 / 2 * transpose(f - K * phi) * pinv(sigma) * (f - K * phi))
    #     return norm * P
    # end
    #
    # function sigma2_n(S::Array{Float64, 1}, n::Int64)
    #     return Integral_mcmc(
    #         x::Array{Float64, 1} ->
    #         (x[n] - S[n])^2 * P_f_phi(data, x, kernel, data_errors)
    #         )
    # end
    #
    # function P_phi_f(
    #     f::Array{Float64, 1},
    #     phi::Array{Float64, 1},
    #     kernel::Array{Float64, 2},
    #     sigma::Array{Float64, 2}
    #     )
    #
    #     norm = Integral_mcmc(
    #         x::Array{Float64, 1} ->
    #         P_phi(x) * P_f_phi(f, x, kernel, sigma)
    #         )
    #     P = P_phi(phi) * P_f_phi(f, phi, kernel, sigma) / norm
    #
    #     return P
    # end
    #
    # function P_phi(phi::Array{Float64, 1})
    #
    #     alpha = unfolder.alphas
    #     omega = unfolder.omegas
    #     num = alpha[i]^(rank(omega[i])) * abs(det(omega[i]))
    #     den = (2. * pi)^(length())
    #     pow = - 1 / 2 * transpose(phi) * alpha[i] * omega[i] * phi
    #
    #     return sqrt(num / den) * exp(- 1 / 2 * pow)
    # end


# end
