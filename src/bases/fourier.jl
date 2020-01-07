"""
Fourier basis with length ``2n+1``: {``0.5``, ``\\sin(\\frac{\\pi (x - \\frac{a+b}{2})}{b-a})``, ``\\cos(\\frac{\\pi (x - \\frac{a+b}{2})}{b-a})``, ..., ``\\sin(\\frac{\\pi n (x - \\frac{a+b}{2})}{b-a})``, ``\\cos(\\frac{\\pi n (x - \\frac{a+b}{2})}{b-a})``}.

```julia
FourierBasis(a::Real, b::Real, n::Int)
```
`a`, `b` -- the beginning and the end of the segment
`n` -- number of basis functions

**Fields**

* `a::Real` -- beginning of the support
* `b::Real` -- end of the support
* `n::Int` -- number of basis functions
* `basis_functions::AbstractVector{BaseFunction}` -- array of basis functions
"""
struct FourierBasis <: Basis
    a::Real
    b::Real
    n::Int
    basis_functions::AbstractVector{BaseFunction}

    function basis_functions_fourier(n::Int, a::Real, b::Real)
        l = (b - a) / 2.
        mid = (a + b) / 2.
        func = [BaseFunction(x::Float64 -> 0.5, a, b)]
        for i = 1:n
            push!(func, BaseFunction(x::Float64 ->
            cos(i * pi * (x - mid) / l), a, b))
            push!(func, BaseFunction(x::Float64 ->
            sin(i * pi * (x - mid) / l), a, b))
        end
        return func
    end

    function FourierBasis(a::Real, b::Real, n::Int)
        if a >= b
            @error "Incorrect specification of a segment: `a` should be less than `b`."
            Base.error("Incorrect specification of a segment: `a` should be less than `b`.")
        end
        if n <= 0
            @error "Number of basis functions should be positive."
            Base.error("Number of basis functions should be positive.")
        end
        basis_functions = basis_functions_fourier(n, a, b)
        @info "Fourier basis is created."
        return new(a, b, n, basis_functions)
    end
end


@memoize function omega(basis::FourierBasis, order::Int)
    @info "Calculating omega matrix for Fourier basis derivatives of order $order..."
    if order < 0
        @error "Order of derivative should be positive."
        Base.error("Order of derivative should be positive.")
    end
    a, b = basis.a, basis.b
    delta = (b - a) / 2
    temp = zeros(Float64, 2 * basis.n + 1)
    if order == 0
        temp[1] = delta
    end
    for i = 1:basis.n
        val = ((i * pi) / delta) ^ (2 * order) * delta / 2
        temp[2 * i] = val
        temp[2 * i + 1] = val
    end
    @info "Omega caclulated successfully."
    return cat(temp...; dims=(1,2))
end
