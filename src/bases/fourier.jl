"""
Fourier basis with length ``2n+1``: {``0.5``, ``\\sin(\\frac{\\pi (x - \\frac{a+b}{2})}{b-a})``, ``\\cos(\\frac{\\pi (x - \\frac{a+b}{2})}{b-a})``, ..., ``\\sin(\\frac{\\pi n (x - \\frac{a+b}{2})}{b-a})``, ``\\cos(\\frac{\\pi n (x - \\frac{a+b}{2})}{b-a})``}.

```julia
FourierBasis(a::Real, b::Real, n::Int)
```
`a`, `b` -- the beginning and the end of the segment
`n` -- number of harmonics

**Fields**

* `a::Real` -- beginning of the support
* `b::Real` -- end of the support
* `n::Int` -- number of harmonics
* `basis_functions::AbstractVector` -- array of basis functions
"""
struct FourierBasis <: Basis
    a::Real
    b::Real
    n::Int
    basis_functions::AbstractVector

    function basis_functions_fourier(n::Int, a::Real, b::Real)
        l = (b - a) / 2.
        mid = (a + b) / 2.
        func = []
        push!(func, x::Float64 -> 0.5)
        for i = 1:n
            push!(func, x::Float64 -> cos(i * pi * (x - mid) / l))
            push!(func, x::Float64 -> sin(i * pi * (x - mid) / l))
        end
        return func
    end

    function FourierBasis(a::Real, b::Real, n::Int)
        @assert a < b "Incorrect specification of a segment: `a` should be less than `b`"
        @assert n > 0 "Number of basis functions should be positive"
        basis_functions = basis_functions_fourier(n, a, b)
        return new(a, b, n, basis_functions)
    end
end


@memoize function omega(basis::FourierBasis, order::Int)
    @info "Calculating omega matrix for Fourier basis derivatives of order $order..."
    @assert order >= 0 "Order of derivative should be non-negative."
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
