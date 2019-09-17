include("piecewise_polynomials.jl")

struct BSpline
    i::Int
    k::Int
    knots::AbstractVector{<:Real}
    func::SegmentPiecewisePoly

    function BSpline(i::Int, k::Int, knots::AbstractVector{<:Real})
        if i < 0
            @error "BSline number should be positive."
            Base.error("BSline number should be positive.")
        end
        if k < 0
            @error "BSline order should be positive."
            Base.error("BSline order should be positive.")
        end

        function b_spline_function(i::Int, k::Int, knots::AbstractVector{<:Real})
            if  k == 0
                polys = [Poly([0]) for i in range(1, stop=length(knots)-1)]
                polys[i + 1] = Poly([1])
                return SegmentPiecewisePoly(polys, knots)
            end

            first = SegmentPiecewisePoly([Poly([0])], [knots[1], knots[end]])
            second = SegmentPiecewisePoly([Poly([0])], [knots[1], knots[end]])
            if !isapprox(abs(knots[i+k+1]-knots[i+1]) + 1, 1)
                first = SegmentPiecewisePoly(
                    [Poly([-knots[i+1]/(knots[i+k+1]-knots[i+1]), 1/(knots[i+k+1]-knots[i+1])])],
                    [knots[1], knots[end]]
                    ) * b_spline_function(i, k-1, knots)
            end

            if !isapprox(abs(knots[i+k+1+1]-knots[i+1+1]) + 1, 1)
                second = SegmentPiecewisePoly(
                [Poly([knots[i+k+1+1]/(knots[i+k+1+1]-knots[i+1+1]), -1/(knots[i+k+1+1]-knots[i+1+1])])],
                [knots[1], knots[end]]
                ) * b_spline_function(i+1, k-1, knots)
            end

            return first + second
        end

        return new(i, k, knots, b_spline_function(i, k, knots))
    end
end

(b_spline::BSpline)(x::Real) = b_spline.func(x)