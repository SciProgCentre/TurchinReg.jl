using QuadGK, LinearAlgebra, Dierckx, Memoize, ApproxFun

struct BSpline
    i::Int64
    k::Int64
    knots::Array{Float64}
    func::Function

    function BSpline(i::Int64, k::Int64, knots::Array{Float64})

        function b_spline_function(i::Int64, k::Int64, x::Float64, knots::Array{Float64})
            if  k == 0
                if x >= knots[i+1] && x <= knots[i+2]
                    return 1.
                else
                    return 0.
                end
            end
            first = 0.
            second = 0.
            if knots[i+k+1]-knots[i+1] != 0
                first = (x-knots[i+1])/
                    (knots[i+k+1]-knots[i+1])*b_spline_function(i, k-1, x, knots)
            end

            if knots[i+k+1+1]-knots[i+1+1] != 0
                second = (knots[i+k+1+1]-x)/
                    (knots[i+k+1+1]-knots[i+1+1])*b_spline_function(i+1, k-1, x, knots)
            end
            return first+second
        end

        return new(i, k, knots, x -> b_spline_function(i, k, x, knots))
    end
end


function derivative(b_spline::BSpline, x::Float64, deg::Int64)
    if deg == 0
        return b_spline.func(x)
    end
    knots = b_spline.knots
    i = b_spline.i
    k = b_spline.k
    if k == 0
        return 0.
    end
    first = 0.
    second = 0.
    if knots[i+k+1]-knots[i+1] != 0
        first = k * derivative(BSpline(i, k-1, knots), x, deg-1) / (knots[i+k+1] - knots[i+1])
    end
    if knots[i+k+1+1]-knots[i+1+1] != 0
        second = k * derivative(BSpline(i+1, k-1, knots), x, deg-1) / (knots[i+k+1+1] - knots[i+1+1])
    end
    return k * (first - second)
end

(b_spline::BSpline)(x::Float64) = b_spline.func(x)
