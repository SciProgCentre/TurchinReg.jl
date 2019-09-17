struct SegmentPiecewisePoly
    polys::Vector
    knots::Vector{<:Real}

    function SegmentPiecewisePoly(
        f::Vector,
        knots::Vector{<:Real}
        )
        if length(f) + 1 != length(knots)
            Base.error("length(poly) + 1 != length(knots), $(length(f)), $(length(knots))")
        end
        #TODO:: Проверка за O(size(knots)) не хорошо
        if !issorted(knots)
            Base.error("Knots should be sorted")
        end
        return new(f, knots)
    end
end


function find_poly(poly::SegmentPiecewisePoly, x::Real, segmetize::Bool=true, atol=1e-7)
    if poly.knots[1] > x
        return Poly([0])
        Base.error("x should be inside the segment")
    end
    for (index, value) in enumerate(poly.knots)
        if value > x
            return poly.polys[index - 1]
        end
    end

    if segmetize == true
        if isapprox(x, poly.knots[end], atol=atol) && x <= poly.knots[end]
            return poly.polys[end]
        end
    end
    return Poly([0])
    #TODO:: что возвращать вне области определения?
    Base.error("x should be inside the segment")
end


(poly::SegmentPiecewisePoly)(x::Real) = find_poly(poly, x)(x)

der(poly::SegmentPiecewisePoly, x::Real) = polyder(find_poly(poly, x))(x)
der(poly::SegmentPiecewisePoly) = SegmentPiecewisePoly(map(polyder, poly.polys), poly.knots)

function der_order(poly::SegmentPiecewisePoly, order::Int)
    if order == 0
        return poly
    end
    return der(der_order(poly, order-1))
end

function antider(poly::SegmentPiecewisePoly)::SegmentPiecewisePoly
    return SegmentPiecewisePoly(map(polyint, poly.polys), poly.knots)
end

function int(poly::SegmentPiecewisePoly, a::Real, b::Real)
    if !(a >= poly.knots[1] && a <= b && b <= poly.knots[end])
        Base.error("a or b not in the segment")
    end
    integral = 0.
    for (index, value) in enumerate(poly.polys)
        if a < poly.knots[index + 1] && b > poly.knots[index]
            s = polyint(value, poly.knots[index], poly.knots[index + 1])
            if a > poly.knots[index]
                s -= polyint(value, poly.knots[index], a)
            end
            if b < poly.knots[index + 1]
                s -= polyint(value, b, poly.knots[index + 1])
            end
            integral += s
        end
    end
    return integral
end

function merge_knots(a::AbstractVector{<:Real}, b::AbstractVector{<:Real}, atol::Real=1e-7)
    # TODO:: Check a and b are sorted ???????
    i = 1
    j = 1
    result = []
    while (i != length(a) + 1 && j != length(b) + 1)
        if isapprox(a[i], b[j], atol=atol)
            append!(result, a[i])
            i += 1
            j += 1
        elseif a[i] < b[j]
            append!(result, a[i])
            i += 1
        else
            append!(result, b[j])
            j += 1
        end
    end
    while i != length(a) + 1
        append!(result, a[i])
        i += 1
    end
    while j != length(b) + 1
        append!(result, b[j])
        j += 1
    end
    return result
end


import Base.+
function (+)(a::SegmentPiecewisePoly, b::SegmentPiecewisePoly)
    new_knots = merge_knots(a.knots, b.knots)
    new_polys = []
    for (l, r) in collect(zip(new_knots[1:end-1], new_knots[2:end]))
        mid = l / 2 + r / 2 # TODO:: Узнать как правильно искать среднее
        append!(new_polys, [find_poly(a, mid) + find_poly(b, mid)])
    end
    return SegmentPiecewisePoly(new_polys, convert(Array{Float64}, new_knots))
end

import Base.-
function (-)(a::SegmentPiecewisePoly, b::SegmentPiecewisePoly)
    return a + SegmentPiecewisePoly(-b.polys, b.knots)
end

import Base.*
function (*)(a::SegmentPiecewisePoly, b::SegmentPiecewisePoly)
    new_knots = merge_knots(a.knots, b.knots)
    new_polys = []
    for (l, r) in collect(zip(new_knots[1:end-1], new_knots[2:end]))
        mid = l / 2 + r / 2 # TODO:: Узнать как правильно искать среднее
        append!(new_polys, [find_poly(a, mid) * find_poly(b, mid)])
    end
    return SegmentPiecewisePoly(new_polys, convert(Array{Float64}, new_knots))
end
