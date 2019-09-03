macro returntrue(f::Expr)
    try
        eval(f)
    catch e
        print(e)
        return false
    end
    return true
end
