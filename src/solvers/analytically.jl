function _solve(algo::Analytically, alphas, omegas, B, b, phi_bounds, basis)
    Ba0 = B + transpose(alphas.alphas) * omegas
    iBa0 = sym_inv(Ba0)
    r = iBa0 * b
    @info "Solved analytically"
    return PhiVec(r, basis, iBa0, alphas.alphas)
end
