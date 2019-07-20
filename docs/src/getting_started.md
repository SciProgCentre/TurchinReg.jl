# Getting started

## Installation
To install StatReg.jl, start up Julia and type the following code-snipped into the REPL.
```julia
import Pkg
Pkg.add("StatReg.jl")
```

## Usage

To reconstruct function you need to load data (``f(y)``) and data errors (``\delta f(y)``) and define Kernel ``K(x, y)``.
There are two possibilities: use vector & matrix form or continuous form.
In the first case ``K(x, y)`` is matrix ``n \times m``, ``f(y)`` and ``\delta f(y)`` - n-dimensional vectors.
In the second case ``K(x, y)`` is function, ``f(y)`` and ``\delta f(y)`` can be either functions or vectors.
If they are functions, knot vector ``y`` should be specified.

* Define data and errors (`y` is a list of measurement points, `f` is a list of function values at these points, `delta_f` is a list of error in these points)

```julia
using StatReg

y = collect(range(a, stop=b, length=20))
f = [-0.00906047, 0.0239243, 0.168545, 0.503525, 1.27398,
    2.08793, 2.78204, 2.5939, 1.90542, 1.13321,
    0.88324, 1.04642, 1.14465, 1.25853, 0.977623,
    0.621435, 0.310866, 0.117351, 0.0240617, 0.0226408]
delta_f = [0.00888777, -0.00480116, 0.0396684, -0.00968618, -0.0195524,
    -0.0988949, 0.00829277, 0.0494844, -0.0445353, 0.0556071,
    0.00347363, 0.0317405, 0.0539028, 0.0545216, 0.0870875,
    0.0338332, -0.0121158, 0.00790281, 0.00138139, 0.00662381]

```

* Then define kernel:
```julia
function kernel(x::Float64, y::Float64)
    if abs(x-y) <= 1
        return 1. -  abs(x-y)
    end
    return 0.
end
```
* Basis:

We use Cubic Spline Basis with knots in data points and zero boundary conditions on the both sides.

```julia
basis = CubicSplineBasis(y, ("dirichlet", "dirichlet"))
```

* Model:

To reconstruct the function we use matrix of the second derivatives as a prior information. Then we choose a solution model.

```julia
omega = omega(basis, 2)
model = GaussErrorUnfolder(basis, omega)
```

* Reconstruction:

To reconstruct the function we use ``solve()`` that returns dictionary containing coefficients of basis function in the sum ``\varphi(x) = \sum_{k=1}^N coeff_n \psi_n(x)``, their errors ``sig_n`` (``\delta \varphi =  \sum_{k=1}^N sig_n \psi_n(x)``) and optimal parameter of smoothness ``\alpha``.

```julia
phi_reconstruct = solve(model, kernel, f, delta_f, y)
```
* Results

Presentation of results in a convenient way is possible with `PhiVec`:
```julia
phivec = PhiVec(phi_reconstruct["coeff"], basis, phi_reconstruct["sig"])

phi_reconstructed = call(phivec, x)
phi_reconstructed_errors = errors(phivec, x)

plot(x, phi_reconstructed)
fill_between(x, phi_reconstructed - phi_reconstructed_errors, phi_reconstructed + phi_reconstructed_errors, alpha=0.3)
```
