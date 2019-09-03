# Statreg.jl

This is the documentation for `Statreg.jl`; a Julia package that allows to apply Turchin's method of statistical regularisation to solve the Fredholm equation of the first kind.


Let's consider the equation

```math
f(y) = \int\limits_{a}^{b} K(x, y) \varphi(x) dx
```

 The problem is, given kernel function ``K(x, y)`` and the function ``f(y)``, to find the function ``\varphi(x)``. ``f(y)`` contains a random noise factor both from initial statistical uncertainty of ``\varphi(x)`` and additional noise from measurement procedure.
The equation is ill-posed: a small error in the measurement of ``f(y)`` leads to big instability of ``\varphi(x)``. Solving such ill-posed problems requires operation called regularisation. It means we need to introduce additional information to make the problem well-posed one.

The idea of statistical regularisation is to look on the problem from the point of view of Bayesian statistics approach: unknown statistical value ``\varphi(x)`` could be reconstructed using measured value ``f(y)``, the kernel ``K(x, y)`` and some prior information about ``\varphi(x)`` behaviour: smoothness, constraints on boundary conditions, non-negativity, etc.
Also it is important to note that statistical regularisation allows to estimate errors of obtained solution. More information about the theory of statistical regularisation you can find in **insert link**, but the main concepts will be explained further in this documentation.

# Description of statistical regularisation’s method

Firstly, it is necessary to go from the continuous space of functions to the parameterised discrete representation. We should introduce basis ``\{ \psi_k \}_{k=1}^N``, in which the required function will be calculated. Then the Fredholm equation will go to the matrix equation:

```math
f_m = K_{mn} \varphi_n,
```
where ``f_m = f(y_m)``,  ``\varphi_n`` :  ``\varphi(x) = \sum\limits_{k=1}^{N} \varphi_n \psi_k(x)``,  ``K_{mn} = \int\limits_{a}^{b} K(x, y_m) \psi_n(x) dx``.

Let's introduce function ``\overrightarrow{S}`` that will evaluate ``\overrightarrow{\varphi}`` based on the function ``\overrightarrow{f}`` and loss function ``L(\overrightarrow{\widehat{\varphi}}, \overrightarrow{S}) = \sum\limits_{n=1}^{N} \mu_n (\widehat{\varphi}_n - S_n)^2``, where ``\overrightarrow{\widehat{\varphi}}=\overrightarrow{\widehat{S}}(\overrightarrow{f})`` -- the best solution.

For this loss function the best strategy is

```math
\overrightarrow{\widehat{S}}[f]=E[\overrightarrow{\varphi}|\overrightarrow{f}]=\int \overrightarrow{\varphi} P(\overrightarrow{\varphi}|\overrightarrow{f}) d\overrightarrow{\varphi}
```
```math
P(\overrightarrow{\varphi}|\overrightarrow{f}) = \frac{P(\overrightarrow{\varphi})P(\overrightarrow{f}|\overrightarrow{\varphi})}{\int d\overrightarrow{\varphi}P(\overrightarrow{\varphi})P(\overrightarrow{f}|\overrightarrow{\varphi})}
```
Errors of the solution:

```math
< \sigma_n^2 > = \int (\varphi_n - \widehat{S}_n)^2 P(\overrightarrow{\varphi}|\overrightarrow{f})d\overrightarrow{\varphi}
```

# Smoothness as a prior information

We expect ``\varphi(x)`` to be relatively smooth and can choose this information as a prior.
The matrix of the mean value of derivatives of order ``p`` can be used as a prior information about the solution.
```math
\Omega_{mn} = \int\limits_{a}^{b} \left( \frac{d^p \psi_m(x)}{dx} \right) \left( \frac{d^p \psi_n(x)}{dx} \right) dx
```

We require a certain value of the smoothness functional to be achieved:

```math
\int (\overrightarrow{\varphi}, \Omega \overrightarrow{\varphi}) P(\overrightarrow{\varphi})d\overrightarrow{\varphi}=\omega
```

Then the ``\overrightarrow{\varphi}`` probability distribution depending on the parameter:

```math
P_{\alpha}(\overrightarrow{\varphi})=\frac{\alpha^{Rg(\Omega)/2} \sqrt{det(\Omega)}}{(2\pi)^{N/2}}exp\left( -\frac{1}{2} (\overrightarrow{\varphi}, \Omega \overrightarrow{\varphi}) \right)
```
where ``\alpha=\frac{1}{\omega}``.

The value of the parameter ``\alpha`` is unknown, and can be obtained in the following ways:
* directly from some external data or manually selected
* as a maximum of a posterior information ``P(\alpha|\overrightarrow{f})``
* as the mean of all possible ``\alpha``, defining the prior probability density as ``P(\overrightarrow{\varphi})=\int d\alpha P(\alpha) P(\overrightarrow{\varphi}|\alpha)`` (all alphas are equally probable).

# Gaussian random process
The most common case is when the variation of the experimental results is subject to the normal distribution. At that rate the regularisation has an analytical solution. Let the measurement vector f have errors described by a multidimensional Gaussian distribution with a covariance matrix ``\Sigma``:
```math
P(\overrightarrow{f}|\overrightarrow{\varphi})=\frac{1}{(2\pi)^{N/2}|\Sigma|^{1/2}}exp\left( -\frac{1}{2} (\overrightarrow{f} - K\overrightarrow{\varphi})^T \Sigma^{-1} (\overrightarrow{f} - K\overrightarrow{\varphi}) \right)
```
Using the most probable ``\alpha``, one can get the best solution:
```math
\overrightarrow{\widehat{S}} = (K^T \Sigma^{-1} K + \alpha^* \Omega)^{-1} K^T \Sigma^{-1 T} \overrightarrow{f}
```

```math
cov(\varphi_m, \varphi_n) = ||(K^T \Sigma^{-1} K + \alpha^* \Omega)^{-1}||_{mn}
```


This package allows to apply statistical regularisation in different bases using such a prior information as smoothness and zero boundary conditions or another information provided by user in a matrix form. ``\Omega`` can be set manually or calculated for every derivative  of degree ``p``. ``\alpha`` can be calculated as a maximum of a posterior information or can be set manually.
