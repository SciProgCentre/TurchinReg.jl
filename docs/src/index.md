# Statreg.jl

This is documentation for `Statreg.jl` -- a Julia package that allows to apply Turchin's method of statistical regularisation to solve the Fredholm equation of the first kind.


Let's consider equation

```math
f(y) = \int\limits_{a}^{b} K(x, y) \varphi(x) dx
```

 The problem is, given kernel function ``K(x, y)`` and observed function ``f(y)``, to find the function ``\varphi(x)``. ``f(y)`` contains a random noise factor both from initial statistical uncertainty of ``\varphi(x)`` and additional noise from measurement procedure.
The equation is ill-posed: a small measurement error of ``f(y)`` leads to big instability of ``\varphi(x)``. Solving such ill-posed problems requires operation called regularisation. It means we need to introduce additional information to make the problem well-posed one.

The idea of statistical regularisation is related to the Bayesian statistics approach: unknown statistical value ``\varphi(x)`` could be reconstructed using measured value ``f(y)``, kernel ``K(x, y)`` and some prior information about ``\varphi(x)`` behaviour: smoothness, constraints on boundary conditions, non-negativity, etc.
Also, it is important to note that statistical regularisation allows to estimate errors of the obtained solution. More information about the theory of statistical regularisation you can find [here](https://www.epj-conferences.org/articles/epjconf/abs/2018/12/epjconf_ayss2018_07005/epjconf_ayss2018_07005.html), but the main concepts will be explained further in this documentation.

# Description of statistical regularisation method

Firstly, it is necessary to make a parameterised discrete representation of the continuous functional space. We should introduce basis ``\{ \psi_k \}_{k=1}^N``, in which the required function will be calculated. Thus, the Fredholm equation will be сonverted to the matrix equation:

```math
f_m = K_{mn} \varphi_n,
```
where ``f_m = f(y_m)``,  ``\varphi_n`` :  ``\varphi(x) = \sum\limits_{k=1}^{N} \varphi_k \psi_k(x)``,  ``K_{mn} = \int\limits_{a}^{b} K(x, y_m) \psi_n(x) dx``.

Let's introduce function ``\overrightarrow{S}`` that will evaluate ``\overrightarrow{\varphi}`` based on the function ``\overrightarrow{f}`` and loss function ``L(\overrightarrow{\widehat{\varphi}}, \overrightarrow{S}) = \sum\limits_{n=1}^{N} \mu_n (\widehat{\varphi}_n - S_n)^2``, where ``\overrightarrow{\widehat{\varphi}}=\overrightarrow{\widehat{S}}(\overrightarrow{f})`` -- the best solution.

For this loss function the best strategy is

```math
\overrightarrow{\widehat{S}}[f]=E[\overrightarrow{\varphi}|\overrightarrow{f}]=\int \overrightarrow{\varphi} P(\overrightarrow{\varphi}|\overrightarrow{f}) d\overrightarrow{\varphi}
```

Errors of the solution:

```math
< \sigma_n^2 > = \int (\varphi_n - \widehat{S}_n)^2 P(\overrightarrow{\varphi}|\overrightarrow{f})d\overrightarrow{\varphi}
```

```math
P(\overrightarrow{\varphi}|\overrightarrow{f}) = \frac{P(\overrightarrow{\varphi})P(\overrightarrow{f}|\overrightarrow{\varphi})}{\int d\overrightarrow{\varphi}P(\overrightarrow{\varphi})P(\overrightarrow{f}|\overrightarrow{\varphi})}
```

Thus, ``P(\overrightarrow{\varphi})`` and ``P(\overrightarrow{f}|\overrightarrow{\varphi})`` are required to find the solution. ``P(\overrightarrow{\varphi})`` can be chosen using prior information about ``\overrightarrow{\varphi})``. ``P(\overrightarrow{f}|\overrightarrow{\varphi})`` depends on ``\overrightarrow{f}`` distribution. Let's consider different distributions of ``\overrightarrow{\varphi}`` and ``\overrightarrow{f}``.

# Smoothness as a prior information

We expect ``\varphi(x)`` to be relatively smooth and can choose this information as a prior.
The matrix of the mean value of derivatives of order ``p`` can be used as a prior information about the solution.
```math
\Omega_{mn} = \int\limits_{a}^{b} \left( \frac{d^p \psi_m(x)}{dx^p} \right) \left( \frac{d^p \psi_n(x)}{dx^p} \right) dx
```

We require a certain value of the smoothness functional to be achieved:

```math
\int (\overrightarrow{\varphi}, \Omega \overrightarrow{\varphi}) P(\overrightarrow{\varphi})d\overrightarrow{\varphi}=\omega
```

Thus, the ``\overrightarrow{\varphi}`` probability distribution depends on the parameter:

```math
P_{\alpha}(\overrightarrow{\varphi})=\frac{\alpha^{Rg(\Omega)/2} \sqrt{\text{det}(\Omega)}}{(2\pi)^{N/2}}\text{exp}\left( -\frac{1}{2} (\overrightarrow{\varphi}, \Omega \overrightarrow{\varphi}) \right),
```
where ``\alpha=\frac{1}{\omega}``.

The value of the parameter ``\alpha`` is unknown and can be obtained in following ways:
* directly from some external data or manually selected
* as a maximum of a posterior information ``P(\alpha|\overrightarrow{f})``
* as the mean of all possible ``\alpha``, defining the prior probability density as ``P(\overrightarrow{\varphi})=\int d\alpha P(\alpha) P(\overrightarrow{\varphi}|\alpha)`` (all alphas are equally probable).

`StatReg.jl` allows to apply all of these options.

# Gaussian random process
Experimental data usually follows a normal distribution. At that rate the regularisation has an analytical solution. Let the measurement vector f have errors described by a multidimensional Gaussian distribution with a covariance matrix ``\Sigma``:
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

This package allows to apply statistical regularisation in different bases using such prior information as smoothness or zero boundary conditions, or another information provided by user in a matrix form. ``\Omega`` can be set manually or calculated for every derivative  of degree ``p``. ``\alpha`` can be calculated as a maximum of a posterior information or can be set manually.


# Non-Gaussian random process
If the ``f`` function errors do not follow Gaussian distribution, the strategy ``\overrightarrow{\widehat{S}}`` can not be calculated analytically in general case.
```math
\overrightarrow{\widehat{S}}[f]=E[\overrightarrow{\varphi}|\overrightarrow{f}]=\int \overrightarrow{\varphi} P(\overrightarrow{\varphi}|\overrightarrow{f}) d\overrightarrow{\varphi}
```
The posterior probability ``P(\overrightarrow{\varphi}|\overrightarrow{f})`` should be obtained from MCMC sampling. It is applied in the `StatReg.jl` using `BAT.jl`, `AHMC.jl` and `DHMC.jl` packages.


# Non-negativity and other conditions
If `phi` is known to be non-negative or has some value restrictions, it can be used to improve the solution. To take it into account, posterior probability is multiplied by fast declining function if `phi_i` reached the bound.

```math
P(\varphi | f)_{\varphi_{min} <\varphi < \varphi_{max}}  = P(\varphi | f) \cdot
e^{- C_{lower}} \cdot e^{- C_{higher}}
```
```math
C_{lower} = \sum_{m=1}^M I(\varphi_m < \varphi_{min\text{ }m}) \cdot \frac{\varphi_{min\text{ }m} - \varphi_m}{\varphi_{0 m}}
```

```math
P(\varphi | f)_{\varphi_{min} <\varphi < \varphi_{max}}  = P(\varphi | f) \cdot
e^{- C_{lower}} \cdot e^{- C_{higher}}
```

```math
C_{higher} = \sum_{m=1}^M I(\varphi_m > \varphi_{max\text{ }m}) \cdot \frac{\varphi_{m} - \varphi_{max\text{ }m}}{\varphi_{0 m}}
```
