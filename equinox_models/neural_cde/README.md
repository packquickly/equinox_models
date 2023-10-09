# Neural Controlled Differential Equation

A neural controlled differential equation (neural CDE) is a continuous-time
sequence-to-sequence model originally described in
[Neural Controlled Differential Equations for Irregular Time Series](https://arxiv.org/abs/2005.08926).

## Basic architecture

The neural CDE architecture is a model of the form:

$$Y_t = \int_0^t f_\theta(Y_s) \,d X_s$$

where $f_\theta$ is a neural network, $X_t$ is the input time-series,
and $Y_t$ is the output time-series. This integral is interpreted in the
[Riemann-Stieltjes](https://en.wikipedia.org/wiki/Riemann%E2%80%93Stieltjes_integral)
sense.

Neural CDEs (as presented here) assume that $X$ is differentiable, reduce the
above integral to

$$Y_t = \int_0^t f_\theta(Y_s) \dot{X}_s \,ds.$$

and solve this numerically using a neural ODE solver.

## Implementation
We use [Diffrax](https://github.com/patrick-kidger/diffrax) for solving the neural
CDE. This implementation is a very simple example for classification, and is 
not particularly flexible. It uses a feedforward network for $f_\theta$.
