# Monte Carlo Simulations

Monte Carlo methods are **stochastic simulation techniques** used to
model **uncertainty and randomness** by performing large numbers of
random experiments.

Used in: - Finance
- Risk modeling
- Reinforcement learning
- Bayesian inference

------------------------------------------------------------------------

## Core Intuition

Monte Carlo answers:

> *"What happens if I simulate this random process thousands of times?"*

------------------------------------------------------------------------

## Markov Property

\[ P(X\_{t+1} \| X_t) = P(X\_{t+1} \| X_t, X\_{t-1}, ...) \]

The future depends only on the present.

------------------------------------------------------------------------

## States & Transitions

States represent system configurations.

Transition probability:

\[ P\_{ij} = P(X\_{t+1}=j \| X_t=i) \]

------------------------------------------------------------------------

## Transition Matrix

\[ P =
```{=tex}
\begin{bmatrix}
P_{11} & P_{12} \\
P_{21} & P_{22}
\end{bmatrix}
```
\]

------------------------------------------------------------------------

## Standard Workflow

1.  Define stochastic process
2.  Sample random variables
3.  Simulate transitions
4.  Aggregate statistics

------------------------------------------------------------------------

## MCMC

Monte Carlo Markov Chain methods: - Metropolis-Hastings
- Gibbs Sampling
- Hamiltonian MC

------------------------------------------------------------------------

## Expected Value

\[ E\[f(X)\] `\approx `{=tex}`\frac{1}{N}`{=tex} `\sum `{=tex}f(x_i) \]

------------------------------------------------------------------------

## Use Cases

### Financial Risk

*Example:* (Insert Example Repo when completed)

### Reinforcement Learning

*Example:* (Insert Example Repo when completed)

### System Reliability

*Example:* (Insert Example Repo when completed)

------------------------------------------------------------------------

## Pros

-   Handles complex systems
-   Easy to parallelize
-   No closed form needed

## Cons

-   Computationally expensive
-   Slow convergence

------------------------------------------------------------------------

## Mental Model

> **Brute-force probability estimation via simulation**

------------------------------------------------------------------------
