# Details of Algorithms

## Nearest Correlation Matrix

This algorithm is trying to solve the optimization problem

$\begin{aligned}
    \mathrm{min}\quad & \frac{1}{2} \Vert G - X \Vert^2 \\
    \mathrm{s.t.}\quad & X_{ii} = 1, \quad i = 1, \ldots , n, \\
    & X \in S_{+}^{n}
\end{aligned}$

## Pearson Matching

## NORTA

Given:

* A target correlation matrix, $\rho$
* A list of marginal distributions, $F$

Do:

* Generate $Z_{n \times d} = \mathcal{N}(0, 1)$ IID standard normal samples
* Transform $Y = ZC$ where $C$ is the upper Cholesky factor of $\rho$
* Transform $U = \Phi(Y)$ where $\Phi(\cdot)$ is the CDF of the standard normal distribution
* Transform $X_i = F_{i}^{-1}(U_i)$

## Correlation Conversion