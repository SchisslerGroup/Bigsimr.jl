# Details of Algorithms

## Correlation Conversion

Given:

* A target Spearman's $\rho$ or Kendall's $\tau$ correlation

Do:

* Determine the Pearson correlation for use in a Gaussian copula such that the resulting random samples using the NORTA algorithm have an estimated Spearman or Kendall correlation that matches the target correlation

From Proposition 22 (Lebrun and Dutfow, 2009), we have the result for Gaussian copula that the copula correlation $r$ is related to Spearman's $\rho$ and Kendall's $\tau$ by the following relations:

$r = 2 \sin \left(\frac{\pi}{6} \rho\right) \\
r = \sin \left(\frac{\pi}{2} \tau\right)$

* Lebrun, R., & Dutfoy, A. (2009). An innovating analysis of the Nataf transformation from the copula viewpoint. Probabilistic Engineering Mechanics, 24(3), 312-320.

## Pearson Matching

Given:

* A target Pearson correlation matrix, $R$
* A list of marginal distributions, $F$

Do:

* Determine the input correlation, $\tilde{P}$, such that the resulting random samples using the NORTA algorithm have an estimated Pearson correlation that matches $P$

This method relies on Equations 19, 39, and 49 (Xiao and Zhou, 2019).

* Xiao, Q., & Zhou, S. (2019). Matching a correlation coefficient by a Gaussian copula. Communications in Statistics-Theory and Methods, 48(7), 1728-1747.

## Nearest Correlation Matrix

This algorithm is trying to find a correlation matrix $X$ close to $G$ in the convex optimization problem:

$\begin{aligned}
    \mathrm{min}\quad & \frac{1}{2} \Vert G - X \Vert^2 \\
    \mathrm{s.t.}\quad & X_{ii} = 1, \quad i = 1, \ldots , n, \\
    & X \in S_{+}^{n}
\end{aligned}$

* Qi, H., & Sun, D. (2006). A quadratically convergent Newton method for computing the nearest correlation matrix. SIAM journal on matrix analysis and applications, 28(2), 360-385.

## Normal to Anything (NORTA)

Given:

* A target correlation matrix, $\rho$
* A list of marginal distributions, $F$

Do:

* Generate $Z_{n \times d} = \mathcal{N}(0, 1)$ IID standard normal samples
* Transform $Y = ZC$ where $C$ is the upper Cholesky factor of $\rho$
* Transform $U = \Phi(Y)$ where $\Phi(\cdot)$ is the CDF of the standard normal distribution
* Transform $X_i = F_{i}^{-1}(U_i)$
