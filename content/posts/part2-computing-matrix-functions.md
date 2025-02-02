---
title: "Computing Matrix Functions (Part 2)"
date: 2025-01-31T18:54:15+01:00
draft: true
---

_This is the second part of a series of posts exploring
how to make exponential integrators fast._

In our
[previous exploration](https://konradha.github.io/posts/part1-exponential-integrators/)
we came to understand what "nice" integrators for integrable systems can
look like. How does one actually compute these trigonometric matrix functions in practice? 

Let's reformulate our current issue as: We want to compute an update for our numerical
integration in the temporal dimension, ie. (short form here)

$$u(t + \tau) \approx \exp(\tau A) u(t)$$


We're in the business of computing matrix functions for general discrete Laplacians.
Computing the Taylor expansion should be fairly easy, right? Turns out this has some
issues. For instance, it converges terribly. Doing 100+ matmuls isn't too difficult
especially if $A$ stays constant. The problem, however, gets worse.


Let's look at what [Scipy does](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html).
Relying on Higham's and Al-Mohy's work it states
> essentially a Pade approximation with a variable order that is decided based on the array data

Ok, cool. Can we do that as well, translating, optimizing this for the matrices arising in our problem?
Let's inspect our matrix more precisely.

Using a finite differences, first-order approximation of our discrete Laplacian $\bar{\Delta}$
with no-flux boundary conditions is going to be tridiagonal, symmetric with 2 more bands on the
$+/- n\_x$ diagonals. So we have $n + 2 (n - 1) + 2 (n - nx)$ nonzeros total. Why would we then
save the entire dense matrix to store all elements? We could make use of the plethora of sparse 
matrix formats in existence: CSR, COO,
[BCSR](https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.sparse.BCSR.html) ...
[take your pick](https://docs.nvidia.com/nvpl/latest/sparse/storage_format/sparse_matrix.html).

Computing the k-th power of such a matrix won't preserve the sparsity pattern however. So,
expanding $\exp\left(\tau A\right) = \sum\_{k=0}^{\infty} \frac{\left(\tau A\right)^{k}}{k!} $
is, in general, dense. As an aside, $\bar{\Delta}^{-1}$ will, in general, not be dense either.
One can observe this fact by looking at the spectrum of $\bar{\Delta}$ and how the inverse spectrum
would decay.

So, well, if we don't want to actually have a huge not-so-sparse object in memory to compute for
every time step -- is it actually feasible?

#### Krylov subspaces

A Krylov subspace $\mathcal{K}\_l$ is defined from a matrix-vector pair $(A, u)$, $A \in \mathbb{K}^{m,  n}$ and
$u \in \mathbb{K}^{m}$:

$$\mathcal{K}\_l := \text{span}\[u, Au, ... A^{l-1} u \]$$

It's a richly developed theory in numerical linear algebra to give us some theoretical background on the question:
If a matrix A is sparse, expensive to store and relatively expensive to compute, how can we nicely approximate
$f(tA)$ without making too many "bad decisions"?

There's considerations about "norming" the initial vector u which we'll glance over for now. Will be expanded on later.


#### Arnoldi iteration

So finally, after all of these consideration we have arrived at a way of performing a single time-step integration.
We compute $\exp(\tau A) u(t)$ for every time step without explicitly computing either the matrix exponential or the
full matrix-vector product of it. 

The Arnoldi iteration now gives us a way how to compute the Krylov subspace, apply the matrix function we're looking for
and project it back into our original $\mathbb{K}^{m, n}$ space.

The basic form of this iteration is

$$A V\_m = V\_m H\_m + h\_{m+1, m} v\_{m+1} e\_m^{T}$$


To understand this form we'll have to take a step back and make use of our knowledge of complex analysis.
The matrix function $f(A)$ applied to a vector may be expressed as contour integral in the following manner

$$f(A)v = \frac{1}{2 \pi i } \int_{\Gamma} f(\lambda) (\lambda \mathbf{I} - A)^{-1} v d \lambda$$

where we define 

$$\mathcal{F} (A) = \bigl\{x^\* A x : x \in \mathbb{K}^{m}, ||x|| = 1 \bigl\}$$







 Arnoldi iteration for matrix exponent

- Then finally, let's implement it in Eigen: exp(t A) u


