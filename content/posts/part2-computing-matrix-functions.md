---
title: "Computing Matrix Functions"
date: 2025-02-04T18:00:15+01:00
draft: false
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

Let's look at what [Scipy does](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html).
Relying on Higham's and Al-Mohy's work it states
> essentially a Pade approximation with a variable order that is decided based on the array data

Ok, cool. Can we do that as well, translating, optimizing this for the matrices arising in our problem?
Let's inspect our matrix more precisely.

Using a finite differences, first-order approximation of our discrete Laplacian $\bar{\Delta}$
with no-flux boundary conditions is going to be tridiagonal, symmetric with 2 more bands on the
$+/- n\_x$ diagonals. So we have $n + 2 (n - 1) + 2 (n - nx)$ nonzeros total. Would we then
save the entire dense matrix to store all elements? We could make use of the plethora of sparse 
matrix formats in existence: CSR, COO,
[BCSR](https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.sparse.BCSR.html) ...
[take your pick](https://docs.nvidia.com/nvpl/latest/sparse/storage_format/sparse_matrix.html).

Computing the k-th power of such a matrix won't preserve the sparsity pattern however. So,
expanding $\exp\left(\tau A\right) = \sum\_{k=0}^{\infty} \frac{\left(\tau A\right)^{k}}{k!} $
is, in general, dense. As an aside, $\bar{\Delta}^{-1}$ will, in general, not be dense either.
One can observe this fact by looking at the spectrum of $\bar{\Delta}$ and how the its inverse
would decay. We can derive a closed-form solution to $A - \lambda I = 0$ where $A = \bar{\Delta}$
and reason from this result.


So, well, if we don't want to actually have a huge not-so-sparse object in memory to compute for
every time step -- is it actually feasible?

We'll need some tools from linear algebra and complex analysis to actually build this.

#### Krylov subspaces

A Krylov subspace $\mathcal{K}\_l$ is defined from a matrix-vector pair $(A, u)$, $A \in \mathbb{K}^{m,  n}$ and
$u \in \mathbb{K}^{m}$:

$$\mathcal{K}\_l := \text{span}\[u, Au, ... A^{l-1} u \]$$

It's a richly developed theory in numerical linear algebra to give us some background on the question:
If A is sparse, difficult to handle -- how can we nicely approximate
$f(tA) u$ without making too many "bad decisions" in terms of performance and correctness?

Applying the sparse matrix-vector products here yields something that looks similar to
what the above span offers. However: To be sure that we can actually use this, we need to come up with
a procedure that can generate a basis for this Krylov space. Ie. for any element in our newly-built basis $\mathcal{B}$
we require something like $\langle v\_i, v\_j \rangle = 0$ for $i \neq j, v\_i \in \mathcal{B}$.

That's what the iterations related to Arnoldi and Lanczos processes are about.

#### Arnoldi iterations

Originally, this type of process had been constructed to approximate an inverse computation that arose often
in the earlier days of scientific computing. Well, it still arises, historically it's just from that area.

$$\left(\lambda I - A\right)^{-1} v$$

ie. approximating the inverse of a matrix was the main interest for this method.
The iteration finds matrices $V\_m, H\_m$. It might also have to find a normalization factor $\beta$ to correctly
project into the actual space we can find $A$ in, more on that later.

$V\_m$ is an orthonormal basis of the Krylov space required.
$H\_m$ is a Hessenberg matrix. Both are of small size and we can finally get to understand this ominous parameter $m$.
It denotes the number of iterations we need to run this process for to get a "good" result -- and also the matrix shapes
of $V, H$. 

Now, the iteration approximates the inverse computation via

$$\left(\lambda I - A\right)^{-1} \approx V\_m (\lambda I - H\_m)^{-1} e\_1$$ where $e\_i$ denotes the i-th basis vector
of $\mathcal{R}^{n}$.
It's a general eigenvalue problem being posed.
The form suggested is

$$A V\_m = V\_m H\_m + h\_{m+1, m} v\_{m+1} e\_m^{T}$$

The above approximation holds if the spectrum of $A$ is different than the one from $H\_m$
and in general if $\lambda \notin \sigma(A), \lambda \notin \sigma(H\_m)$.

Let's define a set

$$\mathcal{F} (A) := \left[x^\* A x: x \in \mathbf{C}^n, ||x|| = 1 \right]$$

which is crucial to show that the above considerations of $\lambda$'s spectrum hold.

Complex analysis gives us access to compute the matrix function via contour integral: Let $f$ be a function that's
analytic around the neighborhood of $\mathcal{F} (A)$: 

$$f(A) v = \frac{1}{2 \pi i} \int_\Gamma f(\lambda) (\lambda I - A)^{-1}v d\lambda$$

where $\Gamma$ is a contour surrounding $\mathcal{F} (A)$.

But wait, now we can identify terms from above and exchange them!

$$\frac{1}{2 \pi i} \int_\Gamma f(\lambda) V\_m (\lambda I - H\_m)^{-1} e\_1 d\lambda = V\_m f(H\_m) e\_1$$

Finally, we have an expression for our approximation!

$$f(A)v \approx V\_m f(H\_m) e\_1$$

If we assume $A$ to have properties such as positive definiteness and symmetry we can even realize the following:
$H\_m$ is a symmetric Hessenberg matrix and thus necessarily tridiagonal. Which is easily diagonalizable:
Hence we can write, without further ado:

$$f(A)v \approx V\_m Q^\* f(S\_m) Q e\_1$$

where $H\_m = Q^\* S\_m Q$ and $S\_m$ is a diagonal matrix containing the $m$ eigenvalues of $H\_m$, $Q$ containing
$H\_m$'s eigenvectors.


Here we've jumped quite a few steps -- I'll refer the interested reader to the elaborations mentioned in the end of this
article. Assumptions on symmetry are of course not very general. We're however allowed to do that in our
highly-symmetric case, as our discrete operator $\bar{\Delta}$ is so regular. 
This lets us exploit a subclass of Arnoldi processes, namely so-called Lanczos iterations where we can exploit this
symmetry when implementing the iteration. For numerical stability reasons we can always introduce some orthogonalization
procedure. Let's see what this finally looks like in practice.

#### Implementation

We'll be making use of Eigen here. Observe the specialized function calls
for this complex-type procedure (ie the calls to `adjoint()`).

```cpp
template <typename Float>
std::tuple<Eigen::MatrixX<Float>, Eigen::MatrixX<Float>, Float>
lanczos_L(const Eigen::SparseMatrix<Float> &L, const Eigen::VectorX<Float> &u,
          const uint32_t m) {

  const uint32_t n = L.rows();
  Eigen::MatrixX<Float> V = Eigen::MatrixX<Float>::Zero(n, m);
  Eigen::MatrixX<Float> T = Eigen::MatrixX<Float>::Zero(m, m);

  Float beta = u.norm();
  V.col(0) = u / beta;

  for (uint32_t j = 0; j < m - 1; j++) {
    Eigen::VectorX<Float> w = L * V.col(j);
    if (j > 0)
      w -= T(j - 1, j) * V.col(j - 1);
    T(j, j) = V.col(j).adjoint() * w;
    w -= T(j, j) * V.col(j);

    // Modified Gram-Schmidt orthogonalization
    for (uint32_t i = 0; i <= j; i++) {
      Float coeff = V.col(i).adjoint() * w;
      w.noalias() -= coeff * V.col(i);
    }
    T(j + 1, j) = w.norm();
    T(j, j + 1) = T(j + 1, j);
    // could use early stopping \approx beta
    V.col(j + 1) = w / T(j + 1, j);
  }
  return {V, T, beta};
}
```

This now gives us access to $H\_m, V\_m$. Nice. Finally, we can approximate our matrix function
for a given vector like so

```cpp
template <typename Float>
Eigen::VectorX<Float> expm_multiply(const Eigen::SparseMatrix<Float> &L,
                                    const Eigen::VectorX<Float> &u, Float t,
                                    const uint32_t m = 10) {
  const auto [V, T, beta] = lanczos_L(L, u, m);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Float>> es(T);
  Eigen::MatrixX<Float> exp_T =
      (es.eigenvectors() *
       (t * es.eigenvalues().array().abs())
           .unaryExpr([](Float x) { return std::exp(x); })
           .matrix()
           .asDiagonal() *
       es.eigenvectors().transpose());
  Eigen::VectorX<Float> e1 = Eigen::VectorX<Float>::Zero(T.rows());
  e1(0) = 1.0;
  return beta * V * exp_T * e1;
}

```

An unfortunately very dense snippet. The main action happens inside the chained action on the eigensolver:

```cpp
    (es.eigenvectors() *
       (t * es.eigenvalues().array().abs())
           .unaryExpr([](Float x) { return std::exp(x); })
           .matrix()
           .asDiagonal() *
       es.eigenvectors().transpose());
```

where we apply
$$Q \left(t \exp(S\_m) \right) Q^{T} $$

and then scale it back using our normalization factor $\beta$ computed during the Lanczos iteration before.
The exponential function application happens element-wise for the diagonal matrix $S$.

It's important to note that this function definition is not perf-ready: we allocate lots of memory for every
function call. Ideally, when calling this repeatedly (ie. to wiggle $u(t)$ to $u(t + \tau)$), we would
want the function to not present any obvious bottlenecks.


#### Recap

We saw that it's problematic to store functions of large matrices in memory. Especially if the function is a
sum of terms that "densify" our structure.

We've seen a procedure that can project the "interesting parts" of a sparse linear operator into a smaller subspace.
We then observed that this can be projected back into the space the original space, the function application
happening in between. This makes it feasible for us to find an approximation of the matrix function to finally do what 
we wanted, ie. twist and turn our solution vector from one time step to another. "Twisting and turning" here meaning
turning over $n$ orthogonal directions.


Next time we'll plug this into some larger structure to show that we can employ this procedure for actual simulations.
And create a few nice first films from it.
