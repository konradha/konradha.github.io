---
title: "Truncated SVD's missing bwd"
date: 2026-03-18T22:48:41+01:00
draft: true
---
For some applications, truncated eigensolves are an important
gate to differentiate through so I looked up what the physics literature has to offer. I found [Naumann's recent paper](https://arxiv.org/abs/2511.14651) 
and liked the style it's written in. Looking into it I found that there really is no complete implementation of the exact truncated backward through the tSVD/tED gate.
Multiple or near-multiple eigenvalues are also often treated only approximately. Degenerate spectra are important in physics though, so I want to
jump into this here. Don't worry, there are some LLM applications as well so there's an _actual_ use for $current\_year technology.

At the core sits

$$
F_{ij} = \frac{1}{\sigma_j^2 - \sigma_i^2}, \quad i \neq j
$$

which diverges when two singular values approach each other. The map $A \mapsto U$ is non-differentiable at eigenvalue coalescence.
Any rotation within the degenerate subspace is an equally valid choice of singular vectors.
Individual sorted eigenvalues can be nonsmooth at crossings, but symmetric quantities on isolated clusters remain smooth ([Gravesen, Evgrafov & Nguyen, 2011](https://backend.orbit.dtu.dk/ws/files/127276986/multieig_final.pdf)).

An AD pattern I've seen is to compute the full SVD and slice to $k$ columns, which can miss discarded-spectrum contributions.
In the regimes tested here, this can produce a relative gradient error.
The correction requires solving a Sylvester equation coupling the kept and discarded subspaces ([Francuz, Schuch & Vanhecke, 2025](https://arxiv.org/abs/2311.11894)).
Some of it is derived on paper and not fully written in LaTeX. For now I just
wanted to implement and test whether this works at all.

## Projector decoupling

The exact truncated pullback needs information about the discarded subspace. [Naumann](https://arxiv.org/abs/2511.14651) introduces projectors: write $P_U = I - UU^\top$, $P_V = I - VV^\top$ and represent everything discarded through $P_U A$ and $AP_V$. Decompose the differential:

$$
dU = UK + dU_2, \qquad dV = VL + dV_2
$$

$K, L$ are skew-symmetric. $dU_2, dV_2$ live in the orthogonal complement, the part that couples kept to discarded. Projecting $dA$:

$$
U^\top dA\,V = d\Sigma + K\Sigma + \Sigma L^\top
$$

This is closed on the kept block. That's where $F$ appears and where degeneracy causes trouble. The cross-subspace part,

$$
P_U\,dA\,V = dU_2\,\Sigma + P_U A\,dV_2
$$

couples the two worlds through a Sylvester system. No discarded bases needed, just $P_U A$ and $AP_V$.

In reverse mode the kept-space core becomes

$$
\Omega = F \odot \bigl((R - R^\top)\Sigma + \Sigma(T - T^\top)\bigr)
$$

with $R = U^\top\bar U$, $T = V^\top\bar V$, and

$$
\bar A_0 = U\bigl(\mathrm{diag}(\bar s) + \Omega\bigr)V^\top
$$

The cross-subspace correction adds a projected Sylvester solve. For $n \le m$:

$$
\mathcal{S}_A(Z) = Z\Sigma^2 - AA^\top Z, \qquad Z \in \operatorname{range}(P_U)
$$

$$
\mathcal{S}_A(Z) = P_U\bar U + AP_V\bar V\,\Sigma^{-1}, \qquad Z \leftarrow P_U Z
$$

$$
\bar A = \bar A_0 + Z\Sigma V^\top + UZ^\top AP_V + U\Sigma^{-1}(P_V\bar V)^\top
$$

The identity $\langle\bar U, dU\rangle + \langle\bar s, ds\rangle + \langle\bar V^h, dV^h\rangle = \langle\bar A, dA\rangle$ is a necessary check, but not sufficient on its own. I also check projector finite differences on $dP_U$ and $dP_V$.


## Handling degeneracies

The exact $F = 1/d$ blows up when kept singular values collide. Some regularization strategies:

**Lorentzian**: $F = d/(d^2 + \varepsilon^2)$. Smooth suppression to zero at degeneracy. A physics resolvent approach, used e.g. in [Francuz et al.](https://arxiv.org/abs/2311.11894)

**Freeze**: Set $F = 0$ wherever $|d| < \tau$.

**Taylor**: Near degeneracies, replace $1/d$ with a geometric series $\sum_{j=0}^{K-1} r^j / \sigma_i^2$, which stays finite as the ratio $\sigma_j/\sigma_i \to 1$. Exact $1/d$ elsewhere. This is what [Dobi-SVD](https://arxiv.org/abs/2502.02723) uses internally.


**Degpert**: Detect clusters of near-degenerate singular values, diagonalize $U_c^\top AA^\top U_c$ within each cluster to lift the degeneracy with effective eigenvalues. Set $F = 0$ within clusters, use effective gaps between them. This is degenerate perturbation theory, see [Kasim (2020)](https://arxiv.org/abs/2011.04366) for the EVD analogue.

## Dobi-SVD

[Dobi-SVD](https://arxiv.org/abs/2502.02723) is a compression pipeline that learns truncation rank per layer via a smooth tanh gate, turning the discrete rank-selection problem into a differentiable one. It's the most complete recent attempt at making truncated SVD end-to-end trainable for LLM compression, which makes it the natural baseline here. They clamp tiny singular values, branch into a Taylor-style geometric series near small gaps, use inverse-gap structure elsewhere. The suggested pullback is

$$
G_{\mathrm{Dobi}} = U\bigl(\Omega_U\Sigma + \Sigma\Omega_V + \mathrm{diag}(\bar s)\bigr)V^\top + T_U + T_V
$$

where $\Omega_U, \Omega_V$ use their piecewise $E$ matrix and $T_U, T_V$ handle cross-subspace terms through $P_U(\bar U / \tilde S)V^\top$ and $U(\tilde S^{-1}\bar V^\top)P_V$. Those cross-subspace terms are scaled cotangents. In our analysis, this behaves like a low-rank surrogate of what the full Sylvester solve computes. What does this approximation actually cost in terms of gradient geometry?

I measured that cost on synthetic matrices and on actual GPT-2 weights, comparing Dobi against Lorentzian ($\varepsilon = 10^{-4}$), taylor ($\tau = 10^{-8}$, $K = 16$), degpert ($\tau = 10^{-6}$), and exact ($\varepsilon = 0$). For the synthetics I swept boundary gaps ($|\sigma_{k-1} - \sigma_k|$ from $10^{-1}$ down to $10^{-6}$) and near-zero kept spectra. Dobi's adjointness error stayed high across the entire gap range and degraded badly when kept singular values approached zero. Lorentzian and taylor tracked exact closely. Degpert handled interior degeneracies well but can't help at the truncation boundary since it only clusters within the kept spectrum.

## On actual GPT-2 layers

To check this isn't just synthetic, I ran the backward on weight matrices from all 12 GPT-2 transformer blocks — attention projections and MLP layers — picking truncation rank at 90% spectral energy per layer:

<img src="/plots/adj_activations.png">

Dobi sits at $10^{-1}$ to $10^{-2}$ adjointness error across the board. Degpert lands at $10^{-3}$ to $10^{-7}$. The other methods (lorentzian, freeze, taylor, exact) all overlap with degpert so I'm only showing two lines here. The gap is consistent across all layers and sublayer types.

## Decomposing the bias

Knowing the error is large isn't enough. Where does it go? Any gradient $G \in \mathbb{R}^{n \times m}$ decomposes into five orthogonal projector channels relative to the truncated bases:

- Kept diagonal: $U\,\mathrm{diag}(\cdot)\,V^\top$, the singular value part
- Kept tangent: off-diagonal within the kept block, within-subspace rotations
- Cross-left: $P_U(\cdot)VV^\top$, discarded left, kept right
- Cross-right: $UU^\top(\cdot)P_V$, kept left, discarded right
- Residual: $P_U(\cdot)P_V$, fully discarded

Project both the surrogate and exact gradient into each channel and measure the per-channel gain:

$$
\mathrm{gain}_c = \frac{\langle G_{\mathrm{sur},c},\, G_{\mathrm{exact},c}\rangle}{\|G_{\mathrm{exact},c}\|_F^2}
$$

Gain near one means the channel matches the exact gradient in magnitude and sign. If it's larger than one you amplify the gradient dir;
and in the opposed case you suppress it. This tells you not just *how much* a surrogate is off, but *where* in the matrix geometry the error concentrates.

The structure suggests which channel breaks first. Dobi's cross-subspace terms $T_U, T_V$ approximate the Sylvester coupling through scaled cotangents $\bar U / \tilde S$. When $\tilde S$ is small or the kept-to-discarded coupling is strong, that estimate overshoots, mostly in the cross-left channel $P_U(\cdot)VV^\top$, where discarded left singular vectors meet kept right ones. In our experiments, the other channels are much less affected because Dobi's within-subspace $E$ matrix handles them better. The error is concentrated rather than diffuse.

This has been written about in [Francuz, Schuch & Vanhecke (2025)](https://arxiv.org/abs/2311.11894), who showed that missing truncated-spectrum contributions create practical mismatch in tensor network AD. [Kasim (2020)](https://arxiv.org/abs/2011.04366) derives the analogous EVD correction with the same kept/cross split. [Naumann (2025)](https://arxiv.org/abs/2511.14651) formalizes the projector decoupling on the forward sensitivity side. The full-rank SVD derivative formulas go back to [Giles (2008)](https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf), with a clean derivation in [Townsend's note](https://j-towns.github.io/papers/svd-derivative.pdf). [Wang et al. (2021)](https://arxiv.org/abs/2104.03821) already predict the near-collision instability that makes stabilized surrogates tempting.
