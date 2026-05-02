---
title: "Model parallelism for continuous Fourier kernels"
date: 2026-05-02T17:38:40+02:00
draft: false
---

Continuous kernels you say? I am honestly not sure why they're called continuous:
Read [here](https://s2.smu.edu/~eclarson/pubs/2025_AIPR_Fourier.pdf). 
It's a technique about large kernels and how with a large number of stencil points added your training becomes essentially infeasible.
Hence, you have to introduce some sampling. Even better, you can pipe an MLP between input and kernel so you can learn what
the good positions in your kernels are. The CF-Conv method as we dub it here avoids evaluating the full continuous kernel
at every step. Instead it keeps a stateful spectral representation and refreshes a sparse subset of that every iteration.
Getting this right is fundamentally a systems problem. And: You can figure out ways to make this scalable and sharded across
several devices -- opening up kernel sizes previously unheard of!

Here's why I like this work:

> This [better understood nonlinearities inside the operators] would allow the network to fully utilize phase and amplitude information, making the approach particularly appealing for domains like audio, radar, and sonar image processing, where such information is critical.


## The operator

Let the input be

$$x \in \mathbb{R}^{B \times C_{\mathrm{in}} \times H \times W}.$$

For real-valued images, the implementation works with the non-redundant real
FFT half-spectrum

$$W_{\mathrm{r}} = \left\lfloor W/2 \right\rfloor + 1$$

and computes

$$X = \operatorname{rfft2}(x) \in \mathbb{C}^{B \times C_{\mathrm{in}} \times H \times W_{\mathrm{r}}}.$$

$\operatorname{rfft2}$ denotes Pytorch's 2d FFT here.

A CF-Conv layer stores a complex spectral kernel

$$\widetilde K^t \in \mathbb{C}^{H \times W_{\mathrm{r}} \times C_{\mathrm{in}} \times C_{\mathrm{out}}}.$$

At every frequency $(h,w)$, the slice

$$\widetilde K^t_{h,w} \in \mathbb{C}^{C_{\mathrm{in}} \times C_{\mathrm{out}}}$$

is a channel-transfer matrix. The layer applies one such matrix per Fourier
mode:

$$Y_{b,o,h,w} = \sum_{i=1}^{C_{\mathrm{in}}} X_{b,i,h,w}\,K^t_{+,h,w,i,o}.$$

Then it returns to real space:

$$y = \operatorname{irfft2}(Y).$$

Roughly, Cf-Conv is not a small spatial stencil but instead a grid of spectrally-indexed mixers.

The continuous part is the map that proposes spectral kernel entries. For

$$u=(h,w,i,o) \in [H]\times[W_{\mathrm{r}}]\times[C_{\mathrm{in}}]\times[C_{\mathrm{out}}],$$

the code evaluates a neural field

$$\Phi_\Theta(u) = \Phi^\Re_\Theta(z(u)) + i\,\Phi^\Im_\Theta(z(u)).$$

Here $z(u)$ is the normalized coordinate vector. The paper normalizes
coordinates to $[0,1]$; this implementation uses $[-1,1]$.

## The shadow kernel

Instead of regenerating every entry of the Fourier kernel on every step, sample coordinates $S_t$ and update only those:

$$K_+^t(u)=\widetilde K^t(u)+\mathbf{1}_{\{u\in S_t\}}\,\gamma_t(u)\left(\Phi_{\Theta_t}(u)-\widetilde K^t(u)\right).$$

For the baseline sparse EMA, $\gamma_t(u)=\alpha$ and entries not selected remain cached. Selected entries are the only ones that carry fresh gradient into $\Theta$ on that step.

After the optimizer step, the sampled values are committed back into $\widetilde K$. So the forward pass is not using the dense implicit kernel
$\Phi_{\Theta_t}$. Instead it is using a stochastic hybrid of a stale dense cache and a sparse fresh refresh.


## Where memory goes

The state buffer has

$$N_\Omega = H W_{\mathrm{r}} C_{\mathrm{in}} C_{\mathrm{out}}$$

complex entries. If each real component uses $b$ bytes, the raw state memory is

$$M_{\mathrm{state}} = 2 b H W_{\mathrm{r}} C_{\mathrm{in}} C_{\mathrm{out}}.$$

The factor of two is due to complex numbers being treated separately. The $W_{\mathrm{r}}$ denotes the
RFFT half-spectrum. The $C_{\mathrm{in}}C_{\mathrm{out}}$ might need to be sharded when you scale channel width.

This is exactly the limitation the CF-Conv paper leaves open: each layer still needs a stateful Fourier kernel of shape roughly
$H \times W \times C_{\mathrm{in}} \times C_{\mathrm{out}} \times 2$. While it does not remove the state itself, sparse refresh reduces MLP evaluation and gradient storage. 

Data parallelism does not solve that. In DDP you'd roughly end up with

$$M_{\mathrm{per\ GPU}}^{\mathrm{DDP}} = M_{\mathrm{state}}.$$

You get more samples per step, maybe better throughput, but the kernel still has
to fit on each GPU.
What we really want is to shard the kernel state itself.

Dryden and friends's [channel and filter parallelism work](https://www.ndryden.com/data/papers/sc2019-chanfilt.pdf) explores this in-depth.
The principle behind distributed convolutions translates somewhat: once channel/filter tensors dominate memory and communication, sample-parallel replication
is insufficient. You start diving your data along C,F axes and add sufficiently many collectives to keep the data in-sync.


## Filter parallelism

With $P_F$ filter shards,
rank $q$ stores

$$\widetilde K^{(q)} \in \mathbb{C}^{H \times W_{\mathrm{r}} \times C_{\mathrm{in}} \times (C_{\mathrm{out}}/P_F)}.$$

The per-rank state drops to

$$M_{\mathrm{state}}^{(q)} = \frac{1}{P_F}M_{\mathrm{state}}.$$

Each rank computes a disjoint output-channel slab:

$$Y^{(q)}_{b,o,h,w} = \sum_{i=1}^{C_{\mathrm{in}}} X_{b,i,h,w}K^{(q)}_{h,w,i,o}.$$


## A two-dimensional channel/filter mesh

The more complete decomposition shards both input and output channel axes. Let
the mesh be $P_C \times P_F$. Rank $(p,q)$ stores

$$\widetilde K^{(p,q)} \in \mathbb{C}^{H \times W_{\mathrm{r}} \times (C_{\mathrm{in}}/P_C) \times (C_{\mathrm{out}}/P_F)}.$$

Now the per-rank state is

$$M_{\mathrm{state}}^{(p,q)} = \frac{1}{P_C P_F}M_{\mathrm{state}}.$$

The forward contraction is local over the input-channel slice:

$$G^{(p,q)}_{b,o,h,w} = \sum_{i\in I_C^{(p)}} X_{b,i,h,w}K_{h,w,i,o}.$$

To finish the output-channel shard $q$, we sum over the input-channel mesh axis:

$$Y^{(q)}_{b,o,h,w} = \sum_{p=1}^{P_C}G^{(p,q)}_{b,o,h,w}.$$

Ie. an allreduce over the $C_{\mathrm{in}}$ axis. The backward path has the
dual communication over the output-filter axis so that input gradients see all
output-channel contributions.

## Measured memory frontier

<img src="/plots/cfconv-model-parallelism-mp-scaling.png">

_Synthetic training-step benchmark. Four ranks, image size $256\times256$, batch size 8 per rank, six CF-Conv blocks, $2^{18}$ sampled kernel positions.
Crosses mark configurations that did not finish successfully in the benchmark row._

Stateful MP=1 reaches the frontier quickly whereas larger and more ranks move it outward.
Shard the $C_{\mathrm{out}}$ side and the stateful kernel cost drops approximately with the number of shards.

For experiments I also added a K-free factorized variant that removes the shadow kernel and trains a low-rank spectral operator directly that
transfer nicely between devices. Of course, its properties on whether training can work similarly for this guy would need to be proven.


## Realizations

The moment you shard you shard the kernel, you also need to decide what the sampler is allowed to mean. A replicated run can draw one
global set $S_t \subset \Omega$. A filter-parallel run can instead draw local sets $S_t^{(q)}$ on each output-channel slab. Both choices can have the same
marginal sampling rate

$$\pi = k/N_\Omega,$$

but they are not the same stochastic process. Global fixed-size sampling couples all coordinates through one budget while ocal fixed-size sampling is closer to a
stratified law. There, each shard gets its own budget and the cross-shard covariance changes. For a linear cache update this mostly washes out in the first moment.
For training, I would not assume this is harmless.


This is also where the method becomes more interesting than "put the big tensor on many GPUs". A bad sharded sampler gives you memory relief but may quietly
change the estimator. A better one can use the mesh: keep fixed sampling density per shard, allocate more samples to shards with larger residuals, or draw a
shared global sample and route entries to their owner ranks.

What's unclear is understanding the shadow gap. You track it as
$$\|\widetilde K^t-\Phi_{\Theta_t}\|,$$

the variance of the sparse update, and whether local-stratified sampling behaves
like the single-device law once gradients and nonlinearities are involved. If
that turns out to matter, the model-parallel story is not merely scaling a
continuous Fourier kernel but instead could give you a way to an estimator-aware parallelism.


## References

- Harper, Wood et al.,
  [Scaling Continuous Kernels with Sparse Fourier Domain Learning](https://s2.smu.edu/~eclarson/pubs/2025_AIPR_Fourier.pdf).
- Dryden, Maruyama et al.,
  [Channel and Filter Parallelism for Large-Scale CNN Training](https://www.ndryden.com/data/papers/sc2019-chanfilt.pdf), SC 2019.
