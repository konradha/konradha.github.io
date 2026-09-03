---
title: "Tradeoffs down to MXFP4: Mixed precision on Metal 4"
date: 2026-09-03T13:36:39+02:00
draft: true
author: Konrad
---

Apple keeps teasing capabilities with little library support and black boxes of hardware software integrations.
A lot of people would probably immediately agree that the FLOPs/W number Apple's M-series achieve is incredible. I want to make use of it, explore it, exploit it.
Especially the unified CPU <> GPU memory. The main blocker has been documentation for a lot of folks, I think. Let us look through the capabilities
current Xcode and macOS betas give us access to -- I do not recommend doing this from scratch if you do not already want to load betas with some detours,
it feels like a long waste of time to get your hands
on these betas. I have explored the following on a simple Macbook Air with an M3 chip, 2024 AD (precisely: MacOS: 27.0, Xcode: 27 beta 6). 

Well. [Looky here](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) -- the MSL 4.1 docs. With some digging you run into this quote

```text
For example, you can create a tensor whose primary data plane is 4-bit floating-point organized
into blocks of 32 elements, with an auxiliary scale plane of 8-bit floating-point values (see
Figure 4)
```

Huh. Funny! Without too much foreshadowing: The grouped tensor packing is a templated alias.
It does look like there is not general HW support for lower precision data types or grouped tensor types. More on this later.


The good news is there is support for native packed formats in general. And contractions such as a GEMM `matmul2d`, more precisely, 
`mpp::tensor_ops::matmul2d` importing `Metal.hpp`. I like these advancements a lot -- you get access to more natural primitives without having to hand-roll
your own datatypes or worse your own runtime that calls into Objective-C or so.
On to the main subject. We are interested in numerics here. Numerics, numerics, numerics. Specifically: How can we make dense numerics scale using mixed precision?


# Matrix roots

In a [recent paper](https://arxiv.org/abs/2607.12430), Gao and friends show a simple application of mixed precision numerics. They show that without loss
in precision you can switch to single-precision for their problem setting while keeping double precision accuracy! And get some performance gains. Naturally I wondered
how much lower we can go (half precision? grouped fp4?) while keeping our algorithm correct.

We are interested in matrix square roots here. They go into p-th roots in their paper, we will only care about the simple case. A matrix square root
is the matrix $X$ such that $X^2 = A$ for a matrix $A$. $A$'s spectrum must have all its values on the positive real axis otherwise we are ill-defined here.



In general people use Schur's algorithm to solve for the matrix root. Building this for general $A$ is quite an involved task so we will limit ourselves to SPD matrices in 
this exploration. Using SPD matrices immediately gives you nice access to what our solution would look like. Indeed, for an SPD matrix $A$ you can write

$$ A = Q \Lambda Q^T = Q \text{diag}\left(s_i^2 \right) Q^T $$ where $s_i^2$ denotes positive values where $Q$'s columns are orthogonal. Whence: Define

$$S = \Lambda^{1/2} = \text{diag}\left(s_i \right)$$.

Let $X_k$ be the current estimate for $X$. Then define the residual $ R_k := A - X_k^2$. An exact Newton step solves: $R_k = X_k \Delta_k + \Delta_k X_k $.
Gao and friends freeze this to an initial operator. Define

$$\mathcal{L}_0 \left(E \right) := X_0 E + E X_0 $$

then one refinement step is $\mathcal{L}_0 \left(\Delta_k \right) = R_k$ and $X_{k+1} = X_{k} + \Delta_k$.
Under our easy SPD assumption we have: $X_0 = Q S Q^T$ and thus (please refer to the paper's derivations) 

$$ G_k := Q^T R_k Q, \quad \quad \quad       (C_k)_{ij} := \frac{(G_k)_{ij}}{s_i + s_j}, \quad \quad \quad         \Delta_k := Q C_k Q^T$$ where all $n^2$ reciprocals $s_i + s_j$ are
precomputed in FP32. A thing that should be immediately noted is that the MSL-speaking GPU does not support native double precision. Hence our baseline
has to be implemented entirely using Apple's highly optimized Accelerate library containing generalized Schur implementations.

Now, for real mixed precision we need to encode ("cast") our values into the lower precision realm. What does the correction look like in that case?
Due to rounding effects in encoding and decoding stages, we cannot assume all cols of $Q$ to be pairwise orthogonal anymore. Let $\hat{Q}$ be the basis
supplied to the product kernels then. Let $\hat{P}$ be an approximation to $Q^{-1}$ and let $\hat{s}$ be the decoded spectrum.
Define 

$$ \hat{H}_{ij} := \frac{1}{\hat{s_i} + \hat{s_j}}, \quad \quad \quad      \text{sym} \left(Y \right) := \frac{Y+Y^T}{2}$$.

We can the define the fixed linear correction as

$$X_{k+1} = X_k + \mathcal{\hat{M}}(R_k), \quad \quad \quad \mathcal{\hat{M}}(X) := \text{sym} \left( \hat{Q} \left[ \hat{P}R\hat{Q} \circ \hat{H} \right] \right) $$   

where $\circ$ denotes elementwise multiplication. That is all a mouthful but again, please inspect the paper by Gao and Kressner for my information.
Important to note is that implicitly we encode and decode information up and down. Product roundoff is present in every measurement.

Input matrix, iterate and residual use as "working precision" $y = y_{\text{lo}} + y_{\text{hi}}$ with $y_{\text{lo}}, y_{\text{hi}} \in \text{FP}32$.
Again, we do not have access to native double precision in MSL.
The solver condition is met when

$$\frac{||A-X_k^2||_F}{||A||_F} \leq \tau$$ where we set $\tau = 10^{-12}$ and $k \in \{1,... 30\}$. Note though that $\tau$ might be dependent on your current-stage
precision ... be aware.

Further, one can show that $\kappa \left( \mathcal{L}_X \right) = \sqrt{\kappa_2 (A)}$ making small eigenvalues a sensitive feature of the correction step.

# Data types and Implementation 
There are several nice scalar datatypes we have native access to on MSL: $\text{FP}32$, $\text{FP}16$, $\text{BF}16$. Then there are two novelties:
First, E4M3 allow for a single sign bit, 4 exponents and a mantissa -- AND they are packed into a single precision lane. Similarly, E5M2 which allow for more range.
Then there are the OCI's innovations, MXFP8 and MXFP4 that pack 32 values together (E4M3 and E2M1 respectively) and a single UE8M0 scale. Natively present in
NVIDIA's Blackwell arch or in recent AMD datacenter architectures, they allow for tensors to be operated on assuming close values function on similar scales and regions. 

An MX tensor has separate data and scale planes. Its scale plane uses `MTLTensorPlaneTypeScales`, UE8M0 values, and a block factor of 32.
The scale plane adds 8/32 = 0.25 bit per value. The logical traffic rate for an encode-decode roundtrip (it is casting but more nuanced than a constant number of instructions
when going from `int32\_t` to `float`... unfortunately): $\frac{8N + 2B_{\text{stored}}}{t_{\text{GPU}}}$ 
where $N$ denotes the number of transferred values, and $B_{\text{stored}}$ is the encoded payload in bytes.

While I did take a lot of pride in making good-looking plots using matplotlib we live in fortunate times in which we can make our agent (in this case, Sol-5.6-high)
give us something visually pleasing to digest. Consider the following:

<img src="/plots/codec-accuracy.png">

This plot is a result of me testing whether, depending on different distributions of the tensors / payload, traffic patterns would change. They in fact did not under different
assumptions, hence we present a single plot. I alluded to this earlier: It does not look like there is anything in the hardware letting us do grouped tensor ops
in an accelerated fashion. Apart from packed floats. Max DRAM <> GPU bandwidth is spec'd as 100 GB/s so I think we are on a good pattern here.

Let us first look at a reproduction experiment. Again, we need to bridge the capability gap that is missing native double precision datatypes on the Apple GPU.

<img src="/plots/baseline-scaling.png">

The native FP64 Schur decomposition we can call into with Accelerate is a tiny bit slower. Baking our SPD-aware algorithm with LAPACK exposed with Accelerate on the
other hand makes for a faster-than-GPU scaling. Interestingly, I did not manage to beat it. Scaling would suggest there to be some crossover though ... to be experimented with.
The plots are all running the procedure described above. We will jump into more detail on this now.

# Study














<!--- have strong conviction on the capacities and strengths of the lower-precision machinery that has been making waves throughout
the world with the seemingly unlimited datacenter buildouts. The industries privvy to supercomputing capabilities from the pre-LLM era have not had
the time to adapt their workloads to the manifold capability jumps we have had. Concretely: Assume you are an industrial giant with lots of physically moving
parts but also large budget allocations for simulation and prediction workflows. The software running this is at best running some double precision
procedures on single GPUs. What's more: It is probably written in FORTRAN or using Perforce. Imagine the pain to make this something using modern sparse
matrix methods, using mixed precision, using multi-cluster setups ... but before that you will have to fix a bunch of different issues. Anyway.
---!>
