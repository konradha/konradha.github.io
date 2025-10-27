---
title: "On the necessity of removing the 𝚏𝚞𝚗𝚌𝚝𝚒𝚘𝚗𝚊𝚕 requirement in traceable functional collectives"
date: 2025-10-26T19:45:36+01:00
draft: False
---

Recent Pytorch-related [releases](https://pytorch.org/blog/torchcomms/)
might make some of the discussion here
irrelevant. I do think that people -- as long as the experimental features recently
published aren't in Pytorch main -- will keep relying on Pytorch's compilation
capabilities (e.g. vLLM).

Overall, this post is me trying to get some ideas across and present my current understanding.
Happy to receive feedback on where my understanding is off or what can be investigated further!

There is ample amount of content to deep-dive into the Pytorch codebase. We will not assume
proficiency but overall familiarity with Pytorch and compilation here.

Big thank you to Tal Ben-Nun for mentoring and guidance for this project.

#### Introduction
I've recently opened a [PR](https://github.com/pytorch/pytorch/pull/161213)
to allow for Pytorch's compilation pipeline to understand pointwise communications.

Curious, one might say, as most operators we pass through (either forwards or
during autodiff) are collectives, so why would you want this specific feature?
Different applications
come to mind. I will offer two applications that are often used in HPC and ML.


1. [Halo exchanges](https://wgropp.cs.illinois.edu/courses/cs598-s15/lectures/lecture25.pdf) 

2. [RingAttention](https://arxiv.org/abs/2310.01889) & friends. 


#### Current state
One could argue that using NCCL's [SymmetricMemory](https://dev-discuss.pytorch.org/t/pytorch-symmetricmemory-harnessing-nvlink-programmability-with-ease/2798)
feature might be sufficient
We want to squeeze any and all overhead and want to enable
optimizations allowing for the scheduler infrastructure and [cudagraphs](https://dev-discuss.pytorch.org/t/understanding-cudagraph-trees/1967/2)
to capture all paths that lead to lower walltime. Exploiting Pytorch's compilation
pipeline fully is for me then the proper channel to do that.

There's been a considerable push for functionalization in Pytorch's infrastructure to reduce
complexity when reasoning about the different compilation stages, e.g. [here](https://github.com/vllm-project/vllm/issues/14703) or [here](https://dev-discuss.pytorch.org/t/functionalization-in-pytorch-everything-you-wanted-to-know/965). To that end, the collectives have been designed
to be fully functional, see the [Traceable functional collectives design document](https://docs.google.com/document/d/1Jqa68gvuVeFWZJFOiukmb58jAaUEET1GVMkd1GOMRT4).

However, here's an immediate issue. Consider the following function.

```python
def f(x):
  return torch.ops._c10d_functional.all_reduce_coalesced_([x], "sum",)
```

As is per the usual Pytorchisms, the suffixed underscore indicates an in-place operation.
Even though this function call technically makes part of the traceable functional collectives
infrastructure, this will yield an error upon calling (any option) of `torch.compile` on it.
(Technically you can call `torch.compile` on it but any subsequent execution of the compiled
function will yield an error).

```bash
RuntimeError: Found a custom (non-ATen) operator whose output has alias annotations: _c10d_functional::all_reduce_coalesced_(Tensor[](a!) inputs, str reduce_op, str group_name) -> Tensor[](a!). We only support functionalizing operators whose outputs do not have alias annotations
```

In CS speak, in-place ops induce side effects. Side effects notably are forbidden in functional
paradigms. I'm not entirely sure inductor's scheduler tracks down any collective call
to reduce the number of copies -- there is a small number of [issues](https://github.com/pytorch/pytorch/issues/134388)
suggesting otherwise.

#### My changes
In my [PR](https://github.com/pytorch/pytorch/pull/161213), I've sort of abused the 
existing infrastructure for collectives to allow for pointwise communications to be
traced using Dynamo. Notably, we construct temporary nodes in the graph for `P2POps`
which are often used together with `batch_isend_irecv` to issue coalesced pointwise
comms. The entire idea of this batching is to allow reducing overhead when creating new 
communicators (highly recommend looking inside `torch/csrc/distributed/c10d` and 
[e.g. this file](https://github.com/pytorch/pytorch/blob/main/torch/csrc/cuda/nccl.cpp) for 
deeper information).

We build `P2POpVariable`s for `P2POp`s that are verbosely passing the information to the
compiled graph such that this

```python
work = dist.batch_isend_irecv(
        [
          dist.P2POp(dist.isend, x, 0),
          dist.P2POp(dist.recv, y, 1)
        ])

some_work(x, z)

for w in work: w.wait()
```

can become something more akin to

```bash
tensors_to_wait = ops.batch_p2p_ops(
  ["isend", "irecv"],
  [0, 1],
  [x, y]
  )

some_work_compiled(x, z)

for t in tensors_to_wait:
    dist.wait_tensor(t)

```

In my PR, none of the ops appear in-place and we induce extra copies
to allow for the operations to take place, see in 
`torch.distributed._functional_collectives.py` 
(compare the notation to what the error before told us about in-place operations):

```python
"batch_p2p_ops(str[] op_list, int[] peer_list, int[] tag_list, Tensor[] tensors, str group_name) -> Tensor[]"
```

As a WIP this is fine -- to allow for efficient interleaving and reducing expensive
data movement we would want zero-copy comms though! I still want to change it to be
fully in-place -- as we've seen earlier, this is one of the reasons why we want to
relax the `functional` requirement in the traceable functional collectives infrastructure.

#### Changes inducing bugs

So far it's great, we can use this async comms feature now to do distributed convolutions
and whatever involves _blocking_ halo exchanges. What if we want to interleave
communication and computation?

Consider this code snippet:

```python
def kernel(x0, x1, y0, y1):
  r = dist.get_rank()
  w = dist.get_world_size()
  nxt = (r + 1) % w
  prv = (r - 1) % w

  work = dist.batch_isend_irecv([
      dist.P2POp(dist.isend, x0, nxt),
      dist.P2POp(dist.irecv, y0, prv),
  ])
  t0 = x0 * 2 + 1
  for ww in work: ww.wait()
  a = y0 + t0

  work = dist.batch_isend_irecv([
      dist.P2POp(dist.isend, a, nxt),
      dist.P2POp(dist.irecv, y1, prv),
  ])
  t1 = a * 1.000244140625
  for ww in work: ww.wait()
  out = y1 + t1
  return out
```

It's unfortunately a little involved but it'll show us: The current implementation
fails to correctly take care of dependencies wrt. data flow.
Comparing eager and compiled runs of this function will yield garbage differences
indicating something is off.

Let's investigate:

```bash
TORCH_COMPILE_DEBUG=1 \
  TORCH_LOGS=output_code \
  TORCHINDUCTOR_CACHE_DIR=$PWD/inductor_cache/$RANK \
  TRITON_CACHE_DIR=$PWD/triton_cache/$RANK \
  torchrun --nproc-per-node=4 --standalone tester.py
# tester contains the above snippet and some numerics checks
```

Inspecting the inductor cache (here, explicitly set in the flags,
`inductor_cache/4y/c4yjdzyq3kj4da7e2xi5gnbvzd4laxipv66ohspyo4nuecce37pv.py`):

```python
buf0 = _c10d_functional.batch_p2p_ops(['isend','irecv'], [2,0], [0,0], [arg0_1, arg1_1], '0')
buf1 = buf0[0]
buf2 = buf0[1]
wait_tensor(buf1); wait_tensor(buf2)
```

```python
triton_poi_fused_add_mul_0.run(arg1_1, arg0_1, arg2_1, buf7, buf15, 1048576, stream=stream1)
```

and only after the kernel call, it reads

```python
buf8 = _c10d_functional.batch_p2p_ops(['isend','irecv'], [2,0], [0,0], [buf7, arg2_1], '0')
buf9 = buf8[0]; buf10 = buf8[1]
wait_tensor(buf9); wait_tensor(buf10)
```

Verdict: Inductor hoisted calls to a wrong argument. Inductor does not carry dependency correctly.
Surely, we can try and track down the wrongly-assigned dependencies here.
Or: Would making TFC allow for side effects be a smart choice to allow for Inductor to do it?

We would get zero-copy `all_reduce` and my newly-introduced pointwise
comms would compile much more easily.
