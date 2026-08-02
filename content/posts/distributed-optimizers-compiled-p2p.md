---
title: "Distributed optimizers on compiled point-to-point comms"
date: 2026-08-02T18:00:00+02:00
draft: true
---

Zyphra's AMD pretraining paper ([Anthony et al., 2025](https://arxiv.org/abs/2511.17127)) has a section I keep thinking about.
Their ZeRO-1 setup flattens and concatenates all parameters, then cuts the flat buffer into contiguous shards — so shard boundaries
split at most two parameters per rank. Their Muon step needs the *full* gradient matrix for Newton–Schulz orthogonalization,
and the reference implementation (Kimi's Megatron-LM Muon) does a full AllGather to get it: 99% of optimizer memory was that
AllGather spike. Their fix is a neighbor SendRecv exchange (their Algorithm 1, rank-ordered to avoid deadlock): assemble only
the boundary-split parameters, reshape everything else in place.

Separately, I landed [pytorch#161213](https://github.com/pytorch/pytorch/pull/161213) earlier this year: `isend` / `irecv` /
`batch_isend_irecv` now compile through torch.compile (behind `TORCHDYNAMO_ENABLE_P2P_COMPILATION`), with the p2p ops
rewritten into functional collectives so inductor can see and schedule them.

The obvious project: reimplement the paper's SendRecv-Muon as a *compiled* optimizer step and see what the compiler does
with comm/compute overlap when the communication is visible in the graph instead of hidden behind graph breaks.

## The zoo

The interesting axis: what global structure does the update rule need under ZeRO-1 flat sharding?

- **Adam / AdamW** — elementwise. Nothing to assemble; shards are independent. The baseline.
- **Lion** — elementwise (sign update). Same story, cheaper state. Control case.
- **Muon** — needs the full 2D parameter matrix per Newton–Schulz step. Neighbor SendRecv suffices under flat sharding. The paper's case.
- **Shampoo** — needs full matrices *and* maintains left/right preconditioner factors with periodic eigendecomposition (or NS^(-1/4)). Assembly plus heavy state.
- **Distributed SOAP** — Shampoo in Adam's eigenbasis; the basis itself rotates over training. Cross-step coupling, the hardest of the set.

## Plan

1. ZeRO-1 flat-sharded AdamW baseline, compiled.
2. Muon with AllGather assembly (reference) vs. SendRecv assembly (paper's Algorithm 1), eager vs. compiled.
3. Measure: peak optimizer memory, step time, comm overlap quality in the inductor schedule.
4. Stretch: Shampoo, then SOAP, same harness.

More once there are numbers.
