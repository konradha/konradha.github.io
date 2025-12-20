---
title: "Balltrees"
date: 2025-12-19T13:28:21+01:00
draft: false
---

Physics usually has sparse and weird scales. From unstructured grids to irregular geometries to
long-range interactions there are different ideas to bring ML models to understand "point clouds"
with associated context and scales.

When solving the NLSE in 2+1 or 3+1 dimensions you might be well-served in refining your grid
to avoid catastrophically collapsing solitons. When running particle-in-cell codes, you build localized
data structures to keep "close elements close". [Barnes-Hut](https://www.cs.princeton.edu/courses/archive/fall03/cs126/assignments/barnes-hut.html)
is an algorithmic innovation to keep elements close as well.


I stumbled on [recent work](https://arxiv.org/abs/2502.17019) while browsing X as one does,
and got interested in the internals of an old idea newly formulated. The Erwin transformer tries to group
elements together that are "close" in some metric -- what's nice about this novel operator then, is that
you can set this up as a flat data structure highly adaptable to a diverse set of architectures and nice temporal as well as 
spatial locality.

Let's try to understand the algorithm to construct this data structure.
The implementation used in the paper can be found [here](https://github.com/maxxxzdn/balltree/blob/main/balltree/balltree.h).

Of note first from the paper is the word "perfect" describing the data structure: Perfect binary trees allow for 
more or less contiguous access throughout memory -- we can make use of heaps.

More qualitaively interesting is that you're not required to partition the entirety of the space but instead localize the data structure
to the clusters that _matter_.

Look at the below widget to see how the tree is constructed, graphically only. You recursively refine the distances of points until at
the lowest level you end with pairs that are represented by the "perfect" final level of the tree.
We have $2^{L+1}$ index slots for $L = \lceil \log_2 n \rceil$ levels assuming we have $n$ points on our domain.
<iframe src="/widgets/balltrees-embed.html"
        style="width:100%; height:420px; border:none; margin:1.5em 0;">
</iframe>

Let's carefully write down what the algortihm that's running here does. Again, we recurse through until the lowest level (two-node pairs),
group them and subsequently build up the tree. We recursive on the number of points n. 
```text
BUILD_BALLTREE(points):
    L ← ⌈log₂(n)⌉
    output ← array of 2^(L+1) index slots

    RECURSE(indices, level, out_pos):
        if level = L:
            output[out_pos]   ← indices[0]
            output[out_pos+1] ← indices[1] or indices[0]  # duplicate if single
            return

        dim ← dimension with max spread over indices
        left, right ← split indices at median along dim

        RECURSE(left,  level+1, out_pos)
        RECURSE(right, level+1, out_pos + 2^(L-level))

    RECURSE([0..n-1], 0, 0)
    return output
```


Now, let's compare this to what the cpp header tells us it's doing

```txt
build_tree(data, idx_array, output_indices, output_mask,
           idx_start, idx_end,      # range in idx_array
           output_start,            # write position in output
           num_features,
           current_level, max_level):

    if current_level = max_level:
        write pair to output_indices[output_start], set output_mask
        return

    split_dim ← find_split_dimension(data, idx_array, idx_start, idx_end, num_features)

    partition_node_indices(data, idx_array + idx_start, split_dim,
                           mid - idx_start,  # split position
                           num_features, num_points)

    parallel:
        build_tree(..., idx_start, mid, output_start, ..., current_level+1)
        build_tree(..., mid, idx_end, output_start + 2^(L-level), ..., current_level+1)
```

We can recurse in parallel, neat!

<iframe src="/widgets/balltree-multicore-embed.html"
        style="width:100%; height:420px; border:none; margin:1.5em 0;">
</iframe>

<p></p>

Due to the recursion it looks difficult to imagine how we can adapt this for an embarrassingly parallel
procedure. The word "perfect" however allows us to really push the frontier: We can lay out the recursion
as we know EXACTLY how many points there are. Sure, there might be divergences in execution and
lots of function or kernel calls which make our lives difficult. But let's see hwo we can implement this as well.

We will use Triton. Triton does not need an introduction. Spearheaded by OpenAI, it is one way to max your FLOPs
for the architectures you have at hand.

The C++ implementation recurses cleanly. At each level, partition points and spawn two child calls.
We cannot do that on device.
No call stack, no dynamic recursion. Instead, we flatten the recursion into a level-by-level loop and track segments explicitly.
THe main thing is that we maintain a list of active segments (start, end pairs into the index array).
At each level, compute splits for all segments in parallel, partition them, then double the segment count for the next level.

```python
for level in range(max_level):
    # kernel 1: compute split dimension + pivot for each segment
    _compute_split_kernel[n_segs,](data, indices, seg_starts, seg_ends, ...)

    # kernel 2: partition indices and split segments
    seg_starts, seg_ends = _partition_and_split(indices, pivots, ...)
```

Why do we use this sort of hybrid? Two reasons:
Finding the max-spread dimension requires iterating over dimensions and computing min/max per segment.
Irregular access patterns that don't map to standard Pytorch ops. Triton lets us write this as a single fused kernel with one thread block per segment.

On the other hand, rearranging indices around a pivot is fundamentally a sorting-like operation.
Pytorch's argsort is backed by CUB/Thrust radix sort AFAIK.
Yes, it's O(n \log n) per level instead of O(n) for true nth\_element. We're on acceptable sizes though where I imagine it difficult to
be faster than highly optimized CUB procedures that have grown for years.


```python
# cat balltree_triton/kernel.py
import torch
import triton
import triton.language as tl


def build_balltree_triton(data, batch_idx):
    device = data.device
    n, dim = data.shape
    dtype = data.dtype

    if n == 0:
        return torch.tensor([], device=device, dtype=torch.int64), torch.tensor(
            [], device=device, dtype=torch.bool
        )

    starts, ends, counts, max_levels, tree_offsets, tree_sizes, total_tree_size = (
        _get_batch_info(batch_idx, device)
    )
    n_batches = starts.shape[0]

    if n_batches == 0:
        return torch.tensor([], device=device, dtype=torch.int64), torch.tensor(
            [], device=device, dtype=torch.bool
        )

    global_max_level = max_levels.max().item()

    indices = torch.empty(n, device=device, dtype=torch.int64)
    for i in range(n_batches):
        s, e = starts[i].item(), ends[i].item()
        indices[s:e] = torch.arange(e - s, device=device, dtype=torch.int64)

    proj = torch.empty(n, device=device, dtype=dtype)

    seg_starts = starts.clone()
    seg_ends = ends.clone()
    seg_batch = torch.arange(n_batches, device=device, dtype=torch.int64)
    current_level = torch.zeros(n_batches, device=device, dtype=torch.int64)

    BLOCK = 256

    for level in range(global_max_level):
        n_segs = seg_starts.shape[0]
        if n_segs == 0:
            break

        seg_levels = current_level[seg_batch]
        active = (seg_levels < max_levels[seg_batch]).to(torch.int32)

        split_dims = torch.empty(n_segs, device=device, dtype=torch.int64)
        pivots = torch.empty(n_segs, device=device, dtype=dtype)

        _compute_split_kernel[(n_segs,)](
            data,
            indices,
            proj,
            split_dims,
            pivots,
            seg_starts,
            seg_ends,
            seg_batch,
            starts,
            n_segs,
            dim,
            BLOCK=BLOCK,
        )

        seg_starts, seg_ends, seg_batch = _partition_and_split(
            proj, indices, seg_starts, seg_ends, seg_batch, pivots, active
        )

        current_level += 1

    n_segs = seg_starts.shape[0]

    if n_segs == 0:
        return torch.zeros(
            total_tree_size, device=device, dtype=torch.int64
        ), torch.zeros(total_tree_size, device=device, dtype=torch.bool)

    seg_batch_sorted, batch_order = seg_batch.sort()
    leaf_ids = torch.zeros(n_segs, device=device, dtype=torch.int64)
    batch_counts = torch.bincount(seg_batch, minlength=n_batches)
    batch_offsets = torch.cat(
        [torch.zeros(1, device=device, dtype=torch.int64), batch_counts.cumsum(0)[:-1]]
    )
    positions_in_batch = (
        torch.arange(n_segs, device=device) - batch_offsets[seg_batch_sorted]
    )
    leaf_ids[batch_order] = positions_in_batch

    out_idx = torch.zeros(total_tree_size, device=device, dtype=torch.int64)
    out_mask = torch.zeros(total_tree_size, device=device, dtype=torch.bool)

    _write_leaves_kernel[(n_segs,)](
        indices,
        starts,
        tree_offsets,
        seg_starts,
        seg_ends,
        seg_batch,
        leaf_ids,
        out_idx,
        out_mask,
        n_segs,
    )

    return out_idx, out_mask
```

Processing: We compute per-batch metadata: where each batch starts/ends in the point array, how many levels each batch's tree needs,
and offsets into the output array. This runs once before the main loop and we fully rely on Pytorch functionality.
```python
def _get_batch_info(batch_idx, device):
    n = batch_idx.shape[0]
    if n == 0:
        empty = torch.tensor([], device=device, dtype=torch.int64)
        return empty, empty, empty, empty, empty, empty, 0

    batch_ids, inverse = torch.unique(batch_idx, sorted=True, return_inverse=True)
    n_batches = batch_ids.shape[0]
    counts = torch.bincount(inverse, minlength=n_batches)

    ends = counts.cumsum(0)
    starts = torch.cat([torch.zeros(1, device=device, dtype=torch.int64), ends[:-1]])

    max_levels = (counts.float().log2().ceil() - 1).clamp(min=0).long()
    n_leaves = (1 << max_levels).long()
    tree_sizes = n_leaves * 2

    tree_offsets = torch.cat(
        [torch.zeros(1, device=device, dtype=torch.int64), tree_sizes.cumsum(0)[:-1]]
    )
    total_tree_size = tree_sizes.sum().item()

    return starts, ends, counts, max_levels, tree_offsets, tree_sizes, total_tree_size
```

A segment is a contiguous slice of the index array representing points to be split. Each one gets one thread block.
The kernel does two passes over the segment's points:

Pass 1: Find split dimension. For each of the dim coordinate axes, compute min and max across all points in the segment.
Track which dimension has the largest spread. This is the balltree heuristic: split along the axis where points are most dispersed.

Pass 2: Project and compute pivot. Read the coordinate values along the chosen dimension, write them to proj
(used by the subsequent partitioning step), and accumulate their sum. The pivot is the mean of these values.

Median would give balanced splits by count; mean gives balanced splits by coordinate value.
For roughly uniform point distributions these coincide.
For clustered data the tree structure differs slightly, but the spatial coherence property is preserved.
The tiled loop lets one thread block handle arbitrarily large segments by processing `BLOCK` points at a time, accumulating partial min/max/sum across tiles.

```python
@triton.jit
def _compute_split_kernel(
    data_ptr,               # [n_total, dim] point coordinates
    indices_ptr,            # [n_total] current permutation of point indices
    proj_ptr,               # [n_total] output: projected values for partitioning
    split_dims_ptr,         # [n_segs] output: chosen split dimension per segment
    pivots_ptr,             # [n_segs] output: pivot value (mean) per segment
    seg_starts_ptr,         # [n_segs] segment start indices into indices_ptr
    seg_ends_ptr,           # [n_segs] segment end indices (exclusive)
    seg_batch_ptr,          # [n_segs] which batch each segment belongs to
    batch_data_starts_ptr,  # [n_batches] offset into data_ptr for each batch
    n_segs,
    dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    seg_id = tl.program_id(0)
    if seg_id >= n_segs:
        return

    start = tl.load(seg_starts_ptr + seg_id)
    end = tl.load(seg_ends_ptr + seg_id)
    n = end - start

    if n <= 1: # nothing to split
        tl.store(split_dims_ptr + seg_id, 0)
        tl.store(pivots_ptr + seg_id, 0.0)
        return

    batch_id = tl.load(seg_batch_ptr + seg_id)
    data_offset = tl.load(batch_data_starts_ptr + batch_id)

    first_idx = tl.load(indices_ptr + start)
    best_dim = 0
    best_spread = tl.load(data_ptr + (data_offset + first_idx) * dim).to(tl.float64)
    best_spread = best_spread - best_spread

    for d in range(dim):
        min_val: tl.float64 = 1e308
        max_val: tl.float64 = -1e308
        # tile over points in segment
        for block_start in range(0, n, BLOCK):
            offs = block_start + tl.arange(0, BLOCK)
            mask = offs < n
            local_idx = tl.load(indices_ptr + start + offs, mask=mask, other=0)
            vals = tl.load(
                data_ptr + (data_offset + local_idx) * dim + d, mask=mask, other=0.0
            ).to(tl.float64)
            min_val = tl.minimum(min_val, tl.min(tl.where(mask, vals, 1e308)))
            max_val = tl.maximum(max_val, tl.max(tl.where(mask, vals, -1e308)))
        spread = max_val - min_val
        if spread > best_spread:
            best_spread = spread
            best_dim = d

    # natural barrier between "passes"
    tl.store(split_dims_ptr + seg_id, best_dim)

    sum_val = best_spread - best_spread
    for block_start in range(0, n, BLOCK):
        offs = block_start + tl.arange(0, BLOCK)
        mask = offs < n
        local_idx = tl.load(indices_ptr + start + offs, mask=mask, other=0)
        vals = tl.load(
            data_ptr + (data_offset + local_idx) * dim + best_dim, mask=mask, other=0.0
        ).to(tl.float64)
        sum_val += tl.sum(tl.where(mask, vals, 0.0))
        tl.store(proj_ptr + start + offs, vals, mask=mask)

    pivot = sum_val / n.to(tl.float64)
    tl.store(pivots_ptr + seg_id, pivot)
```





This part now uses pure Pytorch. We rearrange indices so that within each segment, points below the pivot come before points above.
This is a segmented sort: we construct a composite key (segment\_id, proj\_value) and call argsort.
Segments that are already singletons or belong to batches that have finished subdividing (tracked by `active`) get passed through unchanged. The `valid` mask filters out empty segments.
Yes, this is $O(n \log n)$ but again, we assume that these procedures optimized over years will hide huge latency spikes ...
Maybe I'll revisit this later.

```python
def _partition_and_split(
    proj, indices, seg_starts, seg_ends, seg_batch, pivots, active
):
    device = proj.device
    n = proj.shape[0]
    n_segs = seg_starts.shape[0]

    if n == 0 or n_segs == 0:
        empty = torch.tensor([], device=device, dtype=torch.int64)
        return empty, empty, empty

    active_bool = active.bool()
    seg_lengths = seg_ends - seg_starts

    point_seg = (
        torch.bucketize(torch.arange(n, device=device), seg_starts, right=True) - 1
    )
    point_seg = point_seg.clamp(0, n_segs - 1)

    max_proj = proj.abs().max() + 1
    sort_key = point_seg.double() * max_proj * 2 + proj.double()
    sorted_order = sort_key.argsort()

    new_indices = indices[sorted_order]
    indices.copy_(new_indices)

    left_counts = seg_lengths // 2
    left_counts = torch.where(active_bool & (seg_lengths > 1), left_counts, seg_lengths)
    left_counts = torch.clamp(left_counts, min=1)
    left_counts = torch.where(seg_lengths <= 1, seg_lengths, left_counts)

    mids = seg_starts + left_counts

    n_out = n_segs * 2
    new_starts = torch.empty(n_out, device=device, dtype=torch.int64)
    new_ends = torch.empty(n_out, device=device, dtype=torch.int64)
    new_batch = torch.empty(n_out, device=device, dtype=torch.int64)

    should_split = active_bool & (seg_lengths > 1)

    new_starts[0::2] = seg_starts
    new_ends[0::2] = torch.where(should_split, mids, seg_ends)
    new_starts[1::2] = torch.where(should_split, mids, seg_ends)
    new_ends[1::2] = seg_ends
    new_batch[0::2] = seg_batch
    new_batch[1::2] = seg_batch

    valid = (new_ends - new_starts) > 0
    return new_starts[valid], new_ends[valid], new_batch[valid]
```

Now, one thread per leaf segment: At termination, each segment contains 1-2 points.
This kernel writes them into the output heap at the correct position. Single points get duplicated; pairs get written directly.
The heap layout means `leaf\_id * 2` gives the output position. We omit any ludicrous pointer chasing.

```python
@triton.jit
def _write_leaves_kernel(
    indices_ptr,            # [n_total] final permutation of point indices
    batch_data_starts_ptr,  # [n_batches] offset into global point array per batch
    tree_offsets_ptr,       # [n_batches] offset into output array per batch's tree
    seg_starts_ptr,         # [n_segs] segment start indices
    seg_ends_ptr,           # [n_segs] segment end indices
    seg_batch_ptr,          # [n_segs] which batch each segment belongs to
    leaf_ids_ptr,           # [n_segs] position of this leaf within its batch's tree
    out_idx_ptr,            # output: [total_tree_size] point indices in heap layout
    out_mask_ptr,           # output: [total_tree_size] validity mask (False = duplicate)
    n_segs,
):
    seg_id = tl.program_id(0)
    if seg_id >= n_segs:
        return

    start = tl.load(seg_starts_ptr + seg_id)
    end = tl.load(seg_ends_ptr + seg_id)
    n = end - start
    batch = tl.load(seg_batch_ptr + seg_id)
    leaf_id = tl.load(leaf_ids_ptr + seg_id)

    # compute where this leaf's pair belongs in the output heap
    data_offset = tl.load(batch_data_starts_ptr + batch)
    tree_offset = tl.load(tree_offsets_ptr + batch)
    out_pos = tree_offset + leaf_id * 2 # exactly two slots per leaf

    if n == 0:
        tl.store(out_idx_ptr + out_pos, 0)
        tl.store(out_idx_ptr + out_pos + 1, 0)
        tl.store(out_mask_ptr + out_pos, False)
        tl.store(out_mask_ptr + out_pos + 1, False)
    elif n == 1:
        # single point, need duplication
        local_idx = tl.load(indices_ptr + start)
        gidx = data_offset + local_idx
        tl.store(out_idx_ptr + out_pos, gidx)
        tl.store(out_idx_ptr + out_pos + 1, gidx)
        tl.store(out_mask_ptr + out_pos, True)
        tl.store(out_mask_ptr + out_pos + 1, False)
    else:
        # two or more points 
        li0 = tl.load(indices_ptr + start)
        li1 = tl.load(indices_ptr + start + 1)
        tl.store(out_idx_ptr + out_pos, data_offset + li0)
        tl.store(out_idx_ptr + out_pos + 1, data_offset + li1)
        tl.store(out_mask_ptr + out_pos, True)
        tl.store(out_mask_ptr + out_pos + 1, True)
```

The output is a flat array in heap layout: leaf i occupies slots 2i and 2i+1. The
`out_mask` distinguishes real points from padding.
When a segment has a single point, we duplicate it to maintain the pair structure but mark the duplicate as invalid.


The "perfect" binary tree isn't just an aesthetic choice — it's what makes GPU execution tractable.
Consider the alternative: variable-size neighborhoods. Point A has 12 neighbors, point B has 47, point C has 3.
To process these in parallel, you either pad everything to 47 (wasting compute) or use sparse formats. Neither scales.

A perfect binary tree sidesteps this entirely. At level $l$, every node covers exactly $2^l$ points.
When you need to gather data from "nearby points," the tree tells you exactly which indices to fetch. The count is identical across all queries at that level.

This uniformity lets you pack operations into dense rectangular tensors. The tree construction pays $O(n \log n)$
upfront so that all subsequent operations can be embarrassingly parallel with fixed-size workloads.
BUT: Samples in a batch should have similar point counts. A 1024-point cloud and a 65536-point cloud produce trees with different depths (10 vs 16 levels).
You can pad the smaller tree, but then you're back to wasting compute.


<iframe src="/widgets/balltree-triton-embed.html"
        style="width:100%; height:520px; border:none; margin:1.5em 0;">
</iframe>


### Scaling

Let us run this finally. This has thankfully been possible due to $company's help on modern
RTX 6000 Pro Blackwell Server cards which give us considerable access to compute.

```bash
# status is a correctness check comparing the distance and node existence of both created trees
===========================================================
batch benchmark (n=16384, dim=3)
===========================================================
 batches             impl         time    speedup   status  
-----------------------------------------------------------
       1       cpu/cython      18.10ms      1.00x        ✓  
       1       gpu/triton       7.63ms      2.37x        ✓  
       8       cpu/cython      13.84ms      1.00x        ✓  
       8       gpu/triton       6.14ms      2.26x        ✓  
      64       cpu/cython      16.14ms      1.00x        ✓  
      64       gpu/triton       6.36ms      2.54x        ✓  

===========================================================
dimension benchmark (n=16384, single batch)
===========================================================
     dim             impl         time    speedup   status  
-----------------------------------------------------------
       2       cpu/cython      16.81ms      1.00x        ✓  
       2       gpu/triton       7.47ms      2.25x        ✓  
       3       cpu/cython      23.37ms      1.00x        ✓  
       3       gpu/triton       7.65ms      3.06x        ✓  
       4       cpu/cython      17.03ms      1.00x        ✓  
       4       gpu/triton       7.79ms      2.19x        ✓  
```

Due to the symmetry of the sampling process, all results match exactly: It's hwoever easy to construct cases
where our two different implementations will NOT yield the same result -- again, I'll investigate later
how to get the median choice as a pivot point going within Triton and/or Pytorch.

Overall though, preprocessing and thinking some more about how to make the kernels faster 

### Thread divergence


The balltree construction has a fundamental efficiency problem on GPUs that's worth understanding precisely.

At level $l$, we have $2^l$ segments, each containing $n / 2^l$ points.
Each segment gets one thread block (256 threads). The utilization at level $l$ is:

$$
\text{utilization}(l) = \min\left(1, \frac{n / 2^l}{256}\right)
$$

At the root (level 0): One segment with all $n$ points.
For $n \geq 256$, we're at 100% utilization — though we're underusing parallelism since only one block is active.

At the leaves (level $L-1$ where $L = \lceil \log_2 n \rceil$): We have $2^{L-1}$
segments with ~2 points each. We launch $2^{L-1} \times 256$ threads to process $n$ points:

$$
\text{utilization}(L-1) = \frac{n}{2^{L-1} \times 256} = \frac{2}{256} \approx 0.78\%
$$

The _crossover point_ — where utilization drops below 50% — occurs when $n / 2^l < 128$:

$$
l_{\text{crossover}} = \log_2 n - 7
$$

| $n$ | depth $L$ | crossover level |
|-----|-----------|-----------------|
| 1,024 | 10 | 3 |
| 16,384 | 14 | 7 |
| 65,536 | 16 | 9 |
| 1,048,576 | 20 | 13 |

The last 7 levels are *always* below 50% utilization regardless of $n$. This is intrinsic to the recursive halving — late levels have many small segments, each paying the full overhead of a thread block.

This is why the host-only implementation remains competitive for small $n$: the recursive structure maps naturally to
CPU execution, while the Triton pays fixed overhead per segment.

For applications where the tree is built once per forward pass and then queried many times, this construction cost is amortized.
The queries themselves have much better parallelism since each query is independent.
