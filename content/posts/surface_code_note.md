---
title: "Minimum weight error correction in the surface code"
date: 2023-05-12T15:38:47+02:00
draft: false
---

I've gotten a little confused by different terminologies and complexities
involved in correcting errors in the surface code so I've rewritten the
MWPM-involving algorithm to better understand it. A good reference (with
_fast_ accompanying library) is [^1].

Picture the surface code. Consider the following procedure to estimate
and correct the error in a given word. 


0. Generate the parity check matrix _H_ for one of the operators
    (X-type or Z-type) over mod 2.

1. Generate the so-called <span style="color:#89cff0">matching graph
    </span>: Vertices consist of Z-type
    syndromes, edges are the qubit connecting the syndromes, ie. vertices
    u, v are connected if and only if the qubit k involved in syndromes u
    and v flips. If a vertex is only associated to a single stabilizer, add
    a _boundary node_ and set its weight to 0. Additionally, attribute every
    edge with the qubit associated with said edge (boundary) (with distance,
    ie. weight, equal to zero).

2. Multiply _H_ with error word _e_ mod 2, get all your syndromes.

3. Run all-pairs shortest paths on the <span style="color:#89cff0">
    matching graph</span>, we receive the
    so-called <span style="color:#90EE90">path graph</span>
    (boundary included).

4. Create a <span style="color:#FFCCCB">syndrome graph</span> 
    in the following way: For every defect
    generate a complete graph. The edges of this graph are the distances
    we receive from the <span style="color:#90EE90">path graph</span>
    connecting the syndromes. To correct
    an error, we need to match up syndromes in a pairwise manner. The
    <span style="color:#FFCCCB">syndrome graph</span> needs to have an
    even number of vertices for us to 
    be able to find a perfect matching. Hence, if the number of non-zero
    syndromes is uneven, assign the boundary node from the matching graph
    as another another vertex here.
    Run any Minimum Weight Perfect Matching algorithm here[^2].

5. From the perfect matching in the syndrome graph we know which syndromes
    need to be connected. We can extract their respective paths from the
    <span style="color:#90EE90">path graph</span>. Now we can subsequently
    flip all the qubit in this path and do this for all syndromes that have
    been matched. 

6. Combine X-type and Z-type error correction if needed.


_Note (June 28 2023)_ Apparently we're solving the 
[Chinese Postman Problem](https://en.wikipedia.org/wiki/Chinese_postman_problem#T-joins) here.
I'll investigate this further.


[^1]: [PyMatching](https://arxiv.org/abs/2105.13082)
[^2]: [Edmonds algorithm, Blossom](https://en.wikipedia.org/wiki/Blossom_algorithm)
