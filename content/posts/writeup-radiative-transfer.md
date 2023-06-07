---
title: "Solving instances of radiative transfer using PINNs"
date: 2023-06-07T18:57:35+02:00
draft: true
---


This semester, I've taken a tremendous lecture on using ML to solve PDEs.
What an amazing thing. Instead of passing 3 semesters learning about approximation 
theory, test functions, finite elements, C++, function theory and then putting it
together to solve (seemingly!) trivial heat or wave equations, you can just take
your old reliable Pytorch and have Adam walk the space that minimizes a residuum you
define for a given PDE. And: you don't even need any data. Choosing your points in a
smart manner (ie. not sampling from some centered distribution), you can even deduce
very nice bounds on convergence and precision of the solution you arrive at. Tremendous.
Let's dive in.


### The setup -- Radiative Transfer PDE

### The background -- PINNs and Surrogate models

### Implementation -- Ease and bottlenecks; acceleration

### TL;DR
