---
title: "T=0 in a glassy system = ?"
date: 2024-03-11T11:19:54+01:00
draft: false
---

I've been researching a [lattice-based glassy system](https://arxiv.org/abs/2003.02872).
The authors of this work explain _a little_ how we find the ground state in the Hamiltonian

$$H = \sum_{i \in L^k} n\_i \left(\sum_{j \in \partial i} n\_j - l\_i\right)^2$$

at T=0 and k=3 with periodic boundary conditions. The terms n and l denote site occupation and
how many particles are ideal per particle type, respectively. We have no external field.
It assumes a fixed particle density ϱ with two types of particles, each with different
(fixed) density. The first type has l=3, ie. its energy is minimal if three out of its 
six neighbors are occupied. The second type has l=5.


Running a Monte Carlo simulation for this problem is quite a lenghty process. Of course,
the main thing to do is to use local moves. That is, choose a particle at random, choose
one of its neighboring sites and attempt to swap them using the Metropolis-Hastings
criterion. We can magically accelerate reaching an equilibrium by introducing so-called
swap moves: Instead of randomly choosing one of the first, randomly chosen particle's neighbors,
we now choose another random particle on the lattice and attempt, again a swap move using
the MH-criterion. Interestingly, this enhances equilibration and lets our system decorrelate
more rapidly. Depending on system size, dynamics and temperature, this "speedup" can enhance
our simulation by two to three orders of magnitude. That's not too shabby. It's a magical
enhancement as, for now, we're not quite sure yet how this works so nicely in glassy systems.

But even introducing sweeps over our entire grid (ie. L³ swap trials), the authors suggest
that to reach the ground state at T=0, we have to run the simulation for a huge number of sweeps.



I'm not satisfied with this approach of just abusing compute to get a somewhat minimal state.
Let's ask another question: Can we map the problem to some known structure and then
deduce smarter algorithms? I would sure hope that we can rely on decades of statistical 
physics research to yield solid intution with regards to the type of problem we're dealing with.

There are other connections to be found, though. Notably, we can find a correspondence between
graph partitioning and finding the ground state in this system. In the following I'll just mention
different notions that helped arrive at the heuristic to reach the ground state in our glassy
system. It's written as a stream of consciousness, don't expect every argument and intuition to be
formally correct.

### Diffusion
Let's first talk about the [heat equation](https://en.wikipedia.org/wiki/Heat_equation). The applications
of the idea of smoothing out hotter temperatures into nearby regions has manifold applications. 
Most importantly,it works without applying an external fields and just diffuses heat in the space
we apply our initial conditions in. But what happens when we look at the microscale? Looking at the
[Langevin equation](https://en.wikipedia.org/wiki/Langevin_equation) is arguably most instructive to
see why we choose Monte Carlo methods often times -- it allows us to traverse the free energy landscape
of a given system by assigning every particle in a system some acceleration to understand where it can move.

Well, what does equilibration mean in this context? In the continuous-space, continuous-time view,
we're looking at solving a PDE that fully characterizes the evolution of our system. The solution space here
is all minimal energies we can reach with the conditions and geometry we can find us in. Let's look at the 
discrete-space, regular domain case: Well, particles just move around at Manhattan distance of one another.
What length-scales they find themselves in before, during and after simulating this system, may stronlgy
inform us of what kind of problem we're actually dealing with.

Now, what good are these ideas for our current problem? What I've found during investigations of trying to
equilibrate the system above with local sweeps and swap MC is that we can almost always find regions in our
glassy lattice where locally, the average temperature of each particle is higher. Surely, this can tell us
which region is most likely to mix in the next k epochs compared to cooler regions. An idea is to guide the
system towards a certain kind of equilibration: Maybe we can put an emphasis of averaging the energy per
particle out, maybe it's a solid idea to just have locally-neighboring regions move particles locally?
Are we better off always trying to equilibrate the higher temperature regions or is it smart to try to keep
the regions of low temperature as small as possible to move more opportunistically in the free energy landscape?

What's visible is that, in our current system, there's a strong distinction between low and high temperatures.
Here, we define higher temperatures as inverse temperature between zero and 1 + some delta. No matter the kind
of equilibration you choose from a random initial configuration, this distinction will always be present in 
your simulation.

Let's gather some more ideas, keeping in mind our reflection on what heat does to a system.


### Constraint satisfaction problems (CSP) and k-SAT
What's a clause? What's constraint satisfaction?
We're now going into seemingly non-related fields. We're lucky that truthfulness is usually binary in mathematical
logic. Connecting two binary variables that can be either true or false is usually done using logical `and` and
logical `or`, written as ∧ and ∨ respectively. To negate, write ¬. A clause is a collection of binary variables
connected through these logical operators.
Why do we care? Well, you can map lots of problems in theoretical CS into this framework. We're interested in two
of them. CSP:  maximizing the number of satisfied clauses in a collection of clauses that are logically connected.
The other one is k-SAT: Satisfying a collection of fixed-size (k) clauses.
Looking at our initial Hamiltonian: Getting to energy 0 in this Hamiltonian is equivalent to saying that
every single particle has exactly the number of neighbors it prefers to have. If we can actually do this is part
of a more involved process of looking at crystalline phases of the particles and their mixtures at different rates.
What we can do is upper-bound the number of configurations that could potentially satisfy the Hamiltonian up until
a certain energy threshold $$E\_t \gt 0$$

Both CSP and k-SAT are _hard_, both are in NP. We approximate close-to-optimal solutions: People usually introduce
energy functions that have to be minimized using Monte Carlo techniques. They closely relate to our problem,
we'll get back to this later.

### Graph partitioning
We can find another mapping of our problem of finding GS in our lattice glass into Computer Science.
Let's look at the lattice L^k we're in. Due to the periodic boundary conditions we impose, we find ourselves in a
2k-regular lattice. What's more is that every minimal cycle has same length, 4. This holds for k=2 and 3, but does
it hold for all dimensions? Our problem now is that: We want to find two induced subgraphs with the following
conditions: One 3-regular graph S and one 5-regular graph T. They don't need to necessarily be connected. Their
intersection might not be empty. How do we characterize this problem?

There's [work](https://none) indicating that indeed, we find ourselves in the realm of hard problems again: Finding
a k/2-regular induced subgraph of size v/2 in a k-regular graph is, again, NP-complete. Enumerating them using
dynamic programming is in O(n • 2^n) (at least that's the best one I could come up with).

We can maybe stochastically or greedily grow these graphs. But we can probably lower-bound the time we'd actually
need to get close to a large sample of the configuration space with some exponential integer.

This novel description however can give us something more to think about: What are now ideal geometries for the
glassy dynamics we're looking for to emerge? Can we cluster / reduce our system size using some solid arguments 
from graph theory? Is there a minimal graph that shows emergent glassy behavior? Can we use these tools to gather
more insight of what the "glassy dynamics" are in this system?


### Simulated annealing
Can we more naturally follow the evolution of the system? One approach is to slowly decrease the temperature when in
between Monte Carlo steps. This lets particles group and move at a higher rate in the beginning and makes the system
much more rigid towards the end. This procedure is what's known as "simulated annealing". Combinatorial optimization
is a field where this type of stochastic optimization procedure has been employed widely to varying success. Seeing
that we can map our problem to some well-defined discrete space we can certainly make use of this technique here.

It's interesting to see what kind of schedule is _good_ or optimal for the temperature decrease. I'd like to think
of annealing as turning on a fridge that's been defrosted: Slowly getting all items to rest at fridge temperature.
I think it's somewhat natural to look at this as an inverse exponential such that taking the range for our temperatures
as a geometric series could be useful.


### Putting it together
Now, let's look at the heuristic I've come up with to finally equilibrate the glassy system.
I'm using Python-pseudocode to make it as clear as possible.

```python
nsweeps = 10 ** 4
ncool = nsweeps // 2
nfind = nsweeps // 2
betas = np.geomspace(1 - delta, 20., ncool)

# cooling down close to T ~ 0
for i in range(ncool):
    beta = betas[i] 
    num_partitions = np.random.choice([2, 4, 8, 16], 1)
    for j in range(L** 3 // num_partitions // 2):
        l, h = find_partitions(lattice, beta, num_partitions)
        coin = np.random.randint(0, 1)
        if coin:
          nonlocal_sweep_partitioned(lattice, l, h)
        else:
          local_sweep_partitioned(lattice, l, h)


beta = 20.
# setting T ~ 0
for i in range(nfind):
    coin = np.random.randint(0, 1)
    if coin:
        local_sweep(lattice, beta)
    else:
        nonlocal_sweep(lattice, beta)

    handson_sweep(lattice, beta)
```

Okay, what did I write down here?
The first stage consists of repeatedly  partitioning our lattice into random sizes (`num_partitions`). We repeatedly 
try to move particles using the MH-criterion, half the time purely locally, ie. only swap particles with their immediate
neighbors, half the time try to do fully nonlocal swaps. Here we're free to either diffuse particles throughout the
lattice or we try to localize the diffusion to exchange high-energy regions with low-energy regions.
We do this for a fixed amount of time (here we're taking half the number of all sweeps to be part of the first stage).

The second stage is not trying to traverse the entire configuration space anymore but instead drills down on the
"optimal" configuration that's been worked out after the first stage. We can now assume that the energy is below a
certain threshold `E_t`, ie. there's a number of particles which have energy 0. We now only consider the particles
that have energy greater than a certain threshold and try to either locally or nonlocally swap them. This now
concentrates the possible moves. More importantly, we've reduced the number of particles we touch at every optimization sweep and
can now assume that the lattice is somewhat minimized in energy to drill down even further. Of course, we may get
trapped in some local optimum. That's why for every "handson"-optimization we do local or nonlocal sweeps half the time
to ensure that all lattice sites get looked at during this optimization phase.

The risk to stay trapped in a very deep local optimum is of course not zero. What we do get is that we're now dealing
with 10^4 sweeps overall in this heuristic. We're reaching very low-energy regions however. This method needs about four
(4) orders of magnitudes less equilibration sweeps than what the authors of the study we're trying to reproduce suggest.

The good part is: We can now equilibrate the glassy system quickly. We can now bound this procedure and benchmark it.
We've however introduced lots of moving parts that may influence how well we can traverse the entirety of possibilities.


<!----
Algorithm / schedule to get close to GS, respecting physics
1. diffusion on partitions with lowest / highest energy -- coin flips for local/nonlocal moves
   -> speedup
2. reaching H(config) / (rho * L ** 3) <= .4
   -> introduce new measures
3. concentrated SAT sweeps; nonlocal sweeps
   -> slowing down to perform actual optimization
4. getting to zero energy for small systems (python procedure)

Some plots: (mine, in thesis then)
- reaching GS with algorithm versus what happens just using local/nonlocal sweeps
- varying convergence for different betas
- crystalline phase
- energy grid for our rho for different values of rho1, rho2 (ranges)


### Future checkbox (all with question marks)
- [ ] optimality of procedure
- [ ] investigating model in higher dimensions
- [ ] different geometries (diameter, girth, honeycomb)
- [ ] iterated k-sat
- [ ] simulated annealing
- [ ] fully investigating PT for glassy systems
- [ ] 
--->
