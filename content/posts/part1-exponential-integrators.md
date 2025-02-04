---
title: "Exponential Integrators for PDEs"
date: 2025-01-31T15:15:24+01:00
draft: false
---

_Disclaimer: Lots of sources are missing here. Will be added in the future._

This is the first one of a sequence of posts during which we'll work through
efficiently writing a simple finite differences simulation. We'll touch on some physics,
some numerical methods, some C++, some CUDA. Eventually we'll see some pretty animations.


#### Integrable Systems

Understanding nature is hard. For instance, the Millenium prize to find analytical solutions
to the Navier-Stokes equation is still up for grabs. Fermion systems scale exponentially
in the number of particles in a system. Lots of systems do not give us easy access to 
analytical or even numerical solutions we can be sure are correct.

However, nature leaves breadcrumbs. We can model detached phenomena using systems we
understand well. Predicting a tsunami trajectories, modelling the trajectory of rogue waves, optical 
data transmission in glass fibre, or magnetic confinement -- we can model all of these
using integrable systems that we can sum up as `Nonlinear Wave Equations`.

What makes these equations special is that they're so-called [integrable
systems](https://en.wikipedia.org/wiki/Integrable_system). It's a deep field with connections
from dynamical systems to algebraic geometry -- we'll not go into too much detail here.

A few notions are, however, interesting to see.
Integrable systems consume an infinite number of observables. Ie. in the context of numerical
methods, we can check the behavior of different such quantities. We thus have easy access to
proxies letting us understand the behavior of our methods in a diverse set of ways.
We can, for instance, find two quantities emerging from a simple energy conservation
consideration. Without going to deep theory-wise, let's look at the sine-Gordon equation

$$u\_{tt} = c(x, y) \left(u\_{xx} + u\_{yy}\right) - m(x, y) \sin(u)$$

with corresponding Hamiltonian (let's for the sake of argument forget about the focus/nonlinear $c, m$ terms)

$$H = \int \left(\frac{1}{2} \left(|\nabla u|^2 + |u\_t|^2\right) + (1 - \cos(u)) \right) d\Omega $$

With a corresponding real domain $\Omega \subseteq \mathbf{R}^{k}$ which we're integrating over.
Using calculus of variations we can derive two fields which should yield identical parts
of the Hamiltonian: The energy density and the energy flux. The interested reader can find the expressions
of both very quickly, taking the time derivative of the Hamiltonian density and observing the arising
expression for $u\_{tt}$.

From this we can observe at different time steps what our numerical method might get wrong, where it
might unnecessarily oscillate, what non-smooth features of a given system it might not be able to capture.
It's a solid tool to debug numerical methods.

Apart from this very practical consideration, integrable systems give us access to much more.
Let's look at [Lax pairs](https://en.wikipedia.org/wiki/Lax_pair): Without going into too much detail,
reformulating an integrable system in terms of its correspodning Lax pair lets us understand the
underlying _structure_ of a given system. We can get access to different classes of analytical
solutions, their geometries, their long-term behavior. A huge accelerant for progress in the field
of integrable systems has been the Inverse Scattering Transform, one of the main methods to 
construct analytical solutions from Lax pairs. It's been developed further recently to yield
[more advanced numerical methods](https://arxiv.org/abs/2312.11780).

In the interest of not wasting too much time on mathematical details, we'll cut the exploration here. See the linked
references if you're interested in the physics and the actual development of the related methods :)

#### Structure-preserving integrators
The integrable equation we'll be treating are evolution equations. Hence, we'll have to think deeply about
how to integrate every given system for a time step $\tau$.

What are good _integrators_ for our _integrable systems_?

For systems evolving under $u\_{tt}$, nature has given us a great method, given we don't have difficult spectral
or integral operators that we need to take care of. One can show that the Stormer-Verlet update step is
symplectic. It's explicit and _preserves structure_. Geometric and integrability properties are respected in this
setting. For our sine-Gordon equation we can quickly write it down

$$u\_{n+1} = 2 u\_n - u\_{n-1} + \tau^2 \left( c(x, y) \Delta u\_{n} - m(x, y) \sin(u\_{n}) \right)$$

Assuming we integrate over $n\_t$ time steps in $[0,T]$ with $\tau = \frac{T}{n\_t}$.

But that's not an exponential integrator. Honoring the promise of this article's title,
there are more, maybe better integrators. Assume a general time-dependent problem formulated as

$$u\_{tt} + A u = g\left(u\right)$$

where $A$ is a linear operator and $g$ is a constant inhomogeneity, we can rely on [Hairer, Lubich, Grimm's and
friend's work](https://ludwiggauckler.github.io/habil-web.pdf)[PDF]. It stems from considerations of the [variation of constants
formula](https://en.wikipedia.org/wiki/Variation_of_parameters), where we can express the update -- given
$L$ is a sufficiently stiff operator such that we need such an involved integrator -- as follows

$$ u\_{n+1} - 2 \cos \left(\tau \Omega \right) u\_n  + u\_{n-1} = \tau^2 \psi^2 \left(\frac{\tau}{2} \Omega \right)
g\left( \phi \left(\tau \Omega \right) u\_{n} \right)$$

where $\psi$ and $\phi$ are appropriately chosen _filter functions_ and $\Omega^2= A$.

Okay, great, we have a nice exponential integrator for systems involving $u\_{tt}$. And [we can even show that it
performs really well compared to Stormer-Verlet](TODO) (under the right constraints).


##### What if our model does NOT evolve in $\_{tt}$, though?


For instance, the Alfvén or NLS equations look something like $i u\_t = ...$. Here, the solution that preserves
structure (is symplectic) needs to look differently. We can make use of the fundamental solution to the NLS which
looks like

$$u(\tau) = \exp\left(-i \tau H\right) u\_0$$

which we see is already problematic as for us, $H = \Delta + |u|^2$. How would we evaluate this function?
The [BCH formula](https://en.wikipedia.org/wiki/Baker%E2%80%93Campbell%E2%80%93Hausdorff_formula) already tells us
that we can't arbitrarily switch order of application of operators here. Not cool. What else is there? Well,
instead of trying to run $u(t + \tau) = \exp\left(-i \tau H\right) u(t)$, we could split our Hamiltonian into
$$H = L + N$$ in terms of linear and nonlinear operators. That way, we can define another _symplectic_ integrator

$$\exp(-i\tau H) = \exp\left(N(-i\tau/2) \cdot L(-i\tau) \cdot N(-i\tau/2) \right) + \mathcal{O} \left(\tau^3\right)$$

In practice for the NLSE this looks like

$$u\_{n+1} = \exp(-i \tau/2 N) \exp(-i \tau L) \exp(-i \tau/2 N) u_{n}$$

which is nice and all, but we still don't know what we're computing here.

#### Finite differences
We've now investigated discretization in terms of time. Unfortunately, we need to represent our systems on a machine
that has finite precision and so we need to discretize in space as well. A great teacher of mine -- a physicist --
told us to never employ finite differences to evolve the Schrödinger equation. He used the words "brain-dead" and
"embarrassed" in that context.


Anyway. We'll look at finite differences for our 2 + 1 dimensional system now. It's a method simple enough for us
to just neglect meshing and other considerations. It's one part of the discretization we can easily[\*] exchange.


For finite differences we could apply a stencil. We can nicely describe different physical boundary conditions
in finite differences as well. People love to use their spectral methods, but they're not entirely physical in
that very few natural domain really show us this highly regular periodic behavior. Hm.


Finite differences matrices for a square domain are computed quickly. They have nice structure one can exploit.
More on that later.

Looking at our integrators for the SGE and the NLSE, we can quickly observe that the linear operator L is
what we need to discretize here.

The usual formulation is the first-order 5-point stencil

$$u^n_{i, j} = \frac{u^n_{i+1, j} - 2 u^n_{i, j} + u^n_{i-1, j}}{h\_x^2} + \frac{u^n_{i, j+1} - 2 u^n_{i, j} + u^n_{i, j-1}}{h\_y^2}$$


which we can also very concisely express as a np "kernel":

```python
def u_xx_yy(buf, a, dx, dy):
    uxx_yy = buf
    uxx_yy[1:-1, 1:-1] = (
        (a[2:, 1:-1] + a[:-2, 1:-1] - 2 * a[1:-1, 1:-1]) / (dx ** 2) +
        (a[1:-1, 2:] + a[1:-1, :-2] - 2 * a[1:-1, 1:-1]) / (dy ** 2)
    )
    return uxx_yy
```

Let's assume, for simplicity that our domain is $[-L\_x, L\_x]^2$ discretized with $n\_x$, $n\_y$ points.
For no-flux, homogeneous von-Neumann boundary conditions: $\frac{\partial u}{\partial \vec{n}} = 0$.
Discretizing along the Cartesian x-y axis as $(n\_x, n\_y)$ yields a sparse matrix which is triadiagonal with
2 bands at the diagonals $+/- n\_x$. 


#### What could be done
This is all very nice with some very simple building blocks. Of course, to get predictably good behavior for our
numerical methods we'd like to employ finite elements or -- even better -- finite volume methods. The latter give
us access to actual flow and energy control on a granular level impossible when employing finite differences. It
however also implies lots of work considering building up the related sparse systems. We'll leave that for another
day. We also haven't touched on spectral methods and any of the deeper details of IST, integrable systems
and solving Riemann-Hilbert Problems. This we'll leave for another day as well.


#### Next time
We've looked at some fairly basic methods and gained a preliminary understanding of integrable systems. In the next part
we'll be exploring some of the technical difficulties we'll have to come up with solutions for before implemementing
any advanced kernel-based methods. For instance, how do we build sparse matrices we can use further down the line?
How do we compute matrix functions, even moreso trigonometric matrix functions?
