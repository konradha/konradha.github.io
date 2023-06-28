---
title: "Solving instances of radiative transfer using PINNs"
date: 2023-06-07T18:57:35+02:00
draft: true
---


I've wanted to write about this topic a long time ago: Solving PDE's using Neural Nets.
What an amazing thing. Instead of passing years learning about approximation 
theory, test functions, finite elements, C++, functional analysis, visualization libraries and then putting it
together to solve (seemingly trivial) heat or wave equations (and then falling back to using Ansys in industry), you can just take
your good ol' reliable Pytorch and have Adam walk the space that minimizes a residual you
define for a given PDE. And: you don't even need any data. Choosing your points in a
[smart manner](Sobol low discrepancy sequence) (ie. not sampling from some centered distribution), you can even deduce
very nice bounds on convergence and precision of the solution you arrive at. Tremendous.
Let's dive in.


### The setup -- Radiative Transfer PDE
<!---
At the heart of the problem lies the following PDE

$\frac{1}{c} u_{t} + \omega \dot \nabla_{x} u + ku
= + \sigma \left(u - \frac{1}{s_{d}} \int_{\Lambda} \int{S} \Phi \left ( 
  \omega, \omega', \nu, \nu' \right) d\omega' d\nu' \right) = f $

Where u takes values from $\mathbb{R}^{d}$, $\Lambda$ is a real interval defining
chromaticity, $S = \mathbb{S}^{d-1}$ and $f$ is some source term.
--->
For this example, we're going to look at the [radiative tranfer PDE](wiki) which models
different radiation phenomenae at different scales in nature. (...)

For actual applications, this is a problem in high dimensions. This is _not good_ by itself.
Solving this thing numerically is even worse as it's not only a PDE but also a partial-integro-
differential equation, ie. there's a so-called _scattering kernel_ phi that needs to be evaluated
at every point we want to solve this PDE for.  Depending on how precise you want to solve this may be
a really expensive thing to do.


 


### The background -- PINNs and Surrogate models
Let's keep this short: Your PDE can be solved by a vanilla neural net. Choosing your activation
function(s) carefully, you might even have some [guarantees](...) on how well this should work theoretically.

### Implementation -- Ease and bottlenecks; acceleration
Now: Roll your own PINN object for fun and profit. This approach may be extended to your given problem
(ie. another PDE or maybe an inverse problem by introducing another neural net that approximates your
parameter estimate ... but I digress) pretty quickly.

Class definition
```python
class PINN:
    def __init__(self, n_int, n_sb, n_tb, n_out):
               # here come a bunch of parameters you need in your model
        ....

        # domain extrama here: using a 2d geometry
        self.domain_extrema = torch.tensor([[0, T],  # Time dimension
                                            [0, X]])  # Space dimension
        self.approximate_solution = torch.NN 
        self.training_set_sb, self.training_set_tb, self.training_set_int = self.assemble_datasets()
```

Geometry
```python
    def assemble_datasets(self):
        input_sb, output_sb = self.add_spatial_boundary_points()   # S_sb
        input_tb, output_tb = self.add_temporal_boundary_points()  # S_tb
        input_int, output_int = self.add_interior_points()         # S_int
        ....

    def add_temporal_boundary_points(self):
        ....

    def add_spatial_boundary_points(self):
        ....

    def add_interior_points(self):
        ....
```

Boundary conditions
```python
    def apply_boundary_conditions(self, input_sb):
    ....

    def apply_initial_condition(self, input_tb):
    ....
```

Residual: PDE
```python
    def compute_pde_residual(self, input_int):
    ....
```

Residual: Boundary conditions
```python
    def compute_loss(self, inp_train_sb, u_train_sb, inp_train_tb, u_train_tb, inp_train_int, verbose=True):
        u_pred_sb = self.apply_boundary_conditions(inp_train_sb)
        u_pred_tb = self.apply_initial_condition(inp_train_tb)

        assert (u_pred_sb.shape[1] == u_train_sb.shape[1])
        assert (u_pred_tb.shape[1] == u_train_tb.shape[1])

        r_int = self.compute_pde_residual(inp_train_int)
        r_sb_f_0 = u_train_sb[:self.n_sb,0] - u_pred_sb[:self.n_sb,0]

        grad_Tf = torch.autograd.grad(u_pred_sb[:,0].sum(), inp_train_sb, create_graph=True)[0]
        grad_Ts = torch.autograd.grad(u_pred_sb[:,1].sum(), inp_train_sb, create_graph=True)[0]
        grad_Tf_x = grad_Tf[:, 1]
        grad_Ts_x = grad_Ts[:, 1]
        r_sb_f_1 =  grad_Tf_x[self.n_sb:]
        r_sb_s = grad_Ts_x
    ....
```

Training loop + Plotting

### TL;DR
See code @ ...

