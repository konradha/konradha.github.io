---
title: "Solving PDEs using ✨Machine Learning✨"
date: 2023-06-07T18:57:35+02:00
draft: false
---

At university some of us choose to learn how to solve PDEs numerically.
Usually, you pass years learning about approximation 
theory, test functions, finite elements, C++, functional analysis, visualization libraries, parallelization. Finally,
you put it together to solve heat or wave equations. Then, when you're more of a practitioner in industry, you fall back to using Ansys.
In 2023 however, we can just take our good ol' reliable Pytorch and _have Adam walk the space that minimizes the
residuals we define for a given PDE_.

And: you don't even need any external data. Choosing your points in a
[smart manner](https://pytorch.org/docs/stable/generated/torch.quasirandom.SobolEngine.html)
(ie. not sampling from some centered distribution), you can even deduce
very nice bounds on convergence and precision of the solution you arrive at. Tremendous.

Let's not talk about the hard numerical analysis terms and other concerns. I want to give an overview
on how you can _quickly_ build solvers for instances of PDE problems on your own.

### The setup
We start from a very common formulation of a PDE problem: Take some operator L defined on some
region O, have it equal some source term f. Define some boundary conditions that are close to your application and you're
done.

Our goal is to build a pattern that lets us include everything needed to instantiate the problem, understand
the geometry and fix parameters in a way that makes the entire computation reproduceable.
We also want to reuse as much as possible from our existing knowledge of how our ML framework works.

In the beginning, you don't need to worry too much about the model architecture you'll need to use. Just take
a vanilla MLP with several hidden layers. Choosing your activation functions carefully however, you might even have some [guarantees](https://arxiv.org/abs/2104.08938)
on how well this works on paper.

I'm going to show the main ideas in somewhat Pythonic pseudocode. For details, follow [this](https://github.com/konradha/DLSC/blob/main/Pinns.ipynb)
tutorial to build a solver for the 1D heat equation. This [collection](https://github.com/mroberto166/CAMLab-DLSCTutorials) of tutorials
is probably a little more polished and complete. If you actually want to see the code running, just clone
one of the repos I'm linking and see how the models converge to a specific analytic solution.


### Implementation
Now: Roll your own PINN object. "PINN" is short for [Phyics-informed neural network](https://en.wikipedia.org/wiki/Physics-informed_neural_networks).
This approach may be extended to your given problem fairly quickly
(ie. another PDE or maybe an inverse problem by introducing another neural net that approximates your
parameter ... see the lecture I've linked below for more possibilities).


Create an object that contains your solver, the parameter space and all the methods
you'd need to compute your problem: On instantiation you want to define the most important variables; ie.
how many boundary points you want to sample on, inner points, the depth of your network.
`assemble_datasets` defines a method to fill our object with data representing our problem's region.


```python
class PINN:
    def __init__(self, n_int, n_sb, n_tb, n_out):
        # domain extrema here: using a rectangular 2d geometry
        self.domain_extrema = torch.tensor([[0, T],  # Time dimension
                                            [0, X]])  # Space dimension
        self.approximate_solution = torch.NN 
        self.training_set_sb, self.training_set_tb, self.training_set_int = self.assemble_datasets()
```

We want to make use of the existing ML patterns to train models. Hence, we adapt our class structure
such that we can call a single `compute_loss` function that does all of the numerical compute.
`tb` and `sb` are short for temporal and spatial boundary respectively.

```python
    def fit(self, num_epochs, optimizer, verbose=True):
        history = list()
        for epoch in range(num_epochs):
            for j, ((inp_train_sb, u_train_sb), (inp_train_tb, u_train_tb),
                    (inp_train_int, u_train_int)) in enumerate(
                            zip(self.training_set_sb, self.training_set_tb, self.training_set_int)):
                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss(inp_train_sb, u_train_sb,
                                             inp_train_tb, u_train_tb, inp_train_int, verbose=verbose)
                    loss.backward()
                    history.append(loss.item())
                    return loss
                optimizer.step(closure=closure)
        return history
``` 


Defining your geometry in a sensible manner is of huge importance. We will need clear
separation of concerns for our geometry as else we'll be debugging for hours on end: Pytorch and other frameworks
are really good at hiding complexity. Debugging your model later on when you're unsure about the basic definition
can become inconvenient. Thus I highly recommend to visualize your sample and your geometry before optimizing.


```python
    def assemble_datasets(self):
        input_sb, output_sb = self.add_spatial_boundary_points() 
        input_tb, output_tb = self.add_temporal_boundary_points()  
        input_int, output_int = self.add_interior_points()         
        ....

    def add_temporal_boundary_points(self):
        ....

    def add_spatial_boundary_points(self):
        ....

    def add_interior_points(self):
        ....

    def show_geometry(self):
        ....
```


Now that we've got the geometry out of the way, we can compute what our model should converge to
at the boundary. Here, we can either immediately return a residual for our model to 
optimize later on. Or we just return our model's output for the boundary as is done
in `compute_loss` below.
We will maybe have to compute gradients for the boundary. It's nice to have all of this in a 
single place, making for easier debugging.


```python
    def apply_boundary_conditions(self, input_sb):
        ....

    def apply_initial_condition(self, input_tb):
        ....
```


Finally, we can use the ML framework we're in to encode the PDE we want to solve. This is the
"interior" region where the PDE we defined in the beginning is calculated. Again, taking our
operator L (involving gradients, integrals etc), our source term f. we want its residual to go to zero,
ie.

$$ L[u]\left( x \right) - f\left(x\right) = r \ \forall x \in O$$

with (hopefully)

$$ r \rightarrow 0 $$ the more we train our model.

```python
    def compute_pde_residual(self, input_int):
        ....
```

The final loss computation which we need for every training step will look somewhat akin to `compute_loss`.
Some details: Our model's predictions are returned from the methods we defined above.
Sometimes we also want to leave in some asserts to make sure our tensor manipulations don't quietly destroy
our model.
We get the residual for the interior of our region O as we don't have to compute anything of its variables
anymore.

Finally, we have an important thing to take care of: Training our model turns out to be a multi-objective 
optimization problem. Suddenly the interior PDE and the boundary residuals are in competition to be optimized.
Hence, to have our optimizer fully acknowledge the importance of boundary conditions, we define a weight `Lambda` 
which is chosen arbitrarily such that we solve in the direction of the boundary conditions. 



```python
    def compute_loss(self, inp_train_sb, u_train_sb, inp_train_tb, u_train_tb, inp_train_int, verbose=True):
        inp_train_sb.requires_grad = True
        u_pred_sb = self.apply_boundary_conditions(inp_train_sb)
        u_pred_tb = self.apply_initial_condition(inp_train_tb)

        assert (u_pred_sb.shape[1] == u_train_sb.shape[1])

        r_int = self.compute_pde_residual(inp_train_int)
        r_sb_u_0 = u_train_sb[:self.n_sb,0] - u_pred_sb[:self.n_sb,0]

        grad_u   = torch.autograd.grad(u_pred_sb[:,0].sum(), inp_train_sb, create_graph=True)[0]
        grad_v   = torch.autograd.grad(u_pred_sb[:,1].sum(), inp_train_sb, create_graph=True)[0]
        grad_u_x = grad_u[:, 1]
        grad_v_x = grad_v[:, 1]

        ...
        r_sb_u_1 =  grad_u_x[self.n_sb:]
        r_sb_v   = grad_v_x

        Lambda = 10
        loss = Lambda * (r_sb_s + r_sb_u_0 + r_sb_u_1) + r_int
```



### A quick example

To exemplify how quickly you can adapt this, I tried to build this entire flow using
[tinygrad](https://github.com/tinygrad/tinygrad). Let's quickly define a dense model.


```python
class Net:
    def __init__(self, in_channels=1, out_channels=1, num_layers=5, dims=4):
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.num_layers   = num_layers
        self.dims         = dims

        self.input_layer  = Linear(in_channels, dims)
        self.output_layer = Linear(dims, out_channels)

        self.activation   = Tensor.tanh
        self.layers       = [Linear(dims, dims, bias=False,) for _ in range(self.num_layers)]


    def __call__(self, x: Tensor, transform = None):
        if transform is None:
            return self.forward(x)
        raise NotImplementedError("No-transform case not handled yet")

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for l in self.layers:
            x = self.activation(l(x))
        return self.output_layer(x)
```

Now, the PINN model definition flows easily from our existing knowledge of Pytorch.
We use a very simple dense model to approximate the 1+1d heat equation on a rectangular domain. See
the notebooks linked above for comparison.

```python
class PINN:
    def __init__(self, n_interior, n_boundary, n_layers=7, n_dim=8):
        # 2d geometry, heat equation
        self.approximate_solution = Net(in_channels=2, out_channels=1, num_layers=n_layers, dims=n_dim)
        self.domain_extrema       = Tensor([[0., 0.1], [-1., 1.]])

        self.n_interior = n_interior
        self.n_boundary = n_boundary
        self.Lambda     = 10

        self.spatial_boundary_sample, self.temporal_boundary_sample, self.interior_sample = \
                self.assemble_dataset()

    @staticmethod
    def convert(x: Tensor, a: float, b: float):
        # need to adapt to tinygrad's Uniform sampling mechanism
        # ie. the usual pullback looks a little different
        return .5 * (b-a) * (x + 1) + a


    def assemble_dataset(self):
        spatial_boundary_input, spatial_boundary_output   = self.add_spatial_boundary_points()
        temporal_boundary_input, temporal_boundary_output = self.add_temporal_boundary_points()
        interior_input, interior_output                   = self.add_interior_points()
        return (spatial_boundary_input, spatial_boundary_output), (temporal_boundary_input, temporal_boundary_output), (interior_input, interior_output)
```

We can add our geometry and visualize it.


```python
    def add_spatial_boundary_points(self):
        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]

        t0 = self.domain_extrema[0, 0]
        tL = self.domain_extrema[0, 1]

        points_per_side = self.n_boundary // 2
        t_values_0 = self.convert(Tensor.uniform(points_per_side,), t0, tL)
        x_values_0 = Tensor.ones(points_per_side) * x0
        input_spatial_boundary_0 = Tensor.stack((t_values_0, x_values_0), 1)

        t_values_L = self.convert(Tensor.uniform(points_per_side,), t0, tL)
        x_values_L = Tensor.ones(points_per_side) * xL
        input_spatial_boundary_L = Tensor.stack((t_values_L, x_values_L), 1)

        # redundancy for more readable training loop
        output_spatial_boundary_0 = Tensor.zeros(points_per_side)
        output_spatial_boundary_L = Tensor.zeros(points_per_side)

        return Tensor.cat(input_spatial_boundary_0, input_spatial_boundary_L), Tensor.cat(output_spatial_boundary_0, output_spatial_boundary_L)


    def add_temporal_boundary_points(self):
        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]

        t0 = self.domain_extrema[0, 0]
        tL = self.domain_extrema[0, 1]

        x_values_0 = self.convert(Tensor.uniform(self.n_boundary), x0, xL)
        t_values_0 = Tensor.ones(self.n_boundary) * t0

        input_temporal_boundary  = Tensor.stack((t_values_0, x_values_0), 1)
        output_temporal_boundary = - Tensor.sin(np.pi * x_values_0)

        return input_temporal_boundary, output_temporal_boundary


    def add_interior_points(self):
        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]

        t0 = self.domain_extrema[0, 0]
        tL = self.domain_extrema[0, 1]

        t_values_interior = self.convert(Tensor.uniform(self.n_interior), t0, tL)
        x_values_interior = self.convert(Tensor.uniform(self.n_interior), x0, xL)

        u_values_interior = Tensor.zeros_like(x_values_interior)

        return Tensor.stack((t_values_interior, x_values_interior), 1), u_values_interior


    def show_sample(self):
        x_in, x_out = p.add_spatial_boundary_points()
        t_in, t_out = p.add_temporal_boundary_points()
        interior_in = p.add_interior_points()

        plt.scatter(x_in[:, 1].cpu().numpy(), x_in[:, 0].cpu().numpy(), )
        plt.scatter(t_in[:, 1].cpu().numpy(), t_in[:, 0].cpu().numpy(), )
        plt.scatter(interior_in[:, 1].cpu().numpy(), interior_in[:, 0].cpu().numpy(), )
        plt.show()
```

And we can quickly encode the numerics we need to solve the heat equation.

```python
    def apply_initial_condition(self, input_temporal):
        u_pred_temporal = self.approximate_solution(input_temporal)
        return u_pred_temporal

    def apply_spatial_condition(self, input_spatial):
        u_pred_spatial = self.approximate_solution(input_spatial)
        return u_pred_spatial

    def compute_pde_residual(self, input_interior):

        input_interior.requires_grad = True
        x = input_interior[:, 1]
        t = input_interior[:, 0]
        x.requires_grad = True; t.requires_grad = True
        inp = Tensor.stack((t, x), 1)
        u = self.approximate_solution(inp)

        u_sum = u.sum()
        u_sum.backward()
        # get first order derivatives using single backward pass
        grad_u_x = x.grad
        grad_u_t = t.grad

        # "scalarize" grad_u_x
        intermediate_grad_u_x = grad_u_x.sum()
        intermediate_grad_u_x.backward()
        grad_u_xx = intermediate_grad_u_x

        residual = grad_u_t - grad_u_xx
        return residual.realize()

    def compute_loss(self, spatial_boundary_input, spatial_boundary_output, temporal_boundary_input,
            temporal_boundary_output, interior_input, interior_output):
        u_pred_spatial  = self.apply_spatial_condition(spatial_boundary_input)
        u_pred_temporal = self.apply_initial_condition(temporal_boundary_input)

        residual_interior = self.compute_pde_residual(interior_input)
        residual_spatial  = u_pred_spatial - spatial_boundary_output
        residual_temporal = u_pred_temporal - temporal_boundary_output

        loss_boundary = self.Lambda * ((residual_spatial.abs()).mean() ** 2 + (residual_temporal.abs()).mean() ** 2)
        loss_interior = (residual_interior.abs()).mean() ** 2
        loss = loss_boundary + loss_interior

        return loss
``` 

And we're done. This can now be tuned to fully converge to the analytical solution.





### Final thoughts
There's lots to discover in this new world of SciML: Solving non-deterministic models in high dimensions,
operator learning, foundation models, PDE-constrained optimization ... We're seeing the fruits of an intense
decade of investment in ML research, enabling us to solve hard problems on retail machines in no time. 




Thanks to Professor Mishra and colleagues, you can follow the lecture series on [Deep Learning in Scientific
Computing](https://www.youtube.com/watch?v=y6wHpRzhhkA&ab_channel=CAMLab%2CETHZ%C3%BCrich) to 
get _all_ the details.
