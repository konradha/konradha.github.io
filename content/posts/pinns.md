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
on how you can _quickly_ build solvers for instances of hard problems on your own.

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
is probably a little more polished and complete.


### Implementation
Now: Roll your own PINN object. This approach may be extended to your given problem fairly quickly.
(ie. another PDE or maybe an inverse problem by introducing another neural net that approximates your
parameter ... see the lecture I've linked below for more possibilities).


Create an object that contains your solver, the parameter space and all the methods
you'd need to compute your problem: On instantiation you want to define the most important variables; ie.
how many boundary points you want to sample on, inner points, the depth of your network.
`assemble_datasets` defines a method to fill our object with data representing our problem's region.


```python
class PINN:
    def __init__(self, n_int, n_sb, n_tb, n_out):
        # domain extrama here: using a 2d geometry
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


Now that we've got the geometry out of the way, we can compute what our model should look
like at the boundary. Here, we can either immediately return a residual for our model to 
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



### Final thoughts
There's lots to discover in this new world of SciML: Solving non-deterministic models in high dimensions,
operator learning, foundation models ... We're seeing the fruits of an intense decade of investment in ML
research, enabling us to solve hard problems on retail machines in no time. 




Thanks to Professor Mishra and colleagues, you can follow the lecture series on [Deep Learning in Scientific
Computing](https://www.youtube.com/watch?v=y6wHpRzhhkA&ab_channel=CAMLab%2CETHZ%C3%BCrich) to 
get _all_ the details.
