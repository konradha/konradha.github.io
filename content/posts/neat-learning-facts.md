---
title: "A few notes on Learning Theory"
date: 2023-03-13T17:33:09+01:00
mathjax: true
draft: true
---
<!---I've taken up an exciting challenge this year: Going back to school
to finish my Masters and formalize a) my education and b) my thinking.
Here, I want to explore a few points on Learning Theory I've found neat
or intriguing.--->
In university contexts around machine learning
it usually happens that you get introduced to some sort of formalism
around function compositions, linear algebra and some 
factoids on probability and statistics. You arrive at an orchestration
of linear function compositions and wonder how you suddenly have strange
nonlinearities while some randomized walker descents a loss landscape. 

At least that's been the case for
me at lot of times. Let's summarize some of the formalism we need
to maybe show one or two things and then take a look at what 'state of
the art' techniques actually boil down to.

I really liked George Hotz's introduction in one of his talks
(unfortunately I don't remember where it's from) which goes like this:
"Take a function f mapping elements from sets X to set Y. Our fundamental
question is this: f of x equals y. What's f?" To me, this is much more concise
than handwaving + getting lost in activation functions and collections of
linear maps. Maybe you're a rather slow thinker as I am, let's find out 
what's actually going on.

#### Abstract setting: Formalism

Define a set of pairs in real space
$$ \\{ (x_1, y_1), ..., (x_m, y_m) \\} \subset \mathbb{R} \times \mathbb{R} $$

This is our *sample*. Our goal here is to get _as close as possible_ to the function
$ h: \mathbb{R} \longrightarrow \mathbb{R} $, our so-called *ground truth* which should yield
$ h (x_i) = y_i $ for all $i = 1,...,m$. Reality, however is full of things we do not know about.
We model our *sample* as a collection of data plus some noise
$$ Y_i = h(X_i) + Z_i $$ for all $i=1,...,m$. The Z's account for _small_ pertubations
so they're random variables centered around 0. 
To get more abstract, consider a measurable mapping $f$ from a compact metric space
$$ f: \mathbf{X} \longrightarrow \mathbb{R} $$ 
We define the *generalization error* as follows:
$$ \epsilon_{\rho} (f) = \int (f(x) - y)^{2}d\rho(x,y) $$
for a suitable probability measure $ \rho$. Simple fact that to me feels worth pointing out:
The probability space your data is sampled from will never be explored in its entirety as you're integrating
$\rho$-everywhere. In other words, your generalization error will always skew towards the data you're
using to train your model. Deeply thinking about 
the data you're using is very well worth it as you'll never be able to portray reality in its essence.






<!---

-> formalism until bias-variance tradeoff

-> Monte Carlo integration in this formalism


- Deriving well-known procedures / objects and tackling them using
  our new-found tools from learning theory

- Neat tricks: RHKS & Mercer kernels (+ tricks from exercises I didn't
  know)
--->


<!---$$\left{\left(x_1, y_1\right), ..., \left(x_1, y_1\right)\right}$$.--->
