---
title: "A thesis"
date: 2025-11-07T15:06:05+01:00
draft: true
---
<!---
- what's a PDE
- what is simulation used for
- what's a model weight
- simulation is expensive and hard to understand, limited
- RL simulation is imperfect, not physical
- UQ is really fucking expensive
- no simulation and experiment-heavy industry wants to share their IP
- PDE FMs are good: not too difficult-to-train, well-explored architectures (at scale!),
  VERY good generalizability
- data is hard to come by
- with smart data exploration, close to industry (VPC / on-prem WITH customers):
- unlocking: interpretable data sources and diversity of problems and deisgn choices
- unlocking: smart, quick exploration of the design space for actual engineering problems
  -> electrodynamics, automotive, robotics and everything we can do in the continuum 
- unlocking: UQ at scale for a wide variety of problems
- unlocking: distilling complex multi-scale multi-physics problems into a few activations
- unlocking: (far ahead) RL enhanced with actual physics, downscaled from proprietary models
- looking to build this step-by-step:

1. data handling (and understanding with models)
2. design space utilities
3. UQ at scale
4. distilling models for quicker runs
5. serving large design decisions and expanding scale of problems (micro <-> macro?)
6. understand models sufficiently to distill them into so quick predictions we can feed them into RL
7. replace (too expensive, not all!) simulation in industry by learned surrogates and only enhance, bit-by-bit, the model
   once inference is deemed too brittle

--->


Engineers do not need chatbots. The need cheap, nearly-instant evaluation of design, ideas and constantly
refined physics.
Reducing the overhead in feedback loops is the game.

Simulation is the bottleneck. Legacy solvers, brittle scripts, memory-bound kernels,
and scarcely-available experts make iteration slow and fragile. Compute exists; the horizontal glue is missing.


### State of affairs

Synopsys now dominates huge swaths of the simulation market after acquiring Ansys. The reality is this though:
ML adoption lags behind, engineers struggle with coherent workflows. Every analysis on simulation data and
process management (SPDM) shows knowledge workers
unsatisfied with existing infrastructure due to brittle setups and difficult-to-bridge silos.

Sovereignty isn't optional anymore -- it's a buying constraint. The integrated giants build in-house.
Everyone else is stuck between legacy tooling and cloud vendors they can't trust with trade secrets.
Execution infrastructure that runs on customer hardware, trains on customer data, and extracts nothing without permission.
Deploy per-site and learn per-domain. Aggregate only what customers explicitly share.
Behemoths sell centralized suites but selling the ability to actually use ML _without handing over your process IP_
to a SaaS platform is what will differentiate.


<!----
We could wait. Or we can make it easy and safe for R&D to try new methods at low cost, on their own infrastructure,
with provenance. Vertically integrated giants will keep building in-house, sovereignty and regulation push them that way.
But sovereignty is a buying constraint for everyone.
Without a global view of data and decisions, you can’t see inefficiencies or place ML where it helps rather than harms.
Especially when stakes are as high as in, say, semiconductor manufacturing.

Make it easier to have R&D departments check out novel developments at little cost. Maybe even design a sufficiently
general framework to have it run on their own infrastructure. In the end, everything is computer, caches and cuda.
Have them train and refine ML models precisely adapted to their problems and engineering designs.
And then use the acquired knowledge to further refine model design, data generation and data
management. The IP stays with everyone as much as they require, their pipelines will see slowly emerging enhancements
in any case.
--->


### Foundation models work

Take an existing PDE foundation model, finetune with 20-100 samples, hit 1% precision.
Orders of magnitude speedup.
This isn't speculative; the theory around operator learning is mature.
What's missing is deployment infrastructure that customers actually trust. And people pushing the envelope on the next
k iterations of foundation models.

<!---
What's the state of the art in ML for simulations though? Well, you can reuse and refine existing transformer architectures.
Theory describing them and their implications has matured to a point of "just throw more of your problem at it".
Eventually they will have their moment, will be distilled, quantized and reduced to run on edge devices, enhancing
and guiding subterrain exploration and waveguide design. The numbers are eye-watering: Take an existing foundation model,
finetune it with 20-100 samples of your domain and you're within 1% precision. And this while running orders of magnitude speedups.
--->


### Need for speed

*Map reality.* Simulation data is IP-sensitive. We need to make it indexable,
cheaply understanding geometries, fields units and make the space of explored designs searchable.Build per-customer infrastructure
from composable blocks. First efficiency gain: eliminate redundancy in existing design considerations.

*Automate and reduce walltime.* Time-to-solution can be helped with finetuned models. Actively sampling the design spaces
for the problem at hand -- either by heuristics or constantly learning models -- will enhance the ground truth samples
to further enhance the models that enhance engineers' design iteration time.

*UQ at unprecedented scale.* Uncertainty Quantification becomes prohibitively expensive. Especially when the requirements
imposed by regulators and insurance companies imply heavy penalties when you're wrong. Let's run large-scale simulations at
inference speed: This  allows us to sample the future much more precisely and thoroughly. At a fraction of
the cost. What could be better than running frameworks
optimized for multiple exaflops of training time such as Pytorch and Jax?

*Domain by domain. Learn without leaking. Better decisions.* The requirements per-industry vary in orders of magnitude.
Semiconductor fabs need ångström precision. Automotive needs crash dynamics. Energy needs reservoir modeling.
In the end it's all the same: Integrate a solver environment, introduce better sampling mechanisms, streamline
curation and analysis, and slowly allow for fast inference to take over. What's common is this though: The need for secrecy
and sovereignty for each and every customer is paramount. The decision quality however will be enhanced deeply
with fast inference pipelines. Build once per customer. IP never leaves.

Further out: *Differentiability, transferability and environmental learning.* ML models are differentiable.
Inverse design becomes tractable. Model distillation reduces the sim2real gap.
Deploy where Synopsys can't — edge systems, proprietary processes, regulated environments.

Build the infrastructure. The engineering will follow.







[^1]: [FNOs in reservoir modelling](https://www.sciencedirect.com/science/article/pii/S0098300424003091)
[^2]: [Operator Learning Theory](https://www.jmlr.org/papers/volume22/21-0806/21-0806.pdf)
[^3]: [EU rules](https://www.reuters.com/technology/deutsche-telekom-airbus-slam-plan-allowing-big-tech-access-eu-cloud-data-2024-04-10/)
[^4]: [Merger](https://www.reuters.com/markets/deals/eu-approves-synopsys-35-billion-ansys-deal-under-conditions-2025-01-10)
