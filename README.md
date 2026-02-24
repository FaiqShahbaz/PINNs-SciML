# PINNs-SciML

![python](https://img.shields.io/badge/python-3.8%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![framework](https://img.shields.io/badge/built%20with-PyTorch-orange)
![domain](https://img.shields.io/badge/domain-PINNs-black)
![area](https://img.shields.io/badge/field-SciML-blueviolet)

A curated collection of **Physics-Informed Neural Networks (PINNs)** implementations for solving partial differential equations within the framework of **Scientific Machine Learning (SciML)**.

This repository is designed as an evolving laboratory of PINN-based models, benchmarks, and research experiments.

---

## Table of Contents

- [Overview](#what-are-physics-informed-neural-networks)
- [Repository Structure](#repository-structure)
- [Current Projects](#current-projects)
- [Upcoming Projects](#upcoming-projects)
- [Goals](#goals-of-this-repository)
- [Scientific Foundations](#scientific-foundations)
- [Author](#author)

---

## What Are Physics-Informed Neural Networks?

Physics-Informed Neural Networks (PINNs) incorporate governing physical laws directly into the training objective of neural networks.

Consider a nonlinear partial differential equation of the form:

$$
u_t + \mathcal{N}[u] = 0,
$$

where $\mathcal{N}[\cdot]$ represents a nonlinear differential operator.

A neural network $u_\theta(x,t)$ is used to approximate the solution $u(x,t)$.  
The model is trained by minimizing a composite loss function:

$$
\mathcal{L} = \mathcal{L}_{IC} + \mathcal{L}_{BC} + \mathcal{L}_{PDE},
$$

where:

- $\mathcal{L}_{IC}$ enforces initial conditions  
- $\mathcal{L}_{BC}$ enforces boundary conditions  
- $\mathcal{L}_{PDE}$ minimizes the physics residual  

The physics residual is defined as:

$$
r(x,t) = u_t + \mathcal{N}[u].
$$

Automatic differentiation enables exact computation of derivatives ($u_t$, $u_x$, $u_{xx}$, etc.), allowing the neural network to satisfy the governing equation without traditional mesh-based discretization.

This framework integrates physics directly into deep learning models, forming a core paradigm in Scientific Machine Learning (SciML).

---

## Repository Structure

```
PINNs-SciML/
│
├── PINNs-Burgers/       # Continuous-time PINN for Burgers’ equation
│   ├── README.md        # Detailed project documentation
│   ├── src/             # Implementation
│   ├── outputs/         # Results, figures, metrics
│   └── docs/            # Project-specific slides (optional)
│
└── README.md            # Repository overview (this file)
```

Each project folder contains:

- Source code (`src/`)
- Generated results (`outputs/`)
- Project documentation
- Independent README explaining methodology and results
- Project-specific `requirements.txt`

---

## Current Projects

### 1️⃣ PINNs-Burgers

Continuous-time PINN implementation for the **1D viscous Burgers’ equation**.

**Highlights:**
- Autodiff-based PDE residual enforcement
- Adam → L-BFGS optimization strategy
- Finite-difference reference benchmark
- Quantitative error analysis (relative L2, residual metrics)
- Structured visualization and validation

📂 See: `PINNs-Burgers/`

---

## Upcoming Projects

The following extensions are planned:

- Discrete-time PINNs (Runge–Kutta formulation)
- Allen–Cahn equation
- Inverse problems (parameter identification)
- Multi-dimensional PDEs
- Navier–Stokes PINNs
- Hybrid PINN + classical solver frameworks

This repository will continue expanding as part of an ongoing SciML research portfolio.

---

## Goals of This Repository

- Provide clean, reproducible PINN implementations
- Benchmark PINNs against classical numerical solvers
- Explore stabilization strategies and optimization techniques
- Serve as a structured portfolio of SciML research work

---

## Scientific Foundations

This repository builds upon the foundational work:

M. Raissi, P. Perdikaris, G.E. Karniadakis  
*Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations*  
arXiv:1711.10561, 2017.

---

## Author

Faiq Shahbaz  
GitHub: https://github.com/FaiqShahbaz