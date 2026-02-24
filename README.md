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

Physics-Informed Neural Networks (PINNs) embed governing physical laws directly into the training objective of neural networks.

Given a PDE of the general form:

u_t + N[u] = 0

a neural network u_θ(x, t) is trained to:

- Satisfy initial and boundary conditions  
- Minimize the PDE residual computed using automatic differentiation  

This approach enables solving PDEs without traditional mesh-based discretization and integrates physics directly into deep learning models.

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