# PINNs-SciML

A curated collection of **Physics-Informed Neural Networks (PINNs)** implementations for solving partial differential equations within the framework of **Scientific Machine Learning (SciML)**.

This repository is designed as an evolving laboratory of PINN-based models, benchmarks, and research experiments.

---

## What Are Physics-Informed Neural Networks?

Physics-Informed Neural Networks (PINNs) embed governing physical laws directly into the training objective of neural networks.

Given a PDE of the general form:

u_t + N[u] = 0

a neural network u_θ(x, t) is trained to:

• Satisfy initial and boundary conditions  
• Minimize the PDE residual computed using automatic differentiation  

This approach enables solving PDEs without traditional mesh-based discretization.

---

## Repository Structure

```
PINNs-SciML/
│
├── PINNs-burgers/       # Continuous-time PINN for Burgers’ equation
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

---

## Current Projects

### 1️⃣ PINNs-Burgers

Continuous-time PINN implementation for the 1D viscous Burgers’ equation.

Highlights:
- Autodiff-based PDE residual enforcement
- Adam → L-BFGS optimization strategy
- Finite-difference reference benchmark
- Residual diagnostics and quantitative error analysis
- Energy decay validation

See: `PINNs-burgers/`

---

## Research Direction

Planned extensions include:

- Discrete-time PINNs (Runge–Kutta formulation)
- Allen–Cahn equation
- Inverse problems (parameter discovery)
- Multi-dimensional PDEs
- Fluid dynamics PINNs
- Hybrid PINN + classical solver frameworks

---

## Scientific Foundations

This repository builds upon the foundational work:

M. Raissi, P. Perdikaris, G.E. Karniadakis  
*Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations*  
arXiv:1711.10561, 2017.

---

## Goals of This Repository

- Provide clean, reproducible PINN implementations
- Benchmark PINNs against classical numerical solvers
- Explore stabilization strategies and optimization techniques
- Serve as a structured portfolio of SciML research work

---

## Author

Faiq Shahbaz  
GitHub: https://github.com/FaiqShahbaz