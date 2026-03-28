# Parallel Phase-Field Simulation of Metal Plating Dynamics

A high-performance Python implementation for simulating electrochemical metal plating (electrodeposition) in battery systems. This project focuses on transitioning a serial, single-core phase-field model into a parallelized framework to handle large-scale spatial domains and long-term temporal evolution.

## Overview
Phase-field models are computationally demanding due to the need to solve coupled partial differential equations (PDEs) over fine grids to capture interface dynamics. This project optimizes the simulation of metal ion reduction and morphology evolution (e.g., dendrite growth) by leveraging multi-core CPU architectures.

The primary objective is to reduce wall-clock time while maintaining the physical accuracy of the electrochemical flux and interface kinetics.

## Key Features
* **Massively Parallelized Solvers:** Migrating from single-threaded execution to parallel processing using `mpi4py` (Message Passing Interface).
* **Domain Decomposition:** Efficient splitting of the simulation grid across multiple CPU cores.
* **Electrochemical Coupling:** Solves the evolution of the order parameter ($\xi$) coupled with the concentration field ($c$) and electrostatic potential ($\phi$).
* **Performance Benchmarking:** Includes scripts to analyze scaling efficiency (Strong vs. Weak scaling).

## Tech Stack
* **Language:** Python 3.x
* **Scientific Libraries:** `NumPy`, `SciPy`
* **Parallelization:** `mpi4py`
* **Visualization:** `Matplotlib`
* **Environment Management:** `Conda`

## Performance Improvement
The transition to parallel execution aims to address the following bottleneck in the original code:
> **Original Bottleneck:** The $O(N^2)$ or $O(N^3)$ complexity of the spatial stencil operations limited simulations to small 2D domains. Parallelization allows for larger grids and the exploration of 3D plating dynamics.