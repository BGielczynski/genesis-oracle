# Genesis Oracle - Problem Set 6 (Stochastic Simulations)

This repository contains the implementation of **Problem Set 6** for the module *Applied Modeling and System Simulation (AMS)*. The focus is on stochastic simulations using NumPy and JAX.

## Overview of Exercises

### [Exercise 1: The Antiquated Circle](./src/classical_pi.py)
Estimation of $\pi$ using a classical Monte Carlo approach with NumPy.
- **Method**: 5 million random points in a 2D square.
- **Result**: $\pi \approx 3.1418968$.
- **Visualization**: [docs/classical_pi_disp.png](./docs/classical_pi_disp.png)

### [Exercise 2: The Quantum Leap](./src/monte_carlo.py)
Stochastic Business Revenue simulation using JAX.
- **Variables**: Market Demand (Normal), Asset Cost (Log-Normal), Penalty Rate (Uniform).
- **Techniques**: `jax.vmap` for parallel simulation of 10,000 scenarios.
- **Metrics**: Expected Revenue and 95% Value-at-Risk (VaR).
- **Visualization**: [docs/monte_carlo_revenue.png](./docs/monte_carlo_revenue.png)

### [Exercise 3: Agentic Automation](./docs/Swarm_Stress_Report.md)
Automated analysis using specialized sub-agents.
- **Subagent-Alpha**: Stress-testing the revenue model to find breaking points.
- **Subagent-Beta**: Performance profiling comparing JIT (Cold vs. Warm) execution.
- **Report**: [docs/Swarm_Stress_Report.md](./docs/Swarm_Stress_Report.md)

### [Exercise 4: Boss Fight – The Black Swan](./src/markov_boss.py)
Macro-economic simulation with exogenic shocks.
- **Model**: 3-state Markov Chain (Bull, Stagnation, Recession).
- **Shock**: "Black Swan" event at Day 180-190 (probability mass shifted to Recession).
- **Technique**: `jax.lax.scan` for efficient time-series simulation.
- **Visualization**: [docs/markov_states_history.png](./docs/markov_states_history.png)

## Project Structure
- `src/`: Python source code for all exercises.
- `docs/`: Generated visualizations and reports.
- `pyproject.toml`: Dependency management via `uv`.

## Documentation
The full detailed report is available as a PDF: `C:/GitHub/data/AMS/Set6/Set-6.pdf`.

---
**Author**: Benjamin Gielczynski
**Date**: June 2026
