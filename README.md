# Sparse Group Lasso Solver

This project implements a **Sparse Group Lasso (SGL) optimization algorithm** from scratch in Python.  
Sparse Group Lasso combines **group sparsity** (turning entire groups of features on/off) with **feature-level sparsity** (driving individual coefficients to zero).

⚠️ **Note:** This project is a **work in progress**. It is currently intended for learning and demonstration purposes, and is **not yet optimized for performance or production use**. Further improvements are planned.

The current solver uses **coordinate descent** with group-wise and feature-wise updates and runs on **synthetic test data** to demonstrate coefficient recovery.

## Key Points
- From-scratch implementation using only `numpy` and `scipy` (no external ML frameworks).
- Modular design:
  - `feature_update.py` – single-coordinate updates within a group
  - `group_update.py` – group-level updates and coordination
  - `group_zero_test.py` – fast test to zero-out entire groups
  - `solver.py` – overall optimization loop
  - `main.py` – minimal runnable demo on synthetic data
- Clear foundation for extension to real data and benchmarks.

## Quick Start
```bash
# (optional) create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install dependencies
pip install numpy scipy

# run the synthetic demo
python main.py
```

## Status & Roadmap
This is an **active, evolving project**. It is not fully optimized yet. Planned improvements include:
- Apply to real datasets (text, genetics, or finance-style grouped features).
- Improve optimization speed and numerical stability.
- Add benchmarks vs. Lasso and Group Lasso; include timing and accuracy metrics.
- Expand documentation and add plots of learned coefficients.

—
*Author: Your Name* • *License: MIT (adjust as needed)*
