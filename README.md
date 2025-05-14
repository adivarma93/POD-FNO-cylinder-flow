
# Flow Past a Cylinder: Prediction with POD + FNO

This repository implements a workflow for simulating 2D flow past a cylinder using the Lattice Boltzmann Method (LBM), applying Proper Orthogonal Decomposition (POD) for dimensionality reduction, and then training a Fourier Neural Operator (FNO) to predict the temporal evolution of the POD coefficients. This allows for efficient forecasting of the flow field.

## Overview

The core idea is to leverage the strengths of both classical model order reduction techniques and modern deep learning:

1.  **Data Generation (LBM):** A 2D Lattice Boltzmann Method (D2Q9 model) simulates the unsteady flow around a cylinder at a specified Reynolds number. Velocity field snapshots are collected.
2.  **Dimensionality Reduction (POD):** Proper Orthogonal Decomposition is applied to the collected snapshots. This extracts a set of optimal orthogonal basis functions (POD modes) that capture the most energetic spatial structures in the flow. The high-dimensional flow field is then projected onto these modes, yielding a low-dimensional set of time-varying coefficients.
3.  **Time-Series Prediction (FNO):** A 1D Fourier Neural Operator (FNO1d) is trained to predict the future POD coefficients based on a sequence of past coefficients. FNOs are well-suited for learning operators in function spaces and can effectively capture complex temporal dynamics.
4.  **Flow Reconstruction:** Future flow fields can be reconstructed by multiplying the FNO-predicted coefficients with the corresponding POD modes and adding back the mean flow.

This hybrid approach aims to achieve faster-than-real-time prediction while maintaining reasonable accuracy.

## Features

*   **Lattice Boltzmann Method (LBM) Simulation:**
    *   D2Q9 model for 2D flow.
    *   Zou-He velocity inlet, extrapolation outlet, bounce-back walls, and cylinder boundary conditions.
    *   GPU acceleration using CuPy (with NumPy fallback for CPU).
    *   Configurable Reynolds number, grid resolution, and simulation parameters.
*   **Proper Orthogonal Decomposition (POD):**
    *   Snapshot-based POD using Singular Value Decomposition (SVD).
    *   Selection of modes based on energy capture threshold.
    *   Standardization of temporal coefficients.
*   **Fourier Neural Operator (FNO):**
    *   `FNO1d` from the `neuralop` library for predicting POD coefficient time series.
    *   Configurable FNO architecture (modes, width, layers).
    *   Training using PyTorch with GPU support.
*   **Visualization:**
    *   LBM flow field visualization.
    *   POD mode energy spectrum and temporal coefficient plots.
    *   FNO training loss curves.
    *   Comparison of true and predicted reconstructed flow fields.

## Requirements

*   Python 3.8+
*   NumPy
*   CuPy (Optional, for GPU acceleration of LBM. Ensure CUDA toolkit is installed and compatible.)
    *   Install with e.g., `pip install cupy-cuda12x` (adjust `cudaXX` based on your CUDA version)
*   PyTorch (with CUDA support for GPU acceleration of FNO)
    *   Install from [pytorch.org](https://pytorch.org/)
*   `neuralop`
    *   `pip install neuralop`
*   Scikit-learn
*   Matplotlib
*   tqdm

You can install most dependencies using:
```bash
pip install numpy torch torchvision torchaudio neuralop scikit-learn matplotlib tqdm
# For CuPy, see specific instructions for your CUDA version (e.g., pip install cupy-cuda12x)
