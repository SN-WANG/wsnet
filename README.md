# wsnet: A Deep Learning Library for Engineering Surrogate Modeling

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**wsnet** is an integrated deep learning library specifically designed for **high-fidelity surrogate modeling in engineering applications**. It contains both **classical response surface algorithms**, **standard deep learning algorithms** and **modern neural operator algorithms**, providing a unified pipeline for **fluid dynamics emulation**, **structural analysis**, and **design optimization**.

---

## 🏗 System Architecture

The library is organized into three primary pillars, ensuring a clear separation between algorithms, utilities, and applications.

### 1. Algorithms (`nets/`)
A modular repository of algorithms categorized by their mathematical formulation:
* **`surfaces/`**: Classical response surface algorithms.
    * Includes: **PRS** (Polynomial Response Surface), **RBF** (Radial Basis Function), **KRG** (Kriging), and **SVR** (Support Vector Regression).
* **`baselines/`**: Standard deep learning algorithms.
    * Includes: **MLP** (Multi-Layer Perceptron).
* **`operators/`**: Modern neural operator algorithms.
    * Includes: **DeepONet** (Deep Operator Network), **GeoFNO** (Geometry-aware Fourier Neural Operator).

### 2. Utilities (`utils/`)
Production-grade utilities tailored for physical datasets:
* **`DoE`**: Implementation of **LHS** (Latin Hypercube Sampling) with Maximin distance optimization for maximum information gain in parameter Design of Experimnets.
* **`Engine`**: Fully encapsulated **training pipeline** serving as a "One-Stop" solution for DL workflows. It features:
    * **TensorScaler**: Channel-wise normalization tailored for multi-physics fields.
    * **AutoregressiveTrainer**: Trainer specialized for sequence rollout with **pushforward logic** and **noise injection** to ensure long-term spatio-temporal stability.
* **`CFDParser`**: Automated **data ETL (Extract, Transform and Load) pipeline** for **ANSYS Fluent** exports. It handles raw `.txt` exports, supports spatial/temporal subsampling, and implements high-speed `.pt` caching.
* **`CFDRender`**: Automated **visualization pipeline** for CFD results, generating side-by-side animations of Ground Truth, Prediction, and Absolute Error.

### 3. Applications (`apps/`)
End-to-end research workflows demonstrating the application of `wsnet` to complex physical domains. These applications provide best practices for integrating datasets, models, and trainers to solve production-level problems.

* **`HyperFlow-Net` (Fluid Dynamics Emulation)**: 
    * **Application**: Real-time emulation of hydrogen energy pipelines driven by extreme high-pressure differentials.
    * **Methodology**: Utilizes **GeoFNO** for mesh-independent mapping to capture non-linear shock wave propagation and rapid pressure transients in complex piping topologies.
* **`FId-Net` (Structural Analysis & Inverse Problems)**: 
    * **Application**: Force Identification (FId) for plate structures.
    * **Methodology**: A deep learning approach for solving **Inverse Problems**. It processes multi-sensor vibration/strain data to reconstruct the magnitude and spatial coordinates of external impact loads with high precision.
* **`AeroOpt-Solver` (Aerospace Design Optimization)**: 
    * **Application**: High-fidelity aerodynamic design optimization for aerospace components.
    * **Methodology**: Leverages **wsnet**'s surrogate models (KRG/RBF) and DoE utilities to accelerate the optimization loop, significantly reducing the computational cost compared to traditional CFD-based adjoint methods.

---

## 🚀 Key Features

* **CFD-Ready Pipeline**: Direct ingestion of Fluent data with automatic coordinate and field mapping (Vx, Vy, Vz, P, T).
* **Physics-Inspired Training**: The engine supports **Curriculum Learning** for rollout steps, allowing models to learn short-term dynamics before tackling long-term trajectories.
* **Optimization-Driven DoE**: Generate space-filling designs using optimized LHS to maximize information gain in the parameter space.
* **Quality Inspection**: Integrated rendering tools to monitor model performance across multiple physical fields simultaneously.

---

## 🛠 Quick Start

### Data Preparation
Place your ANSYS Fluent `.txt` exports in the `dataset/` folder following the `case_0001/` naming convention.

### Training an Autoregressive Model
```python
import torch
from wsnet.utils.CFDParser import CFDataset
from wsnet.utils.Engine import AutoregressiveTrainer
from wsnet.nets.operators.GeoFNO import GeoFNO # Example

# 1. Load Data
train_data, val_data, _ = CFDataset.build_datasets(data_dir='./dataset', spatial_dim=2)

# 2. Initialize Model & Trainer
model = GeoFNO(...)
trainer = AutoregressiveTrainer(
    model=model,
    output_dir='./runs'
    pushforward_steps=5, 
    noise_std=0.005,
)

# 3. Fit
trainer.fit(train_loader, val_loader)
