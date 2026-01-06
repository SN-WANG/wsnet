# wsnet: A Deep Learning Library for Surrogate Modeling in Mechanical Engineering

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**wsnet** is an integrated deep learning framework specifically designed for mechanical engineering applications, focusing on **Surrogate Modeling**, **Fluid Dynamics (CFD) Emulation**, and **Neural Operators**. It provides a seamless pipeline from raw simulation data to high-fidelity spatio-temporal predictions.

---

## 🏗 Project Architecture

The library is organized into two core modules: `nets` (Model Zoo) and `utils` (Engineering Toolkit).

### 1. Model Zoo (`nets/`)
* **`operators/`** (Primary Research): Advanced Neural Operators for learning mapping between function spaces.
    * **GeoFNO**: Geometry-aware Fourier Neural Operators.
    * **DeepONet**: Learning operators via branch and trunk network architectures.
* **`surfaces/`**: Classical surrogate models for response surface methodology.
    * Includes: **PRS** (Polynomial Response Surface), **RBF** (Radial Basis Function), **KRG** (Kriging), and **SVR** (Support Vector Regression).
* **`baselines/`**: Standard neural architectures.
    * **MLP**: Multi-Layer Perceptron for fundamental regression tasks.

### 2. Engineering Toolkit (`utils/`)
* **`Engine.py`**: A robust training engine featuring:
    * **AutoregressiveTrainer**: Specialized for sequence rollout with pushforward logic and noise injection to ensure long-term stability.
    * **TensorScaler**: Channel-wise normalization utilities for physical fields.
* **`CFDParser.py`**: Automated data pipeline for **ANSYS Fluent**. It handles raw `.txt` exports, supports spatial/temporal subsampling, and implements `.pt` caching for efficient I/O.
* **`CFDRender.py`**: High-performance visualization for CFD results, generating side-by-side animations of Ground Truth, Prediction, and Absolute Error.
* **`DoE.py`**: **Design of Experiments** via Latin Hypercube Sampling (LHS) with Maximin distance optimization.

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
