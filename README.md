# WSNet: A Deep Learning Library for Engineering Surrogate Modeling

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**WSNet** is an integrated deep learning library specifically designed for **high-fidelity surrogate modeling in engineering applications**. It provides a unified pipeline for **fluid dynamics emulation**, **structural analysis**, and **design optimization** with comprehensive support for classical surrogate models, neural networks, and modern neural operator algorithms.

## 🏗 System Architecture

WSNet features a completely reorganized architecture with clear separation of concerns:

```
wsnet/
├── models/        # Surrogate models (classical, neural, multi-fidelity, ensemble)
├── training/      # Training frameworks and utilities
├── data/          # Data loading and preprocessing
├── sampling/      # Design of Experiments and infill strategies
└── utils/         # Core utilities
```

### 1. Models (`models/`)
A modular repository of surrogate models categorized by their mathematical formulation:

* **`classical/`**: Classical response surface algorithms
    * Includes: **PRS** (Polynomial Response Surface), **RBF** (Radial Basis Function), **KRG** (Kriging), **SVR** (Support Vector Regression)
* **`neural/`**: Neural network models
    * Includes: **MLP** (Multi-Layer Perceptron), **DeepONet**, **GeoFNO** (Geometry-aware Fourier Neural Operator), **U-WNO** (Universal Window-based Neural Operator)
* **`multi_fidelity/`**: Multi-fidelity models
    * Includes: **CCA-MFS**, **MFS-MLS**, **MMFS**
* **`ensemble/`**: Ensemble models
    * Includes: **T-AHS**, **AES-MSI**

### 2. Training (`training/`)
Training frameworks and utilities:

* **`base_trainer.py`**: Base trainer class for custom training workflows
* **`std_trainer.py`**: Standard trainer for static regression tasks
* **`rollout_trainer.py`**: Trainer for autoregressive sequence prediction
* **`base_criterion.py`**: Base loss functions
* **`physics_criterion.py`**: Physics-informed training support

### 3. Data (`data/`)
Data loading and preprocessing utilities:

* **`flow_data.py`**: CFD data loading and preprocessing
* **`scaler.py`**: Data scaling utilities
* **`CFDRender.py`**: CFD visualization and rendering
* **`flow_vis.py`**: Flow field visualization tools

### 4. Sampling (`sampling/`)
Design of Experiments and infill strategies:

* **`doe.py`**: Design of Experiments (LHS, optimized LHS)
* **`infill.py`**: Infill sampling strategies
* **`mico_infill.py`**: MICO infill strategy for multi-fidelity optimization

### 5. Utilities (`utils/`)
Core utilities and helper functions.

## 🚀 Key Features

* **CFD-Ready Pipeline**: Direct ingestion of ANSYS Fluent data with automatic coordinate and field mapping
* **Physics-Informed Training**: Support for physics constraints and loss functions
* **Multi-Fidelity Support**: Comprehensive multi-fidelity modeling capabilities
* **Ensemble Methods**: Advanced ensemble techniques for improved accuracy
* **Neural Operator Algorithms**: Modern neural operator implementations for complex physics
* **Comprehensive Visualization Tools**: Built-in CFD visualization and rendering
* **Standardized API**: Consistent "initialize, fit, predict" pattern across all models

## 📚 API Reference

### Core Classes

- `wsnet.models.base.Model`: Base model interface
- `wsnet.training.base_trainer.Trainer`: Base trainer interface
- `wsnet.data.flow_data.FlowData`: CFD data loader
- `wsnet.sampling.doe.lhs_design`: Optimized Latin Hypercube Sampling

### Common Patterns

The library follows a consistent "initialize, fit, predict" pattern:

```python
# Initialize
model = SomeModel(parameters)

# Fit/Train
trainer = SomeTrainer(model=model)
trainer.fit(train_data, val_data)

# Predict
predictions = model.predict(test_data)
```

## 🚀 Examples and Tutorials

Check the `examples/` directory for complete workflow examples:

- Basic regression example
- CFD emulation example
- Multi-fidelity modeling example
- Ensemble modeling example
- Sequential sampling modeling example

## ⚙️ Installation and Dependencies

### Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy
- SciPy
- Matplotlib
- PiVista
- tqdm

### Installation

```bash
git clone https://github.com/SN-WANG/wsnet.git
cd wsnet
pip install -e .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions and support, please contact:
- Shengning Wang (王晟宁) - snwang2023@163.com
- Project Website: [https://github.com/SN-WANG/wsnet](https://github.com/SN-WANG/wsnet)

## 📖 Citation

If you use WSNet in your research, please cite:

```
@software{wsnet2026,
  author = {Shengning Wang},
  title = {WSNet: A Deep Learning Library for Engineering Surrogate Modeling},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/SN-WANG/wsnet}}
}
```