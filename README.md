# Spinal Cord Hyperflexion Model

A lumped-parameter model for simulating spinal cord injury during hyperflexion trauma, implementing the mathematical framework described in the manuscript "A Lumped-Parameter Model of Spinal Cord Failure in Hyperflexion Trauma."

## Overview

This repository contains a complete implementation of a biomechanical model that investigates the transient strain patterns in spinal cord tissue during high-speed hyperflexion events. The model integrates rigid vertebral dynamics with a viscoelastic representation of spinal soft tissues to provide a mechanistic explanation for neural tissue failure in the absence of gross structural damage.

## Key Features

- **Quasi-static analysis**: Predicts failure moments and strain responses under slow loading conditions
- **Dynamic simulation**: Uses fourth-order Runge-Kutta integration to capture transient effects
- **Sensitivity analysis**: Identifies critical parameters affecting injury risk
- **Validation**: Compares predictions with experimental literature data
- **Visualization**: Generates publication-quality figures matching the manuscript

## Model Components

### Mathematical Formulation

The model is based on a second-order nonlinear ordinary differential equation:

```
I_θ φ̈(t) + C_θ φ̇(t) + (K_θ + k_c r_c²) φ(t) + K_nl φ(t)³ = M_ext(t)
```

Where:
- `φ(t)`: Flexion angle [rad]
- `I_θ`: Rotational inertia [kg·m²]
- `C_θ`: Damping coefficient [Nms/rad]
- `K_θ`: Linear rotational stiffness [Nm/rad]
- `k_c`: Cord axial stiffness [N/m]
- `r_c`: Cord moment arm [m]
- `K_nl`: Nonlinear stiffness coefficient [Nm/rad³]
- `M_ext(t)`: External moment [Nm]

### Key Parameters

| Parameter | Value | Units | Source |
|-----------|-------|-------|--------|
| K_θ | 60.0 | Nm/rad | Wilke et al. 2017 |
| C_θ | 0.12 | Nms/rad | Estimated |
| E_c | 0.3×10⁶ | Pa | Bartlett et al. 2016 |
| A_c | 3.2×10⁻⁵ | m² | Tenny & Varacallo 2023 |
| L_0 | 0.102 | m | Geometric |
| r_c | 0.019 | m | Geometric |
| ε_fail | 0.10 | - | Yamada 1970 |
| I_θ | 4.8×10⁻⁴ | kg·m² | Estimated |
| K_nl | 50.0 | Nm/rad³ | Shetye et al. 2014 |

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Batchu-Sai/traumatic-hyperflexion-model.git
cd traumatic-hyperflexion-model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from src.spinal_model import SpinalCordModel
from src.visualization import SpinalModelVisualizer

# Initialize model
model = SpinalCordModel()

# Quasi-static analysis
M_fail = model.get_quasi_static_failure_moment()
print(f"Quasi-static failure moment: {M_fail:.1f} Nm")

# Dynamic simulation
M_ext_func = model.step_input_moment(20.0)
results = model.simulate_dynamics(M_ext_func, t_final=0.1)

# Generate figures
visualizer = SpinalModelVisualizer(model)
visualizer.generate_all_figures()
```

### Running the Demo

```bash
python demo.py
```

This will:
- Run model validation tests
- Perform quasi-static and dynamic analyses
- Generate all figures from the manuscript
- Display sensitivity analysis results

### Generating Specific Figures

```python
# Quasi-static analysis
fig1 = visualizer.plot_quasi_static_analysis("quasi_static.png")

# Dynamic response
fig2 = visualizer.plot_dynamic_response(save_path="dynamic_response.png")

# Sensitivity analysis
fig3 = visualizer.plot_sensitivity_analysis("sensitivity.png")

# Model schematic
fig4 = visualizer.plot_model_schematic("schematic.png")
```

## Results

### Model Validation

| Metric | Model Prediction | Experimental Data Range |
|--------|------------------|------------------------|
| Failure Strain | 0.10 | 0.08-0.12 [Yamada 1970] |
| Failure Moment (Nm) | ≈40.0 | 23-53 [Lopez-Valdes 2011] |

### Key Findings

1. **Dynamic Overshoot**: The model predicts transient strain amplification where peak strains exceed quasi-static failure thresholds under sub-critical loading conditions.

2. **Critical Parameters**: Sensitivity analysis identifies disc-ligament stiffness (K_θ) and cord stiffness (k_c) as the most influential parameters.

3. **Computational Efficiency**: The lumped-parameter approach provides rapid parameter space exploration while maintaining physical transparency.

## Project Structure

```
spinal_cord_model/
├── src/
│   ├── __init__.py
│   ├── spinal_model.py      # Core model implementation
│   └── visualization.py     # Figure generation
├── figures/                 # Generated figures
├── data/                    # Data files (if any)
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── demo.py                  # Main demonstration script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Mathematical Implementation

### RK4 Solver

The dynamic model uses a fourth-order Runge-Kutta integration scheme:

```python
def rk4_step(self, t, phi, phi_dot, M_ext_func):
    # Calculate k1, k2, k3, k4 coefficients
    # Update state variables
    return new_phi, new_phi_dot
```

### Quasi-Static Analysis

For slow loading conditions, the equilibrium equation is:

```
M_ext = (K_θ + k_c r_c²) φ + K_nl φ³
```

### Strain Calculation

Cord strain is computed as:

```
ε(φ) = (r_c φ) / L_0
```

## Limitations

1. **Fixed ICR**: Assumes instantaneous center of rotation remains at anterior disc margin
2. **Planar Motion**: Constrains motion to sagittal plane only
3. **Linear Elastic Cord**: Assumes Hooke's Law behavior for spinal cord tissue
4. **Single Point Parameters**: Uses literature-based values rather than subject-specific data

## Future Work

- Implement translating ICR for more realistic kinematics
- Add multi-segment spinal column modeling
- Incorporate subject-specific tissue properties
- Extend to three-dimensional motion analysis
- Integrate with finite element models for validation

## Citation

If you use this code in your research, please cite:

```
Batchu, S., Al-Atrache, Z., Mossop, C. & Thomas, A.J. (2025). 
A Lumped-Parameter Model of Spinal Cord Failure in Hyperflexion Trauma. 
[Journal Name], [Volume], [Pages].
```

## Authors

- **Sai Batchu** - Department of Neurosurgery, Cooper University Health Systems
- **Zein Al-Atrache** - Department of Neurosurgery, Cooper University Health Systems  
- **Corey Mossop** - Department of Neurosurgery, Cooper University Health Systems
- **Ajith J Thomas** - Department of Neurosurgery, Cooper University Health Systems

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

The model parameters are based on experimental data from the biomechanical literature cited in the manuscript. 
