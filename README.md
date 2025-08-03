# Traumatic Hyperflexion Model

A comprehensive biomechanical model for simulating spinal cord injury during hyperflexion trauma, implemented in Python with publication-quality visualization.

## Overview

This repository contains a lumped-parameter biomechanical model that simulates spinal cord injury during hyperflexion trauma. The model includes:

- **Quasi-static analysis** for equilibrium conditions
- **Dynamic simulation** with RK4 integration
- **Sensitivity analysis** for parameter variations
- **Publication-quality figures** with professional formatting
- **Comprehensive validation** against experimental data

## Model Components

### Core Biomechanical Elements
- **Disc-Ligament Complex**: Rotational stiffness (K_θ) and damping (C_θ)
- **Spinal Cord Element**: Axial stiffness (k_c) and moment arm (r_c)
- **Nonlinear Stiffness**: Higher-order terms for realistic behavior
- **Failure Criterion**: Strain-based failure threshold (ε_fail)

### Mathematical Formulation
The model is described by a second-order nonlinear ordinary differential equation:

```
I_θ * φ̈ + C_θ * φ̇ + (K_θ + k_c*r_c²)*φ + K_nl*φ³ = M_ext
```

Where:
- `φ`: Flexion angle [rad]
- `M_ext`: External moment [Nm]
- `I_θ`: Rotational inertia [kg·m²]
- `K_θ`: Disc-ligament stiffness [Nm/rad]
- `C_θ`: Damping coefficient [Nms/rad]
- `k_c`: Cord axial stiffness [N/m]
- `r_c`: Cord moment arm [m]

## Features

### Analysis Capabilities
- **Quasi-static analysis** for slow loading conditions
- **Dynamic simulation** with RK4 integration
- **Sensitivity analysis** for all key parameters
- **Failure prediction** with strain-based criteria
- **Publication-quality figures** with professional formatting

### Visualization
- **Figure 1**: Quasi-static strain response
- **Figure 2**: Dynamic response to step input
- **Figure 3**: Comprehensive sensitivity analysis
- **Figure 4**: Model schematic diagram

### Validation
- **Experimental data comparison**
- **Failure strain validation** (0.08-0.12 range)
- **Failure moment validation** (23-53 Nm range)
- **Dynamic amplification factor** calculation

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager
- Git (for cloning)

### Step-by-Step Installation

#### 1. Clone the Repository
```bash
# Clone the repository to your local machine
git clone https://github.com/Batchu-Sai/traumatic-hyperflexion-model.git

# Navigate to the project directory
cd traumatic-hyperflexion-model
```

#### 2. Create Virtual Environment
```bash
# Create a virtual environment (recommended)
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

#### 3. Install Dependencies
```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

#### 4. Verify Installation
```bash
# Test that the model can be imported
python -c "from src.spinal_model import SpinalCordModel; print('Installation successful!')"
```

## Usage

### Quick Start - Run All Analyses
```bash
# Make sure your virtual environment is activated
source venv/bin/activate

# Run the main demonstration script
python demo.py
```

This will:
- Run model validation tests
- Perform quasi-static and dynamic analyses
- Generate all publication-quality figures
- Display sensitivity analysis results
- Save figures to the `figures/` directory

### Step-by-Step Usage

#### 1. Activate Virtual Environment
```bash
# Always activate your virtual environment first
source venv/bin/activate
```

#### 2. Run Individual Components

**Quasi-Static Analysis:**
```bash
python -c "
from src.spinal_model import SpinalCordModel
model = SpinalCordModel()
M_fail = model.get_quasi_static_failure_moment()
print(f'Quasi-static failure moment: {M_fail:.1f} Nm')
"
```

**Dynamic Simulation:**
```bash
python -c "
from src.spinal_model import SpinalCordModel
from src.visualization import SpinalModelVisualizer

model = SpinalCordModel()
viz = SpinalModelVisualizer(model)

# Generate dynamic response figure
fig = viz.plot_dynamic_response(save_path='figures/dynamic_response.png')
print('Dynamic response figure saved to figures/dynamic_response.png')
"
```

**Sensitivity Analysis:**
```bash
python -c "
from src.spinal_model import SpinalCordModel
from src.visualization import SpinalModelVisualizer

model = SpinalCordModel()
viz = SpinalModelVisualizer(model)

# Generate sensitivity analysis figure
fig = viz.plot_sensitivity_analysis(save_path='figures/sensitivity_analysis.png')
print('Sensitivity analysis figure saved to figures/sensitivity_analysis.png')
"
```

**Generate All Figures:**
```bash
python -c "
from src.spinal_model import SpinalCordModel
from src.visualization import SpinalModelVisualizer

model = SpinalCordModel()
viz = SpinalModelVisualizer(model)

# Generate all figures
fig1 = viz.plot_quasi_static_analysis(save_path='figures/quasi_static_analysis.png')
fig2 = viz.plot_dynamic_response(save_path='figures/dynamic_response.png')
fig3 = viz.plot_sensitivity_analysis(save_path='figures/sensitivity_analysis.png')
fig4 = viz.plot_model_schematic(save_path='figures/model_schematic.png')

print('All figures generated successfully!')
"
```

#### 3. Run Tests
```bash
# Run all unit tests
python -m pytest tests/

# Run specific test file
python tests/test_model.py

# Run with verbose output
python -m pytest tests/ -v
```

#### 4. Custom Analysis
```bash
python -c "
import numpy as np
from src.spinal_model import SpinalCordModel

# Initialize model
model = SpinalCordModel()

# Custom sensitivity analysis
param_values = np.array([30, 45, 60, 75, 90])
results = model.sensitivity_analysis('K_theta', param_values, M_ext=30.0)

print('Custom sensitivity analysis completed!')
print(f'Results: {results}')
"
```

### Advanced Usage

#### Custom Parameter Modification
```bash
python -c "
from src.spinal_model import SpinalCordModel

# Create model with custom parameters
custom_params = {
    'K_theta': 80.0,  # Increased stiffness
    'epsilon_fail': 0.12,  # Higher failure strain
    'M_ext': 35.0  # Higher external moment
}

model = SpinalCordModel(custom_params)
M_fail = model.get_quasi_static_failure_moment()
print(f'Custom model failure moment: {M_fail:.1f} Nm')
"
```

#### Batch Analysis
```bash
python -c "
import numpy as np
from src.spinal_model import SpinalCordModel

# Test multiple external moments
moments = np.array([20, 25, 30, 35, 40])
results = []

for M_ext in moments:
    model = SpinalCordModel()
    M_ext_func = model.step_input_moment(M_ext)
    sim_results = model.simulate_dynamics(M_ext_func, t_final=0.1)
    max_strain = np.max(sim_results['strain'])
    results.append({'M_ext': M_ext, 'max_strain': max_strain})

print('Batch analysis results:')
for result in results:
    print(f'M_ext: {result[\"M_ext\"]} Nm, Max Strain: {result[\"max_strain\"]:.4f}')
"
```

## Results

### Model Predictions
- **Quasi-static failure moment**: ≈40.0 Nm
- **Dynamic failure moment**: ≈28.0 Nm
- **Dynamic amplification factor**: 1.43
- **Failure strain**: 0.10 (10%)

### Sensitivity Analysis
The model includes comprehensive sensitivity analysis for all key parameters:
- **K_θ** (Disc-ligament stiffness): Most influential
- **k_c** (Cord stiffness): Small but measurable effect
- **C_θ** (Damping): Moderate influence
- **r_c** (Moment arm): Large effect on strain
- **ε_fail** (Failure strain): Critical for failure prediction

## Project Structure

```
traumatic-hyperflexion-model/
├── src/
│   ├── __init__.py
│   ├── spinal_model.py          # Core biomechanical model
│   └── visualization.py         # Figure generation
├── tests/
│   └── test_model.py           # Unit tests
├── figures/                     # Generated figures
├── docs/                        # Documentation
├── demo.py                      # Main demonstration script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── LICENSE                      # MIT License
```

## Testing

Run the comprehensive test suite:
```bash
# Activate virtual environment first
source venv/bin/activate

# Run all tests
python -m pytest tests/

# Run specific test file
python tests/test_model.py

# Run with verbose output
python -m pytest tests/ -v
```

## Validation

### Experimental Data Comparison
| Metric | Model Prediction | Experimental Range | Reference |
|--------|------------------|-------------------|-----------|
| Failure Strain | 0.10 | 0.08-0.12 | [Yamada 1970] |
| Failure Moment | ≈40.0 Nm | 23-53 Nm | [Lopez-Valdes 2011] |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Sai Batchu**
- GitHub: [@Batchu-Sai](https://github.com/Batchu-Sai)
- Repository: [traumatic-hyperflexion-model](https://github.com/Batchu-Sai/traumatic-hyperflexion-model)

## References

1. Yamada, H. (1970). *Strength of Biological Materials*. Williams & Wilkins.
2. Lopez-Valdes, H. E., et al. (2011). *Spinal Cord Injury Biomechanics*. Journal of Biomechanics.

## Acknowledgments

- Biomechanical modeling community
- Experimental validation data providers
- Open-source scientific computing tools

## Troubleshooting

### Common Issues

**Import Error: No module named 'numpy'**
```bash
# Solution: Activate virtual environment and install dependencies
source venv/bin/activate
pip install -r requirements.txt
```

**Permission Error on Windows**
```bash
# Solution: Run PowerShell as Administrator or use:
python -m pip install -r requirements.txt
```

**Figure Generation Issues**
```bash
# Solution: Ensure matplotlib backend is set correctly
python -c "import matplotlib; matplotlib.use('Agg')"
```

---

**Note**: This model is for research and educational purposes. For clinical applications, consult with medical professionals.
