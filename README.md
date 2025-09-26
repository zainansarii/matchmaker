# Matchmaker™ 💞

A GPU-accelerated user/user matching engine with evaluation framework designed for building recommendation systems and analysing user interaction patterns.


## System Requirements

**⚠️ IMPORTANT: This library requires a Linux system with NVIDIA GPU and proper drivers ⚠️**

### Hardware Requirements
- Linux operating system (Ubuntu 20.04+, RHEL 8+, or equivalent with glibc>=2.28)
- NVIDIA GPU with compute capability 7.0+ (Volta™ architecture or newer)
- Compatible GPUs include: RTX 20xx/30xx/40xx series, Tesla V100+, A100, H100, L4, etc.
- At least 8GB GPU RAM / 16GB system RAM recommended

### CUDA & NVIDIA Driver Requirements

Based on [RAPIDS official requirements](https://docs.rapids.ai/install/):

- **CUDA 12** with Driver 525.60.13 or newer
- **CUDA 13** compatibility coming soon

**Check your current setup:**
```bash
nvidia-smi  # Should show CUDA Version 12.0+ in top right
```

### NVIDIA Driver Installation

**Only install drivers if `nvidia-smi` doesn't work or shows older CUDA version**

#### Ubuntu/Debian Systems:
```bash
# Update package list
sudo apt update

# Install NVIDIA drivers (automatically selects appropriate version)
sudo ubuntu-drivers autoinstall
# OR install specific version
ubuntu-drivers devices  # Check available drivers
sudo apt install nvidia-driver-XXX  # Replace XXX with version number

# Reboot system (REQUIRED!)
sudo reboot
```

#### RHEL/CentOS/Rocky Linux:
```bash
# Install EPEL repository
sudo dnf install epel-release

# Install NVIDIA drivers
sudo dnf install nvidia-driver nvidia-settings

# Reboot system (REQUIRED!)
sudo reboot
```

#### Verify Driver Installation
```bash
nvidia-smi  # Should show GPU info and CUDA version 12.0+
```

## Getting started (development)

**Prerequisites: Linux system with NVIDIA GPU and drivers supporting CUDA 12+**

### Install Miniforge (Recommended by RAPIDS)

RAPIDS recommends using Miniforge over Miniconda/Anaconda for better compatibility:

```bash
# Download and install Miniforge
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash "Miniforge3-$(uname)-$(uname -m).sh"

# Follow installation prompts and enable conda
source ~/miniforge3/bin/activate

# Close and reopen terminal
```

### Set up the development environment

```bash
git clone https://github.com/zainansarii/matchmaker.git
cd matchmaker

# Create environment using official RAPIDS pattern
conda env create -f environment.yml
conda activate matchmaker-dev
```

The environment.yml is based on the [official RAPIDS installation guide](https://docs.rapids.ai/install/) and installs:
- RAPIDS 25.08 with CUDA 12.x support
- Python 3.10
- NetworkX + nx-cugraph for graph processing
- JupyterLab for development
- All necessary CUDA dependencies

### Verify Installation

```bash
# Test NVIDIA drivers and CUDA
nvidia-smi

# Test RAPIDS installation
python -c "import cudf; print(cudf.Series([1, 2, 3]))"

# Test CuPy (included with RAPIDS)
python -c "import cupy; print(f'CuPy version: {cupy.__version__}'); print(f'CUDA device: {cupy.cuda.device.Device().id}')"

# Test matchmaker import
python -c "from matchmaker import matchmaker; print('Matchmaker imported successfully')"
```

**Important Notes**:
- The environment uses official RAPIDS 25.08 with CUDA 12.x support
- Compatible with NVIDIA drivers 525.60.13 or newer
- Requires compute capability 7.0+ GPU (Volta™ architecture or newer)
- For cloud instances (AWS, GCP, Azure), choose GPU-enabled instances with pre-installed drivers
- Windows users should use WSL2 with Ubuntu 20.04+

## Basic Usage

```python
from matchmaker import matchmaker

# Create a matching engine
engine = matchmaker.MatchingEngine()

# Load interaction data (requires CSV with appropriate columns)
engine.load_interactions("path/to/your/data.csv", 
  decider_col='user_id', 
  other_col='other_user_id', 
  like_col='liked', 
  timestamp_col='timestamp')
```
