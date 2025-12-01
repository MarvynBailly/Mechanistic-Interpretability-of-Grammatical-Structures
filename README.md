# Mechanistic Interpretability of Grammatical Structures

This project implements path patching experiments for analyzing the Indirect Object Identification (IOI) task using GPT-2 small, based on the mechanistic interpretability framework.

## Setup

### Environment Setup (Windows)

1. **Create and activate virtual environment:**
   ```powershell
   # One-time setup (or to recreate environment)
   powershell -ExecutionPolicy Bypass -File .\setup_ioi_env.ps1 -EnvName "ioi-env"
   ```

2. **Activate environment for future sessions:**
   ```powershell
   . .\ioi-env\Scripts\Activate.ps1
   ```

The setup script will install all required dependencies including:
- `torch` - PyTorch for neural network operations
- `transformer-lens` - TransformerLens for model analysis
- `transformers` - Hugging Face transformers
- `einops` - Tensor operations
- `matplotlib` - Plotting and visualization

If torch isn't finding your cuda device, try overriding the torch download with:
```powershell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Basic Usage

### Running a Simple Experiment

The simplest way to run an IOI path patching experiment:

```powershell
. .\ioi-env\Scripts\Activate.ps1
py .\path_patching\example.py
```

This will:
- Load GPT-2 small model (auto-detects GPU/CPU)
- Generate 2000 IOI examples
- Evaluate model performance
- Create attention heatmaps
- Run residual path patching analysis
- Save all plots to the current directory

### Custom Configuration

Create a custom experiment with specific settings:

```python
from IOI_pathpatching_gpu import IOIConfig, run_ioi

# Configure experiment
cfg = IOIConfig(
    device="cuda",              # Force GPU (or "cpu", "mps", None for auto)
    n_examples=1000,            # Number of dataset examples
    n_heatmap_examples=100,     # Subset for attention analysis
    model_name="gpt2-small",    # Model to analyze
    seed=42,                    # Random seed for reproducibility
    output_dir="results/exp1",  # Directory to save plots
)

# Run experiment
run_ioi(cfg)
```

### Output Files

Each experiment generates the following visualizations:

- `attn_end_to_io.png` - Attention from END token to IO (indirect object) position
- `attn_end_to_s2.png` - Attention from END token to S2 (repeated subject) position
- `attn_end_io_minus_s2.png` - Difference heatmap showing IO preference
- `resid_patching_heatmap.png` - Residual stream path patching effects


## Project Structure

```
├── path_patching/
│   ├── IOI_pathpatching_gpu.py    # Main experiment implementation
│   ├── utils.py                    # Dataset and evaluation utilities
│   ├── plotting.py                 # Visualization functions
│   ├── example.py                  # Basic usage example
│   ├── gpu_cpu_comp.py            # GPU vs CPU benchmark
│   └── example_with_output_dir.py # Custom output directory demo
├── data_generation/
│   ├── generate_dataset.py         # Dataset generation scripts
│   └── input/                      # Templates and word lists
├── requirements.txt                # Python dependencies
├── setup_ioi_env.ps1              # Environment setup script
└── .gitignore                     # Git ignore rules (includes ioi-env/)
```

## What is IOI (Indirect Object Identification)?

The IOI task tests whether language models can correctly identify indirect objects in sentences like:
- "When **Mary** and John went to the store, John gave a drink to [**Mary**]"

The model must predict the indirect object (IO = Mary) rather than the repeated subject (S = John). Path patching helps identify which model components are responsible for this behavior.

## To Do
- [x] Native speaker needs to read through translated [templates](data_generation/input/templates.json)
- [x] Native speaker needs to read through translated [words](data_generation/input/words.json)
- [ ] Implement unmasked IOI on English dataset
- [ ] Implement unmasked IOI on Chinese dataset
- [ ] Implement masked IOI on English dataset
- [ ] Implement masked IOI on Chinese dataset
- [ ] Implement masked versus unmasked correlation code