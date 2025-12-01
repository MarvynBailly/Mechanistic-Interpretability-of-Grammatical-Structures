# Mechanistic Interpretability of Grammatical Structures

This project implements path patching experiments for analyzing the Indirect Object Identification (IOI) task using GPT-2 small, based on the mechanistic interpretability framework.


### IOI Sentences
A sentence containing IOI begins with an initial dependent clause (e.g., "When Mary and John went to the store,") and ends with a main clause (e.g., "John gave a drink to Mary"). The sentence will introduce a subject (S) and an indirect object (IO) in the dependent clause. In the main clause, the subject will preform an action onto the indirect object (e.g., "John gave a drink to Mary" has John as S and Mary as IO). 

### LLM Prediction Task
Now given an IOI sentence, we remove the finally word(this is the indirect object, e.g., "When Mary and John went to the store, John gave a drink to") and ask the language model to predict the next token. The model should predict the indirect object rather than the repeated subject (e.g., "John gave a drink to Mary" not "John gave a drink to John"). We will refer to the first occurance of the subject as S1 (in the dependent clause) and the second occurance of the subject as S2 (in the main clause). Thus an example IOI sentence is:

- "When IO and S1 went to the store, S2 gave a drink to" 
- model should predict IO rather than S.

The goal of this project is to understand how a small LLM is able to solve this task using mechanistic interpretability methods.

### Transformer Architecture
We will use GPT-2 small, a decoder-only transformer with 12 layers and 12 attention heads per attention layer.

### Propused Circuit for IOI
The authors of "Interpretability in the Wild" propose a circuit for how GPT-2 small solves the IOI task (quotes and images taken from this paper for this subsection). To understand, let's take the example "When Mary and John went to the store, John gave a drink to". A human-interpretable algorithm for solving IOI:

1. Identify all previous names in the sentence (Mary, John, John)
2. Remove the duplicate names (John)
3. Output the last remaining name (Mary) 

The paper proposes that GPT-2 small implements a similar algorithm using attention heads. The primary class of heads are:
- "**Duplicate Token Heads**, identify tokens that have already appeared in the sentence. They are active at the S2 token, attend primarily to the S1 token, and signal that token duplication has occurred by writing the position of the duplicate token."
- "**S-Inhibition Heads** remove duplicate tokens from Name Mover Heads’ attention. They are active at the END token, attend to the S2 token, and write in the query of the Name Mover Heads, inhibiting their attention to S1 and S2 tokens."
- "**Name Mover Heads** output the remaining name. They are active at END, attend to previous names in the sentence, and copy the names they attend to. Due to the S-Inhibition Heads, they attend to the IO token over the S1 and S2 tokens."

![From the "Interpretability in the Wild" paper](results/README/gpt2-circuit.png)












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

Please reference `path_pathing\example.py` and `path_pathing\gpu_cpu_comp.py` for example usage.

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


## To Do
- [x] Native speaker needs to read through translated [templates](data_generation/input/templates.json)
- [x] Native speaker needs to read through translated [words](data_generation/input/words.json)
- [ ] Understand what is going on
- [ ] Implement unmasked IOI on English dataset
- [ ] Implement unmasked IOI on Chinese dataset
- [ ] Implement masked IOI on English dataset
- [ ] Implement masked IOI on Chinese dataset
- [ ] Implement masked versus unmasked correlation code