import time
from IOI_pathpatching_gpu import IOIConfig, run_ioi


# ------------- Names --------------
# Single-token names (for GPT-2 tokenizer). Must include leading space.
ONE_TOKEN_NAMES = [" John"," Mary"," James"," Susan"," Robert"," Linda"]


# ------------- Templates ----------
# These are pIOI-style templates (no {PLACE}/{OBJECT}, but fixed "store"/"drink").
# [IO] is the first name, [S] is the second name and also the subject of "gave".
# The correct next token after "to" is always the IO name (non-repeated name).
TEMPLATES = [
    "{IO} and{S} went to the store,{S} gave a drink to",
    "After {IO} and{S} went to the store,{S} gave a drink to",
    "Then {IO} and{S} went to the store,{S} gave a drink to",
    "While {IO} and{S} were working at the store,{S} gave a drink to",
    "When {IO} and{S} visited the store,{S} gave a drink to",
    "After {IO} and{S} found a drink at the store,{S} gave it to",
]



def compare():
    # Force CPU
    cpu_cfg = IOIConfig(device="cpu")
    start = time.time()
    run_ioi(cpu_cfg)
    t_cpu = time.time() - start

    # Force GPU
    gpu_cfg = IOIConfig(device="cuda")
    start = time.time()
    run_ioi(gpu_cfg)
    t_gpu = time.time() - start

    print("\n=== Summary ===")
    print(f"CPU time: {t_cpu:.2f} s")
    print(f"GPU time: {t_gpu:.2f} s")
    if t_gpu > 0:
        print(f"Speedup (CPU / GPU): {t_cpu / t_gpu:.2f}x")

if __name__ == "__main__":
    compare()