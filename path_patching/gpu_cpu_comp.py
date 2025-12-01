import time
from IOI_pathpatching_gpu import IOIConfig, run_ioi


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