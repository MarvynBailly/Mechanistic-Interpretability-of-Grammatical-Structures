from IOI_pathpatching_gpu import IOIConfig, run_ioi

# Default config (auto device, 2000 examples)
cfg = IOIConfig()
# run main function
run_ioi(cfg)

# cfg_test = IOIConfig(n_examples=100, n_heatmap_examples=50)
# run_ioi(cfg_test)