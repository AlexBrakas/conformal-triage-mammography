""" Parameter count and inference latency for each architecture. """
import time
import torch
from model import BaselineCNN, HybridDualTopology, ViTOnlyBranch


# edge-deployment cost of each model
def run_hardware_benchmarks():
    """ Prints total parameters and mean latency per model.
        inputs: none
        outputs: none (printed)
    """
    print("--- Booting Hardware Diagnostics ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target Device: {device.type.upper()}")

    # batch of 1, 1 channel, 224x224
    dummy_tensor = torch.randn(1, 1, 224, 224).to(device)

    models_to_test = [
        ("Baseline CNN", BaselineCNN().to(device)),
        ("ViT-Only", ViTOnlyBranch().to(device)),
        ("Hybrid ViT", HybridDualTopology().to(device))
    ]

    for name, target_model in models_to_test:
        target_model.eval()

        # frozen + trainable
        total_params = sum(p.numel() for p in target_model.parameters())

        # warm-up so GPU kernels reach steady state
        with torch.no_grad():
            for _ in range(20):
                _ = target_model(dummy_tensor)

        # 100 timed passes; sync so queued GPU work is counted
        if device.type == "cuda": torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                _ = target_model(dummy_tensor)
        if device.type == "cuda": torch.cuda.synchronize()
        end_time = time.perf_counter()

        avg_latency_ms = ((end_time - start_time) / 100) * 1000

        print(f"\n[{name} Metrics]")
        print(f"Total Parameters:  {total_params:,}")
        print(f"Inference Latency: {avg_latency_ms:.2f} ms / tensor")

if __name__ == "__main__":
    run_hardware_benchmarks()
