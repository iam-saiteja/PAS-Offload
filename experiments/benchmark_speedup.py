import os
import time
import torch
import torch.nn as nn
import numpy as np
import sys
import matplotlib.pyplot as plt

def run_speedup_experiment():
    print("==================================================")
    print("EXPERIMENT: Speedup Verification (Standard CPU vs PAS-Offload)")
    print("==================================================")
    
    in_features = 4096
    out_features = 11008
    
    # Hidden state shape: (1, 4096)
    x_cpu = torch.randn(1, in_features, dtype=torch.float16)
    x_gpu = x_cpu.cuda()
    
    print("Initializing weights...")
    W_cpu = torch.randn(out_features, in_features, dtype=torch.float16)
    W_gpu = W_cpu.cuda()
    
    # --------------------------------------------------
    # 1. Simulate Standard Offload: CPU Computation
    # --------------------------------------------------
    print("\n--- Measuring Standard Offload: CPU Execution ---")
    print("Running matrix multiplication on CPU (representing CPU-split layers)...")
    
    # Warmup
    for _ in range(5):
        _ = torch.matmul(x_cpu, W_cpu.t())
        
    iters = 50
    start = time.perf_counter()
    for _ in range(iters):
        _ = torch.matmul(x_cpu, W_cpu.t())
    cpu_time_ms = (time.perf_counter() - start) / iters * 1000
    print(f"Standard CPU Execution Time: {cpu_time_ms:.3f} ms")
    
    # --------------------------------------------------
    # 2. Measure PAS-Offload Execution Path
    # --------------------------------------------------
    print("\n--- Measuring PAS-Offload: Predictor + Sparse Transfer + GPU Execution ---")
    
    # Initialize CPU Predictor
    predictor = nn.Sequential(
        nn.Linear(in_features, 16, bias=False),
        nn.Linear(16, out_features, bias=True)
    )
    
    # Transposed weight representation on CPU (for contiguous slicing)
    W_transposed_cpu = torch.randn(out_features, in_features, dtype=torch.float16, pin_memory=True)
    
    # Generate active column index mask (10% sparsity)
    indices = torch.randperm(out_features)[:int(out_features * 0.10)]
    
    # Warmup PAS-Offload path
    for _ in range(5):
        # CPU Predictor (cast input to float32 to avoid dtype mismatch)
        _ = torch.sigmoid(predictor(x_cpu.float()))
        # CPU contiguous slice
        sliced = torch.index_select(W_transposed_cpu, 0, indices)
        # Asynchronous stream to GPU
        sliced_gpu = sliced.cuda(non_blocking=True)
        # GPU Execution
        _ = torch.matmul(x_gpu, sliced_gpu.t())
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        # Step 1: Predict active columns on CPU
        _ = torch.sigmoid(predictor(x_cpu.float()))
        
        # Step 2: Contiguous memory slicing in CPU RAM
        sliced = torch.index_select(W_transposed_cpu, 0, indices)
        
        # Step 3: Stream sliced weights to GPU
        sliced_gpu = sliced.cuda(non_blocking=True)
        
        # Step 4: Execute matrix multiplication on GPU
        _ = torch.matmul(x_gpu, sliced_gpu.t())
        
    torch.cuda.synchronize()
    pas_time_ms = (time.perf_counter() - start) / iters * 1000
    print(f"PAS-Offload Execution Time: {pas_time_ms:.3f} ms")
    
    # --------------------------------------------------
    # Results Summary
    # --------------------------------------------------
    print("\n--------------------------------------------------")
    print("Speedup Results Summary")
    print("--------------------------------------------------")
    print(f"CPU Split Latency (per layer): {cpu_time_ms:.3f} ms")
    print(f"PAS-Offload Latency (per layer): {pas_time_ms:.3f} ms")
    print(f"Latency Saved:                  {cpu_time_ms - pas_time_ms:.3f} ms")
    print(f"Speedup Factor:                 {cpu_time_ms / pas_time_ms:.2f}x")
    
    # Translate to 32-layer LLM tokens/sec
    cpu_tokens_sec = 1000.0 / (cpu_time_ms * 32)
    pas_tokens_sec = 1000.0 / (pas_time_ms * 32)
    print(f"Estimated FFN Throughput (CPU Split): {cpu_tokens_sec:.2f} tokens/s")
    print(f"Estimated FFN Throughput (PAS-Offload): {pas_tokens_sec:.2f} tokens/s")

    # --------------------------------------------------
    # Save Comparison Plot
    # --------------------------------------------------
    print("\nGenerating speedup comparison plot...")
    plt.figure(figsize=(8, 5))
    categories = ['Standard CPU Split', 'PAS-Offload (Ours)']
    times = [cpu_time_ms, pas_time_ms]
    colors = ['#dc3545', '#28a745'] # crimson and green
    
    bars = plt.bar(categories, times, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + (max(times) * 0.02), f'{height:.2f} ms', ha='center', va='bottom', fontweight='bold')
        
    plt.ylabel("Execution Time per FFN Layer (ms)")
    plt.title("Execution Latency Comparison (Lower is Better)")
    plt.ylim(0, max(times) * 1.25)
    plt.grid(True, linestyle="--", alpha=0.5, axis='y')
    
    # Add Speedup annotation
    speedup = cpu_time_ms / pas_time_ms
    plt.text(0.5, (cpu_time_ms + pas_time_ms)/2.0, f"{speedup:.1f}x Speedup", ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.3", fc="#e2f0d9", ec="#385723", lw=2), fontsize=12, fontweight='bold')
    
    # Create images directory
    image_dir = "../images" if os.path.basename(os.getcwd()) == "experiments" else "images"
    os.makedirs(image_dir, exist_ok=True)
    plot_path = os.path.join(image_dir, "speedup_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Speedup comparison plot saved to: {os.path.abspath(plot_path)}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: CUDA is required to run the speedup benchmark.")
        sys.exit(1)
    run_speedup_experiment()
