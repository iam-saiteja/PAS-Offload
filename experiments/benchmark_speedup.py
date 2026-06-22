import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Add root folder to python path to import pas_offload
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pas_offload.engine import PASOffloadEngine

def run_speedup_experiment():
    print("======================================================================")
    print("EXPERIMENT: Speedup Verification (CPU vs Sequential vs Pipelined)")
    print("======================================================================")
    
    in_features = 4096
    out_features = 11008
    iters = 100
    
    # Initialize CPU-side input tensors
    xs_cpu = [torch.randn(1, in_features, dtype=torch.float16) for _ in range(iters)]
    xs_gpu = [x.cuda() for x in xs_cpu]
    
    print("Initializing weights...")
    W_cpu = torch.randn(out_features, in_features, dtype=torch.float16)
    W_gpu = W_cpu.cuda()
    
    # --------------------------------------------------
    # 1. Standard CPU Execution (representing CPU Split)
    # --------------------------------------------------
    print("\n--- 1. Measuring Standard CPU Execution ---")
    
    # Warmup
    for _ in range(5):
        _ = torch.matmul(xs_cpu[0], W_cpu.t())
        
    start = time.perf_counter()
    for i in range(iters):
        _ = torch.matmul(xs_cpu[i], W_cpu.t())
    cpu_time_ms = (time.perf_counter() - start) / iters * 1000
    print(f"Standard CPU Execution Time: {cpu_time_ms:.3f} ms")
    
    # --------------------------------------------------
    # 2. Baseline PAS-Offload (Sequential GPU)
    # --------------------------------------------------
    print("\n--- 2. Measuring Baseline PAS-Offload (Sequential) ---")
    engine = PASOffloadEngine(in_features, out_features, rank=16)
    engine.load_weights(W_cpu)
    
    # Warmup
    for _ in range(5):
        _ = engine.forward(xs_gpu[0], threshold=0.15, top_k=1100)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for i in range(iters):
        _, _ = engine.forward(xs_gpu[i], threshold=0.15, top_k=1100)
    torch.cuda.synchronize()
    v1_time_ms = (time.perf_counter() - start) / iters * 1000
    print(f"Sequential PAS-Offload Time:    {v1_time_ms:.3f} ms")
    
    # --------------------------------------------------
    # 3. Optimized PAS-Offload (Pipelined Double-Buffered)
    # --------------------------------------------------
    print("\n--- 3. Measuring Optimized PAS-Offload (Pipelined) ---")
    
    # Warmup pipelined path
    ticket = engine.submit_forward(xs_gpu[0], threshold=0.15, buffer_idx=0, top_k=1100, x_cpu_hint=xs_cpu[0])
    for i in range(1, 5):
        buf_idx = i % 2
        next_ticket = engine.submit_forward(xs_gpu[i], threshold=0.15, buffer_idx=buf_idx, top_k=1100, x_cpu_hint=xs_cpu[i])
        _ = engine.execute_forward(xs_gpu[i-1], ticket)
        ticket = next_ticket
    _ = engine.execute_forward(xs_gpu[4], ticket)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    # Pipeline loop
    # Submit first layer/token
    ticket = engine.submit_forward(xs_gpu[0], threshold=0.15, buffer_idx=0, top_k=1100, x_cpu_hint=xs_cpu[0])
    for i in range(1, iters):
        buf_idx = i % 2
        # Asynchronously schedule prediction, slicing, and transfer of L + 1
        next_ticket = engine.submit_forward(xs_gpu[i], threshold=0.15, buffer_idx=buf_idx, top_k=1100, x_cpu_hint=xs_cpu[i])
        
        # Execute layer L
        _ = engine.execute_forward(xs_gpu[i-1], ticket)
        
        ticket = next_ticket
        
    # Execute the final token/layer
    _ = engine.execute_forward(xs_gpu[iters-1], ticket)
    torch.cuda.synchronize()
    v2_time_ms = (time.perf_counter() - start) / iters * 1000
    print(f"Pipelined PAS-Offload Time: {v2_time_ms:.3f} ms")
    
    # --------------------------------------------------
    # Results Summary
    # --------------------------------------------------
    print("\n" + "=" * 50)
    print("Performance Comparison Results")
    print("=" * 50)
    print(f"Standard CPU Latency:         {cpu_time_ms:.3f} ms")
    print(f"Sequential PAS-Offload Latency: {v1_time_ms:.3f} ms (V1 Speedup: {cpu_time_ms / v1_time_ms:.2f}x)")
    print(f"Pipelined PAS-Offload Latency:  {v2_time_ms:.3f} ms (V2 Speedup: {cpu_time_ms / v2_time_ms:.2f}x)")
    print("-" * 50)
    print(f"Pipelined speedup over CPU:    {cpu_time_ms / v2_time_ms:.2f}x")
    print("=" * 50)
    
    # --------------------------------------------------
    # Save Comparison Plot
    # --------------------------------------------------
    print("\nGenerating speedup comparison plot...")
    plt.figure(figsize=(9, 5.5))
    categories = ['Standard CPU Split', 'PAS-Offload (Sequential)', 'PAS-Offload (Pipelined)']
    times = [cpu_time_ms, v1_time_ms, v2_time_ms]
    colors = ['#dc3545', '#ffc107', '#28a745'] # red, amber, green
    
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
    speedup = cpu_time_ms / v2_time_ms
    plt.text(1.0, max(times) * 0.7, f"{speedup:.1f}x Total Speedup", ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.4", fc="#e2f0d9", ec="#385723", lw=2.5), fontsize=12, fontweight='bold')
    
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
