import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np

# Add root folder to python path to import baseline pas_offload
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pas_offload.engine import PASOffloadEngine
from working.engine_v2 import PASOffloadEngineV2

def run_speedup_v2_experiment():
    print("======================================================================")
    print("EXPERIMENT V2: Pipelined Speedup Verification (CPU vs Baseline vs V2)")
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
    engine_v1 = PASOffloadEngine(in_features, out_features, rank=16)
    engine_v1.load_weights(W_cpu)
    
    # Warmup
    for _ in range(5):
        _ = engine_v1.forward(xs_gpu[0], threshold=0.15)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for i in range(iters):
        _, _ = engine_v1.forward(xs_gpu[i], threshold=0.15)
    torch.cuda.synchronize()
    v1_time_ms = (time.perf_counter() - start) / iters * 1000
    print(f"Baseline PAS-Offload Time:    {v1_time_ms:.3f} ms")
    
    # --------------------------------------------------
    # 3. Optimized PAS-Offload V2 (Pipelined Double-Buffered)
    # --------------------------------------------------
    print("\n--- 3. Measuring Optimized PAS-Offload V2 (Pipelined) ---")
    engine_v2 = PASOffloadEngineV2(in_features, out_features, rank=16)
    engine_v2.load_weights(W_cpu)
    
    # Warmup pipelined path
    ticket = engine_v2.submit_forward(xs_gpu[0], threshold=0.15, buffer_idx=0)
    for i in range(1, 5):
        buf_idx = i % 2
        next_ticket = engine_v2.submit_forward(xs_gpu[i], threshold=0.15, buffer_idx=buf_idx)
        _ = engine_v2.execute_forward(xs_gpu[i-1], ticket)
        ticket = next_ticket
    _ = engine_v2.execute_forward(xs_gpu[4], ticket)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    # Pipeline loop
    # Submit first layer/token
    ticket = engine_v2.submit_forward(xs_gpu[0], threshold=0.15, buffer_idx=0)
    for i in range(1, iters):
        buf_idx = i % 2
        # Asynchronously schedule prediction, slicing, and transfer of L + 1
        next_ticket = engine_v2.submit_forward(xs_gpu[i], threshold=0.15, buffer_idx=buf_idx)
        
        # Execute layer L
        _ = engine_v2.execute_forward(xs_gpu[i-1], ticket)
        
        ticket = next_ticket
        
    # Execute the final token/layer
    _ = engine_v2.execute_forward(xs_gpu[iters-1], ticket)
    torch.cuda.synchronize()
    v2_time_ms = (time.perf_counter() - start) / iters * 1000
    print(f"Optimized PAS-Offload V2 Time: {v2_time_ms:.3f} ms")
    
    # --------------------------------------------------
    # Results Summary
    # --------------------------------------------------
    print("\n" + "=" * 50)
    print("Performance Comparison Results")
    print("=" * 50)
    print(f"Standard CPU Latency:         {cpu_time_ms:.3f} ms")
    print(f"Baseline PAS-Offload Latency: {v1_time_ms:.3f} ms (V1 Speedup: {cpu_time_ms / v1_time_ms:.2f}x)")
    print(f"Optimized V2 Latency:         {v2_time_ms:.3f} ms (V2 Speedup: {cpu_time_ms / v2_time_ms:.2f}x)")
    print("-" * 50)
    print(f"V2 speedup over V1 Baseline:  {v1_time_ms / v2_time_ms:.2f}x")
    print(f"Total speedup over CPU:       {cpu_time_ms / v2_time_ms:.2f}x")
    print("=" * 50)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: CUDA is required to run the V2 speedup benchmark.")
        sys.exit(1)
    run_speedup_v2_experiment()
