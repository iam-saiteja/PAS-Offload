# /// script
# dependencies = [
#   "torch",
#   "numpy",
# ]
# ///

import time
import torch
import torch.nn as nn
import numpy as np

def benchmark_gather_transfer():
    print("==========================================")
    print("1. Transposed Gather & Transfer Benchmark")
    print("==========================================")
    
    in_features = 4096
    out_features = 11008
    sparsity = 0.10  # 10% active columns
    active_cols = int(out_features * sparsity)
    
    print(f"FFN Size: {in_features} x {out_features}")
    print(f"Target Sparsity: {sparsity*100:.0f}% ({active_cols} columns loaded)")
    
    # 1. Full Matrix CPU-to-GPU Transfer (float16: 90.2MB)
    weights_full_cpu = torch.randn(in_features, out_features, dtype=torch.float16, pin_memory=True)
    
    # Benchmark full transfer
    iters = 100
    start = time.perf_counter()
    for _ in range(iters):
        _ = weights_full_cpu.cuda(non_blocking=True)
    torch.cuda.synchronize()
    full_time = (time.perf_counter() - start) / iters * 1000
    print(f"Full transfer time (90.2 MB): {full_time:.3f} ms")
    
    indices = torch.randperm(out_features)[:active_cols]
    
    # 2. Transposed / Col-major Gather + Transfer
    weights_transposed_cpu = torch.randn(out_features, in_features, dtype=torch.float16, pin_memory=True)
    
    # Warmup
    for _ in range(5):
        gathered = torch.index_select(weights_transposed_cpu, 0, indices)
        _ = gathered.cuda(non_blocking=True)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        gathered = torch.index_select(weights_transposed_cpu, 0, indices)
        _ = gathered.cuda(non_blocking=True)
    torch.cuda.synchronize()
    col_major_time = (time.perf_counter() - start) / iters * 1000
    print(f"Col-major Sparse Gather + Transfer (9.0 MB): {col_major_time:.3f} ms")
    print(f"Raw Transfer Speedup: {full_time / col_major_time:.2f}x")
    
    return full_time, col_major_time

def evaluate_predictors():
    print("\n==========================================")
    print("2. Predictor Performance: CPU vs GPU")
    print("==========================================")
    
    dim = 4096
    hidden_dim = 11008
    
    # For token decoding, the batch size is 1, and sequence length is 1
    # Hidden state shape: (1, 4096)
    x_gpu = torch.randn(1, dim, device='cuda', dtype=torch.float32)
    x_cpu = x_gpu.cpu()
    
    # Test different ranks
    ranks = [128, 16]
    
    for rank in ranks:
        print(f"\n--- Predictor Rank {rank} Configuration ---")
        
        # GPU Predictor
        gpu_net = nn.Sequential(
            nn.Linear(dim, rank, bias=False),
            nn.Linear(rank, hidden_dim, bias=True)
        ).cuda()
        
        # Warmup GPU
        for _ in range(10):
            _ = torch.sigmoid(gpu_net(x_gpu))
        torch.cuda.synchronize()
        
        # Benchmark GPU
        iters = 500
        start = time.perf_counter()
        for _ in range(iters):
            preds = torch.sigmoid(gpu_net(x_gpu))
            # Simulate CPU synchronization (copying indices back to CPU)
            _ = preds.cpu()
        torch.cuda.synchronize()
        gpu_time = (time.perf_counter() - start) / iters * 1000
        print(f"GPU Predictor + Host Copy Latency: {gpu_time:.3f} ms")
        
        # CPU Predictor
        cpu_net = nn.Sequential(
            nn.Linear(dim, rank, bias=False),
            nn.Linear(rank, hidden_dim, bias=True)
        )
        # Warmup CPU
        for _ in range(10):
            _ = torch.sigmoid(cpu_net(x_cpu))
            
        # Benchmark CPU
        start = time.perf_counter()
        for _ in range(iters):
            _ = torch.sigmoid(cpu_net(x_cpu))
        cpu_time = (time.perf_counter() - start) / iters * 1000
        print(f"CPU Predictor Latency:             {cpu_time:.3f} ms")
        print(f"CPU Speedup vs GPU:                {gpu_time / cpu_time:.2f}x")

def main():
    if not torch.cuda.is_available():
        print("Error: CUDA not available!")
        return
        
    benchmark_gather_transfer()
    evaluate_predictors()

if __name__ == "__main__":
    main()
