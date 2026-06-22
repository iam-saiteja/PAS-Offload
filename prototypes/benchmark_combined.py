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

def benchmark_combined_system():
    print("==========================================")
    print("Combined System: Bit-Slicing + Column-Streaming")
    print("==========================================")
    
    in_features = 4096
    out_features = 11008
    sparsity = 0.10  # 10% active columns
    active_cols = int(out_features * sparsity)
    
    print(f"FFN Size: {in_features} x {out_features}")
    print(f"Sparsity: {sparsity*100:.0f}% ({active_cols} active columns)")
    
    # ------------------------------------------
    # 1. Baseline: Contiguous Full float16 Transfer (90.2 MB)
    # ------------------------------------------
    weights_full_cpu = torch.randn(in_features, out_features, dtype=torch.float16, pin_memory=True)
    
    # Warmup
    for _ in range(5):
        _ = weights_full_cpu.cuda(non_blocking=True)
    torch.cuda.synchronize()
    
    iters = 100
    start = time.perf_counter()
    for _ in range(iters):
        _ = weights_full_cpu.cuda(non_blocking=True)
    torch.cuda.synchronize()
    full_time = (time.perf_counter() - start) / iters * 1000
    print(f"Baseline Full Transfer (90.2 MB):       {full_time:.3f} ms")
    
    indices = torch.randperm(out_features)[:active_cols]
    
    # ------------------------------------------
    # 2. PAS-FFN (Float16 columns: 9.0 MB)
    # ------------------------------------------
    # Transposed layout (11008, 4096)
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
    sparse_f16_time = (time.perf_counter() - start) / iters * 1000
    print(f"PAS-FFN Float16 Slicing + Tx (9.0 MB):  {sparse_f16_time:.3f} ms")
    
    # ------------------------------------------
    # 3. Combined PAS-FFN + 2-bit Slicing (1.125 MB payload)
    # ------------------------------------------
    # In a 2-bit sliced representation:
    # - Each weight takes 2 bits (1/8th of a float16).
    # - We store the weights as packed bytes.
    # - A 2-bit slice of 4096 elements takes 1024 bytes (1 KB) instead of 8192 bytes (8 KB).
    # - We simulate the memory layout by allocating a packed uint8 matrix of size (11008, 512).
    #   (Since 512 bytes = 4096 elements * 2 bits / 8 bits per byte).
    
    bytes_per_column_2bit = in_features * 2 // 8  # 1024 bytes
    
    weights_2bit_packed_cpu = torch.randint(0, 256, (out_features, bytes_per_column_2bit), dtype=torch.uint8, pin_memory=True)
    
    # Warmup
    for _ in range(5):
        gathered_2bit = torch.index_select(weights_2bit_packed_cpu, 0, indices)
        _ = gathered_2bit.cuda(non_blocking=True)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        # 1. Gather active columns on host CPU (contiguous row selection on transposed packed array)
        gathered_2bit = torch.index_select(weights_2bit_packed_cpu, 0, indices)
        # 2. Transfer 2-bit packed weights to GPU (1.125 MB)
        gathered_2bit_gpu = gathered_2bit.cuda(non_blocking=True)
        # 3. Unpack/Dequantize to float16 on the GPU (Simulated by a fast bitwise shift kernel/tensor op)
        # We simulate the GPU unpacking latency (takes <0.05 ms using element-wise ops)
        _ = gathered_2bit_gpu.to(torch.float16) / 127.0
        
    torch.cuda.synchronize()
    combined_time = (time.perf_counter() - start) / iters * 1000
    print(f"Combined (2-bit PAS-FFN) Slicing + Tx (1.13 MB): {combined_time:.3f} ms")
    
    # Speedup Calculations
    print("\n------------------------------------------")
    print("Inference Speedup Performance")
    print("------------------------------------------")
    print(f"PAS-FFN (Float16) Speedup: {full_time / sparse_f16_time:.2f}x")
    print(f"Combined (2-bit PAS-FFN) Speedup: {full_time / combined_time:.2f}x")
    
    return full_time, sparse_f16_time, combined_time

def main():
    if not torch.cuda.is_available():
        print("Error: CUDA not available!")
        return
        
    benchmark_combined_system()

if __name__ == "__main__":
    main()
