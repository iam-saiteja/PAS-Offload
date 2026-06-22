import os
import sys
import time
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pas_offload.engine import PASOffloadEngine
from working.engine_v2 import PASOffloadEngineV2
from working.engine_v3 import PASOffloadEngineV3

def run_speedup_v3_experiment():
    print("======================================================================")
    print("EXPERIMENT V3: Pipelined + LUT Speedup Verification")
    print("======================================================================")
    
    in_features = 4096
    out_features = 11008
    iters = 100
    
    # Initialize CPU-side input tensors
    xs_cpu = [torch.randn(1, in_features, dtype=torch.float16) for _ in range(iters)]
    xs_gpu = [x.cuda() for x in xs_cpu]
    
    print("Initializing weights...")
    W_cpu = torch.randn(out_features, in_features, dtype=torch.float16)
    
    # --------------------------------------------------
    # 1. Standard CPU Execution
    # --------------------------------------------------
    print("\n--- 1. Measuring Standard CPU Execution ---")
    for _ in range(5):
        _ = torch.matmul(xs_cpu[0], W_cpu.t())
        
    start = time.perf_counter()
    for i in range(iters):
        _ = torch.matmul(xs_cpu[i], W_cpu.t())
    cpu_time_ms = (time.perf_counter() - start) / iters * 1000
    print(f"Standard CPU Execution Time: {cpu_time_ms:.3f} ms")
    
    # --------------------------------------------------
    # 2. Baseline PAS-Offload V1
    # --------------------------------------------------
    print("\n--- 2. Measuring Baseline PAS-Offload V1 ---")
    engine_v1 = PASOffloadEngine(in_features, out_features, rank=16)
    engine_v1.load_weights(W_cpu)
    
    for _ in range(5):
        _ = engine_v1.forward(xs_gpu[0], threshold=0.15)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for i in range(iters):
        _, _ = engine_v1.forward(xs_gpu[i], threshold=0.15, top_k=1100)
    torch.cuda.synchronize()
    v1_time_ms = (time.perf_counter() - start) / iters * 1000
    print(f"Baseline PAS-Offload Time:    {v1_time_ms:.3f} ms")

    # --------------------------------------------------
    # 3. Optimized PAS-Offload V2 (Pipelined)
    # --------------------------------------------------
    print("\n--- 3. Measuring Optimized PAS-Offload V2 (Pipelined) ---")
    engine_v2 = PASOffloadEngineV2(in_features, out_features, rank=16)
    engine_v2.load_weights(W_cpu)
    
    ticket = engine_v2.submit_forward(xs_gpu[0], threshold=0.15, buffer_idx=0, top_k=1100)
    for i in range(1, 5):
        buf_idx = i % 2
        next_ticket = engine_v2.submit_forward(xs_gpu[i], threshold=0.15, buffer_idx=buf_idx, top_k=1100)
        _ = engine_v2.execute_forward(xs_gpu[i-1], ticket)
        ticket = next_ticket
    _ = engine_v2.execute_forward(xs_gpu[4], ticket)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    ticket = engine_v2.submit_forward(xs_gpu[0], threshold=0.15, buffer_idx=0, top_k=1100)
    for i in range(1, iters):
        buf_idx = i % 2
        next_ticket = engine_v2.submit_forward(xs_gpu[i], threshold=0.15, buffer_idx=buf_idx, top_k=1100)
        _ = engine_v2.execute_forward(xs_gpu[i-1], ticket)
        ticket = next_ticket
    _ = engine_v2.execute_forward(xs_gpu[iters-1], ticket)
    torch.cuda.synchronize()
    v2_time_ms = (time.perf_counter() - start) / iters * 1000
    print(f"Optimized PAS-Offload V2 Time: {v2_time_ms:.3f} ms")

    # --------------------------------------------------
    # 4. Optimized PAS-Offload V3 (LUT + Pipelined)
    # --------------------------------------------------
    print("\n--- 4. Measuring Optimized PAS-Offload V3 (LUT) ---")
    engine_v3 = PASOffloadEngineV3(in_features, out_features, rank=16)
    engine_v3.load_weights(W_cpu)
    
    ticket = engine_v3.submit_forward(xs_gpu[0], threshold=0.15, buffer_idx=0, x_cpu_hint=xs_cpu[0], top_k=1100)
    for i in range(1, 5):
        buf_idx = i % 2
        next_ticket = engine_v3.submit_forward(xs_gpu[i], threshold=0.15, buffer_idx=buf_idx, x_cpu_hint=xs_cpu[i], top_k=1100)
        _ = engine_v3.execute_forward(xs_gpu[i-1], ticket)
        ticket = next_ticket
    _ = engine_v3.execute_forward(xs_gpu[4], ticket)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    ticket = engine_v3.submit_forward(xs_gpu[0], threshold=0.15, buffer_idx=0, x_cpu_hint=xs_cpu[0], top_k=1100)
    for i in range(1, iters):
        buf_idx = i % 2
        next_ticket = engine_v3.submit_forward(xs_gpu[i], threshold=0.15, buffer_idx=buf_idx, x_cpu_hint=xs_cpu[i], top_k=1100)
        _ = engine_v3.execute_forward(xs_gpu[i-1], ticket)
        ticket = next_ticket
    _ = engine_v3.execute_forward(xs_gpu[iters-1], ticket)
    torch.cuda.synchronize()
    v3_time_ms = (time.perf_counter() - start) / iters * 1000
    print(f"Optimized PAS-Offload V3 Time: {v3_time_ms:.3f} ms")
    
    # --------------------------------------------------
    # Results Summary
    # --------------------------------------------------
    print("\n" + "=" * 50)
    print("Performance Comparison Results")
    print("=" * 50)
    print(f"Standard CPU Latency:         {cpu_time_ms:.3f} ms")
    print(f"Baseline PAS V1 Latency:      {v1_time_ms:.3f} ms (Speedup: {cpu_time_ms / v1_time_ms:.2f}x)")
    print(f"Optimized V2 Latency:         {v2_time_ms:.3f} ms (Speedup: {cpu_time_ms / v2_time_ms:.2f}x)")
    print(f"Optimized V3 (LUT) Latency:   {v3_time_ms:.3f} ms (Speedup: {cpu_time_ms / v3_time_ms:.2f}x)")
    print("-" * 50)
    print(f"V3 speedup over V2 Baseline:  {v2_time_ms / v3_time_ms:.2f}x")
    print(f"Total speedup over CPU:       {cpu_time_ms / v3_time_ms:.2f}x")
    print("=" * 50)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: CUDA is required to run the V3 speedup benchmark.")
        sys.exit(1)
    run_speedup_v3_experiment()
