import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from working.engine_v2 import PASOffloadEngineV2

def run_profiling():
    print("======================================================================")
    print("PROFILING V2: Bottleneck Analysis")
    print("======================================================================")
    
    in_features = 4096
    out_features = 11008
    iters = 100
    
    # Initialize CPU-side input tensors
    xs_cpu = [torch.randn(1, in_features, dtype=torch.float16) for _ in range(iters)]
    xs_gpu = [x.cuda() for x in xs_cpu]
    
    print("Initializing weights...")
    W_cpu = torch.randn(out_features, in_features, dtype=torch.float16)
    
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
    
    # Events for detailed profiling
    cpu_predictor_times = []
    host_slice_times = []
    gpu_dma_times = []
    gpu_unpack_times = []
    gpu_matmul_times = []
    total_submit_times = []
    total_execute_times = []
    
    print("Starting detailed profiling...")
    start_time = time.perf_counter()
    
    for i in range(5, iters):
        buf_idx = i % 2
        x = xs_gpu[i]
        
        # --- PROFILING SUBMIT FORWARD ---
        t_sub_start = time.perf_counter()
        
        # 1. CPU Predictor
        t0 = time.perf_counter()
        x_cpu = x[0]
        indices = engine_v2.predictor.predict_indices(x_cpu, 0.15)
        active_cols = len(indices)
        t1 = time.perf_counter()
        cpu_predictor_times.append((t1 - t0) * 1000)
        
        # 2. Host Slicing
        t0 = time.perf_counter()
        torch.index_select(
            engine_v2.packed_weights_cpu, 0, indices,
            out=engine_v2.sliced_packed_pinned[buf_idx][:active_cols]
        )
        torch.index_select(
            engine_v2.scales_cpu, 0, indices,
            out=engine_v2.scales_pinned[buf_idx][:active_cols]
        )
        torch.index_select(
            engine_v2.min_vals_cpu, 0, indices,
            out=engine_v2.min_vals_pinned[buf_idx][:active_cols]
        )
        t1 = time.perf_counter()
        host_slice_times.append((t1 - t0) * 1000)
        
        # 3. GPU DMA & Unpack
        stream = engine_v2.streams[buf_idx]
        
        start_dma = torch.cuda.Event(enable_timing=True)
        end_dma = torch.cuda.Event(enable_timing=True)
        start_unpack = torch.cuda.Event(enable_timing=True)
        end_unpack = torch.cuda.Event(enable_timing=True)
        
        with torch.cuda.stream(stream):
            start_dma.record(stream)
            engine_v2.sliced_packed_gpu[buf_idx][:active_cols].copy_(
                engine_v2.sliced_packed_pinned[buf_idx][:active_cols], non_blocking=True
            )
            engine_v2.scales_gpu[buf_idx][:active_cols].copy_(
                engine_v2.scales_pinned[buf_idx][:active_cols], non_blocking=True
            )
            engine_v2.min_vals_gpu[buf_idx][:active_cols].copy_(
                engine_v2.min_vals_pinned[buf_idx][:active_cols], non_blocking=True
            )
            end_dma.record(stream)
            
            start_unpack.record(stream)
            from working.quantizer_v2 import unpack_2bit_vectorized_v2
            unpack_2bit_vectorized_v2(
                engine_v2.sliced_packed_gpu[buf_idx][:active_cols],
                engine_v2.scales_gpu[buf_idx][:active_cols],
                engine_v2.min_vals_gpu[buf_idx][:active_cols],
                (active_cols, engine_v2.in_features),
                out_tensor=engine_v2.unpacked_weights_gpu[buf_idx][:active_cols],
                temp_uint8_buffer=engine_v2.temp_uint8_gpu[buf_idx]
            )
            end_unpack.record(stream)
            
        ticket = {
            'buffer_idx': buf_idx,
            'active_cols': active_cols,
            'indices': indices,
            'stream': stream,
            'dma_events': (start_dma, end_dma),
            'unpack_events': (start_unpack, end_unpack)
        }
        t_sub_end = time.perf_counter()
        total_submit_times.append((t_sub_end - t_sub_start) * 1000)
        
        # --- PROFILING EXECUTE FORWARD ---
        # Note: to accurately profile, we execute immediately (no pipelining loop here, 
        # since we just want to measure individual components for bottleneck analysis)
        t_exec_start = time.perf_counter()
        
        start_matmul = torch.cuda.Event(enable_timing=True)
        end_matmul = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.current_stream().wait_stream(stream)
        
        start_matmul.record()
        weights_gpu = engine_v2.unpacked_weights_gpu[buf_idx][:active_cols]
        out = torch.matmul(x, weights_gpu.t())
        end_matmul.record()
        
        torch.cuda.synchronize()
        t_exec_end = time.perf_counter()
        total_execute_times.append((t_exec_end - t_exec_start) * 1000)
        
        gpu_dma_times.append(start_dma.elapsed_time(end_dma))
        gpu_unpack_times.append(start_unpack.elapsed_time(end_unpack))
        gpu_matmul_times.append(start_matmul.elapsed_time(end_matmul))

    print("\n" + "=" * 50)
    print("Granular Profiling Results (Average over runs)")
    print("=" * 50)
    print(f"1. CPU Predictor:   {np.mean(cpu_predictor_times):.3f} ms")
    print(f"2. Host Slicing:    {np.mean(host_slice_times):.3f} ms")
    print(f"3. GPU DMA (PCIe):  {np.mean(gpu_dma_times):.3f} ms")
    print(f"4. GPU Dequantize:  {np.mean(gpu_unpack_times):.3f} ms")
    print(f"5. GPU MatMul:      {np.mean(gpu_matmul_times):.3f} ms")
    print("-" * 50)
    print(f"Total Submit Time:  {np.mean(total_submit_times):.3f} ms")
    print(f"Total Execute Time: {np.mean(total_execute_times):.3f} ms")
    print("=" * 50)
    
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: CUDA is required.")
        sys.exit(1)
    run_profiling()
