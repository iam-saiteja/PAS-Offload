import os
import sys
import time
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from working.engine_v3 import PASOffloadEngineV3

def run_profiling_v3():
    in_features = 4096
    out_features = 11008
    iters = 100
    
    xs_cpu = [torch.randn(1, in_features, dtype=torch.float16) for _ in range(iters)]
    xs_gpu = [x.cuda() for x in xs_cpu]
    W_cpu = torch.randn(out_features, in_features, dtype=torch.float16)
    
    engine_v3 = PASOffloadEngineV3(in_features, out_features, rank=16)
    engine_v3.load_weights(W_cpu)
    
    # Warmup
    ticket = engine_v3.submit_forward(xs_gpu[0], threshold=0.15, buffer_idx=0)
    for i in range(1, 5):
        buf_idx = i % 2
        next_ticket = engine_v3.submit_forward(xs_gpu[i], threshold=0.15, buffer_idx=buf_idx)
        _ = engine_v3.execute_forward(xs_gpu[i-1], ticket)
        ticket = next_ticket
    _ = engine_v3.execute_forward(xs_gpu[4], ticket)
    torch.cuda.synchronize()
    
    cpu_predictor_times = []
    host_slice_times = []
    gpu_dma_unpack_times = []
    gpu_matmul_times = []
    total_submit_times = []
    total_execute_times = []
    wait_times = []
    
    for i in range(5, iters):
        buf_idx = i % 2
        x = xs_gpu[i]
        
        t_sub_start = time.perf_counter()
        
        # 1. CPU Predictor
        t0 = time.perf_counter()
        x_cpu = x[0]
        indices = engine_v3.predictor.predict_indices(x_cpu, 0.15)
        active_cols = len(indices)
        t1 = time.perf_counter()
        cpu_predictor_times.append((t1 - t0) * 1000)
        
        # 2. Host Slicing
        t0 = time.perf_counter()
        torch.index_select(
            engine_v3.packed_weights_cpu, 0, indices,
            out=engine_v3.sliced_packed_pinned[buf_idx][:active_cols]
        )
        torch.index_select(
            engine_v3.scales_cpu, 0, indices,
            out=engine_v3.scales_pinned[buf_idx][:active_cols]
        )
        torch.index_select(
            engine_v3.min_vals_cpu, 0, indices,
            out=engine_v3.min_vals_pinned[buf_idx][:active_cols]
        )
        t1 = time.perf_counter()
        host_slice_times.append((t1 - t0) * 1000)
        
        stream = engine_v3.streams[buf_idx]
        
        start_dma_unpack = torch.cuda.Event(enable_timing=True)
        end_dma_unpack = torch.cuda.Event(enable_timing=True)
        
        with torch.cuda.stream(stream):
            start_dma_unpack.record(stream)
            engine_v3.sliced_packed_gpu[buf_idx][:active_cols].copy_(
                engine_v3.sliced_packed_pinned[buf_idx][:active_cols], non_blocking=True
            )
            engine_v3.scales_gpu[buf_idx][:active_cols].copy_(
                engine_v3.scales_pinned[buf_idx][:active_cols], non_blocking=True
            )
            engine_v3.min_vals_gpu[buf_idx][:active_cols].copy_(
                engine_v3.min_vals_pinned[buf_idx][:active_cols], non_blocking=True
            )
            
            from working.quantizer_v3 import unpack_2bit_lut_v3
            unpack_2bit_lut_v3(
                engine_v3.sliced_packed_gpu[buf_idx][:active_cols],
                engine_v3.scales_gpu[buf_idx][:active_cols],
                engine_v3.min_vals_gpu[buf_idx][:active_cols],
                (active_cols, engine_v3.in_features),
                out_tensor=engine_v3.unpacked_weights_gpu[buf_idx][:active_cols]
            )
            end_dma_unpack.record(stream)
            
        ticket = {
            'buffer_idx': buf_idx,
            'active_cols': active_cols,
            'indices': indices,
            'stream': stream,
        }
        t_sub_end = time.perf_counter()
        total_submit_times.append((t_sub_end - t_sub_start) * 1000)
        
        t_exec_start = time.perf_counter()
        
        start_matmul = torch.cuda.Event(enable_timing=True)
        end_matmul = torch.cuda.Event(enable_timing=True)
        
        t_wait_start = time.perf_counter()
        torch.cuda.current_stream().wait_stream(stream)
        t_wait_end = time.perf_counter()
        wait_times.append((t_wait_end - t_wait_start) * 1000)
        
        start_matmul.record()
        weights_gpu = engine_v3.unpacked_weights_gpu[buf_idx][:active_cols]
        out = torch.matmul(x, weights_gpu.t())
        end_matmul.record()
        
        torch.cuda.synchronize()
        t_exec_end = time.perf_counter()
        total_execute_times.append((t_exec_end - t_exec_start) * 1000)
        
        gpu_dma_unpack_times.append(start_dma_unpack.elapsed_time(end_dma_unpack))
        gpu_matmul_times.append(start_matmul.elapsed_time(end_matmul))

    print("\n" + "=" * 50)
    print("V3 Granular Profiling Results (Average over runs)")
    print("=" * 50)
    print(f"1. CPU Predictor:     {np.mean(cpu_predictor_times):.3f} ms")
    print(f"2. Host Slicing:      {np.mean(host_slice_times):.3f} ms")
    print(f"3. GPU DMA + Unpack:  {np.mean(gpu_dma_unpack_times):.3f} ms")
    print(f"   -> CPU Wait Stream:{np.mean(wait_times):.3f} ms")
    print(f"4. GPU MatMul:        {np.mean(gpu_matmul_times):.3f} ms")
    print("-" * 50)
    print(f"Total Submit Time:    {np.mean(total_submit_times):.3f} ms")
    print(f"Total Execute Time:   {np.mean(total_execute_times):.3f} ms")
    print("=" * 50)
    
if __name__ == "__main__":
    run_profiling_v3()
