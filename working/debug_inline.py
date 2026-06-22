import torch
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from working.engine_v3 import PASOffloadEngineV3
from working.quantizer_v3 import unpack_2bit_lut_v3

def debug_inline():
    in_features = 4096
    out_features = 11008
    iters = 100
    
    xs_cpu = [torch.randn(1, in_features, dtype=torch.float16) for _ in range(iters)]
    xs_gpu = [x.cuda() for x in xs_cpu]
    W_cpu = torch.randn(out_features, in_features, dtype=torch.float16)
    
    engine = PASOffloadEngineV3(in_features, out_features, rank=16)
    engine.load_weights(W_cpu)
    
    # Warmup
    for i in range(5):
        t = engine.submit_forward(xs_gpu[i], threshold=0.15, buffer_idx=i%2)
        engine.execute_forward(xs_gpu[i], t)
    torch.cuda.synchronize()
    
    t_predict = []
    t_slice = []
    t_launch = []
    t_wait = []
    t_matmul = []
    
    # submit 0
    x_cpu = xs_gpu[0][0]
    indices = engine.predictor.predict_indices(x_cpu, 0.15)
    active_cols = len(indices)
    torch.index_select(engine.packed_weights_cpu, 0, indices, out=engine.sliced_packed_pinned[0][:active_cols])
    torch.index_select(engine.scales_cpu, 0, indices, out=engine.scales_pinned[0][:active_cols])
    torch.index_select(engine.min_vals_cpu, 0, indices, out=engine.min_vals_pinned[0][:active_cols])
    
    stream = engine.streams[0]
    with torch.cuda.stream(stream):
        engine.sliced_packed_gpu[0][:active_cols].copy_(engine.sliced_packed_pinned[0][:active_cols], non_blocking=True)
        engine.scales_gpu[0][:active_cols].copy_(engine.scales_pinned[0][:active_cols], non_blocking=True)
        engine.min_vals_gpu[0][:active_cols].copy_(engine.min_vals_pinned[0][:active_cols], non_blocking=True)
        unpack_2bit_lut_v3(engine.sliced_packed_gpu[0][:active_cols], engine.scales_gpu[0][:active_cols], engine.min_vals_gpu[0][:active_cols], (active_cols, engine.in_features), out_tensor=engine.unpacked_weights_gpu[0][:active_cols])
        
    ticket = {'buffer_idx': 0, 'active_cols': active_cols, 'stream': stream}
    
    for i in range(1, iters):
        buf_idx = i % 2
        x_cpu = xs_gpu[i][0]
        
        # Predict
        t0 = time.perf_counter()
        indices = engine.predictor.predict_indices(x_cpu, 0.15)
        active_cols = len(indices)
        t1 = time.perf_counter()
        t_predict.append((t1-t0)*1000)
        
        # Slice
        t0 = time.perf_counter()
        torch.index_select(engine.packed_weights_cpu, 0, indices, out=engine.sliced_packed_pinned[buf_idx][:active_cols])
        torch.index_select(engine.scales_cpu, 0, indices, out=engine.scales_pinned[buf_idx][:active_cols])
        torch.index_select(engine.min_vals_cpu, 0, indices, out=engine.min_vals_pinned[buf_idx][:active_cols])
        t1 = time.perf_counter()
        t_slice.append((t1-t0)*1000)
        
        # Launch kernels
        t0 = time.perf_counter()
        stream = engine.streams[buf_idx]
        with torch.cuda.stream(stream):
            engine.sliced_packed_gpu[buf_idx][:active_cols].copy_(engine.sliced_packed_pinned[buf_idx][:active_cols], non_blocking=True)
            engine.scales_gpu[buf_idx][:active_cols].copy_(engine.scales_pinned[buf_idx][:active_cols], non_blocking=True)
            engine.min_vals_gpu[buf_idx][:active_cols].copy_(engine.min_vals_pinned[buf_idx][:active_cols], non_blocking=True)
            unpack_2bit_lut_v3(engine.sliced_packed_gpu[buf_idx][:active_cols], engine.scales_gpu[buf_idx][:active_cols], engine.min_vals_gpu[buf_idx][:active_cols], (active_cols, engine.in_features), out_tensor=engine.unpacked_weights_gpu[buf_idx][:active_cols])
        t1 = time.perf_counter()
        t_launch.append((t1-t0)*1000)
        
        next_ticket = {'buffer_idx': buf_idx, 'active_cols': active_cols, 'stream': stream}
        
        # Wait
        t0 = time.perf_counter()
        old_idx = ticket['buffer_idx']
        old_active_cols = ticket['active_cols']
        old_stream = ticket['stream']
        torch.cuda.current_stream().wait_stream(old_stream)
        t1 = time.perf_counter()
        t_wait.append((t1-t0)*1000)
        
        # Matmul
        t0 = time.perf_counter()
        weights_gpu = engine.unpacked_weights_gpu[old_idx][:old_active_cols]
        out = torch.matmul(xs_gpu[i-1], weights_gpu.t())
        t1 = time.perf_counter()
        t_matmul.append((t1-t0)*1000)
        
        ticket = next_ticket
        
    torch.cuda.synchronize()
    
    print(f"Predict: {sum(t_predict)/len(t_predict):.3f} ms")
    print(f"Slice:   {sum(t_slice)/len(t_slice):.3f} ms")
    print(f"Launch:  {sum(t_launch)/len(t_launch):.3f} ms")
    print(f"Wait:    {sum(t_wait)/len(t_wait):.3f} ms")
    print(f"Matmul:  {sum(t_matmul)/len(t_matmul):.3f} ms")

if __name__ == '__main__':
    debug_inline()
