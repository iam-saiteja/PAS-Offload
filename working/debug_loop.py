import torch
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from working.engine_v3 import PASOffloadEngineV3

def debug_loop():
    in_features = 4096
    out_features = 11008
    iters = 100
    
    xs_cpu = [torch.randn(1, in_features, dtype=torch.float16) for _ in range(iters)]
    xs_gpu = [x.cuda() for x in xs_cpu]
    W_cpu = torch.randn(out_features, in_features, dtype=torch.float16)
    
    engine_v3 = PASOffloadEngineV3(in_features, out_features, rank=16)
    engine_v3.load_weights(W_cpu)
    
    # Warmup
    for i in range(5):
        t = engine_v3.submit_forward(xs_gpu[i], threshold=0.15, buffer_idx=i%2)
        engine_v3.execute_forward(xs_gpu[i], t)
    torch.cuda.synchronize()
    
    submit_times = []
    wait_times = []
    exec_cpu_times = []
    
    start = time.perf_counter()
    ticket = engine_v3.submit_forward(xs_gpu[0], threshold=0.15, buffer_idx=0)
    
    for i in range(1, iters):
        buf_idx = i % 2
        
        t0 = time.perf_counter()
        next_ticket = engine_v3.submit_forward(xs_gpu[i], threshold=0.15, buffer_idx=buf_idx)
        t1 = time.perf_counter()
        submit_times.append((t1-t0)*1000)
        
        # Manually inline execute_forward to measure wait
        t2 = time.perf_counter()
        old_idx = ticket['buffer_idx']
        active_cols = ticket['active_cols']
        stream = ticket['stream']
        
        torch.cuda.current_stream().wait_stream(stream)
        t3 = time.perf_counter()
        wait_times.append((t3-t2)*1000)
        
        weights_gpu = engine_v3.unpacked_weights_gpu[old_idx][:active_cols]
        out = torch.matmul(xs_gpu[i-1], weights_gpu.t())
        
        t4 = time.perf_counter()
        exec_cpu_times.append((t4-t3)*1000)
        
        ticket = next_ticket
        
    _ = engine_v3.execute_forward(xs_gpu[iters-1], ticket)
    torch.cuda.synchronize()
    
    print(f"Avg Submit Time: {sum(submit_times)/len(submit_times):.3f} ms")
    print(f"Avg Wait Time:   {sum(wait_times)/len(wait_times):.3f} ms")
    print(f"Avg Exec Launch: {sum(exec_cpu_times)/len(exec_cpu_times):.3f} ms")

if __name__ == '__main__':
    debug_loop()
