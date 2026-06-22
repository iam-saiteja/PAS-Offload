import torch
import time

def run():
    in_features = 4096
    out_features = 11008
    active_cols = 1100
    bytes_per_col = in_features * 2 // 8
    
    # Simulate pinned memory
    pinned_packed = torch.randint(0, 256, (active_cols, bytes_per_col), dtype=torch.uint8).pin_memory()
    pinned_scales = torch.randn(active_cols, dtype=torch.float16).pin_memory()
    pinned_min_vals = torch.randn(active_cols, dtype=torch.float16).pin_memory()
    
    gpu_packed = torch.zeros(active_cols, bytes_per_col, dtype=torch.uint8, device='cuda')
    gpu_scales = torch.zeros(active_cols, dtype=torch.float16, device='cuda')
    gpu_min_vals = torch.zeros(active_cols, dtype=torch.float16, device='cuda')
    gpu_out = torch.zeros(active_cols, in_features, dtype=torch.float16, device='cuda')
    
    lut = torch.zeros((256, 4), dtype=torch.float16, device='cuda')
    for i in range(256):
        lut[i, 0] = i & 0x03
        lut[i, 1] = (i >> 2) & 0x03
        lut[i, 2] = (i >> 4) & 0x03
        lut[i, 3] = (i >> 6) & 0x03

    stream = torch.cuda.Stream()
    
    # Warmup
    for _ in range(10):
        with torch.cuda.stream(stream):
            gpu_packed.copy_(pinned_packed, non_blocking=True)
            gpu_scales.copy_(pinned_scales, non_blocking=True)
            gpu_min_vals.copy_(pinned_min_vals, non_blocking=True)
            
            unpacked_float = lut[gpu_packed.long()]
            gpu_out.copy_(unpacked_float.view(active_cols, -1)[:, :in_features])
            gpu_out.mul_(gpu_scales.unsqueeze(1))
            gpu_out.add_(gpu_min_vals.unsqueeze(1))
            
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    dma_event = torch.cuda.Event(enable_timing=True)
    lut_event = torch.cuda.Event(enable_timing=True)
    copy_event = torch.cuda.Event(enable_timing=True)
    math_event = torch.cuda.Event(enable_timing=True)
    
    with torch.cuda.stream(stream):
        start_event.record(stream)
        gpu_packed.copy_(pinned_packed, non_blocking=True)
        gpu_scales.copy_(pinned_scales, non_blocking=True)
        gpu_min_vals.copy_(pinned_min_vals, non_blocking=True)
        dma_event.record(stream)
        
        idx = gpu_packed.long()
        unpacked_float = lut[idx]
        lut_event.record(stream)
        
        gpu_out.copy_(unpacked_float.view(active_cols, -1)[:, :in_features])
        copy_event.record(stream)
        
        gpu_out.mul_(gpu_scales.unsqueeze(1))
        gpu_out.add_(gpu_min_vals.unsqueeze(1))
        math_event.record(stream)
        
    torch.cuda.synchronize()
    
    print(f"DMA Time:   {start_event.elapsed_time(dma_event):.3f} ms")
    print(f"LUT Gather: {dma_event.elapsed_time(lut_event):.3f} ms")
    print(f"Copy Out:   {lut_event.elapsed_time(copy_event):.3f} ms")
    print(f"Math:       {copy_event.elapsed_time(math_event):.3f} ms")
    print(f"Total:      {start_event.elapsed_time(math_event):.3f} ms")

if __name__ == '__main__':
    run()
