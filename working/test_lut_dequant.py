import torch
import time

def test_lut_dequant():
    active_cols = 1100
    in_features = 4096
    bytes_per_col = in_features // 4
    
    device = 'cuda'
    
    # Fake packed data
    packed_tensor = torch.randint(0, 256, (active_cols, bytes_per_col), dtype=torch.uint8, device=device)
    scales = torch.randn(active_cols, dtype=torch.float16, device=device)
    min_vals = torch.randn(active_cols, dtype=torch.float16, device=device)
    
    # -----------------------------
    # 1. Original approach
    # -----------------------------
    def original_unpack(packed_tensor, scales, min_vals):
        flat_uint8 = torch.empty((active_cols, bytes_per_col, 4), dtype=torch.uint8, device=device)
        flat_uint8[..., 0] = packed_tensor & 0x03
        flat_uint8[..., 1] = (packed_tensor >> 2) & 0x03
        flat_uint8[..., 2] = (packed_tensor >> 4) & 0x03
        flat_uint8[..., 3] = (packed_tensor >> 6) & 0x03
        out_tensor = torch.empty((active_cols, in_features), dtype=torch.float16, device=device)
        out_tensor.copy_(flat_uint8.view(active_cols, -1))
        out_tensor.mul_(scales.unsqueeze(1))
        out_tensor.add_(min_vals.unsqueeze(1))
        return out_tensor

    # -----------------------------
    # 2. LUT approach
    # -----------------------------
    # Precompute LUT on GPU
    # For every byte from 0 to 255, extract the 4 2-bit values
    lut = torch.zeros((256, 4), dtype=torch.float16, device=device)
    for i in range(256):
        lut[i, 0] = i & 0x03
        lut[i, 1] = (i >> 2) & 0x03
        lut[i, 2] = (i >> 4) & 0x03
        lut[i, 3] = (i >> 6) & 0x03

    def lut_unpack(packed_tensor, scales, min_vals, out_tensor=None):
        # packed_tensor is (active_cols, bytes_per_col) uint8
        # Because PyTorch embedding/indexing needs long, cast to int/long
        # wait, we can do lut[packed_tensor.long()] which gives (active_cols, bytes_per_col, 4)
        unpacked_float = lut[packed_tensor.long()] 
        
        if out_tensor is None:
            out_tensor = unpacked_float.view(active_cols, in_features)
        else:
            # Note: out_tensor.copy_ or we just reshape.
            # If we want in-place directly? Unpacked_float is a new allocation.
            out_tensor.copy_(unpacked_float.view(active_cols, in_features))
            
        out_tensor.mul_(scales.unsqueeze(1))
        out_tensor.add_(min_vals.unsqueeze(1))
        return out_tensor

    # Warmup
    for _ in range(10):
        o1 = original_unpack(packed_tensor, scales, min_vals)
        o2 = lut_unpack(packed_tensor, scales, min_vals)
        
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(100):
        o1 = original_unpack(packed_tensor, scales, min_vals)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    t2 = time.perf_counter()
    for _ in range(100):
        o2 = lut_unpack(packed_tensor, scales, min_vals)
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    
    print(f"Original Unpack: {(t1-t0)*10:.3f} ms")
    print(f"LUT Unpack:      {(t3-t2)*10:.3f} ms")
    
    # Check correctness
    diff = (o1 - o2).abs().max()
    print(f"Max Diff: {diff}")

if __name__ == '__main__':
    test_lut_dequant()
