import torch

def pack_2bit(tensor: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """
    Symmetrically quantizes a float16/float32 tensor to 2-bit integers [0, 3]
    and packs 4 weights per byte into a uint8 tensor.
    """
    numel = tensor.numel()
    
    pad_len = (4 - (numel % 4)) % 4
    if pad_len > 0:
        flat = torch.cat([tensor.flatten(), torch.zeros(pad_len, dtype=tensor.dtype, device=tensor.device)])
    else:
        flat = tensor.flatten()
        
    min_val = flat.min().item()
    max_val = flat.max().item()
    
    range_val = (max_val - min_val) if (max_val - min_val) > 1e-5 else 1.0
    
    quantized = torch.round((flat - min_val) / range_val * 3.0).to(torch.uint8)
    quantized = torch.clamp(quantized, 0, 3)
    
    w0 = quantized[0::4]
    w1 = quantized[1::4] << 2
    w2 = quantized[2::4] << 4
    w3 = quantized[3::4] << 6
    
    packed = w0 | w1 | w2 | w3
    
    return packed, range_val / 3.0, min_val


# ----------------------------------------------------------------------------
# LUT Initialization for V3 (Lookup Table Dequantization)
# ----------------------------------------------------------------------------
_LUT_2BIT_FLOAT16 = None

def get_lut(device='cuda'):
    global _LUT_2BIT_FLOAT16
    if _LUT_2BIT_FLOAT16 is None or _LUT_2BIT_FLOAT16.device != torch.device(device):
        lut = torch.zeros((256, 4), dtype=torch.float16, device=device)
        for i in range(256):
            lut[i, 0] = i & 0x03
            lut[i, 1] = (i >> 2) & 0x03
            lut[i, 2] = (i >> 4) & 0x03
            lut[i, 3] = (i >> 6) & 0x03
        _LUT_2BIT_FLOAT16 = lut
    return _LUT_2BIT_FLOAT16


def unpack_2bit_lut_v3(packed_tensor: torch.Tensor, scales: torch.Tensor, min_vals: torch.Tensor, out_shape: tuple, out_tensor: torch.Tensor = None) -> torch.Tensor:
    """
    Ultra-fast 2-bit dequantization on GPU using a Lookup Table (LUT).
    Reduces the number of GPU kernels from 8 (in V2) down to 3.
    """
    active_cols, bytes_per_col = packed_tensor.shape
    in_features = out_shape[1]
    
    lut = get_lut(packed_tensor.device)
    
    # Fast gather: converts (N, M) uint8 -> (N, M, 4) float16
    unpacked_float = lut[packed_tensor.long()] 
    
    if out_tensor is None:
        out_tensor = unpacked_float.view(active_cols, -1)[:, :in_features].clone()
    else:
        out_tensor.copy_(unpacked_float.view(active_cols, -1)[:, :in_features])
        
    out_tensor.mul_(scales.unsqueeze(1))
    out_tensor.add_(min_vals.unsqueeze(1))
    
    return out_tensor
