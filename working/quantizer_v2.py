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

def unpack_2bit_vectorized_v2(packed_tensor: torch.Tensor, scales: torch.Tensor, min_vals: torch.Tensor, out_shape: tuple, out_tensor: torch.Tensor = None, temp_uint8_buffer: torch.Tensor = None) -> torch.Tensor:
    """
    Optimized 2-bit dequantization on GPU.
    Avoids temporary tensor allocations by performing all conversions and math in-place.
    """
    active_cols, bytes_per_col = packed_tensor.shape
    in_features = out_shape[1]
    
    if out_tensor is None:
        out_tensor = torch.empty((active_cols, in_features), dtype=torch.float16, device=packed_tensor.device)
        
    # Reuse pre-allocated temp buffer or allocate if None
    if temp_uint8_buffer is None:
        flat_uint8 = torch.empty((active_cols, bytes_per_col, 4), dtype=torch.uint8, device=packed_tensor.device)
    else:
        flat_uint8 = temp_uint8_buffer[:active_cols]
    
    # Perform unpacked assignments in-place
    flat_uint8[..., 0] = packed_tensor & 0x03
    flat_uint8[..., 1] = (packed_tensor >> 2) & 0x03
    flat_uint8[..., 2] = (packed_tensor >> 4) & 0x03
    flat_uint8[..., 3] = (packed_tensor >> 6) & 0x03
    
    # Reshape view (zero copy)
    flat_unpacked = flat_uint8.view(active_cols, -1)
    
    # Perform dequantization 100% in-place directly in out_tensor
    out_tensor.copy_(flat_unpacked[:, :in_features])
    out_tensor.mul_(scales.unsqueeze(1))
    out_tensor.add_(min_vals.unsqueeze(1))
    
    return out_tensor
