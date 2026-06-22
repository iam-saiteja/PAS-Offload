import torch

def pack_2bit(tensor: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """
    Symmetrically quantizes a float16/float32 tensor to 2-bit integers [0, 3]
    and packs 4 weights per byte into a uint8 tensor.
    
    Returns:
        packed_tensor (torch.Tensor): Packed uint8 weights.
        scale (float): Scale factor for dequantization.
        min_val (float): Minimum value offset.
    """
    original_shape = tensor.shape
    numel = tensor.numel()
    
    # Ensure element count is a multiple of 4 (padding if necessary)
    pad_len = (4 - (numel % 4)) % 4
    if pad_len > 0:
        flat = torch.cat([tensor.flatten(), torch.zeros(pad_len, dtype=tensor.dtype, device=tensor.device)])
    else:
        flat = tensor.flatten()
        
    min_val = flat.min().item()
    max_val = flat.max().item()
    
    range_val = (max_val - min_val) if (max_val - min_val) > 1e-5 else 1.0
    
    # Scale to [0, 3] and round
    quantized = torch.round((flat - min_val) / range_val * 3.0).to(torch.uint8)
    quantized = torch.clamp(quantized, 0, 3)
    
    # Pack 4 weights (2-bits each) into a single uint8 byte
    w0 = quantized[0::4]
    w1 = quantized[1::4] << 2
    w2 = quantized[2::4] << 4
    w3 = quantized[3::4] << 6
    
    packed = w0 | w1 | w2 | w3
    
    return packed, range_val / 3.0, min_val

def unpack_2bit(packed_tensor: torch.Tensor, scale: float, min_val: float, out_shape: tuple) -> torch.Tensor:
    """
    Unpacks a uint8 packed 2-bit tensor on the GPU and dequantizes back to float16.
    """
    w0 = packed_tensor & 0x03
    w1 = (packed_tensor >> 2) & 0x03
    w2 = (packed_tensor >> 4) & 0x03
    w3 = (packed_tensor >> 6) & 0x03
    
    flat_packed = torch.stack([w0, w1, w2, w3], dim=1)
    flat_unpacked = flat_packed.flatten()
    
    unpacked_f16 = flat_unpacked.to(torch.float16) * scale + min_val
    
    target_numel = 1
    for dim in out_shape:
        target_numel *= dim
    unpacked_f16 = unpacked_f16[:target_numel]
    
    return unpacked_f16.reshape(out_shape)

def unpack_2bit_vectorized(packed_tensor: torch.Tensor, scales: torch.Tensor, min_vals: torch.Tensor, out_shape: tuple) -> torch.Tensor:
    """
    Unpacks a 2D uint8 packed 2-bit tensor of shape (active_cols, bytes_per_col) on the GPU
    and dequantizes it using per-column scale and min_val vectors of shape (active_cols,).
    """
    active_cols, bytes_per_col = packed_tensor.shape
    
    w0 = packed_tensor & 0x03
    w1 = (packed_tensor >> 2) & 0x03
    w2 = (packed_tensor >> 4) & 0x03
    w3 = (packed_tensor >> 6) & 0x03
    
    # Interleave to restore flat columns
    # Stack along dim 2, shape: (active_cols, bytes_per_col, 4)
    stacked = torch.stack([w0, w1, w2, w3], dim=2)
    # Reshape to (active_cols, total_unpacked_elements_per_col)
    flat_unpacked = stacked.reshape(active_cols, -1)
    
    # Dequantize using broadcast
    # scales shape: (active_cols, 1), min_vals shape: (active_cols, 1)
    unpacked_f16 = flat_unpacked.to(torch.float16) * scales.unsqueeze(1) + min_vals.unsqueeze(1)
    
    # Trim padding if necessary
    return unpacked_f16[:, :out_shape[1]]
