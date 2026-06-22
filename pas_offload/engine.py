import torch
import torch.nn as nn
from .predictor import LowRankPredictor
from .quantizer import pack_2bit, unpack_2bit_vectorized

class PASOffloadEngine:
    """
    PAS-Offload Coordination Engine.
    Handles host-side transposed weight caching, CPU-side active-column prediction,
    pinned memory DMA transfer, and on-GPU weight unpacking and execution.
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 16):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # CPU-side low-rank predictor
        self.predictor = LowRankPredictor(in_features, rank, out_features)
        
        # Calculate packed 2-bit bytes per column vector (4 weights per byte)
        self.bytes_per_col = in_features * 2 // 8
        
        # Page-locked (pinned) CPU host buffers for fast asynchronous DMA transfers.
        # Conditionally pinned only if a CUDA device is available.
        if torch.cuda.is_available():
            self.packed_weights_cpu = torch.zeros(out_features, self.bytes_per_col, dtype=torch.uint8).pin_memory()
            self.scales_cpu = torch.zeros(out_features, dtype=torch.float16).pin_memory()
            self.min_vals_cpu = torch.zeros(out_features, dtype=torch.float16).pin_memory()
        else:
            self.packed_weights_cpu = torch.zeros(out_features, self.bytes_per_col, dtype=torch.uint8)
            self.scales_cpu = torch.zeros(out_features, dtype=torch.float16)
            self.min_vals_cpu = torch.zeros(out_features, dtype=torch.float16)
        
    def load_weights(self, weights_f16: torch.Tensor):
        """
        Compresses and loads a standard float16 weight matrix of shape (out_features, in_features)
        into the transposed, column-major CPU host cache.
        """
        assert weights_f16.shape == (self.out_features, self.in_features), \
            f"Weight shape mismatch. Expected {(self.out_features, self.in_features)}, got {weights_f16.shape}"
            
        # Allocate temp tensors for bulk packing, then copy to pinned buffers
        packed_temp = torch.zeros(self.out_features, self.bytes_per_col, dtype=torch.uint8)
        scales_temp = torch.zeros(self.out_features, dtype=torch.float16)
        min_vals_temp = torch.zeros(self.out_features, dtype=torch.float16)
        
        # Pack each column vector individually (yielding per-column scale factors)
        for i in range(self.out_features):
            col_weights = weights_f16[i]
            packed_col, scale, min_val = pack_2bit(col_weights)
            
            packed_temp[i] = packed_col
            scales_temp[i] = scale
            min_vals_temp[i] = min_val
            
        # Copy into host memory
        self.packed_weights_cpu.copy_(packed_temp)
        self.scales_cpu.copy_(scales_temp)
        self.min_vals_cpu.copy_(min_vals_temp)

    def forward(self, x: torch.Tensor, threshold: float = 0.15) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Executes the optimized sparse transfer and GPU computation pipeline.
        
        Args:
            x (torch.Tensor): Input hidden state vector on CUDA, shape (1, in_features).
            threshold (float): Prediction activation threshold.
            
        Returns:
            out (torch.Tensor): Sparse FFN projection on CUDA, shape (1, active_cols).
            indices (torch.Tensor): Active column indices on CPU, shape (active_cols,).
        """
        assert x.is_cuda, "Input tensor x must be on a CUDA device."
        assert x.shape == (1, self.in_features), f"Input shape mismatch. Expected {(1, self.in_features)}, got {x.shape}"
        
        # 1. Run CPU Predictor
        # Extract the vector to CPU, make prediction and get indices
        indices = self.predictor.predict_indices(x[0], threshold)
        active_cols = len(indices)
        
        # 2. Slice the column-major weights in CPU RAM (contiguous row-select)
        sliced_packed = torch.index_select(self.packed_weights_cpu, 0, indices)
        
        # 3. Stream active weights and metadata to GPU asynchronously via DMA
        sliced_packed_gpu = sliced_packed.cuda(non_blocking=True)
        scales_gpu = self.scales_cpu[indices].cuda(non_blocking=True)
        min_vals_gpu = self.min_vals_cpu[indices].cuda(non_blocking=True)
        
        # 4. Unpack 2-bit weights in parallel on the GPU
        unpacked_shape = (active_cols, self.in_features)
        weights_gpu = unpack_2bit_vectorized(sliced_packed_gpu, scales_gpu, min_vals_gpu, unpacked_shape)
        
        # 5. Execute matrix multiplication on the GPU
        # x: (1, in_features) @ weights_gpu.t(): (in_features, active_cols) -> out: (1, active_cols)
        out = torch.matmul(x, weights_gpu.t())
        
        return out, indices
