import unittest
import torch
from pas_offload.quantizer import pack_2bit, unpack_2bit, unpack_2bit_vectorized

class TestQuantizer(unittest.TestCase):
    def test_pack_unpack_scalar(self):
        # 1. Generate float16 weights
        original = torch.randn(100, dtype=torch.float16)
        
        # 2. Pack to 2-bit
        packed, scale, min_val = pack_2bit(original)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.shape, (25,))  # 100 elements / 4 = 25 bytes
        
        # 3. Unpack back to float16
        unpacked = unpack_2bit(packed, scale, min_val, original.shape)
        self.assertEqual(unpacked.shape, original.shape)
        self.assertEqual(unpacked.dtype, torch.float16)
        
        # 2-bit quantization has a high error but shape and bounds should match
        self.assertTrue((unpacked >= original.min() - 0.1).all())
        self.assertTrue((unpacked <= original.max() + 0.1).all())

    def test_unpack_vectorized(self):
        # 2D matrix of shape (32, 128)
        out_features = 32
        in_features = 128
        
        original_matrix = torch.randn(out_features, in_features, dtype=torch.float16)
        
        # Pack each row (column in the transposed representation) individually
        bytes_per_col = in_features // 4
        packed_cols = torch.zeros(out_features, bytes_per_col, dtype=torch.uint8)
        scales = torch.zeros(out_features, dtype=torch.float16)
        min_vals = torch.zeros(out_features, dtype=torch.float16)
        
        for i in range(out_features):
            packed, scale, min_val = pack_2bit(original_matrix[i])
            packed_cols[i] = packed
            scales[i] = scale
            min_vals[i] = min_val
            
        # Unpack vectorized on GPU (or CPU since PyTorch supports bits operations on CPU too)
        unpacked_matrix = unpack_2bit_vectorized(
            packed_cols, scales, min_vals, original_matrix.shape
        )
        
        self.assertEqual(unpacked_matrix.shape, original_matrix.shape)
        self.assertEqual(unpacked_matrix.dtype, torch.float16)

if __name__ == "__main__":
    unittest.main()
