import unittest
import torch
from pas_offload.predictor import LowRankPredictor

class TestLowRankPredictor(unittest.TestCase):
    def setUp(self):
        self.in_features = 128
        self.rank = 16
        self.out_features = 256
        self.predictor = LowRankPredictor(self.in_features, self.rank, self.out_features)
        
    def test_predictor_shapes(self):
        x = torch.randn(1, self.in_features)
        probs = self.predictor(x)
        self.assertEqual(probs.shape, (1, self.out_features))
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())
        
    def test_predict_indices(self):
        # Generate input vector
        x = torch.randn(self.in_features)
        
        # Test default threshold
        indices = self.predictor.predict_indices(x, threshold=0.5)
        self.assertIsInstance(indices, torch.Tensor)
        self.assertEqual(indices.dim(), 1)
        self.assertTrue(indices.numel() > 0)  # Verify fallback behavior at least returns 1 column
        
        # Test custom low threshold (should select many indices)
        indices_low = self.predictor.predict_indices(x, threshold=0.0)
        self.assertEqual(indices_low.numel(), self.out_features)

if __name__ == "__main__":
    unittest.main()
