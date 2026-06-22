import torch
import torch.nn as nn

class LowRankPredictor(nn.Module):
    """
    CPU-side Low-Rank Linear Predictor for forecasting active FFN columns.
    Uses a low-rank factorization (in_features -> rank -> out_features)
    to minimize memory footprint and execution latency on the CPU.
    """
    def __init__(self, in_features: int, rank: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.rank = rank
        self.out_features = out_features
        
        # Low-rank linear layers
        self.w1 = nn.Linear(in_features, rank, bias=False)
        self.w2 = nn.Linear(rank, out_features, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the predictor forward pass.
        Expects input x shape: (batch_size, in_features) or (in_features,)
        Returns Sigmoid probabilities of active neurons: (batch_size, out_features)
        """
        # Ensure input is 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        h = self.w1(x)
        out = self.w2(h)
        return torch.sigmoid(out)

    @torch.no_grad()
    def predict_indices(self, x: torch.Tensor, threshold: float = 0.15) -> torch.Tensor:
        """
        Predicts the active column indices for a single input vector.
        Expects x shape: (1, in_features) or (in_features,)
        Returns a 1D CPU tensor containing the sorted indices of predicted active columns.
        """
        # Run forward pass (always forced on CPU for low latency and zero host sync overhead)
        probs = self.forward(x.cpu())
        
        # Squeeze to 1D
        probs_1d = probs[0]
        
        # Get indices where prob > threshold
        indices = torch.nonzero(probs_1d > threshold, as_tuple=False).squeeze(-1)
        
        # Fallback: if no columns are predicted, select at least the top-1 column 
        # to prevent downstream empty transfers
        if indices.numel() == 0:
            _, top_idx = torch.topk(probs_1d, 1)
            return top_idx
            
        return indices
