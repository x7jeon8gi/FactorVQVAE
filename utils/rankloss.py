import torch
import torch.nn as nn
import torch.nn.functional as F

class RankLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(RankLoss, self).__init__()
        self.alpha = alpha

    def forward(self, r_pred, r_true):
        """
        Arguments:
        r_pred -- Predicted rankings (tensor of shape [B, N])
        r_true -- Ground truth rankings (tensor of shape [B, N])

        Returns:
        total_loss -- Computed rank loss
        """
        r_pred = r_pred.squeeze()
        r_true = r_true.squeeze()

        # First term: MSE loss
        mse_loss = F.mse_loss(r_pred, r_true, reduction='mean')
        # mse_loss = F.smooth_l1_loss(r_pred, r_true)

        # Compute pairwise differences for r_pred and r_true
        r_pred_diff = r_pred.unsqueeze(2) - r_pred.unsqueeze(1)  # [B, N, N]
        r_true_diff = r_true.unsqueeze(2) - r_true.unsqueeze(1)  # [B, N, N]

        # Exclude diagonal elements
        # N = r_pred.shape[1]
        # diag_mask = ~torch.eye(N, dtype=bool, device=r_pred.device).unsqueeze(0)  # [1, N, N]
        # diag_mask = diag_mask.expand(r_pred.shape[0], -1, -1)  # [B, N, N]

        # Compute rank consistency loss with normalized differences
        rank_loss_matrix = torch.relu(-(r_pred_diff * r_true_diff))  # [B, N*(N-1)]

        # Adjust rank_loss scaling
        #B, N = r_pred.shape[:2]
        #rank_loss = torch.sum(rank_loss_matrix) / (B * N * (N - 1))

        # Combine the two terms
        rank_loss = rank_loss_matrix.mean()
        total_loss = mse_loss + self.alpha * rank_loss

        return total_loss
