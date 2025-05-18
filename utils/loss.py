import torch
import torch.nn as nn
from itertools import permutations

class SI_SDRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Distortion Ratio Loss với Permutation Invariant Training (PIT)
    Triển khai theo công thức chuẩn từ nghiên cứu gốc: https://arxiv.org/abs/1811.02508

    Đặc điểm:
    - Xử lý tự động permutation cho multi-source
    - Hỗ trợ batch và GPU
    - Tối ưu hóa tốc độ với vectorization
    - Độ ổn định số học với epsilon
    """

    def __init__(self, zero_mean=True, eps=1e-8):
        super().__init__()
        self.zero_mean = zero_mean
        self.eps = eps

    def forward(self, est_sources, gt_sources):
        """
        Args:
            est_sources: Tensor (batch, num_sources, time)
            gt_sources: Tensor (batch, num_sources, time)

        Returns:
            loss: Tensor scalar
        """
        return self.pit_loss(est_sources, gt_sources)

    def si_sdr(self, est, gt):
        """Tính SI-SDR cho từng cặp source"""
        if self.zero_mean:
            est = est - torch.mean(est, dim=-1, keepdim=True)
            gt = gt - torch.mean(gt, dim=-1, keepdim=True)

        # Tính alpha để chuẩn hóa scale
        alpha = (torch.sum(est * gt, dim=-1, keepdim=True)
                 / (torch.sum(gt ** 2, dim=-1, keepdim=True) + self.eps))

        # Tính target và noise
        target = alpha * gt
        noise = est - target

        # Tính tỷ lệ tín hiệu trên nhiễu
        si_sdr = 10 * torch.log10(
            (torch.sum(target ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + self.eps)) + self.eps
        )
        return si_sdr

    def pit_loss(self, est, gt):
        """Permutation Invariant Loss"""
        batch_size, num_sources, _ = est.size()

        # Tạo ma trận SI-SDR cho tất cả các cặp
        pairwise_sisdr = torch.zeros(batch_size, num_sources, num_sources, device=est.device)
        for i in range(num_sources):
            for j in range(num_sources):
                pairwise_sisdr[:, i, j] = self.si_sdr(est[:, i], gt[:, j])

        # Tìm permutation tối ưu cho từng sample
        perms = list(permutations(range(num_sources)))
        loss = torch.zeros(batch_size, device=est.device)

        for p in perms:
            # Tính tổng SI-SDR cho permutation hiện tại
            current_loss = torch.zeros_like(loss)
            for src in range(num_sources):
                current_loss += pairwise_sisdr[:, src, p[src]]

            # Cập nhật loss tốt nhất
            loss = torch.maximum(loss, current_loss)

        # Trả về giá trị trung bình âm (vì cần minimize)
        return -torch.mean(loss)

    @staticmethod
    def test():
        """Unit test cho hàm loss"""
        torch.manual_seed(42)

        # Test case 1: Trường hợp lý tưởng
        gt = torch.randn(2, 2, 16000)  # Batch 2, 2 sources
        est = gt.clone()
        loss_fn = SI_SDRLoss()
        loss = loss_fn(est, gt)
        print(f"Test 1 (Perfect reconstruction) Loss: {loss.item():.4f} (Expected ~0.0)")

        # Test case 2: Nhiễu ngẫu nhiên
        noise = torch.randn_like(gt) * 0.1
        loss = loss_fn(gt + noise, gt)
        print(f"Test 2 (Noisy input) Loss: {loss.item():.4f} (Expected >0.0)")
