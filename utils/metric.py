import torch
import numpy as np
from itertools import permutations
from torch import nn
from typing import List

class SeparationMetrics(nn.Module):
    """
    Triển khai các metrics đánh giá chất lượng tách âm
    Bao gồm:
    - SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
    - SDR (Signal-to-Distortion Ratio)
    - PESQ (Tùy chọn, cần cài đặt thư viện bổ sung)
    - STOI (Tùy chọn, cần cài đặt thư viện bổ sung)

    Hỗ trợ:
    - Permutation Invariant evaluation
    - Batch processing
    - GPU acceleration
    """

    def __init__(self, sample_rate: int = 8000, metrics: List[str] = ['sisdr', 'sdr']):
        super().__init__()
        self.sample_rate = sample_rate
        self.metrics = metrics
        self.eps = 1e-8

    def forward(self, est_sources: torch.Tensor, gt_sources: torch.Tensor) -> dict:
        """
        Tính toán metrics cho batch hiện tại

        Args:
            est_sources: Tensor (batch, num_sources, time)
            gt_sources: Tensor (batch, num_sources, time)

        Returns:
            Dict chứa các metrics đã tính
        """
        results = {}

        # Tìm permutation tối ưu
        best_perm = self.find_best_permutation(est_sources, gt_sources)
        est_ordered = est_sources.gather(1, best_perm.unsqueeze(-1).expand(-1, -1, est_sources.size(-1)))

        # Tính các metrics
        if 'sisdr' in self.metrics:
            results['sisdr'] = self.calculate_sisdr(est_ordered, gt_sources)

        if 'sdr' in self.metrics:
            results['sdr'] = self.calculate_sdr(est_ordered, gt_sources)

        return results

    def find_best_permutation(self, est_sources, gt_sources):
        """
        Tìm hoán vị tối ưu sử dụng pairwise SI-SDR
        """
        batch_size, num_sources, _ = est_sources.shape
        device = est_sources.device

        # Tính ma trận SI-SDR cho tất cả các cặp
        pairwise_sisdr = torch.zeros(batch_size, num_sources, num_sources, device=device)
        for i in range(num_sources):
            for j in range(num_sources):
                pairwise_sisdr[:, i, j] = self._pairwise_sisdr(est_sources[:, i], gt_sources[:, j])

        # Tìm permutation tốt nhất
        perms = torch.tensor(list(permutations(range(num_sources))), device=device)
        perm_scores = torch.zeros(batch_size, len(perms), device=device)

        for idx, perm in enumerate(perms):
            for src in range(num_sources):
                perm_scores[:, idx] += pairwise_sisdr[:, src, perm[src]]

        best_perm_idx = torch.argmax(perm_scores, dim=1)
        return perms[best_perm_idx]

    def _pairwise_sisdr(self, est, gt):
        """Tính SI-SDR cho từng cặp nguồn đơn"""
        est = est - torch.mean(est, dim=-1, keepdim=True)
        gt = gt - torch.mean(gt, dim=-1, keepdim=True)

        alpha = (torch.sum(est * gt, dim=-1, keepdim=True) /
            (torch.sum(gt ** 2, dim=-1, keepdim=True) + self.eps))

        target = alpha * gt
        noise = est - target

        return 10 * torch.log10(
            (torch.sum(target ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + self.eps)) + self.eps
        )

    def calculate_sisdr(self, est, gt):
        """Tính SI-SDR trung bình"""
        return torch.mean(self._pairwise_sisdr(est, gt))

    def calculate_sisdr(self, est, gt):
        B, N, T = est.shape
        est = est.view(B * N, T)
        gt = gt.view(B * N, T)
        return torch.mean(self._pairwise_sisdr(est, gt))


class MetricTracker:
    """Lớp helper để theo dõi metrics qua các batch"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}

    def update(self, batch_metrics: dict):
        for metric, value in batch_metrics.items():
            if metric not in self.metrics:
                self.metrics[metric] = []
            self.metrics[metric].append(value.detach().cpu().item())

    def get_means(self):
        return {metric: np.mean(values) for metric, values in self.metrics.items()}

def test_metrics():
    """Unit test cho các metrics"""
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test case 1: Trường hợp lý tưởng
    gt = torch.randn(2, 2, 16000, device=device)  # Batch 2, 2 sources
    est = gt.clone()

    metrics = SeparationMetrics()
    results = metrics(est, gt)

    print("\nTest 1 - Perfect reconstruction:")
    print(f"SI-SDR: {results['sisdr'].item():.2f} dB (Expected: ∞)")
    print(f"SDR: {results['sdr'].item():.2f} dB (Expected: ∞)")

    # Test case 2: Nhiễu Gaussian
    noise = torch.randn_like(gt) * 0.1
    results = metrics(gt + noise, gt)

    print("\nTest 2 - Noisy input:")
    print(f"SI-SDR: {results['sisdr'].item():.2f} dB (Expected: >0)")
    print(f"SDR: {results['sdr'].item():.2f} dB (Expected: >0)")
