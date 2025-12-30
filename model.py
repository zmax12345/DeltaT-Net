import torch
import torch.nn as nn


class DeltaTNet(nn.Module):
    def __init__(self, seq_len=1024):
        super().__init__()

        # --- 1. Per-event Embedding (时间尺度滤波器组) ---
        # 输入: [Batch, 1, N] -> 输出: [Batch, 64, N]
        # 1x1 卷积，相当于对每个 dt 做全连接，提取高维特征
        self.embedding = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.GELU()  # GELU 比 ReLU 在处理时间序列时表现更好
        )

        # --- 2. Temporal Feature Extraction (扩张卷积堆叠) ---
        # 通过堆叠 dilation，感受野指数级增大，看清长程相关性
        self.tcn = nn.Sequential(
            # Block 1: 看局部 (Dilation=1, RF=3)
            nn.Conv1d(64, 64, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(64),
            nn.GELU(),

            # Block 2: 看稍远 (Dilation=2, RF=5)
            nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(128),
            nn.GELU(),

            # Block 3: 看中程 (Dilation=4, RF=9)
            nn.Conv1d(128, 128, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(128),
            nn.GELU(),

            # Block 4: 看全局 (Dilation=8, RF=17...)
            nn.Conv1d(128, 256, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )

        # --- 3. Aggregation & Regression ---
        # Global Average Pooling: 统计这段时间内的平均特征
        # 这就是你说的 "回归 tau_c" 的那一步
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # 输出标量 v (也就是隐含了 1/tau_c * d)
        )

    def forward(self, x):
        # x: [Batch, 1, N] (Log Delta-t Sequence)

        # 1. 逐事件映射
        x = self.embedding(x)

        # 2. 时域扩张卷积
        x = self.tcn(x)

        # 3. 全局统计
        x = self.gap(x).squeeze(-1)  # [Batch, 256]

        # 4. 物理参数回归
        v = self.regressor(x)

        return v