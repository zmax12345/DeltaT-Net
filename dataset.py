import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os


class DeltaTSequenceDataset(Dataset):
    def __init__(self, config, is_train=True):
        self.seq_len = 1024  # 固定序列长度，例如每次取 1024 个间隔
        self.samples = []

        print(f"🚀 初始化 Delta-T 数据集 ({'训练' if is_train else '测试'})...")

        # 加载数据逻辑 (简化版，请根据实际文件路径补全)
        # 假设 config['files'] 包含清洗后的 .npy 文件路径
        target_files = config['files_train'] if is_train else config['files_test']

        for file_path in target_files:
            try:
                # 加载清洗后的数据 [x, y, p, t] (这里假设你已经清洗并保存了)
                # 注意：如果是 csv，按你的新格式：col(0), row(1), t_in(2), t_ex(3)
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path, header=None, usecols=[2], names=['t_in'])
                    t_seq = df['t_in'].values.astype(np.float32)
                else:  # .npy
                    data = np.load(file_path)
                    # 假设 npy 也是存的 [col, row, t_in, t_ex]
                    t_seq = data[:, 2]

                    # 计算 Delta t
                # 排序是必须的，虽然物理产生时就是有序的，但保险起见
                t_seq = np.sort(t_seq)
                delta_t = np.diff(t_seq)

                # 🌟 关键预处理：Log 变换 + 归一化
                # log(dt) 能把跨度巨大的微秒级差异压缩到合理范围
                # 加 1.0 是为了防止 dt=0 导致 log 负无穷
                delta_t = np.log1p(delta_t)

                # 切分成样本
                num_samples = len(delta_t) // self.seq_len
                for i in range(num_samples):
                    segment = delta_t[i * self.seq_len: (i + 1) * self.seq_len]

                    # 获取该段数据的真实流速标签 (从文件名解析)
                    # 比如 "0.2mm_1.csv" -> 0.2
                    label = self.parse_velocity_from_name(file_path)

                    self.samples.append({
                        'dt_seq': segment,
                        'label': label
                    })
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")

            # ... (加载循环结束) ...
            print(f"✅ 加载完成: 共 {len(self.samples)} 个样本")

            # 🌟 新增：检查标签有没有读对
            labels = [s['label'] for s in self.samples]
            print(f"📊 标签分布检查: Min={min(labels)}, Max={max(labels)}")
            print(f"   样例标签: {labels[:10]}")

    def parse_velocity_from_name(self, path):
        # 简单的文件名解析逻辑
        name = os.path.basename(path)
        if "0.2" in name: return 0.2
        if "0.5" in name: return 0.5
        if "0.8" in name: return 0.8
        if "1.0" in name: return 1.0
        if "1.2" in name: return 1.2
        if "1.5" in name: return 1.5
        if "1.8" in name: return 1.8
        if "2.0" in name: return 2.0
        if "2.2" in name: return 2.2
        if "2.5" in name: return 2.5
        return 0.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        x = torch.from_numpy(item['data']).float().unsqueeze(0)

        # 🌟 修改点：对标签进行归一化
        # 假设最大流速是 2.5 mm/s (或者你预计的最大值比如 3.0)
        # 这样 label 就变成了 0.0 ~ 1.0 之间，这对神经网络更友好
        raw_label = item['label']
        normalized_label = raw_label / 2.5

        y = torch.tensor([normalized_label], dtype=torch.float32)
        return x, y