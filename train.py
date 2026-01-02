import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ================= 配置区域 =================
CONFIG = {
    # 你的数据路径 (保持不变)
    'data_dir': '/data/zm/12_29_InTensity/',

    # 文件名匹配模式
    'file_pattern': '*_*_clip.csv',

    # ROI 设置
    'roi': {
        'row_min': 400, 'row_max': 499,  # 第0列
        'col_min': 0, 'col_max': 1280  # 第1列
    },

    # 训练超参数
    'seq_len': 2048,
    'batch_size': 32,
    'lr': 1e-3,
    'epochs': 100,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'save_path': '/data/zm/DeltaTNET_model/best_deltat_model.pth'
}


# ================= 1. 数据集定义 (已修复编码问题) =================
class DeltaTDataset(Dataset):
    def __init__(self, file_list, config, is_train=True):
        self.seq_len = config['seq_len']
        self.roi = config['roi']

        print(f"🔄 正在加载 {'训练' if is_train else '测试'} 数据...")

        # 1. 临时列表收集数据 (只存 Numpy 数组，减少对象开销)
        temp_data_list = []
        temp_label_list = []

        # 使用 tqdm 显示进度
        for file_path in tqdm(file_list, desc="Loading Files"):
            try:
                # ... (解析文件名和读取 CSV 的代码保持不变) ...
                # 1. 解析文件名
                basename = os.path.basename(file_path)
                velocity_str = basename.split('_')[0].replace('mm', '')
                label = float(velocity_str)

                # 2. 读取 CSV (兼容性读取)
                try:
                    df = pd.read_csv(file_path, header=None, usecols=[0, 1, 2],
                                     names=['row', 'col', 't_in'], encoding='utf-8')
                except:
                    try:
                        df = pd.read_csv(file_path, header=None, usecols=[0, 1, 2],
                                         names=['row', 'col', 't_in'], encoding='gbk')
                    except:
                        continue

                # 3. ROI 过滤
                mask = (df['row'] >= self.roi['row_min']) & (df['row'] <= self.roi['row_max']) & \
                       (df['col'] >= self.roi['col_min']) & (df['col'] <= self.roi['col_max'])
                valid_data = df[mask]

                # 4. 核心逻辑: 按像素排序 + 分组差分
                # (这里使用之前提供的"先按坐标排"的正确逻辑)
                data_val = valid_data.values
                if len(data_val) < 2: continue

                # 排序: Time(2), Col(1), Row(0)
                sort_idx = np.lexsort((data_val[:, 2], data_val[:, 1], data_val[:, 0]))
                sorted_data = data_val[sort_idx]

                # 差分
                diffs = sorted_data[1:] - sorted_data[:-1]

                # 筛选同像素事件 (d_row==0 & d_col==0)
                valid_pixel_mask = (diffs[:, 0] == 0) & (diffs[:, 1] == 0)
                true_isi = diffs[valid_pixel_mask, 2]

                # 剔除异常值
                true_isi = true_isi[true_isi > 0]

                if len(true_isi) < self.seq_len:
                    continue

                # Log 变换 (强制转为 float32 以省内存)
                delta_t = np.log1p(true_isi).astype(np.float32)

                # 切分
                num_samples = len(delta_t) // self.seq_len
                for i in range(num_samples):
                    segment = delta_t[i * self.seq_len: (i + 1) * self.seq_len]

                    # 存入临时列表
                    temp_data_list.append(segment)
                    temp_label_list.append(label)

            except Exception as e:
                pass  # 忽略错误文件

        # 2. 🌟 关键优化：将列表转换为紧凑的 Tensor 🌟
        # 这会释放掉列表产生的巨大额外开销
        print("⚡️ 正在进行内存压缩 (List -> Tensor)...")
        if len(temp_data_list) > 0:
            # data_tensor 本来就是从 numpy 转过来的，保持 np.float32 没问题（torch.from_numpy 会自动推断）
            self.data_tensor = torch.from_numpy(np.array(temp_data_list, dtype=np.float32))

            # label_tensor 是直接用 torch.tensor 创建的，必须用 torch.float32
            self.label_tensor = torch.tensor(temp_label_list, dtype=torch.float32)  # ✅ 修正为 torch.float32

            # 标签归一化 (直接在 Tensor 上操作)
            # 假设最大流速 2.5
            self.label_tensor = self.label_tensor / 2.5
        else:
            self.data_tensor = torch.empty(0)
            self.label_tensor = torch.empty(0)

        # 手动清理临时列表，立刻释放内存
        del temp_data_list
        del temp_label_list
        import gc
        gc.collect()

        print(f"✅ 加载完成: 共 {len(self.data_tensor)} 个样本")

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        # 直接从 Tensor 取数，速度极快且不占额外内存
        x = self.data_tensor[idx].unsqueeze(0)  # [1, seq_len]
        y = self.label_tensor[idx].unsqueeze(0)  # [1]
        return x, y


# ================= 2. 模型定义 (DeltaTNet) =================
class DeltaTNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Embedding
        self.embedding = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )

        # 2. Dilated TCN
        self.tcn = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm1d(256), nn.GELU(),
        )

        # 3. Regression
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.tcn(x)
        x = self.gap(x).flatten(1)
        x = self.regressor(x)
        return x


# ================= 3. 训练流程 (已添加进度条) =================
def train():
    search_path = os.path.join(CONFIG['data_dir'], CONFIG['file_pattern'])
    all_files = glob.glob(search_path)
    if not all_files:
        print("❌ 未找到数据文件，请检查路径！")
        return

    train_files = []
    val_files = []

    print("🔄 正在按文件名规则切分数据集...")
    for f in all_files:
        basename = os.path.basename(f)
        # 文件名格式: 0.2_1_clip.csv
        # parts: ['0.2', '1', 'clip.csv']
        try:
            parts = basename.split('_')
            group_idx = parts[1]  # 获取中间那个数字 '1', '2', '3'

            if group_idx == '3':
                val_files.append(f)  # 第3组用于验证
            else:
                train_files.append(f)  # 第1、2组用于训练
        except IndexError:
            print(f"⚠️ 文件名格式异常，跳过: {basename}")
            continue

    print(f"📊 数据集切分结果:")
    print(f"   - 训练集文件数: {len(train_files)} (包含 _1, _2)")
    print(f"   - 验证集文件数: {len(val_files)}   (包含 _3)")

    # 安全检查
    if len(train_files) == 0 or len(val_files) == 0:
        print("❌ 切分失败！请检查文件名是否包含 _1, _2, _3 结构。")
        return


    train_ds = DeltaTDataset(train_files, CONFIG, is_train=True)
    val_ds = DeltaTDataset(val_files, CONFIG, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

    model = DeltaTNet().to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    print("🚀 开始训练...")

    for epoch in range(CONFIG['epochs']):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0
        # 🌟 tqdm 进度条: 显示 Epoch 信息和实时 Loss
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} [Train]", leave=False)

        for x, y in train_bar:
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # 实时更新进度条上的 Loss 显示
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0
        # 验证集通常不需要太详细的进度条，用简单的即可
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
                pred = model(x)
                val_loss += criterion(pred, y).item()

        val_loss /= len(val_loader)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # 打印本轮总结
        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), CONFIG['save_path'])
            print(f"   💾 模型保存 (New Best: {best_loss:.4f})")

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Convergence (DeltaTNet)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/data/zm/12.30/training_curve.png')
    print("\n✅ 训练结束！收敛图已保存至 training_curve.png")


if __name__ == "__main__":
    train()