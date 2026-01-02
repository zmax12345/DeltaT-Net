import numpy as np
import matplotlib.pyplot as plt
from train import DeltaTDataset, CONFIG
import glob
import os
from tqdm import tqdm

CONFIG['device'] = 'cpu'
search_path = os.path.join(CONFIG['data_dir'], CONFIG['file_pattern'])
all_files = glob.glob(search_path)

# 阈值：判断什么是"短间隔" (单位: us)
# log(1 + 500us) ≈ 6.2
THRESHOLD_LOG = np.log1p(500)

velocities = []
ratios = []  # 短间隔占比
means = []  # 平均 Log 间隔

print("正在扫描所有文件进行统计...")
# 为了速度，每个流速只抽 100 个样本点
for f in tqdm(all_files):
    try:
        v_str = os.path.basename(f).split('_')[0].replace('mm', '')
        v = float(v_str)

        # 快速加载，不走 Dataset 的重型逻辑，直接读文件
        # 注意这里要复刻 Dataset 的处理逻辑
        import pandas as pd

        df = pd.read_csv(f, header=None, usecols=[0, 1, 2], names=['r', 'c', 't'], encoding='utf-8')
        # 简单 ROI 筛选
        df = df[(df['r'] >= 400) & (df['r'] <= 499) & (df['c'] >= 0) & (df['c'] <= 1280)]
        if len(df) < 1000: continue

        t = np.sort(df['t'].values)
        dt = np.diff(t)
        log_dt = np.log1p(dt)

        # 计算统计量
        ratio = np.mean(log_dt < THRESHOLD_LOG)
        mean_val = np.mean(log_dt)

        velocities.append(v)
        ratios.append(ratio)
        means.append(mean_val)

    except Exception as e:
        continue

# 绘图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(velocities, ratios, alpha=0.5, c='blue')
plt.xlabel('Velocity (mm/s)')
plt.ylabel('Ratio of Short Intervals (<500us)')
plt.title('Short Interval Ratio vs Velocity')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(velocities, means, alpha=0.5, c='green')
plt.xlabel('Velocity (mm/s)')
plt.ylabel('Mean Log(Delta T)')
plt.title('Mean Interval vs Velocity')
plt.grid(True)

plt.tight_layout()
plt.savefig('/data/zm/12.30/check_trend_result.png')
print("✅ 趋势分析图已保存至 check_trend_result.png")