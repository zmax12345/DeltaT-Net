import numpy as np
import matplotlib.pyplot as plt
from train import DeltaTDataset, CONFIG
import glob
import os

# 1. 强制使用 CPU 进行检查，避免显存冲突
CONFIG['device'] = 'cpu'

# 2. 加载所有文件
search_path = os.path.join(CONFIG['data_dir'], CONFIG['file_pattern'])
all_files = glob.glob(search_path)
if not all_files:
    print("❌ 没找到文件，请检查路径！")
    exit()

# 3. 按流速分组
groups = {}
for f in all_files:
    try:
        # 解析流速: "2.0_1_clip.csv" -> 2.0
        v_str = os.path.basename(f).split('_')[0].replace('mm', '')
        v = float(v_str)
        groups.setdefault(v, []).append(f)
    except:
        continue

# 排序流速
velocities = sorted(groups.keys())
print(f"🔍 发现流速标签: {velocities}")

# 4. 绘图：比较不同流速的 Log(Delta T) 分布
plt.figure(figsize=(10, 6))

# 选取几个代表性流速 (最慢、中等、最快)
selected_vs = [velocities[0], velocities[len(velocities) // 2], velocities[-1]]

for v in selected_vs:
    print(f"正在分析流速 {v} mm/s ...")
    # 只取该流速下的第一个文件做样本
    files = groups[v][:1]
    ds = DeltaTDataset(files, CONFIG, is_train=False)

    all_log_dt = []
    # 收集该文件内所有样本的 dt
    for i in range(len(ds)):
        # data 是 log1p 后的数据
        log_dt = ds[i][0].numpy().flatten()
        all_log_dt.extend(log_dt)

    # 绘制直方图 (Density=True 抵消光强差异)
    plt.hist(all_log_dt, bins=100, density=True, alpha=0.5, label=f'{v} mm/s', histtype='step', linewidth=2)

plt.xlabel('Log(Delta T)  [Network Input]')
plt.ylabel('Density (Probability)')
plt.title('Log-Interval Distribution Comparison (Shape Only)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/data/zm/12.30/check_dist_result.png')
print("✅ 分布对比图已保存至 check_dist_result.png")