import numpy as np
import matplotlib.pyplot as plt
from train import DeltaTDataset, CONFIG, glob, train_test_split
import os

# 加载全部文件
search_path = os.path.join(CONFIG['data_dir'], CONFIG['file_pattern'])
all_files = glob.glob(search_path)

# 按速度分组（假设 label 是速度）
groups = {}
for f in all_files:
    v = float(os.path.basename(f).split('_')[0])
    groups.setdefault(v, []).append(f)

# 选 3~4 个速度
velocities = sorted(groups.keys())
sel_vs = velocities[::max(1, len(velocities)//4)][:4]

plt.figure(figsize=(8,6))

for v in sel_vs:
    ds = DeltaTDataset(groups[v][:2], CONFIG, is_train=False)
    all_dt = []
    for s in ds.samples[:5000]:
        all_dt.extend(np.exp(np.array(s['data'])))  # 还原 logdt
    all_dt = np.array(all_dt)
    plt.hist(all_dt, bins=100, density=True, histtype='step', label=f'v={v}')

plt.xlabel('Δt (seconds)')
plt.ylabel('PDF')
plt.legend()
plt.title('Δt Distribution vs Velocity')
plt.xscale('log')
plt.show()
