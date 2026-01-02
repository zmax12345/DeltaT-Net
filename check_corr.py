import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from train import CONFIG

file_path = glob.glob(os.path.join(CONFIG['data_dir'], CONFIG['file_pattern']))[0]
print(f"正在分析空间分布: {file_path}")

df = pd.read_csv(file_path, header=None, usecols=[0, 1], names=['row', 'col'])
# 筛选 ROI
df = df[(df['row']>=400)&(df['row']<=499)&(df['col']>=0)&(df['col']<=1280)]

plt.figure(figsize=(10, 4))
plt.hist(df['col'], bins=128, color='purple', alpha=0.7)
plt.xlabel('Column Index (0-1280)')
plt.ylabel('Event Count')
plt.title('Spatial Event Distribution (Check for Non-uniformity)')
plt.grid(True, alpha=0.3)
plt.savefig('/data/zm/12.30/check_spatial_result.png')
print("✅ 空间分布图已保存至 check_spatial_result.png")