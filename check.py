# diagnose_signal.py
import numpy as np
import pickle, os
from collections import defaultdict

# 载入 dataset 实例（或直接读取 samples 文件夹）
# 这里假定你能 import DeltaTDataset 的构造并用 train_files 读取一批小样本
from train import DeltaTDataset, CONFIG, glob, train_test_split

search_path = os.path.join(CONFIG['data_dir'], CONFIG['file_pattern'])
all_files = glob.glob(search_path)
train_files, _ = train_test_split(all_files, test_size=0.2, random_state=42)
ds = DeltaTDataset(train_files[:50], CONFIG, is_train=True)  # 只读 50 文件做快速诊断

means = []
labels = []
for s in ds.samples:
    arr = np.asarray(s['data'], dtype=np.float32)
    means.append(arr.mean())
    labels.append(s['label'])

means = np.array(means)
labels = np.array(labels)
print("samples:", len(means))
print("mean(logdt) stats: mean, std:", means.mean(), means.std())
print("labels stats: mean, std:", labels.mean(), labels.std())
# 相关性
corr = np.corrcoef(means, labels)[0,1]
print("Pearson corr(mean(logdt), label) = ", corr)
