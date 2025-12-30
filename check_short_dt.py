import numpy as np
import matplotlib.pyplot as plt
from train import DeltaTDataset, CONFIG, glob
import os

search_path = os.path.join(CONFIG['data_dir'], CONFIG['file_pattern'])
all_files = glob.glob(search_path)

ratios = []
labels = []

TH = 5e-4  # 0.5 ms，可调

for f in all_files[:200]:
    ds = DeltaTDataset([f], CONFIG, is_train=False)
    for s in ds.samples[:200]:
        dt = np.exp(np.array(s['data']))
        ratio = np.mean(dt < TH)
        ratios.append(ratio)
        labels.append(s['label'])

plt.figure(figsize=(6,5))
plt.scatter(labels, ratios, s=2, alpha=0.5)
plt.xlabel('Velocity or τc label')
plt.ylabel('Ratio of short Δt')
plt.title('Short-Δt Ratio vs Label')
plt.show()
