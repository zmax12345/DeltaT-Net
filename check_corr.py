import numpy as np
import matplotlib.pyplot as plt
from train import DeltaTDataset, CONFIG, glob
import os

search_path = os.path.join(CONFIG['data_dir'], CONFIG['file_pattern'])
all_files = glob.glob(search_path)

corrs = []
labels = []

for f in all_files[:200]:
    ds = DeltaTDataset([f], CONFIG, is_train=False)
    for s in ds.samples[:200]:
        dt = np.exp(np.array(s['data']))
        if len(dt) > 5:
            c = np.corrcoef(dt[:-1], dt[1:])[0,1]
            corrs.append(c)
            labels.append(s['label'])

plt.figure(figsize=(6,5))
plt.scatter(labels, corrs, s=2, alpha=0.5)
plt.xlabel('Velocity or τc label')
plt.ylabel('corr(Δt_i, Δt_{i+1})')
plt.title('Temporal Correlation vs Label')
plt.show()
