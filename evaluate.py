import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 导入你的模型和数据集类
# ⚠️ 注意：这里假设你已经把 dataset.py 和 model.py 放在同级目录
from dataset import DeltaTSequenceDataset
from model import DeltaTNet

# ================= 评估配置 =================
EVAL_CONFIG = {
    # 1. 数据路径 (必须与训练时一致的格式)
    'data_dir': '/data/zm/12_29_InTensity/',
    'file_pattern': '*_*_clip.csv',

    # 2. 模型路径
    'model_dir': '/data/zm/DeltaTNET_model/',
    # 如果想指定特定模型，填文件名，否则填 None (自动找最新的)
    'manual_model_name': None,

    # 3. 物理参数
    'norm_factor': 2.5,  # 训练时除以了2.5，这里要乘回来

    # 4. 硬件
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'batch_size': 64
}


def load_test_data():
    """
    重新复现训练时的切分逻辑，只提取测试集文件
    """
    search_path = os.path.join(EVAL_CONFIG['data_dir'], EVAL_CONFIG['file_pattern'])
    all_files = glob.glob(search_path)

    # 复用 train.py 中的切分逻辑 (sklearn random_state=42)
    from sklearn.model_selection import train_test_split
    _, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

    print(f"📊 提取测试集: 共 {len(test_files)} 个文件")

    # 构造 Dataset (注意: is_train=False 会读取 files_test)
    # 我们这里手动构造一个 config 字典传给 Dataset
    dummy_config = {
        'files_test': test_files,
        'seq_len': 2048,  # 必须与训练一致
        'roi': {'row_min': 400, 'row_max': 499, 'col_min': 0, 'col_max': 1280}
    }

    ds = DeltaTSequenceDataset(dummy_config, is_train=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=EVAL_CONFIG['batch_size'], shuffle=False)
    return loader


def get_best_model_path():
    if EVAL_CONFIG['manual_model_name']:
        path = os.path.join(EVAL_CONFIG['model_dir'], EVAL_CONFIG['manual_model_name'])
        if os.path.exists(path): return path
        print(f"❌ 指定模型不存在: {path}")

    # 自动找最新的
    files = glob.glob(os.path.join(EVAL_CONFIG['model_dir'], "*.pth"))
    if not files:
        raise FileNotFoundError("没有找到任何 .pth 模型文件")

    # 按修改时间排序
    latest_file = max(files, key=os.path.getmtime)
    print(f"🔎 自动选择最新模型: {os.path.basename(latest_file)}")
    return latest_file


def evaluate():
    # 1. 准备数据
    test_loader = load_test_data()

    # 2. 加载模型
    model_path = get_best_model_path()
    model = DeltaTNet(seq_len=2048).to(EVAL_CONFIG['device'])

    # 加载权重 (处理可能的 DataParallel module 前缀)
    state_dict = torch.load(model_path, map_location=EVAL_CONFIG['device'])
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # 去掉多卡训练的前缀
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    # 3. 推理
    preds = []
    trues = []

    print("🚀 开始推理评估...")
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x = x.to(EVAL_CONFIG['device'])

            # 前向传播
            output = model(x)

            # 反归一化 (还原为真实 mm/s)
            pred_real = output.cpu().numpy().flatten() * EVAL_CONFIG['norm_factor']
            true_real = y.numpy().flatten() * EVAL_CONFIG['norm_factor']

            preds.extend(pred_real)
            trues.extend(true_real)

    preds = np.array(preds)
    trues = np.array(trues)

    # 4. 计算指标
    r2 = r2_score(trues, preds)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)

    print("\n" + "=" * 40)
    print(f"🏆 最终评估结果")
    print(f"R² Score : {r2:.4f}  (越接近1越好)")
    print(f"RMSE     : {rmse:.4f} mm/s")
    print(f"MAE      : {mae:.4f} mm/s")
    print("=" * 40 + "\n")

    # 5. 绘图
    plt.figure(figsize=(12, 5))

    # 图1: 散点图 (预测值 vs 真实值)
    plt.subplot(1, 2, 1)
    plt.scatter(trues, preds, alpha=0.05, s=2, color='blue', label='Samples')
    # 画对角线 y=x
    mi = min(trues.min(), preds.min())
    ma = max(trues.max(), preds.max())
    plt.plot([mi, ma], [mi, ma], 'r--', linewidth=2, label='Perfect Fit')
    plt.xlabel('True Velocity (mm/s)')
    plt.ylabel('Predicted Velocity (mm/s)')
    plt.title(f'True vs Predicted (R²={r2:.3f})')
    plt.legend()
    plt.grid(alpha=0.3)

    # 图2: 误差直方图
    plt.subplot(1, 2, 2)
    errors = preds - trues
    plt.hist(errors, bins=100, color='purple', alpha=0.7)
    plt.axvline(0, color='r', linestyle='--')
    plt.xlabel('Error (mm/s)')
    plt.ylabel('Count')
    plt.title(f'Error Distribution (RMSE={rmse:.3f})')
    plt.grid(alpha=0.3)

    save_path = 'evaluation_result.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ 图表已保存至: {save_path}")


if __name__ == "__main__":
    evaluate()