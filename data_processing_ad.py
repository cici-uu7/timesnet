import numpy as np
import pandas as pd
import scipy.io
import os
from sklearn.preprocessing import StandardScaler

# ================= 配置 =================
DATA_PATH = './dataset/CWRU_AD'  # 保存路径
RAW_PATH = './dataset/CWRU_AD/raw'  # 原始 .mat 下载路径

# CWRU 文件映射 (参考 Anomaly-Transformer)
FILES = {
    'Normal': '97.mat',  # 正常 (用于训练 + 测试)
    'IR': '105.mat',  # 内圈故障 (只用于测试)
    'Ball': '118.mat',  # 滚动体故障 (只用于测试)
    'OR': '130.mat'  # 外圈故障 (只用于测试)
}

# 确保目录存在
if not os.path.exists(RAW_PATH):
    os.makedirs(RAW_PATH)


def load_mat(path):
    """读取 .mat 并提取 DE_time 驱动端振动数据"""
    try:
        mat = scipy.io.loadmat(path)
        for key in mat.keys():
            if 'DE_time' in key:
                return mat[key].flatten()
    except Exception as e:
        print(f"Error loading {path}: {e}")
    return None


def download_data():
    """简单的下载逻辑 (如果本地没有文件)"""
    base_url = 'https://engineering.case.edu/sites/default/files/'
    import urllib.request
    for name, mat_name in FILES.items():
        file_path = os.path.join(RAW_PATH, mat_name)
        if not os.path.exists(file_path):
            print(f"Downloading {mat_name}...")
            urllib.request.urlretrieve(base_url + mat_name, file_path)


def main():
    download_data()

    # 1. 加载数据
    print("Loading data...")
    normal_data = load_mat(os.path.join(RAW_PATH, FILES['Normal']))
    ir_data = load_mat(os.path.join(RAW_PATH, FILES['IR']))
    ball_data = load_mat(os.path.join(RAW_PATH, FILES['Ball']))
    or_data = load_mat(os.path.join(RAW_PATH, FILES['OR']))

    # 2. 划分训练集和测试集 (参考 Anomaly-Transformer 逻辑)
    # 训练集：只取 Normal 的前 60%
    train_len = int(len(normal_data) * 0.6)
    train_seq = normal_data[:train_len]

    # 测试集：Normal 的后 40% + 所有故障数据
    # 为了模拟连续时间流，我们将它们拼接起来
    # 注意：真实场景中异常通常是突发的，这里直接拼接
    test_normal = normal_data[train_len:]

    # 构建测试集序列
    test_seq = np.concatenate([test_normal, ir_data, ball_data, or_data])

    # 构建测试集标签 (0: 正常, 1: 异常)
    test_labels = np.concatenate([
        np.zeros(len(test_normal)),  # Normal
        np.ones(len(ir_data)),  # IR
        np.ones(len(ball_data)),  # Ball
        np.ones(len(or_data))  # OR
    ])

    # 3. 标准化 (只在训练集上拟合，防止数据泄露)
    print("Normalizing...")
    scaler = StandardScaler()
    scaler.fit(train_seq.reshape(-1, 1))

    train_seq_norm = scaler.transform(train_seq.reshape(-1, 1))
    test_seq_norm = scaler.transform(test_seq.reshape(-1, 1))

    # 4. 保存为 CSV (TimesNet Loader 喜欢的格式)
    # train.csv: 只有一列特征
    pd.DataFrame(train_seq_norm).to_csv(os.path.join(DATA_PATH, 'train.csv'), index=False, header=['value'])

    # test.csv: 只有一列特征
    pd.DataFrame(test_seq_norm).to_csv(os.path.join(DATA_PATH, 'test.csv'), index=False, header=['value'])

    # test_label.csv: 只有一列标签
    pd.DataFrame(test_labels).to_csv(os.path.join(DATA_PATH, 'test_label.csv'), index=False, header=['label'])

    print(f"Done.")
    print(f"Train size: {train_seq_norm.shape}")
    print(f"Test size: {test_seq_norm.shape}")


if __name__ == '__main__':
    main()