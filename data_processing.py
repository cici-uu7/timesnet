"""
Modified: 适配10个目标CWRU文件（1730rpm，含不同故障类型/程度）
添加独立验证集（VAL），划分比例：60% Train, 20% Val, 20% Test
增加每个样本数据量（sample_length=1000）以扩充数据总量
"""
# 随机选取起始位置，截取固定长度样本

from scipy.io import loadmat
import random
import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------- 1. 文件设置 --------------------------
# 数据根目录
root = Path("data")
# 数据文件路径
data_dir = root / "cwru_dataset" / "CWRU_10"
# 输出路径
save_dir = root / "cwru_dataset" / "CWRU_10"

target_files = [
    {"filename": "1730-0.000-Normal.mat", "de_var": "X100_DE_time", "label_str": "0.000-Normal"},
    {"filename": "1730-0.007-Ball.mat", "de_var": "X121_DE_time", "label_str": "0.007-Ball"},
    {"filename": "1730-0.007-InnerRace.mat", "de_var": "X108_DE_time", "label_str": "0.007-InnerRace"},
    {"filename": "1730-0.007-OuterRace6.mat", "de_var": "X133_DE_time", "label_str": "0.007-OuterRace6"},
    {"filename": "1730-0.014-Ball.mat", "de_var": "X188_DE_time", "label_str": "0.014-Ball"},
    {"filename": "1730-0.014-InnerRace.mat", "de_var": "X172_DE_time", "label_str": "0.014-InnerRace"},
    {"filename": "1730-0.014-OuterRace6.mat", "de_var": "X200_DE_time", "label_str": "0.014-OuterRace6"},
    {"filename": "1730-0.021-Ball.mat", "de_var": "X225_DE_time", "label_str": "0.021-Ball"},
    {"filename": "1730-0.021-InnerRace.mat", "de_var": "X212_DE_time", "label_str": "0.021-InnerRace"},
    {"filename": "1730-0.021-OuterRace6.mat", "de_var": "X237_DE_time", "label_str": "0.021-OuterRace6"},
]

# 采样参数（已增加 sample_length 以扩充数据量）
sample_length = 1000  # 每个样本的采样点数量（从500增加到1000，数据量翻倍）
sample_number = 200  # 每个文件生成的总样本数（可进一步增加到300以生成更多样本）
train_val_test_rate = [0.6, 0.2, 0.2]  # 训练/验证/测试集占比（60%/20%/20%）

# 标签映射（10类，与target_files的label_str一一对应）
label_map = {
    "0.000-Normal": 0,
    "0.007-Ball": 1,
    "0.007-InnerRace": 2,
    "0.007-OuterRace6": 3,
    "0.014-Ball": 4,
    "0.014-InnerRace": 5,
    "0.014-OuterRace6": 6,
    "0.021-Ball": 7,
    "0.021-InnerRace": 8,
    "0.021-OuterRace6": 9
}

# -------------------------- 2. 读取数据+生成样本 --------------------------
trains = None  # 初始化训练集
vals = None  # 初始化验证集
tests = None  # 初始化测试集

for file_info in target_files:
    # 构建文件完整路径
    file_path = data_dir / file_info["filename"]
    # 检查文件是否存在
    if not file_path.exists():
        print(f"文件不存在 → {file_path}")
        continue
    try:
        # 读取.mat文件
        mat_data = loadmat(file_path)
        # 提取驱动端数据
        de_var = file_info["de_var"]
        if de_var not in mat_data.keys():
            print(f"变量不存在：{de_var}，文件实际变量：{list(mat_data.keys())}")
            continue
        de_time = mat_data[de_var].reshape(-1)  # 直接转为一维数组
        print(f" 驱动端数据长度：{len(de_time)}")

        # 3. 截取前120000个采样点（若 sample_length=1000，可考虑增加到 e.g., 200000 以支持更多采样）
        de_time = de_time[:120000]
        valid_length = len(de_time)
        if valid_length < sample_length:
            print(f" 数据过短：有效长度{valid_length} < 样本长度{sample_length}")
            continue

        # 4. 随机生成起始位置
        # 起始位置范围：[0, valid_length - sample_length]，确保截取不越界
        max_begin = valid_length - sample_length
        if max_begin <= 0:
            print(f"无法采样：有效长度{valid_length} < 样本长度{sample_length}")
            continue
        # 随机选取sample_number个不重复的起始位置（防超限）
        actual_sample_number = min(sample_number, max_begin + 1)
        begins = random.sample(range(0, max_begin + 1), actual_sample_number)

        # 5. 生成样本（标签+振动序列）
        records = []
        for begin in begins:
            # 截取从begin开始的sample_length个采样点
            sample = de_time[begin:begin + sample_length].tolist()
            # 拼接标签和样本数据
            record = [label_map[file_info["label_str"]]] + sample
            records.append(record)

        # 6. 划分训练/验证/测试集
        temp = np.array(records)
        # 计算各集数量
        train_count = int(actual_sample_number * train_val_test_rate[0])
        val_count = int(actual_sample_number * train_val_test_rate[1])
        test_count = actual_sample_number - train_count - val_count  # 剩余为测试
        # 随机打乱并划分
        indices = np.random.permutation(actual_sample_number)
        train_idx = indices[:train_count]
        val_idx = indices[train_count:train_count + val_count]
        test_idx = indices[train_count + val_count:]
        train_data, val_data, test_data = temp[train_idx], temp[val_idx], temp[test_idx]

        # 7. 拼接所有文件的训练/验证/测试集
        if trains is None:
            trains = train_data
            vals = val_data
            tests = test_data
        else:
            trains = np.r_[trains, train_data]
            vals = np.r_[vals, val_data]
            tests = np.r_[tests, test_data]

        print(f"生成样本数：{len(records)}（训练{len(train_data)}，验证{len(val_data)}，测试{len(test_data)}）")

    except Exception as e:
        print(f"处理文件{file_info['filename']}出错：{str(e)}")

# -------------------------- 3. 保存训练/验证/测试集 --------------------------
if trains is not None and vals is not None and tests is not None:
    # 保存训练集
    train_df = pd.DataFrame(trains)
    train_df[0] = train_df[0].astype(int)  # 标签转为整数
    train_path = save_dir / "TRAIN.tsv"
    train_df.to_csv(train_path, header=False, sep="\t", index=False)
    print(f"\n训练集保存：{train_path}，形状：{trains.shape}")

    # 保存验证集
    val_df = pd.DataFrame(vals)
    val_df[0] = val_df[0].astype(int)  # 标签转为整数
    val_path = save_dir / "VAL.tsv"
    val_df.to_csv(val_path, header=False, sep="\t", index=False)
    print(f"验证集保存：{val_path}，形状：{vals.shape}")

    # 保存测试集
    test_df = pd.DataFrame(tests)
    test_df[0] = test_df[0].astype(int)  # 标签转为整数
    test_path = save_dir / "TEST.tsv"
    test_df.to_csv(test_path, header=False, sep="\t", index=False)
    print(f"测试集保存：{test_path}，形状：{tests.shape}")

    # 打印整体统计
    total_samples = len(trains) + len(vals) + len(tests)
    print(f"\n总体统计：训练{len(trains)}，验证{len(vals)}，测试{len(tests)}，总{total_samples}")
    print(f"类别分布（训练）：{np.bincount(trains[:, 0].astype(int))}")
    print(f"类别分布（验证）：{np.bincount(vals[:, 0].astype(int))}")
    print(f"类别分布（测试）：{np.bincount(tests[:, 0].astype(int))}")
else:
    print(f"\n未生成数据：所有文件处理失败或无有效数据")