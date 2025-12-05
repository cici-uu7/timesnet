#!/bin/bash

# -------------------------- 1. GPU环境配置 --------------------------
export CUDA_VISIBLE_DEVICES=0  # 仅启用第0号GPU
echo "已设置GPU环境：仅使用GPU0"

# -------------------------- 2. 实验核心参数 --------------------------
# 数据路径
ROOT_PATH="./dataset/CWRU_10"
TRAIN_FILE="TRAIN.tsv"
TEST_FILE="TEST.tsv"
VAL_FILE="VAL.tsv"
# 模型与任务配置
MODEL_ID="CWRU_10_GPU0_opt"   # 标识为优化版实验
MODEL="TimesNet"
TASK_NAME="classification"
DATA_TYPE="CWRU_10"

# 数据格式（保持与原配置一致）
FEATURES="S"
SEQ_LEN=1000
PRED_LEN=0
ENC_IN=1
C_OUT=10
NUM_CLASS=10
FREQ="s"

# 模型结构
D_MODEL=128                 # CHANGED: 增大到128，提升模型容量（原64）
D_FF=512                    # 前馈网络维度
E_LAYERS=2                  # 编码器层数
N_HEADS=2                   # 注意力头数
TOP_K=3                     # TimesBlock 的 top-k 周期
USE_NORM=1                  # 启用归一化
DROPOUT=0.2                 # Dropout 比率

# 训练参数
BATCH_SIZE=16               # 批次大小
TRAIN_EPOCHS=50             # CHANGED: 延长到100，结合早停（原50）
LEARNING_RATE=0.0002        # 学习率
LOSS="CE"                   # 损失函数
PATIENCE=15                 # CHANGED: 延长耐心到15，允许恢复（原10）
LRADJ="type3"               # 学习率调整策略（需在Exp中实现ReduceLROnPlateau）
NUM_WORKERS=4               # 数据加载线程数
DES="CWRU_vib_class_GPU0_opt"  # 实验描述
CHECKPOINTS="./checkpoints/GPU0"  # 模型保存路径

# 数据增强参数
AUGMENTATION_RATIO=2        # 增强倍数（统一，避免重复）

# 预热参数（新增）
WARMUP_EPOCHS=5             # LR 预热 epochs（稳定早期训练，减少波动）
# 稳定性参数（新增：可在此统一修改数值）
WEIGHT_DECAY=1e-3           # L2权重衰减（防过拟合，原1e-4）
GRAD_NORM=1.0               # 梯度裁剪阈值（稳定梯度，原4.0）
OPTIMIZER_TYPE="AdamW"      # 优化器类型（AdamW内置权重衰减）

# -------------------------- 3. 执行训练命令 --------------------------
echo -e "\n====================================================="
echo "实验启动：CWRU_10 分类任务（TimesNet·单GPU0·优化版）"
echo "模型保存路径：${CHECKPOINTS}/${MODEL_ID}"
echo "训练数据路径：${ROOT_PATH}/${TRAIN_FILE}"
echo "验证数据路径：${ROOT_PATH}/${VAL_FILE}"
echo "测试数据路径：${ROOT_PATH}/${TEST_FILE}"
echo "核心配置：d_model=${D_MODEL}, batch_size=${BATCH_SIZE}, lr=${LEARNING_RATE}, epochs=${TRAIN_EPOCHS}"
echo "增强配置：aug_ratio=${AUGMENTATION_RATIO}, jitter/scaling/magwarp enabled"
echo "预热配置：warmup_epochs=${WARMUP_EPOCHS}"
echo "=====================================================\n"

python -u run2.py \
  --task_name ${TASK_NAME} \
  --is_training 1 \
  --root_path ${ROOT_PATH} \
  --model_id ${MODEL_ID} \
  --model ${MODEL} \
  --data ${DATA_TYPE} \
  --train_file ${TRAIN_FILE} \
  --test_file ${TEST_FILE} \
  --val_file ${VAL_FILE} \
  --features ${FEATURES} \
  --seq_len ${SEQ_LEN} \
  --pred_len ${PRED_LEN} \
  --enc_in ${ENC_IN} \
  --c_out ${C_OUT} \
  --num_class ${NUM_CLASS} \
  --d_model ${D_MODEL} \
  --d_ff ${D_FF} \
  --e_layers ${E_LAYERS} \
  --n_heads ${N_HEADS} \
  --top_k ${TOP_K} \
  --freq ${FREQ} \
  --use_norm ${USE_NORM} \
  --dropout ${DROPOUT} \
  --batch_size ${BATCH_SIZE} \
  --train_epochs ${TRAIN_EPOCHS} \
  --learning_rate ${LEARNING_RATE} \
  --loss ${LOSS} \
  --patience ${PATIENCE} \
  --lradj ${LRADJ} \
  --num_workers ${NUM_WORKERS} \
  --des ${DES} \
  --checkpoints ${CHECKPOINTS} \
  --use_gpu True \
  --gpu 0 \
  --augmentation_ratio ${AUGMENTATION_RATIO} \
  --jitter \
  --scaling \
  --magwarp \
  --embed fixed \
  --warmup_epochs ${WARMUP_EPOCHS} \
  --weight_decay ${WEIGHT_DECAY} \
  --grad_norm ${GRAD_NORM} \
  --optimizer_type ${OPTIMIZER_TYPE} \
  --use_amp

# -------------------------- 4. 实验结束提示 --------------------------
if [ $? -eq 0 ]; then
  echo -e "\n====================================================="
  echo "实验成功结束！"
  echo "结果路径：${CHECKPOINTS}/${MODEL_ID}"
  echo "GPU0显存已自动清理"
  echo "预期优化：acc应>0.94（波动减少，泛化提升）"
  echo "若Acc低：检查日志，调--d_model=256或添加--timewarp"
  echo "====================================================="
else
  echo -e "\n====================================================="
  echo "实验异常终止，请查看上方日志排查错误"
  echo "建议检查：GPU0显存（nvidia-smi）、数据文件存在、Exp中type3实现"
  echo "====================================================="
  exit 1
fi