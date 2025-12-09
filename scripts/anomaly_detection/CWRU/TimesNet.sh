export CUDA_VISIBLE_DEVICES=0

# 1. 运行预处理 (确保生成了 train.csv, test.csv)
python ./data_processing_ad.py

# 2. 运行 TimesNet
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/CWRU_AD \
  --model_id CWRU_AD \
  --model TimesNet \
  --data CWRU_AD \
  --features S \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 32 \
  --d_ff 32 \
  --e_layers 2 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --top_k 3 \
  --anomaly_ratio 0.85 \
  --batch_size 128 \
  --train_epochs 10