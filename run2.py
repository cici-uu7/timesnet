import argparse
import os
import torch
import torch.backends
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np
import pandas as pd
# --- 新增依赖：用于可视化 ---
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as osp
from torch.utils.data import DataLoader

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    torch.cuda.manual_seed_all(fix_seed)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status: 1=train+test, 0=only test')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='unique model id for logging')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type (e.g., CWRU_10)')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of data files')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv',
                        help='legacy data file (for non-classification tasks)')
    parser.add_argument('--train_file', type=str, default='TRAIN.tsv',
                        help='train file name for classification (TSV format)')
    parser.add_argument('--val_file', type=str, default='VAL.tsv',
                        help='val file name for classification (TSV format)')
    parser.add_argument('--test_file', type=str, default='TEST.tsv',
                        help='test file name for classification (TSV format)')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task type: [M:multivariate->multivariate, S:univariate->univariate, MS:multivariate->univariate]')
    parser.add_argument('--target', type=str, default='OT', help='target feature for S/MS forecasting tasks')
    parser.add_argument('--freq', type=str, default='h',
                        help='time feature encoding freq: [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='path to save model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length (for forecasting)')
    parser.add_argument('--label_len', type=int, default=48, help='start token length (for decoder in forecasting)')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length (for forecasting)')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4 dataset')
    parser.add_argument('--inverse', action='store_true', help='inverse normalize output (for forecasting)',
                        default=False)

    # imputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio for imputation task')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25,
                        help='prior anomaly ratio (%%) for anomaly detection')

    # classification task
    parser.add_argument('--num_class', type=int, default=10,
                        help='number of classes for classification task (CWRU_10 uses 10)')
    parser.add_argument('--input_dim', type=int, default=1000,
                        help='feature dimension of input sequence (CWRU uses 1000)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba model')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba model')
    parser.add_argument('--top_k', type=int, default=3, help='top-k value for TimesBlock (adjusted for CWRU)')
    parser.add_argument('--num_kernels', type=int, default=6, help='number of kernels for Inception module')
    parser.add_argument('--enc_in', type=int, default=7,
                        help='encoder input size (will be overwritten for classification)')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size (unused for classification)')
    parser.add_argument('--c_out', type=int, default=7,
                        help='model output size (will be overwritten for classification)')
    parser.add_argument('--d_model', type=int, default=256,
                        help='model hidden dimension (reduced from 512 for CWRU efficiency)')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='number of attention heads (reduced from 8 for CWRU efficiency)')
    parser.add_argument('--e_layers', type=int, default=2, help='number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='number of decoder layers (unused for classification)')
    parser.add_argument('--d_ff', type=int, default=1024,
                        help='dimension of feed-forward network (reduced from 2048 for CWRU)')
    parser.add_argument('--moving_avg', type=int, default=25,
                        help='window size for moving average (unused for classification)')
    parser.add_argument('--factor', type=int, default=1, help='attention factor')
    parser.add_argument('--distil', action='store_false',
                        help='disable distilling in encoder (default: enable distilling)',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout rate (increased to 0.2 for CWRU overfitting prevention)')
    parser.add_argument('--embed', type=str, default='fixed',
                        help='time feature encoding: [timeF:formula, fixed:fixed embedding, learned:learned embedding] (fixed for CWRU)')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0:channel dependence, 1:channel independence (for FreTS model)')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='series decomposition method: [moving_avg, dft_decomp] (unused for classification)')
    parser.add_argument('--use_norm', type=int, default=1,
                        help='1:enable data normalization, 0:disable (enable for CWRU)')
    parser.add_argument('--down_sampling_layers', type=int, default=0,
                        help='number of down-sampling layers (unused for CWRU)')
    parser.add_argument('--down_sampling_window', type=int, default=1,
                        help='down-sampling window size (unused for CWRU)')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down-sampling method: [avg, max, conv] (unused for CWRU)')
    parser.add_argument('--seg_len', type=int, default=96, help='segment length for SegRNN (unused for CWRU)')

    # optimization
    parser.add_argument('--num_workers', type=int, default=4,
                        help='data loader workers (reduced from 10 for CWRU memory)')
    parser.add_argument('--itr', type=int, default=1, help='number of repeated experiments')
    parser.add_argument('--train_epochs', type=int, default=30,
                        help='training epochs (increased to 30 for CWRU convergence)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size (increased to 64 for CWRU efficiency)')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience (increased to 5 for CWRU)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate (increased to 0.001 for CWRU)')
    parser.add_argument('--des', type=str, default='CWRU_10_classification',
                        help='experiment description (default for CWRU)')
    parser.add_argument('--loss', type=str, default='MSE',
                        help='loss function: [MSE:forecasting, CE:classification] (will be overwritten for classification)')
    parser.add_argument('--lradj', type=str, default='type2',
                        help='learning rate adjust strategy: [type1:step, type2:cosine] (type2 for CWRU)')
    parser.add_argument('--use_amp', action='store_true', help='enable automatic mixed precision training',
                        default=False)

    # GPU
    parser.add_argument('--use_gpu', type=str, default='True', help='1:use GPU, 0:use CPU/MPS')  # 改为str，避免bool解析问题
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id (for single GPU)')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='GPU type: [cuda:NVIDIA, mps:Apple Silicon]')
    parser.add_argument('--use_multi_gpu', action='store_true', help='enable multi-GPU training', default=False)
    parser.add_argument('--devices', type=str, default='0', help='GPU device ids for multi-GPU (e.g., "0,1,2")')

    # de-stationary projector params (unused for classification)
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of de-stationary projector')
    parser.add_argument('--p_hidden_layers', type=int, default=2,
                        help='number of hidden layers in de-stationary projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='enable DTW metric (time-consuming, disable for CWRU)')

    # Augmentation (for classification)
    parser.add_argument('--augmentation_ratio', type=int, default=1, help='data augmentation ratio (1=no augmentation)')
    parser.add_argument('--seed', type=int, default=2021, help='random seed for augmentation (align with global seed)')
    parser.add_argument('--jitter', default=False, action="store_true", help='enable jitter augmentation (for CWRU)')
    parser.add_argument('--scaling', default=False, action="store_true", help='enable scaling augmentation (for CWRU)')
    parser.add_argument('--permutation', default=False, action="store_true",
                        help='enable equal-length permutation augmentation')
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help='enable random-length permutation augmentation')
    parser.add_argument('--magwarp', default=False, action="store_true",
                        help='enable magnitude warp augmentation (for CWRU)')
    parser.add_argument('--timewarp', default=False, action="store_true",
                        help='enable time warp augmentation (for CWRU)')
    parser.add_argument('--windowslice', default=False, action="store_true", help='enable window slice augmentation')
    parser.add_argument('--windowwarp', default=False, action="store_true", help='enable window warp augmentation')
    parser.add_argument('--rotation', default=False, action="store_true",
                        help='enable rotation augmentation (unused for CWRU)')
    parser.add_argument('--spawner', default=False, action="store_true",
                        help='enable SPAWNER augmentation (unused for CWRU)')
    parser.add_argument('--dtwwarp', default=False, action="store_true",
                        help='enable DTW warp augmentation (time-consuming)')
    parser.add_argument('--shapedtwwarp', default=False, action="store_true",
                        help='enable shape DTW warp augmentation (time-consuming)')
    parser.add_argument('--wdba', default=False, action="store_true",
                        help='enable weighted DBA augmentation (unused for CWRU)')
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help='enable discriminative DTW augmentation (unused for CWRU)')
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help='enable discriminative shape DTW augmentation (unused for CWRU)')
    parser.add_argument('--extra_tag', type=str, default="", help='extra_tag for experiment logging')

    # TimeXer (unused for CWRU)
    parser.add_argument('--patch_len', type=int, default=16, help='patch length for TimeXer model')

    # --- 新增：可视化输出路径 ---
    parser.add_argument('--vis_path', type=str, default='./results/', help='path to save visualization PNG files')

    # --- 新增：解决波动参数 ---
    parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup epochs for LR scheduler (to stabilize early training)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 weight decay for optimizer (prevent overfitting)')
    parser.add_argument('--grad_norm', type=float, default=4.0, help='gradient clipping norm')

    # --- 新增：优化器类型 ---
    parser.add_argument('--optimizer_type', type=str, default='Adam', help='optimizer type: Adam or AdamW')

    args = parser.parse_args()

    # 修复：转换 use_gpu 为 bool
    args.use_gpu = args.use_gpu.lower() in ['true', '1', 'yes', 'y']

    # --- 新增：确保可视化输出目录存在 ---
    os.makedirs(args.vis_path, exist_ok=True)

    # --- 分类任务数据预处理 ---
    if args.task_name == 'classification':
        train_full_path = os.path.join(args.root_path, args.train_file)
        val_full_path = os.path.join(args.root_path, args.val_file)
        test_full_path = os.path.join(args.root_path, args.test_file)

        if not os.path.exists(train_full_path):
            raise FileNotFoundError(
                f"Train file not found: {train_full_path}\nPlease check --root_path and --train_file")
        if not os.path.exists(val_full_path):
            raise FileNotFoundError(
                f"Val file not found: {val_full_path}\nPlease check --root_path and --val_file")
        if not os.path.exists(test_full_path):
            raise FileNotFoundError(f"Test file not found: {test_full_path}\nPlease check --root_path and --test_file")

        try:
            df_train = pd.read_csv(train_full_path, sep='\t', header=None)
            df_val = pd.read_csv(val_full_path, sep='\t', header=None)
            df_test = pd.read_csv(test_full_path, sep='\t', header=None)
        except Exception as e:
            raise RuntimeError(f"Failed to load TSV files: {str(e)}\nPlease ensure files are in valid TSV format")

        train_feature_dim = df_train.shape[1] - 1
        val_feature_dim = df_val.shape[1] - 1
        test_feature_dim = df_test.shape[1] - 1
        if train_feature_dim != args.input_dim or val_feature_dim != args.input_dim or test_feature_dim != args.input_dim:
            raise ValueError(
                f"CWRU data feature dimension mismatch:\n"
                f"Expected {args.input_dim}, got train={train_feature_dim}, val={val_feature_dim}, test={test_feature_dim}\n"
                f"Please check data files or adjust --input_dim"
            )

        args.seq_len = args.input_dim
        args.enc_in = 1
        args.c_out = args.num_class
        args.features = 'S'
        args.loss = 'CE'
        args.label_len = 0
        args.pred_len = 0

        args.train_data = df_train.values
        args.val_data = df_val.values
        args.test_data = df_test.values
        args.train_labels = df_train.iloc[:, 0].values
        args.val_labels = df_val.iloc[:, 0].values
        args.test_labels = df_test.iloc[:, 0].values
        print(f"Train label distribution: {np.unique(args.train_labels, return_counts=True)}")
        print(f"Val label distribution: {np.unique(args.val_labels, return_counts=True)}")
        print(f"Test label distribution: {np.unique(args.test_labels, return_counts=True)}")

        unique_train_labels = sorted(np.unique(args.train_labels))
        unique_val_labels = sorted(np.unique(args.val_labels))
        unique_test_labels = sorted(np.unique(args.test_labels))
        print(
            f"Successfully loaded CWRU classification data:\n"
            f"Train: {df_train.shape} (samples: {df_train.shape[0]}, features: {train_feature_dim}, labels: {unique_train_labels})\n"
            f"Val: {df_val.shape} (samples: {df_val.shape[0]}, features: {val_feature_dim}, labels: {unique_val_labels})\n"
            f"Test: {df_test.shape} (samples: {df_test.shape[0]}, features: {test_feature_dim}, labels: {unique_test_labels})\n"
            f"Model config adjusted: seq_len={args.seq_len}, enc_in={args.enc_in}, c_out={args.c_out}, loss={args.loss}"
        )

        # --- 数据统计打印（移入 if 内） ---
        print(f"Train X sample min: {args.train_data[:, 1:].min()}, max: {args.train_data[:, 1:].max()}")
        print(f"Train X std: {np.std(args.train_data[:, 1:])}")
        print(f"Train label dtype: {args.train_labels.dtype}")

        # --- 新增：针对波动问题的参数调整（仅在默认值时覆盖，允许命令行覆盖） ---
        # 启用数据增强：增加训练多样性，减少过拟合
        args.augmentation_ratio = 2  # 每个样本生成1个增强版本（默认1 → 2）
        args.jitter = True
        args.scaling = True
        args.magwarp = True  # 适合振动信号

        # 优化设置：仅在未指定（默认值）时设置
        default_lr = 0.001
        default_batch = 64
        default_epochs = 30
        default_patience = 5
        default_lradj = 'type2'

        if args.learning_rate == default_lr:
            args.learning_rate = 0.0005  # 更低lr，稳定梯度
        if args.batch_size == default_batch:
            args.batch_size = 32  # 更小batch，减少噪声
        if args.patience == default_patience:
            args.patience = 10  # 更长patience，允许恢复
        if args.train_epochs == default_epochs:
            args.train_epochs = 50  # 更多epochs，结合早停
        if args.lradj == 'type1':  # 只在非推荐时覆盖（type2 是默认推荐）
            args.lradj = 'type2'  # cosine退火，更平滑

        # --- 增强稳定性参数（高优先：针对波动） ---
        args.warmup_epochs = 5  # 预热5个epoch，避免早期震荡（默认已5）
        args.weight_decay = 1e-3  # 增加L2正则，防止过拟合（默认1e-4 → 1e-3）
        args.grad_norm = 1.0  # 梯度裁剪阈值（原4.0太松，减少梯度爆炸）
        args.optimizer_type = 'AdamW'  # 切换到AdamW，内置权重衰减（默认'Adam' → 'AdamW'）

        # 传递新参数到Exp（假设Exp_Classification支持warmup_epochs, weight_decay, grad_norm, optimizer_type）
        # 这些将在Exp_Classification._select_optimizer和train中用于预热、clip和优化器选择

        print(f"[Anti-Volatility] Enabled augmentations: jitter={args.jitter}, scaling={args.scaling}, magwarp={args.magwarp}")
        print(f"[Anti-Volatility] Adjusted (if default): lr={args.learning_rate}, batch={args.batch_size}, epochs={args.train_epochs}, patience={args.patience}")
        print(f"[Stability Enhancement] Warmup={args.warmup_epochs}, Weight Decay={args.weight_decay}, Grad Norm={args.grad_norm}, Optimizer={args.optimizer_type}")

    # --- 设备初始化 ---
    if args.use_gpu:
        if args.gpu_type == 'cuda' and torch.cuda.is_available():
            args.device = torch.device(f'cuda:{args.gpu}')
            print(f"Using CUDA GPU: {args.device}")
            if args.use_multi_gpu:
                args.devices = args.devices.replace(' ', '').split(',')
                args.device_ids = [int(id_) for id_ in args.devices if id_.isdigit()]
                if len(args.device_ids) == 0:
                    raise ValueError(f"Invalid GPU devices: {args.devices}\nPlease use format like '0,1,2'")
                args.gpu = args.device_ids[0]
                print(f"Multi-GPU enabled: devices {args.device_ids}")
        elif args.gpu_type == 'mps' and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = torch.device("mps")
            print("Using Apple MPS GPU")
        else:
            args.device = torch.device("cpu")
            print(f"GPU unavailable (cuda/mps not found), using CPU")
    else:
        args.device = torch.device("cpu")
        print("Using CPU (--use_gpu set to False)")

    print('\nArgs in experiment:')
    print_args(args)

    task_exp_map = {
        'long_term_forecast': Exp_Long_Term_Forecast,
        'short_term_forecast': Exp_Short_Term_Forecast,
        'imputation': Exp_Imputation,
        'anomaly_detection': Exp_Anomaly_Detection,
        'classification': Exp_Classification
    }
    Exp = task_exp_map.get(args.task_name, Exp_Long_Term_Forecast)


    # --- 新增：可视化函数 ---
    def visualize_classification_results(preds, labels, inputs, setting, vis_path, num_samples=5):
        """
        可视化分类任务的预测结果和时间序列波形，保存为 PNG。

        Args:
            preds: 模型预测的 softmax 概率，形状 [N, num_class]
            labels: 真实标签，形状 [N]
            inputs: 输入时间序列，形状 [N, seq_len]
            setting: 实验标识字符串
            vis_path: 保存图片的路径
            num_samples: 可视化的样本数量
        """
        # 确保样本数量不超过实际数据量
        num_samples = min(num_samples, len(labels))
        sample_indices = np.random.choice(len(labels), num_samples, replace=False)

        # 设置 Matplotlib 风格
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(15, 5 * num_samples))

        for idx, sample_idx in enumerate(sample_indices):
            # 子图 1：时间序列波形
            plt.subplot(num_samples, 2, 2 * idx + 1)
            plt.plot(inputs[sample_idx], label='Input Sequence', color='blue')
            plt.title(f'Sample {sample_idx} - True Label: {labels[sample_idx]}')
            plt.xlabel('Time Step')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True)

            # 子图 2：预测概率柱状图
            plt.subplot(num_samples, 2, 2 * idx + 2)
            pred_probs = preds[sample_idx]
            plt.bar(range(len(pred_probs)), pred_probs, color='orange', alpha=0.7)
            plt.axvline(labels[sample_idx], color='red', linestyle='--', label='True Label')
            plt.title(f'Sample {sample_idx} - Predicted Probabilities')
            plt.xlabel('Class')
            plt.ylabel('Probability')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        output_path = osp.join(vis_path, f'{setting}_classification_vis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {output_path}")


    if args.is_training:
        for exp_idx in range(args.itr):
            setting = (
                f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_"
                f"ft{args.features}_sl{args.seq_len}_cl{args.num_class}_"
                f"dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_"
                f"df{args.d_ff}_dp{args.dropout}_lr{args.learning_rate}_"
                f"{args.des}_{exp_idx}_{args.extra_tag}"
            ).strip('_')

            print(f'\n>>>>>>> Start training: {setting} (Experiment {exp_idx + 1}/{args.itr}) >>>>>>>>>')
            exp = Exp(args)
            exp.train(setting)

            print(f'>>>>>>> Start testing: {setting} (Experiment {exp_idx + 1}/{args.itr}) <<<<<<<<<')
            # --- 修改：运行测试以获取指标，但不使用 return_preds ---
            exp.test(setting)
            # --- 新增：手动获取预测结果用于可视化（修复：使用 _get_data 获取 loader） ---
            if args.task_name == 'classification':
                print('Getting predictions for visualization...')
                exp.model.eval()
                preds = []
                labels_list = []
                inputs_list = []
                # 修复：动态获取 test_loader
                _, test_loader = exp._get_data(flag='TEST')
                with torch.no_grad():
                    for batch in test_loader:
                        if isinstance(batch, dict):
                            batch_x = batch['seq_x'].float().to(exp.device)
                            batch_y = batch['labels'].long().to(exp.device)
                        else:
                            batch_x = batch[0].float().to(exp.device)
                            batch_y = batch[1].long().to(exp.device)
                        # 修复：分类任务模型调用，传入 None 参数适配 forward 签名
                        output = exp.model(batch_x, None, None,
                                           None)  # x_enc=batch_x, x_mark_enc=None, x_dec=None, x_mark_dec=None
                        pred = output  # logits [B, num_class]
                        preds.append(pred.detach().cpu().numpy())
                        labels_list.append(batch_y.detach().cpu().numpy())
                        # 假设输入形状为 [B, seq_len, 1] 或 [B, 1, seq_len]，挤压通道维度得到 [B, seq_len]
                        if len(batch_x.shape) == 3 and batch_x.shape[1] == 1:
                            inputs_list.append(batch_x.squeeze(1).detach().cpu().numpy())
                        elif len(batch_x.shape) == 3 and batch_x.shape[2] == 1:
                            inputs_list.append(batch_x.squeeze(2).detach().cpu().numpy())
                        else:
                            inputs_list.append(batch_x.detach().cpu().numpy())  # fallback
                preds = np.vstack(preds)
                labels = np.hstack(labels_list)
                inputs = np.vstack(inputs_list)
                # 转换为 softmax 概率
                preds_soft = torch.softmax(torch.tensor(preds), dim=-1).numpy()
                visualize_classification_results(preds_soft, labels, inputs, setting, args.vis_path)

            if args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
            elif args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
    else:
        setting = (
            f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_"
            f"ft{args.features}_sl{args.seq_len}_cl{args.num_class}_"
            f"dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_"
            f"df{args.d_ff}_dp{args.dropout}_lr{args.learning_rate}_"
            f"{args.des}_0_{args.extra_tag}"
        ).strip('_')

        print(f'\n>>>>>>> Start testing (only): {setting} <<<<<<<<<')
        exp = Exp(args)
        # --- 修改：运行测试以获取指标，但不使用 return_preds ---
        exp.test(setting, test=1)
        # --- 新增：手动获取预测结果用于可视化（修复：使用 _get_data 获取 loader） ---
        if args.task_name == 'classification':
            print('Getting predictions for visualization...')
            exp.model.eval()
            preds = []
            labels_list = []
            inputs_list = []
            # 修复：动态获取 test_loader
            _, test_loader = exp._get_data(flag='TEST')
            with torch.no_grad():
                for batch in test_loader:
                    if isinstance(batch, dict):
                        batch_x = batch['seq_x'].float().to(exp.device)
                        batch_y = batch['labels'].long().to(exp.device)
                    else:
                        batch_x = batch[0].float().to(exp.device)
                        batch_y = batch[1].long().to(exp.device)
                    # 对于分类任务，模型前向传播（假设无时间标记；如果需要 x_mark_enc，可添加）
                    output = exp.model(batch_x)
                    pred = output  # logits [B, num_class]
                    preds.append(pred.detach().cpu().numpy())
                    labels_list.append(batch_y.detach().cpu().numpy())
                    # 假设输入形状为 [B, seq_len, 1] 或 [B, 1, seq_len]，挤压通道维度得到 [B, seq_len]
                    if len(batch_x.shape) == 3 and batch_x.shape[1] == 1:
                        inputs_list.append(batch_x.squeeze(1).detach().cpu().numpy())
                    elif len(batch_x.shape) == 3 and batch_x.shape[2] == 1:
                        inputs_list.append(batch_x.squeeze(2).detach().cpu().numpy())
                    else:
                        inputs_list.append(batch_x.detach().cpu().numpy())  # fallback
            preds = np.vstack(preds)
            labels = np.hstack(labels_list)
            inputs = np.vstack(inputs_list)
            # 转换为 softmax 概率
            preds_soft = torch.softmax(torch.tensor(preds), dim=-1).numpy()
            visualize_classification_results(preds_soft, labels, inputs, setting, args.vis_path)

        if args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
        elif args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()