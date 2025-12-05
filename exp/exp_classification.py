from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
from torch.optim import AdamW  # 新增：支持AdamW
import os
import time
import numpy as np
from utils.timefeatures import TimeFeatureEmbedding
from torch.optim.lr_scheduler import LambdaLR  # 用于预热
# 新增：AMP 支持导入（仅在需要时使用）
if torch.cuda.is_available():
    from torch.cuda.amp import autocast, GradScaler


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
        self.time_encoder = TimeFeatureEmbedding(4, args.d_model) if args.embed == 'timeF' else None

    def _build_model(self):
        # 硬码 enc_in=1 (CWRU 单通道)
        self.args.enc_in = 1
        if not hasattr(self.args, 'num_class') or self.args.num_class == 0:
            self.args.num_class = 10  # fallback

        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        # 支持 optimizer_type: 'AdamW' 或默认 'RAdam'
        optimizer_type = getattr(self.args, 'optimizer_type', 'RAdam')
        weight_decay = getattr(self.args, 'weight_decay', 1e-4)

        if optimizer_type == 'AdamW':
            return AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=weight_decay)
        else:
            # 默认 RAdam
            from torch_optimizer import RAdam  # 假设已安装 torch_optimizer；否则用 optim.Adam
            return RAdam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=weight_decay)

    def _select_criterion(self):
        return nn.CrossEntropyLoss()

    def _generate_time_mark(self, batch_size, seq_len):
        if self.args.task_name == 'classification':
            return None  # 分类无时间标记
        # ... 原代码 ...

    def _unpack_batch(self, batch):
        if isinstance(batch, dict):
            batch_x = batch['seq_x'].float().to(self.device)
            label = batch['labels'].long().to(self.device)
        else:
            batch_x = batch[0].float().to(self.device)
            label = batch[1].long().to(self.device)
        if len(batch_x.shape) == 2:
            batch_x = batch_x.unsqueeze(-1)  # [B, L] -> [B, L, 1]
        return batch_x, label

    def train(self, setting):
        train_data, train_loader = self._get_data('TRAIN')
        vali_data, vali_loader = self._get_data('VAL')  # VAL 而非 TEST
        test_data, test_loader = self._get_data('TEST')

        # 修复 DEBUG：用 _unpack_batch 处理 batch
        try:
            batch = next(iter(train_loader))
            batch_x, _ = self._unpack_batch(batch)
            mean = batch_x.mean().item()
            std = batch_x.std().item()
            print(f"\n[DEBUG] Train data mean: {mean:.4f}, std: {std:.4f}")
        except Exception as e:
            print(f"[DEBUG] Failed to get train_data mean/std: {e}")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 新增：AMP 支持（仅 GPU 启用）
        scaler = None
        use_amp = self.args.use_amp and self.device.type == 'cuda'
        if use_amp:
            scaler = GradScaler()

        # 新增：学习率预热（线性从 0 到 lr，在前 warmup_epochs）
        warmup_epochs = getattr(self.args, 'warmup_epochs', 0)  # 默认0，无预热
        if warmup_epochs > 0:
            def warmup_lambda(epoch):
                if epoch < warmup_epochs:
                    return float(epoch + 1) / float(warmup_epochs)  # 线性预热
                return 1.0

            scheduler_warmup = LambdaLR(model_optim, lr_lambda=warmup_lambda)
        else:
            scheduler_warmup = None

        # 原 lradj scheduler（假设框架有 _select_lr_scheduler；此处简化，结合warmup后用cosine等）
        # 注意：若有原scheduler，warmup后切换；此处假设warmup覆盖早期

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss, start_time = [], time.time()

            for batch in train_loader:
                batch_x, label = self._unpack_batch(batch)
                x_mark_enc = self._generate_time_mark(batch_x.size(0), batch_x.size(1))

                # Forward 和 loss 计算
                if use_amp:
                    with autocast():
                        outputs = self.model(batch_x, x_mark_enc, None, None)
                        loss = criterion(outputs, label)
                else:
                    outputs = self.model(batch_x, x_mark_enc, None, None)
                    loss = criterion(outputs, label)

                model_optim.zero_grad()

                # Backward 和优化
                if use_amp:
                    scaler.scale(loss).backward()
                    # 梯度裁剪（AMP 下需在 unscale 前）
                    grad_norm = getattr(self.args, 'grad_norm', 4.0)
                    scaler.unscale_(model_optim)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_norm)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    # 梯度裁剪（用 args.grad_norm）
                    grad_norm = getattr(self.args, 'grad_norm', 4.0)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_norm)
                    model_optim.step()

                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            vali_loss, vali_acc = self.vali(vali_loader)
            test_loss, test_acc = self.vali(test_loader)

            print(f"Epoch {epoch + 1}, Time: {time.time() - start_time:.1f}s | "
                  f"Train Loss: {train_loss:.3f} | Val Loss: {vali_loss:.3f}, Val Acc: {vali_acc:.3f} | "
                  f"Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}")

            # 修复：每个epoch后step warmup scheduler（无条件，如果启用）
            if scheduler_warmup is not None:
                scheduler_warmup.step()

            # 可添加原 lradj scheduler.step()，如果warmup结束后切换
            # e.g., if epoch >= warmup_epochs: self.scheduler.step()

            early_stopping(-vali_acc, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
        return self.model

    def vali(self, loader):
        self.model.eval()
        preds, trues, losses = [], [], []
        criterion = self._select_criterion()
        use_amp = self.args.use_amp and self.device.type == 'cuda'
        with torch.no_grad():
            for batch in loader:
                batch_x, label = self._unpack_batch(batch)
                x_mark_enc = self._generate_time_mark(batch_x.size(0), batch_x.size(1))
                # 可选：验证也用 autocast（节省内存）
                if use_amp:
                    with autocast():
                        outputs = self.model(batch_x, x_mark_enc, None, None)
                        loss = criterion(outputs, label)
                else:
                    outputs = self.model(batch_x, x_mark_enc, None, None)
                    loss = criterion(outputs, label)
                preds.append(outputs)
                trues.append(label)
                losses.append(loss.item())

        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        acc = cal_accuracy(torch.argmax(preds, dim=1).cpu().numpy(), trues.cpu().numpy())
        return np.average(losses), acc

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data('TEST')
        if test:
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            print('Loaded model for testing')

        # 优化：复用 vali 计算 loss/acc
        test_loss, test_acc = self.vali(test_loader)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
        return test_loss, test_acc  # 返回值，便于日志