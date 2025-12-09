from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, CWRUADLoader
from data_provider.uea import collate_fn
from data_provider.CWRU_dataset import Dataset_CWRU  # 导入CWRU数据集类
from torch.utils.data import DataLoader
import torch

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'CWRU_10': Dataset_CWRU,  # 分类任务
    'CWRU_AD': CWRUADLoader,  # 异常检测任务
}


def cwru_collate_fn(batch):
    """CWRU分类任务专用的collate_fn"""
    seq_x_list = [item['seq_x'] for item in batch]
    batch_x = torch.stack(seq_x_list, dim=0)
    batch_labels = torch.stack([item['labels'] for item in batch])
    return batch_x, batch_labels, None


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    # =======================================================
    # 1. 异常检测任务 (Anomaly Detection)
    # =======================================================
    if args.task_name == 'anomaly_detection':
        drop_last = False
        # 修复：显式传递 root_path，这对于 PSMSegLoader, CWRUADLoader 等是必须的
        data_set = Data(
            args=args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag
        )
        print(f"{flag} dataset size: {len(data_set)}")
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader

    # =======================================================
    # 2. 分类任务 (Classification)
    # =======================================================
    elif args.task_name == 'classification':
        drop_last = False
        # 修复：显式传递 root_path，防止 UEAloader 或 修改后的 Dataset_CWRU 报错
        data_set = Data(
            args=args,
            root_path=args.root_path,
            flag=flag
        )

        if args.data == 'CWRU_10':
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=cwru_collate_fn
            )
        else:
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
            )

        print(f"{flag} dataset size: {len(data_set)}")
        return data_set, data_loader

    # =======================================================
    # 3. 其他任务 (Forecasting / Imputation)
    # =======================================================
    else:
        if args.data == 'm4':
            drop_last = False

        data_set = Data(
            args=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(f"{flag} dataset size: {len(data_set)}")
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader