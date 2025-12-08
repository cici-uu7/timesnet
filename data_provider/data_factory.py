from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader,CWRUADLoader
from data_provider.uea import collate_fn
from data_provider.CWRU_dataset import Dataset_CWRU  # 导入CWRU数据集类
from torch.utils.data import DataLoader
import torch  # 新增：用于torch.stack

# 数据集映射字典，添加CWRU_10的映射
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
    'CWRU_10': Dataset_CWRU, # CWRU数据集映射
    'CWRU_AD': CWRUADLoader, # CWRU_AD数据集映射
}


def cwru_collate_fn(batch):
    """
    自定义collate_fn for CWRU数据集
    输入: list of dict {'seq_x': tensor(seq_len, enc_in), 'labels': tensor()}
    输出: (batch_x: tensor(batch, seq_len, enc_in), batch_labels: tensor(batch,), None for padding_mask)
    """
    seq_x_list = [item['seq_x'] for item in batch]
    batch_x = torch.stack(seq_x_list, dim=0)  # (batch, seq_len, enc_in)
    batch_labels = torch.stack([item['labels'] for item in batch])  # (batch,)
    return batch_x, batch_labels, None  # 兼容 (batch_x, label, padding_mask)


def data_provider(args, flag):
    """
    数据提供器：根据不同任务和数据集类型返回相应的数据集和数据加载器
    修复：统一通过args传递参数，避免直接传递root_path等单独参数
    """
    # 获取对应的数据集类
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1  # 时间编码方式

    # 通用参数设置
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True  # 测试集不打乱
    drop_last = False  # 默认不丢弃最后一个不完整批次
    batch_size = args.batch_size
    freq = args.freq

    # 异常检测任务
    if args.task_name == 'anomaly_detection':
        drop_last = False
        # 仅传递args参数，避免直接传递root_path
        data_set = Data(
            args=args,
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

    # 分类任务（重点修复CWRU数据集初始化）
    elif args.task_name == 'classification':
        drop_last = False

        # 对于所有分类数据集（包括CWRU），统一初始化：传递args和flag
        data_set = Data(
            args=args,  # 所有参数通过args传递
            flag=flag  # 标识是训练集还是测试集
        )

        # 对于CWRU，使用自定义collate_fn；其他使用UEA的
        if args.data == 'CWRU_10':
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=cwru_collate_fn  # 使用自定义collate_fn
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

    # 其他任务（预测等）
    else:
        if args.data == 'm4':
            drop_last = False

        data_set = Data(
            args=args,
            root_path=args.root_path,  # 修复：添加缺失参数
            data_path=args.data_path,  # 修复：添加缺失参数
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