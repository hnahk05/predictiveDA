import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from utils.datasets import MTSDataset
from utils.log import clock, load_pkl
from utils.adj_mx import *


def max_min_scale(x, **kwargs):
    _min, _max = kwargs["min"][0, 0, 0], kwargs["max"][0, 0, 0]
    x = (x + 1.0) / 2.0
    x = 1.0 * x * (_max - _min) + _min
    return x


def standard_scale(x, **kwargs):
    mean, std = kwargs["mean"], kwargs["std"]
    x = x * std
    x = x + mean
    return x


@clock
def load_dataset(data_dir, batch_size, dataset_name, seq_len, num_core, device, pretrain):
    data_dict = {}
    raw_file_path = data_dir + "/data.pkl"
    # dataset
    for mode in ["train", "valid", "test"]:
        index_file_path = data_dir + "/" + mode + "_index.pkl"
        dataset = MTSDataset(raw_file_path, index_file_path, seq_len, batch_size, device=device, pretrain=pretrain)
        data_dict[mode + "_loader"] = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=min(batch_size, num_core),
            shuffle=(True if mode == "train" else False),
        )
    # scaler
    if dataset_name in ["PEMS04", "PEMS08"]:
        _min = load_pkl("datasets/" + dataset_name + "/min.pkl")
        _max = load_pkl("datasets/" + dataset_name + "/max.pkl")
        data_dict["scaler"] = max_min_scale
        data_dict["scaler_args"] = {"min": _min, "max": _max}
    elif dataset_name in ["PEMSBAY", "METRLA"]:
        mean = load_pkl("datasets/" + dataset_name + "/mean.pkl")
        std = load_pkl("datasets/" + dataset_name + "/std.pkl")
        data_dict["scaler"] = standard_scale
        data_dict["scaler_args"] = {"mean": mean, "std": std}
    else:
        raise Exception("Unknown Dataset.")
    return data_dict


@clock
def load_adj(file_path, adj_type):
    file_path = Path(file_path).resolve()

    # Kiểm tra file có tồn tại không
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Khong tim thay file adjacency matrix: {file_path}"
        )

    # Đọc đúng file được truyền vào
    data = load_pkl(str(file_path))

    # METR-LA, PEMS-BAY: (sensor_ids, sensor_id_to_ind, adj_mx)
    # PEMS04, PEMS08: có thể chỉ chứa adj_mx
    if isinstance(data, (tuple, list)) and len(data) == 3:
        sensor_ids, sensor_id_to_ind, adj_mx = data
    else:
        adj_mx = data

    # Kiểm tra ma trận kề
    if not hasattr(adj_mx, "shape") or len(adj_mx.shape) != 2:
        raise ValueError(
            f"Du lieu adjacency khong phai ma tran 2 chieu: {file_path}"
        )

    if adj_mx.shape[0] != adj_mx.shape[1]:
        raise ValueError(
            f"Adjacency matrix khong vuong: {adj_mx.shape}"
        )

    if adj_type == "scaled_laplacian":
        adj = [calculate_scaled_laplacian(adj_mx).astype(np.float32).todense()]

    elif adj_type == "normalized_laplacian":
        adj = [calculate_symmetric_normalized_laplacian(adj_mx).astype(np.float32).todense()]

    elif adj_type == "symmetric":
        adj = [symmetric_message_passing_adj(adj_mx).astype(np.float32).todense()]

    elif adj_type == "transition":
        adj = [transition_matrix(adj_mx).T]

    elif adj_type == "doubletransition":
        adj = [
            transition_matrix(adj_mx).T,
            transition_matrix(adj_mx.T).T
        ]

    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]

    elif adj_type == "original":
        adj = adj_mx

    else:
        raise NameError("Unknown adjacency matrix type")

    return adj, adj_mx