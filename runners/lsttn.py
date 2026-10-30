from utils.reproduction import ReproductionMixin
import time
import math
import torch
import torch.nn as nn
from easytorch.core.data_loader import build_data_loader
from easytorch.utils.dist import master_only
from easytorch.device import to_device
from sklearn.metrics import mean_absolute_error
from easytorch import Runner
from models.lsttn import LSTTN
from utils.log import load_pkl
from utils.datasets import MTSDataset
from utils.load_data import max_min_scale, standard_scale
from utils.metrics import masked_mae, masked_rmse, masked_mape, all_metrics


class LSTTNRunner(ReproductionMixin, Runner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.cfg = cfg
        self.clip = 3
        dataset_name = cfg["DATASET_NAME"]
        # scaler
        if dataset_name in ["PEMS04", "PEMS08"]:
            _min = load_pkl("datasets/" + dataset_name + "/min.pkl")
            _max = load_pkl("datasets/" + dataset_name + "/max.pkl")
            self.scaler = max_min_scale
            self.scaler_args = {"min": _min, "max": _max}
        elif dataset_name in ["PEMS-BAY", "METR-LA"]:
            mean = load_pkl("datasets/" + dataset_name + "/mean.pkl")
            std = load_pkl("datasets/" + dataset_name + "/std.pkl")
            self.scaler = standard_scale
            self.scaler_args = {"mean": mean, "std": std}
        self.loss = masked_mae
        # self.loss = masked_mae_loss

        self.dataset_name = cfg["DATASET_NAME"]
        self.output_seq_len = 12
        self.cl_len = self.output_seq_len
        self.if_cl = True

    def init_training(self, cfg):
        super().init_training(cfg)
        self.register_epoch_meter("train_loss", "train", "{:.4f}")
        self.register_epoch_meter("train_MAPE", "train", "{:.4f}")
        self.register_epoch_meter("train_RMSE", "train", "{:.4f}")

    def init_validation(self, cfg: dict):
        super().init_validation(cfg)
        self.register_epoch_meter("val_loss", "val", "{:.4f}")
        self.register_epoch_meter("val_MAPE", "val", "{:.4f}")
        self.register_epoch_meter("val_RMSE", "val", "{:.4f}")

    @staticmethod
    def define_model(cfg: dict) -> nn.Module:
        return {"LSTTN": LSTTN}[cfg["MODEL"]["NAME"]](cfg, **cfg.MODEL.PARAM)

    def build_train_dataset(self, cfg: dict):
        raw_file_path = cfg["TRAIN"]["DATA"]["DIR"] + "/data.pkl"
        index_file_path = cfg["TRAIN"]["DATA"]["DIR"] + "/train_index.pkl"
        seq_len = cfg["TRAIN"]["DATA"]["SEQ_LEN"]
        batch_size = cfg["TRAIN"]["DATA"]["BATCH_SIZE"]
        dataset = MTSDataset(raw_file_path, index_file_path, seq_len, throw=False, pretrain=False)

        warmup_epochs = cfg["TRAIN"]["WARMUP_EPOCHS"]
        cl_epochs = cfg["TRAIN"]["CL_EPOCHS"]
        self.init_lr = cfg["TRAIN"]["OPTIM"]["PARAM"]["lr"]
        self.itera_per_epoch = math.ceil(len(dataset) / batch_size)
        self.warmup_steps = self.itera_per_epoch * warmup_epochs
        self.cl_steps = self.itera_per_epoch * cl_epochs
        print("cl_steps:{0}".format(self.cl_steps))
        print("warmup_steps:{0}".format(self.warmup_steps))

        return dataset

    @staticmethod
    def build_val_dataset(cfg: dict):
        raw_file_path = cfg["VAL"]["DATA"]["DIR"] + "/data.pkl"
        index_file_path = cfg["VAL"]["DATA"]["DIR"] + "/valid_index.pkl"
        seq_len = cfg["VAL"]["DATA"]["SEQ_LEN"]
        dataset = MTSDataset(raw_file_path, index_file_path, seq_len, pretrain=False)
        print("val len: {0}".format(len(dataset)))
        return dataset

    @staticmethod
    def build_test_dataset(cfg: dict):
        raw_file_path = cfg["TEST"]["DATA"]["DIR"] + "/data.pkl"
        index_file_path = cfg["TEST"]["DATA"]["DIR"] + "/test_index.pkl"
        seq_len = cfg["TEST"]["DATA"]["SEQ_LEN"]
        dataset = MTSDataset(raw_file_path, index_file_path, seq_len, pretrain=False)
        print("test len: {0}".format(len(dataset)))
        return dataset

    def train_iters(self, epoch, iter_index, data):
        iter_num = (epoch - 1) * self.itera_per_epoch + iter_index

        y, x_short, x_long = data
        y = to_device(y)
        x_short = to_device(x_short)
        x_long = to_device(x_long)

        output = self.model(x_short, x_long)
        output = output.transpose(1, 2)

        if iter_num < self.warmup_steps:
            self.cl_len = self.output_seq_len
        elif iter_num == self.warmup_steps:
            self.cl_len = 1
            for param_group in self.optim.param_groups:
                param_group["lr"] = self.init_lr
            print("======== Start curriculum learning... reset the learning rate to {0}. ========".format(self.init_lr))
        else:
            if (iter_num - self.warmup_steps) % self.cl_steps == 0 and self.cl_len <= self.output_seq_len:
                self.cl_len += int(self.if_cl)

        if "max" in self.scaler_args.keys():
            predict = self.scaler(output.transpose(1, 2).unsqueeze(-1), **self.scaler_args).transpose(1, 2).squeeze(-1)
            real_val = self.scaler(y.transpose(1, 2).unsqueeze(-1), **self.scaler_args).transpose(1, 2).squeeze(-1)
            real_val = real_val[..., 0]
            mae_loss = self.loss(predict[:, : self.cl_len, :], real_val[:, : self.cl_len, :])
        else:
            # inverse transform for both predict and real value.
            predict = self.scaler(output, **self.scaler_args)
            real_val = self.scaler(y[:, :, :, 0], **self.scaler_args)
            mae_loss = self.loss(predict[:, : self.cl_len, :], real_val[:, : self.cl_len, :], 0)

        loss = mae_loss
        # metrics
        mape = masked_mape(predict, real_val, 0.0)
        rmse = masked_rmse(predict, real_val, 0.0)

        self.update_epoch_meter("train_loss", loss.item())
        self.update_epoch_meter("train_MAPE", mape.item())
        self.update_epoch_meter("train_RMSE", rmse.item())

        return loss

    def val_iters(self, iter_index, data):
        y, x_short, x_long = data
        y = to_device(y)
        x_short = to_device(x_short)
        x_long = to_device(x_long)

        output = self.model(x_short, x_long)
        output = output.transpose(1, 2)

        if "max" in self.scaler_args.keys():
            predict = self.scaler(output.transpose(1, 2).unsqueeze(-1), **self.scaler_args).transpose(1, 2).squeeze(-1)
            real_val = self.scaler(y.transpose(1, 2).unsqueeze(-1), **self.scaler_args).transpose(1, 2).squeeze(-1)
            real_val = real_val[..., 0]
        else:
            predict = self.scaler(output, **self.scaler_args)
            real_val = self.scaler(y[:, :, :, 0], **self.scaler_args)

        loss = self.loss(predict, real_val, 0.0)
        mape = masked_mape(predict, real_val, 0.0)
        rmse = masked_rmse(predict, real_val, 0.0)

        self.update_epoch_meter("val_loss", loss.item(), n=len(y))
        self.update_epoch_meter("val_MAPE", mape.item())
        self.update_epoch_meter("val_RMSE", rmse.item())

    def on_training_end(self):
        from pathlib import Path
        best = Path(self.ckpt_save_dir) / (self.model_name + "_best_val_loss.pt")
        if not best.is_file():
            raise RuntimeError("No best validation checkpoint: " + str(best))
        self.load_model(str(best))
        self.test(self.cfg)
        super().on_training_end()

    @master_only
    def init_test(self, cfg: dict):
        self.register_epoch_meter("test_loss", "test", "{:.4f}")
        self.register_epoch_meter("test_MAPE", "test", "{:.4f}")
        self.register_epoch_meter("test_RMSE", "test", "{:.4f}")

        self.test_interval = cfg['TEST'].get('INTERVAL', 1)
        self.test_data_loader = self.build_test_data_loader(cfg)
        self.register_epoch_meter('test_time', 'test', '{:.2f} (s)', plt=False)

    def build_test_data_loader(self, cfg: dict):
        dataset = self.build_test_dataset(cfg)
        return build_data_loader(dataset, cfg['TEST']['DATA'])

    @torch.no_grad()
    @master_only
    def test(self, cfg: dict = None, train_epoch: int = None):
        if train_epoch is None:
            self.init_test(cfg)

        test_start_time = time.time()
        self.model.eval()

        from pathlib import Path
        from utils.prediction_export import export_batches, evaluate_export
        directory = Path(self.ckpt_save_dir) / ("test_export_" + str(time.time_ns()))
        export_batches(self, directory)
        scores = evaluate_export(directory)
        for horizon, values in scores.items():
            self.logger.info("horizon %s: %s", horizon, values)
        self.update_epoch_meter("test_time", time.time() - test_start_time)
        return

    def test_iters(self, iter_index: int, data: torch.Tensor or tuple):
        y, x_short, x_long = data
        y = to_device(y)
        x_short = to_device(x_short)
        x_long = to_device(x_long)

        output = self.model(x_short, x_long)
        output = output.transpose(1, 2)
        return output, y


if __name__ == "__main__":
    from configs.PEMS04.forecasting import CFG

    runner = LSTTNRunner(CFG)
    test_loader = runner.build_test_data_loader(CFG)
    runner.test(CFG)

    test_start_time = time.time()
    for iter_index, data in enumerate(runner.test_data_loader):
        runner.test_iters(iter_index, data)
        test_end_time = time.time()
        print(test_end_time - test_start_time)
