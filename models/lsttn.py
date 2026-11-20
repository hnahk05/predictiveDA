import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.paths import PRETRAINED

import configs
from models.transformer import Transformer
from models.graph_wavenet import GraphWaveNet

import time
import warnings


class StackedDilatedConv(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=2, dilation=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, dilation=2, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, dilation=4, padding=4)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, dilation=8, padding=8)
        self.pool4 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        h = self.conv1(x)
        h = F.gelu(h)
        h = self.pool1(h)
        h = self.conv2(h)
        h = F.gelu(h)
        h = self.pool2(h)
        h = self.conv3(h)
        h = F.gelu(h)
        h = self.pool3(h)
        h = self.conv4(h)
        h = F.gelu(h)
        h = self.pool4(h)
        return h


class DynamicGraphConv(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, dropout, support_len=3, order=2, graph_mode="author"):
        super().__init__()
        self.node_vec1 = nn.Parameter(torch.randn(num_nodes, 10))
        self.node_vec2 = nn.Parameter(torch.randn(10, num_nodes))
        input_dim = (order * support_len + 1) * input_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.order = order
        self.graph_mode = graph_mode
        if graph_mode not in ('author', 'baseline', 'corrected'):
            raise ValueError('Unknown graph_mode: ' + graph_mode)

    def conv(self, x, adj_mx):
        if self.graph_mode == 'corrected':
            # Same source n -> destination m convention as GraphWaveNet nconv.
            # Supports are already transposed by doubletransition; do not transpose again.
            return torch.einsum('bnh,nm->bmh', (x, adj_mx)).contiguous()
        return torch.einsum("bnh,nm->bnh", (x, adj_mx)).contiguous()

    def forward(self, x, supports):
        outputs = [x]
        new_supports = []
        new_supports.extend(supports)
        adaptive = torch.softmax(torch.relu(torch.mm(self.node_vec1, self.node_vec2)), dim=1)
        new_supports.append(adaptive)
        for adj_mx in new_supports:
            adj_mx = adj_mx.to(x.device)
            x1 = self.conv(x, adj_mx)
            outputs.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.conv(x1, adj_mx)
                outputs.append(x2)
                x1 = x2
        outputs = torch.cat(outputs, dim=2)
        outputs = self.linear(outputs)
        outputs = self.dropout(outputs)
        return outputs


class LSTTN(nn.Module):
    def __init__(self, cfg, **model_args):
        super().__init__()
        transformer_args = model_args["TRANSFORMER"]
        stgnn_args = model_args["STGNN"]
        lsttn_args = model_args["LSTTN"]
        self.dataset_name = cfg.DATASET_NAME
        self.transformer1 = Transformer(mode="pretrain", **transformer_args)
        self.transformer = Transformer(mode="inference", **transformer_args)
        stgnn = GraphWaveNet(**stgnn_args.GWNET)
        self.pretrained_path = lsttn_args.get("pretrained_path")
        self.ablation = lsttn_args.get("ablation", "full")
        self.ablation_mode = lsttn_args.get('ablation_mode', 'legacy')
        self.transformer_policy = lsttn_args.get('transformer_policy', 'baseline')
        if self.ablation_mode not in ('legacy', 'mask_only', 'skip_compute'):
            raise ValueError(self.ablation_mode)
        if self.transformer_policy not in ('baseline', 'fixed_eval'):
            raise ValueError(self.transformer_policy)
        if self.ablation != 'full':
            warnings.warn('Ablation changes checkpoint semantics; strict load alone does not '
                          'validate this variant. Retrain for an architecture comparison.', UserWarning)
        if self.ablation not in ("full", "wo_lt", "wo_p", "wo_st", "st_only"):
            raise ValueError(self.ablation)
        if self.ablation != "st_only" or self.ablation_mode == 'mask_only':
            self.load_pretrained_transformer()
        if self.transformer_policy == 'fixed_eval':
            self.transformer.eval()

        self.num_nodes = lsttn_args["num_nodes"]
        self.pre_len = lsttn_args["pre_len"]
        self.long_trend_hidden_dim = lsttn_args["long_trend_hidden_dim"]
        self.seasonality_hidden_dim = lsttn_args["seasonality_hidden_dim"]
        self.mlp_hidden_dim = lsttn_args["mlp_hidden_dim"]
        self.dropout = lsttn_args["dropout"]
        self.transformer_hidden_dim = self.transformer.encoder.d_model
        self.supports = lsttn_args["supports"]

        self.long_term_trend_extractor = StackedDilatedConv(self.transformer_hidden_dim, self.long_trend_hidden_dim)
        self.short_term_trend_extractor = stgnn
        self.weekly_seasonality_extractor = DynamicGraphConv(
            self.num_nodes, self.transformer_hidden_dim, self.seasonality_hidden_dim, self.dropout, graph_mode=lsttn_args.get("graph_mode", "author")
        )
        self.daily_seasonality_extractor = DynamicGraphConv(
            self.num_nodes, self.transformer_hidden_dim, self.seasonality_hidden_dim, self.dropout, graph_mode=lsttn_args.get("graph_mode", "author")
        )

        self.trend_seasonality_mlp = nn.Sequential(
            nn.Linear(self.long_trend_hidden_dim + self.seasonality_hidden_dim * 2, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_hidden_dim + stgnn_args["GWNET"]["out_dim"], self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, self.pre_len),
        )

    def load_pretrained_transformer(self):
        if self.pretrained_path:
            state_dict = torch.load(self.pretrained_path, map_location="cpu")
        elif self.dataset_name in ("METRLA", "METR-LA"):
            state_dict = torch.load(PRETRAINED / "METR-LA.pt", map_location="cpu")
        elif self.dataset_name in ("PEMSBAY", "PEMS-BAY"):
            state_dict = torch.load(PRETRAINED / "PEMS-BAY.pt", map_location="cpu")
        elif self.dataset_name == "PEMS04":
            state_dict = torch.load(PRETRAINED / "PEMS04.pt", map_location="cpu")
        elif self.dataset_name == "PEMS08":
            state_dict = torch.load(PRETRAINED / "PEMS08.pt", map_location="cpu")
        else:
            raise ValueError("Unknown dataset: " + self.dataset_name)
        self.transformer.load_state_dict(state_dict["model_state_dict"])
        for param in self.transformer.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, 'transformer_policy', 'baseline') == 'fixed_eval':
            self.transformer.eval()
        return self

    def encode_long(self, x, cache=None):
        if cache is None:
            return self.transformer(x)
        from utils.transformer_cache import encode_cached
        return encode_cached(self.transformer, x, cache, self.transformer_policy)

    def forward(self, short_x, long_x, transformer_cache=None):
        if self.ablation_mode == 'legacy':
            return self._forward_legacy(short_x, long_x, transformer_cache)
        skip = self.ablation_mode == 'skip_compute'
        remove_lt = self.ablation in ('wo_lt', 'st_only')
        remove_p = self.ablation in ('wo_p', 'st_only')
        remove_st = self.ablation == 'wo_st'
        batch, _, nodes, _ = short_x.shape
        lt = short_x.new_zeros(batch, nodes, self.long_trend_hidden_dim)
        week = short_x.new_zeros(batch, nodes, self.seasonality_hidden_dim)
        day = torch.zeros_like(week)
        if not (skip and remove_lt and remove_p):
            x = long_x[..., [0]].permute(0, 2, 3, 1)
            representation = self.encode_long(x, transformer_cache)
            count, width = representation.shape[-2:]
            if not (skip and remove_lt):
                lt = self.long_term_trend_extractor(
                    representation.reshape(-1, count, width).transpose(1, 2))[:, :, -1]
                lt = lt.reshape(batch, nodes, -1)
            if not (skip and remove_p):
                week = self.weekly_seasonality_extractor(representation[:, :, -169, :], self.supports)
                day = self.daily_seasonality_extractor(representation[:, :, -25, :], self.supports)
        if remove_lt:
            lt = torch.zeros_like(lt)
        if remove_p:
            week, day = torch.zeros_like(week), torch.zeros_like(day)
        # st_only excludes the entire long/periodic path including its fusion biases.
        if skip and self.ablation == 'st_only':
            trend = short_x.new_zeros(batch, nodes, self.mlp_hidden_dim)
        else:
            trend = self.trend_seasonality_mlp(torch.cat((lt, week, day), dim=-1))
            if self.ablation == 'st_only':
                trend = torch.zeros_like(trend)
        short_width = self.mlp[0].in_features - self.mlp_hidden_dim
        if skip and remove_st:
            short = short_x.new_zeros(batch, nodes, short_width)
        else:
            short = self.short_term_trend_extractor(short_x.transpose(1, 3)).squeeze(-1).transpose(1, 2)
            if remove_st:
                short = torch.zeros_like(short)
        # Full baseline uses [short, trend], including when either is masked.
        return self.mlp(torch.cat((short, trend), dim=-1))

    def _forward_legacy(self, short_x, long_x, transformer_cache=None):
        if self.ablation == "st_only":
            short = self.short_term_trend_extractor(short_x.transpose(1, 3)).squeeze(-1).transpose(1, 2)
            zeros = short.new_zeros((*short.shape[:2], self.mlp_hidden_dim))
            return self.mlp(torch.cat((zeros, short), dim=-1))
        long_x = long_x[..., [0]]
        long_x = long_x.permute(0, 2, 3, 1)
        batch_size, num_nodes, _, long_seq_len = long_x.size()
        long_repr = self.encode_long(long_x, transformer_cache)
        batch_size, num_nodes, num_subseq, _ = long_repr.size()
        # (batch_size * num_nodes, transformer_hidden_dim, num_subseq)
        long_trend_hidden = self.long_term_trend_extractor(
            torch.reshape(long_repr, (-1, num_subseq, self.transformer_hidden_dim)).transpose(1, 2)
        )[:, :, -1]
        long_trend_hidden = torch.reshape(long_trend_hidden, (batch_size, num_nodes, -1))

        # (batch_size, num_nodes, tsf_hidden_dim)
        last_week_repr = long_repr[:, :, -7 * 24 - 1, :]
        last_day_repr = long_repr[:, :, -25, :]
        # (batch_size, num_nodes, s_hidden_dim)
        weekly_hidden = self.weekly_seasonality_extractor(last_week_repr, self.supports)
        daily_hidden = self.daily_seasonality_extractor(last_day_repr, self.supports)
        # (batch_size, num_nodes, lt_hidden_dim + s_hidden_dim * 2)
        if self.ablation == "wo_lt":
            long_trend_hidden = torch.zeros_like(long_trend_hidden)
        if self.ablation == "wo_p":
            weekly_hidden = torch.zeros_like(weekly_hidden)
            daily_hidden = torch.zeros_like(daily_hidden)
        trend_seasonality_hidden = torch.cat((long_trend_hidden, weekly_hidden, daily_hidden), dim=-1)
        # (batch_size, num_nodes, mlp_hidden_dim)
        trend_seasonality_hidden = self.trend_seasonality_mlp(trend_seasonality_hidden)

        # (batch_size, num_nodes, hidden_dim)
        # supports = [self.support_forward, self.support_backward]
        # (batch_size, num_features, num_nodes, seq_len)
        short_x = torch.transpose(short_x, 1, 3)
        # (batch_size, stgnn_hidden_dim, num_nodes, num_features)
        short_term_hidden = self.short_term_trend_extractor(short_x)
        # (batch_size, num_nodes, stgnn_hidden_dim)
        short_term_hidden = short_term_hidden.squeeze(-1).transpose(1, 2)
        # (batch_size, num_nodes, mlp_hidden_dim * 2)
        hidden = torch.cat((short_term_hidden, trend_seasonality_hidden), dim=-1)
        # (batch_size, num_nodes, pre_len)
        if self.ablation == "wo_st":
            short_term_hidden = torch.zeros_like(short_term_hidden)
            hidden = torch.cat((trend_seasonality_hidden, short_term_hidden), dim=-1)
        output = self.mlp(hidden)
        return output


if __name__ == "__main__":
    from torchinfo import summary
    from configs.METRLA.forecasting import CFG


    model = LSTTN(CFG, **CFG.MODEL.PARAM)

    x_short = torch.randn((32, 12, 207, 3))
    x_long = torch.randn((32, 4032, 207, 3))
    # batch_size, long_seq_len, num_nodes, num_features,

    transformer = model.transformer

    x_long = x_long[..., [0]]
    x_long = x_long.permute(0, 2, 3, 1)
    start = time.time()
    long_repr = transformer(x_long)
    end_transformer = time.time()
    print(end_transformer - start)

    x_short = x_short[..., [0]]
    x_short = x_short.permute(0, 2, 3, 1)
    start = time.time()
    long_repr = transformer(x_short)
    end_transformer = time.time()
    print(end_transformer - start)
