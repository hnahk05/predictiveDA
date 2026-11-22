"""Inference sensitivity only, NOT retrained ablation; train-only input baselines and short-history IG."""
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import runpy
import sys
import time

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANALYSIS = "inference_sensitivity_not_retrained_ablation"
HORIZONS = (3, 6, 12)
ORIGIN = datetime(2016, 7, 1)


def require(value, message):
    if not value:
        raise ValueError(message)


def integrated_gradients(scalar_outputs, original, baseline, steps=20):
    """20 trapezoidal intervals = 21 alpha points, gradients w.r.t. inputs only."""
    gradient_sum = None
    endpoints = []
    for k in range(steps + 1):
        interpolated = (baseline + (k / steps) * (original - baseline)).detach().requires_grad_(True)
        values = scalar_outputs(interpolated)
        require(values.ndim == 1 and torch.isfinite(values).all().item(), "Invalid IG scalar outputs")
        if gradient_sum is None:
            gradient_sum = torch.zeros((len(values),) + original.shape, device=original.device, dtype=original.dtype)
        for j in range(len(values)):
            gradient = torch.autograd.grad(values[j], interpolated, retain_graph=j < len(values) - 1)[0]
            require(torch.isfinite(gradient).all().item(), "Nonfinite IG gradients")
            gradient_sum[j] += gradient.detach() * (0.5 if k in (0, steps) else 1.0)
        if k in (0, steps):
            endpoints.append(values.detach().cpu().numpy())
        del values, interpolated
    attribution = (original - baseline).unsqueeze(0) * gradient_sum / steps
    return attribution.detach(), endpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    tick = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    sys.path.insert(0, str(root))
    from scripts.extension.run_baselines_pems08 import sha256, target_times, fit_ha, sufficient_stats, finish_stats
    from utils.datasets import MTSDataset
    from utils.load_data import max_min_scale
    from utils import cuda_softmax_compat as compat
    from models.lsttn import LSTTN
    require(platform.python_version() == "3.9.25" and torch.__version__ == "1.8.1+cu111", "Use verified WSL environment")
    require(torch.cuda.is_available(), "CUDA unavailable: stop")
    require("3050" in torch.cuda.get_device_name(0), "Expected RTX 3050")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    os.environ["LSTTN_SPLIT_SOFTMAX"] = "1"
    compat_enabled = compat.activate_from_env()
    split_count = [0]
    original_split = compat.split_softmax

    def counted_split(*positional, **keywords):
        split_count[0] += 1
        return original_split(*positional, **keywords)

    compat.split_softmax = counted_split
    out = root / "results/extension"
    extra = out / "sensitivity"
    extra.mkdir(parents=True, exist_ok=True)
    protected = [root / name for name in ("README.md", ".gitignore", "utils/datasets.py", "utils/load_data.py", "utils/metrics.py",
                 "utils/cuda_softmax_compat.py", "scripts/extension/run_baselines_pems08.py", "results/experiment_metrics.csv",
                 "results/experiment_manifest.json", "datasets/PEMS08/data.pkl", "datasets/PEMS08/train_index.pkl",
                 "datasets/PEMS08/valid_index.pkl", "datasets/PEMS08/test_index.pkl", "datasets/PEMS08/min.pkl", "datasets/PEMS08/max.pkl")]
    protected += sorted((root / "models").glob("*.py")) + sorted((root / "configs").rglob("*.py"))
    before = {p.relative_to(root).as_posix(): sha256(p) for p in protected}
    inputs = root / "artifacts/report-inputs/PEMS08"
    manifest = json.loads((root / "results/experiment_manifest.json").read_text(encoding="utf-8"))
    reference = next(r for r in manifest["runs"] if r["dataset"] == "PEMS08")
    locator = json.loads((out / "inference_cost.json").read_text(encoding="utf-8"))["checkpoint"]
    checkpoint = root / locator["path"]
    run_reference = json.loads((inputs / "result.json").read_text(encoding="utf-8"))
    expected = [v for k, v in run_reference["artifact_sha256"].items() if k.endswith("/LSTTN_best_val_loss.pt")]
    require(len(expected) == 1 and sha256(checkpoint) == expected[0] == locator["sha256"], "Checkpoint hash mismatch")
    pretrained = root / "pretrained_transformers/PEMS08.pt"
    require(sha256(pretrained) == reference["provided_pretrained_sha256"], "Pretrained hash mismatch")
    cfg = runpy.run_path(str(root / "configs/PEMS08/forecasting.py"))["CFG"]
    cfg.MODEL.PARAM.LSTTN["pretrained_path"] = str(pretrained)
    model = LSTTN(cfg, **cfg.MODEL.PARAM)
    saved = torch.load(str(checkpoint), map_location="cpu")
    require(saved["epoch"] == reference["best_epoch"] == 68, "Wrong epoch")
    keys = model.load_state_dict(saved["model_state_dict"], strict=True)
    require(not keys.missing_keys and not keys.unexpected_keys, "Strict load failed")
    del saved
    model.eval()
    require(all(not m.training for m in model.modules()), "Model not eval")

    def model_hash():
        digest = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    state_before = model_hash()
    device = torch.device("cuda:0")
    model.to(device)
    dataset = MTSDataset(str(root / "datasets/PEMS08/data.pkl"), str(root / "datasets/PEMS08/test_index.pkl"), 4032, throw=False, pretrain=False)
    require(len(dataset) == 3566, "Unexpected test size")
    indices = np.asarray(dataset.index, dtype=np.int64)
    with (root / "datasets/PEMS08/train_index.pkl").open("rb") as stream:
        train_index = np.asarray(pickle.load(stream, encoding="latin1"))
    with (root / "datasets/PEMS08/valid_index.pkl").open("rb") as stream:
        valid_index = np.asarray(pickle.load(stream, encoding="latin1"))
    with (root / "datasets/PEMS08/min.pkl").open("rb") as stream:
        minimum = pickle.load(stream, encoding="latin1")
    with (root / "datasets/PEMS08/max.pkl").open("rb") as stream:
        maximum = pickle.load(stream, encoding="latin1")
    scaler_args = {"min": minimum, "max": maximum}
    raw_scale = float(maximum[0, 0, 0] - minimum[0, 0, 0]) / 2
    train_ids = target_times(train_index)
    require(len(train_ids) == 10711 and not np.intersect1d(train_ids, target_times(indices)).size, "Unexpected train timestamps")
    overlap = len(np.intersect1d(train_ids, target_times(valid_index)))
    data = dataset.data.numpy()
    steps = np.arange(len(data))
    group = ((ORIGIN.weekday() + steps // 288) % 7) * 288 + steps % 288
    profile, counts = fit_ha(data[..., 0], train_ids, group)
    profile = profile.astype(np.float32)
    mean_node = data[train_ids].astype(np.float64).mean(axis=0).astype(np.float32)
    rng = np.random.default_rng(0)
    drawn = rng.choice(len(dataset), size=256, replace=False)
    selected, ig_ids = np.sort(drawn), np.sort(drawn[:32])
    shuffle_rng = np.random.default_rng(1)
    permutation = shuffle_rng.permutation(256)
    while np.any(permutation == np.arange(256)):
        permutation = shuffle_rng.permutation(256)
    donors = selected[permutation]
    noise = np.random.default_rng(2).normal(0, 0.01, size=(256, 12, 170)).astype(np.float32)
    selection = pd.DataFrame(dict(analysis_type=ANALYSIS, test_position=selected,
        input_start=indices[selected, 0], forecast_start=indices[selected, 1], forecast_end=indices[selected, 2],
        long_shuffle_donor_test_position=donors, used_for_IG=np.isin(selected, ig_ids)))
    selection.to_csv(extra / "selected_windows.csv", index=False)
    print(ANALYSIS, "— strict checkpoint epoch 68; CUDA inference batch", args.batch_size, flush=True)

    # Existing exports are the control; no new baseline target is invented.
    hashes = json.loads((inputs / "export_hashes.json").read_text(encoding="utf-8"))
    batch_files = sorted(inputs.glob("batch_*.npz"))
    require(len(hashes) == 113 and len(batch_files) == 112, "Incomplete export")
    require({item["file"] for item in hashes} == {p.name for p in batch_files} | {"metadata.json"}, "Unexpected export manifest")
    for item in hashes:
        require(Path(item["file"]).name == item["file"] and sha256(inputs / item["file"]) == item["sha256"], "Export hash mismatch")
    reference_p = np.empty((256, 12, 170), dtype=np.float32)
    reference_y = np.empty_like(reference_p)
    offset = 0
    seen = 0
    for file in batch_files:
        with np.load(file, allow_pickle=False) as batch:
            idx = batch["index"]
            require(np.array_equal(idx, indices[offset:offset + len(idx)]), "Export index mismatch")
            local = np.flatnonzero((selected >= offset) & (selected < offset + len(idx)))
            positions = selected[local] - offset
            reference_p[local] = batch["prediction"][positions]
            reference_y[local] = batch["target"][positions]
            seen += len(local)
            offset += len(idx)
    require(offset == 3566 and seen == 256, "Incomplete subset reference")

    def score(prediction):
        result = {}
        for h in HORIZONS:
            s = finish_stats(sufficient_stats(prediction[:, h - 1], reference_y[:, h - 1]))
            result[h] = dict(MAE=s["MAE"], RMSE=s["RMSE"])
        return result

    def infer(condition):
        predictions = np.empty_like(reference_p)
        with torch.no_grad():
            for start in range(0, 256, args.batch_size):
                positions = selected[start:start + args.batch_size]
                items = [dataset[int(i)] for i in positions]
                short = torch.stack([item[1] for item in items])
                long = torch.stack([item[2] for item in items])
                y = max_min_scale(torch.stack([item[0] for item in items])[..., 0], **scaler_args).numpy()
                require(np.array_equal(y, reference_y[start:start + len(items)]), "Subset target differs from export")
                if condition == "long_train_calendar_profile":
                    for j, pos in enumerate(positions):
                        end = int(indices[pos, 1])
                        long[j, ..., 0] = torch.from_numpy(profile[group[end - 4032:end]])
                elif condition == "long_shuffled_windows":
                    long = torch.stack([dataset[int(i)][2] for i in donors[start:start + len(items)]])
                elif condition == "short_train_node_mean":
                    short = torch.from_numpy(mean_node).view(1, 1, 170, 3).expand(len(items), 12, -1, -1).clone()
                elif condition == "short_flow_gaussian_noise":
                    short[..., 0] += torch.from_numpy(noise[start:start + len(items)])
                output = model(short.to(device), long.to(device))
                require(not output.requires_grad and torch.isfinite(output).all().item(), "Invalid no_grad inference")
                # Inverse scaling on CPU exactly as the saved export pipeline.
                prediction = max_min_scale(output.transpose(1, 2).cpu(), **scaler_args)
                predictions[start:start + len(items)] = prediction.numpy()
                del short, long, output, prediction
        return predictions

    print("Sanity: running original 256-window subset...", flush=True)
    original_predictions = infer("original")
    original_scores = score(original_predictions)
    export_scores = score(reference_p)
    sanity = [dict(horizon_min=h * 5, original_MAE=original_scores[h]["MAE"], export_MAE=export_scores[h]["MAE"],
        MAE_abs_difference=abs(original_scores[h]["MAE"] - export_scores[h]["MAE"]),
        RMSE_abs_difference=abs(original_scores[h]["RMSE"] - export_scores[h]["RMSE"])) for h in HORIZONS]
    # Predeclared tolerance permits FP32 changes from GPU architecture and batch size.
    require(all(row["MAE_abs_difference"] <= 1e-4 for row in sanity), "Original subset MAE does not match export (atol=1e-4): stop")
    prediction_difference = float(np.max(np.abs(original_predictions - reference_p)))
    print("SANITY PASS:", json.dumps(sanity), "max prediction diff", prediction_difference, flush=True)
    conditions = ["original", "long_train_calendar_profile", "long_shuffled_windows", "short_train_node_mean", "short_flow_gaussian_noise"]
    rows, condition_times = [], {}
    for condition in conditions:
        start = time.perf_counter()
        scores = original_scores if condition == "original" else score(infer(condition))
        condition_times[condition] = time.perf_counter() - start
        for h in HORIZONS:
            row = dict(analysis_type=ANALYSIS, condition=condition, n_windows=256, horizon_min=h * 5, **scores[h])
            for metric in ("MAE", "RMSE"):
                change = scores[h][metric] - original_scores[h][metric]
                row[metric + "_delta_vs_original"] = change
                row[metric + "_change_percent"] = change / original_scores[h][metric] * 100
            rows.append(row)
        print("CONDITION DONE:", condition, scores, flush=True)
    perturbation = pd.DataFrame(rows)
    perturbation.to_csv(out / "perturbation_pems08.csv", index=False)

    # Self-check the same IG routine on a linear function with known attributions.
    test_x = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3) / 12
    test_baseline = torch.zeros_like(test_x)
    weights = torch.ones_like(test_x)
    weights[..., 2] = 0
    def linear_outputs(z):
        scalar = (z * weights).sum()
        return torch.stack((scalar, 2 * scalar, 3 * scalar))
    test_ig, _ = integrated_gradients(linear_outputs, test_x, test_baseline, 20)
    expected_ig = torch.stack([test_x * weights * multiplier for multiplier in (1, 2, 3)])
    require(torch.allclose(test_ig, expected_ig, atol=1e-6, rtol=0), "IG linear self-check failed")
    del test_ig, test_x, test_baseline, weights
    abs_accumulator = np.zeros((3, 12, 2), dtype=np.float64)
    completeness = []
    ig_started = time.perf_counter()
    ig_peak = 0
    for number, pos in enumerate(ig_ids):
        _, short, long = dataset[int(pos)]
        short, long = short.unsqueeze(0).to(device), long.unsqueeze(0).to(device)
        baseline = torch.from_numpy(mean_node).view(1, 1, 170, 3).expand(1, 12, -1, -1).contiguous().to(device)
        def scalar_outputs(interpolated):
            output = model(interpolated, long)
            return torch.stack([output[:, :, h - 1].mean() * raw_scale + float(minimum[0, 0, 0]) for h in HORIZONS])
        torch.cuda.reset_peak_memory_stats()
        with torch.enable_grad():
            attribution, endpoints = integrated_gradients(scalar_outputs, short, baseline, 20)
        torch.cuda.synchronize()
        ig_peak = max(ig_peak, torch.cuda.max_memory_allocated())
        attribution = attribution.cpu().numpy()[:, 0]  # horizon, time, node, channel
        require(np.isfinite(attribution).all() and np.all(attribution[..., 2] == 0), "Invalid or unexpected third-channel IG")
        abs_accumulator += np.abs(attribution[..., :2]).mean(axis=2)
        for j, h in enumerate(HORIZONS):
            delta = float(endpoints[1][j] - endpoints[0][j])
            integral_sum = float(attribution[j].astype(np.float64).sum())
            residual = integral_sum - delta
            completeness.append(dict(analysis_type=ANALYSIS, test_position=int(pos), horizon_min=h * 5,
                scalar_original=float(endpoints[1][j]), scalar_baseline=float(endpoints[0][j]),
                output_difference=delta, signed_IG_sum=integral_sum, completeness_residual=residual,
                abs_residual=abs(residual), relative_abs_residual=abs(residual) / max(abs(delta), 1e-6)))
        del short, long, baseline, attribution
        if (number + 1) % 4 == 0:
            print("IG", number + 1, "/32 windows; 20 intervals each; batch=1", flush=True)
    mean_abs = abs_accumulator / 32
    ig_rows = []
    for j, h in enumerate(HORIZONS):
        denominator = mean_abs[j].sum()
        for step in range(12):
            for channel in range(2):
                value = mean_abs[j, step, channel]
                ig_rows.append(dict(analysis_type=ANALYSIS, horizon_min=h * 5, aggregation="time_channel",
                    time_step=step + 1, lag_minutes=(11 - step) * 5, channel=channel,
                    mean_abs_attribution=value, share_percent=100 * value / denominator))
            value = mean_abs[j, step].sum()
            ig_rows.append(dict(analysis_type=ANALYSIS, horizon_min=h * 5, aggregation="time",
                time_step=step + 1, lag_minutes=(11 - step) * 5, channel="both",
                mean_abs_attribution=value, share_percent=100 * value / denominator))
        for channel in range(2):
            value = mean_abs[j, :, channel].sum()
            ig_rows.append(dict(analysis_type=ANALYSIS, horizon_min=h * 5, aggregation="channel",
                time_step="all", lag_minutes="all", channel=channel,
                mean_abs_attribution=value, share_percent=100 * value / denominator))
    ig = pd.DataFrame(ig_rows)
    ig.to_csv(out / "ig_short_history.csv", index=False)
    completeness = pd.DataFrame(completeness)
    completeness.to_csv(extra / "ig_completeness.csv", index=False)
    require(all(p.grad is None for p in model.parameters()), "Parameter gradients unexpectedly accumulated")
    require(model_hash() == state_before, "Model state changed")
    require(all(sha256(root / p) == digest for p, digest in before.items()), "Protected source/input changed")
    require(sha256(checkpoint) == expected[0], "Checkpoint changed")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    source = "Độ nhạy suy luận, KHÔNG phải ablation huấn luyện lại; run PEMS08 / 0001, best epoch 68."
    labels = ["Gốc", "Dài: hồ sơ train", "Dài: xáo cửa sổ", "Ngắn: trung bình train", "Ngắn: nhiễu Flow"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ("MAE", "RMSE")):
        for condition, label in zip(conditions, labels):
            part = perturbation[perturbation.condition == condition]
            ax.plot(part.horizon_min, part[metric], marker="o", label=label)
        ax.set(xticks=[15, 30, 60], xlabel="Khoảng dự báo (phút)", ylabel=metric + " (xe/5 phút/node)")
    fig.suptitle("PEMS08 — độ nhạy khi thay đổi đầu vào suy luận")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(.5, .06), ncol=3, fontsize=9)
    fig.text(.01, .01, source, fontsize=8)
    fig.tight_layout(rect=[0, .19, 1, .94])
    fig.savefig(extra / "perturbation_metrics.png", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for j, h in enumerate(HORIZONS):
        axes[j, 0].bar(np.arange(12), mean_abs[j].sum(axis=1), color="#2378a8")
        axes[j, 0].set(xticks=np.arange(12), xticklabels=np.arange(-55, 1, 5),
                       title=f"{h * 5} phút — theo bước lịch sử", xlabel="Độ trễ so với quan sát cuối (phút)", ylabel="Trung bình |IG|, cộng hai kênh")
        axes[j, 1].bar(["Flow chuẩn hóa", "Mã giờ"], mean_abs[j].sum(axis=0), color=["#2378a8", "#d18324"])
        axes[j, 1].set(title=f"{h * 5} phút — theo kênh", ylabel="Trung bình |IG|, cộng 12 bước")
    fig.suptitle("IG lịch sử ngắn — độ nhạy suy luận có điều kiện trên lịch sử dài gốc")
    fig.text(.01, .01, source, fontsize=8)
    fig.tight_layout(rect=[0, .05, 1, .95])
    fig.savefig(extra / "ig_short_history.png", dpi=200)
    plt.close(fig)

    protocol = dict(analysis_type=ANALYSIS, status="success", run_id=reference["run_id"], attempt=1, best_epoch=68,
        checkpoint_sha256=expected[0], strict_load=True, eval=True, no_training=True,
        versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
        gpu=torch.cuda.get_device_name(0), inference_batch=args.batch_size, ig_batch=1,
        flags=dict(no_grad_perturbations=True, enable_grad_IG=True, FP32=True, TF32=False,
                   softmax_compat=compat_enabled, actual_split_calls=split_count[0]),
        selection=dict(generator="numpy.default_rng", seed=0, n_test=3566, n_selected=256, n_IG=32,
                       rule="draw 256 without replacement; evaluate sorted IDs; IG is first 32 unsorted draws, then sorted",
                       file="selected_windows.csv", selected_positions=selected.tolist(), IG_positions=ig_ids.tolist()),
        train_only=dict(source="unique label timestamps from actual train_index", count=len(train_ids),
                        first=int(train_ids[0]), last=int(train_ids[-1]), shared_validation_labels=overlap,
                        test_labels_used=0, groups="weekday Monday=0 and 5-minute slot; origin 2016-07-01 Friday",
                        missing_groups=int((counts == 0).sum()), fallback="per-node train mean",
                        note="Shared 11 train/validation timestamps retained as requested repo protocol; not target-disjoint"),
        normalization=dict(input="MTSDataset float32 normalized channels", scaler="repo max_min_scale on CPU for metric/export",
                           min=float(minimum[0, 0, 0]), max=float(maximum[0, 0, 0]), clipping=False),
        metrics=dict(MAE="all finite targets, zeros included", RMSE="target != 0; global sums then square root", horizons=[3, 6, 12]),
        sanity=dict(MAE_atol_predeclared=1e-4, rtol=0, passed=True, comparisons=sanity,
                    max_abs_prediction_difference=prediction_difference,
                    note="Comparison against SAME selected export windows; original batch32/A6000 vs batch2-or1/RTX3050 allows FP32 rounding differences, not bitwise identity"),
        conditions=dict(long_train_calendar_profile="replace only long Flow with normalized train weekday/slot mean; ignored long time codes retained",
                        long_shuffled_windows="full long input from a deranged donor in selected test windows, seed1; recipient short and target unchanged",
                        short_train_node_mean="all short channels replaced by per-node normalized train means, broadcast across 12 steps; third channel unused by model",
                        short_flow_gaussian_noise=dict(channel=0, std_normalized=.01, std_original_flow=.01 * raw_scale, seed=2, time_codes_preserved=True, clipping=False)),
        IG=dict(steps=20, alpha_points=21, quadrature="trapezoidal endpoints half-weighted", baseline="normalized per-node train mean for all short channels",
                target_scalar="mean forecast Flow over 170 nodes, separately at horizons 3/6/12, original units",
                long_history="fixed original at every alpha; it includes the original recent 12 steps",
                aggregation="abs before averaging across 32 windows and 170 nodes; time sum over used channels, channel sum over 12 steps",
                ignored_third_channel_zero_verified=True, linear_self_check_passed=True,
                peak_allocated_bytes=ig_peak, elapsed_seconds=time.perf_counter() - ig_started,
                completeness_abs_mean=float(completeness.abs_residual.mean()), completeness_abs_max=float(completeness.abs_residual.max()),
                note="20-step approximation, not assumed exact; per-window/horizon residuals in ig_completeness.csv; input gradients only, model parameters unchanged"),
        protected_files_unchanged=True, protected_file_sha256=before, model_state_unchanged=True,
        condition_seconds=condition_times, elapsed_seconds=time.perf_counter() - tick, script_sha256=sha256(Path(__file__)),
        limitations=["Sensitivity is not causal contribution or retrained ablation; no new model was fitted.",
                     "Replacing means/time codes or shuffling can create out-of-distribution and inconsistent long/short inputs.",
                     "Shuffled donors can be later test windows; this is a diagnostic intervention, NOT a valid forecasting protocol.",
                     "Long input contains recent observations too; keeping it fixed makes short IG conditional, not independent attribution.",
                     "Results describe only one selected subset, noise realization, checkpoint and chosen train-mean baseline.",
                     "IG explains a node-averaged scalar; cross-node gradient cancellation can occur before taking absolute values."])
    (extra / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    notes = ["# Độ nhạy đầu vào khi suy luận — KHÔNG phải ablation huấn luyện lại", "",
        "Run PEMS08-repo96-author-pretrained / 0001, best epoch 68; strict-load, eval. Không cập nhật tham số.",
        "256 cửa sổ lấy ngẫu nhiên không hoàn lại bằng seed 0; danh sách và donor xáo trộn nằm trong selected_windows.csv. IG dùng 32 cửa sổ con.",
        "Hồ sơ dài và baseline ngắn chỉ dùng 10.711 timestamp nhãn train duy nhất, gồm 11 timestamp chung validation theo repo; không dùng nhãn test. Nhóm lịch là thứ và khoảng 5 phút từ 2016-07-01 (thứ Sáu).", "",
        "## Sanity gốc so với export cùng index", ""]
    notes += [f"- {r['horizon_min']} phút: MAE suy luận {r['original_MAE']:.8f}; export {r['export_MAE']:.8f}; chênh lệch tuyệt đối {r['MAE_abs_difference']:.3g}." for r in sanity]
    notes += [f"Ngưỡng MAE khai báo trước 1e-4; khác GPU/batch FP32 không yêu cầu khớp từng bit. Max chênh lệch prediction {prediction_difference:.6g}.", "",
        "## Quan sát", ""]
    for condition in conditions:
        row = perturbation[(perturbation.condition == condition) & (perturbation.horizon_min == 60)].iloc[0]
        notes.append(f"- {condition}, 60 phút: MAE {row.MAE:.4f}, RMSE {row.RMSE:.4f}; MAE thay đổi {row.MAE_change_percent:+.2f}% so với gốc cùng subset.")
    notes += ["", "## Cách hiểu IG và giới hạn", "",
        "IG tự cài bằng 20 khoảng nội suy (21 điểm, quy tắc hình thang), batch 1. Đầu ra giải thích là Flow dự báo trung bình trên 170 node tại từng horizon, không phải một node riêng. Baseline là trung bình chuẩn hóa theo node từ train; lịch sử dài giữ nguyên.",
        "CSV lấy trị tuyệt đối attribution trước khi trung bình qua cửa sổ và node. Theo thời gian cộng hai kênh, theo kênh cộng 12 bước. Tỷ trọng |IG| chỉ là cách tổng hợp độ nhạy, không phải tỷ lệ đóng góp nhân quả.",
        f"Sai số completeness tuyệt đối trung bình {protocol['IG']['completeness_abs_mean']:.6g}, lớn nhất {protocol['IG']['completeness_abs_max']:.6g} đơn vị Flow dự báo trung bình. Xem ig_completeness.csv; 20 bước chỉ là xấp xỉ, không khẳng định đã hội tụ tích phân.",
        "Nhiễu chỉ cộng vào Flow ngắn: sigma=0,01 miền chuẩn hóa, tương đương 5,385 xe/5 phút; giữ mã thời gian, không clip. Mean ngắn thay cả hai kênh được model sử dụng bằng trung bình train nên không còn lịch thời gian thực.",
        "Mô hình được huấn luyện với đầu vào thật. Thay trung bình, xáo trộn hoặc thay clock có thể tạo đầu vào ngoài phân phối. Long và short có phần lịch sử chồng nhau, nên không phải các nguồn độc lập. Donor xáo trộn có thể ở tương lai so với cửa sổ nhận; không coi đây là cách dự báo hợp lệ.",
        "Độ nhạy KHÔNG bằng đóng góp nhân quả; không phải ablation huấn luyện lại. Một subset, một nhiễu và một baseline IG không cho phép kết luận tổng quát hoặc so sánh nhân quả giữa các nhánh.", ""]
    (extra / "NOTES.md").write_text("\n".join(notes), encoding="utf-8")
    print("SUCCESS", ANALYSIS, "— CSV, PNG, NOTES and protocol saved; no model updates.", flush=True)


if __name__ == "__main__":
    main()
