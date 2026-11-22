"""CPU-only Persistence/HA on the unchanged PEMS08 forecast test indices.

Run from either Windows or WSL. No training, downloads, CUDA or model changes.
Only the two small result files in results/extension/ are written.
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import platform
import sys
import time

import numpy as np


HORIZONS = (3, 6, 12)
START = datetime(2016, 7, 1)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pickle(path):
    # These are the project's trusted existing dataset files.
    with path.open("rb") as stream:
        return pickle.load(stream, encoding="latin1")


def timestamp(position):
    return (START + timedelta(minutes=5 * int(position))).isoformat()


def target_times(index):
    """Union of label timestamps, not repeated occurrences in sliding windows."""
    return np.unique((index[:, 1, None] + np.arange(12)[None, :]).ravel())


def sufficient_stats(prediction, target):
    """Finite PEMS08 protocol, float64 sums; never average batch RMSEs."""
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    require(p.shape == y.shape, "Prediction/target shape mismatch")
    require(np.isfinite(p).all() and np.isfinite(y).all(), "Nonfinite data: stop")
    rmse_mask = y != 0
    mape_mask = y >= 0.01
    return np.array([
        np.abs(p - y).sum(), y.size,
        np.square(p[rmse_mask] - y[rmse_mask]).sum(), rmse_mask.sum(),
        (np.abs(p[mape_mask] - y[mape_mask]) / y[mape_mask]).sum(), mape_mask.sum(),
    ], dtype=np.float64)


def finish_stats(values):
    require(values[1] > 0 and values[3] > 0 and values[5] > 0, "Empty metric mask")
    return dict(MAE=float(values[0] / values[1]),
                RMSE=float(np.sqrt(values[2] / values[3])),
                MAPE_percent=float(100 * values[4] / values[5]))


def metric_self_check(root):
    """Cross-check masks/threshold against the current author metric functions."""
    import torch
    spec = importlib.util.spec_from_file_location("author_metrics", root / "utils/metrics.py")
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    p = np.array([2.0, 9.0, 6.0, 10.0])
    y = np.array([0.0, 0.005, 3.0, 5.0])
    ours = finish_stats(sufficient_stats(p, y))
    pt, yt = torch.from_numpy(p), torch.from_numpy(y)
    expected = dict(MAE=metrics.masked_mae(pt, yt).item(),
                    RMSE=metrics.masked_rmse(pt, yt, 0.0).item(),
                    MAPE_percent=metrics.masked_mape(pt, yt, 0.0).item() * 100)
    for key in expected:
        require(np.isclose(ours[key], expected[key], atol=1e-6, rtol=2e-7),
                "Metric self-check failed: " + key)
    return dict(passed=True, predictions=p.tolist(), targets=y.tolist(),
                numpy_float64=ours, author_torch=expected,
                torch_version=torch.__version__, device="cpu",
                note="Torch masks normalize in float32; global float64 sums can round differently.")


def fit_ha(flow, train_ids, group_ids):
    """Fit only the explicit training-label timestamp union; zeros are retained."""
    selected = flow[train_ids].astype(np.float64)
    counts = np.bincount(group_ids[train_ids], minlength=7 * 288)
    sums = np.zeros((7 * 288, flow.shape[1]), dtype=np.float64)
    np.add.at(sums, group_ids[train_ids], selected)
    # Missing (weekday,slot) -> that node's mean over the same training IDs only.
    table = np.broadcast_to(selected.mean(axis=0), sums.shape).copy()
    present = counts > 0
    table[present] = sums[present] / counts[present, None]
    return table, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="CPU evaluation windows per batch; no CUDA is used")
    args = parser.parse_args()
    require(args.batch_size > 0, "batch-size must be positive")
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "datasets/PEMS08"
    export_dir = root / "artifacts/report-inputs/PEMS08"
    out = root / "results/extension"
    data_paths = [data_dir / name for name in (
        "data.pkl", "min.pkl", "max.pkl", "train_index.pkl", "valid_index.pkl", "test_index.pkl")]
    protected = data_paths + [root / name for name in (
        "README.md", "utils/metrics.py", "utils/datasets.py", "utils/load_data.py",
        "results/experiment_metrics.csv", "results/experiment_manifest.json")]
    protected += sorted((root / "configs").rglob("*.py"))
    protected += sorted((root / "models").glob("*.py"))
    before_hashes = {p.relative_to(root).as_posix(): sha256(p) for p in protected}
    self_check = metric_self_check(root)

    data = np.asarray(read_pickle(data_dir / "data.pkl"))
    require(data.shape == (17856, 170, 3), "Unexpected PEMS08 data shape")
    require(np.isfinite(data).all(), "Nonfinite dataset: stop without imputing")
    minimum = np.float32(np.asarray(read_pickle(data_dir / "min.pkl"))[0, 0, 0])
    maximum = np.float32(np.asarray(read_pickle(data_dir / "max.pkl"))[0, 0, 0])
    # Match MTSDataset.float() and each float32 inverse-scaler operation.
    normalized = data[..., 0].astype(np.float32)
    flow = (normalized + np.float32(1)) / np.float32(2)
    flow = np.float32(1) * flow * (maximum - minimum) + minimum
    require(np.isfinite(flow).all(), "Invalid inverse scaler result")
    indices = {s: np.asarray(read_pickle(data_dir / (s + "_index.pkl")), dtype=np.int64)
               for s in ("train", "valid", "test")}
    require([len(indices[s]) for s in ("train", "valid", "test")] == [10700, 3567, 3566],
            "Unexpected split sizes")
    for split, idx in indices.items():
        require(idx.shape[1:] == (3,), "Invalid index shape: " + split)
        require(np.all(idx[:, 1] - idx[:, 0] == 12) and np.all(idx[:, 2] - idx[:, 1] == 12),
                "Unexpected window length: " + split)
        require(np.all(idx[:, 0] >= 0) and np.all(idx[:, 2] <= len(flow)), "Out-of-bounds index")
        require(np.all(np.diff(idx[:, 1]) == 1), "Noncontinuous forecast starts")
    test_index = indices["test"]
    ids = {s: target_times(idx) for s, idx in indices.items()}
    train_ids = ids["train"]
    train_mask = np.zeros(len(flow), dtype=bool)
    train_mask[train_ids] = True
    val_only_ids = np.setdiff1d(ids["valid"], train_ids)
    require(not np.intersect1d(train_ids, val_only_ids).size, "HA contains validation-only IDs")
    require(not np.intersect1d(train_ids, ids["test"]).size, "HA contains test IDs")
    overlaps = {a + "/" + b: int(np.intersect1d(ids[a], ids[b]).size)
                for a, b in (("train", "valid"), ("valid", "test"), ("train", "test"))}
    require(overlaps == {"train/valid": 11, "valid/test": 11, "train/test": 0}, "Unexpected overlaps")

    steps = np.arange(len(flow))
    weekdays = (START.weekday() + steps // 288) % 7  # Monday=0; 2016-07-01=Friday=4.
    slots = steps % 288
    groups = weekdays * 288 + slots
    require(START.weekday() == 4, "Incorrect calendar origin")
    require(np.allclose(data[..., 1], slots[:, None] / 288, atol=1e-10, rtol=0), "Time code mismatch")
    require(np.array_equal(data[..., 2], np.broadcast_to((steps // 288 % 7)[:, None], flow.shape)),
            "Relative day code mismatch")

    # Verify every existing export hash and target/index, then recompute LSTTN as a control.
    export_started = time.perf_counter()
    manifest = json.loads((export_dir / "export_hashes.json").read_text(encoding="utf-8"))
    entries = {item["file"]: item["sha256"] for item in manifest}
    batch_files = sorted(export_dir.glob("batch_*.npz"))
    require(len(batch_files) == 112 and len(entries) == len(manifest) == 113, "Incomplete export")
    require(set(entries) == {p.name for p in batch_files} | {"metadata.json"}, "Unexpected export manifest")
    for name, expected in entries.items():
        require(Path(name).name == name and sha256(export_dir / name) == expected, "Export hash mismatch: " + name)
    metadata = json.loads((export_dir / "metadata.json").read_text(encoding="utf-8"))
    require(metadata["dataset"] == "PEMS08" and metadata["samples"] == len(test_index), "Export metadata mismatch")
    lsttn_stats = {h: np.zeros(6) for h in HORIZONS}
    offset, target_error = 0, 0.0
    for file in batch_files:
        with np.load(file, allow_pickle=False) as exported:
            idx = exported["index"]
            require(np.array_equal(idx, test_index[offset:offset + len(idx)]), "Export index mismatch")
            y = flow[idx[:, 1, None] + np.arange(12)]
            error = float(np.max(np.abs(y - exported["target"])))
            target_error = max(target_error, error)
            require(np.array_equal(y, exported["target"]), "Float32 target is not exactly equal to export")
            require(np.array_equal(exported["mask"], np.isfinite(y) & (y != 0)), "Export mask mismatch")
            for h in HORIZONS:
                lsttn_stats[h] += sufficient_stats(exported["prediction"][:, h - 1], y[:, h - 1])
            offset += len(idx)
    require(offset == len(test_index), "Incomplete test coverage")
    with (root / "results/experiment_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        recorded = {int(r["minutes"]): r for r in csv.DictReader(stream) if r["dataset"] == "PEMS08"}
    lsttn_comparison = []
    for h in HORIZONS:
        row = finish_stats(lsttn_stats[h])
        require(all(np.isclose(row[k], float(recorded[h * 5][k]), atol=1e-8, rtol=0) for k in row),
                "Recomputed LSTTN metrics differ from reference")
        lsttn_comparison.append(dict(method="LSTTN_reference", horizon_min=h * 5, **row))
    export_seconds = time.perf_counter() - export_started

    fit_started = time.perf_counter()
    ha_table, group_counts = fit_ha(flow, train_ids, groups)
    fit_seconds = time.perf_counter() - fit_started
    # Causal membership check: modifying every timestamp outside train must not alter HA.
    altered = flow.copy()
    altered[~train_mask] = np.float32(-999999)
    altered_table, altered_counts = fit_ha(altered, train_ids, groups)
    require(np.array_equal(ha_table, altered_table) and np.array_equal(group_counts, altered_counts),
            "HA depends on timestamps outside train")
    del altered, altered_table

    stats = {m: {h: np.zeros(6) for h in HORIZONS} for m in ("Persistence", "HA")}
    timing = {m: 0.0 for m in stats}
    for start in range(0, len(test_index), args.batch_size):
        idx = test_index[start:start + args.batch_size]
        for method in stats:
            tick = time.perf_counter()
            for h in HORIZONS:
                target_positions = idx[:, 1] + h - 1
                y = flow[target_positions]
                p = flow[idx[:, 1] - 1] if method == "Persistence" else ha_table[groups[target_positions]]
                stats[method][h] += sufficient_stats(p, y)
            timing[method] += time.perf_counter() - tick
    rows = [dict(method=m, horizon_min=h * 5, **finish_stats(stats[m][h])) for m in stats for h in HORIZONS]

    manual = []
    for sample, node in ((0, 0), (1, 85), (len(test_index) - 1, 169)):
        idx = test_index[sample]
        last = int(idx[1] - 1)
        prediction = np.repeat(float(flow[last, node]), 12)
        require(np.all(prediction == flow[int(idx[0]):int(idx[1]), node][-1]), "Persistence check failed")
        position = int(idx[1] + 2)  # Horizon 3 for checking the HA group manually.
        calendar = START + timedelta(minutes=5 * position)
        require(groups[position] == calendar.weekday() * 288 + calendar.hour * 12 + calendar.minute // 5,
                "Calendar grouping mismatch")
        members = train_ids[groups[train_ids] == groups[position]]
        require(len(members) > 0, "Manual sample group is missing")
        direct_mean = float(flow[members, node].astype(np.float64).mean())
        require(np.isclose(direct_mean, ha_table[groups[position], node], atol=1e-12, rtol=0), "HA mean mismatch")
        manual.append(dict(test_window=sample, node=node, last_input_position=last,
                           last_input_time=timestamp(last), last_flow=float(flow[last, node]),
                           persistence_steps_3_6_12=prediction[[2, 5, 11]].tolist(),
                           ha_horizon3_group=dict(weekday_monday_zero=calendar.weekday(),
                               five_minute_slot=calendar.hour * 12 + calendar.minute // 5),
                           ha_training_positions=members.tolist(),
                           ha_training_values=flow[members, node].astype(float).tolist(),
                           ha_horizon3=float(ha_table[groups[position], node])))

    for relative, original_hash in before_hashes.items():
        require(sha256(root / relative) == original_hash, "Protected input/source changed: " + relative)
    protocol = dict(
        status="success", created_utc=datetime.now(timezone.utc).isoformat(),
        script="scripts/extension/run_baselines_pems08.py", script_sha256=sha256(Path(__file__)),
        versions=dict(python=platform.python_version(), numpy=np.__version__, torch=self_check["torch_version"]),
        platform=platform.platform(), device="cpu", cuda_used=False, batch_size=args.batch_size,
        dataset="PEMS08", target="Flow", units="vehicles per 5 minutes per node", nodes=170,
        split_window_counts={s: len(idx) for s, idx in indices.items()},
        horizons_steps=list(HORIZONS), horizons_minutes=[h * 5 for h in HORIZONS],
        scaler=dict(min=float(minimum), max=float(maximum),
                    formula="(normalized_float32 + 1) / 2 * (max - min) + min; no clipping",
                    reason="Matches MTSDataset.float() and existing float32 export targets exactly"),
        masks=dict(MAE="all finite targets, including zero", RMSE="finite targets != 0",
                   MAPE="finite targets >= 0.01; ratio multiplied by 100", zero_imputation=False,
                   nonfinite_policy="stop; no silent omission or filling",
                   accumulation="global float64 sums/counts, not mean of batch metrics"),
        metric_self_check=self_check,
        persistence=dict(rule="repeat Flow at forecast_start - 1 for all 12 future steps"),
        historical_average=dict(
            group="calendar weekday Monday=0 plus 5-minute slot 0..287; separate mean per node",
            calendar_start=START.isoformat(), frequency_minutes=5, timezone="naive local benchmark clock",
            timestamp_origin="reconstructed metadata, not embedded timestamps",
            relative_day_channel_used_for_grouping=False,
            fit_rule="union of actual train-index label timestamps, each timestamp counted once; includes zeros",
            unique_training_timestamps=int(len(train_ids)), training_first=int(train_ids[0]), training_last=int(train_ids[-1]),
            training_first_time=timestamp(train_ids[0]), training_last_time=timestamp(train_ids[-1]),
            available_groups=int((group_counts > 0).sum()), total_groups=2016,
            group_observation_count_min=int(group_counts.min()), group_observation_count_max=int(group_counts.max()),
            missing_groups=int((group_counts == 0).sum()),
            fallback="node's global mean on the same train-label timestamp union only",
            fallback_test_uses_per_horizon={str(h): int((group_counts[groups[test_index[:, 1] + h - 1]] == 0).sum()) for h in HORIZONS},
            outside_train_perturbation_test_passed=True,
            validation_only_timestamps_used=0, test_timestamps_used=0,
            boundary_overlap=overlaps,
            caveat="11 included train-label timestamps also belong to validation labels. No validation-only or test timestamps used. This is repo protocol, not target-disjoint."),
        export_verification=dict(path="artifacts/report-inputs/PEMS08", sha_verified_files=len(entries),
            batch_files=len(batch_files), samples=offset, target_max_abs_error=target_error,
            target_exact_match=True, masks_match=True, index_exact_match=True, lsttn_metrics_match=True),
        metric_counts={str(h): dict(MAE=int(stats["HA"][h][1]), RMSE=int(stats["HA"][h][3]), MAPE=int(stats["HA"][h][5])) for h in HORIZONS},
        manual_checks=manual, input_and_source_sha256=before_hashes, protected_files_unchanged=True,
        timings_seconds=dict(ha_fit=fit_seconds, evaluation_by_method=timing,
                             export_verification=export_seconds, total_before_output_write=time.perf_counter() - started),
        side_by_side=rows + lsttn_comparison,
        interpretation="Descriptive side-by-side only. No causal comparison or statistical significance claim.",
        original_result_files_overwritten=False,
    )
    # Write final outputs only after every verification succeeds.
    out.mkdir(parents=True, exist_ok=True)
    with (out / "baselines_pems08.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "horizon_min", "MAE", "RMSE", "MAPE_percent"])
        writer.writeheader()
        writer.writerows(rows)
    (out / "baselines_pems08_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("MASK: MAE includes zero; RMSE excludes zero; MAPE target>=0.01, percent.")
    print("HA train timestamps:", len(train_ids), "from", timestamp(train_ids[0]), "to", timestamp(train_ids[-1]))
    print("HA shared train/validation labels:", overlaps["train/valid"], "; validation-only/test used: 0/0")
    print("HA missing groups:", int((group_counts == 0).sum()))
    for item in manual:
        print("MANUAL", json.dumps(item, ensure_ascii=False))
    print("method,horizon_min,MAE,RMSE,MAPE_percent")
    for row in rows + lsttn_comparison:
        print("{method},{horizon_min},{MAE:.8f},{RMSE:.8f},{MAPE_percent:.8f}".format(**row))
    print("SUCCESS: 3566 test windows; protected files unchanged; outputs:", out)


if __name__ == "__main__":
    main()
