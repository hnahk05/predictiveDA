"""CPU analysis of an existing PEMS08 export; no model inference or training."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HORIZONS = (3, 6, 12)
ORIGIN = datetime(2016, 7, 1)
SOURCE = "Nguồn: export run PEMS08-repo96-author-pretrained / 0001; best epoch 68."


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def summary(p, y):
    # All selected inputs are required finite before this function is called.
    error = p - y
    valid = y != 0
    mape_valid = y >= 0.01
    return dict(n_MAE=int(y.size), n_RMSE=int(valid.sum()), n_MAPE=int(mape_valid.sum()),
                mean_target_flow=float(y.mean()), MAE=float(np.abs(error).mean()),
                RMSE=float(np.sqrt(np.square(error[valid]).mean())) if valid.any() else np.nan,
                MAPE_percent=float(100 * (np.abs(error[mape_valid]) / y[mape_valid]).mean()) if mape_valid.any() else np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()
    check(args.bootstrap_reps >= 100, "Use at least 100 bootstrap replicates")
    tick = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    inputs = root / "artifacts/report-inputs/PEMS08"
    output = root / "results/extension/error_analysis"
    protected = [root / x for x in ("README.md", ".gitignore", "utils/metrics.py", "utils/prediction_export.py",
                  "results/experiment_metrics.csv", "results/experiment_manifest.json")]
    protected += sorted((root / "configs").rglob("*.py")) + sorted((root / "models").glob("*.py"))
    original_hashes = {p.relative_to(root).as_posix(): sha256(p) for p in protected}

    hashes = json.loads((inputs / "export_hashes.json").read_text(encoding="utf-8"))
    batch_files = sorted(inputs.glob("batch_*.npz"))
    entries = {x["file"]: x["sha256"] for x in hashes}
    check(len(batch_files) == 112 and len(hashes) == len(entries) == 113, "Incomplete export")
    check(set(entries) == {p.name for p in batch_files} | {"metadata.json"}, "Unexpected manifest entries")
    for name, expected in entries.items():
        check(Path(name).name == name and sha256(inputs / name) == expected, "Hash mismatch: " + name)
    metadata = json.loads((inputs / "metadata.json").read_text(encoding="utf-8"))
    check(metadata["dataset"] == "PEMS08" and metadata["units"] == "original", "Unexpected target units/dataset")
    pp, yy, mm, ii = [], [], [], []
    for file in batch_files:
        with np.load(file, allow_pickle=False) as batch:
            pp.append(batch["prediction"][:, [h - 1 for h in HORIZONS], :])
            yy.append(batch["target"][:, [h - 1 for h in HORIZONS], :])
            mm.append(batch["mask"][:, [h - 1 for h in HORIZONS], :])
            ii.append(batch["index"])
    predictions, targets = np.concatenate(pp).astype("float64"), np.concatenate(yy).astype("float64")
    mask, index = np.concatenate(mm), np.concatenate(ii)
    del pp, yy, mm, ii
    check(predictions.shape == targets.shape == mask.shape == (3566, 3, 170), "Unexpected export shape")
    check(np.isfinite(predictions).all() and np.isfinite(targets).all(), "Nonfinite values: stop")
    check(np.array_equal(mask, targets != 0), "Export null mask differs from target != 0")
    import pickle
    with (root / "datasets/PEMS08/test_index.pkl").open("rb") as stream:
        test_index = np.asarray(pickle.load(stream, encoding="latin1"))
    check(np.array_equal(index, test_index), "Export does not match repo test index")
    check(np.all(np.diff(index[:, 1]) == 1), "Non-continuous test forecast starts")
    check(np.all(index[:, 2] - index[:, 1] == 12), "Unexpected target length")

    reference = pd.read_csv(root / "results/experiment_metrics.csv")
    reference = reference[reference["dataset"] == "PEMS08"].set_index("minutes")
    verified = []
    for j, h in enumerate(HORIZONS):
        values = summary(predictions[:, j], targets[:, j])
        for metric in ("MAE", "RMSE", "MAPE_percent"):
            check(np.isclose(values[metric], reference.loc[h * 5, metric], atol=1e-8, rtol=0),
                  "Metric verification failed: " + metric)
        verified.append(dict(horizon_min=h * 5, **values,
                             max_metric_abs_difference=max(abs(values[k] - reference.loc[h * 5, k])
                                                            for k in ("MAE", "RMSE", "MAPE_percent"))))
    print("PASS: all global metrics match experiment_metrics.csv (atol=1e-8, rtol=0).", flush=True)
    # No subgroup analysis or outputs before the global gate above has passed.
    verification_seconds = time.perf_counter() - tick
    output.mkdir(parents=True, exist_ok=True)
    tables, figures = {}, []

    def save_table(name, rows):
        frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        frame.to_csv(output / (name + ".csv"), index=False, encoding="utf-8-sig", float_format="%.12g")
        tables[name] = frame
        return frame

    def save_figure(fig, name, title, note=""):
        fig.suptitle(title, fontsize=14)
        fig.text(.01, .025, SOURCE, fontsize=8)
        if note:
            fig.text(.01, .005, note, fontsize=8)
        fig.tight_layout(rect=[0, .07, 1, .94])
        fig.savefig(output / name, dpi=200, facecolor="white")
        plt.close(fig)
        figures.append(name)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    save_table("overall_metrics_verified", verified)
    node_rows, correlations, extremes = [], [], []
    hours_rows, daytype_rows, quantile_rows, daily_rows = [], [], [], []
    quantile_metadata, day_metadata = {}, {}
    daily_sums, daily_counts = [], []
    target_datetimes = [pd.DatetimeIndex([ORIGIN + timedelta(minutes=5 * int(t))
                        for t in index[:, 1] + h - 1]) for h in HORIZONS]
    calendar_dates = np.unique(np.concatenate([t.strftime("%Y-%m-%d").to_numpy() for t in target_datetimes]))

    for j, h in enumerate(HORIZONS):
        p, y = predictions[:, j], targets[:, j]
        dates = target_datetimes[j]
        node_frame = pd.DataFrame([dict(horizon_min=h * 5, node=n, **summary(p[:, n], y[:, n])) for n in range(170)])
        node_rows.extend(node_frame.to_dict("records"))
        pearson = node_frame["MAE"].corr(node_frame["mean_target_flow"], method="pearson")
        spearman = node_frame["MAE"].rank().corr(node_frame["mean_target_flow"].rank())
        correlations.append(dict(horizon_min=h * 5, n_nodes=170, pearson_r=pearson, spearman_rho=spearman))
        for label, ascending in (("highest_MAE", False), ("lowest_MAE", True)):
            ranked = node_frame.sort_values(["MAE", "node"], ascending=[ascending, True]).head(10)
            extremes.extend(dict(category=label, rank=rank, **row) for rank, row in enumerate(ranked.to_dict("records"), 1))

        for hour in range(24):
            selected = dates.hour == hour
            hours_rows.append(dict(horizon_min=h * 5, hour=hour, n_windows=int(selected.sum()),
                                   n_dates=int(len(np.unique(dates[selected].date))), **summary(p[selected], y[selected])))
        for label, selected in (("weekday", dates.dayofweek < 5), ("weekend", dates.dayofweek >= 5)):
            daytype_rows.append(dict(horizon_min=h * 5, day_type=label, n_windows=int(selected.sum()),
                                     n_dates=int(len(np.unique(dates[selected].date))), **summary(p[selected], y[selected])))

        # Per-horizon target quartiles; equal targets stay together, no random tie splitting.
        edges = np.unique(np.quantile(y, [0, .25, .5, .75, 1]))
        if len(edges) == 1:
            assignment, n_groups = np.zeros(y.shape, dtype=int), 1
        else:
            assignment, n_groups = np.searchsorted(edges[1:-1], y, side="left"), len(edges) - 1
        quantile_metadata[str(h)] = dict(edges=edges.tolist(), actual_groups=n_groups)
        for q in range(n_groups):
            selected = assignment == q
            lo, hi = (edges[0], edges[0]) if len(edges) == 1 else (edges[q], edges[q + 1])
            quantile_rows.append(dict(horizon_min=h * 5, group="Q" + str(q + 1),
                lower_edge=lo, upper_edge=hi, interval=("[" if q == 0 else "(") + f"{lo:g}, {hi:g}]",
                observation_percent=float(selected.mean() * 100), **summary(p[selected], y[selected])))

        sums, counts, details = [], [], []
        for day in calendar_dates:
            selected = dates.strftime("%Y-%m-%d") == day
            check(selected.any(), "Bootstrap date missing from a horizon")
            day_times = dates[selected]
            complete = len(day_times) == 288 and day_times[0].hour == 0 and day_times[0].minute == 0
            values = summary(p[selected], y[selected])
            daily_rows.append(dict(horizon_min=h * 5, date=day, n_windows=int(selected.sum()),
                                   complete_day=complete, first_target=str(day_times[0]), last_target=str(day_times[-1]), **values))
            sums.append(float(np.abs(p[selected] - y[selected]).sum()))
            counts.append(int(y[selected].size))
            details.append(dict(date=day, n_windows=int(selected.sum()), complete_day=complete))
        daily_sums.append(sums)
        daily_counts.append(counts)
        day_metadata[str(h)] = details

    nodes = save_table("node_metrics", node_rows)
    extremes = save_table("node_extremes", extremes)
    correlations = save_table("node_correlations", correlations)
    hours = save_table("hourly_metrics", hours_rows)
    daytypes = save_table("day_type_metrics", daytype_rows)
    quantiles = save_table("flow_quantile_metrics", quantile_rows)
    daily = save_table("daily_metrics", daily_rows)
    # Group totals must reconstruct the global MAE for each disjoint partition.
    for table in (nodes, hours, daytypes, quantiles, daily):
        for item in verified:
            group = table[table.horizon_min == item["horizon_min"]]
            check(group.n_MAE.sum() == item["n_MAE"], "Subgroup observation coverage mismatch")
            reconstructed = (group.MAE * group.n_MAE).sum() / group.n_MAE.sum()
            check(np.isclose(reconstructed, item["MAE"], atol=1e-10, rtol=0), "Subgroup MAE reconstruction failed")
            check(group.n_RMSE.sum() == item["n_RMSE"], "Subgroup RMSE count mismatch")
            reconstructed_rmse = np.sqrt((group.RMSE.fillna(0) ** 2 * group.n_RMSE).sum() / group.n_RMSE.sum())
            check(np.isclose(reconstructed_rmse, item["RMSE"], atol=1e-10, rtol=0), "Subgroup RMSE reconstruction failed")

    # One draw selects calendar-day blocks; all windows/nodes within a day stay together.
    # Same sampled dates across horizons; weight by each resample's observation counts.
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(calendar_dates), size=(args.bootstrap_reps, len(calendar_dates)))
    bootstrap_rows, bootstrap_summary = [], []
    for j, h in enumerate(HORIZONS):
        sums, counts = np.asarray(daily_sums[j]), np.asarray(daily_counts[j])
        values = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        lower, upper = np.quantile(values, [.025, .975])
        bootstrap_summary.append(dict(horizon_min=h * 5, MAE=verified[j]["MAE"], percentile_2_5=lower,
            percentile_97_5=upper, bootstrap_mean=float(values.mean()), bootstrap_std=float(values.std(ddof=1)),
            n_days=len(calendar_dates), reps=args.bootstrap_reps, seed=args.seed))
        bootstrap_rows.extend(dict(horizon_min=h * 5, replicate=i, MAE=float(v)) for i, v in enumerate(values))
        check(np.isclose(sums.sum() / counts.sum(), verified[j]["MAE"], atol=1e-10, rtol=0), "Day sums mismatch")
    boot = save_table("bootstrap_mae", bootstrap_summary)
    save_table("bootstrap_samples", bootstrap_rows)

    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for j, h in enumerate(HORIZONS):
        for ax, category, title in zip(axes[j], ("highest_MAE", "lowest_MAE"), ("10 node sai nhiều nhất", "10 node sai ít nhất")):
            selected = extremes[(extremes.horizon_min == h * 5) & (extremes.category == category)].sort_values("rank")
            ax.barh([str(n) for n in selected.node], selected.MAE, color="#2378a8")
            ax.invert_yaxis()
            ax.set(title=f"{h * 5} phút — {title}", xlabel="MAE (xe/5 phút/node)", ylabel="Chỉ số node")
    save_figure(fig, "01_node_extremes.png", "PEMS08 — các node theo MAE", "Xếp hạng riêng từng horizon; chỉ số node bắt đầu từ 0.")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, h in zip(axes, HORIZONS):
        selected = nodes[nodes.horizon_min == h * 5]
        r = correlations[correlations.horizon_min == h * 5].iloc[0]
        ax.scatter(selected.mean_target_flow, selected.MAE, s=18, alpha=.7)
        ax.set(title=f"{h * 5} phút; Pearson r={r.pearson_r:.3f}\nSpearman ρ={r.spearman_rho:.3f}",
               xlabel="Flow mục tiêu trung bình (xe/5 phút)", ylabel="MAE (xe/5 phút/node)")
    save_figure(fig, "02_node_flow_correlation.png", "PEMS08 — Flow trung bình và MAE theo node", "170 node; tương quan mô tả, không diễn giải nhân quả hoặc kiểm định độc lập.")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for ax, metric in zip(axes, ("MAE", "RMSE")):
        for h in HORIZONS:
            selected = hours[hours.horizon_min == h * 5]
            ax.plot(selected.hour, selected[metric], marker="o", ms=3, label=f"{h * 5} phút")
        ax.set(xticks=range(24), xlabel="Giờ của timestamp mục tiêu", ylabel=metric + " (xe/5 phút/node)")
        ax.legend()
        ax.grid(alpha=.2)
    save_figure(fig, "03_hourly_errors.png", "PEMS08 — sai số theo giờ", "Gom theo giờ mục tiêu, không phải giờ phát dự báo; số mẫu có trong CSV.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, metric in zip(axes, ("MAE", "RMSE")):
        for j, h in enumerate(HORIZONS):
            selected = daytypes[daytypes.horizon_min == h * 5].set_index("day_type").loc[["weekday", "weekend"]]
            ax.bar(np.arange(2) + (j - 1) * .23, selected[metric], width=.23, label=f"{h * 5} phút")
        ax.set(xticks=[0, 1], xticklabels=["Thứ Hai–Thứ Sáu", "Thứ Bảy–Chủ nhật"], ylabel=metric + " (xe/5 phút/node)")
        ax.legend()
    save_figure(fig, "04_day_type_errors.png", "PEMS08 — sai số theo loại ngày", "Phân loại theo lịch, không loại ngày lễ; số ngày và số mẫu hai nhóm khác nhau.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, metric in zip(axes, ("MAE", "RMSE")):
        for h in HORIZONS:
            selected = quantiles[quantiles.horizon_min == h * 5]
            ax.plot(selected.group, selected[metric], marker="o", label=f"{h * 5} phút")
        ax.set(xlabel="Nhóm phân vị Flow mục tiêu (thấp → cao)", ylabel=metric + " (xe/5 phút/node)")
        ax.legend()
    save_figure(fig, "05_flow_quantile_errors.png", "PEMS08 — sai số theo phân vị Flow", "Ngưỡng tính riêng trên target test từng horizon; giữ ties cùng nhóm, không dùng để chọn model.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.vlines(boot.horizon_min, boot.percentile_2_5, boot.percentile_97_5, lw=3, label="Percentile bootstrap 2,5–97,5%")
    ax.scatter(boot.horizon_min, boot.MAE, color="#d18324", zorder=3, label="MAE test đầy đủ")
    ax.set(xticks=[15, 30, 60], xlabel="Khoảng dự báo (phút)", ylabel="MAE (xe/5 phút/node)")
    ax.legend(loc="upper left")
    save_figure(fig, "06_bootstrap_mae.png", "PEMS08 — khoảng dao động MAE theo khối ngày", f"{args.bootstrap_reps} lần lấy mẫu từ {len(calendar_dates)} ngày test; không phải dao động giữa seed mô hình.")

    fig, ax = plt.subplots(figsize=(12, 5))
    for h in HORIZONS:
        selected = daily[daily.horizon_min == h * 5]
        ax.plot(selected.date, selected.MAE, marker="o", label=f"{h * 5} phút")
    ax.set(xlabel="Ngày mục tiêu (2016)", ylabel="MAE (xe/5 phút/node)")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    save_figure(fig, "07_daily_mae.png", "PEMS08 — MAE từng ngày test", "Ngày đầu và cuối không đủ 288 cửa sổ tại một số horizon; xem complete_day và n_windows trong CSV.")

    elapsed = time.perf_counter() - tick
    for relative, original in original_hashes.items():
        check(sha256(root / relative) == original, "Protected file changed: " + relative)
    protocol = dict(status="success", created_utc=datetime.now(timezone.utc).isoformat(),
        run_id="PEMS08-repo96-author-pretrained", attempt="0001", best_epoch=68,
        source_export="artifacts/report-inputs/PEMS08", device="cpu", model_executed=False,
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__),
        export_sha_verified_files=len(entries), test_windows=3566, nodes=170, horizons_steps=list(HORIZONS),
        metric_global_gate=dict(atol=1e-8, rtol=0, passed=True),
        masks=dict(MAE="all finite targets, zeros included", RMSE="target != 0", MAPE="target >=0.01; percent",
                   finite_policy="stop on any nonfinite prediction or target", sums="global float64"),
        timestamps=dict(origin=ORIGIN.isoformat(), sampling_minutes=5, calendar="naive local benchmark clock",
                        grouping_timestamp="forecast_start + horizon - 1; target clock, not issue clock",
                        weekday="Monday=0; Friday at origin", weekend="Saturday/Sunday only; holidays not modeled"),
        node_correlation=dict(flow_mean="mean target Flow per node over that horizon's full test windows, including zeros",
                              coefficients=["Pearson", "Spearman (average ranks for ties)"], n_nodes=170, p_values_computed=False),
        quantiles=dict(scope="each horizon's test target sensor-time values", requested_groups=4,
                       boundaries="first interval closed both sides, subsequent (lower,upper]; ties not split",
                       duplicate_quantile_edges="collapse duplicate edges; fewer groups if necessary", per_horizon=quantile_metadata),
        bootstrap=dict(reps=args.bootstrap_reps, seed=args.seed, unit="target-calendar-day block",
                       n_days=len(calendar_dates), dates=calendar_dates.tolist(),
                       statistic="sum absolute errors / sum observation counts in sampled blocks",
                       sample_size="same number of dates, replacement; partial first/last blocks retained",
                       paired_horizons=True, interval="percentile 2.5 to 97.5, nominal 95%",
                       interpretation="test-sampling variation of fixed trained model; NOT between-training-seed variation",
                       limitations="only 13 dates; unequal/partial blocks; between-day dependence and future distribution shifts not modeled",
                       per_horizon_day_coverage=day_metadata),
        figures=dict(dpi=200, files=figures), verification_seconds=verification_seconds,
        total_seconds_before_protocol_and_notes=elapsed, protected_files_unchanged=True,
        protected_file_sha256=original_hashes, source_index_sha256=sha256(root / "datasets/PEMS08/test_index.pkl"),
        export_hash_manifest_sha256=sha256(inputs / "export_hashes.json"), script_sha256=sha256(Path(__file__)))
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# Ghi chú phân tích sai số PEMS08", "",
             "Nguồn: run `PEMS08-repo96-author-pretrained`, attempt `0001`, best epoch 68. Không chạy lại mô hình.",
             "", "## Kiểm chứng và cách tính", "",
             "Đã kiểm SHA-256 113 file export/metadata; đủ 3.566 cửa sổ, 170 node. Metric tổng khớp bảng chuẩn với atol=1e-8, rtol=0.",
             "MAE giữ nhãn 0; RMSE loại nhãn 0; MAPE loại nhãn <0,01. Tổng sai số được cộng ở float64. Các phân nhóm tái tạo được MAE/RMSE tổng.",
             "Gom theo timestamp mục tiêu = forecast_start + horizon - 1. Lịch tái dựng từ 2016-07-01 00:00, mỗi 5 phút; không đổi múi giờ.",
             "", "## Quan sát mô tả", ""]
    for j, h in enumerate(HORIZONS):
        n = nodes[nodes.horizon_min == h * 5]
        worst = n.loc[n.MAE.idxmax()]
        best = n.loc[n.MAE.idxmin()]
        r = correlations[correlations.horizon_min == h * 5].iloc[0]
        hr = hours[hours.horizon_min == h * 5]
        highest_hour, lowest_hour = hr.loc[hr.MAE.idxmax()], hr.loc[hr.MAE.idxmin()]
        dt = daytypes[daytypes.horizon_min == h * 5].set_index("day_type")
        q = quantiles[quantiles.horizon_min == h * 5]
        b = boot[boot.horizon_min == h * 5].iloc[0]
        lines.append(f"- **{h * 5} phút:** MAE {verified[j]['MAE']:.4f}; node MAE cao nhất {int(worst.node)} ({worst.MAE:.4f}), thấp nhất {int(best.node)} ({best.MAE:.4f}). Pearson với Flow trung bình {r.pearson_r:.4f}, Spearman {r.spearman_rho:.4f}.")
        lines.append(f"  Theo giờ: MAE lớn nhất {int(highest_hour.hour)}h ({highest_hour.MAE:.4f}), nhỏ nhất {int(lowest_hour.hour)}h ({lowest_hour.MAE:.4f}). Thứ Hai–Sáu {dt.loc['weekday','MAE']:.4f}; Thứ Bảy–Chủ nhật {dt.loc['weekend','MAE']:.4f}.")
        lines.append(f"  Theo phân vị: Q1 MAE {q.iloc[0].MAE:.4f}, nhóm cao nhất {q.iloc[-1]['group']} MAE {q.iloc[-1].MAE:.4f}. Khoảng percentile bootstrap 95%: [{b.percentile_2_5:.4f}, {b.percentile_97_5:.4f}].")
    lines += ["", "## Giới hạn", "",
              "Các quan sát là mô tả, không chứng minh Flow, khung giờ hoặc loại ngày gây ra sai số. Không có p-value hoặc kết luận về khác biệt có ý nghĩa thống kê.",
              "Phân vị được tính riêng từ target test từng horizon; các quan sát bằng ngưỡng ở cùng nhóm nên số mẫu nhóm có thể không bằng nhau. Đây là phân tích sau đánh giá, không chọn model bằng test.",
              "Bootstrap lấy lại toàn bộ khối ngày, giữ các node và cửa sổ trong ngày đi cùng nhau. Có 13 ngày (9 ngày thứ Hai–Sáu, 4 ngày cuối tuần); ngày đầu/cuối có khối chưa đầy đủ ở một số horizon. MAE được tính bằng tỷ số tổng sai số/tổng số quan sát, không trung bình không trọng số giữa các ngày.",
              f"Khoảng 2,5–97,5% từ {args.bootstrap_reps} lần bootstrap mô tả dao động do lấy mẫu test của **một model cố định**, không phải dao động giữa seed huấn luyện. Seed {args.seed} chỉ điều khiển lấy mẫu bootstrap. Phương pháp không mô hình hóa phụ thuộc giữa các ngày; số ngày ít nên khoảng có tính thăm dò.",
              "MAE tuyệt đối có cùng đơn vị xe/5 phút/node; node lưu lượng thấp có thể có MAE thấp mà chưa chứng minh dự báo tương đối tốt hơn. Chưa có nhãn sự kiện để suy diễn sai số do tai nạn hoặc tắc đường.",
              "", "## Tệp kết quả", "",
              "`node_metrics.csv` chứa cả 170 node mỗi horizon; `node_extremes.csv` chứa 10 cao/10 thấp; `node_correlations.csv` chứa tương quan. Các bảng còn lại tương ứng giờ, loại ngày, phân vị, từng ngày và bootstrap. PNG đều dpi=200 và có dòng nguồn.",
              f"Thời gian đến trước bước ghi protocol/ghi chú: {elapsed:.2f} giây. Không sửa README, config, graph, split hoặc results/experiment_*.", ""]
    (output / "NOTES.md").write_text("\n".join(lines), encoding="utf-8")
    print(boot.to_string(index=False), flush=True)
    print(correlations.to_string(index=False), flush=True)
    print("SUCCESS:", len(tables), "CSV tables;", len(figures), "PNG (200 dpi); protocol.json and NOTES.md ->", output, flush=True)


if __name__ == "__main__":
    main()
