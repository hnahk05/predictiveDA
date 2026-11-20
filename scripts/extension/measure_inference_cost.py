"""Measure checkpoint inference; optional isolated optimizer probe is restored, never saved."""
import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import statistics
import sys
import time
import threading
import psutil

import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def describe(values):
    return dict(mean=statistics.mean(values), std_sample=statistics.stdev(values),
                min=min(values), max=max(values), observations=len(values))


def count_parameters(module):
    parameters = list(module.parameters())
    return dict(total=sum(p.numel() for p in parameters),
                requires_grad_true=sum(p.numel() for p in parameters if p.requires_grad),
                frozen=sum(p.numel() for p in parameters if not p.requires_grad))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--softmax-compat", choices=("on", "off"), default="on")
    parser.add_argument('--output-prefix', type=Path, default=Path('results/extension/inference_cost_v2'))
    parser.add_argument('--parameter-audit', action='store_true', help='One isolated optimizer diagnostic, restored in memory')
    args = parser.parse_args()
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    os.chdir(root)
    prefix = args.output_prefix.resolve()
    require(not prefix.with_suffix('.json').exists() and not prefix.with_suffix('.csv').exists(),
            'Refusing to overwrite existing results')
    require(torch.__version__ == "1.8.1+cu111", "Unexpected Torch version")
    require(torch.cuda.is_available(), "CUDA unavailable; stop without fabricating timings")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    require("3050" in properties.name, "Expected the user's RTX 3050")
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    protected = [root / name for name in ("README.md", ".gitignore", "utils/cuda_softmax_compat.py",
                 "utils/datasets.py", "utils/load_data.py", "results/experiment_manifest.json", "results/experiment_metrics.csv",
                 "datasets/PEMS08/data.pkl", "datasets/PEMS08/test_index.pkl", "datasets/sensor_graph/adj_mx_08.pkl")]
    protected += sorted((root / "models").glob("*.py")) + sorted((root / "configs").rglob("*.py"))
    before = {p.relative_to(root).as_posix(): sha256(p) for p in protected}
    manifest = json.loads((root / "results/experiment_manifest.json").read_text(encoding="utf-8"))
    reference = next(r for r in manifest["runs"] if r["dataset"] == "PEMS08")
    result_path = root / "artifacts/report-inputs/PEMS08/result.json"
    run_result = json.loads(result_path.read_text(encoding="utf-8"))
    require(run_result["id"] == reference["run_id"] and run_result["attempt"] == 1, "Wrong reference run")
    expected_hashes = [v for k, v in run_result["artifact_sha256"].items() if k.endswith("/LSTTN_best_val_loss.pt")]
    require(len(expected_hashes) == 1, "Cannot identify checkpoint hash")
    if args.checkpoint:
        checkpoint = args.checkpoint.expanduser().resolve()
    else:
        locator = root / "results/extension/graph_conv_check.json"
        if locator.exists():
            checkpoint = root / json.loads(locator.read_text(encoding="utf-8"))["checkpoint"]["path"]
        else:
            candidates = sorted((root / ".local_checks/restored-pems08").glob(
                "*/runs/PEMS08-repo96-author-pretrained/0001/checkpoints/*/LSTTN_best_val_loss.pt"))
            require(len(candidates) == 1, "Specify --checkpoint; no unique restored checkpoint found")
            checkpoint = candidates[0]
    require(checkpoint.is_file() and sha256(checkpoint) == expected_hashes[0], "Checkpoint missing or hash mismatch")
    pretrained = root / "pretrained_transformers/PEMS08.pt"
    require(sha256(pretrained) == reference["provided_pretrained_sha256"], "Pretrained hash mismatch")

    from models.lsttn import LSTTN
    from utils.datasets import MTSDataset
    from utils import cuda_softmax_compat as compat
    os.environ["LSTTN_SPLIT_SOFTMAX"] = "1" if args.softmax_compat == "on" else "0"
    compat_enabled = compat.activate_from_env()
    split_calls = [0]
    if compat_enabled:
        original_split = compat.split_softmax

        def counted_split(*positional, **keywords):
            split_calls[0] += 1
            return original_split(*positional, **keywords)

        compat.split_softmax = counted_split  # Process-local observation; source file unchanged.
    cfg = runpy.run_path(str(root / "configs/PEMS08/forecasting.py"))["CFG"]
    cfg.MODEL.PARAM.LSTTN["pretrained_path"] = str(pretrained)
    model = LSTTN(cfg, **cfg.MODEL.PARAM).cpu()
    state = torch.load(str(checkpoint), map_location="cpu")
    require(state["epoch"] == reference["best_epoch"] == 68, "Wrong checkpoint epoch")
    loaded = model.load_state_dict(state["model_state_dict"], strict=True)
    require(not loaded.missing_keys and not loaded.unexpected_keys, "Strict-load mismatch")
    del state
    model.eval()
    require(all(not module.training for module in model.modules()), "Model not completely in eval mode")
    total = count_parameters(model)
    blocks = {name: count_parameters(module) for name, module in model.named_children()}
    require(sum(row["total"] for row in blocks.values()) == total["total"], "Parameter blocks do not partition model")
    require(blocks["transformer"]["requires_grad_true"] == 0, "Active Transformer is not frozen")
    heads = model.transformer.encoder.transformer_encoder.layers[0].self_attn.num_heads
    tokens = 4032 // model.transformer.patch_size
    require((heads, tokens) == (4, 336), "Unexpected attention geometry")
    dataset = MTSDataset(str(root / "datasets/PEMS08/data.pkl"), str(root / "datasets/PEMS08/test_index.pkl"),
                         seq_len=4032, throw=False, pretrain=False)
    require(len(dataset) == 3566, "Wrong test dataset size")
    device = torch.device("cuda:0")
    model.to(device)
    print("STRICT LOAD: epoch 68; eval/no_grad; GPU:", properties.name, flush=True)
    print("REGISTERED PARAMETERS:", total, flush=True)
    print("SOFTMAX COMPAT:", compat_enabled, "; threshold=2**30 elements", flush=True)

    observed_calls = {}
    results, details = [], []
    progress = {}
    process = psutil.Process()

    def get_inputs(offset, batch):
        items = [dataset[i] for i in range(offset, offset + batch)]
        short = torch.stack([item[1] for item in items]).to(device)
        long = torch.stack([item[2] for item in items]).to(device)
        return short, long

    def validate(output, batch):
        require(tuple(output.shape) == (batch, 170, 12), "Unexpected model output shape")
        require(not output.requires_grad and torch.isfinite(output).all().item(), "Invalid inference output")

    def measure(batch):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        base_allocated = torch.cuda.memory_allocated()
        call_count_start = split_calls[0]
        print("BATCH", batch, "warm-up 5", flush=True)
        progress.update(phase="warmup", warmup_completed=0, measured_completed=0, profiling_completed=0)
        with torch.no_grad():
            handles = []
            if not observed_calls:
                for name, module in model.named_children():
                    def count_call(_module, _inputs, _output, key=name):
                        observed_calls[key] = observed_calls.get(key, 0) + 1
                    handles.append(module.register_forward_hook(count_call))
            try:
                for i in range(5):
                    short, long = get_inputs(i * batch, batch)
                    output = model(short, long)
                    torch.cuda.synchronize()
                    validate(output, batch)
                    progress["warmup_completed"] += 1
                    del output, short, long
                    if i == 0:
                        for handle in handles:
                            handle.remove()
                        handles = []
            finally:
                for handle in handles:
                    handle.remove()

            wall_ms, peaks_allocated, peaks_reserved = [], [], []
            rss_before = process.memory_info().rss
            rss_samples = [rss_before]
            rss_stop = threading.Event()
            def sample_rss():
                while not rss_stop.wait(.01):
                    rss_samples.append(process.memory_info().rss)
            rss_thread = threading.Thread(target=sample_rss, daemon=True)
            rss_thread.start()
            progress["phase"] = "unhooked_measurement"
            try:
                for i in range(20):
                    short, long = get_inputs((5 + i) * batch, batch)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    output = model(short, long)
                    torch.cuda.synchronize()
                    wall_ms.append((time.perf_counter() - start) * 1000)
                    peaks_allocated.append(torch.cuda.max_memory_allocated())
                    peaks_reserved.append(torch.cuda.max_memory_reserved())
                    validate(output, batch)
                    progress["measured_completed"] += 1
                    del output, short, long
            finally:
                rss_samples.append(process.memory_info().rss)
                rss_stop.set()
                rss_thread.join()
            print("BATCH", batch, "main measured; separate CUDA-event block profiling", flush=True)

            events = {}
            handles = []
            for name, module in (("transformer", model.transformer), ("graph_wavenet", model.short_term_trend_extractor)):
                def pre_hook(_module, _inputs, key=name):
                    require(key not in events, "Block called more than once")
                    events[key] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    events[key][0].record()

                def post_hook(_module, _inputs, _output, key=name):
                    events[key][1].record()

                handles.extend([module.register_forward_pre_hook(pre_hook), module.register_forward_hook(post_hook)])
            profile_rows = []
            progress["phase"] = "separate_profile"
            try:
                for i in range(20):
                    events.clear()
                    short, long = get_inputs((5 + i) * batch, batch)
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    begin.record()
                    output = model(short, long)
                    end.record()
                    torch.cuda.synchronize()
                    require(set(events) == {"transformer", "graph_wavenet"}, "Missing block events")
                    full = begin.elapsed_time(end)
                    transformer = events["transformer"][0].elapsed_time(events["transformer"][1])
                    graph_wavenet = events["graph_wavenet"][0].elapsed_time(events["graph_wavenet"][1])
                    remainder = full - transformer - graph_wavenet
                    require(remainder >= 0, "Negative event-time remainder")
                    profile_rows.append(dict(full_event_ms=full, transformer_ms=transformer,
                                             graph_wavenet_ms=graph_wavenet, remainder_ms=remainder))
                    validate(output, batch)
                    progress["profiling_completed"] += 1
                    del output, short, long
            finally:
                for handle in handles:
                    handle.remove()

        elements = batch * 170 * heads * tokens ** 2
        attention_bytes = elements * 4
        profile_stats = {key: describe([row[key] for row in profile_rows]) for key in profile_rows[0]}
        row = dict(device="cuda:0", batch=batch, status="success", warmup_batches=5, measured_batches=20,
            latency_mean_ms=statistics.mean(wall_ms), latency_std_ms=statistics.stdev(wall_ms),
            peak_allocated_bytes=max(peaks_allocated), peak_allocated_GB=max(peaks_allocated) / 1e9,
            peak_allocated_GiB=max(peaks_allocated) / 2 ** 30,
            peak_reserved_bytes=max(peaks_reserved), model_baseline_allocated_bytes=base_allocated,
            attention_elements_one_layer=elements, attention_FP32_bytes_one_matrix=attention_bytes,
            attention_FP32_GB_one_matrix=attention_bytes / 1e9,
            peak_allocated_to_one_attention_matrix_ratio=max(peaks_allocated) / attention_bytes,
            profile_transformer_mean_ms=profile_stats["transformer_ms"]["mean"],
            profile_graph_wavenet_mean_ms=profile_stats["graph_wavenet_ms"]["mean"],
            profile_remainder_mean_ms=profile_stats["remainder_ms"]["mean"],
            profile_full_event_mean_ms=profile_stats["full_event_ms"]["mean"],
            softmax_compat_enabled=compat_enabled, split_softmax_calls=split_calls[0] - call_count_start,
            torch_version=torch.__version__, gpu=properties.name,
            total_parameters=total["total"], frozen_parameters=total["frozen"], requires_grad_true=total["requires_grad_true"])
        detail = dict(batch=batch, unhooked_wall_ms=wall_ms, unhooked_peak_allocated_bytes=peaks_allocated,
                      separate_profile_per_batch=profile_rows, separate_profile_statistics=profile_stats,
                      test_window_ranges=dict(warmup=[0, 5 * batch - 1], measured=[5 * batch, 25 * batch - 1]),
                      note="Separate profiling repeats the same 20 measured input batches; it does not replace the unhooked wall-time measurement.")
        row.update(process_rss_before_bytes=rss_before, process_rss_sampled_peak_bytes=max(rss_samples),
                   process_rss_after_bytes=process.memory_info().rss)
        return row, detail

    for batch in (1, 2, 4):
        try:
            row, detail = measure(batch)
            results.append(row)
            details.append(detail)
            print("DONE:", json.dumps(row), flush=True)
        except RuntimeError as error:
            status = "oom" if "out of memory" in str(error).lower() else "error"
            results.append(dict(device="cuda:0", batch=batch, status=status, torch_version=torch.__version__,
                                gpu=properties.name, softmax_compat_enabled=compat_enabled,
                                **{key: value for key, value in progress.items() if key != "phase"}))
            details.append(dict(batch=batch, status=status, phase=progress.get("phase"), error=str(error),
                                skipped_larger_batches=[b for b in (1, 2, 4) if b > batch]))
            print("STOP at batch", batch, status, str(error), flush=True)
            gc.collect()
            if status == "oom":
                torch.cuda.empty_cache()
            break
    require(all(p.grad is None for p in model.parameters()), "Unexpected parameter gradients")
    for relative, digest in before.items():
        require(sha256(root / relative) == digest, "Protected file changed: " + relative)
    require(sha256(checkpoint) == expected_hashes[0], "Checkpoint changed")
    uncalled = {name: blocks[name] for name in blocks if not observed_calls.get(name)}
    parameter_diagnostic = None
    if args.parameter_audit:
        from utils.parameter_audit import audit_parameters
        inputs = get_inputs(0, 1)
        target = dataset[0][0][...,0].transpose(0,1).unsqueeze(0).to(device)
        parameter_diagnostic = audit_parameters(model, inputs, cfg.TRAIN.OPTIM, target)
        require(sha256(checkpoint) == expected_hashes[0], 'Checkpoint changed')
    report = dict(status="success" if len(results) == 3 and all(r["status"] == "success" for r in results) else "stopped",
        created_utc=datetime.now(timezone.utc).isoformat(), purpose="INFERENCE cost, not training cost or A6000 training benchmark",
        versions=dict(python=platform.python_version(), torch=torch.__version__, cuda_build=torch.version.cuda,
                      cudnn=torch.backends.cudnn.version()), platform=platform.platform(),
        hardware=dict(gpu=properties.name, total_memory_bytes=properties.total_memory, device=0),
        checkpoint=dict(path=os.path.relpath(checkpoint, root), sha256=expected_hashes[0], epoch=68,
                        strict_load=True, missing_keys=list(loaded.missing_keys), unexpected_keys=list(loaded.unexpected_keys)),
        model=dict(width=96, hidden=[32, 32], nodes=170, short_steps=12, long_steps=4032,
                   patch_steps=12, tokens=tokens, heads=heads, encoder_layers=len(model.transformer.encoder.transformer_encoder.layers)),
        parameter_counts=dict(registered=total, by_top_level_block=blocks,
                              top_level_calls_observed_first_warmup=observed_calls,
                              registered_blocks_not_called=uncalled,
                              warning="requires_grad=True counts registered parameters, not proof of updates during training. transformer1 is an extra registered model not called in this forward. The active Transformer also registers pretraining-head parameters unused in inference."),
        flags=dict(eval=True, no_grad=True, precision="FP32", AMP=False,
                   TF32_matmul=False, TF32_cudnn=False, cudnn_benchmark=False, cudnn_deterministic=False,
                   softmax_compat_enabled=compat_enabled,
                   softmax_wrapper_installed=bool(getattr(torch.nn.functional.softmax, "_lsttn_chunked", False)),
                   softmax_split_threshold_elements=2 ** 30, observed_split_calls=split_calls[0]),
        measurement=dict(requested_batches=[1, 2, 4], warmup_per_batch=5, measured_per_batch=20,
            main_timer="perf_counter around forward, CUDA synchronized before/after; no hooks",
            std="sample standard deviation, ddof=1", input="real first test windows, dataset float32",
            excluded="checkpoint loading, CPU data slicing/stacking, input H2D copies, inverse scaling, metrics and export",
            included="model forward including any internal support CPU-to-GPU copies retained from author code",
            block_timer="separate pass of 20 forwards; pre/post hooks and CUDA events on default stream",
            remainder="profile full CUDA-event interval minus Transformer and Graph WaveNet; includes other blocks, reshapes, internal copies and dispatch gaps; not isolated kernel compute",
            memory="max_memory_allocated, reset after resident inputs and synchronization each measured iteration; includes model, inputs, live activations and outputs. Not nvidia-smi VRAM; reserved memory reported separately",
            attention="B * 170 * 4 * 336^2 elements, FP32 4 bytes; one dense per-head attention matrix in one layer only, not full peak memory or sum over layers",
            CPU_measurement="CPU inference timing not performed; process RSS sampled", no_training=not args.parameter_audit,
            full_training=False, isolated_optimizer_probe=args.parameter_audit),
        results=results, per_batch_details=details, parameter_diagnostic=parameter_diagnostic,
        ram_measurement='Process RSS sampled every 10ms during main benchmark incl input preparation; sampled peak may miss brief spikes; not system-wide RAM.',
        protected_files_unchanged=True,
        protected_input_source_sha256=before, script_sha256=sha256(Path(__file__)),
        elapsed_seconds=time.perf_counter() - started,
        limitations=["Single checkpoint/device session; no between-seed or cross-device statistical claim.",
                     "Separate hook/event measurements have instrumentation overhead and are not the main unhooked timing.",
                     "Laptop power, temperature and other processes can affect timings; hardware clock/power settings were not changed.",
                     "One attention-matrix estimate excludes other buffers, scores/softmax copies, parameters and allocator overhead."])
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in results for key in row))
    with prefix.with_suffix('.csv').open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    with prefix.with_suffix('.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print("SAVED", prefix, "status:", report["status"], flush=True)


if __name__ == "__main__":
    main()
