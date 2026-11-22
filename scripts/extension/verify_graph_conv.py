"""Check the author's graph expression on CPU; never modify model/config files."""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import runpy
import sys
import time

import numpy as np
import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Optional explicit trained PEMS08 checkpoint path")
    args = parser.parse_args()
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    os.chdir(root)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    protected = [root / x for x in ("README.md", ".gitignore", "utils/adj_mx.py", "utils/load_data.py",
                 "results/experiment_manifest.json", "results/experiment_metrics.csv")]
    protected += sorted((root / "configs").rglob("*.py")) + sorted((root / "models").glob("*.py"))
    original_hashes = {p.relative_to(root).as_posix(): sha256(p) for p in protected}

    # Use the real repository classes, not a replacement einsum implementation.
    from models.lsttn import DynamicGraphConv, LSTTN
    from models.graph_wavenet import nconv
    require(Path(inspect.getfile(DynamicGraphConv)).resolve() == root / "models/lsttn.py", "Unexpected class source")
    toy_layer = DynamicGraphConv(num_nodes=2, input_dim=1, output_dim=1, dropout=0).cpu().eval()
    x = torch.tensor([[[1.0], [10.0]]], dtype=torch.float64)
    swap = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    identity = torch.eye(2, dtype=torch.float64)
    with torch.no_grad():
        actual = toy_layer.conv(x, swap)
        conventional = torch.matmul(swap, x[0]).unsqueeze(0)
        same_rowsum = toy_layer.conv(x, identity)
        gwn_actual = nconv()(x.permute(0, 2, 1).unsqueeze(-1), swap).reshape(-1)
    require(torch.equal(actual, x), "Toy author expression did not retain node values")
    require(conventional.reshape(-1).tolist() == [10.0, 1.0], "Incorrect A @ x reference")
    require(torch.equal(same_rowsum, actual), "Same-row-sum graph check failed")
    require(gwn_actual.tolist() == [10.0, 1.0], "GraphWaveNet control failed")
    # Non-stochastic rows also test a scale of 2, 4 and 0, with two feature channels.
    probe3 = torch.tensor([[[1., 2.], [10., 20.], [100., 200.]]], dtype=torch.float64)
    matrix3 = torch.tensor([[0., 2., 0.], [3., 0., 1.], [0., 0., 0.]], dtype=torch.float64)
    layer3 = DynamicGraphConv(3, 2, 2, dropout=0).cpu().eval()
    nonstochastic = layer3.conv(probe3, matrix3)
    require(torch.equal(nonstochastic, probe3 * matrix3.sum(1)[None, :, None]), "Row-sum identity failed")
    toy = dict(class_source="models/lsttn.py:DynamicGraphConv.conv",
               method_source=inspect.getsource(DynamicGraphConv.conv).strip(), used_repo_class=True,
               expression_reimplemented_instead=False, x=[1, 10], adjacency=swap.tolist(),
               repo_result=actual.reshape(-1).tolist(), conventional_A_times_x=conventional.reshape(-1).tolist(),
               identity_graph_result=same_rowsum.reshape(-1).tolist(),
               graph_wavenet_control=dict(source="models/graph_wavenet.py:nconv.forward", result=gwn_actual.tolist(),
                    note="Different operator; it does mix nodes. Swap is symmetric, so transpose convention is immaterial in this control."),
               nonstochastic_test=dict(row_sums=matrix3.sum(1).tolist(), output=nonstochastic.tolist(), passed=True))

    with np.errstate(divide="ignore", invalid="ignore"):
        configuration = runpy.run_path(str(root / "configs/PEMS08/forecasting.py"))
    cfg = configuration["CFG"]
    graph_path = (root / configuration["GRAPH_PKL_PATH"]).resolve()
    graph_before = sha256(graph_path)
    raw = np.asarray(configuration["_"])
    supports = cfg.MODEL.PARAM.LSTTN["supports"]
    require(raw.shape == (170, 170) and len(supports) == 2, "Unexpected PEMS08 graph/support shape")
    real_layer = DynamicGraphConv(170, 2, 2, dropout=0).cpu().eval()
    profiles = []

    def profile(name, matrix, used_by, note):
        matrix = torch.as_tensor(matrix).detach().cpu()
        if not matrix.is_floating_point():
            matrix = matrix.to(torch.float64)
        require(matrix.shape == (170, 170) and torch.isfinite(matrix).all().item(), "Invalid matrix: " + name)
        rows64 = matrix.to(torch.float64).sum(1)
        native_rows = matrix.sum(1)
        probe = torch.linspace(-1, 1, 340, dtype=matrix.dtype).reshape(1, 170, 2)
        with torch.no_grad():
            out = real_layer.conv(probe, matrix)
            scaled = probe * native_rows[None, :, None]
        max_scale_error = float((out - scaled).abs().max())
        require(torch.allclose(out, scaled, atol=1e-5, rtol=1e-5), "Real matrix row-scaling check failed")
        result = dict(name=name, used_by=used_by, note=note, shape=list(matrix.shape), dtype=str(matrix.dtype),
                      row_sum_min=float(rows64.min()), row_sum_max=float(rows64.max()), row_sum_mean=float(rows64.mean()),
                      rows_exactly_zero=int((rows64 == 0).sum()),
                      rows_abs_deviation_from_one_gt_1e_6=int(((rows64 - 1).abs() > 1e-6).sum()),
                      max_abs_row_sum_deviation_from_one=float((rows64 - 1).abs().max()),
                      row_sum_calculation="float64 sum of actual stored matrix entries",
                      native_dtype_row_sum_min=float(native_rows.min()), native_dtype_row_sum_max=float(native_rows.max()),
                      row_sums=rows64.tolist(), probe_operator="models.lsttn.DynamicGraphConv.conv",
                      probe_max_abs_error_against_row_scaling=max_scale_error,
                      probe_max_abs_error_against_identity=float((out - probe).abs().max()),
                      near_identity_condition="all row sums within 1e-6 of one; output error additionally scales with input magnitude")
        profiles.append(result)

    profile("raw_adjacency", raw, "input to graph loader; not the normalized support passed directly to conv",
            "Original PEMS08 adjacency from the file referenced by author config.")
    profile("forward_support", supports[0], "DynamicGraphConv and GraphWaveNet",
            "Repo doubletransition: transition_matrix(A).T. Row-normalization occurs before transpose; output rows need not sum to one.")
    profile("backward_support", supports[1], "DynamicGraphConv and GraphWaveNet",
            "Repo doubletransition: transition_matrix(A.T).T. Inspect actual row sums after transpose.")

    if args.checkpoint:
        checkpoint_path = args.checkpoint.expanduser().resolve()
        require(checkpoint_path.is_file(), "Checkpoint not found")
    else:
        candidates = sorted((root / ".local_checks/restored-pems08").glob(
            "*/runs/PEMS08-repo96-author-pretrained/0001/checkpoints/*/LSTTN_best_val_loss.pt"))
        require(len(candidates) <= 1, "Multiple checkpoints found: specify --checkpoint")
        checkpoint_path = candidates[0] if candidates else None
    checkpoint_info = dict(available=False, strict_load_verified=False,
                           reason="No trained checkpoint found in the restored run; no adaptive matrix fabricated.")
    if checkpoint_path is not None:
        checkpoint_hash = sha256(checkpoint_path)
        result_json = root / "artifacts/report-inputs/PEMS08/result.json"
        recorded_result = json.loads(result_json.read_text(encoding="utf-8"))
        require(recorded_result["id"] == "PEMS08-repo96-author-pretrained" and recorded_result["attempt"] == 1,
                "Wrong run reference")
        expected = [value for key, value in recorded_result["artifact_sha256"].items()
                    if key.endswith("/LSTTN_best_val_loss.pt")]
        require(len(expected) == 1 and checkpoint_hash == expected[0], "Trained checkpoint hash mismatch")
        manifest = json.loads((root / "results/experiment_manifest.json").read_text(encoding="utf-8"))
        recorded_run = next(r for r in manifest["runs"] if r["dataset"] == "PEMS08")
        pretrained = root / "pretrained_transformers/PEMS08.pt"
        require(sha256(pretrained) == recorded_run["provided_pretrained_sha256"], "Pretrained checkpoint hash mismatch")
        cfg.USE_GPU, cfg.GPU_NUM = False, 0  # Only in memory; author config file is unchanged.
        cfg.MODEL.PARAM.LSTTN["pretrained_path"] = str(pretrained)
        model = LSTTN(cfg, **cfg.MODEL.PARAM).cpu().eval()
        state = torch.load(str(checkpoint_path), map_location="cpu")
        require(state["epoch"] == recorded_run["best_epoch"] == 68, "Unexpected best epoch")
        load_result = model.load_state_dict(state["model_state_dict"], strict=True)
        require(not load_result.missing_keys and not load_result.unexpected_keys, "Strict load mismatch")
        require(all(not p.is_cuda for p in model.parameters()), "Model unexpectedly uses CUDA")
        with torch.no_grad():
            for name, branch in (("adaptive_weekly", model.weekly_seasonality_extractor),
                                 ("adaptive_daily", model.daily_seasonality_extractor)):
                adaptive = torch.softmax(torch.relu(torch.mm(branch.node_vec1, branch.node_vec2)), dim=1)
                profile(name, adaptive, "DynamicGraphConv", "Exact softmax(relu(node_vec1 @ node_vec2), dim=1) from strict-loaded trained weights.")
            short = model.short_term_trend_extractor
            adaptive_short = torch.softmax(torch.relu(torch.mm(short.nodevec1, short.nodevec2)), dim=1)
            profile("adaptive_short_term", adaptive_short, "GraphWaveNet nconv, NOT the problematic periodicity conv",
                    "Row-stochastic matrix is shown for completeness. GraphWaveNet's own nconv mixes nodes and is not an identity operator merely because rows sum to one.")
        try:
            recorded_path = checkpoint_path.relative_to(root).as_posix()
        except ValueError:
            recorded_path = str(checkpoint_path)
        checkpoint_info = dict(available=True, path=recorded_path, sha256=checkpoint_hash,
            reference_hash_verified=True, reference_result_sha256=sha256(result_json),
            strict_load_verified=True, loaded_entire_model=True, state_dict_keys=len(state["model_state_dict"]),
            missing_keys=list(load_result.missing_keys), unexpected_keys=list(load_result.unexpected_keys),
            epoch=state["epoch"], device="cpu", full_model_forward_executed=False,
            note="Loaded model weights only; no optimizer resume, training, forecast or accuracy measurement.")
        require(sha256(checkpoint_path) == checkpoint_hash, "Checkpoint changed during verification")

    for relative, original in original_hashes.items():
        require(sha256(root / relative) == original, "Protected source changed: " + relative)
    require(sha256(graph_path) == graph_before, "Graph input changed")
    report = dict(status="success", created_utc=datetime.now(timezone.utc).isoformat(),
        versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
        device="cpu", cuda_used=False, toy=toy,
        real_graph=dict(file=graph_path.relative_to(root).as_posix(), sha256=graph_before,
                        author_config="configs/PEMS08/forecasting.py", transform="doubletransition, exactly as author config"),
        checkpoint=checkpoint_info, matrix_profiles=profiles,
        conclusion=dict(
            algebra="sum_m x[b,n,h]*A[n,m] = x[b,n,h]*row_sum(A)[n]; node n is retained, so no neighbor-node features are summed",
            mixes_nodes=False,
            exact_identity="This conv alone is identity if every row sum is exactly 1 in exact arithmetic",
            approximate_identity="With row sums near 1, absolute output change is bounded by input magnitude times row-sum deviation, plus floating-point rounding",
            not_entire_module_identity="The complete DynamicGraphConv still concatenates terms and applies learned Linear and Dropout; it is not generally an identity module",
            no_graph_wavenet_generalization="GraphWaveNet nconv has a different einsum that sums over input nodes; its behavior is not inferred from the periodicity expression",
            accuracy_impact="Not measured. No graph-fixed model, new training, forecast run, ablation or causal accuracy claim."),
        protected_files_unchanged=True, protected_source_sha256=original_hashes,
        script_sha256=sha256(Path(__file__)), elapsed_seconds=time.perf_counter() - started)
    out = root / "results/extension/graph_conv_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("SOURCE: models/lsttn.py:DynamicGraphConv.conv (actual repo method)")
    print("TOY: x=[1,10], swap adjacency -> repo", toy["repo_result"], "; A@x", toy["conventional_A_times_x"])
    print("CONTROL GraphWaveNet.nconv:", toy["graph_wavenet_control"]["result"])
    print("matrix | row_sum_min | max | mean | zero_rows | abs(row_sum-1)>1e-6")
    for row in profiles:
        print("{name} | {row_sum_min:.10g} | {row_sum_max:.10g} | {row_sum_mean:.10g} | {rows_exactly_zero} | {rows_abs_deviation_from_one_gt_1e_6}".format(**row))
    print("CHECKPOINT strict CPU load:", checkpoint_info["strict_load_verified"])
    print("CONCLUSION: periodicity conv does NOT mix nodes; row sums near 1 imply approximate identity for this conv only.")
    print("Accuracy effect NOT measured; original code/config/graph unchanged.")
    print("SAVED:", out)


if __name__ == "__main__":
    main()
