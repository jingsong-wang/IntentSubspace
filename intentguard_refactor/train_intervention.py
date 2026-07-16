from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from intentguard.intervention import (
    InterventionLossWeights,
    LowRankSafetyAdapter,
    compute_intervention_loss,
)
from intentguard.io import write_json


def cache_array(cache: Any, name: str, required: bool = True) -> np.ndarray | None:
    if name in cache:
        return np.asarray(cache[name])
    if required:
        raise KeyError(f"Intervention cache is missing required array {name!r}")
    return None


def evaluate(
    adapter: LowRankSafetyAdapter,
    hidden: torch.Tensor,
    route_mask: torch.Tensor,
    retain_mask: torch.Tensor,
    indices: torch.Tensor,
    weights: InterventionLossWeights,
    teacher_hidden: torch.Tensor | None,
    refusal_direction: torch.Tensor | None,
    preserve_basis: torch.Tensor | None,
    refusal_margin: float,
) -> dict[str, float]:
    if indices.numel() == 0:
        return {}
    adapter.eval()
    with torch.no_grad():
        base = hidden[indices]
        routed = adapter(base)
        refusal_scores = routed @ refusal_direction if refusal_direction is not None else None
        losses = compute_intervention_loss(
            base_hidden=base,
            routed_hidden=routed,
            route_mask=route_mask[indices],
            retain_mask=retain_mask[indices],
            weights=weights,
            teacher_hidden=teacher_hidden[indices] if teacher_hidden is not None else None,
            refusal_scores=refusal_scores,
            refusal_margin=refusal_margin,
            preserve_basis=preserve_basis,
        )
    return {name: float(value.detach().cpu()) for name, value in losses.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Representation-level pretraining for a conditional safe-layer adapter."
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--refusal-margin", type=float, default=1.0)
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--projection-weight", type=float, default=0.25)
    parser.add_argument("--retain-hidden-weight", type=float, default=0.25)
    parser.add_argument("--semantic-weight", type=float, default=0.25)
    parser.add_argument("--delta-weight", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    cache = np.load(args.cache, allow_pickle=True)
    hidden_np = cache_array(cache, "hidden")
    splits_np = cache_array(cache, "splits")
    route_np = cache_array(cache, "route_mask")
    retain_np = cache_array(cache, "retain_mask")
    assert hidden_np is not None and splits_np is not None
    assert route_np is not None and retain_np is not None
    if hidden_np.ndim not in {2, 3}:
        raise ValueError("hidden must have shape [n, hidden] or [n, tokens, hidden]")
    n = int(hidden_np.shape[0])
    if len(splits_np) != n or len(route_np) != n or len(retain_np) != n:
        raise ValueError("splits, route_mask, and retain_mask must align with hidden rows")

    teacher_np = cache_array(cache, "teacher_hidden", required=False)
    refusal_np = cache_array(cache, "refusal_direction", required=False)
    preserve_np = cache_array(cache, "preserve_basis", required=False)
    if teacher_np is None and refusal_np is None:
        raise ValueError(
            "Training needs teacher_hidden and/or refusal_direction; detector labels alone do not define a safety transform."
        )
    if teacher_np is not None and teacher_np.shape != hidden_np.shape:
        raise ValueError("teacher_hidden must have the same shape as hidden")
    if refusal_np is not None and refusal_np.shape != (hidden_np.shape[-1],):
        raise ValueError("refusal_direction must have shape [hidden]")
    if preserve_np is not None and (preserve_np.ndim != 2 or preserve_np.shape[-1] != hidden_np.shape[-1]):
        raise ValueError("preserve_basis must have shape [rank, hidden]")

    splits = np.asarray(splits_np).astype(str)
    route_np_bool = np.asarray(route_np, dtype=bool)
    retain_np_bool = np.asarray(retain_np, dtype=bool)
    eligible_np = route_np_bool | retain_np_bool
    train_indices_np = np.where((splits == "train") & eligible_np)[0]
    validation_indices_np = np.where((splits == "validation") & eligible_np)[0]
    if len(train_indices_np) == 0:
        raise ValueError("No train rows found; calibration and test rows are never used for fitting")
    if not np.any(route_np_bool[train_indices_np]):
        raise ValueError("Train split contains no route-positive samples")
    if not np.any(retain_np_bool[train_indices_np]):
        raise ValueError("Train split contains no retain/control samples")

    hidden = torch.as_tensor(hidden_np, dtype=torch.float32, device=device)
    route_mask = torch.as_tensor(route_np, dtype=torch.bool, device=device)
    retain_mask = torch.as_tensor(retain_np, dtype=torch.bool, device=device)
    teacher_hidden = (
        torch.as_tensor(teacher_np, dtype=torch.float32, device=device)
        if teacher_np is not None
        else None
    )
    refusal_direction = (
        torch.as_tensor(refusal_np, dtype=torch.float32, device=device)
        if refusal_np is not None
        else None
    )
    preserve_basis = (
        torch.as_tensor(preserve_np, dtype=torch.float32, device=device)
        if preserve_np is not None
        else None
    )
    train_indices = torch.as_tensor(train_indices_np, dtype=torch.long, device=device)
    validation_indices = torch.as_tensor(
        validation_indices_np, dtype=torch.long, device=device
    )

    adapter = LowRankSafetyAdapter(
        hidden_size=int(hidden_np.shape[-1]),
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    weights = InterventionLossWeights(
        teacher=args.teacher_weight,
        refusal_projection=args.projection_weight,
        retain_hidden=args.retain_hidden_weight,
        semantic_preserve=args.semantic_weight,
        delta_l2=args.delta_weight,
    )

    best_metric = float("inf")
    best_state = copy.deepcopy(adapter.state_dict())
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        permutation = train_indices[torch.randperm(len(train_indices), generator=generator).to(device)]
        epoch_total = 0.0
        batch_count = 0
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            base = hidden[indices]
            routed = adapter(base)
            refusal_scores = routed @ refusal_direction if refusal_direction is not None else None
            losses = compute_intervention_loss(
                base_hidden=base,
                routed_hidden=routed,
                route_mask=route_mask[indices],
                retain_mask=retain_mask[indices],
                weights=weights,
                teacher_hidden=teacher_hidden[indices] if teacher_hidden is not None else None,
                refusal_scores=refusal_scores,
                refusal_margin=args.refusal_margin,
                preserve_basis=preserve_basis,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_total += float(losses["total"].detach().cpu())
            batch_count += 1

        train_total = epoch_total / max(batch_count, 1)
        validation = evaluate(
            adapter,
            hidden,
            route_mask,
            retain_mask,
            validation_indices,
            weights,
            teacher_hidden,
            refusal_direction,
            preserve_basis,
            args.refusal_margin,
        )
        selection_metric = validation.get("total", train_total)
        history.append(
            {"epoch": epoch, "train_total": train_total, "selection_total": selection_metric}
        )
        if selection_metric < best_metric - 1e-7:
            best_metric = selection_metric
            best_state = copy.deepcopy(adapter.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break

    adapter.load_state_dict(best_state)
    final_train = evaluate(
        adapter,
        hidden,
        route_mask,
        retain_mask,
        train_indices,
        weights,
        teacher_hidden,
        refusal_direction,
        preserve_basis,
        args.refusal_margin,
    )
    final_validation = evaluate(
        adapter,
        hidden,
        route_mask,
        retain_mask,
        validation_indices,
        weights,
        teacher_hidden,
        refusal_direction,
        preserve_basis,
        args.refusal_margin,
    )
    selected_layer = (
        int(np.asarray(cache["layer"]).reshape(-1)[0]) if "layer" in cache else None
    )
    artifact = {
        "format_version": "CISR_safe_layer_adapter_v1",
        "state_dict": {name: value.detach().cpu() for name, value in adapter.state_dict().items()},
        "hidden_size": adapter.hidden_size,
        "rank": adapter.rank,
        "alpha": adapter.alpha,
        "dropout": args.dropout,
        "layer": selected_layer,
        "source_cache": str(args.cache),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.out)
    summary = {
        "format_version": "CISR_safe_layer_training_v1",
        "artifact": str(args.out),
        "source_cache": str(args.cache),
        "device": str(device),
        "layer": selected_layer,
        "hidden_size": adapter.hidden_size,
        "rank": adapter.rank,
        "alpha": adapter.alpha,
        "fit_split": "train",
        "train_n": int(len(train_indices_np)),
        "validation_n": int(len(validation_indices_np)),
        "calibration_n_excluded": int(np.sum(splits == "calibration")),
        "test_n_excluded": int(np.sum(splits == "test")),
        "route_positive_train_n": int(np.sum(route_np_bool[train_indices_np])),
        "retain_train_n": int(np.sum(retain_np_bool[train_indices_np])),
        "excluded_unjudged_or_audit_train_n": int(
            np.sum((splits == "train") & ~eligible_np)
        ),
        "epochs_completed": len(history),
        "best_selection_total": best_metric,
        "train_losses": final_train,
        "validation_losses": final_validation,
        "history_tail": history[-10:],
        "has_teacher_hidden": teacher_np is not None,
        "has_refusal_direction": refusal_np is not None,
        "has_preserve_basis": preserve_np is not None,
    }
    summary_path = args.summary_out or args.out.with_suffix(".json")
    write_json(summary_path, summary)
    print(f"Wrote safe-layer adapter to {args.out}")
    print(f"Wrote training summary to {summary_path}")


if __name__ == "__main__":
    main()
