from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Literal, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .routing import RoutingDecision


TokenScope = Literal["all", "last"]


def assert_hidden_equivalent(
    captured: Tensor,
    reference: Tensor,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict[str, float]:
    """Fail closed when a module path does not reproduce the detector state."""

    if captured.shape != reference.shape:
        raise ValueError(
            f"Hook/reference shape mismatch: {tuple(captured.shape)} != {tuple(reference.shape)}"
        )
    captured_float = captured.detach().float()
    reference_float = reference.detach().float().to(device=captured_float.device)
    difference = (captured_float - reference_float).abs()
    diagnostics = {
        "max_abs_error": float(difference.max().cpu()) if difference.numel() else 0.0,
        "mean_abs_error": float(difference.mean().cpu()) if difference.numel() else 0.0,
    }
    if not torch.allclose(captured_float, reference_float, atol=atol, rtol=rtol):
        raise ValueError(
            "Selected module output is not equivalent to the detector hidden state: "
            f"max_abs_error={diagnostics['max_abs_error']:.6g}"
        )
    return diagnostics


def resolve_module_path(root: nn.Module, path: str) -> nn.Module:
    """Resolve an explicit dotted module path, including numeric ModuleList indices."""

    current: Any = root
    for part in path.split("."):
        if not part:
            raise ValueError(f"Invalid empty component in module path {path!r}")
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise TypeError(f"Resolved object at {path!r} is not torch.nn.Module")
    return current


def load_adapter_artifact(
    path: str,
    *,
    map_location: torch.device | str = "cpu",
) -> tuple["LowRankSafetyAdapter", dict[str, Any]]:
    artifact = torch.load(path, map_location=map_location)
    if artifact.get("format_version") != "CISR_safe_layer_adapter_v1":
        raise ValueError(f"Unsupported adapter artifact: {artifact.get('format_version')!r}")
    adapter = LowRankSafetyAdapter(
        hidden_size=int(artifact["hidden_size"]),
        rank=int(artifact["rank"]),
        alpha=float(artifact["alpha"]),
        dropout=float(artifact.get("dropout", 0.0)),
    )
    adapter.load_state_dict(artifact["state_dict"])
    adapter.eval()
    return adapter, artifact


class LowRankSafetyAdapter(nn.Module):
    """A zero-initialized low-rank residual transform for one model layer."""

    def __init__(
        self,
        hidden_size: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or rank <= 0:
            raise ValueError("hidden_size and rank must be positive")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.down = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(float(dropout))
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def delta(self, hidden: Tensor) -> Tensor:
        normalized = F.layer_norm(hidden, (self.hidden_size,))
        return self.up(F.gelu(self.down(self.dropout(normalized)))) * self.scaling

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden + self.delta(hidden)


@dataclass
class RoutingRuntimeState:
    active_mask: Tensor
    token_scope: TokenScope = "last"
    token_mask: Tensor | None = None
    max_route_calls: int | None = None
    route_calls: int = 0

    @property
    def exhausted(self) -> bool:
        return self.max_route_calls is not None and self.route_calls >= self.max_route_calls


def decisions_to_mask(
    decisions: Iterable[RoutingDecision],
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    return torch.tensor(
        [decision.triggered for decision in decisions],
        dtype=torch.bool,
        device=device,
    )


def _replace_hidden(output: Any, hidden: Tensor) -> Any:
    if isinstance(output, Tensor):
        return hidden
    if isinstance(output, tuple):
        if not output or not isinstance(output[0], Tensor):
            raise TypeError("Tuple layer output must contain hidden states at index 0")
        return (hidden, *output[1:])
    if isinstance(output, list):
        if not output or not isinstance(output[0], Tensor):
            raise TypeError("List layer output must contain hidden states at index 0")
        return [hidden, *output[1:]]
    if isinstance(output, dict):
        for key in ("hidden_states", "last_hidden_state"):
            if key in output and isinstance(output[key], Tensor):
                copied = dict(output)
                copied[key] = hidden
                return copied
    raise TypeError(
        "Unsupported layer output. Expected Tensor, tuple/list with Tensor first, "
        "or dict containing hidden_states/last_hidden_state."
    )


def _extract_hidden(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    if isinstance(output, dict):
        for key in ("hidden_states", "last_hidden_state"):
            value = output.get(key)
            if isinstance(value, Tensor):
                return value
    raise TypeError("Could not extract hidden states from the selected layer output")


class SafeLayerRouter:
    """Attach a conditional safety adapter to a selected ``torch.nn.Module``.

    A routing context is request-local through ``ContextVar``. The adapter is
    dormant outside that context and for batch rows whose decision is false.
    ``max_route_calls`` bounds prefill/decode forward calls; callers should set
    it to ``K + 1`` when prefill plus the first K decode steps are routed.
    """

    def __init__(self, adapter: LowRankSafetyAdapter) -> None:
        self.adapter = adapter
        self._state: ContextVar[RoutingRuntimeState | None] = ContextVar(
            f"safe_layer_router_{id(self)}", default=None
        )
        self._handle: Any | None = None

    @property
    def attached(self) -> bool:
        return self._handle is not None

    def attach(self, module: nn.Module) -> None:
        if self._handle is not None:
            raise RuntimeError("SafeLayerRouter is already attached")
        self._handle = module.register_forward_hook(self._forward_hook)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @contextmanager
    def routing(
        self,
        active_mask: Tensor | Sequence[bool],
        *,
        token_scope: TokenScope = "last",
        token_mask: Tensor | Sequence[Sequence[bool]] | None = None,
        max_route_calls: int | None = None,
    ) -> Iterator[RoutingRuntimeState]:
        if token_scope not in {"all", "last"}:
            raise ValueError(f"Unsupported token_scope={token_scope!r}")
        if max_route_calls is not None and max_route_calls <= 0:
            raise ValueError("max_route_calls must be positive when provided")
        mask = torch.as_tensor(active_mask, dtype=torch.bool)
        if mask.ndim != 1:
            raise ValueError("active_mask must have shape [batch]")
        if mask.numel() == 0:
            raise ValueError("active_mask must contain at least one batch row")
        sequence_mask = (
            torch.as_tensor(token_mask, dtype=torch.bool) if token_mask is not None else None
        )
        if sequence_mask is not None and (
            sequence_mask.ndim != 2 or sequence_mask.shape[0] != mask.numel()
        ):
            raise ValueError("token_mask must have shape [batch, sequence]")
        state = RoutingRuntimeState(
            active_mask=mask,
            token_scope=token_scope,
            token_mask=sequence_mask,
            max_route_calls=max_route_calls,
        )
        token = self._state.set(state)
        try:
            yield state
        finally:
            self._state.reset(token)

    def route_hidden(self, hidden: Tensor, state: RoutingRuntimeState) -> Tensor:
        if state.exhausted or not bool(state.active_mask.any()):
            return hidden
        if hidden.ndim not in {2, 3}:
            raise ValueError(
                f"Expected layer hidden shape [batch, hidden] or [batch, seq, hidden], got {tuple(hidden.shape)}"
            )
        base_batch = state.active_mask.numel()
        if hidden.shape[0] % base_batch != 0:
            raise ValueError(
                f"active_mask batch={base_batch} cannot expand to hidden batch={hidden.shape[0]}"
            )
        if hidden.shape[-1] != self.adapter.hidden_size:
            raise ValueError(
                f"Adapter hidden_size={self.adapter.hidden_size} does not match layer output={hidden.shape[-1]}"
            )

        adapter_parameter = next(self.adapter.parameters())
        if adapter_parameter.device != hidden.device:
            raise ValueError(
                f"Adapter device={adapter_parameter.device} does not match hidden device={hidden.device}"
            )
        expansion = hidden.shape[0] // base_batch
        active = state.active_mask.to(device=hidden.device).repeat_interleave(expansion)
        delta = self.adapter.delta(hidden.to(dtype=adapter_parameter.dtype)).to(
            dtype=hidden.dtype
        )
        if hidden.ndim == 2:
            mask = active[:, None]
        else:
            mask = active[:, None, None].expand(-1, hidden.shape[1], 1).clone()
            if state.token_scope == "last" and hidden.shape[1] > 1:
                mask.zero_()
                sequence_mask = state.token_mask
                if sequence_mask is not None and sequence_mask.shape[1] == hidden.shape[1]:
                    sequence_mask = sequence_mask.to(device=hidden.device).repeat_interleave(
                        expansion, dim=0
                    )
                    if not bool(sequence_mask.any(dim=1).all()):
                        raise ValueError("Every token_mask row must contain at least one active token")
                    positions = (
                        torch.arange(hidden.shape[1], device=hidden.device)[None, :]
                        .expand(hidden.shape[0], -1)
                        .masked_fill(~sequence_mask, -1)
                        .max(dim=1)
                        .values
                    )
                else:
                    positions = torch.full(
                        (hidden.shape[0],), hidden.shape[1] - 1, device=hidden.device
                    )
                mask[torch.arange(hidden.shape[0], device=hidden.device), positions, 0] = active
        state.route_calls += 1
        return hidden + delta * mask.to(dtype=hidden.dtype)

    def _forward_hook(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        del module, inputs
        state = self._state.get()
        if state is None or state.exhausted or not bool(state.active_mask.any()):
            return output
        hidden = _extract_hidden(output)
        routed = self.route_hidden(hidden, state)
        return _replace_hidden(output, routed)


@dataclass(frozen=True)
class InterventionLossWeights:
    safe_ce: float = 1.0
    teacher: float = 1.0
    refusal_projection: float = 0.25
    retain_kl: float = 1.0
    retain_hidden: float = 0.25
    semantic_preserve: float = 0.25
    delta_l2: float = 1e-4


def _sample_mean(values: Tensor) -> Tensor:
    if values.ndim <= 1:
        return values
    return values.reshape(values.shape[0], -1).mean(dim=1)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    selected = values[mask]
    if selected.numel() == 0:
        return values.sum() * 0.0
    return selected.mean()


def compute_intervention_loss(
    *,
    base_hidden: Tensor,
    routed_hidden: Tensor,
    route_mask: Tensor,
    retain_mask: Tensor,
    weights: InterventionLossWeights = InterventionLossWeights(),
    teacher_hidden: Tensor | None = None,
    refusal_scores: Tensor | None = None,
    refusal_margin: float = 1.0,
    base_logits: Tensor | None = None,
    routed_logits: Tensor | None = None,
    safe_labels: Tensor | None = None,
    preserve_basis: Tensor | None = None,
) -> dict[str, Tensor]:
    """Compose representation and optional token-level CSRL objectives.

    Token CE/KL terms become available when the model-aware trainer supplies
    logits. Cached-hidden pretraining can use teacher, projection, retention,
    preservation, and norm terms without loading the full VLM.
    """

    route = torch.as_tensor(route_mask, dtype=torch.bool, device=base_hidden.device)
    retain = torch.as_tensor(retain_mask, dtype=torch.bool, device=base_hidden.device)
    if route.shape != retain.shape or route.ndim != 1 or route.numel() != base_hidden.shape[0]:
        raise ValueError("route_mask and retain_mask must both have shape [batch]")
    if bool((route & retain).any()):
        raise ValueError("route_mask and retain_mask must be disjoint")
    if routed_hidden.shape != base_hidden.shape:
        raise ValueError("base_hidden and routed_hidden must have identical shapes")

    zero = routed_hidden.sum() * 0.0
    components: dict[str, Tensor] = {}

    if routed_logits is not None and safe_labels is not None:
        labels = safe_labels.to(device=routed_logits.device, dtype=torch.long)
        if routed_logits.shape[:-1] != labels.shape:
            raise ValueError("safe_labels must match routed_logits without the vocabulary dimension")
        token_loss = F.cross_entropy(
            routed_logits.reshape(-1, routed_logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(labels.shape)
        valid_tokens = labels.ne(-100)
        token_loss = (token_loss * valid_tokens).reshape(token_loss.shape[0], -1).sum(dim=1)
        valid_count = valid_tokens.reshape(valid_tokens.shape[0], -1).sum(dim=1).clamp_min(1)
        token_loss = token_loss / valid_count
        components["safe_ce"] = _masked_mean(token_loss, route)
    else:
        components["safe_ce"] = zero

    if teacher_hidden is not None:
        if teacher_hidden.shape != routed_hidden.shape:
            raise ValueError("teacher_hidden must match routed_hidden")
        teacher_error = _sample_mean((routed_hidden - teacher_hidden.detach()).pow(2))
        components["teacher"] = _masked_mean(teacher_error, route)
    else:
        components["teacher"] = zero

    if refusal_scores is not None:
        scores = _sample_mean(refusal_scores)
        if scores.shape != route.shape:
            raise ValueError("refusal_scores must reduce to one value per sample")
        projection_error = F.relu(float(refusal_margin) - scores)
        components["refusal_projection"] = _masked_mean(projection_error, route)
    else:
        components["refusal_projection"] = zero

    if base_logits is not None and routed_logits is not None:
        if base_logits.shape != routed_logits.shape:
            raise ValueError("base_logits and routed_logits must have identical shapes")
        kl = F.kl_div(
            F.log_softmax(routed_logits, dim=-1),
            F.softmax(base_logits.detach(), dim=-1),
            reduction="none",
        ).sum(dim=-1)
        components["retain_kl"] = _masked_mean(_sample_mean(kl), retain)
    else:
        components["retain_kl"] = zero

    hidden_error = _sample_mean((routed_hidden - base_hidden.detach()).pow(2))
    components["retain_hidden"] = _masked_mean(hidden_error, retain)

    if preserve_basis is not None:
        basis = preserve_basis.to(device=base_hidden.device, dtype=base_hidden.dtype)
        if basis.ndim != 2 or basis.shape[-1] != base_hidden.shape[-1]:
            raise ValueError("preserve_basis must have shape [rank, hidden]")
        base_projection = base_hidden.detach() @ basis.T
        routed_projection = routed_hidden @ basis.T
        preservation_error = _sample_mean((routed_projection - base_projection).pow(2))
        components["semantic_preserve"] = _masked_mean(preservation_error, route | retain)
    else:
        components["semantic_preserve"] = zero

    delta_error = _sample_mean((routed_hidden - base_hidden.detach()).pow(2))
    components["delta_l2"] = _masked_mean(delta_error, route | retain)

    total = (
        weights.safe_ce * components["safe_ce"]
        + weights.teacher * components["teacher"]
        + weights.refusal_projection * components["refusal_projection"]
        + weights.retain_kl * components["retain_kl"]
        + weights.retain_hidden * components["retain_hidden"]
        + weights.semantic_preserve * components["semantic_preserve"]
        + weights.delta_l2 * components["delta_l2"]
    )
    components["total"] = total
    return components
