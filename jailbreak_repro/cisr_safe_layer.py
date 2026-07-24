from __future__ import annotations

from pathlib import Path
from typing import Any

from intentguard_refactor.intentguard.intervention import (
    SafeLayerRouter,
    load_adapter_artifact,
    resolve_module_path,
)

from .models import BaseModelRunner, Generation


class CISRSafeLayerRuntime:
    """Request-scoped runtime for a trained CISR safe-layer adapter."""

    def __init__(
        self,
        *,
        runner: BaseModelRunner,
        artifact_path: Path,
        module_path: str,
        token_scope: str = "last",
        max_route_calls: int = 9,
        expected_layer: int | None = None,
    ) -> None:
        if token_scope not in {"last", "all"}:
            raise ValueError("token_scope must be 'last' or 'all'")
        if max_route_calls <= 0:
            raise ValueError("max_route_calls must be positive")
        root_model = getattr(runner, "model", None)
        if root_model is None:
            raise TypeError(
                f"{type(runner).__name__} does not expose a torch model for safe-layer routing"
            )
        module = resolve_module_path(root_model, module_path)
        try:
            device = next(module.parameters()).device
        except StopIteration:
            device = next(root_model.parameters()).device
        adapter, metadata = load_adapter_artifact(
            str(artifact_path), map_location=device
        )
        adapter.to(device=device)
        artifact_layer = metadata.get("layer")
        if (
            expected_layer is not None
            and artifact_layer is not None
            and int(artifact_layer) != int(expected_layer)
        ):
            raise ValueError(
                f"Safe-layer artifact was trained at hidden-state index {artifact_layer}, "
                f"but detector uses {expected_layer}"
            )
        self.runner = runner
        self.artifact_path = artifact_path.resolve()
        self.module_path = module_path
        self.token_scope = token_scope
        self.max_route_calls = int(max_route_calls)
        self.metadata = metadata
        self.router = SafeLayerRouter(adapter)
        self.router.attach(module)

    def generate(
        self,
        prompt: str,
        *,
        image_path: str | None,
        max_new_tokens: int,
        temperature: float,
        top_p: float | None,
    ) -> Generation:
        with self.router.routing(
            [True],
            token_scope=self.token_scope,
            max_route_calls=self.max_route_calls,
        ) as state:
            generation = self.runner.generate(
                prompt,
                image_path=image_path,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        generation.metadata = {
            **generation.metadata,
            "cisr4_safe_layer": {
                "artifact": str(self.artifact_path),
                "module_path": self.module_path,
                "token_scope": self.token_scope,
                "max_route_calls": self.max_route_calls,
                "route_calls": state.route_calls,
                "artifact_layer": self.metadata.get("layer"),
            },
        }
        return generation

    def close(self) -> None:
        self.router.detach()

    def __enter__(self) -> "CISRSafeLayerRuntime":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()
