from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - lightweight CI/runtime
    torch = None


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

if torch is not None:
    from extract_activations import pool_hidden  # noqa: E402


@unittest.skipIf(torch is None, "torch is not installed")
class CISRV4PoolingTest(unittest.TestCase):
    def test_image_and_non_image_means_use_disjoint_valid_tokens(self) -> None:
        hidden = torch.tensor(
            [[[1.0, 0.0], [3.0, 0.0], [10.0, 2.0], [14.0, 2.0], [5.0, 0.0]]]
        )
        attention = torch.tensor([[1, 1, 1, 1, 1]])
        image = torch.tensor([[0, 0, 1, 1, 0]], dtype=torch.bool)

        image_mean = pool_hidden(hidden, attention, "image_mean", image)
        non_image_mean = pool_hidden(hidden, attention, "non_image_mean", image)

        self.assertTrue(torch.allclose(image_mean, torch.tensor([12.0, 2.0])))
        self.assertTrue(torch.allclose(non_image_mean, torch.tensor([3.0, 0.0])))

    def test_image_mean_fails_when_backend_exposes_no_image_tokens(self) -> None:
        hidden = torch.zeros((1, 3, 2))
        with self.assertRaisesRegex(ValueError, "explicit image tokens"):
            pool_hidden(
                hidden,
                torch.ones((1, 3)),
                "image_mean",
                torch.zeros((1, 3), dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
