from __future__ import annotations

import unittest

from ood_intent_study.model_backends import _call_processor_with_mm_types, representation_forward_model


class _LegacyProcessor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        if "return_mm_token_type_ids" in kwargs:
            raise TypeError("unexpected keyword argument 'return_mm_token_type_ids'")
        return kwargs


class ModelBackendTests(unittest.TestCase):
    def test_processor_retries_only_for_optional_mm_keyword(self) -> None:
        processor = _LegacyProcessor()
        self.assertEqual(_call_processor_with_mm_types(processor, {"text": ["x"]}), {"text": ["x"]})
        self.assertEqual(processor.calls, 2)

        def broken(**kwargs):
            raise TypeError("malformed image tensor")

        with self.assertRaisesRegex(TypeError, "malformed image"):
            _call_processor_with_mm_types(broken, {"text": ["x"]})

    def test_representation_forward_uses_multimodal_base(self) -> None:
        base = type("Base", (), {"language_model": object(), "vision_tower": object()})()
        wrapper = type("Wrapper", (), {"model": base})()
        self.assertIs(representation_forward_model(wrapper), base)


if __name__ == "__main__":
    unittest.main()
