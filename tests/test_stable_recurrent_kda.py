"""Regression tests for the Stable-ABI recurrent_kda scratch-state path."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
STABLE_PATH = (
    REPO_ROOT
    / "torch_custom"
    / "fla_npu"
    / "fla_npu"
    / "ops"
    / "ascendc"
    / "_stable.py"
)


def _load_stable():
    spec = importlib.util.spec_from_file_location("recurrent_kda_stable_under_test", STABLE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RecurrentKdaScratchStateTest(unittest.TestCase):
    def test_non_inplace_seeds_scratch_from_initial_state(self) -> None:
        stable = _load_stable()
        q = torch.zeros((1, 2, 1, 128), dtype=torch.bfloat16)
        k = torch.zeros_like(q)
        v = torch.zeros((1, 2, 1, 128), dtype=torch.bfloat16)
        g = torch.zeros((1, 2, 1, 128), dtype=torch.float32)
        beta = torch.zeros((1, 2, 1), dtype=torch.float32)
        initial_state = torch.arange(
            128 * 128, dtype=torch.float32
        ).reshape(1, 1, 128, 128).requires_grad_()
        original_state = initial_state.clone()
        captured = {}

        def fake_op(*args):
            scratch = args[5]
            captured["same_storage"] = scratch.data_ptr() == initial_state.data_ptr()
            torch.testing.assert_close(scratch, initial_state)
            scratch.add_(1.0)
            return torch.zeros_like(v), scratch

        with mock.patch.dict(stable._OP_CACHE, {"npu_recurrent_kda": fake_op}):
            with mock.patch.object(stable, "_current_stream_ptr", return_value=0):
                with mock.patch.object(stable, "_INC_VERSION", lambda tensors: None):
                    out, final_state = stable.npu_recurrent_kda(
                        q,
                        k,
                        v,
                        g,
                        beta,
                        initial_state,
                        layout="BSND",
                        output_final_state=True,
                        inplace_final_state=False,
                        state_v_first=True,
                    )

        self.assertEqual(out.shape, v.shape)
        self.assertIsNot(final_state, initial_state)
        self.assertFalse(captured["same_storage"])
        self.assertFalse(final_state.requires_grad)
        torch.testing.assert_close(initial_state, original_state)
        torch.testing.assert_close(final_state, original_state + 1.0)

    def test_non_inplace_none_creates_zero_state(self) -> None:
        stable = _load_stable()
        q = torch.zeros((1, 2, 1, 128), dtype=torch.bfloat16)
        k = torch.zeros_like(q)
        v = torch.zeros((1, 2, 1, 128), dtype=torch.bfloat16)
        g = torch.zeros((1, 2, 1, 128), dtype=torch.float32)
        beta = torch.zeros((1, 2, 1), dtype=torch.float32)

        def fake_op(*args):
            scratch = args[5]
            self.assertEqual(scratch.shape, (1, 1, 128, 128))
            self.assertEqual(scratch.dtype, torch.float32)
            torch.testing.assert_close(scratch, torch.zeros_like(scratch))
            scratch.add_(1.0)
            return torch.zeros_like(v), scratch

        with mock.patch.dict(stable._OP_CACHE, {"npu_recurrent_kda": fake_op}):
            with mock.patch.object(stable, "_current_stream_ptr", return_value=0):
                with mock.patch.object(stable, "_INC_VERSION", lambda tensors: None):
                    _, final_state = stable.npu_recurrent_kda(
                        q,
                        k,
                        v,
                        g,
                        beta,
                        None,
                        layout="BSND",
                        output_final_state=True,
                        inplace_final_state=False,
                        state_v_first=True,
                    )

        torch.testing.assert_close(final_state, torch.ones_like(final_state))


if __name__ == "__main__":
    unittest.main()
